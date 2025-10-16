from __future__ import annotations

from typing import List, Optional, Set, Dict

import logging

import torch

from sglang.srt.sampling.penaltylib.orchestrator import (
    BatchedPenalizerOrchestrator,
    _BatchedPenalizer,
)
from sglang.srt.sampling.penaltylib.constants import (
    QUOTE_CHARS,
    SENTENCE_END_CHARS,
)

logger = logging.getLogger(__name__)

class BatchedFixedBigramStartGuardPenalizer(_BatchedPenalizer):
    """Hard-blocks the bigram "The word" at sentence/reply starts.

    Behavior:
    - At BOS/start positions, prevents emitting a single token whose decode begins with
      "the word" (case-insensitive), via -inf logit mask.
    - If the last emitted token at a start position decodes to a first token of "The"
      (case-insensitive), then on the next step hard-block second-token IDs that would
      complete "The word" (handling whether the first token ends with a space).
    """

    def __init__(self, orchestrator: BatchedPenalizerOrchestrator):
        self.orchestrator = orchestrator
        self._is_prepared = False

    def _is_required(self) -> bool:
        # Always enforce by default; guard can be disabled via server args
        return True

    def _prepare(self):
        reqs = self.orchestrator.reqs()
        device = self.orchestrator.device
        vocab_size = self.orchestrator.vocab_size

        # Per-request state used across steps
        self.pending_after_the_at_start = torch.zeros(
            (len(reqs),), dtype=torch.bool, device=device
        )

        # Precomputed token ID sets shared across requests (on CPU then moved lazily)
        self.first_token_ids_per_req: List[Optional[torch.Tensor]] = [None] * len(reqs)
        self.first_token_ids_set_per_req: List[Optional[Set[int]]] = [None] * len(reqs)
        self.first_token_requires_space_per_req: List[Optional[Dict[int, bool]]] = [
            None
        ] * len(reqs)

        # Global (request-agnostic) sets for step-1 and step-2 blocking
        self.single_token_blacklist: Optional[torch.Tensor] = None
        self.word_with_space_ids: Optional[torch.Tensor] = None
        self.word_no_space_ids: Optional[torch.Tensor] = None

        # Build shared sets by scanning the vocab once using any tokenizer available.
        # We will attempt using the first non-None tokenizer; if none available, we skip.
        tokenizer0 = None
        for r in reqs:
            tok = getattr(r, "tokenizer", None)
            if tok is not None:
                tokenizer0 = tok
                break

        if tokenizer0 is not None:
            # Step-1: block single tokens that start with "the word" (boundary-aware)
            st_blacklist: Set[int] = set()
            for tid in range(vocab_size):
                try:
                    s = tokenizer0.decode([tid])
                except Exception:
                    continue
                if not s:
                    continue
                s_low = s.lstrip().lower()
                if not s_low.startswith("the word"):
                    continue
                # boundary after "word": next char must be non-alpha or absent
                after = s_low[len("the word") : len("the word") + 1]
                if after and after.isalpha():
                    continue
                st_blacklist.add(int(tid))
            if st_blacklist:
                self.single_token_blacklist = torch.tensor(
                    sorted(st_blacklist), dtype=torch.int64, device=device
                )

            # Step-2: collect second-token candidates depending on presence of leading space
            w_space_ids: Set[int] = set()
            no_space_ids: Set[int] = set()
            # Scan vocab for decodes starting with " word" (and boundary) / "word" (and boundary)
            for tid in range(vocab_size):
                try:
                    s = tokenizer0.decode([tid])
                except Exception:
                    continue
                if not s:
                    continue
                sl = s.lower()
                if sl.startswith(" word"):
                    # check boundary after word
                    pos = len(" word")
                    if pos < len(sl) and sl[pos : pos + 1].isalpha():
                        pass
                    else:
                        w_space_ids.add(int(tid))
                if sl.startswith("word"):
                    pos = len("word")
                    if pos < len(sl) and sl[pos : pos + 1].isalpha():
                        pass
                    else:
                        no_space_ids.add(int(tid))
            if w_space_ids:
                self.word_with_space_ids = torch.tensor(
                    sorted(w_space_ids), dtype=torch.int64, device=device
                )
            if no_space_ids:
                self.word_no_space_ids = torch.tensor(
                    sorted(no_space_ids), dtype=torch.int64, device=device
                )
            logger.info(
                "BigramGuard prepare: tokenizer=found single_token_blacklist=%d word_with_space_ids=%d word_no_space_ids=%d",
                0 if self.single_token_blacklist is None else int(self.single_token_blacklist.numel()),
                0 if self.word_with_space_ids is None else int(self.word_with_space_ids.numel()),
                0 if self.word_no_space_ids is None else int(self.word_no_space_ids.numel()),
            )
        else:
            logger.info("BigramGuard prepare: tokenizer=missing; only per-request first-token prep will run")

        # Per-request preparation for first-token IDs for "The" at starts
        for i, r in enumerate(reqs):
            tok = getattr(r, "tokenizer", None)
            if tok is None:
                logger.info("BigramGuard: request idx=%d has no tokenizer; skipping first-token prep", i)
                continue
            first_ids: Set[int] = set()
            requires_space: Dict[int, bool] = {}

            # Build a small surface set for "The" and lowercase variants with typical prefixes
            variants = [
                "The",
                " the",
                "the",
                " The",
            ]
            for q in QUOTE_CHARS:
                variants.extend([f"{q}The", f" {q}The", f"{q}the", f" {q}the"])

            for surf in variants:
                try:
                    ids = tok.encode(surf)
                except Exception:
                    continue
                if not ids:
                    continue
                fid = int(ids[0])
                if fid in first_ids:
                    continue
                first_ids.add(fid)
                try:
                    decoded = tok.decode([fid])
                except Exception:
                    decoded = ""
                # If the first token decode ends with space, then the next token should be "word" (no leading space)
                # Otherwise, we need next token that starts with " word"
                requires_space[fid] = not (decoded.endswith(" "))

            if first_ids:
                self.first_token_ids_per_req[i] = torch.tensor(
                    sorted(first_ids), dtype=torch.int64, device=device
                )
                self.first_token_ids_set_per_req[i] = set(first_ids)
                self.first_token_requires_space_per_req[i] = requires_space
                logger.info(
                    "BigramGuard: request idx=%d prepared first-token set size=%d",
                    i,
                    int(self.first_token_ids_per_req[i].numel()),
                )
            else:
                self.first_token_ids_per_req[i] = None
                self.first_token_ids_set_per_req[i] = None
                self.first_token_requires_space_per_req[i] = None
                logger.info(
                    "BigramGuard: request idx=%d has empty first-token set for 'The'",
                    i,
                )

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        # Track if we just emitted a first-token for "The" at a start position
        reqs = self.orchestrator.reqs()
        if output_ids is None or output_ids.numel() == 0:
            return
        for i, req in enumerate(reqs):
            first_ids_set = self.first_token_ids_set_per_req[i]
            if not first_ids_set:
                continue
            # Only trigger at sentence/reply starts
            if not self._is_start_position(req):
                continue
            last_id = int(output_ids[i].item())
            if int(last_id) in first_ids_set:
                self.pending_after_the_at_start[i] = True
                rid = getattr(req, "rid", None)
                logger.info(
                    "BigramGuard: pending second-token ban set rid=%s idx=%d last_id=%d",
                    str(rid),
                    i,
                    last_id,
                )

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        # BOS single-token hard block and two-step bigram guard
        reqs = self.orchestrator.reqs()
        for i, req in enumerate(reqs):
            # BOS: Hard block single-token candidates that decode to "the word..." (boundary-aware)
            if len(getattr(req, "output_ids", []) or []) == 0 and self.single_token_blacklist is not None:
                logits[i, self.single_token_blacklist] = -float("inf")
                rid = getattr(req, "rid", None)
                logger.info(
                    "BigramGuard: applied BOS single-token mask rid=%s idx=%d masked=%d",
                    str(rid),
                    i,
                    int(self.single_token_blacklist.numel()),
                )

            # Two-step guard: if the last emitted token at a start was a first-token for "The",
            # then block appropriate second-token IDs
            if self.pending_after_the_at_start[i]:
                first_ids = self.first_token_ids_per_req[i]
                requires_space_map = self.first_token_requires_space_per_req[i] or {}
                out_ids = getattr(req, "output_ids", None) or []
                if out_ids:
                    last_id = int(out_ids[-1])
                    need_space_variant = bool(requires_space_map.get(last_id, True))
                    if need_space_variant and self.word_with_space_ids is not None:
                        logits[i, self.word_with_space_ids] = -float("inf")
                        rid = getattr(req, "rid", None)
                        logger.info(
                            "BigramGuard: blocked second token set variant=space rid=%s idx=%d size=%d last_id=%d",
                            str(rid),
                            i,
                            int(self.word_with_space_ids.numel()),
                            last_id,
                        )
                    elif (not need_space_variant) and self.word_no_space_ids is not None:
                        logits[i, self.word_no_space_ids] = -float("inf")
                        rid = getattr(req, "rid", None)
                        logger.info(
                            "BigramGuard: blocked second token set variant=no_space rid=%s idx=%d size=%d last_id=%d",
                            str(rid),
                            i,
                            int(self.word_no_space_ids.numel()),
                            last_id,
                        )
                # Reset flag after applying for this step
                self.pending_after_the_at_start[i] = False

    def _filter(self, keep_indices: torch.Tensor):
        keep = keep_indices
        self.pending_after_the_at_start = self.pending_after_the_at_start[keep]
        self.first_token_ids_per_req = [self.first_token_ids_per_req[j] for j in keep.tolist()]
        self.first_token_ids_set_per_req = [self.first_token_ids_set_per_req[j] for j in keep.tolist()]
        self.first_token_requires_space_per_req = [
            self.first_token_requires_space_per_req[j] for j in keep.tolist()
        ]

    def _merge(self, their: "BatchedFixedBigramStartGuardPenalizer"):
        self.pending_after_the_at_start = torch.cat(
            [self.pending_after_the_at_start, their.pending_after_the_at_start], dim=0
        )
        self.first_token_ids_per_req.extend(their.first_token_ids_per_req)
        self.first_token_ids_set_per_req.extend(their.first_token_ids_set_per_req)
        self.first_token_requires_space_per_req.extend(
            their.first_token_requires_space_per_req
        )
        # Global sets should be equivalent; prefer keeping ours if both exist
        if self.single_token_blacklist is None:
            self.single_token_blacklist = their.single_token_blacklist
        if self.word_with_space_ids is None:
            self.word_with_space_ids = their.word_with_space_ids
        if self.word_no_space_ids is None:
            self.word_no_space_ids = their.word_no_space_ids

    def _is_start_position(self, req) -> bool:
        out_ids = getattr(req, "output_ids", None) or []
        if len(out_ids) == 0:
            return True
        tok = getattr(req, "tokenizer", None)
        if tok is None:
            return False
        n = min(12, len(out_ids))
        try:
            tail = tok.decode(out_ids[-n:])
        except Exception:
            return False
        if not tail:
            return False
        i = len(tail) - 1
        while i >= 0 and tail[i].isspace():
            i -= 1
        if i < 0:
            return False
        ch = tail[i]
        return ch in QUOTE_CHARS or ch in SENTENCE_END_CHARS
