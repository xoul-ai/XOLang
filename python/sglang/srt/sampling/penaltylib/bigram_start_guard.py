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
        # Canonical token sequences for " word" and "word" (used for multi-step matching)
        self.suffix_seq_space: Optional[List[int]] = None
        self.suffix_seq_nospace: Optional[List[int]] = None

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
            # Canonical encode path for exact suffix matching in multi-step cases
            try:
                self.suffix_seq_space = tokenizer0.encode(" word")
            except Exception:
                self.suffix_seq_space = None
            try:
                self.suffix_seq_nospace = tokenizer0.encode("word")
            except Exception:
                self.suffix_seq_nospace = None
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
        # Initialize per-request FSM for suffix multi-step tracking
        self.active_after_the = torch.zeros((len(reqs),), dtype=torch.bool, device=device)
        self.suffix_variant_space = torch.zeros((len(reqs),), dtype=torch.bool, device=device)  # True => use space variant
        self.suffix_progress = torch.zeros((len(reqs),), dtype=torch.int32, device=device)

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
                self.active_after_the[i] = True
                # Determine variant at the moment we observed the first token
                req_map = self.first_token_requires_space_per_req[i] or {}
                need_space_variant = bool(req_map.get(last_id, True))
                self.suffix_variant_space[i] = need_space_variant
                self.suffix_progress[i] = 0
                rid = getattr(req, "rid", None)
                logger.info(
                    "BigramGuard: pending second-token ban set rid=%s idx=%d last_id=%d need_space_variant=%s",
                    str(rid),
                    i,
                    last_id,
                    str(need_space_variant),
                )
            # If already active (after seeing first token), advance FSM with current token
            if bool(self.active_after_the[i].item()):
                # Initialize variant and reset progress if this is the first step after 'The'
                if not bool(self.pending_after_the_at_start[i].item()):
                    # Advance matching on subsequent steps
                    seq = (
                        self.suffix_seq_space if bool(self.suffix_variant_space[i].item()) else self.suffix_seq_nospace
                    )
                    if seq is None or len(seq) == 0:
                        # No canonical sequence; deactivate to avoid false blocks
                        self.active_after_the[i] = False
                    else:
                        prog = int(self.suffix_progress[i].item())
                        # Only advance if not exceeding sequence
                        if prog < len(seq):
                            if last_id == int(seq[prog]):
                                self.suffix_progress[i] = prog + 1
                            else:
                                # diverged; deactivate
                                self.active_after_the[i] = False
                        logger.info(
                            "BigramGuard: advance FSM rid=%s idx=%d last_id=%d prog=%d/%d variant_space=%s",
                            str(getattr(req, "rid", None)),
                            i,
                            last_id,
                            int(self.suffix_progress[i].item()),
                            len(seq),
                            str(bool(self.suffix_variant_space[i].item())),
                        )

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        # BOS single-token hard block and two-step bigram guard
        reqs = self.orchestrator.reqs()
        for i, req in enumerate(reqs):
            # BOS: Hard block single-token candidates that decode to "the word..." (boundary-aware)
            out_ids_list = getattr(req, "output_ids", []) or []
            rid = getattr(req, "rid", None)
            if len(out_ids_list) == 0 and self.single_token_blacklist is not None:
                logits[i, self.single_token_blacklist] = -float("inf")
                logger.info(
                    "BigramGuard: applied BOS single-token mask rid=%s idx=%d masked=%d",
                    str(rid),
                    i,
                    int(self.single_token_blacklist.numel()),
                )
            # Verbose diagnostics: decode tail and top-10 tokens pre-mask
            try:
                tok = getattr(req, "tokenizer", None)
                if tok is not None:
                    tail_decode = tok.decode(out_ids_list[-12:]) if out_ids_list else ""
                else:
                    tail_decode = ""
            except Exception:
                tail_decode = "<decode_error>"
            # snapshot top-10 before applying multi-step mask
            try:
                vals, idxs = torch.topk(logits[i], k=min(10, logits.shape[1]))
                top_ids = idxs.tolist()
                top_vals = vals.tolist()
            except Exception:
                top_ids, top_vals = [], []
            decoded_tops = []
            tok = getattr(req, "tokenizer", None)
            if tok is not None:
                for tid in top_ids:
                    try:
                        decoded_tops.append(tok.decode([tid]))
                    except Exception:
                        decoded_tops.append("")
            logger.info(
                "BigramGuard: pre-mask diag rid=%s idx=%d out_len=%d tail=['%s'] top10_ids=%s top10_decodes=%s",
                str(rid),
                i,
                len(out_ids_list),
                tail_decode.replace("\n", "\\n"),
                top_ids,
                decoded_tops,
            )

            # Two-step guard: if the last emitted token at a start was a first-token for "The",
            # then block appropriate second-token IDs
            if self.pending_after_the_at_start[i]:
                # Use variant decided at cumulate time to avoid drift
                need_space_variant = bool(self.suffix_variant_space[i].item())
                if need_space_variant and self.word_with_space_ids is not None:
                        logits[i, self.word_with_space_ids] = -float("inf")
                        logger.info(
                            "BigramGuard: blocked second token set variant=space rid=%s idx=%d size=%d",
                            str(rid),
                            i,
                            int(self.word_with_space_ids.numel()),
                        )
                elif (not need_space_variant) and self.word_no_space_ids is not None:
                        logits[i, self.word_no_space_ids] = -float("inf")
                        logger.info(
                            "BigramGuard: blocked second token set variant=no_space rid=%s idx=%d size=%d",
                            str(rid),
                            i,
                            int(self.word_no_space_ids.numel()),
                        )
                # Reset flag after applying for this step
                self.pending_after_the_at_start[i] = False

            # Multi-step guard: if we're in active suffix matching state and are at the point
            # where emitting the next token would complete the suffix (canonical encoding),
            # then hard-block that specific token id.
            if bool(self.active_after_the[i].item()):
                seq = (
                    self.suffix_seq_space if bool(self.suffix_variant_space[i].item()) else self.suffix_seq_nospace
                )
                if seq and len(seq) >= 2:
                    prog = int(self.suffix_progress[i].item())
                    # If we have matched first k tokens of the suffix and k == len(seq) - 1,
                    # then the next token uniquely completes " word" or "word".
                    if prog == len(seq) - 1:
                        next_tid = int(seq[prog])
                        logits[i, next_tid] = -float("inf")
                        rid = getattr(req, "rid", None)
                        logger.info(
                            "BigramGuard: blocked multi-step completion rid=%s idx=%d next_tid=%d variant_space=%s",
                            str(rid),
                            i,
                            next_tid,
                            str(bool(self.suffix_variant_space[i].item())),
                        )
            # snapshot top-10 after masking for visibility
            try:
                vals2, idxs2 = torch.topk(logits[i], k=min(10, logits.shape[1]))
                top_ids2 = idxs2.tolist()
                decoded_tops2 = []
                tok2 = getattr(req, "tokenizer", None)
                if tok2 is not None:
                    for tid in top_ids2:
                        try:
                            decoded_tops2.append(tok2.decode([tid]))
                        except Exception:
                            decoded_tops2.append("")
            except Exception:
                top_ids2, decoded_tops2 = [], []
            logger.info(
                "BigramGuard: post-mask diag rid=%s idx=%d top10_ids=%s top10_decodes=%s active_after_the=%s prog=%s/%s",
                str(rid),
                i,
                top_ids2,
                decoded_tops2,
                str(bool(self.active_after_the[i].item())),
                str(int(self.suffix_progress[i].item())),
                str(len(self.suffix_seq_space or self.suffix_seq_nospace or [])),
            )

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
        is_start = ch in QUOTE_CHARS or ch in SENTENCE_END_CHARS
        logger.info(
            "BigramGuard: start-detect rid=%s out_len=%d last_ch='%s' is_start=%s tail=['%s']",
            str(getattr(req, "rid", None)),
            len(out_ids),
            ch,
            str(is_start),
            tail.replace("\n", "\\n"),
        )
        return is_start
