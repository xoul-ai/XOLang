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
                # Normalize SentencePiece space character (U+2581) to regular space
                s_norm = s.replace("\u2581", " ")
                s_low = s_norm.lstrip().lower()
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
                # Normalize SentencePiece space character (U+2581) to regular space
                s_norm = s.replace("\u2581", " ")
                sl = s_norm.lower()
                if sl.startswith(" word"):
                    # check boundary after word
                    pos = len(" word")
                    if pos < len(sl) and sl[pos : pos + 1].isalpha():
                        pass
                    else:
                        w_space_ids.add(int(tid))
                        logger.info(f"BigramGuard VOCAB: FOUND w_space tid={tid} raw={repr(s)} norm={repr(s_norm)} lower={repr(sl)}")
                if sl.startswith("word"):
                    pos = len("word")
                    if pos < len(sl) and sl[pos : pos + 1].isalpha():
                        pass
                    else:
                        no_space_ids.add(int(tid))
                        logger.info(f"BigramGuard VOCAB: FOUND no_space tid={tid} raw={repr(s)} norm={repr(s_norm)} lower={repr(sl)}")
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

            # Augment: scan entire vocab for tokens that decode to a THE-like start after stripping leading quotes/spaces
            try:
                vocab_size = self.orchestrator.vocab_size
            except Exception:
                vocab_size = None
            if vocab_size is not None:
                ignore_leading = " "
                for q in QUOTE_CHARS:
                    ignore_leading += q
                added = 0
                add_cap = 8192
                for tid in range(vocab_size):
                    if tid in first_ids:
                        continue
                    try:
                        s = tok.decode([tid])
                    except Exception:
                        continue
                    if not s:
                        continue
                    k = 0
                    L = len(s)
                    while k < L and (s[k] in ignore_leading):
                        k += 1
                    if k >= L:
                        continue
                    ch0 = s[k]
                    if not ("A" <= ch0 <= "Z" or "a" <= ch0 <= "z"):
                        continue
                    rem = s[k:].lower()
                    # Require boundary after 'the'
                    if rem.startswith("the"):
                        end = 3
                        if end < len(rem) and rem[end : end + 1].isalpha():
                            continue
                        fid = int(tid)
                        first_ids.add(fid)
                        added += 1
                        # Most THE tokens won't end with space; default mapping
                        if fid not in requires_space:
                            requires_space[fid] = True
                    if added >= add_cap:
                        break

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
        # Track hard blocks applied at last step per request
        self._last_hard_blocks: List[Optional[torch.Tensor]] = [None] * len(reqs)

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        # Track if we just emitted a first-token for "The" at a start position
        reqs = self.orchestrator.reqs()
        if output_ids is None or output_ids.numel() == 0:
            return
        for i, req in enumerate(reqs):
            tok = getattr(req, "tokenizer", None)
            last_id = int(output_ids[i].item())
            rid = getattr(req, "rid", None)

            # LOG: Show what token was actually generated
            if tok is not None:
                try:
                    decoded = tok.decode([last_id])
                    logger.info(
                        "BigramGuard CUMULATE: rid=%s idx=%d generated_tid=%d decoded=%s",
                        str(rid),
                        i,
                        last_id,
                        repr(decoded),
                    )
                except Exception:
                    pass

            first_ids_set = self.first_token_ids_set_per_req[i]
            if not first_ids_set:
                continue
            # Only trigger at sentence/reply starts
            if not self._is_start_position(req):
                continue
            if int(last_id) in first_ids_set:
                self.pending_after_the_at_start[i] = True
                self.active_after_the[i] = True
                # Determine variant at the moment we observed the first token
                req_map = self.first_token_requires_space_per_req[i] or {}
                need_space_variant = bool(req_map.get(last_id, True))
                self.suffix_variant_space[i] = need_space_variant
                self.suffix_progress[i] = 0
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

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        # BOS single-token hard block and two-step bigram guard
        reqs = self.orchestrator.reqs()
        # If reqs unavailable (e.g., weakref died after pickling), skip
        if reqs is None:
            return logits
        # Reset last hard-blocks
        for j in range(len(reqs)):
            self._last_hard_blocks[j] = None
        for i, req in enumerate(reqs):
            # BOS: Hard block single-token candidates that decode to "the word..." (boundary-aware)
            out_ids_list = getattr(req, "output_ids", []) or []
            rid = getattr(req, "rid", None)
            if len(out_ids_list) == 0 and self.single_token_blacklist is not None:
                logits[i, self.single_token_blacklist] = -float("inf")
                if self.single_token_blacklist is not None and self.single_token_blacklist.numel() > 0:
                    self._last_hard_blocks[i] = self.single_token_blacklist
                logger.info(
                    "BigramGuard: applied BOS single-token mask rid=%s idx=%d masked=%d",
                    str(rid),
                    i,
                    int(self.single_token_blacklist.numel()),
                )
            tok = getattr(req, "tokenizer", None)

            # Two-step guard: proactively detect if we are immediately after a THE token at a start,
            # even if cumulate missed due to overlap scheduling. If so, set variant and apply masks.
            just_after_the = False
            first_ids_set2 = self.first_token_ids_set_per_req[i]
            if first_ids_set2 and out_ids_list:
                # We consider we are at start of sentence if out_len == 1 or if start-detect on tail of decoded prefix is True
                is_start_here = False
                if len(out_ids_list) == 1:
                    is_start_here = True
                else:
                    # Reuse start detection logic
                    is_start_here = self._is_start_position(req)
                last_id2 = int(out_ids_list[-1])
                if is_start_here and (last_id2 in first_ids_set2):
                    just_after_the = True
                    # Establish variant if not already set
                    if not bool(self.active_after_the[i].item()):
                        self.active_after_the[i] = True
                        req_map2 = self.first_token_requires_space_per_req[i] or {}
                        need_space_variant2 = bool(req_map2.get(last_id2, True))
                        self.suffix_variant_space[i] = need_space_variant2
                        self.suffix_progress[i] = 0
                        logger.info(
                            "BigramGuard: detected post-THE at apply rid=%s idx=%d last_id=%d need_space_variant=%s",
                            str(rid),
                            i,
                            last_id2,
                            str(need_space_variant2),
                        )

            # Two-step guard: if flagged pending or detected just-after-the, block appropriate second-token IDs
            if self.pending_after_the_at_start[i] or just_after_the:
                # Use variant decided at cumulate time to avoid drift
                need_space_variant = bool(self.suffix_variant_space[i].item())
                if need_space_variant and self.word_with_space_ids is not None:
                    logits[i, self.word_with_space_ids] = -float("inf")
                    if self.word_with_space_ids is not None and self.word_with_space_ids.numel() > 0:
                        self._last_hard_blocks[i] = self.word_with_space_ids
                    logger.info(
                        "BigramGuard: blocked second token set variant=space rid=%s idx=%d size=%d blocked_tids=%s",
                        str(rid),
                        i,
                        int(self.word_with_space_ids.numel()),
                        self.word_with_space_ids.tolist(),
                    )
                    # LOG: Verify blocking worked by checking specific logit values
                    blocked_vals = logits[i, self.word_with_space_ids].tolist()
                    logger.info(
                        "BigramGuard VERIFY_BLOCK: rid=%s idx=%d blocked_tids=%s blocked_logit_values=%s",
                        str(rid),
                        i,
                        self.word_with_space_ids.tolist(),
                        blocked_vals,
                    )
                    # LOG: Show top candidates after blocking
                    top_k = torch.topk(logits[i], k=10)
                    top_ids = top_k.indices.tolist()
                    top_vals = top_k.values.tolist()
                    if tok is not None:
                        try:
                            top_decoded = [tok.decode([tid]) for tid in top_ids]
                            logger.info(
                                "BigramGuard AFTER_BLOCK: rid=%s idx=%d top10_tids=%s top10_decoded=%s",
                                str(rid),
                                i,
                                top_ids,
                                [repr(d) for d in top_decoded],
                            )
                        except Exception:
                            pass
                elif (not need_space_variant) and self.word_no_space_ids is not None:
                    logits[i, self.word_no_space_ids] = -float("inf")
                    if self.word_no_space_ids is not None and self.word_no_space_ids.numel() > 0:
                        self._last_hard_blocks[i] = self.word_no_space_ids
                    logger.info(
                        "BigramGuard: blocked second token set variant=no_space rid=%s idx=%d size=%d blocked_tids=%s",
                        str(rid),
                        i,
                        int(self.word_no_space_ids.numel()),
                        self.word_no_space_ids.tolist(),
                    )
                # Do not reset pending flag here; allow sampler compute-now to observe it reliably

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
                        try:
                            self._last_hard_blocks[i] = torch.tensor([next_tid], dtype=torch.int64, device=logits.device)
                        except Exception:
                            pass
                        rid = getattr(req, "rid", None)
                        logger.info(
                            "BigramGuard: blocked multi-step completion rid=%s idx=%d next_tid=%d variant_space=%s",
                            str(rid),
                            i,
                            next_tid,
                            str(bool(self.suffix_variant_space[i].item())),
                        )
        return logits

    def get_last_hard_block_ids(self):
        return self._last_hard_blocks

    def get_computed_hard_block_ids(self):
        # Compute hard blocks for current step independent of local _apply timing
        reqs = self.orchestrator.reqs()
        if not reqs:
            return None
        out: List[Optional[torch.Tensor]] = [None] * len(reqs)
        for i, req in enumerate(reqs):
            out_ids_list = getattr(req, "output_ids", []) or []
            rid = getattr(req, "rid", None)
            # LOG: Always log what we see in output_ids for first request
            if i == 0:
                logger.info(
                    f"BigramGuard COMPUTE_NOW_ENTRY: rid={rid} idx={i} out_ids_len={len(out_ids_list)} last_few_ids={out_ids_list[-5:] if len(out_ids_list) >= 5 else out_ids_list}"
                )
            # BOS: block any single-token that decodes to "the word..." if we have it
            if len(out_ids_list) == 0 and self.single_token_blacklist is not None and self.single_token_blacklist.numel() > 0:
                out[i] = self.single_token_blacklist
                continue
            # Two-step guard: use cumulated state to decide immediately-after-THE-at-start
            decided = False
            # Path A: use cumulated state if available
            try:
                pending = bool(self.pending_after_the_at_start[i].item())
                logger.info(f"BigramGuard COMPUTE_NOW_PATH_A: rid={rid} idx={i} pending_after_the={pending}")
                if pending:
                    need_space_variant = bool(self.suffix_variant_space[i].item())
                    if need_space_variant and self.word_with_space_ids is not None and self.word_with_space_ids.numel() > 0:
                        out[i] = self.word_with_space_ids
                        decided = True
                    elif (not need_space_variant) and self.word_no_space_ids is not None and self.word_no_space_ids.numel() > 0:
                        out[i] = self.word_no_space_ids
                        decided = True
            except Exception as e:
                logger.info(f"BigramGuard COMPUTE_NOW_PATH_A: rid={rid} idx={i} exception={e}")
            if decided:
                logger.info(
                    f"BigramGuard COMPUTE_NOW: rid={getattr(req,'rid',None)} idx={i} path=A pending_after_the={bool(self.pending_after_the_at_start[i].item())} variant_space={bool(self.suffix_variant_space[i].item())} ids={(out[i][:min(8,out[i].numel())].tolist() if out[i] is not None else [])}"
                )
                continue
            # Path B: infer from request state if cumulated state not set yet on this rank
            first_ids_set2 = self.first_token_ids_set_per_req[i]
            sample_first_ids = sorted(list(first_ids_set2))[:10] if first_ids_set2 else []
            logger.info(f"BigramGuard COMPUTE_NOW_PATH_B: rid={rid} idx={i} has_first_ids_set={first_ids_set2 is not None} set_size={len(first_ids_set2) if first_ids_set2 else 0} sample_ids={sample_first_ids} out_ids_len={len(out_ids_list)}")
            if first_ids_set2 and out_ids_list:
                is_start_here = len(out_ids_list) == 1 or self._is_start_position(req)
                logger.info(f"BigramGuard COMPUTE_NOW_PATH_B: rid={rid} idx={i} is_start_here={is_start_here} out_ids_len={len(out_ids_list)}")
                if is_start_here:
                    last_id2 = int(out_ids_list[-1])
                    is_match = last_id2 in first_ids_set2
                    logger.info(f"BigramGuard COMPUTE_NOW_PATH_B: rid={rid} idx={i} last_id2={last_id2} is_in_first_ids_set={is_match} first_ids_set_size={len(first_ids_set2)}")
                    if is_match:
                        req_map2 = self.first_token_requires_space_per_req[i] or {}
                        need_space_variant2 = bool(req_map2.get(last_id2, True))
                        logger.info(f"BigramGuard COMPUTE_NOW_PATH_B: rid={rid} idx={i} MATCHED! need_space_variant={need_space_variant2}")
                        if need_space_variant2 and self.word_with_space_ids is not None and self.word_with_space_ids.numel() > 0:
                            out[i] = self.word_with_space_ids
                            logger.info(
                                f"BigramGuard COMPUTE_NOW: rid={getattr(req,'rid',None)} idx={i} path=B variant=space ids={self.word_with_space_ids[:min(8,self.word_with_space_ids.numel())].tolist()}"
                            )
                        elif (not need_space_variant2) and self.word_no_space_ids is not None and self.word_no_space_ids.numel() > 0:
                            out[i] = self.word_no_space_ids
                            logger.info(
                                f"BigramGuard COMPUTE_NOW: rid={getattr(req,'rid',None)} idx={i} path=B variant=no_space ids={self.word_no_space_ids[:min(8,self.word_no_space_ids.numel())].tolist()}"
                            )
                    else:
                        logger.info(f"BigramGuard COMPUTE_NOW_PATH_B: rid={rid} idx={i} last_id2={last_id2} NOT in first_ids_set (576 in set? {576 in first_ids_set2})")
        # Final logging
        non_none_count = sum(1 for x in out if x is not None)
        logger.info(f"BigramGuard COMPUTE_NOW_RETURN: returning list with {non_none_count} non-None entries out of {len(out)}")
        return out

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
        return is_start
