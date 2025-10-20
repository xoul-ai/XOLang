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
from sglang.srt.sampling.penaltylib.vocab_cache import get_bigram_cache

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
        from sglang.srt.sampling.penaltylib.toggle import DEBUG

        reqs = self.orchestrator.reqs()
        device = self.orchestrator.device
        vocab_size = self.orchestrator.vocab_size

        if DEBUG:
            logger.info(
                f"BigramGuard _prepare: preparing with {len(reqs)} reqs, self_id={id(self)}, orch_id={id(self.orchestrator)}"
            )

        # Per-request state used across steps
        self.pending_after_the_at_start = torch.zeros(
            (len(reqs),), dtype=torch.bool, device=device
        )

        # Precomputed token ID sets shared across requests (move to device at prepare)
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
        # Use the first non-None tokenizer. If none available, skip.
        tokenizer0 = None
        for r in reqs:
            tok = getattr(r, "tokenizer", None)
            if tok is not None:
                tokenizer0 = tok
                break

        if tokenizer0 is not None:
            cache = get_bigram_cache(tokenizer0, vocab_size, QUOTE_CHARS)
            if cache.single_token_blacklist:
                self.single_token_blacklist = torch.tensor(
                    cache.single_token_blacklist, dtype=torch.int64, device=device
                )
            if cache.word_with_space_ids:
                self.word_with_space_ids = torch.tensor(
                    cache.word_with_space_ids, dtype=torch.int64, device=device
                )
            if cache.word_no_space_ids:
                self.word_no_space_ids = torch.tensor(
                    cache.word_no_space_ids, dtype=torch.int64, device=device
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

        if tokenizer0 is not None:
            cache = get_bigram_cache(tokenizer0, vocab_size, QUOTE_CHARS)
            first_ids_sorted = sorted(cache.the_first_token_ids)
            first_ids_tensor = (
                torch.tensor(first_ids_sorted, dtype=torch.int64, device=device)
                if first_ids_sorted
                else None
            )
            for i, _ in enumerate(reqs):
                if first_ids_tensor is not None:
                    self.first_token_ids_per_req[i] = first_ids_tensor
                    self.first_token_ids_set_per_req[i] = set(cache.the_first_token_ids)
                    self.first_token_requires_space_per_req[i] = dict(cache.requires_space)
                else:
                    self.first_token_ids_per_req[i] = None
                    self.first_token_ids_set_per_req[i] = None
                    self.first_token_requires_space_per_req[i] = None

        # Initialize per-request FSM for suffix multi-step tracking
        self.active_after_the = torch.zeros(
            (len(reqs),), dtype=torch.bool, device=device
        )
        self.suffix_variant_space = torch.zeros(
            (len(reqs),), dtype=torch.bool, device=device
        )  # True => use space variant
        self.suffix_progress = torch.zeros(
            (len(reqs),), dtype=torch.int32, device=device
        )
        # Track hard blocks applied at last step per request
        self._last_hard_blocks: List[Optional[torch.Tensor]] = [None] * len(reqs)

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        from sglang.srt.sampling.penaltylib.toggle import DEBUG

        # Track if we just emitted a first-token for "The" at a start position
        if output_ids is None or output_ids.numel() == 0:
            return
        reqs = self.orchestrator.reqs()
        # If reqs unavailable (e.g., weakref died after pickling), skip
        if reqs is None:
            if DEBUG:
                logger.info(
                    f"BigramGuard _cumulate: reqs is None, skipping, self_id={id(self)}, orch_id={id(self.orchestrator)}"
                )
            return
        # Check for batch size mismatch (can happen during filter/merge)
        if len(reqs) != len(self.active_after_the):
            if DEBUG:
                logger.info(
                    f"BigramGuard _cumulate: batch size mismatch, reqs={len(reqs)} vs active_after_the={len(self.active_after_the)}, skipping, self_id={id(self)}, orch_id={id(self.orchestrator)}"
                )
            return
        for i, req in enumerate(reqs):
            tok = getattr(req, "tokenizer", None)
            last_id = int(output_ids[i].item())
            rid = getattr(req, "rid", None)

            # LOG: Show what token was actually generated
            if tok is not None:
                try:
                    decoded = tok.decode([last_id])

                except Exception:
                    pass

            first_ids_set = self.first_token_ids_set_per_req[i]
            if not first_ids_set:
                continue
            # Only trigger at sentence/reply starts (relative to context before last token)
            if not self._is_start_position_before_last(req):
                continue
            if int(last_id) in first_ids_set:
                self.pending_after_the_at_start[i] = True
                self.active_after_the[i] = True
                # Determine variant at the moment we observed the first token
                req_map = self.first_token_requires_space_per_req[i] or {}
                need_space_variant = bool(req_map.get(last_id, True))
                self.suffix_variant_space[i] = need_space_variant
                self.suffix_progress[i] = 0

            # If already active (after seeing first token), advance FSM with current token
            if bool(self.active_after_the[i].item()):
                # Initialize variant and reset progress if this is the first step after 'The'
                if not bool(self.pending_after_the_at_start[i].item()):
                    # Advance matching on subsequent steps
                    seq = (
                        self.suffix_seq_space
                        if bool(self.suffix_variant_space[i].item())
                        else self.suffix_seq_nospace
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
        B = logits.shape[0]

        # Copy ALL tensor/list references at start to prevent race conditions
        active_after_the = self.active_after_the
        suffix_variant_space = self.suffix_variant_space
        suffix_progress = self.suffix_progress
        first_token_ids_set_per_req = self.first_token_ids_set_per_req
        last_hard_blocks = self._last_hard_blocks
        single_token_blacklist = self.single_token_blacklist
        pending_after_the_at_start = self.pending_after_the_at_start
        word_with_space_ids = self.word_with_space_ids
        word_no_space_ids = self.word_no_space_ids
        first_token_requires_space_per_req = self.first_token_requires_space_per_req
        suffix_seq_space = self.suffix_seq_space
        suffix_seq_nospace = self.suffix_seq_nospace

        reqs = self.orchestrator.reqs()
        # If reqs unavailable or batch size mismatch, skip
        if reqs is None:
            orch_unique_id = getattr(self.orchestrator, "_unique_id", "NO_ID")
            logger.error(
                f"BigramGuard _apply: reqs is None! orch_unique_id={orch_unique_id}, backup_reqs={self.orchestrator._backup_reqs}, batch_ref_alive={self.orchestrator.batch is not None}"
            )
            return logits
        from sglang.srt.sampling.penaltylib.toggle import DEBUG

        if len(reqs) != B:
            if DEBUG:
                logger.warning(f"BigramGuard _apply: reqs size mismatch, len(reqs)={len(reqs)} vs B={B}")
            return logits
        # Check ALL tensor sizes match batch size B
        if len(active_after_the) != B:
            if DEBUG:
                logger.warning(f"BigramGuard _apply: active_after_the size mismatch, len={len(active_after_the)} vs B={B}")
            return logits
        if len(pending_after_the_at_start) != B:
            if DEBUG:
                logger.warning(f"BigramGuard _apply: pending_after_the_at_start size mismatch, len={len(pending_after_the_at_start)} vs B={B}")
            return logits
        if len(suffix_variant_space) != B:
            if DEBUG:
                logger.warning(f"BigramGuard _apply: suffix_variant_space size mismatch, len={len(suffix_variant_space)} vs B={B}")
            return logits
        if len(suffix_progress) != B:
            if DEBUG:
                logger.warning(f"BigramGuard _apply: suffix_progress size mismatch, len={len(suffix_progress)} vs B={B}")
            return logits
        # Check list sizes as well (these are not tensors, so they need separate validation)
        if len(first_token_ids_set_per_req) != B:
            if DEBUG:
                logger.warning(f"BigramGuard _apply: first_token_ids_set_per_req size mismatch, len={len(first_token_ids_set_per_req)} vs B={B}")
            return logits
        if len(last_hard_blocks) != B:
            if DEBUG:
                logger.warning(f"BigramGuard _apply: last_hard_blocks size mismatch, len={len(last_hard_blocks)} vs B={B}")
            return logits
        # Reset last hard-blocks
        for j in range(len(reqs)):
            last_hard_blocks[j] = None
        for i, req in enumerate(reqs):
            # BOS/start: Hard block single-token candidates that decode to "the word..." (boundary-aware)
            out_ids_list = getattr(req, "output_ids", []) or []
            rid = getattr(req, "rid", None)
            tok = getattr(req, "tokenizer", None)

            is_start_here = (len(out_ids_list) == 0) or self._is_start_position(req)
            if is_start_here and single_token_blacklist is not None:
                logits[i, single_token_blacklist] = -float("inf")
                if (
                    single_token_blacklist is not None
                    and single_token_blacklist.numel() > 0
                ):
                    last_hard_blocks[i] = single_token_blacklist
                    orch_unique_id = getattr(self.orchestrator, "_unique_id", "NO_ID")

            # Two-step guard: proactively detect if we are immediately after a THE token at a start,
            # even if cumulate missed due to overlap scheduling. If so, set variant and apply masks.
            just_after_the = False
            first_ids_set2 = first_token_ids_set_per_req[i]
            if first_ids_set2 and out_ids_list:
                # We consider we are at start of sentence if out_len == 1 or if start-detect
                # on the context BEFORE the last token is True
                last_id2 = int(out_ids_list[-1])
                if last_id2 in first_ids_set2:
                    if len(out_ids_list) == 1:
                        is_start_here = True
                    else:
                        # Reuse start detection logic (relative to prefix)
                        is_start_here = self._is_start_position_before_last(req)
                else:
                    is_start_here = False
                if is_start_here:
                    just_after_the = True
                    decoded_last = ""
                    if tok:
                        try:
                            decoded_last = tok.decode([last_id2])
                        except:
                            pass
                    # Establish variant if not already set
                    if not bool(active_after_the[i].item()):
                        self.active_after_the[i] = True
                        req_map2 = first_token_requires_space_per_req[i] or {}
                        need_space_variant2 = bool(req_map2.get(last_id2, True))
                        self.suffix_variant_space[i] = need_space_variant2
                        self.suffix_progress[i] = 0

            # Two-step guard: if flagged pending or detected just-after-the, block appropriate second-token IDs
            if pending_after_the_at_start[i] or just_after_the:
                # Use variant decided at cumulate time to avoid drift
                need_space_variant = bool(suffix_variant_space[i].item())
                orch_unique_id = getattr(self.orchestrator, "_unique_id", "NO_ID")
                if need_space_variant and word_with_space_ids is not None:
                    logits[i, word_with_space_ids] = -float("inf")
                    if (
                        word_with_space_ids is not None
                        and word_with_space_ids.numel() > 0
                    ):
                        last_hard_blocks[i] = word_with_space_ids
                elif (not need_space_variant) and word_no_space_ids is not None:
                    logits[i, word_no_space_ids] = -float("inf")
                    if word_no_space_ids is not None and word_no_space_ids.numel() > 0:
                        last_hard_blocks[i] = word_no_space_ids

                # Do not reset pending flag here; allow sampler compute-now to observe it reliably

            # Multi-step guard: if we're in active suffix matching state and are at the point
            # where emitting the next token would complete the suffix (canonical encoding),
            # then hard-block that specific token id.
            if bool(active_after_the[i].item()):
                seq = (
                    suffix_seq_space
                    if bool(suffix_variant_space[i].item())
                    else suffix_seq_nospace
                )
                if seq and len(seq) >= 2:
                    prog = int(suffix_progress[i].item())
                    # If we have matched first k tokens of the suffix and k == len(seq) - 1,
                    # then the next token uniquely completes " word" or "word".
                    if prog == len(seq) - 1:
                        next_tid = int(seq[prog])
                        logits[i, next_tid] = -float("inf")
                        try:
                            last_hard_blocks[i] = torch.tensor(
                                [next_tid], dtype=torch.int64, device=logits.device
                            )
                        except Exception:
                            pass
                        rid = getattr(req, "rid", None)

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

            # BOS/start: block any single-token that decodes to "the word..." if we have it
            is_start_here = (len(out_ids_list) == 0) or self._is_start_position(req)
            if (
                is_start_here
                and self.single_token_blacklist is not None
                and self.single_token_blacklist.numel() > 0
            ):
                out[i] = self.single_token_blacklist
                continue
            # Two-step guard: use cumulated state to decide immediately-after-THE-at-start
            decided = False
            # Path A: use cumulated state if available
            try:
                pending = bool(self.pending_after_the_at_start[i].item())

                if pending:
                    need_space_variant = bool(self.suffix_variant_space[i].item())
                    if (
                        need_space_variant
                        and self.word_with_space_ids is not None
                        and self.word_with_space_ids.numel() > 0
                    ):
                        out[i] = self.word_with_space_ids
                        decided = True
                    elif (
                        (not need_space_variant)
                        and self.word_no_space_ids is not None
                        and self.word_no_space_ids.numel() > 0
                    ):
                        out[i] = self.word_no_space_ids
                        decided = True
            except Exception as e:
                from sglang.srt.sampling.penaltylib.toggle import DEBUG
                if DEBUG:
                    logger.info(
                        f"BigramGuard COMPUTE_NOW_PATH_A: rid={rid} idx={i} exception={e}"
                    )
            if decided:
                continue
            # Path B: infer from request state if cumulated state not set yet on this rank
            first_ids_set2 = self.first_token_ids_set_per_req[i]
            sample_first_ids = (
                sorted(list(first_ids_set2))[:10] if first_ids_set2 else []
            )

            if first_ids_set2 and out_ids_list:
                last_id2 = int(out_ids_list[-1])
                if last_id2 in first_ids_set2:
                    is_start_here = (
                        len(out_ids_list) == 1 or self._is_start_position_before_last(req)
                    )
                else:
                    is_start_here = False

                if is_start_here:
                    req_map2 = self.first_token_requires_space_per_req[i] or {}
                    need_space_variant2 = bool(req_map2.get(last_id2, True))

                    if (
                        need_space_variant2
                        and self.word_with_space_ids is not None
                        and self.word_with_space_ids.numel() > 0
                    ):
                        out[i] = self.word_with_space_ids

                    elif (
                        (not need_space_variant2)
                        and self.word_no_space_ids is not None
                        and self.word_no_space_ids.numel() > 0
                    ):
                        out[i] = self.word_no_space_ids

        # Final logging
        non_none_count = sum(1 for x in out if x is not None)

        return out

    def _filter(self, keep_indices: torch.Tensor):
        from sglang.srt.sampling.penaltylib.toggle import DEBUG

        keep = keep_indices
        old_len = len(self.active_after_the) if hasattr(self, "active_after_the") else 0
        if DEBUG:
            logger.info(
                f"BigramGuard _filter: filtering from {old_len} to {len(keep)}, self_id={id(self)}, orch_id={id(self.orchestrator)}"
            )

        self.pending_after_the_at_start = self.pending_after_the_at_start[keep]
        self.first_token_ids_per_req = [
            self.first_token_ids_per_req[j] for j in keep.tolist()
        ]
        self.first_token_ids_set_per_req = [
            self.first_token_ids_set_per_req[j] for j in keep.tolist()
        ]
        self.first_token_requires_space_per_req = [
            self.first_token_requires_space_per_req[j] for j in keep.tolist()
        ]
        # Filter FSM tracking tensors
        self.active_after_the = self.active_after_the[keep]
        self.suffix_variant_space = self.suffix_variant_space[keep]
        self.suffix_progress = self.suffix_progress[keep]
        self._last_hard_blocks = [self._last_hard_blocks[j] for j in keep.tolist()]

        new_len = len(self.active_after_the)
        if DEBUG:
            logger.info(
                f"BigramGuard _filter: filtered to {new_len}, self_id={id(self)}, orch_id={id(self.orchestrator)}"
            )

    def _merge(self, their: "BatchedFixedBigramStartGuardPenalizer"):
        from sglang.srt.sampling.penaltylib.toggle import DEBUG

        old_len = len(self.active_after_the) if hasattr(self, "active_after_the") else 0
        their_len = (
            len(their.active_after_the) if hasattr(their, "active_after_the") else 0
        )
        if DEBUG:
            logger.info(
                f"BigramGuard _merge: merging {old_len} + {their_len}, self_id={id(self)}, orch_id={id(self.orchestrator)}"
            )

        self.pending_after_the_at_start = torch.cat(
            [self.pending_after_the_at_start, their.pending_after_the_at_start], dim=0
        )
        self.first_token_ids_per_req.extend(their.first_token_ids_per_req)
        self.first_token_ids_set_per_req.extend(their.first_token_ids_set_per_req)
        self.first_token_requires_space_per_req.extend(
            their.first_token_requires_space_per_req
        )
        # Merge FSM tracking tensors
        self.active_after_the = torch.cat(
            [self.active_after_the, their.active_after_the], dim=0
        )
        self.suffix_variant_space = torch.cat(
            [self.suffix_variant_space, their.suffix_variant_space], dim=0
        )
        self.suffix_progress = torch.cat(
            [self.suffix_progress, their.suffix_progress], dim=0
        )
        self._last_hard_blocks.extend(their._last_hard_blocks)

        new_len = len(self.active_after_the)
        if DEBUG:
            logger.info(
                f"BigramGuard _merge: merged to {new_len}, self_id={id(self)}, orch_id={id(self.orchestrator)}"
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

    def _is_start_position_before_last(self, req) -> bool:
        """Check start-of-sentence relative to the position BEFORE the last token.

        This is critical for detecting that a just-emitted THE-like token occurred at a
        sentence start; checking the full tail including the last token would see an alpha
        char and miss the boundary. Here we examine the context prior to the last token.
        """
        out_ids = getattr(req, "output_ids", None) or []
        if len(out_ids) == 0:
            return True
        if len(out_ids) == 1:
            # Before the first token is BOS, which we treat as start
            return True
        tok = getattr(req, "tokenizer", None)
        if tok is None:
            return False
        prefix = out_ids[:-1]
        n = min(12, len(prefix))
        try:
            tail = tok.decode(prefix[-n:])
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
        return (ch in QUOTE_CHARS) or (ch in SENTENCE_END_CHARS)
