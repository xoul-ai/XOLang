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
from sglang.srt.sampling.penaltylib.vocab_cache import (
    get_bigram_cache,
    get_sentence_end_token_ids,
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
        # Only required if there are actually requests in the batch
        reqs = self.orchestrator.reqs()
        return reqs is not None and len(reqs) > 0

    def _prepare(self):
        reqs = self.orchestrator.reqs()
        device = self.orchestrator.device
        vocab_size = self.orchestrator.vocab_size

        # Per-request state tracking whether next token is immediately after "The" at a start
        # Keep per-request FSM/control state on CPU to avoid GPU syncs
        self.pending_after_the_at_start = torch.zeros(
            (len(reqs),), dtype=torch.bool
        )

        self.first_token_ids_per_req: List[Optional[torch.Tensor]] = [None] * len(reqs)
        self.first_token_ids_set_per_req: List[Optional[Set[int]]] = [None] * len(reqs)
        self.first_token_requires_space_per_req: List[Optional[Dict[int, bool]]] = [
            None
        ] * len(reqs)

        # Global token ID sets for blocking "the word" patterns
        self.single_token_blacklist: Optional[torch.Tensor] = None
        self.word_with_space_ids: Optional[torch.Tensor] = None
        self.word_no_space_ids: Optional[torch.Tensor] = None
        # Canonical sequences for multi-step suffix matching
        self.suffix_seq_space: Optional[List[int]] = None
        self.suffix_seq_nospace: Optional[List[int]] = None

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

        # FSM state for tracking multi-step suffix matching (after "The" token)
        self.active_after_the = torch.zeros((len(reqs),), dtype=torch.bool)
        self.suffix_variant_space = torch.zeros((len(reqs),), dtype=torch.bool)
        self.suffix_progress = torch.zeros((len(reqs),), dtype=torch.int32)
        self._last_hard_blocks: List[Optional[torch.Tensor]] = [None] * len(reqs)
        # Track start-of-sentence flags (CPU). Initialize BOS as start.
        self.prev_pos_is_start = torch.ones((len(reqs),), dtype=torch.bool)
        self.next_pos_is_start = torch.ones((len(reqs),), dtype=torch.bool)

        # PERFORMANCE: Use cached sentence-ending token IDs (built once per vocab)
        self.sentence_end_token_ids: Optional[Set[int]] = None
        if tokenizer0 is not None:
            self.sentence_end_token_ids = get_sentence_end_token_ids(
                tokenizer0, vocab_size, QUOTE_CHARS, SENTENCE_END_CHARS
            )

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        if output_ids is None or output_ids.numel() == 0:
            return
        reqs = self.orchestrator.reqs()
        if reqs is None:
            return
        # Clamp to avoid rare races with scheduler/filter/merge
        B_ids = int(output_ids.numel())
        L = min(
            len(reqs),
            B_ids,
            len(self.active_after_the),
            len(self.pending_after_the_at_start),
            len(self.suffix_variant_space),
            len(self.suffix_progress),
            len(self.first_token_ids_set_per_req),
            len(self.first_token_requires_space_per_req),
        )
        if L == 0:
            return
        # Take local views to reduce risk if self.* mutate during iteration
        active_after_the = self.active_after_the
        pending_after_the_at_start = self.pending_after_the_at_start
        suffix_variant_space = self.suffix_variant_space
        suffix_progress = self.suffix_progress
        first_token_ids_set_per_req = self.first_token_ids_set_per_req
        first_token_requires_space_per_req = self.first_token_requires_space_per_req

        for i in range(L):
            req = reqs[i]
            last_id = int(output_ids[i].item())

            # Update sentence-start flags using only the last emitted token
            if i < len(self.prev_pos_is_start):
                self.prev_pos_is_start[i] = (
                    self.next_pos_is_start[i]
                    if i < len(self.next_pos_is_start)
                    else torch.tensor(False)
                )
            # PERFORMANCE FIX: Use pre-computed set instead of decode()
            if i < len(self.next_pos_is_start):
                if self.sentence_end_token_ids is not None:
                    self.next_pos_is_start[i] = (last_id in self.sentence_end_token_ids)
                else:
                    # Fallback to decode if cache not available (shouldn't happen)
                    tok = getattr(req, "tokenizer", None)
                    if tok is not None:
                        try:
                            s = tok.decode([last_id])
                        except Exception:
                            s = ""
                        j = len(s) - 1
                        while j >= 0 and s[j].isspace():
                            j -= 1
                        if j >= 0:
                            ch = s[j]
                            self.next_pos_is_start[i] = bool(
                                (ch in QUOTE_CHARS) or (ch in SENTENCE_END_CHARS)
                            )
                        else:
                            self.next_pos_is_start[i] = False

            first_ids_set = first_token_ids_set_per_req[i]
            if not first_ids_set:
                continue
            if not (i < len(self.prev_pos_is_start) and bool(self.prev_pos_is_start[i].item())):
                continue
            if last_id in first_ids_set:
                if i < len(pending_after_the_at_start):
                    pending_after_the_at_start[i] = True
                if i < len(active_after_the):
                    active_after_the[i] = True
                # Determine whether next token needs space based on how "The" was encoded
                req_map = first_token_requires_space_per_req[i] or {}
                need_space_variant = bool(req_map.get(last_id, True))
                if i < len(suffix_variant_space):
                    suffix_variant_space[i] = need_space_variant
                if i < len(suffix_progress):
                    suffix_progress[i] = 0

            # Advance FSM for multi-step suffix matching
            if i < len(active_after_the) and bool(active_after_the[i].item()):
                if not (i < len(pending_after_the_at_start) and bool(pending_after_the_at_start[i].item())):
                    seq = (
                        self.suffix_seq_space
                        if (i < len(suffix_variant_space) and bool(suffix_variant_space[i].item()))
                        else self.suffix_seq_nospace
                    )
                    if seq is None or len(seq) == 0:
                        if i < len(active_after_the):
                            active_after_the[i] = False
                    else:
                        prog = int(suffix_progress[i].item()) if i < len(suffix_progress) else 0
                        if prog < len(seq):
                            if last_id == int(seq[prog]):
                                if i < len(suffix_progress):
                                    suffix_progress[i] = prog + 1
                            else:
                                if i < len(active_after_the):
                                    active_after_the[i] = False

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        B = logits.shape[0]
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
        if reqs is None:
            return logits
        # Clamp to the smallest consistent length to avoid rare race conditions
        L = min(
            B,
            len(reqs),
            len(active_after_the),
            len(pending_after_the_at_start),
            len(suffix_variant_space),
            len(suffix_progress),
            len(first_token_ids_set_per_req),
            len(last_hard_blocks),
        )
        if L == 0:
            return logits
        for j in range(L):
            last_hard_blocks[j] = None
        for i in range(L):
            req = reqs[i]
            out_ids_list = getattr(req, "output_ids", []) or []

            is_start_here = (len(out_ids_list) == 0) or (
                i < len(self.next_pos_is_start) and bool(self.next_pos_is_start[i].item())
            )
            if is_start_here and single_token_blacklist is not None:
                logits[i, single_token_blacklist] = -float("inf")
                if single_token_blacklist is not None and single_token_blacklist.numel() > 0:
                    last_hard_blocks[i] = single_token_blacklist

            just_after_the = False
            first_ids_set2 = first_token_ids_set_per_req[i]
            if first_ids_set2 and out_ids_list:
                last_id2 = int(out_ids_list[-1])
                if last_id2 in first_ids_set2:
                    is_start_here = (len(out_ids_list) == 1) or (
                        i < len(self.prev_pos_is_start)
                        and bool(self.prev_pos_is_start[i].item())
                    )
                else:
                    is_start_here = False
                if is_start_here:
                    just_after_the = True
                    # Use local views and guard against races that can shrink buffers
                    if i < len(active_after_the) and not bool(active_after_the[i].item()):
                        active_after_the[i] = True
                        req_map2 = first_token_requires_space_per_req[i] or {}
                        need_space_variant2 = bool(req_map2.get(last_id2, True))
                        if i < len(suffix_variant_space):
                            suffix_variant_space[i] = need_space_variant2
                        if i < len(suffix_progress):
                            suffix_progress[i] = 0

            if (i < len(pending_after_the_at_start) and pending_after_the_at_start[i]) or just_after_the:
                need_space_variant = (
                    bool(suffix_variant_space[i].item()) if i < len(suffix_variant_space) else False
                )
                if need_space_variant and word_with_space_ids is not None:
                    logits[i, word_with_space_ids] = -float("inf")
                    if word_with_space_ids is not None and word_with_space_ids.numel() > 0:
                        last_hard_blocks[i] = word_with_space_ids
                elif (not need_space_variant) and word_no_space_ids is not None:
                    logits[i, word_no_space_ids] = -float("inf")
                    if word_no_space_ids is not None and word_no_space_ids.numel() > 0:
                        last_hard_blocks[i] = word_no_space_ids

            if i < len(active_after_the) and bool(active_after_the[i].item()):
                seq = (
                    suffix_seq_space
                    if (i < len(suffix_variant_space) and bool(suffix_variant_space[i].item()))
                    else suffix_seq_nospace
                )
                if seq and len(seq) >= 2:
                    prog = int(suffix_progress[i].item()) if i < len(suffix_progress) else 0
                    if prog == len(seq) - 1:
                        next_tid = int(seq[prog])
                        logits[i, next_tid] = -float("inf")
                        try:
                            last_hard_blocks[i] = torch.tensor(
                                [next_tid], dtype=torch.int64, device=logits.device
                            )
                        except Exception:
                            pass

        return logits

    def get_last_hard_block_ids(self):
        if not self.is_prepared():
            return None
        return self._last_hard_blocks

    def get_computed_hard_block_ids(self):
        # Compute hard blocks for current step independent of local _apply timing
        if not self.is_prepared():
            return None
        reqs = self.orchestrator.reqs()
        if not reqs:
            return None
        out: List[Optional[torch.Tensor]] = [None] * len(reqs)
        # Clamp to available per-request state to avoid races
        L = min(
            len(reqs),
            len(self.pending_after_the_at_start),
            len(self.suffix_variant_space),
            len(self.first_token_ids_set_per_req),
            len(self.first_token_requires_space_per_req),
            len(self.prev_pos_is_start),
            len(self.next_pos_is_start),
        )
        for i in range(L):
            req = reqs[i]
            out_ids_list = getattr(req, "output_ids", []) or []

            is_start_here = (len(out_ids_list) == 0) or (
                i < len(self.next_pos_is_start) and bool(self.next_pos_is_start[i].item())
            )
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
            pending = (
                bool(self.pending_after_the_at_start[i].item())
                if i < len(self.pending_after_the_at_start)
                else False
            )

            if pending:
                need_space_variant = (
                    bool(self.suffix_variant_space[i].item())
                    if i < len(self.suffix_variant_space)
                    else False
                )
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
            if decided:
                continue
            # Path B: infer from request state if cumulated state not set yet on this rank
            first_ids_set2 = self.first_token_ids_set_per_req[i]
            if first_ids_set2 and out_ids_list:
                last_id2 = int(out_ids_list[-1])
                if last_id2 in first_ids_set2:
                    is_start_here = (len(out_ids_list) == 1) or (
                        i < len(self.prev_pos_is_start)
                        and bool(self.prev_pos_is_start[i].item())
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

        return out

    def _filter(self, keep_indices: torch.Tensor):
        keep = keep_indices.cpu()
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
        self.active_after_the = self.active_after_the[keep]
        self.suffix_variant_space = self.suffix_variant_space[keep]
        self.suffix_progress = self.suffix_progress[keep]
        # Keep start flags in sync
        if hasattr(self, "prev_pos_is_start") and self.prev_pos_is_start.numel() >= keep.numel():
            self.prev_pos_is_start = self.prev_pos_is_start[keep]
        if hasattr(self, "next_pos_is_start") and self.next_pos_is_start.numel() >= keep.numel():
            self.next_pos_is_start = self.next_pos_is_start[keep]
        self._last_hard_blocks = [self._last_hard_blocks[j] for j in keep.tolist()]

    def _merge(self, their: "BatchedFixedBigramStartGuardPenalizer"):

        self.pending_after_the_at_start = torch.cat(
            [self.pending_after_the_at_start, their.pending_after_the_at_start], dim=0
        )
        self.first_token_ids_per_req.extend(their.first_token_ids_per_req)
        self.first_token_ids_set_per_req.extend(their.first_token_ids_set_per_req)
        self.first_token_requires_space_per_req.extend(
            their.first_token_requires_space_per_req
        )
        self.active_after_the = torch.cat(
            [self.active_after_the, their.active_after_the], dim=0
        )
        self.suffix_variant_space = torch.cat(
            [self.suffix_variant_space, their.suffix_variant_space], dim=0
        )
        self.suffix_progress = torch.cat(
            [self.suffix_progress, their.suffix_progress], dim=0
        )
        # Merge start flags
        if hasattr(self, "prev_pos_is_start") and hasattr(their, "prev_pos_is_start"):
            self.prev_pos_is_start = torch.cat(
                [self.prev_pos_is_start, their.prev_pos_is_start], dim=0
            )
        if hasattr(self, "next_pos_is_start") and hasattr(their, "next_pos_is_start"):
            self.next_pos_is_start = torch.cat(
                [self.next_pos_is_start, their.next_pos_is_start], dim=0
            )
        self._last_hard_blocks.extend(their._last_hard_blocks)

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
