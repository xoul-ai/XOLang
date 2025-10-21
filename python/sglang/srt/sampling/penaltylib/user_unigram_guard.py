from __future__ import annotations

import logging
import re
from typing import List, Optional, Set

import torch

logger = logging.getLogger(__name__)

from sglang.srt.sampling.penaltylib.orchestrator import (
    BatchedPenalizerOrchestrator,
    _BatchedPenalizer,
)
from sglang.srt.sampling.penaltylib.constants import (
    STOPWORDS,
    SENTENCE_END_CHARS,
    QUOTE_CHARS,
)
from sglang.srt.sampling.penaltylib.vocab_cache import (
    get_unigram_first_word_index,
    get_unigram_prefix_index,
)


class BatchedUserUnigramStartGuardPenalizer(_BatchedPenalizer):
    """Biases/blocks starting tokens that match user unigrams."""

    # Characters that should count as a start when appearing before the next token.
    # Historically included '*', keep it for markdown/emphasis scenarios.
    _OPENING_QUOTES = QUOTE_CHARS

    def __init__(self, orchestrator: BatchedPenalizerOrchestrator):
        self.orchestrator = orchestrator
        self._is_prepared = False

    def _is_required(self) -> bool:
        reqs = self.orchestrator.reqs()
        if not reqs:
            return False
        for req in reqs:
            cp = getattr(req.sampling_params, "custom_params", None)
            if isinstance(cp, dict) and cp.get("unigrams_text"):
                return True
        return False

    def _prepare(self):
        reqs = self.orchestrator.reqs()
        device = self.orchestrator.device

        self.guard_window: torch.Tensor = torch.zeros(
            (len(reqs),), dtype=torch.int32, device=device
        )
        self.hard_at_bos: torch.Tensor = torch.zeros(
            (len(reqs),), dtype=torch.bool, device=device
        )
        self.hard_at_all_starts: torch.Tensor = torch.zeros(
            (len(reqs),), dtype=torch.bool, device=device
        )
        # Optional: apply prefix-neighbor hard block at starts
        self.hard_prefix_at_starts: torch.Tensor = torch.zeros(
            (len(reqs),), dtype=torch.bool, device=device
        )
        self.bias_vals: torch.Tensor = torch.zeros(
            (len(reqs),), dtype=torch.float32, device=device
        )
        self.generated_counts: torch.Tensor = torch.zeros(
            (len(reqs),), dtype=torch.int32, device=device
        )

        # Track hard blocks applied at last step per request (list of 1-D tensors)
        self._last_hard_blocks: List[Optional[torch.Tensor]] = [None] * len(reqs)

        self.first_token_ids: List[Optional[torch.Tensor]] = [None] * len(reqs)
        self.full_prefixes: List[Optional[List[List[int]]]] = [None] * len(reqs)
        # Optional neighbor-prefix ids
        self.prefix_first_token_ids: List[Optional[torch.Tensor]] = [None] * len(reqs)

        # Track whether the NEXT position is a sentence/reply start for each request.
        # Initialized to True to treat BOS as a start position.
        self.next_pos_is_start: torch.Tensor = torch.ones(
            (len(reqs),), dtype=torch.bool, device=device
        )

        for i, req in enumerate(reqs):
            cp = getattr(req.sampling_params, "custom_params", None) or {}
            text = str(cp.get("unigrams_text", "") or "")

            window = int(cp.get("ban_user_unigram_guard_window", 40) or 40)
            hard_bos = bool(cp.get("ban_user_unigram_hard_at_bos", True))
            hard_all = bool(cp.get("ban_user_unigram_hard_at_all_starts", False))
            bias = float(cp.get("ban_user_unigram_bias", -0.9) or -0.9)
            prefix_len = int(cp.get("ban_user_unigram_prefix_len", 0) or 0)
            hard_prefix = bool(cp.get("ban_user_unigram_hard_prefix_at_starts", True))

            self.guard_window[i] = window
            self.hard_at_bos[i] = hard_bos
            self.hard_at_all_starts[i] = hard_all
            self.bias_vals[i] = bias
            # Only enable hard_prefix if prefix_len > 0
            self.hard_prefix_at_starts[i] = bool(hard_prefix and prefix_len > 0)

            # Proceed only if tokenizer is available

            tokenizer = getattr(req, "tokenizer", None)
            if tokenizer is None:
                # If tokenizer is unavailable, skip for this request
                continue

            # Match words including contractions (e.g., "don't", "we're", "would've")
            matches = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text) if text else []

            # Use cached word->first-token-id index only to avoid
            # over-including generic subword/prefix tokens from ad-hoc surface scans.
            first_ids: Set[int] = set()
            try:
                vocab_size = int(getattr(self.orchestrator, "vocab_size", 0) or 0)
                if matches and tokenizer is not None and vocab_size > 0:
                    index = get_unigram_first_word_index(
                        tokenizer, vocab_size, self._OPENING_QUOTES
                    )
                    banned_words: Set[str] = set()
                    for orig in matches:
                        low = orig.lower()
                        if low and (low not in STOPWORDS):
                            banned_words.add(low)
                    for w in banned_words:
                        ids = index.word_to_token_ids.get(w)
                        if ids:
                            for tid in ids:
                                first_ids.add(int(tid))
            except Exception:
                pass

            if first_ids:
                self.first_token_ids[i] = torch.tensor(
                    sorted(first_ids), dtype=torch.int64, device=device
                )
            else:
                self.first_token_ids[i] = None
            # Keep API parity; we no longer build per-word prefixes here
            self.full_prefixes[i] = None

            # Build optional neighbor-prefix first-token IDs at starts
            if prefix_len > 0 and tokenizer is not None and matches:
                try:
                    vocab_size = int(getattr(self.orchestrator, "vocab_size", 0) or 0)
                    if vocab_size > 0:
                        pref_index = get_unigram_prefix_index(
                            tokenizer, vocab_size, self._OPENING_QUOTES, prefix_len
                        )
                        banned_words: Set[str] = set()
                        for orig in matches:
                            low = orig.lower()
                            if low and (low not in STOPWORDS):
                                banned_words.add(low)
                        prefixes_set: Set[str] = set(
                            w[:prefix_len] for w in banned_words if len(w) >= prefix_len
                        )
                        pref_ids: Set[int] = set()
                        for pref in prefixes_set:
                            ids = pref_index.get(pref)
                            if ids:
                                for tid in ids:
                                    pref_ids.add(int(tid))
                        if pref_ids:
                            self.prefix_first_token_ids[i] = torch.tensor(
                                sorted(pref_ids), dtype=torch.int64, device=device
                            )
                        else:
                            self.prefix_first_token_ids[i] = None
                except Exception:
                    self.prefix_first_token_ids[i] = None

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        if output_ids is None or output_ids.numel() == 0:
            return
        self.generated_counts.add_(torch.ones_like(self.generated_counts))
        # Update per-request next_pos_is_start using only the last emitted token
        reqs = self.orchestrator.reqs()
        if reqs is None:
            return
        B = len(reqs)
        # Guard against size mismatches (e.g., after filter/merge in scheduler)
        if self.next_pos_is_start.size(0) != B:
            return
        step_B = min(B, int(output_ids.numel()))
        for i in range(step_B):
            req = reqs[i]
            tok = getattr(req, "tokenizer", None)
            last_id = int(output_ids[i].item())
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

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        B, V = logits.shape

        # Copy tensor/list references at start to prevent race conditions
        guard_window = self.guard_window
        hard_at_bos = self.hard_at_bos
        hard_at_all_starts = self.hard_at_all_starts
        hard_prefix_at_starts = self.hard_prefix_at_starts
        bias_vals = self.bias_vals
        generated_counts = self.generated_counts
        first_token_ids = self.first_token_ids
        prefix_first_token_ids = self.prefix_first_token_ids
        last_hard_blocks = self._last_hard_blocks

        next_pos_is_start = self.next_pos_is_start
        reqs = self.orchestrator.reqs()
        # If reqs unavailable or batch size mismatch, skip
        if (
            reqs is None
            or len(reqs) != B
            or len(guard_window) != B
            or len(hard_at_bos) != B
            or len(hard_at_all_starts) != B
            or len(bias_vals) != B
            or len(generated_counts) != B
            or len(first_token_ids) != B
            or len(prefix_first_token_ids) != B
            or len(last_hard_blocks) != B
            or next_pos_is_start.size(0) != B
        ):
            return logits
        # Reset last hard-blocks
        for j in range(B):
            last_hard_blocks[j] = None
        # Batch-collect rows needing hard blocks, and apply in one fused op
        to_block: List[Optional[torch.Tensor]] = [None] * B
        for i in range(B):
            req = reqs[i]
            first_ids = first_token_ids[i]
            if first_ids is None or first_ids.numel() == 0:
                continue
            if int(generated_counts[i].item()) >= int(guard_window[i].item()):
                continue

            _out_ids = getattr(req, "output_ids", None) or []
            is_bos = len(_out_ids) == 0
            is_start = True if is_bos else bool(next_pos_is_start[i].item())
            if not is_start:
                continue

            if is_bos and bool(hard_at_bos[i].item()):
                pref_ids = prefix_first_token_ids[i]
                if pref_ids is not None and bool(hard_prefix_at_starts[i].item()):
                    to_block[i] = torch.unique(torch.cat([first_ids, pref_ids]))
                else:
                    to_block[i] = first_ids
                last_hard_blocks[i] = to_block[i]
            elif (not is_bos) and bool(hard_at_all_starts[i].item()):
                pref_ids = prefix_first_token_ids[i]
                if pref_ids is not None and bool(hard_prefix_at_starts[i].item()):
                    to_block[i] = torch.unique(torch.cat([first_ids, pref_ids]))
                else:
                    to_block[i] = first_ids
                last_hard_blocks[i] = to_block[i]
            else:
                # Do not apply soft bias at non-start positions; applying it globally
                # caused broad suppression mid-sentence and unnatural detours.
                # If softer behavior is desired, disable hard_* and rely on soft bias
                # at start positions only.
                pass

        # Apply hard blocks in a batched op
        if any(x is not None for x in to_block):
            from sglang.srt.sampling.penaltylib.mask_utils import (
                apply_blocked_ids_mask_inplace,
            )

            apply_blocked_ids_mask_inplace(logits, to_block, fill_value=-float("inf"))
            # Optional DEBUG: log top-k next tokens when hard block is applied

            for i in range(B):
                if to_block[i] is not None:
                    try:
                        self._debug_log_topk_row(logits, i, reqs[i], tag="unigram-hard")
                    except Exception:
                        pass
        return logits

    def _debug_log_topk_row(self, logits: torch.Tensor, i: int, req, tag: str):
        """Log top-k tokens and their probabilities for row i (DEBUG only)."""
        import torch

        topk = min(10, logits.size(1))
        vals, idx = torch.topk(logits[i], topk)
        probs = torch.softmax(logits[i], dim=-1)[idx]
        tok = getattr(req, "tokenizer", None)
        items = []
        for tid, p in zip(idx.tolist(), probs.tolist()):
            if tok is not None:
                try:
                    s = tok.decode([int(tid)])
                except Exception:
                    s = ""
            else:
                s = ""
            items.append((tid, round(float(p), 5), s))
        logger.info(f"UnigramGuard {tag} row={i} topk={items}")

    def _filter(self, keep_indices: torch.Tensor):
        keep = keep_indices
        self.guard_window = self.guard_window[keep]
        self.hard_at_bos = self.hard_at_bos[keep]
        self.hard_at_all_starts = self.hard_at_all_starts[keep]
        self.hard_prefix_at_starts = self.hard_prefix_at_starts[keep]
        self.bias_vals = self.bias_vals[keep]
        self.generated_counts = self.generated_counts[keep]
        self.first_token_ids = [self.first_token_ids[j] for j in keep.tolist()]
        self.prefix_first_token_ids = [
            self.prefix_first_token_ids[j] for j in keep.tolist()
        ]
        self.full_prefixes = [self.full_prefixes[j] for j in keep.tolist()]
        self._last_hard_blocks = [self._last_hard_blocks[j] for j in keep.tolist()]
        # Keep next_pos_is_start in sync with batch filtering
        if (
            self.next_pos_is_start is not None
            and self.next_pos_is_start.size(0) >= keep.numel()
        ):
            self.next_pos_is_start = self.next_pos_is_start[keep]

    def _merge(self, their: "BatchedUserUnigramStartGuardPenalizer"):
        self.guard_window = torch.cat([self.guard_window, their.guard_window], dim=0)
        self.hard_at_bos = torch.cat([self.hard_at_bos, their.hard_at_bos], dim=0)
        self.hard_at_all_starts = torch.cat(
            [self.hard_at_all_starts, their.hard_at_all_starts], dim=0
        )
        self.hard_prefix_at_starts = torch.cat(
            [self.hard_prefix_at_starts, their.hard_prefix_at_starts], dim=0
        )
        self.bias_vals = torch.cat([self.bias_vals, their.bias_vals], dim=0)
        self.generated_counts = torch.cat(
            [self.generated_counts, their.generated_counts], dim=0
        )
        self.first_token_ids.extend(their.first_token_ids)
        self.prefix_first_token_ids.extend(their.prefix_first_token_ids)
        self.full_prefixes.extend(their.full_prefixes)
        self._last_hard_blocks.extend(their._last_hard_blocks)
        # Merge next_pos_is_start state
        self.next_pos_is_start = torch.cat(
            [self.next_pos_is_start, their.next_pos_is_start], dim=0
        )

    def _is_start_position(self, req) -> bool:
        _out_ids = getattr(req, "output_ids", None) or []
        if len(_out_ids) == 0:
            return True
        tokenizer = getattr(req, "tokenizer", None)
        if tokenizer is None:
            return False

        n = min(12, len(_out_ids))
        try:
            tail = tokenizer.decode(_out_ids[-n:])
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
        is_start = ch in self._OPENING_QUOTES or ch in SENTENCE_END_CHARS
        return is_start

    def get_last_hard_block_ids(self):
        return self._last_hard_blocks

    def get_computed_hard_block_ids(self):
        # Compute current hard blocks based on request state (BOS / start detection)
        reqs = self.orchestrator.reqs()
        if not reqs:
            return None
        out: List[Optional[torch.Tensor]] = [None] * len(reqs)
        for i, req in enumerate(reqs):
            first_ids = self.first_token_ids[i]
            if first_ids is None or (
                hasattr(first_ids, "numel") and first_ids.numel() == 0
            ):
                continue
            # Respect window if we can infer tokens generated
            try:
                total_emitted = len(getattr(req, "output_ids", []) or [])
                if total_emitted >= int(self.guard_window[i].item()):
                    continue
            except Exception:
                pass
            out_ids_list = getattr(req, "output_ids", []) or []
            is_bos = len(out_ids_list) == 0
            if is_bos and bool(self.hard_at_bos[i].item()):
                out[i] = first_ids
                continue
            if (not is_bos) and bool(self.hard_at_all_starts[i].item()):
                if self._is_start_position(req):
                    out[i] = first_ids
        return out
