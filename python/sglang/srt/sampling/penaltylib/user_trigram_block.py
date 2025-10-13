from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import logging

from sglang.srt.sampling.penaltylib.orchestrator import (
    BatchedPenalizerOrchestrator,
    _BatchedPenalizer,
)


class BatchedUserTrigramBlocker(_BatchedPenalizer):
    """
    Blocks exact reuse of user 3-grams in the first ~80 generated tokens.

    Activation is per-request via SamplingParams.custom_params:
      - ban_user_trigrams_text: str (required to activate)
      - ban_user_trigrams_max_tokens: Optional[int] (default 80)

    Behavior:
      - Precomputes token trigrams from the provided text using the same tokenizer
        as generation (encode with add_special_tokens=False).
      - During generation, if the last two generated tokens match the first two
        of any user trigram, it sets the logit for the corresponding third token
        to -inf, preventing the model from continuing an exact 3+ span.
      - Only applied for the first `ban_user_trigrams_max_tokens` output tokens.
    """

    def __init__(self, orchestrator: BatchedPenalizerOrchestrator):
        self.orchestrator = orchestrator
        self._is_prepared = False

        # Per-request data
        self._pair2next: List[Optional[Dict[Tuple[int, int], List[int]]]] = []
        self._max_tokens: List[int] = []
        self._logger = logging.getLogger(__name__)

    def _is_required(self) -> bool:
        reqs = self.orchestrator.reqs()
        if not reqs:
            return False
        for req in reqs:
            cp = getattr(req.sampling_params, "custom_params", None)
            if isinstance(cp, dict) and (cp.get("ban_user_trigrams_text") or cp.get("ban_user_trigrams_ids")):
                return True
        return False

    def _prepare(self):
        self._pair2next = []
        self._max_tokens = []

        for req in self.orchestrator.reqs():
            cp = getattr(req.sampling_params, "custom_params", None) or {}
            debug = bool(cp.get("ban_user_trigrams_debug", False))

            # Allow overriding the active length window; default 80
            limit = cp.get("ban_user_trigrams_max_tokens")
            try:
                limit = int(limit) if limit is not None else 80
            except Exception:
                limit = 80
            self._max_tokens.append(max(0, limit))

            pair2next: Optional[Dict[Tuple[int, int], List[int]]] = None

            # Prefer explicit ids if provided; otherwise tokenize text
            ids = cp.get("ban_user_trigrams_ids")
            if ids is None:
                text = cp.get("ban_user_trigrams_text")
                if isinstance(text, str) and text.strip():
                    tokenizer = getattr(req, "tokenizer", None)
                    try:
                        if tokenizer is not None:
                            ids = tokenizer.encode(text, add_special_tokens=False)
                        else:
                            ids = None
                    except Exception:
                        ids = None

            if isinstance(ids, list) and len(ids) >= 3:
                pair2next: Dict[Tuple[int, int], List[int]] = {}
                tok = getattr(req, "tokenizer", None)

                def add_block_for_pair(pair: Tuple[int, int], third_id: int, third_text: Optional[str]):
                    lst = pair2next.get(pair)
                    if lst is None:
                        lst = []
                        pair2next[pair] = lst
                    if third_id not in lst:
                        lst.append(third_id)
                    # Also block single-token prefixes of the third text to prevent multi-token spillovers
                    if tok is not None and isinstance(third_text, str) and len(third_text) > 0:
                        # Preserve leading spaces from third_text
                        leading = len(third_text) - len(third_text.lstrip(" "))
                        base = third_text[:leading]
                        core = third_text[leading:]
                        for plen in range(1, len(core)):
                            prefix_text = base + core[:plen]
                            try:
                                enc = tok.encode(prefix_text, add_special_tokens=False)
                            except Exception:
                                enc = None
                            if isinstance(enc, list) and len(enc) == 1:
                                cand = int(enc[0])
                                if cand != third_id and cand not in lst:
                                    lst.append(cand)

                def add_trigrams(seq: List[int]):
                    for j in range(len(seq) - 2):
                        p = (int(seq[j]), int(seq[j + 1]))
                        nxt = int(seq[j + 2])
                        third_text = None
                        try:
                            if tok is not None:
                                third_text = tok.decode([nxt])
                        except Exception:
                            third_text = None
                        add_block_for_pair(p, nxt, third_text)
                add_trigrams(ids)
                if debug:
                    try:
                        tok = getattr(req, "tokenizer", None)
                        toks = [tok.decode([t]) if tok is not None else str(t) for t in ids]
                        self._logger.info(
                            "[UTB] rid=%s base_text_ids=%s base_text_tokens=%s",
                            getattr(req, "rid", ""), ids, toks
                        )
                        # Log per-pair block list for visibility
                        for pair, blocked in pair2next.items():
                            blocked_str = [tok.decode([b]) if tok is not None else str(b) for b in blocked]
                            self._logger.info(
                                "[UTB] rid=%s pair=%s blocked_ids=%s blocked_tokens=%s",
                                getattr(req, "rid", ""), pair, blocked, blocked_str
                            )
                    except Exception:
                        pass
                # Also consider a leading-space variant to match common generation prefixes
                try:
                    tokenizer = getattr(req, "tokenizer", None)
                    if tokenizer is not None:
                        ids2 = tokenizer.encode(" " + text, add_special_tokens=False)
                            if isinstance(ids2, list) and len(ids2) >= 3:
                                add_trigrams(ids2)
                            if debug:
                                try:
                                    toks2 = [tokenizer.decode([t]) for t in ids2]
                                    self._logger.info(
                                        "[UTB] rid=%s space_text_ids=%s space_text_tokens=%s",
                                        getattr(req, "rid", ""), ids2, toks2
                                    )
                                    for pair, blocked in pair2next.items():
                                        blocked_str = [tokenizer.decode([b]) for b in blocked]
                                        self._logger.info(
                                            "[UTB] rid=%s pair=%s blocked_ids=%s blocked_tokens=%s",
                                            getattr(req, "rid", ""), pair, blocked, blocked_str
                                        )
                                except Exception:
                                    pass
                except Exception:
                    pass

            self._pair2next.append(pair2next)

        # Log summary once per batch
        try:
            enabled = sum(1 for m in self._pair2next if m)
            self._logger.info(
                "User trigram blocker prepared: %d/%d active, limits=%s",
                enabled,
                len(self._pair2next),
                self._max_tokens,
            )
        except Exception:
            pass

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        # Not required; we read per-request histories directly in _apply.
        return

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._pair2next:
            return logits

        B, V = logits.shape
        reqs = self.orchestrator.reqs()

        for i in range(B):
            mapping = self._pair2next[i]
            if not mapping:
                continue
            req = reqs[i]

            # Apply only for the first N generated tokens
            if len(req.output_ids) >= self._max_tokens[i]:
                continue

            # Need at least two generated tokens to match a trigram prefix
            if len(req.output_ids) < 2:
                try:
                    cp = getattr(req.sampling_params, "custom_params", None) or {}
                    if cp.get("ban_user_trigrams_debug", False):
                        self._logger.debug(
                            "[UTB] step=%d rid=%s not enough history (need >=2)",
                            len(req.output_ids), getattr(req, "rid", "")
                        )
                except Exception:
                    pass
                continue

            t1 = int(req.output_ids[-2])
            t2 = int(req.output_ids[-1])
            next_list = mapping.get((t1, t2))
            if not next_list:
                # Optional step-level debug
                try:
                    cp = getattr(req.sampling_params, "custom_params", None) or {}
                    if cp.get("ban_user_trigrams_debug", False):
                        tok = getattr(req, "tokenizer", None)
                        s1 = tok.decode([t1]) if tok is not None else str(t1)
                        s2 = tok.decode([t2]) if tok is not None else str(t2)
                        self._logger.debug(
                            "[UTB] step=%d rid=%s last2=(%d,%d) last2_str=(%s|%s) -> no match",
                            len(req.output_ids), getattr(req, "rid", ""), t1, t2, s1, s2
                        )
                except Exception:
                    pass
                continue

            # Block all candidate next ids that would complete a matching trigram
            for t3 in next_list:
                if 0 <= t3 < V:
                    logits[i, t3] = -float("inf")
            # Optional debug: show first block action for a req
            try:
                cp = getattr(req.sampling_params, "custom_params", None) or {}
                if cp.get("ban_user_trigrams_debug", False):
                    tok = getattr(req, "tokenizer", None)
                    blocked_str = [tok.decode([t]) if tok is not None else str(t) for t in next_list]
                    s1 = tok.decode([t1]) if tok is not None else str(t1)
                    s2 = tok.decode([t2]) if tok is not None else str(t2)
                    self._logger.info(
                        "[UTB] step=%d rid=%s last2=(%d,%d) last2_str=(%s|%s) blocked_next_ids=%s blocked_next_tokens=%s",
                        len(req.output_ids), getattr(req, "rid", ""), t1, t2, s1, s2, next_list, blocked_str
                    )
            except Exception:
                pass

        return logits

    def _filter(self, keep_indices: torch.Tensor):
        keep = keep_indices.tolist()
        self._pair2next = [self._pair2next[j] for j in keep]
        self._max_tokens = [self._max_tokens[j] for j in keep]

    def _merge(self, their: "BatchedUserTrigramBlocker"):
        self._pair2next.extend(their._pair2next)
        self._max_tokens.extend(their._max_tokens)
