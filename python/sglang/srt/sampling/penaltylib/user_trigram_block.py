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
                pair2next = {}
                def add_trigrams(seq: List[int]):
                    for j in range(len(seq) - 2):
                        p = (int(seq[j]), int(seq[j + 1]))
                        nxt = int(seq[j + 2])
                        if p not in pair2next:
                            pair2next[p] = [nxt]
                        else:
                            pair2next[p].append(nxt)
                add_trigrams(ids)
                # Also consider a leading-space variant to match common generation prefixes
                try:
                    tokenizer = getattr(req, "tokenizer", None)
                    if tokenizer is not None:
                        ids2 = tokenizer.encode(" " + text, add_special_tokens=False)
                        if isinstance(ids2, list) and len(ids2) >= 3:
                            add_trigrams(ids2)
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
                continue

            t1 = int(req.output_ids[-2])
            t2 = int(req.output_ids[-1])
            next_list = mapping.get((t1, t2))
            if not next_list:
                continue

            # Block all candidate next ids that would complete a matching trigram
            for t3 in next_list:
                if 0 <= t3 < V:
                    logits[i, t3] = -float("inf")
            # Optional debug: show first block action for a req
            try:
                if hasattr(req, "_utb_logged") is False or getattr(req, "_utb_logged", False) is False:
                    setattr(req, "_utb_logged", True)
                    self._logger.debug(
                        "Blocked user trigram at step=%d for rid=%s, last2=(%d,%d), blocked_next=%s",
                        len(req.output_ids),
                        getattr(req, "rid", ""),
                        t1,
                        t2,
                        next_list,
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
