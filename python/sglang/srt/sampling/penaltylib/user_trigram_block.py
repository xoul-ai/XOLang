from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import torch
from sglang.srt.sampling.penaltylib.orchestrator import (
    BatchedPenalizerOrchestrator,
    _BatchedPenalizer,
)


class BatchedUserTrigramBlockPenalizer(_BatchedPenalizer):
    """
    Blocks exact contiguous n-gram spans (default n=3) taken from the
    last user-provided text for the first N generated tokens (default: 80).

    Per-request knobs in SamplingParams.custom_params:
      - ban_user_pair_to_next_map: Optional[Dict[Tuple[int,...], Set[int]]]
        Precomputed mapping of prefix (n-1 tokens) -> set(next token ids).
      - ban_user_trigrams_ids: Optional[List[List[int]]] (legacy; n must be 3) 
      - ban_user_trigram_max_tokens: Optional[int] (default: 80)
      - ban_user_ngram_size: Optional[int] (default: 3, minimum 3)
      - ban_user_whitelist_token_ids: Optional[List[int]] (never block these ids)

    Implementation detail: when the last (n-1) generated tokens match a
    prefix in the mapping, we set logits of all mapped next tokens to -inf.
    """

    def __init__(self, orchestrator: BatchedPenalizerOrchestrator):
        self.orchestrator = orchestrator
        self._is_prepared = False

    def _is_required(self) -> bool:
        reqs = self.orchestrator.reqs()
        if not reqs:
            return False
        for req in reqs:
            cp = getattr(req.sampling_params, "custom_params", None)
            if not isinstance(cp, dict):
                continue
            if cp.get("ban_user_pair_to_next_map"):
                return True
            if cp.get("ban_user_trigrams_ids"):
                return True
        return False

    def _prepare(self):
        # Precompute prefix -> {next} mapping and per-request limits
        reqs = self.orchestrator.reqs()
        pair_to_next: List[Optional[Dict[Tuple[int, ...], Set[int]]]] = []
        max_tokens_list: List[int] = []
        n_list: List[int] = []
        whitelist_list: List[Optional[Set[int]]] = []

        for req in reqs:
            cp = getattr(req.sampling_params, "custom_params", None)
            if not isinstance(cp, dict):
                pair_to_next.append(None)
                max_tokens_list.append(0)
                n_list.append(3)
                whitelist_list.append(None)
                continue
            # ngram size (min 3)
            n = cp.get("ban_user_ngram_size", 3)
            try:
                n = int(n)
            except Exception:
                n = 3
            if n < 3:
                n = 3

            # explicit whitelist
            wl_ids = cp.get("ban_user_whitelist_token_ids")
            whitelist = None
            if isinstance(wl_ids, list) and wl_ids:
                try:
                    whitelist = {int(x) for x in wl_ids}
                except Exception:
                    whitelist = None

            # precomputed mapping takes precedence
            mapping = cp.get("ban_user_pair_to_next_map")
            if isinstance(mapping, dict) and mapping:
                # keys can be tuples or strings -> try to normalize to tuple[int,...]
                norm: Dict[Tuple[int, ...], Set[int]] = {}
                for k, v in mapping.items():
                    if isinstance(k, (list, tuple)):
                        key = tuple(int(x) for x in k)
                    else:
                        # Can't reliably parse; skip malformed keys
                        continue
                    if not isinstance(v, (list, set)):
                        continue
                    s = set(int(x) for x in v)
                    if s:
                        norm[key] = s
                pair_to_next.append(norm if norm else None)
                max_tokens = cp.get("ban_user_trigram_max_tokens")
                try:
                    max_tokens = int(max_tokens) if max_tokens is not None else 80
                except Exception:
                    max_tokens = 80
                if max_tokens < 0:
                    max_tokens = 0
                max_tokens_list.append(max_tokens)
                n_list.append(n)
                whitelist_list.append(whitelist)
                continue

            # fallback to trigrams list
            trigrams = cp.get("ban_user_trigrams_ids")
            if not isinstance(trigrams, list) or len(trigrams) == 0:
                pair_to_next.append(None)
                max_tokens_list.append(0)
                n_list.append(n)
                whitelist_list.append(whitelist)
                continue
            mapping2: Dict[Tuple[int, ...], Set[int]] = {}
            for tri in trigrams:
                if not isinstance(tri, list) or len(tri) != 3:
                    continue
                a, b, c = tri
                key = (int(a), int(b))
                mapping2.setdefault(key, set()).add(int(c))
            pair_to_next.append(mapping2 if mapping2 else None)

            max_tokens = cp.get("ban_user_trigram_max_tokens")
            try:
                max_tokens = int(max_tokens) if max_tokens is not None else 80
            except Exception:
                max_tokens = 80
            if max_tokens < 0:
                max_tokens = 0
            max_tokens_list.append(max_tokens)

        self._pair_to_next = pair_to_next
        # store as simple python list; we only read it each step
        self._max_tokens_list = max_tokens_list
        self._ngram_size_list = n_list
        self._whitelist_list = whitelist_list

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        # No internal state to update; we read req.output_ids directly.
        return

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: [B, V]
        B, V = logits.shape
        reqs = self.orchestrator.reqs()
        for i in range(B):
            mapping = self._pair_to_next[i] if i < len(self._pair_to_next) else None
            if not mapping:
                continue
            req = reqs[i]
            # Apply only to the first N generated tokens
            if not isinstance(req.output_ids, list):
                continue
            gen_len = len(req.output_ids)
            n = self._ngram_size_list[i] if i < len(self._ngram_size_list) else 3
            if gen_len < max(2, n - 1) or gen_len >= self._max_tokens_list[i]:
                continue
            key = tuple(req.output_ids[-(n - 1) :])
            nxts = mapping.get(key)
            if not nxts:
                continue
            whitelist = self._whitelist_list[i] if i < len(self._whitelist_list) else None
            # Set logits for all next tokens to -inf
            for c in nxts:
                if whitelist is not None and c in whitelist:
                    continue
                if 0 <= c < V:
                    logits[i, c] = -float("inf")

    def _filter(self, keep_indices: torch.Tensor):
        keep = keep_indices.tolist()
        self._pair_to_next = [self._pair_to_next[j] for j in keep]
        self._max_tokens_list = [self._max_tokens_list[j] for j in keep]
        self._ngram_size_list = [self._ngram_size_list[j] for j in keep]
        self._whitelist_list = [self._whitelist_list[j] for j in keep]

    def _merge(self, their: "BatchedUserTrigramBlockPenalizer"):
        self._pair_to_next.extend(their._pair_to_next)
        self._max_tokens_list.extend(their._max_tokens_list)
        self._ngram_size_list.extend(their._ngram_size_list)
        self._whitelist_list.extend(their._whitelist_list)
