from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import torch

from sglang.srt.sampling.penaltylib.orchestrator import (
    BatchedPenalizerOrchestrator,
    _BatchedPenalizer,
)


class BatchedUserNGramBlockPenalizer(_BatchedPenalizer):
    """
    Blocks exact contiguous user-provided n-grams (default n=3) during the
    first N newly generated tokens (default 80), request-scoped.

    Expected knobs in SamplingParams.custom_params (per request):
      - ban_user_pair_to_next_map: Dict[str, List[int]] mapping of prefix
        (n-1 tokens) to allowed next tokens to block. Prefix key format:
        "a|b|..." (string of int token-ids separated by '|'). Values are
        int token-ids. This is injected by entrypoints.
      - ban_user_trigram_max_tokens: int, default 80 (only count new tokens)
      - ban_user_ngram_size: int, >= 3, default 3
      - ban_user_whitelist_token_ids: Optional[List[int]]

    Behavior: When the last (n-1) generated tokens exactly match a prefix,
    set logits[next] = -inf for each next in the mapped set.
    """

    def __init__(self, orchestrator: BatchedPenalizerOrchestrator):
        self.orchestrator = orchestrator
        self._is_prepared = False

    def _is_required(self) -> bool:
        batch = self.orchestrator.batch
        if batch is None:
            return False
        for req in batch.reqs:
            cp = getattr(req.sampling_params, "custom_params", None)
            if not isinstance(cp, dict):
                continue
            if cp.get("ban_user_pair_to_next_map"):
                return True
        return False

    def _prepare(self):
        reqs = self.orchestrator.reqs()
        # Parsed data per request
        maps: List[Optional[Dict[Tuple[int, ...], Set[int]]]] = []
        n_list: List[int] = []
        max_list: List[int] = []
        wl_list: List[Optional[Set[int]]] = []

        for req in reqs:
            cp = getattr(req.sampling_params, "custom_params", None)
            if not isinstance(cp, dict):
                maps.append(None)
                n_list.append(3)
                max_list.append(0)
                wl_list.append(None)
                continue

            # Parse n
            n = cp.get("ban_user_ngram_size", 3)
            try:
                n = int(n)
            except Exception:
                n = 3
            if n < 3:
                n = 3

            # Parse whitelist
            wl_ids = cp.get("ban_user_whitelist_token_ids")
            whitelist = None
            if isinstance(wl_ids, list) and wl_ids:
                try:
                    whitelist = {int(x) for x in wl_ids}
                except Exception:
                    whitelist = None

            # Parse max tokens
            max_tokens = cp.get("ban_user_trigram_max_tokens")
            try:
                max_tokens = int(max_tokens) if max_tokens is not None else 80
            except Exception:
                max_tokens = 80
            if max_tokens < 0:
                max_tokens = 0

            # Parse provided map
            raw_map = cp.get("ban_user_pair_to_next_map")
            parsed: Optional[Dict[Tuple[int, ...], Set[int]]] = None
            if isinstance(raw_map, dict) and raw_map:
                parsed = {}
                for k, v in raw_map.items():
                    # Accept key formats: tuple/list of ints, or pipe-joined string
                    if isinstance(k, (tuple, list)):
                        try:
                            key = tuple(int(x) for x in k)
                        except Exception:
                            continue
                    elif isinstance(k, str):
                        try:
                            key = tuple(int(x) for x in k.split("|") if x != "")
                        except Exception:
                            continue
                    else:
                        continue
                    if len(key) != (n - 1):
                        # Skip keys that don't match the configured n
                        continue
                    if not isinstance(v, (list, set, tuple)):
                        continue
                    nexts = set()
                    for t in v:
                        try:
                            nexts.add(int(t))
                        except Exception:
                            continue
                    if nexts:
                        parsed[key] = nexts

            maps.append(parsed if parsed else None)
            n_list.append(n)
            max_list.append(max_tokens)
            wl_list.append(whitelist)

        self._maps = maps
        self._ns = n_list
        self._max_new = max_list
        self._whitelists = wl_list

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        # Not used; we read req.output_ids directly.
        return

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        if logits is None:
            return logits
        B, V = logits.shape
        reqs = self.orchestrator.reqs()
        for i in range(B):
            mapping = self._maps[i] if i < len(self._maps) else None
            if not mapping:
                continue
            req = reqs[i]
            if not isinstance(req.output_ids, list):
                continue
            gen_len = len(req.output_ids)
            n = self._ns[i]
            if gen_len < (n - 1) or gen_len >= self._max_new[i]:
                continue
            key = tuple(req.output_ids[-(n - 1) :])
            nexts = mapping.get(key)
            if not nexts:
                continue
            whitelist = self._whitelists[i]
            for t in nexts:
                if whitelist is not None and t in whitelist:
                    continue
                if 0 <= t < V:
                    logits[i, t] = -float("inf")
        return logits

    def _filter(self, keep_indices: torch.Tensor):
        idx = keep_indices.tolist()
        self._maps = [self._maps[j] for j in idx]
        self._ns = [self._ns[j] for j in idx]
        self._max_new = [self._max_new[j] for j in idx]
        self._whitelists = [self._whitelists[j] for j in idx]

    def _merge(self, their: "BatchedUserNGramBlockPenalizer"):
        self._maps.extend(their._maps)
        self._ns.extend(their._ns)
        self._max_new.extend(their._max_new)
        self._whitelists.extend(their._whitelists)

