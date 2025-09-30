
from __future__ import annotations
import math
from typing import List, Optional
import torch
from sglang.srt.sampling.penaltylib.orchestrator import (
    BatchedPenalizerOrchestrator,
    _BatchedPenalizer,
)

class BatchedDRYPenalizer(_BatchedPenalizer):
    """DRY (Don't Repeat Yourself) penalizer for SGLang 0.5.2.
    Supports both string and token-id sequence breakers via custom_params:
      - dry_multiplier: float > 0 enables DRY; 0 disables
      - dry_base: float >= 1.0 (growth base)
      - dry_allowed_length: int >= 0 (free allowance)
      - dry_sequence_breakers: optional List[str] (tokenized if tokenizer is available)
      - dry_sequence_breakers_ids: optional List[List[int]] of token-id sequences
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
            mult = float(cp.get("dry_multiplier", 0.0) or 0.0)
            if mult > 0.0:
                return True
        return False

    def _prepare(self):
        device = self.orchestrator.device
        reqs = self.orchestrator.reqs()
        batch = self.orchestrator.batch
        tokenizer = getattr(batch, "tokenizer", None)

        multipliers, bases, allowed_lengths = [], [], []
        breakers_ids: List[Optional[List[List[int]]]] = []

        for req in reqs:
            cp = getattr(req.sampling_params, "custom_params", None) or {}
            mult = float(cp.get("dry_multiplier", 0.0) or 0.0)
            base = float(cp.get("dry_base", 1.1) or 1.1)
            allow = int(cp.get("dry_allowed_length", 0) or 0)

            brk_ids = cp.get("dry_sequence_breakers_ids", None)
            if brk_ids is None:
                brk_strs = cp.get("dry_sequence_breakers", None)
                if brk_strs and tokenizer is not None:
                    try:
                        tmp = []
                        for s in brk_strs:
                            ids = tokenizer.encode(s, add_special_tokens=False)
                            if ids:
                                tmp.append(ids)
                        brk_ids = tmp if tmp else None
                    except Exception:
                        brk_ids = None

            multipliers.append(mult)
            bases.append(base)
            allowed_lengths.append(allow)
            breakers_ids.append(brk_ids if isinstance(brk_ids, list) else None)

        self.dry_multiplier = torch.tensor(multipliers, dtype=torch.float32, device=device).unsqueeze(1)
        self.dry_base = torch.tensor(bases, dtype=torch.float32, device=device).unsqueeze(1)
        self.dry_allowed_length = torch.tensor(allowed_lengths, dtype=torch.int32, device=device).unsqueeze(1)
        self.breakers = breakers_ids

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        return

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        B, V = logits.shape
        for i in range(B):
            mult = float(self.dry_multiplier[i].item())
            if mult <= 0.0:
                continue
            base = float(self.dry_base[i].item())
            allow = int(self.dry_allowed_length[i].item())
            brks = self.breakers[i]
            req = self.orchestrator.reqs()[i]
            hist = (req.origin_input_ids or []) + (req.output_ids or [])
            if not hist:
                continue
            if brks:
                for seq in brks:
                    if not seq:
                        continue
                    L = len(seq)
                    if L <= len(hist) and hist[-L:] == seq:
                        hist = None
                        break
                if hist is None:
                    continue
            max_search = min(len(hist) - 1, 128)
            if max_search <= allow:
                continue
            tail = hist[-max_search:]
            prefix = hist[:-1]
            candidate = None
            best_k = 0
            for k in range(max_search, allow, -1):
                suffix = tail[-k:]
                indices = _find_all_subsequence(prefix, suffix)
                if indices:
                    candidates = []
                    for idx in indices:
                        j = idx + k
                        if j < len(hist):
                            candidates.append(hist[j])
                    if candidates:
                        best_k = k
                        for candidate_token in set(candidates):
                            pen = mult * (base ** best_k)
                            try:
                                if pen > 0.0 and 0 <= candidate_token < logits.shape[1]:
                                    logits[i, candidate_token] -= float(math.log1p(pen))
                            except Exception:
                                pass
                        break
            if candidate is None or best_k <= 0:
                continue
            try:
                factor = mult * (base ** best_k)
                if factor > 1.0:
                    delta = math.log(factor)
                    if 0 <= candidate < V:
                        logits[i, candidate] -= float(delta)
            except Exception:
                pass

    def _filter(self, keep_indices: torch.Tensor):
        self.dry_multiplier = self.dry_multiplier[keep_indices]
        self.dry_base = self.dry_base[keep_indices]
        self.dry_allowed_length = self.dry_allowed_length[keep_indices]
        self.breakers = [self.breakers[j] for j in keep_indices.tolist()]

    def _merge(self, their: "BatchedDRYPenalizer"):
        self.dry_multiplier = torch.cat([self.dry_multiplier, their.dry_multiplier], dim=0)
        self.dry_base = torch.cat([self.dry_base, their.dry_base], dim=0)
        self.dry_allowed_length = torch.cat([self.dry_allowed_length, their.dry_allowed_length], dim=0)
        self.breakers.extend(their.breakers)


def _find_subsequence(haystack: List[int], needle: List[int]) -> int:
    if not needle:
        return -1
    n = len(needle)
    lim = len(haystack) - n
    if lim < 0:
        return -1
    first = needle[0]
    i = 0
    while i <= lim:
        while i <= lim and haystack[i] != first:
            i += 1
        if i > lim:
            break
        ok = True
        for j in range(1, n):
            if haystack[i + j] != needle[j]:
                ok = False
                break
        if ok:
            return i
        i += 1
    return -1
