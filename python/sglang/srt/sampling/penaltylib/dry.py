from __future__ import annotations
import math
from typing import List, Optional

import torch
from sglang.srt.sampling.penaltylib.orchestrator import (
    BatchedPenalizerOrchestrator,
    _BatchedPenalizer,
)


def _find_all_subsequence(haystack: List[int], needle: List[int]) -> List[int]:
    if not needle:
        return []
    n = len(needle)
    lim = len(haystack) - n
    if lim < 0:
        return []
    out: List[int] = []
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
            out.append(i)
        i += 1
    return out


class BatchedDRYPenalizer(_BatchedPenalizer):
    """Penalizes logits for repeated suffixes (DRY)."""

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

        multipliers: List[float] = []
        bases: List[float] = []
        allowed_lengths: List[int] = []
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
                        tmp: List[List[int]] = []
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

        self.dry_multiplier = torch.tensor(
            multipliers, dtype=torch.float32, device=device
        ).unsqueeze(1)
        self.dry_base = torch.tensor(
            bases, dtype=torch.float32, device=device
        ).unsqueeze(1)
        self.dry_allowed_length = torch.tensor(
            allowed_lengths, dtype=torch.int32, device=device
        ).unsqueeze(1)
        self.breakers = breakers_ids

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        return

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        import logging
        logger = logging.getLogger(__name__)
        B, V = logits.shape
        logger.info(f"DRY _apply: starting with logits shape B={B}, V={V}")
        reqs = self.orchestrator.reqs()
        # If reqs unavailable or batch size mismatch, skip
        if reqs is None:
            logger.info("DRY _apply: reqs is None, skipping")
            return logits
        reqs_len = len(reqs)
        logger.info(f"DRY _apply: got reqs with len={reqs_len}")
        if reqs_len != B:
            logger.info(f"DRY _apply: batch size mismatch, reqs={reqs_len} vs B={B}, skipping")
            return logits
        mult_len = len(self.dry_multiplier)
        logger.info(f"DRY _apply: dry_multiplier len={mult_len}")
        if mult_len != B:
            logger.info(f"DRY _apply: tensor size mismatch, dry_multiplier={mult_len} vs B={B}, skipping")
            return logits
        logger.info(f"DRY _apply: all checks passed, processing {B} requests")
        for i in range(B):
            logger.info(f"DRY _apply: processing request i={i}/{B}")
            mult = float(self.dry_multiplier[i].item())
            if mult <= 0.0:
                continue
            base = float(self.dry_base[i].item())
            allow = int(self.dry_allowed_length[i].item())
            brks = self.breakers[i]
            req = reqs[i]

            # 1) Build token history (origin + output so far)
            hist = (req.origin_input_ids or []) + (req.output_ids or [])
            if not hist or len(hist) < 2:
                continue

            # 2) If any breaker sequence matches the tail of history, skip this step
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

            # 3) Search longest repeating suffix length (> allow), capped to a window
            max_search = min(len(hist) - 1, 128)
            if max_search <= allow:
                continue

            tail = hist[-max_search:]
            prefix = hist[:-1]  # history excluding the final token

            best_k = 0
            candidates: List[int] = []

            for k in range(max_search, allow, -1):
                suffix = tail[-k:]
                indices = _find_all_subsequence(prefix, suffix)
                if indices:
                    # 4) Gather tokens that historically followed the matched suffix
                    tmp: List[int] = []
                    for idx in indices:
                        j = idx + k
                        if j < len(hist):
                            tmp.append(hist[j])
                    if tmp:
                        best_k = k
                        candidates = tmp
                        break

            if best_k <= 0 or not candidates:
                continue

            # 5) Apply a positive penalty => subtract log(1 + penalty) from those candidate logits
            pen = mult * (base**best_k)
            try:
                if pen > 0.0:
                    delta = math.log1p(pen)
                    for token_id in set(candidates):
                        if 0 <= token_id < V:
                            logits[i, token_id] -= float(delta)
            except Exception:
                pass

    def _filter(self, keep_indices: torch.Tensor):
        import logging
        logger = logging.getLogger(__name__)
        old_len = len(self.dry_multiplier)
        new_len = len(keep_indices)
        logger.info(f"DRY _filter: filtering from {old_len} to {new_len} requests")
        self.dry_multiplier = self.dry_multiplier[keep_indices]
        self.dry_base = self.dry_base[keep_indices]
        self.dry_allowed_length = self.dry_allowed_length[keep_indices]
        self.breakers = [self.breakers[j] for j in keep_indices.tolist()]
        logger.info(f"DRY _filter: complete, new dry_multiplier len={len(self.dry_multiplier)}")

    def _merge(self, their: "BatchedDRYPenalizer"):
        import logging
        logger = logging.getLogger(__name__)
        old_len = len(self.dry_multiplier)
        their_len = len(their.dry_multiplier)
        logger.info(f"DRY _merge: merging {old_len} + {their_len} requests")
        self.dry_multiplier = torch.cat(
            [self.dry_multiplier, their.dry_multiplier], dim=0
        )
        self.dry_base = torch.cat([self.dry_base, their.dry_base], dim=0)
        self.dry_allowed_length = torch.cat(
            [self.dry_allowed_length, their.dry_allowed_length], dim=0
        )
        self.breakers.extend(their.breakers)
        logger.info(f"DRY _merge: complete, new dry_multiplier len={len(self.dry_multiplier)}")
