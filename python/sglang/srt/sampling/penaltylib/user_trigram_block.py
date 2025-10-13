from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import logging
import torch

from sglang.srt.sampling.penaltylib.orchestrator import (
    BatchedPenalizerOrchestrator,
    _BatchedPenalizer,
)


class BatchedUserTrigramBlocker(_BatchedPenalizer):
    """
    Blocks exact reuse of user 3-grams in the first ~80 generated tokens.

    Activation per request via SamplingParams.custom_params:
      - ban_user_trigrams_text: str (required to activate)
      - ban_user_trigrams_max_tokens: Optional[int] (default 80)

    Implementation notes:
      - Precomputes token trigrams from provided text (same tokenizer, add_special_tokens=False).
      - During generation, when last two generated tokens match the first two of any user trigram,
        sets logits for the third token to -inf. Also blocks single-token prefixes of the third token
        to prevent reconstruction via multiple tokens.
      - Applies only for the first `ban_user_trigrams_max_tokens` generated tokens.
    """

    def __init__(self, orchestrator: BatchedPenalizerOrchestrator):
        self.orchestrator = orchestrator
        self._is_prepared = False
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

            limit = cp.get("ban_user_trigrams_max_tokens")
            try:
                limit = int(limit) if limit is not None else 80
            except Exception:
                limit = 80
            self._max_tokens.append(max(0, limit))

            pair2next: Optional[Dict[Tuple[int, int], List[int]]] = None

            ids = cp.get("ban_user_trigrams_ids")
            text = cp.get("ban_user_trigrams_text")
            tok = getattr(req, "tokenizer", None)
            if ids is None and isinstance(text, str) and text.strip() and tok is not None:
                try:
                    ids = tok.encode(text, add_special_tokens=False)
                except Exception:
                    ids = None

            if isinstance(ids, list) and len(ids) >= 3:
                pair2next = {}

                def add_block_for_pair(pair: Tuple[int, int], third_id: int, third_text: Optional[str]):
                    lst = pair2next.get(pair)
                    if lst is None:
                        lst = []
                        pair2next[pair] = lst
                    if third_id not in lst:
                        lst.append(third_id)
                    if tok is not None and isinstance(third_text, str) and len(third_text) > 0:
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
                        try:
                            third_text = tok.decode([nxt]) if tok is not None else None
                        except Exception:
                            third_text = None
                        add_block_for_pair(p, nxt, third_text)

                add_trigrams(ids)

                # Leading-space variant to catch typical BPE/SP space prefix starts
                if tok is not None and isinstance(text, str) and text:
                    try:
                        ids2 = tok.encode(" " + text, add_special_tokens=False)
                    except Exception:
                        ids2 = None
                    if isinstance(ids2, list) and len(ids2) >= 3:
                        add_trigrams(ids2)

                if debug:
                    try:
                        toks = [tok.decode([t]) if tok is not None else str(t) for t in ids]
                        self._logger.info("[UTB] rid=%s base_text_ids=%s base_text_tokens=%s", getattr(req, "rid", ""), ids, toks)
                        if tok is not None and isinstance(text, str) and text:
                            ids2 = tok.encode(" " + text, add_special_tokens=False)
                            toks2 = [tok.decode([t]) for t in ids2] if isinstance(ids2, list) else []
                            self._logger.info("[UTB] rid=%s space_text_ids=%s space_text_tokens=%s", getattr(req, "rid", ""), ids2, toks2)
                        for pair, blocked in pair2next.items():
                            blocked_str = [tok.decode([b]) if tok is not None else str(b) for b in blocked]
                            self._logger.info("[UTB] rid=%s pair=%s blocked_ids=%s blocked_tokens=%s", getattr(req, "rid", ""), pair, blocked, blocked_str)
                    except Exception:
                        pass

            self._pair2next.append(pair2next)

        try:
            enabled = sum(1 for m in self._pair2next if m)
            self._logger.info("User trigram blocker prepared: %d/%d active, limits=%s", enabled, len(self._pair2next), self._max_tokens)
        except Exception:
            pass

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
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
            if len(req.output_ids) >= self._max_tokens[i]:
                continue
            if len(req.output_ids) < 2:
                cp = getattr(req.sampling_params, "custom_params", None) or {}
                if cp.get("ban_user_trigrams_debug", False):
                    try:
                        self._logger.debug("[UTB] step=%d rid=%s not enough history (need >=2)", len(req.output_ids), getattr(req, "rid", ""))
                    except Exception:
                        pass
                continue

            t1 = int(req.output_ids[-2])
            t2 = int(req.output_ids[-1])
            next_list = mapping.get((t1, t2))
            if not next_list:
                cp = getattr(req.sampling_params, "custom_params", None) or {}
                if cp.get("ban_user_trigrams_debug", False):
                    try:
                        tok = getattr(req, "tokenizer", None)
                        s1 = tok.decode([t1]) if tok is not None else str(t1)
                        s2 = tok.decode([t2]) if tok is not None else str(t2)
                        self._logger.debug("[UTB] step=%d rid=%s last2=(%d,%d) last2_str=(%s|%s) -> no match", len(req.output_ids), getattr(req, "rid", ""), t1, t2, s1, s2)
                    except Exception:
                        pass
                continue

            for t3 in next_list:
                if 0 <= t3 < V:
                    logits[i, t3] = -float("inf")

            cp = getattr(req.sampling_params, "custom_params", None) or {}
            if cp.get("ban_user_trigrams_debug", False):
                try:
                    tok = getattr(req, "tokenizer", None)
                    blocked_str = [tok.decode([t]) if tok is not None else str(t) for t in next_list]
                    s1 = tok.decode([t1]) if tok is not None else str(t1)
                    s2 = tok.decode([t2]) if tok is not None else str(t2)
                    self._logger.info("[UTB] step=%d rid=%s last2=(%d,%d) last2_str=(%s|%s) blocked_next_ids=%s blocked_next_tokens=%s", len(req.output_ids), getattr(req, "rid", ""), t1, t2, s1, s2, next_list, blocked_str)
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
