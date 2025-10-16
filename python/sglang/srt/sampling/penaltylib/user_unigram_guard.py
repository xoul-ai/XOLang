from __future__ import annotations

import re
from typing import List, Optional, Set

import torch

from sglang.srt.sampling.penaltylib.orchestrator import (
    BatchedPenalizerOrchestrator,
    _BatchedPenalizer,
)
from sglang.srt.sampling.penaltylib.constants import (
    STOPWORDS,
    SENTENCE_END_CHARS,
    QUOTE_CHARS,
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
        self.bias_vals: torch.Tensor = torch.zeros(
            (len(reqs),), dtype=torch.float32, device=device
        )
        self.generated_counts: torch.Tensor = torch.zeros(
            (len(reqs),), dtype=torch.int32, device=device
        )

        self.first_token_ids: List[Optional[torch.Tensor]] = [None] * len(reqs)
        self.full_prefixes: List[Optional[List[List[int]]]] = [None] * len(reqs)

        for i, req in enumerate(reqs):
            cp = getattr(req.sampling_params, "custom_params", None) or {}
            text = str(cp.get("unigrams_text", "") or "")

            window = int(cp.get("ban_user_unigram_guard_window", 40) or 40)
            hard_bos = bool(cp.get("ban_user_unigram_hard_at_bos", True))
            hard_all = bool(cp.get("ban_user_unigram_hard_at_all_starts", False))
            bias = float(cp.get("ban_user_unigram_bias", -0.9) or -0.9)

            self.guard_window[i] = window
            self.hard_at_bos[i] = hard_bos
            self.hard_at_all_starts[i] = hard_all
            self.bias_vals[i] = bias

            # Proceed to surface encoding if tokenizer is available

            tokenizer = getattr(req, "tokenizer", None)
            if tokenizer is None:
                # If tokenizer is unavailable, skip for this request
                continue

            matches = re.findall(r"[A-Za-z]+", text) if text else []

            cap = 1500
            seen: Set[str] = set()
            words_with_variants: List[str] = []
            for orig in matches:
                low = orig.lower()
                if low in STOPWORDS:
                    continue
                # add lower/title/original variants
                for v in (low, orig.title(), orig):
                    if v not in seen:
                        seen.add(v)
                        words_with_variants.append(v)
                        if len(seen) >= cap:
                            break
                if len(seen) >= cap:
                    break

            first_ids: Set[int] = set()
            prefixes: List[List[int]] = []

            # Include common trailing punctuation and quote-like enders. This helps
            # harvest first-token IDs in tokenizers that fuse trailing chars.
            end_punct = [
                ",",
                ".",
                "?",
                "!",
                ":",
                ";",
            ]
            for w in words_with_variants:
                # Build surfaces for plain word, space+word, and all quote-like
                # prefixes with and without a leading space to mirror tokenizer contexts.
                base_surfaces = [w, f" {w}"]
                for q in self._OPENING_QUOTES:
                    base_surfaces.append(f"{q}{w}")
                    base_surfaces.append(f" {q}{w}")
                surfaces = []
                for s in base_surfaces:
                    surfaces.append(s)
                    for p in end_punct:
                        surfaces.append(s + p)
                for surface in surfaces:
                    try:
                        ids = tokenizer.encode(surface, add_special_tokens=False)
                    except Exception:
                        ids = []
                    if not ids:
                        continue
                    first_idx = 0
                    try:
                        for j, tok in enumerate(ids):
                            s = tokenizer.decode([tok])
                            if re.search(r"[A-Za-z]", s):
                                first_idx = j
                                break
                        tok_id = int(ids[first_idx])
                        first_ids.add(tok_id)
                        prefixes.append(ids[first_idx:])
                    except Exception:
                        first_ids.add(int(ids[0]))
                        prefixes.append(ids)

            # Augment watchlist by scanning vocab for single-piece tokens that
            # start with any banned word after stripping spaces/quotes.
            try:
                vocab_size = getattr(self.orchestrator, "vocab_size", None) or 0
                banned_words: Set[str] = set()
                for orig in matches:
                    low = orig.lower()
                    if low and (low not in STOPWORDS):
                        banned_words.add(low)
                if banned_words and 0 < vocab_size <= 200000:
                    # Build set of leading chars to ignore (spaces + quotes + SentencePiece space)
                    ignore_leading = (
                        "\u0020\n\t\r\f\v\u00a0\u2009\u202f\u3000\u1680"
                        "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u205f\u00ad\u180e"
                        "\u2581"  # SentencePiece space character
                    )
                    for q in self._OPENING_QUOTES:
                        ignore_leading += q
                    added = 0
                    add_cap = 4096
                    for tok_id in range(vocab_size):
                        if tok_id in first_ids:
                            continue
                        try:
                            s = tokenizer.decode([tok_id])
                        except Exception:
                            continue
                        if not s:
                            continue
                        k = 0
                        L = len(s)
                        while k < L and (s[k] in ignore_leading):
                            k += 1
                        if k >= L:
                            continue
                        ch0 = s[k]
                        if not ("A" <= ch0 <= "Z" or "a" <= ch0 <= "z"):
                            continue
                        rem = s[k:].lower()
                        for wlow in banned_words:
                            if rem.startswith(wlow):
                                end = len(wlow)
                                # Word boundary check: next char (if any) is not alpha
                                if end < len(rem) and rem[end : end + 1].isalpha():
                                    continue
                                first_ids.add(int(tok_id))
                                added += 1
                                break
                        if added >= add_cap:
                            break
            except Exception:
                # If augmentation fails for any reason, proceed with existing set
                pass

            if first_ids:
                self.first_token_ids[i] = torch.tensor(
                    sorted(first_ids), dtype=torch.int64, device=device
                )
                self.full_prefixes[i] = prefixes
            else:
                self.first_token_ids[i] = None
                self.full_prefixes[i] = None

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        if output_ids is None or output_ids.numel() == 0:
            return
        self.generated_counts.add_(torch.ones_like(self.generated_counts))

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        B, V = logits.shape
        reqs = self.orchestrator.reqs()

        for i in range(B):
            req = reqs[i]
            first_ids = self.first_token_ids[i]
            if first_ids is None or first_ids.numel() == 0:
                continue
            if int(self.generated_counts[i].item()) >= int(self.guard_window[i].item()):
                continue

            # 2) Determine whether we are at a start position (BOS or after quotes/punctuation)
            _out_ids = getattr(req, "output_ids", None) or []
            is_bos = len(_out_ids) == 0
            is_start = True if is_bos else self._is_start_position(req)
            if not is_start:
                continue

            # 3) Hard block at BOS or any start (if enabled); otherwise apply soft bias
            if is_bos and bool(self.hard_at_bos[i].item()):
                logits[i, first_ids] = -float("inf")
            elif (not is_bos) and bool(self.hard_at_all_starts[i].item()):
                logits[i, first_ids] = -float("inf")
            else:
                bias = float(self.bias_vals[i].item())
                if bias != 0.0:
                    logits[i, first_ids] += bias
        return logits

    def _filter(self, keep_indices: torch.Tensor):
        keep = keep_indices
        self.guard_window = self.guard_window[keep]
        self.hard_at_bos = self.hard_at_bos[keep]
        self.hard_at_all_starts = self.hard_at_all_starts[keep]
        self.bias_vals = self.bias_vals[keep]
        self.generated_counts = self.generated_counts[keep]
        self.first_token_ids = [self.first_token_ids[j] for j in keep.tolist()]
        self.full_prefixes = [self.full_prefixes[j] for j in keep.tolist()]

    def _merge(self, their: "BatchedUserUnigramStartGuardPenalizer"):
        self.guard_window = torch.cat([self.guard_window, their.guard_window], dim=0)
        self.hard_at_bos = torch.cat([self.hard_at_bos, their.hard_at_bos], dim=0)
        self.hard_at_all_starts = torch.cat(
            [self.hard_at_all_starts, their.hard_at_all_starts], dim=0
        )
        self.bias_vals = torch.cat([self.bias_vals, their.bias_vals], dim=0)
        self.generated_counts = torch.cat(
            [self.generated_counts, their.generated_counts], dim=0
        )
        self.first_token_ids.extend(their.first_token_ids)
        self.full_prefixes.extend(their.full_prefixes)

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
