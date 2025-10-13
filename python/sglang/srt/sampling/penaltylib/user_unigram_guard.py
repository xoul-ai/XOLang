from __future__ import annotations

import re
import logging
from typing import List, Optional, Set, Tuple

import torch

from sglang.srt.sampling.penaltylib.orchestrator import (
    BatchedPenalizerOrchestrator,
    _BatchedPenalizer,
)

logger = logging.getLogger(__name__)

# A compact English stopword list to avoid over-guarding trivial words
_STOPWORDS: Set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "if",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "will",
    "with",
    "you",
    "your",
}


class BatchedUserUnigramStartGuardPenalizer(_BatchedPenalizer):
    """
    Case-insensitive user-unigram start guard.

    Reads from SamplingParams.custom_params per request:
      - unigrams_text: raw user text to tokenize and extract watchlist words
      - ban_user_unigram_guard_window: int (default 40)
      - ban_user_unigram_hard_at_bos: bool (default True)
      - ban_user_unigram_bias: float (default -0.9)

    Behavior for first N newly generated tokens and only at "start positions":
      - BOS and `hard_at_bos=True`: set logits of matching first-token ids to -inf
      - Otherwise (within window): add a small negative bias to matching first-token ids
    """

    # Start delimiters that define a start position when they are the last non-space char
    _SENTENCE_END_CHARS = {".", "!", "?", "\n", "…"}
    _OPENING_QUOTES = {'"', "“", "‘", "'", "*"}

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
                logger.info(
                    "[UnigramGuard] activation detected for rid=%s",
                    getattr(req, "rid", "<unknown>"),
                )
                return True
        return False

    def _prepare(self):
        reqs = self.orchestrator.reqs()
        device = self.orchestrator.device

        # Per-request state
        self.guard_window: torch.Tensor = torch.zeros(
            (len(reqs),), dtype=torch.int32, device=device
        )
        self.hard_at_bos: torch.Tensor = torch.zeros(
            (len(reqs),), dtype=torch.bool, device=device
        )
        self.bias_vals: torch.Tensor = torch.zeros(
            (len(reqs),), dtype=torch.float32, device=device
        )
        self.generated_counts: torch.Tensor = torch.zeros(
            (len(reqs),), dtype=torch.int32, device=device
        )

        # Token watchlists per request
        # - first token ids for soft/hard operations
        # - full token id sequences (prefixes) kept for possible future extension
        self.first_token_ids: List[Optional[torch.Tensor]] = [None] * len(reqs)
        self.full_prefixes: List[Optional[List[List[int]]]] = [None] * len(reqs)

        for i, req in enumerate(reqs):
            cp = getattr(req.sampling_params, "custom_params", None) or {}
            text = str(cp.get("unigrams_text", "") or "")

            # Defaults
            window = int(cp.get("ban_user_unigram_guard_window", 40) or 40)
            hard_bos = bool(cp.get("ban_user_unigram_hard_at_bos", True))
            bias = float(cp.get("ban_user_unigram_bias", -0.9) or -0.9)

            self.guard_window[i] = window
            self.hard_at_bos[i] = hard_bos
            self.bias_vals[i] = bias

            if not text:
                logger.info(
                    "[UnigramGuard][rid=%s] no unigrams_text provided; skipping",
                    getattr(req, "rid", "<unknown>"),
                )
                continue

            tokenizer = getattr(req, "tokenizer", None)
            if tokenizer is None:
                # If tokenizer is unavailable, skip for this request
                logger.warning(
                    "[UnigramGuard][rid=%s] tokenizer missing; guard disabled",
                    getattr(req, "rid", "<unknown>"),
                )
                continue

            # Extract alphabetic words from raw text, keep originals for casing variants
            matches = re.findall(r"[A-Za-z]+", text)
            if not matches:
                logger.info(
                    "[UnigramGuard][rid=%s] no alphabetic words extracted from text length=%d",
                    getattr(req, "rid", "<unknown>"),
                    len(text),
                )
                continue

            # Build a capped set of unique words (ignore stopwords)
            cap = 1500
            seen: Set[str] = set()
            words_with_variants: List[str] = []
            for orig in matches:
                low = orig.lower()
                if low in _STOPWORDS:
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
            end_punct = [",", ".", "?", "!", ":", ";"]
            for w in words_with_variants:
                # Try common surfaces for sentence starts and quotes
                # Base leading variants
                base_surfaces = [
                    w,
                    f" {w}",
                    f'"{w}',
                    f"“{w}",
                    f"‘{w}",
                    f"'{w}",
                    f" '{w}",
                    f"*{w}",
                    f" *{w}",
                ]
                # Include punctuation-suffixed variants (to match BPE-fused tokens like 'Lawyers?')
                surfaces = []
                for s in base_surfaces:
                    surfaces.append(s)
                    for p in end_punct:
                        surfaces.append(s + p)
                for surface in surfaces:
                    try:
                        ids = tokenizer.encode(surface, add_special_tokens=False)
                    except Exception:
                        logger.exception(
                            "[UnigramGuard][rid=%s] tokenizer.encode failed for surface=%r",
                            getattr(req, "rid", "<unknown>"),
                            surface,
                        )
                        ids = []
                    if not ids:
                        continue
                    # Skip leading tokens that are pure punctuation/space; pick first token with alpha
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
                        # Fallback to the very first id if decode fails
                        first_ids.add(int(ids[0]))
                        prefixes.append(ids)

            if first_ids:
                self.first_token_ids[i] = torch.tensor(
                    sorted(first_ids), dtype=torch.int64, device=device
                )
                self.full_prefixes[i] = prefixes
                logger.info(
                    "[UnigramGuard][rid=%s] prepared: window=%d hard_at_bos=%s bias=%.3f watch_first_ids=%d prefixes=%d",
                    getattr(req, "rid", "<unknown>"),
                    int(window),
                    hard_bos,
                    bias,
                    len(first_ids),
                    len(prefixes),
                )
            else:
                self.first_token_ids[i] = None
                self.full_prefixes[i] = None
                logger.info(
                    "[UnigramGuard][rid=%s] prepared but empty watchlist after tokenization",
                    getattr(req, "rid", "<unknown>"),
                )

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        # Count how many new tokens have been generated per request
        # output_ids has shape [B], int64
        if output_ids is None or output_ids.numel() == 0:
            return
        self.generated_counts.add_(torch.ones_like(self.generated_counts))
        logger.info(
            "[UnigramGuard] updated generated_counts=%s",
            self.generated_counts.tolist(),
        )

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        # Apply guard only within window and at start positions
        B, V = logits.shape
        reqs = self.orchestrator.reqs()

        for i in range(B):
            first_ids = self.first_token_ids[i]
            if first_ids is None or first_ids.numel() == 0:
                continue
            if int(self.generated_counts[i].item()) >= int(self.guard_window[i].item()):
                continue

            req = reqs[i]
            is_bos = len(req.output_ids) == 0
            if not is_bos and not self._is_start_position(req):
                continue

            if is_bos and bool(self.hard_at_bos[i].item()):
                # HARD: ban starting with any watchlisted word at BOS
                logits[i, first_ids] = -float("inf")
                logger.info(
                    "[UnigramGuard][rid=%s] HARD BOS applied to %d tokens",
                    getattr(req, "rid", "<unknown>"),
                    int(first_ids.numel()),
                )
            else:
                # SOFT: add a small negative bias to disfavor starts
                bias = float(self.bias_vals[i].item())
                if bias != 0.0:
                    logits[i, first_ids] += bias
                    logger.info(
                        "[UnigramGuard][rid=%s] SOFT bias %.3f applied to %d tokens",
                        getattr(req, "rid", "<unknown>"),
                        bias,
                        int(first_ids.numel()),
                    )

    def _filter(self, keep_indices: torch.Tensor):
        keep = keep_indices
        self.guard_window = self.guard_window[keep]
        self.hard_at_bos = self.hard_at_bos[keep]
        self.bias_vals = self.bias_vals[keep]
        self.generated_counts = self.generated_counts[keep]
        self.first_token_ids = [self.first_token_ids[j] for j in keep.tolist()]
        self.full_prefixes = [self.full_prefixes[j] for j in keep.tolist()]

    def _merge(self, their: "BatchedUserUnigramStartGuardPenalizer"):
        self.guard_window = torch.cat([self.guard_window, their.guard_window], dim=0)
        self.hard_at_bos = torch.cat([self.hard_at_bos, their.hard_at_bos], dim=0)
        self.bias_vals = torch.cat([self.bias_vals, their.bias_vals], dim=0)
        self.generated_counts = torch.cat(
            [self.generated_counts, their.generated_counts], dim=0
        )
        self.first_token_ids.extend(their.first_token_ids)
        self.full_prefixes.extend(their.full_prefixes)

    def _is_start_position(self, req) -> bool:
        # True if BOS or last non-space char is an opening quote or sentence-ending punctuation
        if len(req.output_ids) == 0:
            return True
        tokenizer = getattr(req, "tokenizer", None)
        if tokenizer is None:
            return False

        # Decode up to the last 12 tokens to capture punctuation even if split/merged
        n = min(12, len(req.output_ids))
        try:
            tail = tokenizer.decode(req.output_ids[-n:])
        except Exception:
            logger.exception(
                "[UnigramGuard][rid=%s] tail decode failed",
                getattr(req, "rid", "<unknown>"),
            )
            return False
        if not tail:
            return False
        # Consider the last non-space character
        # Avoid expensive rstrip on very long strings by scanning from the end
        i = len(tail) - 1
        while i >= 0 and tail[i].isspace():
            i -= 1
        if i < 0:
            return False
        ch = tail[i]
        is_start = ch in self._OPENING_QUOTES or ch in self._SENTENCE_END_CHARS
        logger.info(
            "[UnigramGuard][rid=%s] is_start_position=%s last_char=%r tail=%r (n=%d)",
            getattr(req, "rid", "<unknown>"),
            is_start,
            ch,
            tail,
            n,
        )
        return is_start
