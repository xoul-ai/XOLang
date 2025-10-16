from __future__ import annotations

import re
from typing import List, Optional, Set

import torch

from sglang.srt.sampling.penaltylib.orchestrator import (
    BatchedPenalizerOrchestrator,
    _BatchedPenalizer,
)

# Expanded English stopword list to avoid biasing common words.
# Notes:
# - Words are lowercase since we match against lowercased tokens.
# - Include contraction fragments likely produced by regex splitting
#   (e.g., "you're" -> ["you", "re"]).
_STOPWORDS: Set[str] = {
    # Core determiners/pronouns/auxiliaries
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "cannot",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "let",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    # Adverbs/quantifiers commonly function-word-ish
    "also",
    "already",
    "always",
    "because",
    "beforehand",
    "besides",
    "ever",
    "every",
    "everyone",
    "everything",
    "everywhere",
    "however",
    "indeed",
    "instead",
    "less",
    "least",
    "maybe",
    "moreover",
    "mostly",
    "much",
    "neither",
    "never",
    "non",
    "nothing",
    "now",
    "often",
    "perhaps",
    "quite",
    "rather",
    "really",
    "several",
    "since",
    "still",
    "such",
    "then",
    "thus",
    "today",
    "together",
    "too",
    "toward",
    "towards",
    "well",
    "yet",
    # Contraction fragments commonly split by regex
    "re",
    "ve",
    "ll",
    "d",
    "s",
    "m",
    "t",
    "nt",
    # Negation/auxiliary stems that appear without apostrophes
    "aren",
    "couldn",
    "didn",
    "doesn",
    "don",
    "hadn",
    "hasn",
    "haven",
    "isn",
    "mightn",
    "mustn",
    "needn",
    "shan",
    "shouldn",
    "wasn",
    "weren",
    "won",
    "wouldn",
}


class BatchedUserUnigramStartGuardPenalizer(_BatchedPenalizer):
    """Biases/blocks starting tokens that match user unigrams."""

    # Delimiters considered as start positions when last non-space char
    _SENTENCE_END_CHARS = {".", "!", "?", "\n", "…"}

    # A comprehensive set of quote-like characters. We include both opening and
    # closing variants to be robust to tokenizer/model usage.
    _QUOTE_CHARS = {
        '"',
        "'",
        "`",
        "“",  # U+201C Left double quotation mark
        "”",  # U+201D Right double quotation mark
        "„",  # U+201E Double low-9 quotation mark
        "‟",  # U+201F Double high-reversed-9 quotation mark
        "‘",  # U+2018 Left single quotation mark
        "’",  # U+2019 Right single quotation mark
        "‚",  # U+201A Single low-9 quotation mark
        "‛",  # U+201B Single high-reversed-9 quotation mark
        "«",  # U+00AB Left-pointing double angle quotation mark
        "»",  # U+00BB Right-pointing double angle quotation mark
        "‹",  # U+2039 Single left-pointing angle quotation mark
        "›",  # U+203A Single right-pointing angle quotation mark
        "「",  # U+300C Left corner bracket
        "」",  # U+300D Right corner bracket
        "『",  # U+300E Left white corner bracket
        "』",  # U+300F Right white corner bracket
        "〈",  # U+3008 Left angle bracket
        "〉",  # U+3009 Right angle bracket
        "《",  # U+300A Left double angle bracket
        "》",  # U+300B Right double angle bracket
    }

    # Characters that should count as a start when appearing before the next token.
    # Historically included '*', keep it for markdown/emphasis scenarios.
    _OPENING_QUOTES = _QUOTE_CHARS | {"*"}

    def __init__(self, orchestrator: BatchedPenalizerOrchestrator):
        self.orchestrator = orchestrator
        self._is_prepared = False

    def _is_required(self) -> bool:
        reqs = self.orchestrator.reqs()
        if not reqs:
            return False
        for req in reqs:
            cp = getattr(req.sampling_params, "custom_params", None)
            if isinstance(cp, dict):
                if cp.get("unigrams_text"):
                    return True
                # Also enable if start-bigram guard is explicitly configured
                if cp.get("ban_start_bigram_the_word_hard") or float(
                    cp.get("ban_start_bigram_the_word_bias", 0.0) or 0.0
                ) != 0.0:
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
        # Optional start-bigram guard specifically for discouraging "The word"
        self.bigram_bias_vals: torch.Tensor = torch.zeros(
            (len(reqs),), dtype=torch.float32, device=device
        )
        self.bigram_hard: torch.Tensor = torch.zeros(
            (len(reqs),), dtype=torch.bool, device=device
        )
        self.generated_counts: torch.Tensor = torch.zeros(
            (len(reqs),), dtype=torch.int32, device=device
        )

        self.first_token_ids: List[Optional[torch.Tensor]] = [None] * len(reqs)
        self.full_prefixes: List[Optional[List[List[int]]]] = [None] * len(reqs)
        self.bigram_second_token_ids: List[Optional[torch.Tensor]] = [None] * len(reqs)

        for i, req in enumerate(reqs):
            cp = getattr(req.sampling_params, "custom_params", None) or {}
            text = str(cp.get("unigrams_text", "") or "")

            window = int(cp.get("ban_user_unigram_guard_window", 40) or 40)
            hard_bos = bool(cp.get("ban_user_unigram_hard_at_bos", True))
            hard_all = bool(cp.get("ban_user_unigram_hard_at_all_starts", False))
            bias = float(cp.get("ban_user_unigram_bias", -0.9) or -0.9)
            # Start-bigram ("The word") controls — default disabled (non-interfering)
            bigram_bias = float(cp.get("ban_start_bigram_the_word_bias", 0.0) or 0.0)
            bigram_hard = bool(cp.get("ban_start_bigram_the_word_hard", False))

            self.guard_window[i] = window
            self.hard_at_bos[i] = hard_bos
            self.hard_at_all_starts[i] = hard_all
            self.bias_vals[i] = bias
            self.bigram_bias_vals[i] = bigram_bias
            self.bigram_hard[i] = bigram_hard

            # Continue to compute bigram surfaces even if `text` is empty

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
            # Include common trailing punctuation and quote-like enders. This helps
            # harvest first-token IDs in tokenizers that fuse trailing chars.
            end_punct = [
                ",",
                ".",
                "?",
                "!",
                ":",
                ";",
                "'",
                '"',
                "’",
                "”",
                "*",
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

            if first_ids:
                self.first_token_ids[i] = torch.tensor(
                    sorted(first_ids), dtype=torch.int64, device=device
                )
                self.full_prefixes[i] = prefixes
            else:
                self.first_token_ids[i] = None
                self.full_prefixes[i] = None

            # Precompute second-word first token IDs for the bigram "The word"
            # Only handle raw phrase with a single space: "The word" / "the word".
            if bigram_bias != 0.0 or bigram_hard:
                second_ids: Set[int] = set()
                # Only the space-prefixed surface to capture typical tokenizers (e.g., BPE)
                second_surfaces = [" word"]
                for s in second_surfaces:
                    try:
                        ids = tokenizer.encode(s, add_special_tokens=False)
                    except Exception:
                        ids = []
                    if not ids:
                        continue
                    try:
                        first_idx = 0
                        for j, tok in enumerate(ids):
                            token_str = tokenizer.decode([tok])
                            if re.search(r"[A-Za-z]", token_str):
                                first_idx = j
                                break
                        second_ids.add(int(ids[first_idx]))
                    except Exception:
                        second_ids.add(int(ids[0]))

                if second_ids:
                    self.bigram_second_token_ids[i] = torch.tensor(
                        sorted(second_ids), dtype=torch.int64, device=device
                    )
                else:
                    self.bigram_second_token_ids[i] = None
            else:
                self.bigram_second_token_ids[i] = None

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        if output_ids is None or output_ids.numel() == 0:
            return
        self.generated_counts.add_(torch.ones_like(self.generated_counts))

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        B, V = logits.shape
        reqs = self.orchestrator.reqs()

        for i in range(B):
            # Start-bigram guard: discourage "The word" as the very beginning phrase.
            # This runs independently of the unigram start guard and is only active
            # if configured via custom params.
            req = reqs[i]
            second_ids = (
                self.bigram_second_token_ids[i]
                if i < len(self.bigram_second_token_ids)
                else None
            )
            if (
                second_ids is not None
                and second_ids.numel() > 0
                and int(self.generated_counts[i].item()) < int(self.guard_window[i].item())
                and self._starts_with_word(req, target="the")
            ):
                if bool(self.bigram_hard[i].item()):
                    logits[i, second_ids] = -float("inf")
                else:
                    bb = float(self.bigram_bias_vals[i].item())
                    if bb != 0.0:
                        logits[i, second_ids] += bb

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

    def _filter(self, keep_indices: torch.Tensor):
        keep = keep_indices
        self.guard_window = self.guard_window[keep]
        self.hard_at_bos = self.hard_at_bos[keep]
        self.hard_at_all_starts = self.hard_at_all_starts[keep]
        self.bias_vals = self.bias_vals[keep]
        self.bigram_bias_vals = self.bigram_bias_vals[keep]
        self.bigram_hard = self.bigram_hard[keep]
        self.generated_counts = self.generated_counts[keep]
        self.first_token_ids = [self.first_token_ids[j] for j in keep.tolist()]
        self.full_prefixes = [self.full_prefixes[j] for j in keep.tolist()]
        self.bigram_second_token_ids = [
            self.bigram_second_token_ids[j] for j in keep.tolist()
        ]

    def _merge(self, their: "BatchedUserUnigramStartGuardPenalizer"):
        self.guard_window = torch.cat([self.guard_window, their.guard_window], dim=0)
        self.hard_at_bos = torch.cat([self.hard_at_bos, their.hard_at_bos], dim=0)
        self.hard_at_all_starts = torch.cat(
            [self.hard_at_all_starts, their.hard_at_all_starts], dim=0
        )
        self.bias_vals = torch.cat([self.bias_vals, their.bias_vals], dim=0)
        self.bigram_bias_vals = torch.cat(
            [self.bigram_bias_vals, their.bigram_bias_vals], dim=0
        )
        self.bigram_hard = torch.cat([self.bigram_hard, their.bigram_hard], dim=0)
        self.generated_counts = torch.cat(
            [self.generated_counts, their.generated_counts], dim=0
        )
        self.first_token_ids.extend(their.first_token_ids)
        self.full_prefixes.extend(their.full_prefixes)
        self.bigram_second_token_ids.extend(their.bigram_second_token_ids)

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
        is_start = ch in self._OPENING_QUOTES or ch in self._SENTENCE_END_CHARS
        return is_start

    def _starts_with_word(self, req, target: str) -> bool:
        """True if generated output begins with the target word (case-insensitive),
        with no preceding quotes/asterisks; allows leading spaces only; and there
        are no additional non-space characters after that first word."""
        _out_ids = getattr(req, "output_ids", None) or []
        if len(_out_ids) == 0:
            return False
        tokenizer = getattr(req, "tokenizer", None)
        if tokenizer is None:
            return False
        try:
            text = tokenizer.decode(_out_ids)
        except Exception:
            return False
        if not text:
            return False
        i = 0
        n = len(text)
        # Skip leading spaces
        while i < n and text[i].isspace():
            i += 1
        # Do NOT skip quotes/asterisks; we only want the raw phrase case.
        m = re.match(r"([A-Za-z]+)\b", text[i:])
        if not m or m.group(1).lower() != target.lower():
            return False
        # Ensure we're immediately after the first word (ignore trailing spaces)
        j = i + len(m.group(0))
        remainder = text[j:]
        return remainder.strip() == ""
