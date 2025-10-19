from __future__ import annotations

import logging
import re
from typing import List, Optional, Set

import torch

logger = logging.getLogger(__name__)

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

        # Track hard blocks applied at last step per request (list of 1-D tensors)
        self._last_hard_blocks: List[Optional[torch.Tensor]] = [None] * len(reqs)

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

            # Match words including contractions (e.g., "don't", "we're", "would've")
            matches = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text) if text else []

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
                                # Skip contraction fragments: tokens that are just apostrophe + short suffix
                                # like 't, 's, 're, 'm, 'd, 'll, 've which are parts of contractions
                                s_stripped = s.strip()
                                # Check if it's a contraction fragment: starts with apostrophe/quote and has <=3 chars total
                                is_contraction_fragment = (
                                    len(s_stripped) <= 3
                                    and len(s_stripped) > 0
                                    and s_stripped[0] in ("'", '"', """, """, "`")
                                )
                                if is_contraction_fragment:
                                    continue  # Skip this token, keep looking
                                first_idx = j
                                break
                        tok_id = int(ids[first_idx])
                        # Double-check: skip if it's a contraction fragment
                        s_check = tokenizer.decode([tok_id]).strip()
                        is_frag = (
                            len(s_check) <= 3
                            and len(s_check) > 0
                            and s_check[0] in ("'", '"', """, """, "`")
                        )
                        if not is_frag:
                            first_ids.add(tok_id)
                            prefixes.append(ids[first_idx:])
                    except Exception:
                        # Also check exception path
                        try:
                            s_check = tokenizer.decode([int(ids[0])]).strip()
                            is_frag = (
                                len(s_check) <= 3
                                and len(s_check) > 0
                                and s_check[0] in ("'", '"', """, """, "`")
                            )
                            if not is_frag:
                                first_ids.add(int(ids[0]))
                                prefixes.append(ids)
                        except:
                            pass

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
            except Exception as e:
                # If augmentation fails for any reason, proceed with existing set
                logger.info(f"UNIGRAM_DEBUG: req_idx={i} vocab scan exception: {e}")

            if tokenizer and first_ids:
                dont_tokens = []
                let_tokens = []
                sample_all_tokens = []

                for tid in list(first_ids):  # Check ALL tokens
                    try:
                        decoded = tokenizer.decode([tid])
                        decoded_lower = decoded.lower()
                        if "don" in decoded_lower:
                            dont_tokens.append((tid, repr(decoded)))
                        if "let" in decoded_lower:
                            let_tokens.append((tid, repr(decoded)))
                        # Sample first 80 blocked tokens for analysis
                        if len(sample_all_tokens) < 80:
                            sample_all_tokens.append((tid, repr(decoded)))
                    except:
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
        import logging
        logger = logging.getLogger(__name__)
        B, V = logits.shape

        # Copy tensor/list references at start to prevent race conditions
        guard_window = self.guard_window
        hard_at_bos = self.hard_at_bos
        hard_at_all_starts = self.hard_at_all_starts
        bias_vals = self.bias_vals
        generated_counts = self.generated_counts
        first_token_ids = self.first_token_ids
        last_hard_blocks = self._last_hard_blocks

        reqs = self.orchestrator.reqs()
        # If reqs unavailable or batch size mismatch, skip
        if reqs is None:
            logger.info("UnigramGuard _apply: reqs is None, skipping")
            return logits
        if len(reqs) != B:
            logger.info(f"UnigramGuard _apply: batch size mismatch, reqs={len(reqs)} vs B={B}, skipping")
            return logits
        if len(guard_window) != B:
            logger.info(f"UnigramGuard _apply: tensor size mismatch, guard_window={len(guard_window)} vs B={B}, skipping")
            return logits
        # Check list sizes as well (these are not tensors, so they need separate validation)
        if len(first_token_ids) != B:
            logger.info(f"UnigramGuard _apply: list size mismatch, first_token_ids={len(first_token_ids)} vs B={B}, skipping")
            return logits
        if len(last_hard_blocks) != B:
            logger.info(f"UnigramGuard _apply: list size mismatch, _last_hard_blocks={len(last_hard_blocks)} vs B={B}, skipping")
            return logits
        # Reset last hard-blocks
        for j in range(B):
            last_hard_blocks[j] = None
        for i in range(B):
            req = reqs[i]
            first_ids = first_token_ids[i]
            if first_ids is None or first_ids.numel() == 0:
                continue
            if int(generated_counts[i].item()) >= int(guard_window[i].item()):
                continue

            # 2) Determine whether we are at a start position (BOS or after quotes/punctuation)
            _out_ids = getattr(req, "output_ids", None) or []
            is_bos = len(_out_ids) == 0
            is_start = True if is_bos else self._is_start_position(req)
            if not is_start:
                continue

            # 3) Hard block at BOS or any start (if enabled); otherwise apply soft bias
            rid = getattr(req, "rid", None)
            tok = getattr(req, "tokenizer", None)
            if is_bos and bool(hard_at_bos[i].item()):
                # Check if "don't" is being blocked
                dont_blocked = []
                if tok:
                    for tid in first_ids[:20].tolist():
                        try:
                            decoded = tok.decode([tid])
                            if "don" in decoded.lower():
                                dont_blocked.append((tid, decoded))
                        except:
                            pass
                logits[i, first_ids] = -float("inf")
                last_hard_blocks[i] = first_ids
            elif (not is_bos) and bool(hard_at_all_starts[i].item()):
                dont_blocked = []
                if tok:
                    for tid in first_ids[:20].tolist():
                        try:
                            decoded = tok.decode([tid])
                            if "don" in decoded.lower():
                                dont_blocked.append((tid, decoded))
                        except:
                            pass
                logits[i, first_ids] = -float("inf")
                last_hard_blocks[i] = first_ids
            else:
                bias = float(bias_vals[i].item())
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
        self._last_hard_blocks = [self._last_hard_blocks[j] for j in keep.tolist()]

    def _merge(self, their: "BatchedUserUnigramStartGuardPenalizer"):
        import logging
        logger = logging.getLogger(__name__)
        old_len = len(self.guard_window)
        their_len = len(their.guard_window)

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
        self._last_hard_blocks.extend(their._last_hard_blocks)

        new_len = len(self.guard_window)
        logger.info(f"UnigramGuard _merge: merged tensors {old_len} + {their_len} = {new_len}")

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

    def get_last_hard_block_ids(self):
        return self._last_hard_blocks

    def get_computed_hard_block_ids(self):
        # Compute current hard blocks based on request state (BOS / start detection)
        reqs = self.orchestrator.reqs()
        if not reqs:
            return None
        out: List[Optional[torch.Tensor]] = [None] * len(reqs)
        for i, req in enumerate(reqs):
            first_ids = self.first_token_ids[i]
            if first_ids is None or (
                hasattr(first_ids, "numel") and first_ids.numel() == 0
            ):
                continue
            # Respect window if we can infer tokens generated
            try:
                total_emitted = len(getattr(req, "output_ids", []) or [])
                if total_emitted >= int(self.guard_window[i].item()):
                    continue
            except Exception:
                pass
            out_ids_list = getattr(req, "output_ids", []) or []
            is_bos = len(out_ids_list) == 0
            if is_bos and bool(self.hard_at_bos[i].item()):
                out[i] = first_ids
                continue
            if (not is_bos) and bool(self.hard_at_all_starts[i].item()):
                if self._is_start_position(req):
                    out[i] = first_ids
        return out
