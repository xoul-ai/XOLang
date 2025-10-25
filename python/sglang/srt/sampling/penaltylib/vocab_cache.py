from __future__ import annotations

import threading
from typing import Dict, List, Set

# Utilities for caching tokenizer-wide vocab scans used by guards/penalizers.
# These avoid repeated per-request full-vocab decode loops.

_lock = threading.Lock()


class _BigramCache:
    def __init__(
        self,
        vocab_size: int,
        single_token_blacklist: List[int],
        word_with_space_ids: List[int],
        word_no_space_ids: List[int],
        the_first_token_ids: Set[int],
        requires_space: Dict[int, bool],
    ) -> None:
        self.vocab_size = vocab_size
        self.single_token_blacklist = single_token_blacklist
        self.word_with_space_ids = word_with_space_ids
        self.word_no_space_ids = word_no_space_ids
        self.the_first_token_ids = the_first_token_ids
        self.requires_space = requires_space


class _UnigramIndex:
    def __init__(self, vocab_size: int, word_to_token_ids: Dict[str, List[int]]):
        self.vocab_size = vocab_size
        self.word_to_token_ids = word_to_token_ids


_bigram_cache_by_tok: Dict[tuple, _BigramCache] = {}
_unigram_index_by_tok: Dict[tuple, _UnigramIndex] = {}
_unigram_prefix_index_by_tok: Dict[tuple, Dict[str, List[int]]] = {}
_sentence_end_token_ids_cache: Dict[tuple, Set[int]] = {}
# Cache surface variant token IDs: (tokenizer_key, word, quote_chars_tuple) -> (set[token_ids], list[prefixes])
_surface_variant_cache: Dict[tuple, tuple] = {}


_SP_SPACE = "\u2581"


def _strip_leading_quotes_spaces(s: str, quote_chars: Set[str]) -> str:
    i = 0
    L = len(s)
    while i < L and (s[i].isspace() or s[i] in quote_chars):
        i += 1
    return s[i:]


def _is_alpha(c: str) -> bool:
    return ("a" <= c <= "z") or ("A" <= c <= "Z")


def _boundary_after(prefix: str, full: str) -> bool:
    # Require that next char after prefix (if any) is not alphabetic
    n = len(prefix)
    if len(full) <= n:
        return True
    nxt = full[n : n + 1]
    return not (nxt and nxt.isalpha())


def get_bigram_cache(tokenizer, vocab_size: int, quote_chars: Set[str]) -> _BigramCache:
    """Build or fetch cached bigram-related vocab scans for a tokenizer.

    Returns CPU lists/sets of token IDs; callers can move to device as needed.
    """
    # PERFORMANCE FIX: Use stable key based on class and vocab_size instead of id()
    # This prevents cache invalidation across batches
    key = (vocab_size, type(tokenizer).__name__, type(tokenizer).__module__)
    with _lock:
        cached = _bigram_cache_by_tok.get(key)
        if cached is not None:
            return cached

    # Build cache
    single_token_blacklist: List[int] = []
    word_with_space_ids: List[int] = []
    word_no_space_ids: List[int] = []
    the_first_token_ids: Set[int] = set()
    requires_space: Dict[int, bool] = {}

    for tid in range(vocab_size):
        try:
            s = tokenizer.decode([tid])
        except Exception:
            continue
        if not s:
            continue
        s_norm = s.replace(_SP_SPACE, " ")
        s_lower = s_norm.lower()
        s_lstrip_lower = s_norm.lstrip().lower()

        # Single-token blacklist for variants starting with "the word" at BOS
        if s_lstrip_lower.startswith("the word") and _boundary_after("the word", s_lstrip_lower):
            single_token_blacklist.append(int(tid))

        # Second-token candidates with or without leading space
        if s_lower.startswith(" word") and _boundary_after(" word", s_lower):
            word_with_space_ids.append(int(tid))
        if s_lower.startswith("word") and _boundary_after("word", s_lower):
            word_no_space_ids.append(int(tid))

        # First token detection for THE (after stripping spaces and quotes)
        rem = _strip_leading_quotes_spaces(s_norm, quote_chars).lower()
        if rem.startswith("the") and _boundary_after("the", rem):
            the_first_token_ids.add(int(tid))
            # Whether next token requires a leading space variant depends on whether this token endswith space
            requires_space[int(tid)] = not s.endswith(" ")

    single_token_blacklist.sort()
    word_with_space_ids.sort()
    word_no_space_ids.sort()

    built = _BigramCache(
        vocab_size=vocab_size,
        single_token_blacklist=single_token_blacklist,
        word_with_space_ids=word_with_space_ids,
        word_no_space_ids=word_no_space_ids,
        the_first_token_ids=the_first_token_ids,
        requires_space=requires_space,
    )
    with _lock:
        _bigram_cache_by_tok[key] = built
    return built


def get_unigram_first_word_index(
    tokenizer, vocab_size: int, quote_chars: Set[str]
) -> _UnigramIndex:
    """Build or fetch an index: word -> list[token_ids] such that decode(token)
    starts with that word (case-insensitive), after stripping leading spaces/quotes,
    and with a word boundary immediately after the word.
    """
    # PERFORMANCE FIX: Use stable key based on class and vocab_size instead of id()
    # This prevents cache invalidation across batches
    key = (vocab_size, type(tokenizer).__name__, type(tokenizer).__module__)
    with _lock:
        cached = _unigram_index_by_tok.get(key)
        if cached is not None:
            return cached

    word_to_token_ids: Dict[str, List[int]] = {}
    for tid in range(vocab_size):
        try:
            s = tokenizer.decode([tid])
        except Exception:
            continue
        if not s:
            continue
        s_norm = s.replace(_SP_SPACE, " ")
        rem = _strip_leading_quotes_spaces(s_norm, quote_chars)
        if not rem:
            continue
        ch0 = rem[0]
        if not _is_alpha(ch0):
            continue
        # Extract leading alpha word
        j = 1
        L = len(rem)
        while j < L and rem[j].isalpha():
            j += 1
        word = rem[:j].lower()
        # Boundary check: next char (if any) must not be alpha
        if j < L and rem[j : j + 1].isalpha():
            continue
        word_to_token_ids.setdefault(word, []).append(int(tid))

    # Sort for stability
    for w in word_to_token_ids:
        word_to_token_ids[w].sort()

    built = _UnigramIndex(vocab_size=vocab_size, word_to_token_ids=word_to_token_ids)
    with _lock:
        _unigram_index_by_tok[key] = built
    return built


def get_unigram_prefix_index(
    tokenizer, vocab_size: int, quote_chars: Set[str], prefix_len: int
) -> Dict[str, List[int]]:
    """Build or fetch an index: prefix (lowercased, length P) -> list[token_ids].

    Uses the unigram first-word index to group token ids by the first-word prefix of
    length `prefix_len`.
    """
    if prefix_len <= 0:
        return {}
    # PERFORMANCE FIX: Use stable key based on class and vocab_size instead of id()
    # This prevents cache invalidation across batches
    key = (vocab_size, type(tokenizer).__name__, type(tokenizer).__module__, int(prefix_len))
    with _lock:
        cached = _unigram_prefix_index_by_tok.get(key)
        if cached is not None:
            return cached

    idx = get_unigram_first_word_index(tokenizer, vocab_size, quote_chars)
    prefix_map: Dict[str, List[int]] = {}
    for word, ids in idx.word_to_token_ids.items():
        if not word:
            continue
        pref = word[:prefix_len]
        lst = prefix_map.get(pref)
        if lst is None:
            prefix_map[pref] = list(ids)
        else:
            lst.extend(ids)

    # Deduplicate and sort for stability
    for pref, lst in prefix_map.items():
        # unique while preserving ints
        uniq = sorted(set(int(x) for x in lst))
        prefix_map[pref] = uniq

    with _lock:
        _unigram_prefix_index_by_tok[key] = prefix_map
    return prefix_map


def get_sentence_end_token_ids(
    tokenizer, vocab_size: int, quote_chars: Set[str], sentence_end_chars: Set[str]
) -> Set[int]:
    """Build or fetch cached set of token IDs that end with sentence-ending or quote chars.

    Returns a set of token IDs where the decoded token ends with a sentence-ending
    character (after stripping trailing whitespace). This is used to detect sentence
    boundaries without calling decode() in hot paths.
    """
    # PERFORMANCE: Cache by vocab_size and tokenizer class to avoid rebuilding
    key = (vocab_size, type(tokenizer).__name__, type(tokenizer).__module__)
    with _lock:
        cached = _sentence_end_token_ids_cache.get(key)
        if cached is not None:
            return cached

    # Build cache: scan vocab once
    end_ids: Set[int] = set()
    for tid in range(vocab_size):
        try:
            s = tokenizer.decode([tid])
        except Exception:
            continue
        if not s:
            continue
        # Check last non-whitespace character
        j = len(s) - 1
        while j >= 0 and s[j].isspace():
            j -= 1
        if j >= 0:
            ch = s[j]
            if ch in quote_chars or ch in sentence_end_chars:
                end_ids.add(tid)

    with _lock:
        _sentence_end_token_ids_cache[key] = end_ids
    return end_ids


def get_surface_variant_tokens(
    tokenizer,
    word: str,
    quote_chars: Set[str],
    end_punct: List[str],
) -> tuple:
    """Build or fetch cached token IDs for surface variants of a word.

    Returns: (set[token_ids], list[prefixes])

    PERFORMANCE CRITICAL: This caches the expensive encode/decode operations
    per word globally, so they only run once even if _prepare() is called
    multiple times during batch merges.
    """
    import re

    # Create stable cache key
    key = (
        type(tokenizer).__name__,
        type(tokenizer).__module__,
        word,
        tuple(sorted(quote_chars)),
        tuple(end_punct),
    )

    with _lock:
        cached = _surface_variant_cache.get(key)
        if cached is not None:
            return cached

    # Build surface variants (happens once per word)
    first_ids: Set[int] = set()
    prefixes: List[List[int]] = []

    base_surfaces = [word, f" {word}"]
    for q in quote_chars:
        base_surfaces.append(f"{q}{word}")
        base_surfaces.append(f" {q}{word}")

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
                    # Skip contraction fragments
                    s_stripped = s.strip()
                    is_contraction_fragment = (
                        len(s_stripped) <= 3
                        and len(s_stripped) > 0
                        and s_stripped[0] in ("'", '"', """, """, "`")
                    )
                    if is_contraction_fragment:
                        continue
                    first_idx = j
                    break

            tok_id = int(ids[first_idx])
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
            except Exception:
                pass

    result = (first_ids, prefixes)
    with _lock:
        _surface_variant_cache[key] = result
    return result
