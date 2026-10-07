"""Exact token counting for the round input budgets.

The harness used to approximate tokens as UTF-8 bytes // 4.  On this benchmark
that approximation is not usable: every task is Chinese but the retrieved pages
are a mixture of Chinese and English, and the error runs in both directions.
Measured on real round-one inputs the ratio of server-reported prompt tokens to
the approximation ranged from 0.77 to 1.49, so a "120k token" round was really
anywhere between 58k and 179k.  Since the whole comparison is between engines at
controlled input lengths, the budget has to be counted with the model's own
tokenizer.
"""
from __future__ import annotations

import os
import threading

_DEFAULT_TOKENIZER = "models/GLM-4.7-Flash/tokenizer.json"
_lock = threading.Lock()
_tokenizer = None
_unavailable = False


def _approximate(text: str) -> int:
    return max(1, len(text.encode("utf-8")) // 4)


def _get():
    """Load the tokenizer once per process; fall back to the approximation."""
    global _tokenizer, _unavailable
    if _tokenizer is not None or _unavailable:
        return _tokenizer
    with _lock:
        if _tokenizer is None and not _unavailable:
            path = os.environ.get("MTBENCH_TOKENIZER", _DEFAULT_TOKENIZER)
            try:
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                from tokenizers import Tokenizer

                _tokenizer = Tokenizer.from_file(path)
            except Exception:
                _unavailable = True
    return _tokenizer


def count_tokens(text: str) -> int:
    if not text:
        return 1
    tokenizer = _get()
    if tokenizer is None:
        return _approximate(text)
    try:
        return max(1, len(tokenizer.encode(text, add_special_tokens=False).ids))
    except Exception:
        return _approximate(text)


def truncate_to_tokens(text: str, tokens: int) -> str:
    """Cut text to exactly `tokens` tokens on a character boundary.

    Applied only to the single document that crosses a round budget; every
    earlier document is kept whole.
    """
    if tokens <= 0:
        return ""
    tokenizer = _get()
    if tokenizer is None:
        raw = text.encode("utf-8")
        if len(raw) // 4 <= tokens:
            return text
        return raw[: tokens * 4].decode("utf-8", errors="ignore")
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded.ids) <= tokens:
            return text
        end = encoded.offsets[tokens - 1][1]
        return text[:end]
    except Exception:
        return text
