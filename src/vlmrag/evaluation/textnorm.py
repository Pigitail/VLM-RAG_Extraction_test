"""The one normalisation every evaluation metric uses to compare strings.

Model answers and ground truth often differ only in ways a reader would not
call an error: letter case, line breaks or double spaces, an en dash where the
drawing prints a hyphen, typographic quotes. Every metric therefore reports a
*normalised* comparison next to the exact one, and both use this module so the
definition is written down once:

``normalize_value(s)``
    1. dash variants (U+2010-U+2015, U+2212 minus, U+FE58, U+FE63, U+FF0D)
       become ``-``; single-quote variants (U+2018-U+201B, U+2032 prime,
       U+FF07) become ``'``; double-quote variants (U+201C-U+201F, U+2033,
       U+FF02) become ``"``;
    2. ``str.casefold()``;
    3. every whitespace run becomes one space; leading/trailing space is removed.

Nothing else is changed: no Unicode compatibility folding, no punctuation
stripping, no unit conversion. ``"0.42 kg"`` and ``"0.42kg"`` stay different.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["normalize_value", "exact_match", "normalized_match", "mentions", "NORMALIZATION"]

_DASHES = "‐‑‒–—―−﹘﹣－"
_SINGLE_QUOTES = "‘’‚‛′＇"
_DOUBLE_QUOTES = "“”„‟″＂"
_TRANSLATION = str.maketrans(
    {**{c: "-" for c in _DASHES}, **{c: "'" for c in _SINGLE_QUOTES}, **{c: '"' for c in _DOUBLE_QUOTES}}
)
_WS_RE = re.compile(r"\s+")

#: Short description stored in results.json so a reader knows what "normalised" meant for those numbers.
NORMALIZATION = "unify dash and quote variants, casefold, collapse whitespace runs to one space, strip"


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def normalize_value(value: Any) -> str:
    """See the module docstring; ``None`` becomes ``""``."""
    return _WS_RE.sub(" ", _text(value).translate(_TRANSLATION).casefold()).strip()


def exact_match(predicted: Any, truth: Any) -> bool:
    """Character-for-character equality (``None`` counts as ``""``)."""
    return _text(predicted) == _text(truth)


def normalized_match(predicted: Any, truth: Any) -> bool:
    """Equality after :func:`normalize_value`."""
    return normalize_value(predicted) == normalize_value(truth)


def mentions(text: Any, fact: Any) -> bool:
    """``fact`` occurs in ``text`` after normalising both; an empty fact is never "mentioned".

    Used for must-mention recall: the fact has to appear as a contiguous
    substring of the emitted text, so a paraphrase does not count.
    """
    needle = normalize_value(fact)
    return bool(needle) and needle in normalize_value(text)
