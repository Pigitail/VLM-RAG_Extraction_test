"""Small text helpers shared by all techniques."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Sequence

__all__ = [
    "normalize_ws",
    "normalize_unicode",
    "filtering_text",
    "markdown_table",
    "plain_table",
    "truncate",
    "dedupe_keep_order",
]

_WS_RE = re.compile(r"\s+")
_DASH_RE = re.compile(r"[‐‑‒–—−]")


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def normalize_ws(s: Any) -> str:
    """Collapse every whitespace run (including newlines) to one space and strip."""
    return _WS_RE.sub(" ", _text(s)).strip()


def normalize_unicode(s: Any) -> str:
    """NFKC-normalise, unify curly quotes and dash variants, then :func:`normalize_ws`.

    Text layers and model output mix full-width digits, typographic quotes and
    several dash code points; normalising them makes exact-match comparisons
    (hint matching, field merging) behave as a human would expect.
    """
    text = unicodedata.normalize("NFKC", _text(s))
    text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    text = _DASH_RE.sub("-", text)
    return normalize_ws(text)


def filtering_text(s: Any) -> str:
    """Remove all whitespace: the representation used for exact substring filters."""
    return _WS_RE.sub("", _text(s))


def _cell(value: Any, markdown: bool) -> str:
    text = normalize_ws(value)
    return text.replace("|", "\\|") if markdown else text


def markdown_table(rows: Sequence[Sequence[Any]], header: bool = True) -> str:
    """Render rows as a GitHub-flavoured Markdown table.

    With ``header=False`` an empty header row is emitted, because Markdown
    tables cannot exist without one and promoting a data row to a header would
    misrepresent it to the answering LLM. Ragged rows are padded.
    """
    rows = [list(r) for r in rows if r is not None]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    if width == 0:
        return ""
    padded = [[_cell(c, True) for c in r] + [""] * (width - len(r)) for r in rows]
    if header:
        head, body = padded[0], padded[1:]
    else:
        head, body = [""] * width, padded
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines.extend("| " + " | ".join(r) + " |" for r in body)
    return "\n".join(lines)


def plain_table(rows: Sequence[Sequence[Any]]) -> str:
    """One line per row, cells joined with ``" | "`` (empty cells dropped).

    This is the retrieval rendering of a table: no Markdown separators that
    would only add tokens, and every cell value kept on the same line as its row
    neighbours so BM25 and embeddings see them together.
    """
    lines = []
    for row in rows:
        cells = [_cell(c, False) for c in (row or [])]
        cells = [c for c in cells if c]
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def truncate(s: Any, limit: int) -> str:
    """Cut ``s`` to at most ``limit`` characters (no ellipsis, so limits stay exact)."""
    text = _text(s)
    if limit is None or limit < 0 or len(text) <= limit:
        return text
    return text[:limit]


def dedupe_keep_order(items: Iterable[Any]) -> list:
    """Drop empty and repeated items while preserving first-seen order."""
    seen = set()
    out = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
