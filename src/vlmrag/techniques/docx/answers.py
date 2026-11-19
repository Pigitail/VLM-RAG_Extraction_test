"""Parsing and strict validation of the model's answers (pure functions).

The model's output is treated as untrusted input. Image descriptions are
filtered to well-formed entries. Table answers go through an all-or-nothing
validation of the row assignment: the answer is accepted only if it assigns
every XML row exactly once to a page inside the table's span. Anything else -
a missing row, a duplicate, an unknown page, a non-integer - rejects the whole
assignment, and the caller falls back to the editor's own row markers. A
partially right assignment is never "repaired", because a repair would have to
guess, and a guessed row on the wrong page is worse than a known fallback.

The title is validated separately and can only be *dropped*, never invented:
it must occur (case- and whitespace-insensitively) in the paragraph or first
row the model says it came from, and a ``"low"`` confidence title is ignored.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .prompts import TITLE_CONFIDENCES, TITLE_SOURCES
from .xmlsplit import MultiPageTable

__all__ = [
    "ImageAnswer",
    "TableAnswer",
    "parse_image_answer",
    "validate_table_answer",
    "fallback_distribution",
]

log = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------- image descriptions


@dataclass
class ImageAnswer:
    """Outcome of parsing one image-description answer.

    ``status``: ``"ok"`` (at least one description), ``"no_visuals"`` (a valid
    answer with nothing to describe), ``"invalid"`` (JSON without the expected
    shape) or ``"unparseable"`` (no JSON at all).
    """

    status: str
    descriptions: list[dict] = field(default_factory=list)


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def parse_image_answer(value: Any) -> ImageAnswer:
    """Turn a parsed JSON value into description entries.

    Entries without a description are dropped; a non-empty ``notes`` becomes
    one more entry with location ``"notes"``.
    """
    if value is None:
        return ImageAnswer("unparseable")
    if not isinstance(value, dict) or not isinstance(value.get("image_descriptions"), list):
        return ImageAnswer("invalid")
    descriptions: list[dict] = []
    for item in value["image_descriptions"]:
        if not isinstance(item, dict):
            continue
        description = _text(item.get("description"))
        if description:
            descriptions.append({"location_hint": _text(item.get("location_hint")), "description": description})
    notes = _text(value.get("notes"))
    if notes:
        descriptions.append({"location_hint": "notes", "description": notes})
    return ImageAnswer("ok" if descriptions else "no_visuals", descriptions)


# --------------------------------------------------------------------------- table distribution


@dataclass
class TableAnswer:
    """A validated row assignment (``row_pages`` maps every page of the span to row indexes)."""

    row_pages: dict[int, list[int]]
    title: str = ""
    title_source: str = "none"
    title_confidence: str = "low"
    title_rejected: bool = False


def _as_int(value: Any) -> Optional[int]:
    """Strict integer conversion: ints, integral floats and digit strings only (never booleans)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str) and re.fullmatch(r"\s*-?\d+\s*", value):
        return int(value)
    return None


def _norm(value: Any) -> str:
    return _WS_RE.sub(" ", _text(value)).lower()


def validate_table_answer(value: Any, table: MultiPageTable) -> tuple[Optional[TableAnswer], str]:
    """Validate one table answer; return ``(answer, "")`` or ``(None, reason)``.

    Row checks (any failure rejects the answer):

    * ``row_distribution`` is an object whose keys are integers within
      ``[start_page, end_page]`` and whose values are lists of integers;
    * no row index appears twice;
    * the set of indexes equals ``{0, ..., N-1}`` exactly;
    * reading the pages in order gives the rows in order (a table is drawn
      top to bottom, so row 7 cannot sit on an earlier page than row 6).

    Pages missing from an otherwise valid answer are filled with ``[]``, and
    the indexes listed for one page are sorted: the answer only decides which
    page a row is on, the order of rows is the XML order.
    Title checks (a failure only drops the title): ``has_title`` is JSON ``true``,
    non-empty text, confidence not ``"low"``, a known source, and the text is a
    substring of one preceding paragraph (source ``preceding_paragraph``) or
    of the joined first row (source ``first_row``).
    """
    if value is None:
        return None, "unparseable"
    if not isinstance(value, dict):
        return None, "not_an_object"
    raw = value.get("row_distribution")
    if not isinstance(raw, dict):
        return None, "row_distribution_not_an_object"

    distribution: dict[int, list[int]] = {}
    seen: list[int] = []
    for key, indexes in raw.items():
        page = _as_int(key)
        if page is None:
            return None, f"non_integer_page:{key!r}"
        if not table.start_page <= page <= table.end_page:
            return None, f"page_out_of_range:{page}"
        if page in distribution:
            return None, f"duplicate_page:{page}"
        if not isinstance(indexes, list):
            return None, f"page_value_not_a_list:{page}"
        converted = []
        for index in indexes:
            number = _as_int(index)
            if number is None:
                return None, f"non_integer_row:{index!r}"
            converted.append(number)
        distribution[page] = converted
        seen.extend(converted)

    if len(seen) != len(set(seen)):
        duplicates = sorted({i for i in seen if seen.count(i) > 1})
        return None, f"duplicate_rows:{duplicates}"
    expected = set(range(len(table.rows)))
    if set(seen) != expected:
        missing = sorted(expected - set(seen))
        extra = sorted(set(seen) - expected)
        return None, f"row_set_mismatch:missing={missing},extra={extra}"
    for page in table.pages:
        distribution.setdefault(page, [])
    distribution = {page: sorted(distribution[page]) for page in sorted(distribution)}
    # Rows are laid out top to bottom, so page order must equal row order.
    ordered = [index for page in sorted(distribution) for index in distribution[page]]
    if ordered != sorted(ordered):
        return None, "rows_not_in_page_order"

    answer = TableAnswer(row_pages=distribution)
    has_title = value.get("has_title") is True
    title = _text(value.get("title_text"))
    source = _text(value.get("title_source")).lower()
    confidence = _text(value.get("title_confidence")).lower()
    if confidence not in TITLE_CONFIDENCES:
        confidence = "low"
    if source not in TITLE_SOURCES:
        source = "none"
    if has_title and title and confidence != "low" and source != "none":
        needle = _norm(title)
        if source == "preceding_paragraph":
            found = any(needle in _norm(p) for p in table.preceding_paragraphs)
        else:
            found = bool(table.rows) and needle in _norm(" ".join(table.rows[0]))
        if needle and found:
            answer.title, answer.title_source, answer.title_confidence = title, source, confidence
        else:
            answer.title_rejected = True
            log.warning("table %s: title %r not found in its claimed source %r; dropped", table.table_id, title, source)
    return answer, ""


def fallback_distribution(table: MultiPageTable) -> dict[int, list[int]]:
    """Deterministic row assignment from the editor's rendered break markers.

    Each row stays on the page where the editor last rendered it (recorded
    while splitting the XML). It needs no model, is identical on every run,
    and is exact whenever the file's rendered breaks are current.
    """
    distribution: dict[int, list[int]] = {page: [] for page in table.pages}
    for index, page in enumerate(table.row_pages):
        distribution.setdefault(page, []).append(index)
    return distribution
