"""Turn model answers into title-block fields, pairs and table rows.

The parser is deliberately forgiving because it sits between a model and an
acceptance gate:

* JSON is recovered with :mod:`vlmrag.vlm.jsonparse` (code fences, prose
  around the object, servers that ignore the schema);
* when no JSON object can be recovered, a line-based fallback reads
  ``Label: value`` lines and multi-line ``Label`` / value blocks, so a model
  that answered in plain text still yields usable pairs;
* every pair is kept even when its label maps to no canonical field - those
  "unmapped" pairs are often the most specific retrieval facts on the sheet.

Values are compared and de-duplicated on :func:`normalize_label` forms, and the
longest value wins when the same label is read twice (overlapping tiles
usually cut one reading short).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from ...vlm.jsonparse import extract_json_object, iter_json_candidates, strip_code_fences
from .config import TitleBlockSchema, normalize_label

__all__ = [
    "ParsedTitleBlock",
    "as_text",
    "dedupe_pairs",
    "merge_pairs",
    "first_json_object_text",
    "parse_title_block",
    "parse_table_rows",
    "table_rows_text",
    "table_rows_matrix",
    "pairs_text",
    "clean_field_value",
]

_WS = re.compile(r"\s+")
_ROW_SPLIT = re.compile(r"\s*\|\s*|\n+")
_TRAILING_GRID_LETTER = re.compile(r"\s+[A-Z]$")
_MAX_TABLE_COLUMNS = 64


def _collapse(text: str) -> str:
    return _WS.sub(" ", text).strip()


def as_text(value: Any) -> str:
    """Model values as clean strings (``None`` -> ``""``, lists joined with ``" | "``)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " | ".join(t for t in (as_text(v) for v in value) if t)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _pair(topic: Any, value: Any) -> dict[str, str]:
    return {"topic": as_text(topic), "value": as_text(value)}


def dedupe_pairs(items: Optional[Iterable[Any]]) -> list[dict[str, str]]:
    """One pair per normalised topic; the longer value replaces a shorter one in place.

    Overlapping tiles and focus crops make the model read the same cell several
    times, usually once truncated; keeping the longest reading recovers it.
    """
    out: list[dict[str, str]] = []
    index: dict[str, int] = {}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        pair = _pair(item.get("topic"), item.get("value"))
        norm = normalize_label(pair["topic"])
        if not norm:
            continue
        if norm not in index:
            index[norm] = len(out)
            out.append(pair)
        elif len(pair["value"]) > len(out[index[norm]]["value"]):
            out[index[norm]] = pair
    return out


def merge_pairs(primary: Optional[Iterable[Any]], secondary: Optional[Iterable[Any]]) -> list[dict[str, str]]:
    """``primary`` then ``secondary``; drops empty topics/values and exact (topic, value) repeats.

    Unlike :func:`dedupe_pairs` this keeps several values per topic: two table
    rows labelled "Finish" with different parts are two facts, not one.
    """
    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for source in (primary or [], secondary or []):
        for item in source:
            if not isinstance(item, dict):
                continue
            pair = _pair(item.get("topic"), item.get("value"))
            if not pair["topic"] or not pair["value"]:
                continue
            key = (normalize_label(pair["topic"]), pair["value"])
            if key in seen:
                continue
            seen.add(key)
            merged.append(pair)
    return merged


def first_json_object_text(text: Optional[str]) -> str:
    """The first balanced ``{...}`` in ``text`` (even if it does not parse), else the stripped text.

    Used to forward the vision answers to the normalization pass without the
    prose or fences around them.
    """
    stripped = (text or "").strip()
    for candidate in iter_json_candidates(stripped, openers="{"):
        return candidate
    return stripped


@dataclass
class ParsedTitleBlock:
    """Fields (every schema key, possibly empty) and pairs read from one answer."""

    fields: dict[str, str]
    pairs: list[dict[str, str]] = field(default_factory=list)
    from_json: bool = False

    def has_content(self) -> bool:
        return bool(self.pairs) or any(v for v in self.fields.values())


def parse_title_block(text: Optional[str], schema: TitleBlockSchema) -> ParsedTitleBlock:
    """Parse a vision / table / normalize answer into fields and pairs.

    JSON answers: every schema key is read directly; each pair whose topic maps
    to a field may extend that field (longer value wins); string-valued keys
    outside the schema become pairs. Non-JSON answers go through the line
    fallback described in the module docstring.
    """
    keys = schema.keys()
    label_map = schema.label_map()
    fields = {key: "" for key in keys}
    pairs: list[dict[str, str]] = []

    def add_pair(topic: Any, value: Any) -> None:
        topic_text, value_text = as_text(topic), as_text(value)
        if topic_text:
            pairs.append({"topic": topic_text, "value": value_text})

    def extend_field(key: str, value: str) -> None:
        value = as_text(value)
        if value and len(value) > len(fields.get(key, "")):
            fields[key] = value

    data = extract_json_object(text) if text else None
    if isinstance(data, dict):
        for key in keys:
            fields[key] = as_text(data.get(key))
        raw_pairs = data.get("pairs")
        if isinstance(raw_pairs, list):
            for item in raw_pairs:
                if isinstance(item, dict):
                    add_pair(item.get("topic"), item.get("value"))
                    mapped = label_map.get(normalize_label(item.get("topic")))
                    if mapped:
                        extend_field(mapped, item.get("value"))
        for raw_key, raw_value in data.items():
            if raw_key in fields or raw_key == "pairs" or not isinstance(raw_value, str):
                continue
            add_pair(raw_key, raw_value)
            mapped = label_map.get(normalize_label(raw_key))
            if mapped:
                extend_field(mapped, raw_value)
        return ParsedTitleBlock(fields=fields, pairs=dedupe_pairs(pairs), from_json=True)

    body = strip_code_fences(text or "")
    # Pass 1: "Label: value" on one line.
    for line in body.splitlines():
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        add_pair(label.strip(), value.strip())
        mapped = label_map.get(normalize_label(label))
        if mapped:
            extend_field(mapped, value)

    # Pass 2: a known label on its own line, value on the following line(s) until the next label.
    lines = [_collapse(line) for line in body.splitlines()]
    lines = [line for line in lines if line]

    def match_label(line: str) -> tuple[str, str, str]:
        if ":" in line:
            left, right = line.split(":", 1)
            mapped = label_map.get(normalize_label(left))
            if mapped:
                return left.strip(), mapped, right.strip()
        mapped = label_map.get(normalize_label(line))
        if mapped:
            return line.strip(), mapped, ""
        return "", "", ""

    i = 0
    while i < len(lines):
        label_text, mapped, inline = match_label(lines[i])
        if not mapped:
            i += 1
            continue
        values = [inline] if inline else []
        j = i + 1
        while j < len(lines) and not match_label(lines[j])[1]:
            values.append(lines[j])
            j += 1
        spec = schema.get(mapped)
        joined = (spec.join if spec else " ").join(v for v in values if v)
        display = label_text if label_text.endswith(":") else f"{label_text}:"
        add_pair(display, joined)
        extend_field(mapped, joined)
        i = j
    return ParsedTitleBlock(fields=fields, pairs=dedupe_pairs(pairs), from_json=False)


def parse_table_rows(text: Optional[str]) -> list[dict]:
    """``table_rows`` of a table answer: ``[{"row_index": int, "cells": [{"col_index": int, "text": str}]}]``.

    Missing or non-integer indexes default to the position in the list; empty
    cells and rows without cells are dropped. Anything unparseable yields ``[]``.
    """
    data = extract_json_object(text) if text else None
    if not isinstance(data, dict) or not isinstance(data.get("table_rows"), list):
        return []
    rows: list[dict] = []
    for raw_row in data["table_rows"]:
        if not isinstance(raw_row, dict):
            continue
        row_index = _as_index(raw_row.get("row_index"), len(rows))
        cells = []
        raw_cells = raw_row.get("cells")
        for raw_cell in raw_cells if isinstance(raw_cells, list) else []:
            if not isinstance(raw_cell, dict):
                continue
            col_index = _as_index(raw_cell.get("col_index"), len(cells))
            cell_text = as_text(raw_cell.get("text"))
            if cell_text:
                cells.append({"col_index": col_index, "text": cell_text})
        if cells:
            rows.append({"row_index": row_index, "cells": cells})
    return rows


def _as_index(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def table_rows_text(rows: Optional[Iterable[dict]]) -> str:
    """``ROW 3: A || Initial release || 2025-11-02`` - one line per table row (retrieval text)."""
    lines = []
    for row in rows or []:
        cells = [as_text(c.get("text")) for c in (row or {}).get("cells", []) or [] if isinstance(c, dict)]
        cells = [c for c in cells if c]
        if cells:
            lines.append(f"ROW {(row or {}).get('row_index', '')}: " + " || ".join(cells))
    return "\n".join(lines)


def table_rows_matrix(rows: Optional[Iterable[dict]]) -> list[list[str]]:
    """Rows as a rectangular matrix (cells placed by ``col_index``) for Markdown rendering."""
    matrix: list[list[str]] = []
    for row in rows or []:
        cells: dict[int, str] = {}
        for cell in (row or {}).get("cells", []) or []:
            col = min(max(0, _as_index(cell.get("col_index"), len(cells))), _MAX_TABLE_COLUMNS - 1)
            text = as_text(cell.get("text"))
            cells[col] = f"{cells[col]} {text}".strip() if col in cells else text
        if cells:
            matrix.append([cells.get(c, "") for c in range(max(cells) + 1)])
    width = max((len(r) for r in matrix), default=0)
    return [r + [""] * (width - len(r)) for r in matrix]


def pairs_text(pairs: Optional[Iterable[dict]]) -> str:
    """``Topic: value`` lines (a topic that already ends with ``:`` is not given a second one)."""
    lines = []
    for item in pairs or []:
        if not isinstance(item, dict):
            continue
        topic, value = as_text(item.get("topic")), as_text(item.get("value"))
        if not topic:
            continue
        if topic.endswith(":"):
            lines.append(f"{topic} {value}".rstrip() if value else topic)
        else:
            lines.append(f"{topic}: {value}" if value else f"{topic}:")
    return "\n".join(lines)


def _strip_leading_labels(segment: str, patterns: list[str]) -> str:
    """Remove labels that start ``segment`` ("Surface finish MATTE" -> "MATTE").

    Repeated until nothing changes, so cleaning a cleaned value is a no-op.
    """
    previous = None
    while segment != previous:
        previous = segment
        for pattern in patterns:
            segment = re.sub(rf"(?i)^{pattern}(?!\w)\s*:?\s*", "", segment)
        segment = segment.strip(" :|-")
    return segment


def clean_field_value(key: str, value: Any, schema: TitleBlockSchema) -> str:
    """Whitespace-normalise a field value; table fields also lose echoed labels and grid letters.

    Small table cells are where models most often copy a neighbour's label
    ("Material: PC Mass") or a border zone letter into a value. For table
    fields the labels of the *other* table fields are removed wherever they
    occur, the field's *own* label only where it acts as a label (followed by
    a colon, or at the start of a row segment): table labels such as "Finish"
    or "Material" are ordinary words inside values ("MILL FINISH"), so
    removing them everywhere would cut real content. The value is then split
    into row segments, stray single capital letters (border zone markers) are
    dropped and repeated segments are removed; ``row_field`` values keep their
    segments apart with ``" | "``. When no segment survives (for example a
    value that is a single letter), the label-stripped value is returned with
    only whitespace and trailing-letter cleanup.
    """
    text = as_text(value)
    if not text:
        return ""
    text = text.replace("\r", "\n")
    spec = schema.get(key)
    if spec is None or not spec.table_field:
        return _collapse(text)
    own = schema.field_label_patterns(key)
    for pattern in schema.table_label_patterns():
        suffix = r"\s*:" if pattern in own else r"\s*:?"
        text = re.sub(rf"(?i)(?<!\w){pattern}(?!\w){suffix}", " ", text)
    aliases = schema.table_label_aliases()
    segments: list[str] = []
    seen: set[str] = set()
    for raw in _ROW_SPLIT.split(text):
        segment = _strip_leading_labels(_collapse(raw).strip(" :|-"), own)
        segment = _TRAILING_GRID_LETTER.sub("", segment).strip()
        norm = normalize_label(segment)
        if not norm or norm in aliases or re.fullmatch(r"[A-Z]", segment) or norm in seen:
            continue
        seen.add(norm)
        segments.append(segment)
    if not segments:
        fallback = _strip_leading_labels(_collapse(text).strip(" :|-"), own)
        return _TRAILING_GRID_LETTER.sub("", fallback).strip()
    if spec.row_field:
        return " | ".join(segments).strip(" |")
    return " ".join(segments).strip()
