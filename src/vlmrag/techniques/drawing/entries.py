"""Retrieval-oriented "field entries": one record per (field, value) fact on a sheet.

Canonical fields answer "what is the drawing number?", but users also ask
"which finish does the left-hand housing get?" - a fact that lives in a table
row, not in a canonical field. Field entries flatten everything the passes
found (canonical fields, pairs, table pairs, and signal lines of the text layer)
into uniform records annotated with:

* ``field_key`` - canonical key, the term group a label mentions, or a slug of
  the label;
* ``part`` / ``sub_part`` - which component the fact belongs to (from the
  domain terms);
* ``intent_tags`` - which retrieval-critical term groups it mentions.

They are rendered into the retrieval text as one line each and exposed in the
chunk metadata (``field_entries`` and ``field_values``) for filtering.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

from .config import DomainTerms, TitleBlockSchema, normalize_label
from .parsing import as_text

__all__ = [
    "field_key_for_topic",
    "entry_tags",
    "entry_part",
    "entry_sub_part",
    "is_markup_fragment",
    "make_entry",
    "merge_entries",
    "build_field_entries",
    "entries_from_text",
    "entries_text",
    "field_values",
    "layout_columns",
    "layout_columns_text",
    "domain_synonym_text",
]

_MARKUP = re.compile(r"</?(?:br|p|table|thead|tbody|tr|td)\b|data-category=|style=", re.IGNORECASE)


def field_key_for_topic(topic: Any, schema: TitleBlockSchema, domain: DomainTerms) -> str:
    """Canonical key of a printed label; else the first term group it mentions; else a label slug."""
    norm = normalize_label(topic)
    mapped = schema.label_map().get(norm)
    if mapped:
        return mapped
    for group in domain.groups:
        if group.search(norm):
            return group.field_key or group.tag
    return norm.replace(" ", "_").replace("/", "_")[:80]


def entry_tags(topic: Any, value: Any, domain: DomainTerms) -> list[str]:
    text = f"{as_text(topic)} {as_text(value)}".lower()
    return [group.tag for group in domain.groups if group.search(text)]


def entry_part(topic: Any, value: Any, domain: DomainTerms) -> str:
    text = f"{as_text(topic)} {as_text(value)}".lower()
    for part in domain.parts:
        if any(re.search(p, text, re.IGNORECASE) for p in part.patterns):
            return part.name
    return ""


def entry_sub_part(topic: Any, domain: DomainTerms) -> str:
    if not domain.sub_part_tokens:
        return ""
    alternation = "|".join(re.escape(t) for t in domain.sub_part_tokens)
    match = re.search(rf"\b({alternation})\b", as_text(topic), re.IGNORECASE)
    return match.group(1).upper() if match else ""


def is_markup_fragment(text: Any) -> bool:
    return bool(_MARKUP.search(as_text(text)))


def make_entry(
    topic: Any,
    value: Any,
    source: str,
    *,
    schema: TitleBlockSchema,
    domain: DomainTerms,
    field_key: str = "",
    source_text: str = "",
    chunk: str = "",
    source_text_limit: int = 500,
) -> Optional[dict]:
    """One field entry, or ``None`` when there is neither a topic nor source text."""
    topic_text, value_text, source_line = as_text(topic), as_text(value), as_text(source_text)
    if not topic_text and not source_line:
        return None
    if not value_text and source_line:
        value_text = source_line
    key = as_text(field_key) or field_key_for_topic(topic_text, schema, domain)
    return {
        "topic_raw": topic_text,
        "topic_norm": normalize_label(topic_text),
        "field_key": key,
        "value": value_text,
        "value_norm": normalize_label(value_text),
        "source": as_text(source),
        "part": entry_part(topic_text, value_text, domain),
        "sub_part": entry_sub_part(topic_text, domain),
        "intent_tags": entry_tags(topic_text, value_text, domain),
        "chunk": as_text(chunk),
        "source_text": (source_line or f"{topic_text}: {value_text}".strip(": "))[:source_text_limit],
    }


def merge_entries(*groups: Optional[Iterable[Any]], limit: int = 300) -> list[dict]:
    """Concatenate entry lists, dropping repeats of (field key, topic, value) and stopping at ``limit``."""
    merged: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for group in groups:
        for entry in group or []:
            if not isinstance(entry, dict):
                continue
            topic_norm = as_text(entry.get("topic_norm") or normalize_label(entry.get("topic_raw")))
            value_norm = as_text(entry.get("value_norm") or normalize_label(entry.get("value")))
            key = (as_text(entry.get("field_key")) or topic_norm, topic_norm, value_norm)
            if not any(key) or key in seen:
                continue
            seen.add(key)
            saved = dict(entry, topic_norm=topic_norm, value_norm=value_norm)
            merged.append(saved)
            if len(merged) >= limit:
                return merged
    return merged


def build_field_entries(
    *,
    fields: Optional[dict] = None,
    pairs: Optional[Iterable[dict]] = None,
    table_pairs: Optional[Iterable[dict]] = None,
    schema: TitleBlockSchema,
    domain: DomainTerms,
    limit: int = 300,
    source_text_limit: int = 500,
) -> list[dict]:
    """Entries from title-block pairs, table pairs and canonical fields (in that order)."""
    entries: list[dict] = []
    common = dict(schema=schema, domain=domain, source_text_limit=source_text_limit)
    for source, items in (("title_block", pairs), ("title_block_table", table_pairs)):
        for item in items or []:
            if isinstance(item, dict):
                entry = make_entry(item.get("topic"), item.get("value"), source, **common)
                if entry:
                    entries.append(entry)
    for key, value in (fields or {}).items():
        value_text = as_text(value)
        if not value_text:
            continue
        entry = make_entry(key.replace("_", " ").title(), value_text, "title_block_field", field_key=key, **common)
        if entry:
            entries.append(entry)
    return merge_entries(entries, limit=limit)


def entries_from_text(
    text: Any,
    source: str,
    *,
    schema: TitleBlockSchema,
    domain: DomainTerms,
    chunk: str = "",
    limit: int = 300,
    source_text_limit: int = 500,
) -> list[dict]:
    """Entries for text-layer lines that mention a retrieval-critical term.

    Notes such as ``VENT HOLE DIA 3.2 THRU, 6 PLACES`` never appear in a title
    block but are exactly what users search for; this lifts them into the same
    structured form. A line is split into topic and value at the first
    ``:``/``=`` after the term; otherwise at the first colon; otherwise the whole
    line is the topic and the value.
    """
    signal = domain.signal_regex()
    if signal is None:
        return []
    terms = "|".join(f"(?:{p})" for g in domain.groups for p in g.patterns)
    # Named groups: configured term patterns may contain capturing groups of their own.
    splitter = re.compile(
        rf"(?P<topic>.{{1,140}}?(?:{terms}).{{0,80}}?)(?:\s*[:=]\s*|\s{{2,}}|\t+)(?P<value>.+)$", re.IGNORECASE
    )
    entries: list[dict] = []
    for raw in as_text(text).splitlines():
        line = re.sub(r"\s+", " ", raw.strip())
        if not line or not signal.search(line) or is_markup_fragment(line):
            continue
        topic, value = line[:120], ""
        match = splitter.match(line)
        if match:
            topic, value = match.group("topic").strip(), match.group("value").strip()
        elif ":" in line:
            left, right = line.split(":", 1)
            topic, value = left.strip(), right.strip()
        entry = make_entry(
            topic, value, source, schema=schema, domain=domain, source_text=line, chunk=chunk,
            source_text_limit=source_text_limit,
        )
        if entry:
            entries.append(entry)
        if len(entries) >= limit:
            break
    return merge_entries(entries, limit=limit)


def entries_text(entries: Optional[Iterable[dict]]) -> str:
    """``- Topic: value (field=... / part=... / sub_part=... / tags=...)`` per entry."""
    lines = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        topic, value = as_text(entry.get("topic_raw")), as_text(entry.get("value"))
        tags = ", ".join(entry.get("intent_tags") or [])
        meta = [
            f"field={entry['field_key']}" if entry.get("field_key") else "",
            f"part={entry['part']}" if entry.get("part") else "",
            f"sub_part={entry['sub_part']}" if entry.get("sub_part") else "",
            f"tags={tags}" if tags else "",
        ]
        suffix = " (" + " / ".join(m for m in meta if m) + ")" if any(meta) else ""
        if topic and value:
            lines.append(f"- {topic}: {value}{suffix}")
        elif value:
            lines.append(f"- {value}{suffix}")
    return "\n".join(lines)


def field_values(entries: Optional[Iterable[dict]]) -> dict[str, list[str]]:
    """``{field_key: [distinct values]}`` - the filter-friendly view of the entries."""
    values: dict[str, list[str]] = {}
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        key = as_text(entry.get("field_key") or entry.get("topic_norm"))
        value = as_text(entry.get("value"))
        if key and value:
            bucket = values.setdefault(key, [])
            if value not in bucket:
                bucket.append(value)
    return values


def layout_columns(pairs: Optional[Iterable[dict]]) -> list[str]:
    """Distinct printed topics that carry a value, in reading order."""
    columns: list[str] = []
    for item in pairs or []:
        if not isinstance(item, dict):
            continue
        topic, value = as_text(item.get("topic")), as_text(item.get("value"))
        if normalize_label(topic) and value and topic not in columns:
            columns.append(topic)
    return columns


def layout_columns_text(columns: Optional[Iterable[str]]) -> str:
    """One line listing the title-block labels, so label-only queries ("sheets with a mass cell") match."""
    names = [as_text(c) for c in columns or [] if as_text(c)]
    return "Title block layout columns: " + " | ".join(names) if names else ""


def domain_synonym_text(text: Any, domain: DomainTerms) -> str:
    """Synonym lines of every term group mentioned in ``text``.

    Appending ``Domain synonym: Vent hole = Venthole = Breather hole`` lets a
    query that uses one spelling reach a sheet that prints another, without
    rewriting the source text itself.
    """
    source = as_text(text)
    lines = [group.synonym_line() for group in domain.groups if group.search(source)]
    return "\n".join(line for line in lines if line)
