"""Prompts and output schemas of the three model passes.

* ``drawing.vision`` - all page tiles plus focus crops -> title-block fields and
  the visible topic-value pairs;
* ``drawing.table`` - metadata-table crops -> visually grouped table rows plus
  pairs and fields;
* ``drawing.normalize`` - text only: maps the raw pairs of both passes onto the
  canonical fields.

Each prompt is assembled from the title-block schema (field names, normalisation
rules) and the domain terms (retrieval-critical vocabulary), so adapting the
technique to another organisation means editing two JSON files, not prompts.
The prompts still embed an example object: servers that ignore structured
output (many OpenAI-compatible ones) need the shape in the text. Bump the
``*_PROMPT_VERSION`` constants whenever a template changes: they are part of
the cache and replay keys.
"""

from __future__ import annotations

import json
from typing import Any

from .config import DomainTerms, DrawingConfig, TitleBlockSchema

__all__ = [
    "VISION_TASK",
    "TABLE_TASK",
    "NORMALIZE_TASK",
    "VISION_PROMPT_VERSION",
    "TABLE_PROMPT_VERSION",
    "NORMALIZE_PROMPT_VERSION",
    "PROMPT_VERSIONS",
    "build_vision_prompt",
    "build_table_prompt",
    "build_normalize_prompt",
    "fields_json_schema",
    "table_json_schema",
    "fields_example",
    "table_example",
    "NORMALIZE_PAYLOAD_HEADER",
    "NORMALIZE_PAIRS_HEADER",
]

VISION_TASK = "drawing.vision"
TABLE_TASK = "drawing.table"
NORMALIZE_TASK = "drawing.normalize"
VISION_PROMPT_VERSION = "drawing.vision/1"
TABLE_PROMPT_VERSION = "drawing.table/1"
NORMALIZE_PROMPT_VERSION = "drawing.normalize/1"
# Keyed by task name, like ``Chunk.provenance["prompt_versions"]`` of every technique.
PROMPT_VERSIONS = {
    VISION_TASK: VISION_PROMPT_VERSION,
    TABLE_TASK: TABLE_PROMPT_VERSION,
    NORMALIZE_TASK: NORMALIZE_PROMPT_VERSION,
}

NORMALIZE_PAYLOAD_HEADER = "Raw vision JSON/object:"
NORMALIZE_PAIRS_HEADER = "Observed raw pairs text:"


# --------------------------------------------------------------------------- output contracts


def _string_props(schema: TitleBlockSchema) -> dict[str, Any]:
    return {key: {"type": "string"} for key in schema.keys()}


def _pairs_schema() -> dict:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"topic": {"type": "string"}, "value": {"type": "string"}},
            "required": ["topic", "value"],
            "additionalProperties": False,
        },
    }


def fields_json_schema(schema: TitleBlockSchema) -> dict:
    """Strict-mode schema of the vision and normalize answers: ``pairs`` plus one string per field."""
    properties = {"pairs": _pairs_schema(), **_string_props(schema)}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def table_json_schema(schema: TitleBlockSchema) -> dict:
    """Strict-mode schema of the table answer: ``table_rows`` (row/col indexes + cell text) first."""
    rows = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "row_index": {"type": "integer"},
                "cells": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"col_index": {"type": "integer"}, "text": {"type": "string"}},
                        "required": ["col_index", "text"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["row_index", "cells"],
            "additionalProperties": False,
        },
    }
    properties = {"table_rows": rows, "pairs": _pairs_schema(), **_string_props(schema)}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def fields_example(schema: TitleBlockSchema) -> str:
    example: dict[str, Any] = {"pairs": [{"topic": "", "value": ""}]}
    example.update({key: "" for key in schema.keys()})
    return _compact(example)


def table_example(schema: TitleBlockSchema) -> str:
    example: dict[str, Any] = {
        "table_rows": [{"row_index": 0, "cells": [{"col_index": 0, "text": ""}]}],
        "pairs": [{"topic": "", "value": ""}],
    }
    example.update({key: "" for key in schema.keys()})
    return _compact(example)


# --------------------------------------------------------------------------- prompt text


def _join_terms(terms: list[str]) -> str:
    if not terms:
        return ""
    if len(terms) == 1:
        return terms[0]
    return ", ".join(terms[:-1]) + " or " + terms[-1]


def _header_example(domain: DomainTerms) -> str:
    word = domain.repeated_header_example or (domain.groups[0].terms[0] if domain.groups and domain.groups[0].terms else "")
    return f'"{word} {word}"' if word else ""


_POSITION_LABELS = (
    "A position label precedes every image: it names the crop and gives its normalized page box "
    "(x grows to the right and y grows downward, both from 0 to 1) and its render resolution."
)
_WEAK_INFERENCE = (
    "Keep domain inference weak: engineering-drawing vocabulary may help you understand labels and map "
    "them to fields, but never invent values, never guess text you cannot read, and never join uncertain "
    "fragments without visual support."
)
_JSON_ONLY = "Return exactly one JSON object and nothing else."
_NO_PROSE = "Do not add explanations or markdown."


def build_vision_prompt(cfg: DrawingConfig) -> str:
    """Prompt of the vision pass (tiles + focus crops)."""
    schema, domain = cfg.schema, cfg.domain
    rows, cols = cfg.page_tile_rows, cfg.page_tile_cols
    tiles = rows * cols
    critical = domain.critical_terms()
    sentences = [
        "You are reading the title block of one engineering drawing sheet.",
        f"The input contains {tiles} overlapping page tiles of that sheet: the full page was cut into a grid of "
        f"{cols} columns by {rows} rows, and neighbouring tiles overlap slightly.",
        "Title blocks usually sit in the lower-right area, but inspect every tile, because a field can "
        "straddle a tile boundary.",
        "Close-up crops of the lower-right title block and of the small bottom-right table may follow the "
        "tiles. When they are present, rely on them first for tiny text and table cells.",
        _POSITION_LABELS,
        "Treat the images as evidence for title-block metadata, not for the drawing geometry.",
        _WEAK_INFERENCE,
        "Leave a field empty when its text is not clearly legible.",
        "Important: whenever any label or value is legible in any image, list those visible topic-value "
        "pairs in the pairs array, even if some canonical top-level fields stay empty.",
        "Never answer with an all-empty object while legible metadata is visible in the crops.",
        "Inside ruled table areas, group text by the visible borders, row alignment and cell grouping, and "
        "never move a value into a neighbouring cell that a border or the alignment separates from it.",
        f"Fields commonly found include {', '.join(schema.keys())}, and other title-block metadata.",
    ]
    if critical:
        header = _header_example(domain)
        sentences += [
            f"For downstream retrieval, also keep every legible topic-value pair or table row that mentions "
            f"{_join_terms(critical)}.",
            "Keep those retrieval-critical pairs separate per local row, cell or part instead of folding "
            "several visible rows into one representative value.",
        ]
        if header:
            sentences.append(
                f"Do not turn a repeated stand-alone header such as {header} into a pair unless a legible row "
                "value, cell value or part label sits next to it."
            )
    sentences += [
        "When the same field is visible in several tiles, prefer the most complete value that the images "
        "support.",
        "You may also receive noisy text from the PDF text layer of the same regions, headed by the same "
        "position labels as the images. Use it only as a secondary hint where it matches the visible image "
        "evidence.",
        _JSON_ONLY,
        f"Use exactly this schema: {fields_example(schema)}.",
        "The pairs array must contain the visible topic-value pairs you can actually read in the title block.",
        "Keep each topic close to its printed label.",
        _NO_PROSE,
    ]
    return " ".join(sentences)


def build_table_prompt(cfg: DrawingConfig) -> str:
    """Prompt of the table pass (metadata-table crops)."""
    schema, domain = cfg.schema, cfg.domain
    critical = domain.critical_terms()
    sentences = [
        "You are reading the ruled metadata table in the lower-right title-block area of an engineering "
        "drawing sheet.",
        "The images focus on the table region, where printed topic labels and values are arranged in cells.",
        _POSITION_LABELS,
        "Group text by the visible table borders, row alignment and column alignment.",
        "Never mix text from neighbouring cells when a border, a gap or the alignment separates them.",
        "Return both the visually grouped table rows and the topic-value pairs derived from those rows.",
        "Keep domain inference weak and never invent unreadable text.",
        "If part of a cell is unclear, keep only the legible part.",
    ]
    if critical:
        sentences += [
            f"Rows or cells that mention {_join_terms(critical)} are retrieval-critical: keep them in "
            "table_rows and create a separate pair for every legible row or cell group.",
            "Do not fold several such rows into one value, and do not output a repeated stand-alone header as "
            "a value when no row or cell value belonging to it is legible.",
        ]
    sentences += [
        _JSON_ONLY,
        f"Use exactly this schema: {table_example(schema)}.",
        "In table_rows, keep the visual row order from top to bottom.",
        "In pairs, keep each topic close to its printed label.",
        _NO_PROSE,
    ]
    return " ".join(sentences)


def build_normalize_prompt(cfg: DrawingConfig) -> str:
    """Prompt of the text-only normalization pass."""
    schema, domain = cfg.schema, cfg.domain
    critical = domain.critical_terms()
    sentences = [
        "You are normalizing raw topic-value pairs read from the title block of an engineering drawing sheet.",
        "Topic labels differ a lot between files: synonyms, abbreviations, alternative wording and local "
        "conventions are common.",
        "Choose the best canonical field from the meaning of the label and the shape of the value, not by "
        "exact string matching.",
        "Use only evidence contained in the raw vision output, the raw pairs text and the optional structured "
        "table extraction.",
        "Never invent values; leave a field empty when nothing supports it.",
        "Treat the structured table extraction as high-priority evidence for which topic belongs to which "
        "value in ruled table areas.",
    ]
    if critical:
        sentences.append(
            f"Never drop raw pairs that mention {_join_terms(critical)} just because they do not map to a "
            "canonical top-level field; keep them in the pairs array with stable, retrieval-friendly labels."
        )
    examples = [f"{short} as {meaning}" for short, meaning in domain.abbreviations]
    if domain.sub_part_tokens:
        examples.append(f"{_join_terms(list(domain.sub_part_tokens)).replace(' or ', ' and ')} as sub-part tokens")
    if examples:
        sentences.append(
            "Expand abbreviations only when the source text supports it, for example " + ", ".join(examples) + "."
        )
    if domain.parts and domain.groups and domain.groups[0].terms:
        broad = f"{domain.parts[0].name} {domain.groups[0].terms[0].lower()}"
        sentences.append(
            f"If several rows describe the same broad field, such as {broad}, keep separate row-level pairs "
            "instead of returning one representative value."
        )
    sentences += list(schema.normalize_rules)
    sentences += [
        _JSON_ONLY,
        f"Use exactly this schema: {fields_example(schema)}.",
        "The pairs array should hold the normalized topic-value pairs that are most useful for downstream "
        "retrieval.",
        "Use stable, readable topic labels that follow the canonical meaning where possible.",
        _NO_PROSE,
    ]
    return " ".join(sentences)
