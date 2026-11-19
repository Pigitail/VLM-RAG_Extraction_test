"""Request structure: prompt order, labels, schemas, limits and replay-key hygiene."""

from __future__ import annotations

import json

import pytest

from vlmrag.techniques.docx import pipeline, prompts
from vlmrag.techniques.docx.xmlsplit import MultiPageTable
from vlmrag.vlm.types import ImageInput


def _assert_strict(schema: dict) -> None:
    """Strict structured output: every object lists all properties as required and forbids extras."""
    if schema.get("type") == "object":
        assert schema["additionalProperties"] is False
        assert sorted(schema["required"]) == sorted(schema["properties"])
        for child in schema["properties"].values():
            _assert_strict(child)
    elif schema.get("type") == "array":
        _assert_strict(schema["items"])


@pytest.fixture
def table() -> MultiPageTable:
    return MultiPageTable("t1", 3, 5, [["Item", "Qty"], ["a", "1"]], [3, 5], ["Table 1. Stock"])


def test_image_request_structure():
    request = pipeline.build_image_request(b"png", 2, 4, doc_id="abc", source_name="spec.docx")
    assert request.task == "docx.image_description" == prompts.IMAGE_TASK
    assert request.prompt_version == "docx.image_description/1"
    assert request.parts[0] == prompts.IMAGE_PROMPT
    assert request.parts[1] == "Page context: page 2 of 4."
    assert isinstance(request.parts[2], ImageInput) and request.parts[2].png == b"png"
    assert (request.page, request.source_name, request.doc_id) == (2, "spec.docx", "abc")
    assert request.max_output_tokens == 60000 and request.timeout_s == 600
    assert request.reasoning_effort == "low"
    assert request.json_schema is prompts.IMAGE_SCHEMA
    _assert_strict(request.json_schema)


def test_table_request_structure(table):
    request = pipeline.build_table_request(table, [(3, b"a"), (4, b"b")], 6, source_name="spec.docx")
    assert request.task == "docx.table_distribution"
    assert request.prompt_version == "docx.table_distribution/1"
    assert request.parts[0] == prompts.TABLE_PROMPT
    assert request.parts[1] == "Document context: table spans pages 3-5 of 6 total."
    header, payload = request.parts[2].split("\n", 1)
    assert "ground truth" in header and "do NOT re-OCR" in header
    data = json.loads(payload)
    assert data["rows"] == [{"index": 0, "cells": ["Item", "Qty"]}, {"index": 1, "cells": ["a", "1"]}]
    assert data["preceding_paragraphs"] == ["Table 1. Stock"] and data["row_count"] == 2
    assert [p.label for p in request.images()] == ["PDF page 3 image follows:", "PDF page 4 image follows:"]
    assert request.page == 3 and request.reasoning_effort == "medium"
    schema = request.json_schema
    _assert_strict(schema)
    assert schema["properties"]["row_distribution"]["required"] == ["3", "4", "5"]
    assert schema["properties"]["title_source"]["enum"] == ["preceding_paragraph", "first_row", "none"]


def test_reasoning_effort_can_be_disabled_by_settings():
    class NoEffort:
        reasoning_effort = None

    class Default:
        reasoning_effort = "low"

    assert pipeline._effort(NoEffort(), "medium") is None
    assert pipeline._effort(Default(), "medium") == "medium"


def test_very_wide_tables_are_sent_without_a_schema():
    assert prompts.table_schema(range(1, 200)) is None
    assert prompts.table_schema([]) is None


def test_prompts_keep_their_contract():
    assert '"image_descriptions":[]' in prompts.IMAGE_PROMPT.replace(" ", "")
    assert "do NOT transcribe or re-OCR" in prompts.IMAGE_PROMPT
    for needle in ("EXACTLY ONCE", "empty list []", "Never invent a title", '"row_distribution": {}'):
        assert needle in prompts.TABLE_PROMPT


def test_no_paths_leak_into_text_parts(tmp_path, table):
    request = pipeline.build_table_request(table, [(3, b"a")], 5, source_name="spec.docx")
    for part in request.text_parts() + [img.label for img in request.images()]:
        assert str(tmp_path) not in part and ":\\" not in part and "/tmp" not in part
