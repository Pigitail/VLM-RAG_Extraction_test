"""Prompt contents, versions and strict-mode output schemas."""

from __future__ import annotations

import json
import re

from vlmrag.techniques.drawing import DrawingConfig
from vlmrag.techniques.drawing.config import DomainTerms
from vlmrag.techniques.drawing.prompts import (
    PROMPT_VERSIONS,
    build_normalize_prompt,
    build_table_prompt,
    build_vision_prompt,
    fields_example,
    fields_json_schema,
    table_example,
    table_json_schema,
)


def _assert_strict(node, path="$"):
    if isinstance(node, dict):
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False, path
            assert sorted(node.get("required", [])) == sorted(node["properties"]), path
        for key, value in node.items():
            _assert_strict(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _assert_strict(value, f"{path}[{i}]")


def test_schemas_are_strict_mode_compatible():
    schema = DrawingConfig().schema
    for js in (fields_json_schema(schema), table_json_schema(schema)):
        _assert_strict(js)
        json.dumps(js)
    assert list(table_json_schema(schema)["properties"])[:2] == ["table_rows", "pairs"]
    assert set(fields_json_schema(schema)["properties"]) == {"pairs", *schema.keys()}


def test_examples_match_schemas():
    schema = DrawingConfig().schema
    assert set(json.loads(fields_example(schema))) == set(fields_json_schema(schema)["properties"])
    assert set(json.loads(table_example(schema))) == set(table_json_schema(schema)["properties"])


def test_prompt_versions():
    assert PROMPT_VERSIONS == {
        "drawing.vision": "drawing.vision/1",
        "drawing.table": "drawing.table/1",
        "drawing.normalize": "drawing.normalize/1",
    }
    for task, version in PROMPT_VERSIONS.items():  # "<task>/<n>", the naming every technique uses
        assert re.fullmatch(rf"{re.escape(task)}/\d+", version)


def test_vision_prompt_contents():
    cfg = DrawingConfig()
    prompt = build_vision_prompt(cfg)
    assert "12 overlapping page tiles" in prompt and "4 columns by 3 rows" in prompt
    assert "position label" in prompt and "secondary hint" in prompt
    assert "Vent hole" in prompt and "retrieval-critical" in prompt and '"Finish Finish"' in prompt
    assert fields_example(cfg.schema) in prompt
    for key in cfg.schema.keys():
        assert key in prompt
    other = build_vision_prompt(cfg.replace(page_tile_rows=2, page_tile_cols=2))
    assert "4 overlapping page tiles" in other and "2 columns by 2 rows" in other


def test_prompts_without_domain_terms_skip_domain_sentences():
    cfg = DrawingConfig(domain=DomainTerms.empty())
    for build in (build_vision_prompt, build_table_prompt, build_normalize_prompt):
        prompt = build(cfg)
        assert "retrieval-critical" not in prompt and "Vent hole" not in prompt
        assert "Return exactly one JSON object" in prompt


def test_table_and_normalize_prompts():
    cfg = DrawingConfig()
    table = build_table_prompt(cfg)
    assert table_example(cfg.schema) in table and "top to bottom" in table and "borders" in table
    normalize = build_normalize_prompt(cfg)
    assert "HSG as housing" in normalize and "LH, RH, FR and RR as sub-part tokens" in normalize
    for rule in cfg.schema.normalize_rules:
        assert rule in normalize
    assert "housing surface finish" in normalize


def test_prompts_contain_no_person_fields():
    cfg = DrawingConfig()
    text = " ".join(build(cfg) for build in (build_vision_prompt, build_table_prompt, build_normalize_prompt)).lower()
    for word in ("signature", "designer", "approved by", "drawn by", "checked by"):
        assert word not in text
