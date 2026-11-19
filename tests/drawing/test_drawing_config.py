"""Defaults, packaged JSON, environment overrides and schema/domain validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlmrag.techniques.drawing.config import (
    DomainTerms,
    DrawingConfig,
    FieldSpec,
    TitleBlockSchema,
    normalize_label,
)

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "src" / "vlmrag" / "techniques" / "drawing"
PERSON_WORDS = ("designer", "drawn", "checked", "approved", "approver", "signature", "author", "engineer", "developer", "creator")


def test_defaults_match_the_tuned_values():
    cfg = DrawingConfig()
    assert (cfg.page_dpi, cfg.page_tile_rows, cfg.page_tile_cols) == (400, 3, 4)
    assert (cfg.page_tile_overlap, cfg.page_tile_upscale) == (0.12, 2.0)
    assert (cfg.title_focus_x_start, cfg.title_focus_y_start, cfg.title_focus_dpi, cfg.title_focus_upscale) == (0.68, 0.52, 500, 3.0)
    assert (cfg.table_focus_x_start, cfg.table_focus_y_start, cfg.table_focus_dpi, cfg.table_focus_upscale) == (0.82, 0.70, 650, 4.0)
    assert (cfg.metadata_x_start, cfg.metadata_y_start, cfg.metadata_dpi, cfg.metadata_upscale) == (0.44, 0.62, 650, 4.0)
    assert (cfg.metadata_tile_rows, cfg.metadata_tile_cols, cfg.metadata_tile_overlap) == (2, 3, 0.10)
    assert (cfg.timeout_s, cfg.max_output_tokens, cfg.empty_output_retries) == (600.0, 60000, 2)
    assert (cfg.min_pairs, cfg.field_entry_limit, cfg.field_source_text_limit) == (3, 300, 500)
    assert cfg.title_block_pages == "first"


def test_example_configs_equal_packaged_defaults():
    for example, packaged in (
        ("title_block_schema.example.json", "title_block_schema.json"),
        ("domain_terms.example.json", "domain_terms.json"),
    ):
        a = json.loads((REPO / "configs" / example).read_text(encoding="utf-8"))
        b = json.loads((PACKAGE / packaged).read_text(encoding="utf-8"))
        assert a == b, f"configs/{example} drifted from the packaged default"


def test_default_schema_has_no_person_fields():
    schema = TitleBlockSchema.default()
    words = " ".join(f"{f.key} {f.label} {' '.join(f.aliases)}" for f in schema.fields).lower()
    for word in PERSON_WORDS:
        assert word not in words
    assert {"drawing_number", "title", "revision", "date_of_issue", "sheet", "company"} <= set(schema.keys())


def test_label_map_normalises_printed_variants():
    label_map = TitleBlockSchema.default().label_map()
    assert label_map[normalize_label("DWG. NO.:")] == "drawing_number"
    assert label_map[normalize_label("P / N")] == "part_number"
    assert label_map[normalize_label("p/n")] == "part_number"
    assert label_map[normalize_label("Surface_Finish")] == "surface_finish"
    assert label_map[normalize_label("Drawing No.")] == "drawing_number"


def test_normalize_label():
    assert normalize_label("  Part_No / Rev:  ") == "part no / rev"
    assert normalize_label("P/N") == "p / n"
    assert normalize_label("Mat’l") == "mat'l"
    assert normalize_label(None) == ""


def test_label_map_first_field_wins_on_clash():
    schema = TitleBlockSchema.from_dict(
        {"fields": [{"key": "a", "aliases": ["no"]}, {"key": "b", "aliases": ["no", "number"]}]}
    )
    assert schema.label_map()["no"] == "a"
    assert schema.label_map()["number"] == "b"


def test_table_label_patterns_longest_first_and_optional_dots():
    schema = TitleBlockSchema.default()
    patterns = schema.table_label_patterns()
    assert patterns == sorted(patterns, key=len, reverse=True)
    assert any(p.startswith(r"gen\.?") for p in patterns)
    assert "finish" in schema.table_label_aliases()


@pytest.mark.parametrize(
    "data, message",
    [
        ({"fields": []}, "at least one field"),
        ({"fields": [{"key": "Bad Key"}]}, "invalid title-block field key"),
        ({"fields": [{"key": "pairs"}]}, "invalid title-block field key"),
        ({"fields": [{"key": "a"}, {"key": "a"}]}, "unique"),
    ],
)
def test_invalid_schemas(data, message):
    with pytest.raises(ValueError, match=message):
        TitleBlockSchema.from_dict(data)


def test_field_spec_defaults():
    spec = FieldSpec.from_dict({"key": "sheet_size"})
    assert spec.label == "Sheet Size" and spec.join == " " and not spec.table_field


def test_domain_terms_validation_and_defaults():
    domain = DomainTerms.default()
    assert not domain.is_empty()
    assert "Vent hole" in domain.critical_terms()
    assert domain.signal_regex().search("VENT HOLE DIA 3.2")
    assert DomainTerms.empty().signal_regex() is None
    with pytest.raises(ValueError):
        DomainTerms.from_dict({"sub_part_tokens": ["L H"]})
    with pytest.raises(ValueError):
        DomainTerms.from_dict({"groups": [{"tag": "Bad Tag", "terms": ["x"]}]})
    generated = DomainTerms.from_dict({"groups": [{"tag": "rib", "terms": ["Rib", "Stiffener"]}]})
    assert generated.groups[0].search("main stiffener") and generated.groups[0].synonym_line() == "Domain synonym: Rib = Stiffener"


def test_from_env_overrides_and_files(tmp_path):
    schema_file = tmp_path / "schema.json"
    schema_file.write_text(json.dumps({"name": "tiny", "fields": [{"key": "drawing_number", "aliases": ["dwg"]}]}), encoding="utf-8")
    cfg = DrawingConfig.from_env(
        {
            "VLMRAG_DRAWING_MIN_PAIRS": "5",
            "VLMRAG_DRAWING_PAGE_DPI": "300",
            "VLMRAG_DRAWING_TITLE_BLOCK_PAGES": "all",
            "VLMRAG_DRAWING_INCLUDE_TEXT_LAYER": "0",
            "VLMRAG_DRAWING_MAX_IMAGE_PIXELS": "0",
            "VLMRAG_DRAWING_SCHEMA_PATH": str(schema_file),
            "VLMRAG_DRAWING_DOMAIN_TERMS_PATH": "none",
            "UNRELATED": "x",
        }
    )
    assert cfg.min_pairs == 5 and cfg.page_dpi == 300.0 and isinstance(cfg.page_dpi, float)
    assert cfg.title_block_pages == "all" and cfg.include_text_layer is False and cfg.max_image_pixels == 0
    assert cfg.schema.name == "tiny" and cfg.schema.keys() == ["drawing_number"]
    assert cfg.domain.is_empty()
    assert DrawingConfig.from_env({}, min_pairs=1).min_pairs == 1


@pytest.mark.parametrize(
    "changes",
    [{"title_block_pages": "some"}, {"page_tile_rows": 0}, {"page_dpi": 0}, {"min_pairs": -1}, {"max_image_pixels": -5}],
)
def test_invalid_config_values(changes):
    with pytest.raises(ValueError):
        DrawingConfig(**changes)


def test_invalid_env_bool():
    with pytest.raises(ValueError, match="VLMRAG_DRAWING_INCLUDE_TEXT_LAYER"):
        DrawingConfig.from_env({"VLMRAG_DRAWING_INCLUDE_TEXT_LAYER": "maybe"})
