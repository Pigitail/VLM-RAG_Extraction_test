"""Answer parsing: pairs, fields (JSON and line fallback), table rows, value cleaning."""

from __future__ import annotations

import json

import pytest

from vlmrag.techniques.drawing.config import TitleBlockSchema
from vlmrag.techniques.drawing.parsing import (
    as_text,
    clean_field_value,
    dedupe_pairs,
    first_json_object_text,
    merge_pairs,
    pairs_text,
    parse_table_rows,
    parse_title_block,
    table_rows_matrix,
    table_rows_text,
)

SCHEMA = TitleBlockSchema.default()


def test_as_text():
    assert as_text(None) == "" and as_text("  a ") == "a" and as_text(3) == "3"
    assert as_text(["a", "", None, "b"]) == "a | b"
    assert as_text({"k": 1}) == '{"k": 1}'


def test_dedupe_pairs_keeps_longest_value_in_first_position():
    pairs = dedupe_pairs(
        [
            {"topic": "Drawing No.", "value": "EX-L300-10"},
            {"topic": "Rev", "value": "C"},
            {"topic": "DRAWING NO.", "value": "EX-L300-1001"},
            {"topic": "", "value": "orphan"},
            "not a dict",
            {"topic": "Rev:", "value": ""},
        ]
    )
    assert pairs == [{"topic": "DRAWING NO.", "value": "EX-L300-1001"}, {"topic": "Rev", "value": "C"}]


def test_merge_pairs_keeps_distinct_values_per_topic():
    merged = merge_pairs(
        [{"topic": "Finish", "value": "HOUSING: MATTE"}, {"topic": "Rev", "value": ""}],
        [{"topic": "finish:", "value": "HOUSING: MATTE"}, {"topic": "Finish", "value": "CAP: SATIN"}, None],
    )
    assert merged == [{"topic": "Finish", "value": "HOUSING: MATTE"}, {"topic": "Finish", "value": "CAP: SATIN"}]


def test_parse_json_answer_reads_fields_pairs_and_extra_keys():
    answer = {
        "pairs": [
            {"topic": "DWG. NO.", "value": "EX-L300-1001"},
            {"topic": "Title", "value": "LANTERN HOUSING ASSEMBLY"},
            {"topic": "Rev", "value": None},
        ],
        "title": "LANTERN HOUSING",
        "revision": 3,
        "material": ["PC", "UV"],
        "Paint Code": "RAL 9005",
        "table_rows": [],
    }
    parsed = parse_title_block("Here you go:\n```json\n" + json.dumps(answer) + "\n```\nDone.", SCHEMA)
    assert parsed.from_json
    assert parsed.fields["drawing_number"] == "EX-L300-1001"  # filled from a pair
    assert parsed.fields["title"] == "LANTERN HOUSING ASSEMBLY"  # longer pair value wins
    assert parsed.fields["revision"] == "3" and parsed.fields["material"] == "PC | UV"
    assert {"topic": "Paint Code", "value": "RAL 9005"} in parsed.pairs
    assert {"topic": "Rev", "value": ""} in parsed.pairs
    assert set(parsed.fields) == set(SCHEMA.keys())
    assert parsed.has_content()


def test_parse_line_fallback():
    text = (
        "I could read the following.\n"
        "Drawing No.: EX-L300-1004\n"
        "Rev: B\n"
        "Material\n"
        "ALUMINIUM\n"
        "1050-H14\n"
        "Surface Finish\n"
        "REFLECTOR: BRIGHT\n"
        "ANODIZED\n"
        "Scale: 1:1\n"
    )
    parsed = parse_title_block(text, SCHEMA)
    assert not parsed.from_json
    assert parsed.fields["drawing_number"] == "EX-L300-1004"
    assert parsed.fields["revision"] == "B"
    assert parsed.fields["material"] == "ALUMINIUM / 1050-H14"  # join of the material field
    assert parsed.fields["surface_finish"] == "REFLECTOR: BRIGHT; ANODIZED"
    assert parsed.fields["scale"] == "1:1"
    topics = [p["topic"] for p in parsed.pairs]
    assert "Material:" in topics and "Drawing No." in topics


def test_parse_garbage_and_empty():
    assert not parse_title_block("I cannot help with that", SCHEMA).has_content()
    assert not parse_title_block("", SCHEMA).has_content()
    assert not parse_title_block(None, SCHEMA).has_content()
    assert not parse_title_block("{}", SCHEMA).has_content()


def test_parse_table_rows():
    answer = {
        "table_rows": [
            {"row_index": 0, "cells": [{"col_index": 0, "text": "REV"}, {"col_index": 1, "text": " DESCRIPTION "}]},
            {"row_index": "x", "cells": [{"col_index": None, "text": "A"}, {"col_index": 2, "text": ""}, "junk"]},
            {"row_index": 5, "cells": []},
            "junk",
            {"row_index": True, "cells": [{"col_index": 1, "text": "B"}]},
        ]
    }
    rows = parse_table_rows(json.dumps(answer))
    assert rows == [
        {"row_index": 0, "cells": [{"col_index": 0, "text": "REV"}, {"col_index": 1, "text": "DESCRIPTION"}]},
        {"row_index": 1, "cells": [{"col_index": 0, "text": "A"}]},
        {"row_index": 2, "cells": [{"col_index": 1, "text": "B"}]},
    ]
    assert parse_table_rows("not json") == []
    assert parse_table_rows('{"table_rows": "nope"}') == []
    assert table_rows_text(rows) == "ROW 0: REV || DESCRIPTION\nROW 1: A\nROW 2: B"
    assert table_rows_matrix(rows) == [["REV", "DESCRIPTION"], ["A", ""], ["", "B"]]


def test_table_rows_matrix_merges_colliding_cells_and_bounds_width():
    rows = [{"row_index": 0, "cells": [{"col_index": 0, "text": "a"}, {"col_index": 0, "text": "b"}, {"col_index": 10_000, "text": "z"}]}]
    matrix = table_rows_matrix(rows)
    assert matrix[0][0] == "a b" and matrix[0][-1] == "z" and len(matrix[0]) == 64


def test_pairs_text():
    assert pairs_text([{"topic": "Rev", "value": "C"}, {"topic": "Material:", "value": "PC"}, {"topic": "Mass:", "value": ""}, {"topic": "Scale", "value": ""}, {"topic": "", "value": "x"}]) == (
        "Rev: C\nMaterial: PC\nMass:\nScale:"
    )


def test_first_json_object_text():
    assert first_json_object_text('Sure: {"a": {"b": "}"}} trailing {"c": 1}') == '{"a": {"b": "}"}}'
    assert first_json_object_text("  no json  ") == "no json"
    assert first_json_object_text(None) == ""


@pytest.mark.parametrize(
    "key, value, expected",
    [
        ("title", "  LANTERN \n HOUSING ", "LANTERN HOUSING"),
        ("material", "Material: PC-UV STABILISED", "PC-UV STABILISED"),
        ("general_tolerance", "ISO 2768-m F", "ISO 2768-m"),  # border zone letter picked up by the crop
        ("surface_finish", "Surface Finish | HOUSING: MATTE BLACK | CAP: SATIN GREY | HOUSING: MATTE BLACK", "HOUSING: MATTE BLACK | CAP: SATIN GREY"),
        ("material", "PC-UV\nMass 0.42 kg", "PC-UV 0.42 kg"),  # another table label echoed into the value
        ("mass", "Mass", ""),  # a bare echoed label is not a value
        ("mass", "F", "F"),  # no segment survives: fall back to the label-stripped value
        ("mass", "", ""),
        ("material", "Gen. Tol.: ISO 2768-f", "ISO 2768-f"),
        # The field's own label words inside a value are content, not an echoed label.
        ("surface_finish", "MILL FINISH", "MILL FINISH"),
        ("surface_finish", "Surface Finish SATIN FINISH", "SATIN FINISH"),
        ("surface_finish", "Finish: BRIGHT FINISH | Mass", "BRIGHT FINISH"),
        ("material", "MATERIAL: SEE MATERIAL LIST", "SEE MATERIAL LIST"),
        ("mass", "Weight 0.42 kg", "0.42 kg"),
        # Other table labels are neighbour-cell bleed wherever they appear.
        ("material", "ALUMINIUM Finish", "ALUMINIUM"),
    ],
)
def test_clean_field_value(key, value, expected):
    assert clean_field_value(key, value, SCHEMA) == expected
    assert clean_field_value(key, expected, SCHEMA) == expected  # merge_fields cleans twice


def test_clean_field_value_is_idempotent():
    value = "Surface finish: HOUSING: MATTE BLACK F | CAP: SATIN GREY"
    once = clean_field_value("surface_finish", value, SCHEMA)
    assert clean_field_value("surface_finish", once, SCHEMA) == once
