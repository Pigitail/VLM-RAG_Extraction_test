"""Field priority (normalized > table > raw vision) and the MIN_PAIRS gate."""

from __future__ import annotations

import pytest

from vlmrag.techniques.drawing.config import TitleBlockSchema
from vlmrag.techniques.drawing.parsing import merge_pairs
from vlmrag.techniques.drawing.title_block import (
    TitleBlockGateError,
    apply_gate,
    canonical_pairs,
    merge_fields,
)

SCHEMA = TitleBlockSchema.default()


def _fields(**values):
    base = {k: "" for k in SCHEMA.keys()}
    base.update(values)
    return base


def test_merge_priority_normalized_table_raw():
    normalized = _fields(title="LANTERN HOUSING", drawing_number="EX-L300-1001")
    table = _fields(drawing_number="EX-L300-1009", material="Material: PC-UV STABILISED", mass="0.42 kg")
    raw = _fields(material="PC", mass="0.4 kg", scale="1:2")
    fields, sources = merge_fields(normalized, table, raw, SCHEMA)
    assert fields["drawing_number"] == "EX-L300-1001" and sources["drawing_number"] == "normalize"
    assert fields["material"] == "PC-UV STABILISED" and sources["material"] == "table"
    assert fields["mass"] == "0.42 kg" and sources["mass"] == "table"
    assert fields["scale"] == "1:2" and sources["scale"] == "vision"
    assert fields["sheet"] == "" and "sheet" not in sources
    assert set(fields) == set(SCHEMA.keys())


def test_merge_without_normalized_uses_vision_name():
    raw = _fields(title="T")
    fields, sources = merge_fields(raw, _fields(), raw, SCHEMA, primary_name="vision")
    assert fields["title"] == "T" and sources["title"] == "vision"


def test_canonical_pairs_follow_schema_order_and_labels():
    pairs = canonical_pairs(_fields(revision="C", title="T", drawing_number="D"), SCHEMA)
    assert pairs == [
        {"topic": "Title", "value": "T"},
        {"topic": "Drawing No.", "value": "D"},
        {"topic": "Rev", "value": "C"},
    ]


def test_gate_counts_canonical_plus_raw_pairs():
    canonical = canonical_pairs(_fields(title="T"), SCHEMA)
    raw = [{"topic": "Title", "value": "T"}, {"topic": "Paint code", "value": "RAL 9005"}]
    final = merge_pairs(canonical, raw)
    assert len(final) == 2
    with pytest.raises(TitleBlockGateError) as info:
        apply_gate(final, 3, page=2, source_name="x.pdf", raw_text="{}")
    err = info.value
    assert (err.page, err.pairs_found, err.min_pairs) == (2, 2, 3)
    assert "x.pdf page 2" in str(err) and "fewer than the required 3" in str(err)
    apply_gate(final + [{"topic": "Rev", "value": "C"}], 3, page=1)
    apply_gate([], 0, page=1)


def test_gate_error_survives_pickling():
    import pickle

    with pytest.raises(TitleBlockGateError) as info:
        apply_gate([], 3, page=4, source_name="x.pdf")
    clone = pickle.loads(pickle.dumps(info.value))
    assert (str(clone), clone.page, clone.pairs_found, clone.min_pairs) == (str(info.value), 4, 0, 3)
