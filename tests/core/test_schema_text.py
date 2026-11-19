from __future__ import annotations

import hashlib

import pytest

from vlmrag.schema import Chunk, doc_id_for, read_jsonl, write_jsonl
from vlmrag.text import (
    dedupe_keep_order,
    filtering_text,
    markdown_table,
    normalize_unicode,
    normalize_ws,
    plain_table,
    truncate,
)


def _chunk(**overrides):
    base = dict(
        doc_id="abc",
        source_name="lantern.pdf",
        doc_type="drawing",
        page=1,
        chunk_index=0,
        retrieve_content="Part number L-300",
        llm_content="| Part | L-300 |",
        filtering_content="PartnumberL-300",
    )
    base.update(overrides)
    return Chunk(**base)


def test_doc_id_is_content_hash_prefix(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"hello world")
    assert doc_id_for(path) == hashlib.sha256(b"hello world").hexdigest()[:16]
    copy = tmp_path / "renamed.bin"
    copy.write_bytes(b"hello world")
    assert doc_id_for(copy) == doc_id_for(path)


def test_jsonl_round_trip(tmp_path):
    chunks = [
        _chunk(metadata={"fields": {"title": "Housing"}}, provenance={"model": "m"}),
        _chunk(page=2, chunk_index=1, retrieve_content="도면 번호 L-300-02"),
    ]
    path = tmp_path / "nested" / "out.jsonl"
    write_jsonl(chunks, path)
    assert read_jsonl(path) == chunks
    assert "도면" in path.read_text(encoding="utf-8")  # not ASCII-escaped
    assert chunks[0].to_dict()["metadata"] == {"fields": {"title": "Housing"}}


def test_page_must_be_one_based_source_page():
    for bad in (0, -1, "1", 1.0, True):
        with pytest.raises(ValueError):
            _chunk(page=bad)


def test_from_dict_ignores_unknown_keys():
    data = _chunk().to_dict()
    data["extra"] = 1
    assert Chunk.from_dict(data) == _chunk()


def test_whitespace_helpers():
    assert normalize_ws("  a\n\tb   c ") == "a b c"
    assert normalize_ws(None) == ""
    assert filtering_text(" A 1\n- 2 ") == "A1-2"
    assert normalize_unicode("“Ａ” – １") == '"A" - 1'


def test_markdown_table():
    table = markdown_table([["Item", "Qty"], ["Bolt | M6", 4], ["Nut"]])
    assert table.splitlines() == [
        "| Item | Qty |",
        "| --- | --- |",
        "| Bolt \\| M6 | 4 |",
        "| Nut |  |",
    ]
    no_header = markdown_table([["a", "b"]], header=False).splitlines()
    assert no_header[0] == "|  |  |" and no_header[2] == "| a | b |"
    assert markdown_table([]) == ""


def test_plain_table_and_truncate_and_dedupe():
    assert plain_table([["Item", "Qty"], ["Bolt", None, "4"], [], [" "]]) == "Item | Qty\nBolt | 4"
    assert truncate("abcdef", 3) == "abc"
    assert truncate("abc", 10) == "abc"
    assert truncate(None, 3) == ""
    assert dedupe_keep_order(["a", "", "b", "a", None, "c"]) == ["a", "b", "c"]
