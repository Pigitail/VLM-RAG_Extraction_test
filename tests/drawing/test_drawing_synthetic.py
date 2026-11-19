"""The synthetic drawing generator: determinism, geometry, text layers and truth files."""

from __future__ import annotations

import hashlib
import re

from vlmrag.render.pdf import PdfPages
from vlmrag.synthetic import load_generator
from vlmrag.synthetic.drawing import (
    REVISION_TABLE_BOX,
    TITLE_BLOCK_BOX,
    generate,
    load_truth,
    truth_responder,
)
from vlmrag.vlm.types import VLMRequest

from .conftest import BRACKET, HOUSING, REFLECTOR

A3_LANDSCAPE = (1190.55, 841.89)


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registered_and_writes_only_into_out_dir(tmp_path):
    assert load_generator("drawing") is generate
    paths = generate(tmp_path)
    assert sorted(p.name for p in paths) == sorted([HOUSING, REFLECTOR, BRACKET])
    expected = {p.name for p in paths} | {p.stem + ".truth.json" for p in paths}
    assert {p.name for p in tmp_path.iterdir()} == expected


def test_deterministic_for_a_seed(tmp_path, drawing_docs):
    again = {p.name: p for p in generate(tmp_path / "again", seed=0)}
    other = {p.name: p for p in generate(tmp_path / "other", seed=7)}
    for name, path in drawing_docs.items():
        assert _digest(path) == _digest(again[name])
        assert load_truth(path)["pages"] == load_truth(other[name])["pages"]  # seed never changes truth
    assert any(_digest(drawing_docs[n]) != _digest(other[n]) for n in drawing_docs)


def test_sheets_are_a3_and_housing_page_one_is_rotated(drawing_docs, truths):
    for name, path in drawing_docs.items():
        truth = truths[name]
        with PdfPages(path) as pages:
            assert len(pages) == truth["page_count"]
            for i, page in enumerate(truth["pages"]):
                width, height = pages.page_size(i)
                assert (round(width, 2), round(height, 2)) == A3_LANDSCAPE
                assert pages.rotation(i) == page["rotation"]
    assert [p["rotation"] for p in truths[HOUSING]["pages"]] == [90, 0]


def test_text_layer_matches_truth_flags(drawing_docs, truths):
    """Raster-only regions really have no text layer; vector regions have one (rotation included)."""
    for name, path in drawing_docs.items():
        with PdfPages(path) as pages:
            for i, page in enumerate(truths[name]["pages"]):
                title_text = _squash(pages.text(i, TITLE_BLOCK_BOX))
                rev_text = _squash(pages.text(i, REVISION_TABLE_BOX))
                fields = page["title_block"]
                assert (_squash(fields["drawing_number"]) in title_text) == page["text_layer"]["title_block"]
                assert (_squash(fields["material"]) in title_text) == page["text_layer"]["material_table"]
                if page["table_rows"]:
                    last_row = _squash(" ".join(page["table_rows"][-1][1:2]))
                    assert (last_row in rev_text) == page["text_layer"]["revision_table"]
                full = pages.text(i)
                assert "NOTES:" in full
                # Dense notes are 4.5 pt text in the text layer.
                assert page["notes"][1] in full


def test_queries_are_answerable_only_from_raster_content(drawing_docs, truths):
    for name, path in drawing_docs.items():
        truth = truths[name]
        assert 3 <= len(truth["queries"]) <= 6
        with PdfPages(path) as pages:
            text_layer = _squash(" ".join(pages.text(i) for i in range(len(pages))))
        for query in truth["queries"]:
            assert query["query"].endswith("?")
            for relevant in query["relevant"]:
                assert relevant["source"] == name
                assert 1 <= relevant["page"] <= truth["page_count"]
            assert _squash(query["evidence"]) not in text_layer, query["query"]
            page = truth["pages"][query["relevant"][0]["page"] - 1]
            visible = _squash(" ".join(list(page["title_block"].values()) + [c for row in page["table_rows"] for c in row]))
            assert _squash(query["evidence"]) in visible


def test_truth_structure(truths):
    for name, truth in truths.items():
        assert truth["source"] == name and truth["doc_type"] == "drawing"
        for page in truth["pages"]:
            assert set(page["title_block_labels"]) == set(page["title_block"])
            assert set(page["table_fields"]) <= set(page["title_block"])
            for row in page["table_rows"]:
                assert len(row) == 3
    reflector = truths[REFLECTOR]["pages"][0]
    assert reflector["raster_regions"] == ["revision_table", "title_block"]
    assert not reflector["text_layer"]["title_block"]


def test_truth_responder_answers_each_pass(truths):
    respond = truth_responder(truths.values())

    def ask(task, page=1, source=HOUSING):
        return respond(VLMRequest(task=task, prompt_version="x", parts=[], source_name=source, page=page))

    vision = ask("drawing.vision", page=2)
    assert vision["sheet"] == "2/2" and {"topic": "Sheet", "value": "2/2"} in vision["pairs"]
    table = ask("drawing.table")
    assert table["table_rows"][0]["cells"][1]["text"] == "DESCRIPTION" and "material" in table
    assert "drawing_number" not in table
    normalize = ask("drawing.normalize", source=REFLECTOR)
    assert normalize["drawing_number"] == "EX-L300-1004"
    for bad in ({"task": "other.task"}, {"source": "unknown.pdf"}):
        try:
            ask(bad.get("task", "drawing.vision"), source=bad.get("source", HOUSING))
        except KeyError:
            pass
        else:  # pragma: no cover - the responder must reject unknown requests
            raise AssertionError("expected KeyError")
