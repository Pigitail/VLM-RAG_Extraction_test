"""Document generation and discovery, and the docx layout pin used for portable replay."""

from __future__ import annotations

import json
import shutil

import pytest

from vlmrag.evaluation.documents import discover_documents, docx_layout_pins, generate_documents
from vlmrag.synthetic import accepted_options, generate_all
from vlmrag.synthetic.__main__ import main as synthetic_main
from vlmrag.synthetic.docx import STEM, generate
from vlmrag.techniques.docx.xmlsplit import split_docx


def _truth(path):
    return json.loads(path.with_name(f"{path.stem}.truth.json").read_text(encoding="utf-8"))


def test_docx_layout_pin_reproduces_the_document_byte_for_byte(tmp_path):
    [first] = generate(tmp_path / "a", seed=0, use_libreoffice=False)
    truth = _truth(first)
    assert truth["layout_source"] == "estimated" and len(truth["anchor_pages"]) > 40
    [again] = generate(tmp_path / "b", seed=0, layout=truth["anchor_pages"])
    pinned = _truth(again)
    assert again.read_bytes() == first.read_bytes()
    assert pinned["layout_source"] == "pinned"
    assert {k: v for k, v in pinned.items() if k != "layout_source"} == {
        k: v for k, v in truth.items() if k != "layout_source"
    }


def test_docx_layout_pin_moves_rows_and_markers_consistently(tmp_path):
    [first] = generate(tmp_path / "a", seed=0, use_libreoffice=False)
    truth = _truth(first)
    pages = list(truth["anchor_pages"])
    split_at = pages.index(3)  # first anchor on page 3 (a table row) moves back to page 2
    pages[split_at] = 2
    [moved] = generate(tmp_path / "b", seed=0, layout=pages)
    moved_truth = _truth(moved)
    old_rows, new_rows = truth["tables"][0]["row_pages"], moved_truth["tables"][0]["row_pages"]
    assert new_rows.count(2) == old_rows.count(2) + 1
    assert split_docx(moved).tables[0].row_pages == new_rows


@pytest.mark.parametrize("change", ["short", "start", "skip", "decrease"])
def test_docx_layout_pin_rejects_impossible_layouts(tmp_path, change):
    [first] = generate(tmp_path / "a", seed=0, use_libreoffice=False)
    pages = list(_truth(first)["anchor_pages"])
    if change == "short":
        pages = pages[:-1]
    elif change == "start":
        pages[0] = 2
    elif change == "skip":
        pages[-1] = pages[-1] + 2
    else:
        pages[-1] = pages[-2] - 1
    with pytest.raises(ValueError):
        generate(tmp_path / "b", seed=0, layout=pages)


def test_generator_options_reach_only_generators_that_accept_them(tmp_path):
    from vlmrag.synthetic.drawing import generate as drawing_generate

    assert accepted_options(drawing_generate, {"use_libreoffice": False}) == {}
    assert accepted_options(generate, {"use_libreoffice": False, "other": 1}) == {"use_libreoffice": False}
    written = generate_all(tmp_path, only=["docx"], seed=0, use_libreoffice=False)
    assert _truth(written["docx"][0])["layout_source"] == "estimated"


def test_synthetic_cli_no_libreoffice_flag(tmp_path, capsys):
    assert synthetic_main(["--out", str(tmp_path), "--only", "docx", "--no-libreoffice"]) == 0
    assert _truth(tmp_path / "docx" / f"{STEM}.docx")["layout_source"] == "estimated"
    assert "docx:" in capsys.readouterr().out


def test_generate_documents_uses_pins_over_the_layout_choice(tmp_path):
    docs = generate_documents(tmp_path / "a", ["docx"], seed=0, docx_layout="estimated")
    pins = docx_layout_pins(docs)
    assert list(pins) == [f"{STEM}.docx"]
    again = generate_documents(tmp_path / "b", ["docx"], seed=0, docx_layout="auto", docx_pins=pins)
    assert again[0].truth["layout_source"] == "pinned"
    assert again[0].path.read_bytes() == docs[0].path.read_bytes()
    with pytest.raises(ValueError):
        generate_documents(tmp_path / "c", ["docx"], docx_layout="libreoffice-ish")


def test_discover_documents_pairs_truth_skips_lock_files_and_rejects_duplicate_names(tmp_path, eval_docs):
    root = tmp_path / "docs"
    for doc in eval_docs:
        target = root / doc.technique
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(doc.path, target / doc.source)
        shutil.copy2(doc.path.with_name(f"{doc.path.stem}.truth.json"), target)
    (root / "docx" / "~$lantern_spec.docx").write_bytes(b"lock")
    (root / "docx" / "notes.docx").write_bytes(b"no truth")
    found = discover_documents(root, ["drawing", "slides", "docx"])
    names = [d.source for d in found]
    assert "~$lantern_spec.docx" not in names and "notes.docx" in names
    assert all(d.truth for d in found if d.source != "notes.docx")
    assert next(d for d in found if d.source == "notes.docx").truth is None
    assert [d.source for d in discover_documents(root, ["slides"])] == ["sample_erp_manual.pptx"]
    (root / "copy").mkdir()
    shutil.copy2(root / "slides" / "sample_erp_manual.pptx", root / "copy")
    with pytest.raises(ValueError, match="unique"):
        discover_documents(root, ["slides"])
    with pytest.raises(FileNotFoundError):
        discover_documents(tmp_path / "missing", ["slides"])
