"""The synthetic specification generator: determinism and truth consistency."""

from __future__ import annotations

import json
import zipfile

import pytest

from vlmrag.synthetic import load_generator
from vlmrag.synthetic.docx import STEM, generate
from vlmrag.techniques.docx.xmlsplit import app_page_count, read_docx_parts, split_docx


def test_registered_generator():
    assert load_generator("docx") is generate


def test_estimated_layout_is_byte_deterministic(tmp_path):
    a = generate(tmp_path / "a", seed=0, use_libreoffice=False)[0]
    b = generate(tmp_path / "b", seed=0, use_libreoffice=False)[0]
    c = generate(tmp_path / "c", seed=1, use_libreoffice=False)[0]
    assert a.read_bytes() == b.read_bytes()
    assert (a.parent / f"{STEM}.truth.json").read_text() == (b.parent / f"{STEM}.truth.json").read_text()
    assert a.read_bytes() != c.read_bytes()


def test_truth_matches_the_xml(synthetic_docx):
    path, truth = synthetic_docx
    assert path.parent.joinpath(f"{STEM}.truth.json").is_file()
    assert truth["source"] == path.name and truth["doc_type"] == "docx"
    assert 4 <= truth["page_count"] <= 6
    parts = read_docx_parts(path)
    assert app_page_count(parts.app_xml) == truth["page_count"]
    split = split_docx(path)
    assert split.mode == "rendered" and len(split.pages) == truth["page_count"] and split.padded_pages == 0
    [table] = split.tables
    expected = truth["tables"][0]
    assert expected["multi_page"] and table.table_id == expected["table_id"]
    assert 30 <= expected["row_count"] - 1 <= 50
    assert table.row_pages == expected["row_pages"]
    assert (table.start_page, table.end_page) == (expected["start_page"], expected["end_page"])
    assert expected["title"] in table.preceding_paragraphs
    image_pages = sorted({img["page"] for img in truth["images"]})
    assert [pg.number for pg in split.pages if pg.has_images] == image_pages


def test_queries_point_at_visual_only_facts(synthetic_docx):
    path, truth = synthetic_docx
    assert 3 <= len(truth["queries"]) <= 6
    image_pages = {img["page"] for img in truth["images"]}
    with zipfile.ZipFile(path) as archive:
        xml_text = archive.read("word/document.xml").decode("utf-8").lower()
    for query in truth["queries"]:
        for relevant in query["relevant"]:
            assert relevant["source"] == path.name and relevant["page"] in image_pages
    # The decisive facts live only in the pictures, never in the document text.
    for fact in ("5 v / 2 a", "green"):
        assert fact not in xml_text


def test_no_documents_written_outside_out_dir(tmp_path):
    out = tmp_path / "only_here"
    paths = generate(out, seed=3, use_libreoffice=False)
    assert all(p.parent == out for p in paths)
    assert sorted(p.name for p in out.iterdir()) == [f"{STEM}.docx", f"{STEM}.truth.json"]


@pytest.mark.soffice
def test_libreoffice_layout_matches_its_own_render(tmp_path):
    from vlmrag.render.office import convert
    from vlmrag.render.pdf import PdfPages

    [path] = generate(tmp_path / "lo", seed=0)
    truth = json.loads((path.parent / f"{STEM}.truth.json").read_text(encoding="utf-8"))
    assert truth["layout_source"] == "libreoffice"
    pdf = convert(path, "pdf", tmp_path / "pdf")
    with PdfPages(pdf) as pages:
        assert len(pages) == truth["page_count"]
        # Every data row is found on the page the truth (and the XML markers) assign it to.
        table = truth["tables"][0]
        page_text = [pages.text(i).replace(" ", "") for i in range(len(pages))]
    split = split_docx(path)
    assert split.tables[0].row_pages == table["row_pages"]
    for index, row in enumerate(split.tables[0].rows[1:], start=1):
        assert row[1] in page_text[table["row_pages"][index] - 1]
