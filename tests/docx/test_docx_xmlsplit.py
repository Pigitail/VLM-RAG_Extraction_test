"""Unit tests for the XML-first page split (no model, no LibreOffice)."""

from __future__ import annotations

import zipfile

import pytest

from vlmrag.techniques.docx.xmlsplit import (
    DocxExtractionError,
    app_page_count,
    heading_level,
    read_docx_parts,
    split_docx,
    split_document,
    style_names,
)

MARK = "<w:lastRenderedPageBreak/>"
PAGE_BREAK = '<w:r><w:br w:type="page"/></w:r>'
DRAWING = "<w:r><w:drawing/></w:r>"


def p(text: str = "", style: str = "", mark: bool = False, extra: str = "") -> str:
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    run = f"<w:r>{MARK if mark else ''}<w:t>{text}</w:t></w:r>" if text else ""
    return f"<w:p>{ppr}{run}{extra}</w:p>"


def row(*cells: str, mark: bool = False, mark_all: bool = False) -> str:
    out = []
    for i, cell in enumerate(cells):
        lead = MARK if mark_all or (mark and i == 0) else ""
        out.append(f"<w:tc><w:p><w:r>{lead}<w:t>{cell}</w:t></w:r></w:p></w:tc>")
    return "<w:tr>" + "".join(out) + "</w:tr>"


def tbl(*rows: str) -> str:
    return "<w:tbl>" + "".join(rows) + "</w:tbl>"


def texts(page) -> list[str]:
    return [b.text for b in page.blocks]


# --------------------------------------------------------------------------- page boundaries


def test_rendered_breaks_split_pages(wxml):
    split = split_document(wxml(p("one") + p("two", mark=True) + p("three") + p("four", mark=True)))
    assert [texts(pg) for pg in split.pages] == [["one"], ["two", "three"], ["four"]]
    assert split.mode == "rendered"
    assert split.rendered_breaks == 2
    assert [pg.number for pg in split.pages] == [1, 2, 3]


def test_break_in_the_middle_of_a_paragraph_keeps_text_on_both_pages(wxml):
    body = f"<w:p><w:r><w:t>first half</w:t></w:r><w:r>{MARK}<w:t>second half</w:t></w:r></w:p>"
    split = split_document(wxml(body))
    assert [texts(pg) for pg in split.pages] == [["first half"], ["second half"]]


def test_explicit_page_break_without_rendered_markers(wxml):
    split = split_document(wxml(p("a", extra=PAGE_BREAK) + p("b")))
    assert [texts(pg) for pg in split.pages] == [["a"], ["b"]]
    assert split.mode == "explicit"


def test_rendered_marker_right_after_manual_break_is_the_same_boundary(wxml):
    # Editors write both a manual break and a rendered marker at the top of the next page.
    split = split_document(wxml(p("a", extra=PAGE_BREAK) + p("b", mark=True) + p("c")))
    assert [texts(pg) for pg in split.pages] == [["a"], ["b", "c"]]
    assert split.merged_breaks == 1
    assert split.mode == "rendered"


def test_rendered_marker_after_content_is_not_merged(wxml):
    split = split_document(wxml(p("a", extra=PAGE_BREAK) + p("b") + p("c", mark=True)))
    assert [texts(pg) for pg in split.pages] == [["a"], ["b"], ["c"]]
    assert split.merged_breaks == 0


def test_no_page_information_is_one_page(wxml):
    split = split_document(wxml(p("a") + p("b") + p("c")))
    assert len(split.pages) == 1
    assert split.mode == "none"


def test_app_xml_pages_pads_visual_only_tail_pages(wxml):
    split = split_document(wxml(p("a") + p("b", mark=True)), app_pages=4)
    assert len(split.pages) == 4
    assert split.padded_pages == 2
    tail = split.pages[2:]
    assert all(pg.placeholder and pg.has_images and not pg.blocks for pg in tail)
    assert [pg.number for pg in split.pages] == [1, 2, 3, 4]
    assert split.info()["xml_pages"] == 2


def test_more_pages_than_app_xml_are_kept(wxml):
    split = split_document(wxml(p("a") + p("b", mark=True) + p("c", mark=True)), app_pages=2)
    assert len(split.pages) == 3
    assert split.padded_pages == 0


def test_empty_page_between_breaks_keeps_numbering(wxml):
    split = split_document(wxml(p("a") + p("", extra=f"<w:r>{MARK}</w:r>") + p("b", mark=True)))
    assert [texts(pg) for pg in split.pages] == [["a"], [], ["b"]]
    assert split.pages[1].is_empty


# --------------------------------------------------------------------------- headings and text


@pytest.mark.parametrize(
    "style_id, name, level",
    [("Title", "", 1), ("Subtitle", "", 1), ("Heading1", "", 1), ("Heading3", "", 3), ("HeadingX", "", 2),
     ("Heading0", "", 1), ("Normal", "", None), ("Berschrift1", "heading 2", 2), ("", "", None)],
)
def test_heading_level(style_id, name, level):
    assert heading_level(style_id, name) == level


def test_headings_set_page_and_section_titles(wxml):
    body = p("Doc", style="Title") + p("Intro", style="Heading1") + p("Detail", style="Heading2") + p("text")
    page = split_document(wxml(body)).pages[0]
    assert page.page_title == "Doc"
    assert page.section_title == "Intro"
    assert [b.markdown for b in page.blocks] == ["## Doc", "## Intro", "### Detail", "text"]
    assert [b.kind for b in page.blocks] == ["heading", "heading", "heading", "paragraph"]


def test_localized_style_ids_resolve_through_style_names(wxml):
    styles = style_names(
        f'<w:styles xmlns:w="{"http://schemas.openxmlformats.org/wordprocessingml/2006/main"}">'
        '<w:style w:styleId="Kop1"><w:name w:val="heading 1"/></w:style></w:styles>'.encode()
    )
    assert styles == {"Kop1": "heading 1"}
    page = split_document(wxml(p("Scope", style="Kop1")), styles=styles).pages[0]
    assert page.page_title == "Scope"


def test_tabs_line_breaks_and_whitespace(wxml):
    body = "<w:p><w:r><w:t>a</w:t><w:tab/><w:t>b </w:t><w:br/><w:br/><w:t>c   d</w:t><w:cr/><w:t>e</w:t></w:r></w:p>"
    assert texts(split_document(wxml(body)).pages[0]) == ["a b\nc d\ne"]


def test_embedded_object_preview_flags_the_page(wxml):
    body = "<w:p><w:r><w:object/></w:r></w:p>"
    page = split_document(wxml(p("text") + body)).pages[0]
    assert page.has_images and page.image_count == 1


def test_text_box_paragraphs_join_their_host_paragraph(wxml):
    box = (
        "<w:r><w:drawing><w:txbxContent>"
        "<w:p><w:r><w:t>box one</w:t></w:r></w:p><w:p><w:r><w:t>box two</w:t></w:r></w:p>"
        "</w:txbxContent></w:drawing></w:r>"
    )
    body = f"<w:p><w:r><w:t>before</w:t></w:r>{box}<w:r><w:t> after</w:t></w:r></w:p>" + p("next")
    page = split_document(wxml(body)).pages[0]
    assert texts(page) == ["before\nbox one\nbox two\nafter", "next"]
    assert page.has_images and page.image_count == 1


def test_markup_compatibility_fallback_is_not_read_twice(wxml):
    body = (
        "<w:p><w:r><mc:AlternateContent><mc:Choice Requires=\"wps\"><w:drawing><w:txbxContent><w:p><w:r>"
        "<w:t>label</w:t></w:r></w:p></w:txbxContent></w:drawing></mc:Choice><mc:Fallback><w:pict><w:txbxContent>"
        "<w:p><w:r><w:t>label</w:t></w:r></w:p></w:txbxContent></w:pict></mc:Fallback></mc:AlternateContent>"
        "</w:r></w:p>"
    )
    page = split_document(wxml(body)).pages[0]
    assert texts(page) == ["label"]
    assert page.image_count == 1


def test_images_are_attributed_to_the_page_they_appear_on(wxml):
    body = f"<w:p><w:r><w:t>caption above</w:t></w:r>{DRAWING}<w:r>{MARK}<w:t>next page text</w:t></w:r></w:p>"
    split = split_document(wxml(body))
    assert split.pages[0].has_images and not split.pages[1].has_images

    body = f"<w:p><w:r>{MARK}<w:drawing/></w:r></w:p>" + p("caption")
    split = split_document(wxml(p("first") + body))
    assert not split.pages[0].has_images
    assert split.pages[1].has_images and texts(split.pages[1]) == ["caption"]


def test_picture_inside_a_table_cell_flags_the_page(wxml):
    body = tbl("<w:tr><w:tc><w:p><w:r><w:drawing/></w:r><w:r><w:t>x</w:t></w:r></w:p></w:tc></w:tr>")
    assert split_document(wxml(body)).pages[0].has_images


def test_deleted_text_and_field_codes_are_ignored(wxml):
    body = (
        "<w:p><w:r><w:t>kept</w:t></w:r><w:del><w:r><w:delText>gone</w:delText></w:r></w:del>"
        "<w:r><w:instrText>PAGE</w:instrText></w:r></w:p>"
    )
    assert texts(split_document(wxml(body)).pages[0]) == ["kept"]


# --------------------------------------------------------------------------- tables


def test_single_page_table_becomes_a_block_with_header_hint(wxml):
    body = p("Table 9. Values") + tbl(row("Name", "Value"), row("alpha", "1"), row(" ", " "), row("beta", "2"))
    page = split_document(wxml(body)).pages[0]
    table = page.blocks[-1]
    assert table.kind == "table"
    assert table.text == "Name | Value\nalpha | 1\nbeta | 2"  # the blank row is dropped
    assert table.markdown.splitlines()[0] == "| Name | Value |"
    assert page.table_header_hint == "Name | Value"


def test_multi_page_table_event_and_slot(wxml):
    body = (
        p("intro")
        + p("Table 1. Parts")
        + tbl(row("Item", "Qty"), row("a", "1"), row("", ""), row("b", "2", mark=True), row("c", "3"))
        + p("after")
    )
    split = split_document(wxml(body))
    assert len(split.pages) == 2
    [table] = split.tables
    assert (table.table_id, table.start_page, table.end_page) == ("t1", 1, 2)
    assert table.rows == [["Item", "Qty"], ["a", "1"], ["b", "2"], ["c", "3"]]
    assert table.row_pages == [1, 1, 2, 2]
    assert table.preceding_paragraphs == ["intro", "Table 1. Parts"]
    assert [b.kind for b in split.pages[0].blocks] == ["paragraph", "paragraph"]
    assert [b.kind for b in split.pages[1].blocks] == ["table_slot", "paragraph"]
    assert split.pages[1].blocks[0].table_id == "t1"


def test_table_starting_at_the_top_of_a_page_is_not_a_multi_page_table(wxml):
    # The marker of a table that begins a page sits inside its first cell, after <w:tbl> opened.
    body = p("intro") + tbl(row("Item", "Qty", mark=True), row("a", "1")) + p("after")
    split = split_document(wxml(body))
    assert split.tables == []
    assert [b.kind for b in split.pages[0].blocks] == ["paragraph"]
    assert [b.kind for b in split.pages[1].blocks] == ["table", "paragraph"]
    assert split.pages[1].table_header_hint == "Item | Qty"


def test_markers_in_every_cell_of_a_row_are_one_boundary(wxml):
    # Cells are laid out side by side: each one may carry a marker for the same page start.
    body = p("intro") + tbl(row("Item", "Qty"), row("a", "1"), row("b", "2", mark_all=True), row("c", "3")) + p("end")
    split = split_document(wxml(body))
    assert len(split.pages) == 2
    assert split.rendered_breaks == 2 and split.merged_breaks == 1
    [table] = split.tables
    assert (table.start_page, table.end_page, table.row_pages) == (1, 2, [1, 1, 2, 2])


def test_a_row_crosses_as_many_pages_as_its_most_broken_cell(wxml):
    tall = (
        f"<w:tr><w:tc><w:p><w:r><w:t>x1</w:t></w:r></w:p><w:p><w:r>{MARK}<w:t>x2</w:t></w:r></w:p>"
        f"<w:p><w:r>{MARK}<w:t>x3</w:t></w:r></w:p></w:tc>"
        f"<w:tc><w:p><w:r><w:t>y1</w:t></w:r></w:p><w:p><w:r>{MARK}<w:t>y2</w:t></w:r></w:p></w:tc></w:tr>"
    )
    split = split_document(wxml(tbl(row("h1", "h2"), tall, row("z", "9"))))
    assert len(split.pages) == 3
    [table] = split.tables
    assert table.row_pages == [1, 3, 3]


def test_manual_break_before_a_table_with_markers_in_every_cell(wxml):
    body = p("a", extra=PAGE_BREAK) + tbl(row("Item", "Qty", mark_all=True), row("b", "2")) + p("c")
    split = split_document(wxml(body))
    assert [texts(pg) for pg in split.pages] == [["a"], ["Item | Qty\nb | 2", "c"]]
    assert split.tables == [] and split.merged_breaks == 2


def test_preceding_paragraphs_keep_the_last_five(wxml):
    body = "".join(p(f"p{i}") for i in range(8)) + tbl(row("h"), row("x", mark=True))
    [table] = split_document(wxml(body)).tables
    assert table.preceding_paragraphs == ["p3", "p4", "p5", "p6", "p7"]


def test_nested_table_text_is_flattened_into_the_outer_cell(wxml):
    inner = tbl(row("in1", "in2"))
    body = tbl(f"<w:tr><w:tc><w:p><w:r><w:t>outer</w:t></w:r></w:p>{inner}</w:tc><w:tc><w:p/></w:tc></w:tr>")
    page = split_document(wxml(body)).pages[0]
    assert page.blocks[0].text == "outer in1 in2"


# --------------------------------------------------------------------------- package access


def test_app_page_count_variants():
    assert app_page_count(b"<Properties><Pages>7</Pages></Properties>") == 7
    assert app_page_count(b"<ep:Properties><ep:Pages> 3 </ep:Pages></ep:Properties>") == 3
    assert app_page_count("<Properties><Pages>2</Pages></Properties>") == 2
    assert app_page_count(b"<Properties/>") == 0
    assert app_page_count(b"") == 0


def test_split_docx_reads_app_xml(make_docx):
    path = make_docx(p("a") + p("b", mark=True), pages=3)
    split = split_docx(path)
    assert len(split.pages) == 3 and split.app_pages == 3 and split.padded_pages == 1


def test_main_part_is_resolved_from_package_relationships(tmp_path, wxml):
    path = tmp_path / "moved.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "_rels/.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
            'officeDocument" Target="/content/main.xml"/></Relationships>',
        )
        archive.writestr("content/main.xml", wxml(p("moved text")))
    parts = read_docx_parts(path)
    assert parts.main_part == "content/main.xml"
    assert texts(split_docx(path).pages[0]) == ["moved text"]


def test_not_a_docx_raises_a_clear_error(tmp_path):
    bogus = tmp_path / "bogus.docx"
    bogus.write_bytes(b"not a zip at all")
    with pytest.raises(DocxExtractionError, match="not a DOCX"):
        split_docx(bogus)
    empty = tmp_path / "empty.docx"
    with zipfile.ZipFile(empty, "w") as archive:
        archive.writestr("other.xml", "<x/>")
    with pytest.raises(DocxExtractionError, match="missing"):
        split_docx(empty)


def test_ole_container_gets_a_specific_hint(tmp_path):
    # Legacy .doc files and encrypted or rights-protected Office files are OLE compound files.
    protected = tmp_path / "protected.docx"
    protected.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 504)
    with pytest.raises(DocxExtractionError, match="OLE compound file"):
        split_docx(protected)


def test_invalid_xml_raises_a_clear_error():
    with pytest.raises(DocxExtractionError, match="invalid WordprocessingML"):
        split_document(b"<w:document><w:body><w:p></w:body>")
