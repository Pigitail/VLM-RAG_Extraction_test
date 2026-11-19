"""Placing multi-page table rows on pages and rendering page text."""

from __future__ import annotations

from vlmrag.techniques.docx.assemble import (
    VISUAL_HEADING_MD,
    VISUAL_HEADING_TEXT,
    TableAssignment,
    apply_table_assignments,
    is_table_heavy,
    render_page,
)
from vlmrag.techniques.docx.xmlsplit import split_document

MARK = "<w:lastRenderedPageBreak/>"


def p(text: str, mark: bool = False) -> str:
    return f"<w:p><w:r>{MARK if mark else ''}<w:t>{text}</w:t></w:r></w:p>"


def row(*cells: str, mark: bool = False) -> str:
    out = []
    for i, cell in enumerate(cells):
        lead = MARK if mark and i == 0 else ""
        out.append(f"<w:tc><w:p><w:r>{lead}<w:t>{cell}</w:t></w:r></w:p></w:tc>")
    return "<w:tr>" + "".join(out) + "</w:tr>"


def three_page_split(wxml):
    """Table header+5 rows over pages 1-3 (markers: rows 0-1 on p1, 2-3 on p2, 4-5 on p3), text after it."""
    rows = [row("Item", "Qty"), row("a", "1"), row("b", "2", mark=True), row("c", "3"),
            row("d", "4", mark=True), row("e", "5")]
    body = p("Table 5. Stock") + "<w:tbl>" + "".join(rows) + "</w:tbl>" + p("closing note")
    return split_document(wxml(body))


def test_fallback_places_rows_where_the_editor_rendered_them(wxml):
    split = three_page_split(wxml)
    used = apply_table_assignments(split.pages, split.tables, {})
    assert used["t1"].source == "xml_row_markers" and used["t1"].reason == "not_attempted"
    page1, page2, page3 = split.pages
    assert page1.blocks[-1].row_indexes == [0, 1]
    assert page2.blocks[0].row_indexes == [2, 3]
    assert [b.kind for b in page3.blocks] == ["table", "paragraph"]  # slot replaced in place
    assert page3.blocks[0].row_indexes == [4, 5]
    assert page2.blocks[0].continued_from == 1 and page1.blocks[-1].continued_from is None
    md, text = render_page(page2)
    assert text.startswith("(Table continued from page 1)\nb | 2\nc | 3")
    assert md.startswith("_(Table continued from page 1)_\n\n|  |  |\n| --- | --- |\n| b | 2 |")
    _, text1 = render_page(page1)
    assert "Item | Qty\na | 1" in text1


def test_validated_assignment_and_title_are_applied(wxml):
    split = three_page_split(wxml)
    assignment = TableAssignment("t1", {1: [0, 1, 2], 2: [3, 4, 5], 3: []}, "vlm",
                                 title="Table 5. Stock", title_source="preceding_paragraph", title_confidence="high")
    apply_table_assignments(split.pages, split.tables, {"t1": assignment})
    page1, page2, page3 = split.pages
    assert page1.blocks[-1].row_indexes == [0, 1, 2]
    assert page2.blocks[0].row_indexes == [3, 4, 5]
    assert [b.kind for b in page3.blocks] == ["paragraph"]  # empty slot removed, note kept
    assert page1.table_titles == ["Table 5. Stock"] and page2.table_titles == ["Table 5. Stock"]
    assert page3.table_titles == []
    assert page2.table_header_hint == "Table 5. Stock"
    _, text2 = render_page(page2)
    assert text2.startswith("(Table continued from page 1: Table 5. Stock)")


def test_low_confidence_title_is_not_propagated(wxml):
    split = three_page_split(wxml)
    assignment = TableAssignment("t1", {1: [0, 1], 2: [2, 3], 3: [4, 5]}, "vlm", title="Table 5. Stock",
                                 title_confidence="low")
    apply_table_assignments(split.pages, split.tables, {"t1": assignment})
    assert all(not page.table_titles for page in split.pages)


def test_out_of_range_indexes_are_ignored_defensively(wxml):
    split = three_page_split(wxml)
    assignment = TableAssignment("t1", {1: [0, 1, 99], 2: [2, 3], 3: [4, 5]}, "vlm")
    apply_table_assignments(split.pages, split.tables, {"t1": assignment})
    assert split.pages[0].blocks[-1].row_indexes == [0, 1]


def test_render_page_appends_visual_descriptions_last(wxml):
    page = split_document(wxml(p("Body text"))).pages[0]
    md, text = render_page(page, [
        {"location_hint": "centre diagram", "description": "Three boxes joined by arrows."},
        {"location_hint": "", "description": "A small logo."},
    ])
    assert md == (
        f"Body text\n\n{VISUAL_HEADING_MD}\n- **centre diagram**: Three boxes joined by arrows.\n- A small logo."
    )
    assert text == f"Body text\n\n{VISUAL_HEADING_TEXT}\n- centre diagram: Three boxes joined by arrows.\n- A small logo."


def test_table_heavy_flag(wxml):
    split = three_page_split(wxml)
    apply_table_assignments(split.pages, split.tables, {})
    assert [is_table_heavy(page) for page in split.pages] == [True, True, True]
    assert not is_table_heavy(split_document(wxml(p("a") + p("b"))).pages[0])
