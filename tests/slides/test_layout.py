from __future__ import annotations

import io
import logging
from types import SimpleNamespace

import pytest
from PIL import Image
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.util import Inches

from vlmrag.synthetic.slides import new_presentation
from vlmrag.techniques.slides import layout
from vlmrag.techniques.slides.layout import (
    Transform,
    bbox_norm,
    paragraph_lines,
    picture_crop,
    read_deck,
    sniff_vector_ext,
    sort_shapes_by_position,
    visible_crop_area,
)


def _png(color="white", size=(40, 30)) -> io.BytesIO:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def _save(prs, tmp_path, name="deck.pptx"):
    path = tmp_path / name
    prs.save(str(path))
    return path


# --------------------------------------------------------------------------- synthetic deck


def test_synthetic_deck_blocks_match_truth(synthetic_deck):
    deck_path, truth = synthetic_deck
    deck = read_deck(deck_path)
    assert deck.slide_count == truth["slide_count"]
    for expected in truth["slides"]:
        slide = deck.slides[expected["slide"] - 1]
        assert slide.number == expected["slide"]
        assert slide.title == expected["title"]
        names = {b.shape_name for b in slide.image_blocks}
        assert names == {b["shape_name"] for b in expected["image_blocks"]}
        for annotation in expected["annotation_shapes"]:
            blocks = [b for b in slide.blocks if b.shape_name == annotation]
            assert blocks, f"annotation shape {annotation} should be a layout block"
            assert all(b.type in ("shape", "text") for b in blocks)
        orders = [b.order for b in slide.blocks]
        assert orders == list(range(1, len(orders) + 1))


def test_bboxes_match_placement_including_scaled_group(synthetic_deck):
    deck_path, truth = synthetic_deck
    deck = read_deck(deck_path)
    for expected in truth["slides"]:
        slide = deck.slides[expected["slide"] - 1]
        for block_truth in expected["image_blocks"]:
            block = next(b for b in slide.image_blocks if b.shape_name == block_truth["shape_name"])
            for key in ("x", "y", "w", "h"):
                assert block.bbox_norm[key] == pytest.approx(block_truth["bbox_norm"][key], abs=0.002), (
                    block_truth["shape_name"], key)


def test_detail_crop_shares_source_image_and_records_crop(synthetic_deck):
    deck = read_deck(synthetic_deck[0])
    slide = deck.slides[2]
    main = next(b for b in slide.image_blocks if b.shape_name == "Screen PUR-120")
    detail = next(b for b in slide.image_blocks if b.shape_name == "Detail PUR-120")
    assert main.picture.source_image_id == detail.picture.source_image_id
    assert not main.picture.cropped and main.picture.crop_area_norm == 1.0
    assert detail.picture.cropped and 0.1 < detail.picture.crop_area_norm < 0.5
    data = detail.to_dict()
    assert set(data["crop_norm"]) == {"l", "t", "r", "b"}
    assert "blob" not in data and data["asset_ext"] == "png"


def test_slides_without_pictures_have_no_image_blocks(synthetic_deck):
    deck = read_deck(synthetic_deck[0])
    assert deck.slides[0].image_blocks == [] and deck.slides[5].image_blocks == []


def test_tables_have_plain_and_markdown_renderings(synthetic_deck):
    slide = read_deck(synthetic_deck[0]).slides[4]
    table = next(b for b in slide.blocks if b.type == "table")
    assert "Draft | Saved, not yet sent" in table.text_content
    assert table.markdown_content.startswith("| Status | Meaning |\n| --- | --- |")
    assert "| --- |" in slide.text_and_table and "| --- |" not in slide.text_only


def test_outline_levels_survive(synthetic_deck):
    slide = read_deck(synthetic_deck[0]).slides[5]
    assert "\n\t- Check that every order line has a delivery date." in slide.text_only
    assert "\n\t\t- Lines without a date block the release." in slide.text_only


# --------------------------------------------------------------------------- hand-built decks


def test_reading_order_is_top_then_left_and_ids_are_sequential(tmp_path):
    prs = new_presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "Order test"
    for name, left, top in (("C", 6, 4), ("B", 6, 2), ("A", 1, 2), ("D", 1, 5)):
        box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(2), Inches(0.5))
        box.name = name
        box.text_frame.text = f"text {name}"
    deck = read_deck(_save(prs, tmp_path))
    blocks = deck.slides[0].blocks
    assert [b.shape_name for b in blocks] == ["Title 1", "A", "B", "C", "D"]
    assert [b.order for b in blocks] == [1, 2, 3, 4, 5]
    assert deck.slides[0].text_only.splitlines()[1] == "- text A"


def test_annotation_shapes_become_shape_blocks(tmp_path):
    prs = new_presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "Shapes"
    arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(1), Inches(2), Inches(1), Inches(0.5))
    arrow.name = "Arrow shape"
    box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(3), Inches(2), Inches(1), Inches(0.5))
    box.name = "Highlight"
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(5), Inches(2), Inches(6), Inches(3))
    line.name = "Connector"
    empty_box = slide.shapes.add_textbox(Inches(7), Inches(2), Inches(1), Inches(0.5))
    empty_box.name = "Empty text box"
    deck = read_deck(_save(prs, tmp_path))
    by_name = {b.shape_name: b for b in deck.slides[0].blocks}
    assert by_name["Arrow shape"].type == "shape" and by_name["Arrow shape"].auto_shape_type == "RIGHT_ARROW"
    assert by_name["Highlight"].shape_type_name == "AUTO_SHAPE"
    assert by_name["Connector"].shape_type_name == "LINE" and by_name["Connector"].auto_shape_type == ""
    assert by_name["Empty text box"].shape_type_name == "TEXT_BOX"
    assert deck.slides[0].image_blocks == []


def test_empty_placeholder_is_not_a_block(tmp_path):
    prs = new_presentation()
    prs.slides.add_slide(prs.slide_layouts[0])  # title placeholder left empty
    deck = read_deck(_save(prs, tmp_path))
    assert deck.slides[0].blocks == [] and deck.slides[0].title == ""
    assert deck.slides[0].text_only == ""


def test_hidden_slide_is_flagged(tmp_path):
    prs = new_presentation()
    for title in ("visible", "hidden"):
        prs.slides.add_slide(prs.slide_layouts[0]).shapes.title.text = title
    prs.slides[1]._element.set("show", "0")
    deck = read_deck(_save(prs, tmp_path))
    assert [s.hidden for s in deck.slides] == [False, True]


def test_table_with_blank_header_falls_back_to_plain_and_drops_empty_rows(tmp_path):
    prs = new_presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    shape = slide.shapes.add_table(3, 2, Inches(1), Inches(1), Inches(4), Inches(1.5))
    shape.table.cell(1, 0).text = "Lamp | base"
    shape.table.cell(1, 1).text = "L-300"
    deck = read_deck(_save(prs, tmp_path))
    table = next(b for b in deck.slides[0].blocks if b.type == "table")
    assert table.text_content == "Lamp | base | L-300"
    assert table.markdown_content == table.text_content  # blank header -> plain rendering


def test_markdown_table_escapes_pipes_and_drops_empty_body_rows(tmp_path):
    prs = new_presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    shape = slide.shapes.add_table(3, 2, Inches(1), Inches(1), Inches(4), Inches(1.5))
    shape.table.cell(0, 0).text = "Part"
    shape.table.cell(0, 1).text = "Note"
    shape.table.cell(2, 0).text = "L-300"
    shape.table.cell(2, 1).text = "a | b"
    deck = read_deck(_save(prs, tmp_path))
    table = next(b for b in deck.slides[0].blocks if b.type == "table")
    assert table.markdown_content.splitlines() == ["| Part | Note |", "| --- | --- |", "| L-300 | a \\| b |"]


def test_one_failing_shape_does_not_lose_the_slide(tmp_path, monkeypatch, caplog):
    prs = new_presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "Pictures"
    good = slide.shapes.add_picture(_png("red"), Inches(1), Inches(2), Inches(2))
    good.name = "good"
    bad = slide.shapes.add_picture(_png("blue"), Inches(4), Inches(2), Inches(2))
    bad.name = "bad"
    path = _save(prs, tmp_path)
    real = layout._picture_info

    def flaky(shape):
        if shape.name == "bad":
            raise ValueError("linked picture")
        return real(shape)

    monkeypatch.setattr(layout, "_picture_info", flaky)
    with caplog.at_level(logging.WARNING, logger="vlmrag.techniques.slides.layout"):
        deck = read_deck(path)
    assert [b.shape_name for b in deck.slides[0].image_blocks] == ["good"]
    assert deck.slides[0].title == "Pictures"
    assert "bad" in caplog.text


# --------------------------------------------------------------------------- pure helpers


def test_sort_keeps_unpositioned_shapes_last():
    a = SimpleNamespace(name="a", top=10, left=5)
    b = SimpleNamespace(name="b", top=None, left=None)
    c = SimpleNamespace(name="c", top=10, left=1)
    d = SimpleNamespace(name="d", top=2, left=9)
    assert [s.name for s in sort_shapes_by_position([a, b, c, d])] == ["d", "c", "a", "b"]


def test_paragraph_lines_keep_levels_and_collapse_whitespace():
    frame = SimpleNamespace(paragraphs=[
        SimpleNamespace(text="Open  the   screen", level=0),
        SimpleNamespace(text="   ", level=0),
        SimpleNamespace(text="Pick\vthe vendor", level=1),
        SimpleNamespace(text="Detail", level=2),
    ])
    assert paragraph_lines(frame) == "- Open the screen\n\t- Pick the vendor\n\t\t- Detail"


def test_picture_crop_is_clamped():
    shape = SimpleNamespace(crop_left=-0.2, crop_top=0.25, crop_right=1.7, crop_bottom=None)
    assert picture_crop(shape) == (0.0, 0.25, 1.0, 0.0)


def test_visible_crop_area_and_bbox_norm():
    assert visible_crop_area((0.1, 0.2, 0.3, 0.4)) == pytest.approx(0.6 * 0.4)
    assert visible_crop_area((0.7, 0.0, 0.6, 0.0)) == 0.0
    assert bbox_norm(100, 50, 200, 100, 1000, 500) == {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}
    assert bbox_norm(1, 1, 1, 1, 0, 0) == {}


def test_sniff_vector_ext():
    emf = b"\x01\x00\x00\x00" + b"\x00" * 36 + b" EMF" + b"\x00" * 8
    assert sniff_vector_ext(emf) == "emf"
    assert sniff_vector_ext(b"\xd7\xcd\xc6\x9a" + b"\x00" * 20) == "wmf"
    assert sniff_vector_ext(b"\x01\x00\x09\x00" + b"\x00" * 20) == "wmf"
    assert sniff_vector_ext(_png().getvalue()) is None


def test_transform_composition():
    inner = Transform(sx=0.5, dx=100, sy=2.0, dy=-10)
    outer = Transform(sx=2.0, dx=5, sy=1.0, dy=7)
    composed = inner.then(outer)
    x, y, w, h = composed.apply(10, 20, 30, 40)
    ix, iy, iw, ih = inner.apply(10, 20, 30, 40)
    assert (x, y, w, h) == outer.apply(ix, iy, iw, ih)
