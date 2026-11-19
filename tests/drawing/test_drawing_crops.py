"""Crop geometry, position labels, pixel cap and region-matched text-layer hints."""

from __future__ import annotations

import re

import pytest
from PIL import Image

from vlmrag.render.images import format_bbox
from vlmrag.render.pdf import PdfPages
from vlmrag.techniques.drawing import DrawingConfig
from vlmrag.techniques.drawing.crops import (
    FULL_PAGE,
    corner_box,
    hint_block,
    hint_section,
    plan_crops,
    region_name,
    region_text,
    render_crop,
    scale_for_model,
)

from .conftest import BRACKET, HOUSING, REFLECTOR

TITLE_FOCUS = (0.68, 0.52, 1.0, 1.0)


def _approx(box, expected):
    return all(abs(a - b) < 1e-9 for a, b in zip(box, expected))


def test_default_plan_order_and_geometry():
    crops = plan_crops(DrawingConfig())
    kinds = [c.kind for c in crops]
    assert kinds == ["page_tile"] * 12 + ["title_block_focus", "table_focus", "table_overview"] + ["table_tile"] * 6
    tiles = crops[:12]
    assert {(c.dpi, c.upscale) for c in tiles} == {(400.0, 2.0)}
    # Row 1, column 2 of a 3 x 4 grid grown by 12 % of the tile size on every side.
    tw, th = 1 / 4, 1 / 3
    assert _approx(tiles[1].bbox, (tw - tw * 0.12, 0.0, 2 * tw + tw * 0.12, th + th * 0.12))
    assert _approx(tiles[-1].bbox, (3 * tw - tw * 0.12, 2 * th - th * 0.12, 1.0, 1.0))
    title, table_focus, overview = crops[12:15]
    assert title.bbox == TITLE_FOCUS and (title.dpi, title.upscale) == (500, 3.0)
    assert title.for_vision and not title.for_table
    assert table_focus.bbox == (0.82, 0.70, 1.0, 1.0) and (table_focus.dpi, table_focus.upscale) == (650, 4.0)
    assert table_focus.for_vision and table_focus.for_table
    assert overview.bbox == (0.44, 0.62, 1.0, 1.0) and not overview.for_vision and overview.for_table
    meta_tiles = crops[15:]
    assert all(0.44 <= c.bbox[0] and 0.62 <= c.bbox[1] for c in meta_tiles)
    mw, mh = 0.56 / 3, 0.38 / 2
    assert _approx(meta_tiles[1].bbox, (0.44 + mw - mw * 0.10, 0.62, 0.44 + 2 * mw + mw * 0.10, 0.62 + mh + mh * 0.10))
    assert all((c.dpi, c.upscale) == (650, 4.0) for c in meta_tiles)


def test_labels_are_unique_and_describe_position():
    crops = plan_crops(DrawingConfig())
    labels = [c.label for c in crops]
    assert len(set(labels)) == len(labels)
    for crop in crops:
        assert format_bbox(crop.bbox) in crop.label
        assert re.search(r"dpi=\d+$", crop.label)
    assert crops[0].label.startswith("PAGE_TILE r1c1 top-left x=0.00-")
    assert crops[11].label.startswith("PAGE_TILE r3c4 bottom-right ")
    assert crops[12].label.startswith("TITLE_BLOCK_FOCUS lower-right x=0.68-1.00 y=0.52-1.00 dpi=500")
    assert crops[-1].label.startswith("METADATA_TABLE_TILE r2c3 bottom-right ")
    # Labels never carry file system paths (they are part of the replay key).
    assert not any("\\" in label or "/" in label for label in labels)


@pytest.mark.parametrize(
    "row, col, rows, cols, expected",
    [
        (0, 0, 3, 4, "top-left"),
        (1, 2, 3, 4, "middle-center-right"),
        (1, 1, 2, 3, "bottom-center"),
        (0, 0, 1, 1, "whole"),
        (4, 5, 6, 7, "row5-col6"),
        (0, 1, 1, 2, "right"),
    ],
)
def test_region_names(row, col, rows, cols, expected):
    assert region_name(row, col, rows, cols) == expected


def test_corner_box_clamps_and_disables():
    assert corner_box(-0.2, 0.5) == (0.0, 0.5, 1.0, 1.0)
    assert corner_box(1.0, 0.5) is None
    crops = plan_crops(DrawingConfig(table_focus_x_start=1.0, metadata_y_start=1.2))
    assert {c.kind for c in crops} == {"page_tile", "title_block_focus"}


def test_focus_dpi_never_below_72():
    crops = plan_crops(DrawingConfig(title_focus_dpi=30, page_dpi=30))
    assert crops[12].dpi == 72 and crops[0].dpi == 30


def test_scale_for_model():
    img = Image.new("RGB", (100, 50), "white")
    out, factor, capped = scale_for_model(img, 2.0, 0)
    assert out.size == (200, 100) and factor == 2.0 and not capped
    out, factor, capped = scale_for_model(img, 4.0, 20_000)
    assert capped and out.width * out.height <= 20_000 and 1.0 < factor < 4.0
    out, factor, capped = scale_for_model(img, 1.0, 1_000)
    assert capped and out.width * out.height <= 1_000 and factor < 1.0
    out, factor, capped = scale_for_model(img, 0.5, 0)
    assert out.size == (50, 25) and not capped
    out, _, _ = scale_for_model(img, 1.0, 0)
    assert out is img


def test_render_crop_records_sizes(drawing_docs):
    spec = plan_crops(DrawingConfig(title_focus_dpi=72, title_focus_upscale=2.0))[12]
    with PdfPages(drawing_docs[BRACKET]) as pages:
        prepared = render_crop(pages, 0, spec, max_pixels=0)
    w, h = prepared.info["rendered_size"]
    assert prepared.info["size"] == [w * 2, h * 2]
    assert prepared.image.label == spec.label and prepared.image.png.startswith(b"\x89PNG")
    with PdfPages(drawing_docs[BRACKET]) as pages:
        capped = render_crop(pages, 0, spec, max_pixels=w * h)
    assert capped.info["capped"] and capped.info["size"][0] * capped.info["size"][1] <= w * h


def test_hint_helpers():
    assert hint_section("LBL", "  ") == ""
    assert hint_section("LBL", "x") == "[LBL]\nx"
    assert hint_block("intro", ["", ""]) == ""
    assert hint_block("intro", ["[A]\n1", "", "[B]\n2"]) == "intro\n[A]\n1\n\n[B]\n2"


def test_region_text_matches_the_cropped_region(drawing_docs, truths):
    """Hints come from the same box as the image: present for vector title blocks, absent for raster ones."""
    for name in (BRACKET, HOUSING):
        fields = truths[name]["pages"][0]["title_block"]
        with PdfPages(drawing_docs[name]) as pages:
            focus = region_text(pages, 0, TITLE_FOCUS)
            top_left = region_text(pages, 0, (0.0, 0.0, 0.25, 0.33))
        assert fields["drawing_number"] in focus
        assert fields["drawing_number"] not in top_left
        assert "NOTES:" in top_left and "NOTES:" not in focus
    with PdfPages(drawing_docs[REFLECTOR]) as pages:
        focus = region_text(pages, 0, TITLE_FOCUS)
        full = region_text(pages, 0, FULL_PAGE)
    assert truths[REFLECTOR]["pages"][0]["title_block"]["drawing_number"] not in focus + full
    assert "NOTES:" in full


def test_region_text_survives_text_layer_errors():
    class Broken:
        def text(self, index, bbox=None):
            raise RuntimeError("bad text layer")

    assert region_text(Broken(), 0, TITLE_FOCUS) == ""
    assert region_text(Broken(), 0, FULL_PAGE) == ""
