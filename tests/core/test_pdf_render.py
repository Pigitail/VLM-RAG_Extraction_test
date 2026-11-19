"""PdfPages must speak displayed coordinates for every /Rotate value.

Each generated page has the same displayed size (A6 landscape-ish 400 x 200 pt)
and a red square plus the word MARKTL in the *unrotated* top-left corner and
MARKBR in the unrotated bottom-right corner. Rotating the page clockwise moves
those corners on screen, and render()/text() must follow.
"""

from __future__ import annotations

import pypdfium2 as pdfium
import pytest
from reportlab.pdfgen import canvas

from vlmrag.render.pdf import PdfPages, displayed_to_user_space

DISPLAY_W, DISPLAY_H = 400.0, 200.0
ROTATIONS = (0, 90, 180, 270)
QUADRANTS = {
    "top_left": (0.0, 0.0, 0.5, 0.5),
    "top_right": (0.5, 0.0, 1.0, 0.5),
    "bottom_left": (0.0, 0.5, 0.5, 1.0),
    "bottom_right": (0.5, 0.5, 1.0, 1.0),
}
# Where the unrotated top-left / bottom-right corners appear after a clockwise rotation.
EXPECTED_TL = {0: "top_left", 90: "top_right", 180: "bottom_right", 270: "bottom_left"}
EXPECTED_BR = {0: "bottom_right", 90: "bottom_left", 180: "top_left", 270: "top_right"}


@pytest.fixture(scope="module")
def rotated_pdf(tmp_path_factory):
    path = tmp_path_factory.mktemp("pdf") / "rotated.pdf"
    c = canvas.Canvas(str(path), pagesize=(DISPLAY_W, DISPLAY_H), invariant=1)
    for rotation in ROTATIONS:
        # reportlab treats pagesize as the displayed size and swaps the MediaBox for 90/270,
        # so drawing happens in the unrotated coordinate system below.
        unrot_w, unrot_h = (DISPLAY_W, DISPLAY_H) if rotation in (0, 180) else (DISPLAY_H, DISPLAY_W)
        c.setPageRotation(rotation)
        c.setFillColorRGB(1, 0, 0)
        c.rect(0, unrot_h - 30, 30, 30, fill=1, stroke=0)
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica", 8)
        c.drawString(2, unrot_h - 42, "MARKTL")
        c.drawString(unrot_w - 40, 4, "MARKBR")
        c.showPage()
    c.save()
    # Sanity-check the fixture itself: rotation and unrotated MediaBox are what we intended.
    doc = pdfium.PdfDocument(str(path))
    try:
        for index, rotation in enumerate(ROTATIONS):
            page = doc[index]
            assert page.get_rotation() == rotation
            unrot = (DISPLAY_W, DISPLAY_H) if rotation in (0, 180) else (DISPLAY_H, DISPLAY_W)
            assert tuple(page.get_mediabox()) == (0.0, 0.0, *unrot)
            page.close()
    finally:
        doc.close()
    return path


def _red_pixels(image):
    pixels = image.load()
    return sum(
        1
        for x in range(image.width)
        for y in range(image.height)
        if pixels[x, y][0] > 200 and pixels[x, y][1] < 80 and pixels[x, y][2] < 80
    )


def test_page_count_and_displayed_size(rotated_pdf):
    with PdfPages(rotated_pdf) as pages:
        assert len(pages) == 4
        for index, rotation in enumerate(ROTATIONS):
            assert pages.rotation(index) == rotation
            assert pages.page_size(index) == (DISPLAY_W, DISPLAY_H)


@pytest.mark.parametrize("index, rotation", list(enumerate(ROTATIONS)))
def test_full_render_is_displayed_orientation(rotated_pdf, index, rotation):
    with PdfPages(rotated_pdf) as pages:
        image = pages.render(index, dpi=72)
    assert image.mode == "RGB"
    assert image.size == (400, 200)
    # The 30x30 pt marker lands in the expected displayed corner.
    w, h = image.size
    corners = {
        "top_left": (0, 0, 30, 30),
        "top_right": (w - 30, 0, w, 30),
        "bottom_left": (0, h - 30, 30, h),
        "bottom_right": (w - 30, h - 30, w, h),
    }
    for name, box in corners.items():
        red = _red_pixels(image.crop(box))
        if name == EXPECTED_TL[rotation]:
            assert red > 800, (rotation, name, red)
        else:
            assert red == 0, (rotation, name, red)


@pytest.mark.parametrize("index, rotation", list(enumerate(ROTATIONS)))
def test_region_render_and_text_follow_rotation(rotated_pdf, index, rotation):
    with PdfPages(rotated_pdf) as pages:
        for name, bbox in QUADRANTS.items():
            image = pages.render(index, dpi=72, bbox=bbox)
            assert image.size == (200, 100)  # only the region is rasterised
            red = _red_pixels(image)
            text = pages.text(index, bbox)
            if name == EXPECTED_TL[rotation]:
                assert red > 800
                assert "MARKTL" in text and "MARKBR" not in text
            elif name == EXPECTED_BR[rotation]:
                assert red == 0
                assert text == "MARKBR"
            else:
                assert red == 0
                assert text == ""


def test_render_dpi_and_upscale(rotated_pdf):
    with PdfPages(rotated_pdf) as pages:
        region = pages.render(1, dpi=144, bbox=(0.5, 0.0, 1.0, 0.5))
        assert region.size == (400, 200)
        enlarged = pages.render(1, dpi=72, bbox=(0.5, 0.0, 1.0, 0.5), upscale=2.0)
        assert enlarged.size == (400, 200)
        assert _red_pixels(enlarged) > 3000


def test_text_full_page_and_errors(rotated_pdf):
    with PdfPages(rotated_pdf) as pages:
        full = pages.text(2)
        assert "MARKTL" in full and "MARKBR" in full
        assert "\r" not in full
        with pytest.raises(IndexError):
            pages.render(4, dpi=72)
        with pytest.raises(ValueError):
            pages.render(0, dpi=72, bbox=(0.5, 0.5, 0.5, 0.9))
    with pytest.raises(ValueError):
        pages.text(0)  # closed


def test_pages_from_bytes(rotated_pdf):
    with PdfPages(rotated_pdf.read_bytes()) as pages:
        assert len(pages) == 4


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_displayed_to_user_space(rotation):
    # Unrotated page 200 x 100 pt. The displayed quadrant that shows the unrotated top-left
    # corner must map back to the unrotated top-left quarter for every rotation.
    page_box = (0.0, 0.0, 200.0, 100.0)
    bbox = QUADRANTS[EXPECTED_TL[rotation]]
    assert displayed_to_user_space(bbox, rotation, page_box) == (0.0, 50.0, 100.0, 100.0)
    opposite = QUADRANTS[EXPECTED_BR[rotation]]
    assert displayed_to_user_space(opposite, rotation, page_box) == (100.0, 0.0, 200.0, 50.0)


def test_displayed_to_user_space_respects_page_box_origin():
    # A page box that does not start at the origin (common after cropping).
    left, bottom, right, top = displayed_to_user_space((0.0, 0.0, 1.0, 1.0), 90, (10.0, 20.0, 110.0, 220.0))
    assert (left, bottom, right, top) == (10.0, 20.0, 110.0, 220.0)
    assert displayed_to_user_space((0.0, 0.0, 0.5, 0.5), 0, (10.0, 20.0, 110.0, 220.0)) == (10.0, 120.0, 60.0, 220.0)
