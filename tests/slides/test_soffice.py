"""Real LibreOffice rendering (skipped when soffice is not installed)."""

from __future__ import annotations

import io

import pytest
from PIL import Image
from pptx.util import Inches

from vlmrag.config import Settings
from vlmrag.synthetic.slides import new_presentation
from vlmrag.techniques.slides import TASK, extract
from vlmrag.vlm.types import ImageInput

pytestmark = pytest.mark.soffice


def _image_of(request) -> Image.Image:
    (image,) = [p for p in request.parts if isinstance(p, ImageInput)]
    return Image.open(io.BytesIO(image.png)).convert("RGB")


def test_real_render_end_to_end(deck_path, truth, scripted_client, truth_responder, soffice_path):
    client = scripted_client(by_task={TASK: truth_responder(truth)})
    chunks = {c.page: c for c in extract(deck_path, client, Settings(provider="null", soffice=soffice_path))}
    assert sorted(r.page for r in client.backend.calls) == [2, 3, 4, 5]
    for request in client.backend.calls:
        image = _image_of(request)
        # 13.333 x 7.5 in at 180 dpi
        assert image.size[0] == pytest.approx(2400, abs=3) and image.size[1] == pytest.approx(1350, abs=3)
        assert chunks[request.page].provenance["render"] == {"dpi": 180, "converter": "libreoffice",
                                                             "pdf_page": request.page}
    for slide in truth["slides"]:
        for fact in slide["must_mention"]:
            assert fact in chunks[slide["slide"]].retrieve_content


def _solid(color) -> io.BytesIO:
    buffer = io.BytesIO()
    Image.new("RGB", (160, 90), color).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_hidden_slide_does_not_shift_renders(tmp_path, scripted_client, soffice_path):
    prs = new_presentation()
    colors = {1: (0, 0, 255), 2: (255, 0, 0), 3: (0, 160, 0)}
    for number, color in colors.items():
        slide = prs.slides.add_slide(prs.slide_layouts[0])
        slide.shapes.title.text = f"Slide {number}"
        slide.shapes.add_picture(_solid(color), Inches(0), Inches(1.2), Inches(13.3), Inches(6.3))
    prs.slides[1]._element.set("show", "0")
    path = tmp_path / "hidden.pptx"
    prs.save(str(path))

    seen = {}

    def respond(request):
        image = _image_of(request)
        seen[request.page] = image.getpixel((image.width // 2, int(image.height * 0.6)))
        return {"visual_summary": "", "image_descriptions": [], "flow_notes": [], "notes": ""}

    chunks = {c.page: c for c in extract(path, scripted_client(by_task={TASK: respond}),
                                         Settings(provider="null", soffice=soffice_path))}
    assert set(chunks) == {1, 2, 3}
    for page, pixel in seen.items():
        expected = colors[page]
        assert all(abs(a - b) < 40 for a, b in zip(pixel, expected)), (page, pixel)
    assert 3 in seen and 1 in seen
    if 2 not in seen:  # LibreOffice skipped the hidden slide in the PDF
        assert chunks[2].provenance["fallbacks"] == ["hidden_slide"]
    assert chunks[2].metadata["hidden"] is True
