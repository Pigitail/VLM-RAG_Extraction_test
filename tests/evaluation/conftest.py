"""Fixtures for the evaluation harness tests.

* ``eval_docs`` generates the synthetic set once per session into a pytest
  temp directory, with the LibreOffice-independent docx layout so the tests
  behave the same on every machine.
* ``fake_renders`` replaces the two LibreOffice-backed steps (slide renders,
  DOCX -> PDF) with small stand-ins: the scripted and oracle models never look
  at pixels, and the end-to-end tests must not depend on LibreOffice.
* ``LIGHT`` is a low-resolution drawing geometry (same crops, labels and
  hints, far fewer pixels) used through ``EvalConfig.drawing_overrides``.
* ``chunk(...)`` builds a :class:`Chunk` for hand-computed metric tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest
from PIL import Image

from vlmrag.evaluation.documents import EvalDocument, generate_documents
from vlmrag.render.images import to_png
from vlmrag.schema import Chunk

LIGHT = {
    "page_dpi": 36,
    "page_tile_upscale": 1.0,
    "title_focus_dpi": 72,
    "title_focus_upscale": 1.0,
    "table_focus_dpi": 72,
    "table_focus_upscale": 1.0,
    "metadata_dpi": 72,
    "metadata_upscale": 1.0,
}

#: The same geometry as environment variables (for CLI tests: DrawingConfig.from_env reads them).
LIGHT_ENV = {f"VLMRAG_DRAWING_{key.upper()}": str(value) for key, value in LIGHT.items()}


@pytest.fixture(scope="session")
def eval_docs_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("eval_docs")


@pytest.fixture(scope="session")
def eval_docs(eval_docs_dir) -> list[EvalDocument]:
    return generate_documents(eval_docs_dir, ("drawing", "slides", "docx"), seed=0, docx_layout="estimated")


@pytest.fixture
def light() -> dict:
    return dict(LIGHT)


@pytest.fixture
def fake_renders(monkeypatch) -> dict[str, list]:
    """Patch slide rendering and DOCX conversion; returns what was 'rendered'."""
    from vlmrag.techniques.docx import pipeline as docx_pipeline
    from vlmrag.techniques.docx.xmlsplit import split_docx
    from vlmrag.techniques.slides import pipeline as slides_pipeline

    calls: dict[str, list] = {"slides": [], "docx": []}
    png = to_png(Image.new("RGB", (64, 36), "white"))

    def render_slides(path, deck, numbers, settings, *, dpi=180, timeout_s=60.0):
        calls["slides"].extend(numbers)
        return {n: (png, n) for n in numbers}, {}

    def convert_to_pdf(path: Path, out_dir: Path, settings: Any) -> Path:
        from reportlab.pdfgen import canvas

        calls["docx"].append(Path(path).name)
        pages = len(split_docx(path).pages)
        target = Path(out_dir) / f"{Path(path).stem}.pdf"
        pdf = canvas.Canvas(str(target), pagesize=(200, 280), invariant=1)
        for number in range(1, pages + 1):
            pdf.drawString(20, 250, f"page {number}")
            pdf.showPage()
        pdf.save()
        return target

    monkeypatch.setattr(slides_pipeline, "render_slides", render_slides)
    monkeypatch.setattr(docx_pipeline, "convert_to_pdf", convert_to_pdf)
    return calls


def make_chunk(source: str, page: int, text: str = "", *, doc_type: str = "drawing", index: int = 0,
               metadata: dict | None = None, provenance: dict | None = None) -> Chunk:
    return Chunk(
        doc_id="0" * 16,
        source_name=source,
        doc_type=doc_type,
        page=page,
        chunk_index=index,
        retrieve_content=text,
        llm_content=text,
        filtering_content="".join(text.split()),
        metadata=dict(metadata or {}),
        provenance=dict(provenance or {}),
    )


@pytest.fixture
def chunk() -> Callable[..., Chunk]:
    return make_chunk
