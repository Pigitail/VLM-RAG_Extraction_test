"""Fixtures for the docx technique tests.

This directory deliberately has no ``__init__.py``: with pytest's default
import mode a package here would be imported as the top-level module ``docx``
and shadow the python-docx distribution for the whole session. Test modules
use a ``test_docx_`` prefix instead, which keeps their names unique.
Helpers are exposed as fixtures so they work under every import mode.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"


def _wxml(body: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{W_NS}" xmlns:mc="{MC_NS}"><w:body>{body}</w:body></w:document>'
    ).encode("utf-8")


@pytest.fixture
def wxml() -> Callable[[str], bytes]:
    """``wxml(body)`` -> a complete ``word/document.xml`` with the usual namespaces."""
    return _wxml


@pytest.fixture
def make_docx(tmp_path: Path) -> Callable[..., Path]:
    """``make_docx(body, pages=None, name="doc.docx")`` -> a minimal DOCX package on disk."""
    import zipfile

    def factory(body: str, pages: Optional[int] = None, name: str = "doc.docx") -> Path:
        path = tmp_path / name
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("word/document.xml", _wxml(body))
            if pages is not None:
                archive.writestr(
                    "docProps/app.xml",
                    '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
                    f"<Pages>{pages}</Pages></Properties>",
                )
        return path

    return factory


@pytest.fixture(scope="session")
def synthetic_docx(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    """The synthetic specification with the estimated (LibreOffice-independent) layout, plus its truth."""
    from vlmrag.synthetic.docx import generate

    out = tmp_path_factory.mktemp("docx_synthetic")
    [path] = generate(out, seed=0, use_libreoffice=False)
    truth = json.loads((out / f"{path.stem}.truth.json").read_text(encoding="utf-8"))
    return path, truth


def _table_answer(truth: dict) -> dict:
    table = truth["tables"][0]
    pages = range(table["start_page"], table["end_page"] + 1)
    return {
        "has_title": True,
        "title_text": table["title"],
        "title_source": table["title_source"],
        "title_confidence": "high",
        "row_distribution": {str(p): [i for i, rp in enumerate(table["row_pages"]) if rp == p] for p in pages},
    }


def _image_answer(truth: dict, page: Optional[int]) -> dict:
    return {
        "image_descriptions": [
            {"location_hint": image["location_hint"], "description": image["description"]}
            for image in truth["images"]
            if image["page"] == page
        ],
        "notes": "",
    }


@pytest.fixture
def truth_answers() -> dict[str, Callable[..., Any]]:
    """``{"table": f(truth), "image": f(truth, page)}``: correct answers derived from a truth file."""
    return {"table": _table_answer, "image": _image_answer}


@pytest.fixture
def truth_by_task() -> Callable[[dict], dict]:
    """``truth_by_task(truth)`` -> a ``by_task`` table for ``scripted_client`` answering from the truth."""

    def factory(truth: dict) -> dict:
        return {
            "docx.image_description": lambda request: _image_answer(truth, request.page),
            "docx.table_distribution": lambda request: _table_answer(truth),
        }

    return factory


@pytest.fixture
def fake_pdf(monkeypatch: pytest.MonkeyPatch) -> Callable[[int], list]:
    """Replace the LibreOffice conversion with a reportlab PDF of ``page_count`` A4 pages.

    Returns a list that records every conversion call, so tests can assert
    whether the PDF phase ran. Pixels do not matter here: scripted answers are
    keyed by page number.
    """
    from vlmrag.techniques.docx import pipeline

    calls: list = []

    def install(page_count: int) -> list:
        def fake_convert(path: Path, out_dir: Path, settings: Any) -> Path:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas

            calls.append(Path(path).name)
            target = Path(out_dir) / f"{Path(path).stem}.pdf"
            pdf = canvas.Canvas(str(target), pagesize=A4, invariant=1)
            for number in range(1, page_count + 1):
                pdf.drawString(72, 760, f"rendered page {number}")
                pdf.rect(72, 300, 300, 300)
                pdf.showPage()
            pdf.save()
            return target

        monkeypatch.setattr(pipeline, "convert_to_pdf", fake_convert)
        return calls

    return install
