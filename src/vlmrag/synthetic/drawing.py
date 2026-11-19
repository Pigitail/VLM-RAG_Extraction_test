"""Synthetic A3 engineering drawings with ground truth for the ``drawing`` technique.

Three invented drawings of an invented product ("Outdoor Lantern L-300" by
"Example Manufacturing Co.") exercise every hard case of title-block reading:

* ``lantern_housing.pdf`` - two sheets. Sheet 1 is stored portrait with
  ``/Rotate 90`` (displayed landscape) and its revision table is a pasted raster
  image (no text layer); sheet 2 is an ordinary continuation sheet.
* ``lantern_reflector_scan.pdf`` - the whole title block and revision table are
  raster only, as on a scanned or flattened sheet, so the values can only be
  obtained by vision.
* ``lantern_bracket.pdf`` - vector title block except the material / finish
  rows, which are a raster "stamp".

Every sheet carries dense 4-6 pt notes, border zone markers (single letters
and digits that pollute the text layer near the title block, as on real
sheets), a bottom-right title block and a revision table to its left.

Each document gets ``<stem>.truth.json``: the expected title-block fields per
page, the printed labels, the revision-table rows, which regions have a text
layer, and 3 retrieval ``queries`` answerable only from raster content (each
with the ``evidence`` string that only a vision model can read).
:func:`truth_responder` turns truth files into a scripted model for offline
end-to-end runs (tests and ``examples/drawing_demo.py --simulate``).

Output is deterministic for a given ``seed`` (reportlab ``invariant=1``; the
seed only moves decorative geometry, never truth values).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Union

from PIL import Image, ImageDraw, ImageFont

__all__ = ["generate", "DOCUMENTS", "load_truth", "truth_responder", "TITLE_BLOCK_BOX", "REVISION_TABLE_BOX"]

PAGE_W, PAGE_H = 1190.5511811023623, 841.8897637795277  # A3 landscape, points
RASTER_DPI = 200
TITLE_BLOCK_BOX = (0.700, 0.735, 0.977, 0.968)  # normalised, top-left origin, displayed page
REVISION_TABLE_BOX = (0.450, 0.795, 0.690, 0.968)
NOTES_BOX = (0.035, 0.060, 0.400, 0.300)
LABEL_PT = 4.5
VALUE_PT = 7.0

COMPANY = "Example Manufacturing Co."
PRODUCT = "Outdoor Lantern L-300"

LABELS = {
    "company": "Company",
    "department": "Department",
    "title": "Title",
    "product": "Product",
    "drawing_number": "Drawing No.",
    "part_number": "Part No.",
    "revision": "Rev",
    "date_of_issue": "Date of Issue",
    "scale": "Scale",
    "sheet_size": "Size",
    "sheet": "Sheet",
    "document_type": "Document Type",
    "projection": "Projection",
    "material": "Material",
    "general_tolerance": "General Tolerance",
    "mass": "Mass",
    "surface_finish": "Surface Finish",
}
# Title-block grid: rows of (field key, width fraction). Rows 5-6 form the small material table.
TITLE_ROWS = (
    (("company", 0.6), ("department", 0.4)),
    (("title", 0.6), ("product", 0.4)),
    (("drawing_number", 0.4), ("part_number", 0.4), ("revision", 0.2)),
    (("date_of_issue", 0.3), ("scale", 0.2), ("sheet_size", 0.2), ("sheet", 0.3)),
    (("document_type", 0.3), ("projection", 0.3), ("material", 0.4)),
    (("general_tolerance", 0.3), ("mass", 0.2), ("surface_finish", 0.5)),
)
TABLE_FIELDS = ("material", "general_tolerance", "mass", "surface_finish")
REVISION_HEADER = ("REV", "DESCRIPTION", "DATE")
REVISION_COLUMNS = (0.12, 0.63, 0.25)

BASE_NOTES = (
    "NOTES:",
    "1. UNLESS OTHERWISE SPECIFIED ALL DIMENSIONS ARE IN MILLIMETRES.",
    "2. GENERAL TOLERANCES PER TITLE BLOCK. ANGLES +/- 0.5 DEG.",
    "3. REMOVE ALL BURRS AND BREAK SHARP EDGES 0.2 X 45 DEG.",
    "4. SURFACE FINISH: SEE TITLE BLOCK.",
    "5. DRAFT ANGLE 1 DEG MIN ON ALL VERTICAL WALLS UNLESS NOTED.",
    "6. DIMENSIONS MARKED (C) ARE CRITICAL AND 100 PCT INSPECTED.",
    "7. PART MARKING: PART NO. AND MATERIAL CODE, 2 MM CHARACTERS.",
    "8. NO SINK MARKS, FLOW LINES OR WELD LINES ON VISIBLE SURFACES.",
)

_COMMON = {
    "company": COMPANY,
    "department": "Product Engineering",
    "product": PRODUCT,
    "sheet_size": "A3",
    "projection": "First angle",
    "document_type": "Detail drawing",
}

DOCUMENTS: tuple[dict, ...] = (
    {
        "name": "lantern_housing.pdf",
        "pages": (
            {
                "rotation": 90,
                "fields": dict(
                    _COMMON,
                    title="LANTERN HOUSING",
                    drawing_number="EX-L300-1001",
                    part_number="L300-HSG-01",
                    revision="C",
                    date_of_issue="2026-03-14",
                    scale="1:2",
                    sheet="1/2",
                    material="PC-UV STABILISED",
                    general_tolerance="ISO 2768-m",
                    mass="0.42 kg",
                    surface_finish="HOUSING: MATTE BLACK | CAP: SATIN GREY",
                ),
                "revisions": (
                    ("A", "Initial release", "2025-11-02"),
                    ("B", "Vent hole diameter 3.0 to 3.2", "2026-01-20"),
                    ("C", "Surface finish changed to matte black", "2026-03-14"),
                ),
                "raster": ("revision_table",),
                "extra_notes": ("9. VENT HOLE DIA 3.2 THRU, 6 PLACES EQUALLY SPACED.", "10. GASKET SEAT FREE OF PARTING LINE."),
            },
            {
                "rotation": 0,
                "fields": dict(
                    _COMMON,
                    title="LANTERN HOUSING SECTIONS",
                    drawing_number="EX-L300-1001",
                    part_number="L300-HSG-01",
                    revision="C",
                    date_of_issue="2026-03-14",
                    scale="1:1",
                    sheet="2/2",
                    material="PC-UV STABILISED",
                    general_tolerance="ISO 2768-m",
                    mass="0.42 kg",
                    surface_finish="HOUSING: MATTE BLACK | CAP: SATIN GREY",
                ),
                "revisions": (),
                "raster": (),
                "extra_notes": ("9. SECTION A-A: NOMINAL WALL 2.4.",),
            },
        ),
        "queries": (
            ("In which revision was the vent hole diameter of the lantern housing increased to 3.2?", 1, "Vent hole diameter 3.0 to 3.2"),
            ("When was the surface finish of the lantern housing changed to matte black?", 1, "Surface finish changed to matte black"),
            ("What is the initial release date of the lantern housing drawing?", 1, "2025-11-02"),
        ),
    },
    {
        "name": "lantern_reflector_scan.pdf",
        "pages": (
            {
                "rotation": 0,
                "fields": dict(
                    _COMMON,
                    title="REFLECTOR",
                    drawing_number="EX-L300-1004",
                    part_number="L300-RFL-02",
                    revision="B",
                    date_of_issue="2026-02-09",
                    scale="1:1",
                    sheet="1/1",
                    material="ALUMINIUM 1050-H14",
                    general_tolerance="ISO 2768-f",
                    mass="0.08 kg",
                    surface_finish="REFLECTOR: BRIGHT ANODIZED",
                ),
                "revisions": (
                    ("A", "Initial release", "2025-12-01"),
                    ("B", "Reflector depth 18 to 20", "2026-02-09"),
                ),
                "raster": ("title_block", "revision_table"),
                "extra_notes": ("9. REFLECTIVE SURFACE: NO SCRATCHES OR DENTS.",),
            },
        ),
        "queries": (
            ("What material is the reflector of the Outdoor Lantern L-300 made of?", 1, "ALUMINIUM 1050-H14"),
            ("Which drawing number identifies the lantern reflector?", 1, "EX-L300-1004"),
            ("What surface finish does the lantern reflector get?", 1, "BRIGHT ANODIZED"),
        ),
    },
    {
        "name": "lantern_bracket.pdf",
        "pages": (
            {
                "rotation": 0,
                "fields": dict(
                    _COMMON,
                    title="MOUNTING BRACKET",
                    drawing_number="EX-L300-1007",
                    part_number="L300-BRK-03",
                    revision="A",
                    date_of_issue="2026-01-05",
                    scale="1:1",
                    sheet="1/1",
                    material="STAINLESS STEEL 1.4301",
                    general_tolerance="ISO 2768-m",
                    mass="0.15 kg",
                    surface_finish="BRACKET: BEAD BLASTED | FASTENERS: PASSIVATED",
                ),
                "revisions": (("A", "Initial release", "2026-01-05"),),
                "raster": ("material_table",),
                "extra_notes": ("9. BEND RADIUS 1.5 INSIDE UNLESS NOTED.",),
            },
        ),
        "queries": (
            ("Which stainless steel grade is used for the lantern mounting bracket?", 1, "STAINLESS STEEL 1.4301"),
            ("How is the surface of the lantern mounting bracket finished?", 1, "BEAD BLASTED"),
            ("What is the mass of the lantern mounting bracket?", 1, "0.15 kg"),
        ),
    },
)


# --------------------------------------------------------------------------- layout model


@dataclass(frozen=True)
class _Cell:
    box: tuple[float, float, float, float]  # normalised, top-left origin
    label: str
    value: str
    value_pt: float = VALUE_PT
    bold: bool = True


@dataclass(frozen=True)
class _Region:
    name: str
    box: tuple[float, float, float, float]
    cells: tuple[_Cell, ...]
    caption: str = ""


def _fit_size(text: str, font: str, size: float, width_pt: float) -> float:
    from reportlab.pdfbase.pdfmetrics import stringWidth

    width = stringWidth(text, font, 1.0)
    return size if width <= 0 else min(size, max(3.5, (width_pt - 6.0) / width))


def _title_regions(fields: Mapping[str, str], raster: Iterable[str]) -> list[_Region]:
    x0, y0, x1, y1 = TITLE_BLOCK_BOX
    row_h = (y1 - y0) / len(TITLE_ROWS)
    cells: list[_Cell] = []
    material_cells: list[_Cell] = []
    for r, row in enumerate(TITLE_ROWS):
        cx = x0
        for key, frac in row:
            cw = (x1 - x0) * frac
            box = (cx, y0 + r * row_h, cx + cw, y0 + (r + 1) * row_h)
            size = 9.0 if key == "title" else VALUE_PT
            size = _fit_size(fields.get(key, ""), "Helvetica-Bold", size, cw * PAGE_W)
            cell = _Cell(box, LABELS[key], fields.get(key, ""), size)
            (material_cells if r >= 4 else cells).append(cell)
            cx += cw
    raster = set(raster)
    if "material_table" in raster:
        top = (x0, y0, x1, y0 + 4 * row_h)
        bottom = (x0, y0 + 4 * row_h, x1, y1)
        return [_Region("title_block", top, tuple(cells)), _Region("material_table", bottom, tuple(material_cells))]
    return [_Region("title_block", TITLE_BLOCK_BOX, tuple(cells + material_cells))]


def _revision_region(revisions: Iterable[tuple[str, str, str]]) -> Optional[_Region]:
    rows = [REVISION_HEADER] + [tuple(r) for r in revisions]
    if len(rows) < 2:
        return None
    x0, y0, x1, y1 = REVISION_TABLE_BOX
    caption_h = 0.020
    row_h = (y1 - y0 - caption_h) / len(rows)
    cells = []
    for r, row in enumerate(rows):
        cx = x0
        for c, frac in enumerate(REVISION_COLUMNS):
            cw = (x1 - x0) * frac
            top = y0 + caption_h + r * row_h
            size = _fit_size(row[c], "Helvetica-Bold" if r == 0 else "Helvetica", 5.5, cw * PAGE_W)
            cells.append(_Cell((cx, top, cx + cw, top + row_h), "", row[c], size, bold=(r == 0)))
            cx += cw
    return _Region("revision_table", REVISION_TABLE_BOX, tuple(cells), caption="REVISIONS")


# --------------------------------------------------------------------------- vector rendering


def _xy(u: float, v: float) -> tuple[float, float]:
    """Normalised top-left coordinates -> landscape PDF points (bottom-left origin)."""
    return u * PAGE_W, (1.0 - v) * PAGE_H


def _draw_region_vector(c: Any, region: _Region) -> None:
    c.setLineWidth(0.6)
    c.setStrokeColorRGB(0, 0, 0)
    c.setFillColorRGB(0, 0, 0)
    if region.caption:
        x, y = _xy(region.box[0], region.box[1])
        c.setFont("Helvetica-Bold", 6)
        c.drawString(x + 2, y - 8, region.caption)
    for cell in region.cells:
        u0, v0, u1, v1 = cell.box
        x0, y_top = _xy(u0, v0)
        x1, y_bottom = _xy(u1, v1)
        c.rect(x0, y_bottom, x1 - x0, y_top - y_bottom, stroke=1, fill=0)
        if cell.label:
            c.setFont("Helvetica", LABEL_PT)
            c.drawString(x0 + 2, y_top - LABEL_PT - 1.5, cell.label)
        if cell.value:
            c.setFont("Helvetica-Bold" if cell.bold else "Helvetica", cell.value_pt)
            baseline = y_bottom + (y_top - y_bottom) * (0.25 if cell.label else 0.32)
            c.drawString(x0 + 3, baseline, cell.value)


def _font(px: float) -> Any:
    size = max(6, int(round(px)))
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1: fixed-size bitmap font
        return ImageFont.load_default()


def _render_region_raster(region: _Region) -> Image.Image:
    """Draw ``region`` into a grayscale bitmap (no text layer survives)."""
    rx0, ry0, rx1, ry1 = region.box
    scale = RASTER_DPI / 72.0
    width = max(1, int(round((rx1 - rx0) * PAGE_W * scale)))
    height = max(1, int(round((ry1 - ry0) * PAGE_H * scale)))
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)

    def px(u: float, v: float) -> tuple[float, float]:
        return (u - rx0) * PAGE_W * scale, (v - ry0) * PAGE_H * scale

    if region.caption:
        draw.text((2 * scale, 1 * scale), region.caption, font=_font(6 * scale), fill=0)
    for cell in region.cells:
        u0, v0, u1, v1 = cell.box
        a, b = px(u0, v0)
        c2, d = px(u1, v1)
        draw.rectangle([a, b, min(c2, width - 1), min(d, height - 1)], outline=0, width=max(1, int(scale * 0.6)))
        if cell.label:
            draw.text((a + 2 * scale, b + 1.2 * scale), cell.label, font=_font(LABEL_PT * scale), fill=0)
        if cell.value:
            cell_h = d - b
            top = b + cell_h * (0.45 if cell.label else 0.25)
            draw.text((a + 3 * scale, top), cell.value, font=_font(cell.value_pt * scale), fill=0)
    return img


def _place_raster(c: Any, region: _Region) -> None:
    from reportlab.lib.utils import ImageReader

    image = _render_region_raster(region)
    u0, v0, u1, v1 = region.box
    x0, y_top = _xy(u0, v0)
    x1, y_bottom = _xy(u1, v1)
    c.drawImage(ImageReader(image), x0, y_bottom, width=x1 - x0, height=y_top - y_bottom)


def _draw_frame(c: Any) -> None:
    c.setLineWidth(1.2)
    c.rect(14, 14, PAGE_W - 28, PAGE_H - 28)
    c.setLineWidth(0.5)
    c.rect(26, 26, PAGE_W - 52, PAGE_H - 52)
    c.setFont("Helvetica", 6)
    for i in range(8):  # zone numbers along top and bottom
        x = 26 + (PAGE_W - 52) * (i + 0.5) / 8
        c.drawCentredString(x, PAGE_H - 22, str(i + 1))
        c.drawCentredString(x, 17, str(i + 1))
    for j, letter in enumerate("ABCDEF"):  # zone letters along both sides
        y = PAGE_H - 26 - (PAGE_H - 52) * (j + 0.5) / 6
        c.drawCentredString(20, y - 2, letter)
        c.drawCentredString(PAGE_W - 20, y - 2, letter)


def _draw_geometry(c: Any, rng: random.Random) -> None:
    """Decorative part views: enough line work that tiles are not empty, varied by the seed."""
    c.setLineWidth(0.8)
    cx, cy = _xy(0.24, 0.58)
    outer = 120 + rng.randint(-8, 8)
    c.circle(cx, cy, outer)
    c.circle(cx, cy, outer * 0.55)
    holes = rng.choice((4, 6, 8))
    pitch = outer * 0.78
    for k in range(holes):
        angle = 2 * math.pi * k / holes
        c.circle(cx + pitch * math.cos(angle), cy + pitch * math.sin(angle), 5)
    c.setDash(6, 3)
    c.setLineWidth(0.4)
    c.line(cx - outer - 15, cy, cx + outer + 15, cy)
    c.line(cx, cy - outer - 15, cx, cy + outer + 15)
    c.setDash()
    sx0, sy_top = _xy(0.46, 0.36)
    sx1, sy_bottom = _xy(0.66, 0.74)
    c.setLineWidth(0.8)
    c.rect(sx0, sy_bottom, sx1 - sx0, sy_top - sy_bottom)
    c.rect(sx0 + 12, sy_bottom + 12, sx1 - sx0 - 24, sy_top - sy_bottom - 24)
    c.setFont("Helvetica", 5)
    c.drawString(cx - 12, cy + outer + 20, f"DIA {outer * 2 / 4:.0f}")
    c.drawString(sx0, sy_top + 6, f"{(sx1 - sx0) / 4:.1f}")
    c.drawString(sx1 + 4, (sy_top + sy_bottom) / 2, f"{(sy_top - sy_bottom) / 4:.1f}")
    c.drawString(sx0, sy_bottom - 10, "SECTION A-A")


def _draw_notes(c: Any, notes: Iterable[str]) -> None:
    x, y = _xy(NOTES_BOX[0], NOTES_BOX[1])
    for i, line in enumerate(notes):
        c.setFont("Helvetica-Bold" if i == 0 else "Helvetica", 6 if i == 0 else 4.5)
        c.drawString(x, y - i * 6.0, line)


def _draw_sheet(c: Any, page: Mapping[str, Any], rng: random.Random) -> None:
    raster = set(page.get("raster", ()))
    _draw_frame(c)
    _draw_geometry(c, rng)
    _draw_notes(c, BASE_NOTES + tuple(page.get("extra_notes", ())))
    regions = _title_regions(page["fields"], raster)
    revision = _revision_region(page.get("revisions", ()))
    if revision is not None:
        regions.append(revision)
    for region in regions:
        if region.name in raster:
            _place_raster(c, region)
        else:
            _draw_region_vector(c, region)


# --------------------------------------------------------------------------- documents + truth


def _write_pdf(path: Path, doc: Mapping[str, Any], seed: int) -> None:
    from reportlab.pdfgen import canvas

    rng = random.Random(f"{seed}:{doc['name']}")
    c = canvas.Canvas(str(path), pagesize=(PAGE_W, PAGE_H), invariant=1)
    c.setTitle(Path(doc["name"]).stem.replace("_", " "))
    c.setAuthor(COMPANY)
    for page in doc["pages"]:
        rotation = int(page.get("rotation", 0))
        c.setPageRotation(rotation)
        c.saveState()
        if rotation == 90:
            # reportlab swaps the MediaBox for /Rotate 90: the unrotated page is portrait, PAGE_H wide.
            # Landscape drawing coordinates (a, b) map to unrotated user space (PAGE_H - b, a), so the
            # sheet reads upright once the viewer applies the clockwise rotation.
            c.translate(PAGE_H, 0)
            c.rotate(90)
        elif rotation != 0:
            raise ValueError("the generator only produces /Rotate 0 or 90 pages")
        _draw_sheet(c, page, rng)
        c.restoreState()
        c.showPage()
    c.save()


def _table_rows(revisions: Iterable[Iterable[str]]) -> list[list[str]]:
    rows = [list(r) for r in revisions]
    return [list(REVISION_HEADER)] + rows if rows else []


def _truth(doc: Mapping[str, Any], seed: int) -> dict:
    pages = []
    for number, page in enumerate(doc["pages"], 1):
        raster = set(page.get("raster", ()))
        fields = dict(page["fields"])
        pages.append(
            {
                "page": number,
                "rotation": int(page.get("rotation", 0)),
                "displayed_size_pt": [round(PAGE_W, 2), round(PAGE_H, 2)],
                "title_block": fields,
                "title_block_labels": {k: LABELS[k] for k in fields},
                "table_fields": [k for k in TABLE_FIELDS if k in fields],
                "table_rows": _table_rows(page.get("revisions", ())),
                "raster_regions": sorted(raster),
                "text_layer": {
                    "title_block": "title_block" not in raster,
                    "material_table": not ({"title_block", "material_table"} & raster),
                    "revision_table": bool(page.get("revisions")) and "revision_table" not in raster,
                    "notes": True,
                },
                "notes": list(BASE_NOTES + tuple(page.get("extra_notes", ()))),
            }
        )
    return {
        "generator": "vlmrag.synthetic.drawing",
        "version": 1,
        "seed": seed,
        "source": doc["name"],
        "doc_type": "drawing",
        "page_count": len(pages),
        "boxes": {"title_block": list(TITLE_BLOCK_BOX), "revision_table": list(REVISION_TABLE_BOX)},
        "pages": pages,
        # "evidence" is a string printed only in raster content of the relevant page: the query
        # cannot be answered from the text layer, only from what a vision model reads.
        "queries": [
            {"query": query, "relevant": [{"source": doc["name"], "page": page}], "evidence": evidence}
            for query, page, evidence in doc["queries"]
        ],
    }


def generate(out_dir: Union[str, Path], *, seed: int = 0) -> list[Path]:
    """Write the synthetic drawings and their ``<stem>.truth.json`` into ``out_dir``.

    Returns the document paths (each has a truth file next to it).
    """
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for doc in DOCUMENTS:
        pdf_path = target / doc["name"]
        _write_pdf(pdf_path, doc, seed)
        truth_path = pdf_path.with_name(pdf_path.stem + ".truth.json")
        truth_path.write_text(json.dumps(_truth(doc, seed), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(pdf_path)
    return written


def load_truth(document: Union[str, Path]) -> dict:
    """Truth of a generated document (pass the PDF path or the truth path)."""
    path = Path(document)
    if not path.name.endswith(".truth.json"):
        path = path.with_name(path.stem + ".truth.json")
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- scripted model


def _vision_answer(page: Mapping[str, Any]) -> dict:
    fields = page["title_block"]
    labels = page["title_block_labels"]
    return {"pairs": [{"topic": labels[k], "value": v} for k, v in fields.items()], **fields}


def _table_answer(page: Mapping[str, Any]) -> dict:
    fields = page["title_block"]
    labels = page["title_block_labels"]
    table_keys = page.get("table_fields", [])
    rows = [
        {"row_index": i, "cells": [{"col_index": j, "text": text} for j, text in enumerate(row)]}
        for i, row in enumerate(page.get("table_rows", []))
    ]
    pairs = [{"topic": labels[k], "value": fields[k]} for k in table_keys]
    return {"table_rows": rows, "pairs": pairs, **{k: fields[k] for k in table_keys}}


def _normalize_answer(page: Mapping[str, Any]) -> dict:
    fields = page["title_block"]
    return {"pairs": [{"topic": k.replace("_", " ").title(), "value": v} for k, v in fields.items()], **fields}


def truth_responder(truths: Union[Mapping[str, Any], Iterable[Mapping[str, Any]]]) -> Callable[[Any], dict]:
    """A ``ScriptedBackend`` responder that answers every drawing pass from ground truth.

    It plays a perfect model: the vision pass returns every printed field (also
    for raster-only regions, which only vision can read), the table pass returns
    the revision-table rows plus the material-table fields, and the normalize
    pass returns canonical labels. Useful to run the whole pipeline offline and
    to check that chunks carry what the model read.
    """
    if isinstance(truths, Mapping):
        truths = [truths]
    by_source = {t["source"]: t for t in truths}
    answers = {"drawing.vision": _vision_answer, "drawing.table": _table_answer, "drawing.normalize": _normalize_answer}

    def respond(request: Any) -> dict:
        truth = by_source.get(request.source_name)
        if truth is None:
            raise KeyError(f"no truth for source {request.source_name!r}")
        if request.task not in answers:
            raise KeyError(f"unexpected task {request.task!r}")
        page = truth["pages"][int(request.page) - 1]
        return answers[request.task](page)

    return respond
