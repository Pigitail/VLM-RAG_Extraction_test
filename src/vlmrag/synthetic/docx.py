"""Synthetic specification DOCX with ground truth for the ``docx`` technique.

Writes ``lantern_spec.docx`` (a 4-5 page product specification of the
invented "Outdoor Lantern L-300" by the invented "Example Manufacturing Co.")
and ``lantern_spec.truth.json`` next to it.

What the document exercises
    * headings (``Title`` and ``heading 1`` styles), body paragraphs and two
      manual page breaks;
    * three pictures drawn with PIL (an exploded-view diagram, a block diagram,
      a photo-like render) whose key facts exist *only* in the pixels, so a
      retrieval query about them can only be answered from the image pass;
    * a 44-row parts table that crosses a page boundary;
    * ``<w:lastRenderedPageBreak/>`` markers at the points where the document
      really breaks, and a matching ``docProps/app.xml`` ``<Pages>``.

Where the page breaks come from
    A paginating word processor writes rendered break markers when it saves.
    The generator imitates that: it writes a draft, lets LibreOffice render it
    to PDF (when installed), finds on which PDF page every paragraph and table
    row starts (each one carries a unique text key) and then writes the final
    file with a marker at the start of every block that begins a new page.
    Like a word processor, it also records a rendered break at the top of a
    page that a manual page break started. LibreOffice ignores these markers
    when it lays a document out, so the final file renders exactly like the
    draft. Without LibreOffice the same placement is estimated from a simple
    height model; ``truth["layout_source"]`` says which one was used.

The output is deterministic for a given ``seed`` (and, with LibreOffice, for
a given LibreOffice installation): fixed zip timestamps, seeded content and
seeded image noise. ``truth["anchor_pages"]`` records the page of every
paragraph and table row; ``generate(..., layout=anchor_pages)`` rebuilds the
same file on a machine with a different LibreOffice (or none), which is what
portable record/replay of model answers needs (see ``docs/evaluation.md``).
"""

from __future__ import annotations

import json
import logging
import math
import random
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union
from xml.sax.saxutils import escape, quoteattr

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..render.images import to_png
from ..text import filtering_text, normalize_unicode

__all__ = ["generate", "STEM", "build_document"]

log = logging.getLogger(__name__)

STEM = "lantern_spec"
TABLE_TITLE = "Table 1. Parts list for the L-300 assembly."
TABLE_HEADER = ["Item", "Part number", "Description", "Material", "Qty"]
TABLE_DATA_ROWS = 43

PAGE_W, PAGE_H, MARGIN = 11906, 16838, 1440  # A4 in twips, 1 inch margins
COLUMN_WIDTHS = [700, 1700, 3326, 1800, 1500]  # twips; sum = text width (9026)
EMU_PER_INCH = 914400
ZIP_TIMESTAMP = (2024, 1, 15, 9, 0, 0)
LO_TIMEOUT_S = 120

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"

_PART_NAMES = [
    "Carry handle", "Handle pivot pin", "Top cap", "Top cap gasket", "Glass globe", "Globe retaining ring",
    "Light diffuser", "Reflector cone", "LED module", "LED heat spreader", "LED driver board", "Driver board spacer",
    "Charge controller board", "USB-C port seal", "Battery pack", "Battery holder", "Battery cover",
    "Battery cover latch", "Base housing", "Rubber foot", "Switch knob", "Switch spring", "Switch shaft",
    "Status light pipe", "Status ring lens", "Wire harness, main", "Wire harness, LED", "Screw M3 x 8",
    "Screw M2.5 x 6", "Screw M2 x 4", "Washer M3", "Hanging hook", "Hook spring clip", "Desiccant sachet",
    "Rating label", "Warning label", "Foam pad", "Inner tray", "Outer carton", "Pulp insert",
    "User leaflet", "Warranty card", "Polybag",
]
_MATERIALS = ["ABS", "PC", "Glass", "Aluminium", "Steel", "Silicone", "FR-4", "Copper", "Paper", "PE"]


# --------------------------------------------------------------------------- document model


@dataclass
class _Para:
    text: str = ""
    style: str = ""
    image: Optional[str] = None
    keep_next: bool = False
    page_break_after: bool = False
    page: int = 0
    marker: bool = False

    @property
    def key(self) -> str:
        return self.text


@dataclass
class _Row:
    cells: list[str]
    header: bool = False
    key: str = ""
    page: int = 0
    marker: bool = False


@dataclass
class _Table:
    rows: list[_Row]


@dataclass
class _Figure:
    figure_id: str
    file_name: str
    png: bytes
    width_in: float
    height_in: float
    location_hint: str
    description: str
    must_mention: list[str]


@dataclass
class _Document:
    body: list[Union[_Para, _Table]]
    figures: dict[str, _Figure]
    facts: dict[str, str] = field(default_factory=dict)

    def anchors(self) -> list[Union[_Para, _Row]]:
        items: list[Union[_Para, _Row]] = []
        for item in self.body:
            items.extend(item.rows if isinstance(item, _Table) else [item])
        return items


# --------------------------------------------------------------------------- pictures


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)  # scalable font, Pillow >= 10.1 built with FreeType
    except (TypeError, ImportError, OSError):  # older Pillow, or a build without FreeType
        return ImageFont.load_default()


def _centered_text(d: ImageDraw.ImageDraw, centre: tuple[float, float], text: str, font) -> None:
    # Computed by hand instead of anchor="mm", which bitmap fonts do not support.
    left, top, right, bottom = d.multiline_textbbox((0, 0), text, font=font, align="center")
    x = centre[0] - (right - left) / 2 - left
    y = centre[1] - (bottom - top) / 2 - top
    d.multiline_text((x, y), text, font=font, fill="black", align="center")


def _exploded_view() -> Image.Image:
    img = Image.new("RGB", (1000, 720), "white")
    d = ImageDraw.Draw(img)
    cx = 330
    for y in range(30, 700, 18):  # dashed assembly axis
        d.line([(cx, y), (cx, y + 9)], fill=(150, 150, 150), width=2)
    parts = []
    d.arc([cx - 90, 40, cx + 90, 170], start=180, end=360, fill=(60, 60, 60), width=12)
    parts.append(("HANDLE", 70))
    d.polygon([(cx - 110, 230), (cx + 110, 230), (cx + 80, 180), (cx - 80, 180)], fill=(90, 90, 95), outline="black")
    parts.append(("TOP CAP", 205))
    d.rounded_rectangle([cx - 120, 260, cx + 120, 430], radius=40, fill=(205, 228, 245), outline=(40, 80, 120), width=4)
    for x in (cx - 60, cx, cx + 60):
        d.line([(x, 280), (x, 410)], fill=(170, 200, 225), width=3)
    parts.append(("GLASS GLOBE", 345))
    d.rectangle([cx - 85, 470, cx + 85, 510], fill=(40, 120, 60), outline="black")
    for x in range(cx - 60, cx + 61, 30):
        d.ellipse([x - 9, 481, x + 9, 499], fill=(250, 230, 90), outline="black")
    parts.append(("LED MODULE", 490))
    d.rounded_rectangle([cx - 115, 560, cx + 115, 650], radius=14, fill=(60, 62, 70), outline="black", width=3)
    d.rectangle([cx - 40, 595, cx + 40, 615], fill=(120, 120, 128))
    parts.append(("BATTERY COVER", 605))
    font = _font(30)
    for label, y in parts:
        d.line([(cx + 130, y), (590, y)], fill="black", width=2)
        d.ellipse([cx + 124, y - 6, cx + 136, y + 6], fill="black")
        d.text((605, y - 18), label, font=font, fill="black")
    d.text((20, 690), "EXPLODED VIEW - NOT TO SCALE", font=_font(18), fill=(90, 90, 90))
    return img


def _power_path() -> Image.Image:
    img = Image.new("RGB", (1240, 380), "white")
    d = ImageDraw.Draw(img)
    labels = ["USB-C\nINPUT", "CHARGE\nCONTROLLER", "BATTERY\nPACK", "LED\nDRIVER", "LED\nMODULE"]
    font = _font(26)
    box_w, box_h, gap, top = 196, 130, 60, 160
    x = 20
    centres = []
    for i, label in enumerate(labels):
        fill = (232, 240, 252) if i != 2 else (236, 248, 232)
        d.rounded_rectangle([x, top, x + box_w, top + box_h], radius=16, fill=fill, outline=(30, 60, 110), width=4)
        _centered_text(d, (x + box_w / 2, top + box_h / 2), label, font)
        centres.append(x)
        x += box_w + gap
    for i in range(len(labels) - 1):
        x0 = centres[i] + box_w + 6
        x1 = centres[i + 1] - 6
        y = top + box_h / 2
        d.line([(x0, y), (x1 - 14, y)], fill="black", width=5)
        d.polygon([(x1, y), (x1 - 18, y - 11), (x1 - 18, y + 11)], fill="black")
    d.text((centres[0] + box_w - 40, top - 70), "MAX 5 V / 2 A", font=_font(28), fill=(170, 30, 30))
    d.line([(centres[0] + box_w + 30, top - 30), (centres[0] + box_w + 30, top + box_h / 2 - 8)], fill=(170, 30, 30), width=3)
    d.text((20, 20), "POWER PATH", font=_font(30), fill=(30, 60, 110))
    return img


def _product_photo(rng: random.Random) -> Image.Image:
    w, h = 900, 700
    noise = Image.new("L", (90, 70))
    noise.putdata([rng.randrange(0, 256) for _ in range(90 * 70)])
    noise = noise.resize((w, h), Image.BICUBIC)
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    for y in range(h):  # dusk sky fading into a lawn
        t = y / (h - 1)
        if t < 0.72:
            col = (int(28 + 40 * t), int(36 + 50 * t), int(70 + 60 * t))
        else:
            col = (int(30 + 10 * t), int(70 + 20 * t), int(35 + 5 * t))
        d.line([(0, y), (w, y)], fill=col)
    img = Image.blend(img, Image.merge("RGB", (noise, noise, noise)), 0.08)
    glow = Image.new("RGB", (w, h), "black")
    ImageDraw.Draw(glow).ellipse([270, 150, 630, 520], fill=(255, 214, 140))
    glow = glow.filter(ImageFilter.GaussianBlur(60))
    img = Image.blend(img, Image.composite(glow, img, glow.convert("L")), 0.55)
    d = ImageDraw.Draw(img)
    cx = 450
    d.ellipse([cx - 150, 600, cx + 150, 640], fill=(20, 45, 25))  # ground shadow
    d.arc([cx - 45, 40, cx + 45, 150], start=180, end=360, fill=(170, 170, 175), width=10)  # carry hook
    d.line([(cx, 40), (cx, 10)], fill=(170, 170, 175), width=6)
    d.polygon([(cx - 105, 215), (cx + 105, 215), (cx + 70, 160), (cx - 70, 160)], fill=(45, 45, 50))
    d.rounded_rectangle([cx - 115, 215, cx + 115, 470], radius=46, fill=(255, 238, 200), outline=(200, 170, 120), width=3)
    d.ellipse([cx - 55, 290, cx + 55, 400], fill=(255, 250, 235))
    d.rectangle([cx - 122, 470, cx + 122, 498], fill=(40, 185, 75))  # green status ring
    d.rounded_rectangle([cx - 125, 498, cx + 125, 600], radius=18, fill=(55, 57, 64))
    d.ellipse([cx - 14, 535, cx + 14, 563], fill=(90, 92, 100))
    return img.filter(ImageFilter.GaussianBlur(1.2))


def _figures(rng: random.Random) -> dict[str, _Figure]:
    figures = {
        "fig1": _Figure(
            "fig1", "fig1.png", to_png(_exploded_view()), 5.0, 3.6, "centre diagram",
            "Exploded view of the lantern along a dashed vertical axis. From top to bottom the labelled parts are: "
            "handle, top cap, glass globe, LED module and battery cover; the LED module sits directly below the "
            "glass globe and the battery cover is the lowest part.",
            ["glass globe", "LED module", "battery cover"],
        ),
        "fig2": _Figure(
            "fig2", "fig2.png", to_png(_power_path()), 6.2, 1.9, "centre block diagram",
            "Block diagram titled POWER PATH: USB-C input -> charge controller -> battery pack -> LED driver -> "
            "LED module, connected by arrows from left to right. A red label 'MAX 5 V / 2 A' marks the USB-C input.",
            ["charge controller", "5 V / 2 A", "LED driver"],
        ),
        "fig3": _Figure(
            "fig3", "fig3.png", to_png(_product_photo(rng)), 4.6, 3.58, "centre photograph",
            "Photograph of the finished lantern at dusk on a lawn, hanging from its carry hook and glowing warm "
            "white, with a green status ring at the base of the glass globe and a dark grey base.",
            ["green status ring", "carry hook"],
        ),
    }
    return figures


# --------------------------------------------------------------------------- content


def build_document(seed: int = 0) -> _Document:
    """The document content for ``seed`` (pages are not assigned yet)."""
    rng = random.Random(seed)
    capacity = rng.choice([4400, 5200, 6000])
    base = rng.randrange(100, 500)
    rows = [_Row(list(TABLE_HEADER), header=True, key="Partnumber")]
    for i in range(TABLE_DATA_ROWS):
        part_number = f"L300-{base + i:03d}"
        rows.append(_Row(
            [str(i + 1), part_number, _PART_NAMES[i], rng.choice(_MATERIALS), str(rng.randint(1, 4))],
            key=part_number,
        ))
    figures = _figures(rng)
    body: list[Union[_Para, _Table]] = [
        _Para("Outdoor Lantern L-300 - Product Specification", style="Title"),
        _Para(
            "Document number SPEC-L300-01, revision C. Prepared by Example Manufacturing Co. This specification "
            "describes the construction, electrical characteristics, parts and packaging of the Outdoor Lantern L-300."
        ),
        _Para("1 Scope", style="Heading1"),
        _Para(
            "This document applies to all production lots of the Outdoor Lantern L-300. It defines the requirements "
            "that incoming inspection and final assembly verify. Values are nominal unless a tolerance is given."
        ),
        _Para("2 Product overview", style="Heading1"),
        _Para(
            "The lantern is a portable, rechargeable light for camping and garden use. Figure 1 shows how the main "
            "assemblies are stacked along the vertical axis of the product."
        ),
        _Para(image="fig1", keep_next=True),
        _Para("Figure 1. Exploded view of the lantern assembly.", style="Caption", page_break_after=True),
        _Para("3 Electrical characteristics", style="Heading1"),
        _Para(
            f"The lantern runs from a rechargeable lithium-ion battery pack with a nominal voltage of 3.7 V and a "
            f"capacity of {capacity} mAh. Light output is adjustable in three steps with the switch knob."
        ),
        _Para("Figure 2 shows the power path from the charging port to the light source."),
        _Para(image="fig2", keep_next=True),
        _Para("Figure 2. Power path.", style="Caption"),
        _Para("4 Parts list", style="Heading1"),
        _Para(TABLE_TITLE, keep_next=True),
        _Table(rows),
        _Para(
            "All parts shall be free of sharp edges and burrs. Fasteners are supplied pre-counted in one kit per "
            "assembly.",
            page_break_after=True,
        ),
        _Para("5 Finish and packaging", style="Heading1"),
        _Para(
            "Each lantern is inspected for finish defects before packing. Figure 3 shows a finished unit as it is "
            "presented on the packaging."
        ),
        _Para(image="fig3", keep_next=True),
        _Para("Figure 3. Product photograph.", style="Caption"),
        _Para(
            "The unit is packed in a recycled carton with a moulded pulp insert. The carton carries the rating label "
            "and the recycling mark."
        ),
    ]
    return _Document(body=body, figures=figures, facts={"capacity_mah": str(capacity)})


# --------------------------------------------------------------------------- page assignment


def _assign_from_pdf(doc: _Document, page_texts: list[str]) -> bool:
    """Find the PDF page on which every keyed block starts; ``False`` if the layout is not usable."""
    texts = [filtering_text(normalize_unicode(t)) for t in page_texts]
    page, pos = 0, 0
    anchors = doc.anchors()
    for anchor in anchors:
        if not anchor.key:
            continue
        key = filtering_text(normalize_unicode(anchor.key))[:40]
        found = None
        for p in range(page, len(texts)):
            idx = texts[p].find(key, pos if p == page else 0)
            if idx >= 0:
                found = (p, idx)
                break
        if found is None:
            log.info("layout key %r not found in the LibreOffice render", key)
            return False
        page, pos = found[0], found[1] + len(key)
        anchor.page = page + 1
    _fill_unkeyed(anchors)
    numbers = [a.page for a in anchors]
    if numbers[0] != 1 or any(b - a > 1 for a, b in zip(numbers, numbers[1:])) or numbers[-1] != len(texts):
        log.info("LibreOffice layout has gaps or trailing pages; using the estimated layout")
        return False
    return True


def _fill_unkeyed(anchors: list) -> None:
    # Picture paragraphs carry no text; keep-with-next puts them on their caption's page.
    following = 0
    for anchor in reversed(anchors):
        if anchor.page:
            following = anchor.page
        else:
            anchor.page = following


def _estimate_layout(doc: _Document) -> None:
    """Height-model page assignment used when LibreOffice is not installed."""
    usable = (PAGE_H - 2 * MARGIN) / 1440.0
    line = 0.19
    page, y = 1, 0.0

    def height(item: Union[_Para, _Row]) -> float:
        if isinstance(item, _Row):
            return 0.24
        if item.image:
            return doc.figures[item.image].height_in + 0.15
        if item.style == "Title":
            return 0.75
        if item.style == "Heading1":
            return 0.55
        if item.style == "Caption":
            return 0.35
        return math.ceil(len(item.text) / 95) * line + 0.12

    anchors = doc.anchors()
    for i, item in enumerate(anchors):
        h = height(item)
        if isinstance(item, _Para) and item.keep_next and i + 1 < len(anchors):
            h_block = h + height(anchors[i + 1])
        else:
            h_block = h
        if y > 0 and y + h_block > usable:
            page, y = page + 1, 0.0
        item.page = page
        y += h
        if isinstance(item, _Para) and item.page_break_after:
            page, y = page + 1, line  # the paragraph's empty remainder starts the next page


def _apply_layout(doc: _Document, anchor_pages: Sequence[int]) -> None:
    """Assign the pages of a previously generated layout (``truth["anchor_pages"]``).

    Exists for portable record/replay: a LibreOffice layout depends on the
    LibreOffice version and its fonts, so a recording made on one machine is
    replayed against documents regenerated with *the recorded* layout instead
    of whatever the replaying machine would compute. The pages must describe
    a real pagination: start at 1, never decrease and never skip a page.
    """
    anchors = doc.anchors()
    pages = [int(p) for p in anchor_pages]
    if len(pages) != len(anchors):
        raise ValueError(f"layout has {len(pages)} anchor page(s), the document has {len(anchors)} anchor(s)")
    if not pages or pages[0] != 1 or any(b - a not in (0, 1) for a, b in zip(pages, pages[1:])):
        raise ValueError("layout anchor pages must start at 1 and grow by 0 or 1 from anchor to anchor")
    for anchor, page in zip(anchors, pages):
        anchor.page = page


def _set_markers(doc: _Document) -> int:
    previous = 1
    for anchor in doc.anchors():
        anchor.marker = anchor.page > previous
        previous = anchor.page
    return previous


# --------------------------------------------------------------------------- OOXML writer


def _run_text(text: str, *, bold: bool = False, marker: bool = False) -> str:
    rpr = "<w:rPr><w:b/></w:rPr>" if bold else ""
    mark = "<w:lastRenderedPageBreak/>" if marker else ""
    return f'<w:r>{rpr}{mark}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def _drawing(fig: _Figure, rel_id: str, number: int) -> str:
    cx = int(fig.width_in * EMU_PER_INCH)
    cy = int(fig.height_in * EMU_PER_INCH)
    return (
        f'<w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
        f'<wp:effectExtent l="0" t="0" r="0" b="0"/><wp:docPr id="{number}" name="Picture {number}"/>'
        '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        f'<a:graphic><a:graphicData uri="{PIC_NS}"><pic:pic><pic:nvPicPr><pic:cNvPr id="{number}" '
        f'name={quoteattr(fig.file_name)}/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="{rel_id}"/>'
        '<a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/>'
        f'<a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        "</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing>"
    )


def _paragraph_xml(para: _Para, doc: _Document, rel_ids: dict[str, str], numbers: dict[str, int]) -> str:
    ppr = []
    if para.style:
        ppr.append(f'<w:pStyle w:val="{para.style}"/>')
    if para.keep_next:
        ppr.append("<w:keepNext/>")
    ppr.append("<w:keepLines/>")  # a paragraph never straddles pages, so one marker per page start suffices
    if para.image:
        ppr.append('<w:jc w:val="center"/>')
    runs = ""
    if para.image:
        mark = "<w:lastRenderedPageBreak/>" if para.marker else ""
        fig = doc.figures[para.image]
        runs = f"<w:r>{mark}{_drawing(fig, rel_ids[para.image], numbers[para.image])}</w:r>"
    elif para.text:
        runs = _run_text(para.text, marker=para.marker)
    if para.page_break_after:
        runs += '<w:r><w:br w:type="page"/></w:r>'
    return f"<w:p><w:pPr>{''.join(ppr)}</w:pPr>{runs}</w:p>"


def _table_xml(table: _Table) -> str:
    border = 'w:val="single" w:sz="4" w:space="0" w:color="404040"'
    borders = "".join(f"<w:{side} {border}/>" for side in ("top", "left", "bottom", "right", "insideH", "insideV"))
    parts = [
        f'<w:tbl><w:tblPr><w:tblW w:w="{sum(COLUMN_WIDTHS)}" w:type="dxa"/><w:tblBorders>{borders}</w:tblBorders>'
        '<w:tblLayout w:type="fixed"/><w:tblLook w:val="04A0"/></w:tblPr><w:tblGrid>',
        "".join(f'<w:gridCol w:w="{w}"/>' for w in COLUMN_WIDTHS),
        "</w:tblGrid>",
    ]
    for row in table.rows:
        cells = []
        for col, (value, width) in enumerate(zip(row.cells, COLUMN_WIDTHS)):
            shade = '<w:shd w:val="clear" w:color="auto" w:fill="D9D9D9"/>' if row.header else ""
            run = _run_text(value, bold=row.header, marker=row.marker and col == 0)
            cells.append(
                f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{shade}</w:tcPr>'
                f'<w:p><w:pPr><w:spacing w:before="20" w:after="20"/></w:pPr>{run}</w:p></w:tc>'
            )
        parts.append(f"<w:tr><w:trPr><w:cantSplit/></w:trPr>{''.join(cells)}</w:tr>")
    parts.append("</w:tbl>")
    return "".join(parts)


def _document_xml(doc: _Document, rel_ids: dict[str, str]) -> str:
    numbers = {fid: i + 1 for i, fid in enumerate(sorted(doc.figures))}
    body = []
    for item in doc.body:
        body.append(_table_xml(item) if isinstance(item, _Table) else _paragraph_xml(item, doc, rel_ids, numbers))
    sect = (
        f'<w:sectPr><w:pgSz w:w="{PAGE_W}" w:h="{PAGE_H}"/><w:pgMar w:top="{MARGIN}" w:right="{MARGIN}" '
        f'w:bottom="{MARGIN}" w:left="{MARGIN}" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<w:document xmlns:w="{W_NS}" xmlns:r="{R_NS}" xmlns:wp="{WP_NS}" xmlns:a="{A_NS}" xmlns:pic="{PIC_NS}">'
        f"<w:body>{''.join(body)}{sect}</w:body></w:document>"
    )


_STYLES_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    f'<w:styles xmlns:w="{W_NS}">'
    '<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="Arial" w:cs="Arial"/>'
    '<w:sz w:val="20"/><w:szCs w:val="20"/><w:lang w:val="en-US"/></w:rPr></w:rPrDefault>'
    '<w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="264" w:lineRule="auto"/></w:pPr></w:pPrDefault>'
    "</w:docDefaults>"
    '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>'
    '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/>'
    '<w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:after="240"/></w:pPr>'
    '<w:rPr><w:b/><w:sz w:val="36"/><w:szCs w:val="36"/></w:rPr></w:style>'
    '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/>'
    '<w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before="240" w:after="120"/>'
    '<w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:b/><w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr></w:style>'
    '<w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="caption"/><w:basedOn w:val="Normal"/>'
    '<w:qFormat/><w:pPr><w:spacing w:after="200"/><w:jc w:val="center"/></w:pPr>'
    '<w:rPr><w:i/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr></w:style>'
    "</w:styles>"
)

_SETTINGS_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    f'<w:settings xmlns:w="{W_NS}"><w:defaultTabStop w:val="720"/><w:compat><w:compatSetting '
    'w:name="compatibilityMode" w:uri="http://schemas.microsoft.com/office/word" w:val="15"/></w:compat></w:settings>'
)

_CONTENT_TYPES_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Default Extension="png" ContentType="image/png"/>'
    '<Override PartName="/word/document.xml" '
    'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    '<Override PartName="/word/styles.xml" '
    'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
    '<Override PartName="/word/settings.xml" '
    'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>'
    '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
    '<Override PartName="/docProps/app.xml" '
    'ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
    "</Types>"
)

_ROOT_RELS_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
    'Target="word/document.xml"/>'
    '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" '
    'Target="docProps/core.xml"/>'
    '<Relationship Id="rId3" '
    'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" '
    'Target="docProps/app.xml"/>'
    "</Relationships>"
)

_CORE_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
    'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
    'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
    "<dc:title>Outdoor Lantern L-300 Product Specification</dc:title>"
    "<dc:creator>Example Manufacturing Co.</dc:creator>"
    '<dcterms:created xsi:type="dcterms:W3CDTF">2024-01-15T09:00:00Z</dcterms:created>'
    '<dcterms:modified xsi:type="dcterms:W3CDTF">2024-01-15T09:00:00Z</dcterms:modified>'
    "</cp:coreProperties>"
)


def _app_xml(pages: int, words: int) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        f"<Application>vlmrag synthetic generator</Application><Pages>{pages}</Pages><Words>{words}</Words>"
        "<Company>Example Manufacturing Co.</Company></Properties>"
    )


def _write_docx(doc: _Document, path: Path, pages: int) -> None:
    """Write the package with fixed timestamps and member order (byte-deterministic)."""
    figure_ids = sorted(doc.figures)
    rel_ids = {fid: f"rId{10 + i}" for i, fid in enumerate(figure_ids)}
    doc_rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>',
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" '
        'Target="settings.xml"/>',
    ]
    for fid in figure_ids:
        doc_rels.append(
            f'<Relationship Id="{rel_ids[fid]}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            f'Target="media/{doc.figures[fid].file_name}"/>'
        )
    words = sum(len(a.text.split()) for a in doc.anchors() if isinstance(a, _Para)) + sum(
        len(" ".join(r.cells).split()) for a in doc.body if isinstance(a, _Table) for r in a.rows
    )
    members: list[tuple[str, bytes]] = [
        ("[Content_Types].xml", _CONTENT_TYPES_XML.encode("utf-8")),
        ("_rels/.rels", _ROOT_RELS_XML.encode("utf-8")),
        ("docProps/app.xml", _app_xml(pages, words).encode("utf-8")),
        ("docProps/core.xml", _CORE_XML.encode("utf-8")),
        ("word/document.xml", _document_xml(doc, rel_ids).encode("utf-8")),
        ("word/styles.xml", _STYLES_XML.encode("utf-8")),
        ("word/settings.xml", _SETTINGS_XML.encode("utf-8")),
        (
            "word/_rels/document.xml.rels",
            (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                + "".join(doc_rels) + "</Relationships>"
            ).encode("utf-8"),
        ),
    ]
    members += [(f"word/media/{doc.figures[fid].file_name}", doc.figures[fid].png) for fid in figure_ids]
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members:
            info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)


# --------------------------------------------------------------------------- layout via LibreOffice


def _libreoffice_page_texts(doc: _Document) -> Optional[list[str]]:
    """Render a marker-free draft with LibreOffice and return its page texts (``None`` if unavailable)."""
    from ..render.office import ConversionError, convert, find_soffice

    if find_soffice() is None:
        return None
    work = Path(tempfile.mkdtemp(prefix="vlmrag_gen_"))
    try:
        draft = work / f"{STEM}.docx"
        _write_docx(doc, draft, pages=1)
        try:
            pdf = convert(draft, "pdf", work / "pdf", timeout_s=LO_TIMEOUT_S)
        except (ConversionError, OSError) as exc:
            log.warning("LibreOffice render of the draft failed (%s); using the estimated layout", exc)
            return None
        from ..render.pdf import PdfPages

        with PdfPages(pdf) as pages:
            return [pages.text(i) for i in range(len(pages))]
    finally:
        shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- truth + entry point


def _truth(doc: _Document, source: str, page_count: int, layout_source: str, seed: int) -> dict:
    table = next(item for item in doc.body if isinstance(item, _Table))
    row_pages = [row.page for row in table.rows]
    figure_pages = {p.image: p.page for p in doc.body if isinstance(p, _Para) and p.image}
    pages = []
    for number in range(1, page_count + 1):
        headings = [p.text for p in doc.body if isinstance(p, _Para) and p.style in ("Title", "Heading1") and p.page == number]
        figures = [fid for fid, page in sorted(figure_pages.items()) if page == number]
        must = [fact for fid in figures for fact in doc.figures[fid].must_mention]
        pages.append({
            "page": number,
            "headings": headings,
            "figures": figures,
            "has_images": bool(figures),
            "table_rows": [i for i, p in enumerate(row_pages) if p == number],
            "must_mention": must,
        })
    images = [
        {
            "figure_id": fid,
            "page": figure_pages[fid],
            "location_hint": fig.location_hint,
            "description": fig.description,
            "must_mention": list(fig.must_mention),
        }
        for fid, fig in sorted(doc.figures.items())
    ]
    multi_page = row_pages[0] != row_pages[-1]
    queries = [
        ("Which component sits directly below the glass globe in the exploded view of the lantern?", "fig1"),
        ("What is the maximum charging input of the L-300 lantern?", "fig2"),
        ("In the power path diagram, which block does the USB-C input feed?", "fig2"),
        ("What colour is the status ring on the lantern in the product photograph?", "fig3"),
    ]
    return {
        "source": source,
        "doc_type": "docx",
        "seed": seed,
        "layout_source": layout_source,
        "page_count": page_count,
        "pages": pages,
        "images": images,
        "tables": [{
            "table_id": "t1" if multi_page else None,
            "multi_page": multi_page,
            "title": TABLE_TITLE,
            "title_source": "preceding_paragraph",
            "start_page": row_pages[0],
            "end_page": row_pages[-1],
            "row_count": len(table.rows),
            "header": list(TABLE_HEADER),
            "row_pages": row_pages,
        }],
        "facts": dict(doc.facts),
        "queries": [
            {"query": q, "relevant": [{"source": source, "page": figure_pages[fid]}]} for q, fid in queries
        ],
        # Page of every paragraph and table row in body order: pass it back as
        # ``generate(..., layout=...)`` to reproduce this exact document elsewhere.
        "anchor_pages": [anchor.page for anchor in doc.anchors()],
    }


def generate(
    out_dir: Union[str, Path],
    *,
    seed: int = 0,
    use_libreoffice: Optional[bool] = None,
    layout: Optional[Sequence[int]] = None,
) -> list[Path]:
    """Write ``lantern_spec.docx`` and ``lantern_spec.truth.json`` into ``out_dir``.

    ``use_libreoffice``: ``None`` (default) uses LibreOffice when installed,
    ``False`` forces the estimated layout (fast, environment-independent).
    ``layout``: the ``anchor_pages`` list of an earlier truth file; when given,
    neither LibreOffice nor the estimate is used and that earlier document is
    reproduced byte for byte (``layout_source`` is then ``"pinned"``).
    Returns the document path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = build_document(seed)
    layout_source = "estimated"
    if layout is not None:
        _apply_layout(doc, layout)
        layout_source = "pinned"
    else:
        page_texts = _libreoffice_page_texts(doc) if use_libreoffice is not False else None
        if page_texts and _assign_from_pdf(doc, page_texts):
            layout_source = "libreoffice"
        else:
            _estimate_layout(doc)
    page_count = _set_markers(doc)
    target = out / f"{STEM}.docx"
    _write_docx(doc, target, pages=page_count)
    truth = _truth(doc, target.name, page_count, layout_source, seed)
    (out / f"{STEM}.truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    log.info("wrote %s (%d pages, layout from %s)", target.name, page_count, layout_source)
    return [target]
