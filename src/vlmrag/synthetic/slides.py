"""Synthetic software-manual deck for the ``slides`` technique.

Real user manuals cannot be published, so this module writes a look-alike: a
six-slide training deck for an invented ERP ("Sample ERP") of an invented
company ("Example Manufacturing Co."). It contains exactly the things the
technique has to cope with:

* mock application screenshots drawn with PIL (title bar, menu, form fields,
  buttons, data grids, status bar);
* annotation-only shapes - a callout marker, a red highlight rectangle and
  arrow connectors pointing at buttons - that must *not* be described as
  objects of their own;
* the same screenshot inserted twice, once in full and once cropped in
  PowerPoint and enlarged, which exercises crop-aware image assets and the
  ``detail_crop`` role;
* a small supporting dialog capture (``inline_image``) and a small icon
  (``ui_icon``);
* a scaled group shape (child coordinates differ from slide coordinates);
* a native table and multi-level bullets;
* two slides without pictures (no model call is expected for them).

The facts the retrieval queries ask about (vendor number, unit price, dialog
question, overdue count, status text) are drawn into the screenshots only;
they never appear in the slide XML text, so they can only be found through
the visual descriptions.

Why a hand-written package skeleton
    ``python-pptx`` normally starts from its bundled ``default.pptx``. Building
    the minimal package (content types, presentation, one master, one layout,
    a plain theme, fixed core properties) here instead keeps the output
    byte-for-byte deterministic across python-pptx versions and works on
    machines where that bundled template is unreadable. The zip entries are
    rewritten with a fixed timestamp for the same reason, so ``doc_id`` and
    every cache key stay stable for a given ``seed``.
"""

from __future__ import annotations

import io
import json
import logging
import random
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from PIL import Image, ImageDraw, ImageFont

__all__ = [
    "DECK_STEM",
    "SLIDE_WIDTH_EMU",
    "SLIDE_HEIGHT_EMU",
    "generate",
    "new_presentation",
    "presentation_skeleton",
    "normalize_zip",
]

log = logging.getLogger(__name__)

DECK_STEM = "sample_erp_manual"
SLIDE_WIDTH_EMU = 12192000  # 13.333 in (16:9)
SLIDE_HEIGHT_EMU = 6858000  # 7.5 in
EMU_PER_INCH = 914400
_FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)

# --------------------------------------------------------------------------- package skeleton

_NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
_NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_XML_DECL = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
_P_ROOT_NS = f'xmlns:a="{_NS_A}" xmlns:r="{_NS_R}" xmlns:p="{_NS_P}"'

_CONTENT_TYPES = (
    _XML_DECL
    + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/ppt/presentation.xml" '
    'ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>'
    '<Override PartName="/ppt/slideMasters/slideMaster1.xml" '
    'ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>'
    '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" '
    'ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>'
    '<Override PartName="/ppt/theme/theme1.xml" '
    'ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>'
    '<Override PartName="/docProps/core.xml" '
    'ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
    '<Override PartName="/docProps/app.xml" '
    'ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
    "</Types>"
)

_ROOT_RELS = (
    _XML_DECL
    + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    f'<Relationship Id="rId1" Type="{_REL}/officeDocument" Target="ppt/presentation.xml"/>'
    '<Relationship Id="rId2" '
    'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" '
    'Target="docProps/core.xml"/>'
    f'<Relationship Id="rId3" Type="{_REL}/extended-properties" Target="docProps/app.xml"/>'
    "</Relationships>"
)


def _presentation_xml(width: int, height: int) -> str:
    return (
        _XML_DECL
        + f"<p:presentation {_P_ROOT_NS}>"
        '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
        f'<p:sldSz cx="{width}" cy="{height}"/>'
        '<p:notesSz cx="6858000" cy="9144000"/>'
        "</p:presentation>"
    )


_PRESENTATION_RELS = (
    _XML_DECL
    + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    f'<Relationship Id="rId1" Type="{_REL}/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
    f'<Relationship Id="rId2" Type="{_REL}/theme" Target="theme/theme1.xml"/>'
    "</Relationships>"
)

_EMPTY_GROUP = (
    '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
    '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
    '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
)


def _title_placeholder(name: str, width: int, with_geometry: bool) -> str:
    geometry = (
        f'<a:xfrm><a:off x="457200" y="228600"/><a:ext cx="{width - 914400}" cy="822960"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        if with_geometry
        else ""
    )
    return (
        "<p:sp><p:nvSpPr>"
        f'<p:cNvPr id="2" name="{name}"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr>'
        '<p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr>'
        f"<p:spPr>{geometry}</p:spPr>"
        '<p:txBody><a:bodyPr anchor="ctr"/><a:lstStyle/><a:p><a:endParaRPr lang="en-US"/></a:p></p:txBody>'
        "</p:sp>"
    )


def _master_xml(width: int) -> str:
    def level(n: int, size: int) -> str:
        indent = 342900 * n
        return (
            f'<a:lvl{n}pPr marL="{indent}" indent="-228600"><a:buFont typeface="Arial"/><a:buChar char="&#8226;"/>'
            f'<a:defRPr sz="{size}"><a:solidFill><a:schemeClr val="tx1"/></a:solidFill>'
            '<a:latin typeface="+mn-lt"/></a:defRPr></a:lvl' + f"{n}pPr>"
        )

    return (
        _XML_DECL
        + f"<p:sldMaster {_P_ROOT_NS}>"
        '<p:cSld><p:bg><p:bgRef idx="1001"><a:schemeClr val="bg1"/></p:bgRef></p:bg><p:spTree>'
        + _EMPTY_GROUP
        + _title_placeholder("Title Placeholder 1", width, True)
        + "</p:spTree></p:cSld>"
        '<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" '
        'accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" '
        'folHlink="folHlink"/>'
        '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>'
        "<p:txStyles>"
        '<p:titleStyle><a:lvl1pPr algn="l"><a:defRPr sz="3000" b="1"><a:solidFill><a:schemeClr val="tx2"/>'
        '</a:solidFill><a:latin typeface="+mj-lt"/></a:defRPr></a:lvl1pPr></p:titleStyle>'
        "<p:bodyStyle>" + level(1, 2000) + level(2, 1800) + level(3, 1600) + "</p:bodyStyle>"
        '<p:otherStyle><a:lvl1pPr><a:defRPr sz="1800"><a:solidFill><a:schemeClr val="tx1"/></a:solidFill>'
        '<a:latin typeface="+mn-lt"/></a:defRPr></a:lvl1pPr></p:otherStyle>'
        "</p:txStyles></p:sldMaster>"
    )


_MASTER_RELS = (
    _XML_DECL
    + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    f'<Relationship Id="rId1" Type="{_REL}/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
    f'<Relationship Id="rId2" Type="{_REL}/theme" Target="../theme/theme1.xml"/>'
    "</Relationships>"
)


def _layout_xml(width: int) -> str:
    return (
        _XML_DECL
        + f'<p:sldLayout {_P_ROOT_NS} type="titleOnly" preserve="1">'
        '<p:cSld name="Title Only"><p:spTree>'
        + _EMPTY_GROUP
        + _title_placeholder("Title 1", width, False)
        + "</p:spTree></p:cSld>"
        "<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>"
    )


_LAYOUT_RELS = (
    _XML_DECL
    + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    f'<Relationship Id="rId1" Type="{_REL}/slideMaster" Target="../slideMasters/slideMaster1.xml"/>'
    "</Relationships>"
)


def _theme_xml() -> str:
    colors = {
        "dk1": "000000", "lt1": "FFFFFF", "dk2": "1F2937", "lt2": "F3F4F6",
        "accent1": "2563EB", "accent2": "16A34A", "accent3": "DC2626", "accent4": "D97706",
        "accent5": "7C3AED", "accent6": "0891B2", "hlink": "1D4ED8", "folHlink": "6D28D9",
    }
    scheme = "".join(f'<a:{k}><a:srgbClr val="{v}"/></a:{k}>' for k, v in colors.items())
    solid = '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    lines = "".join(f'<a:ln w="{w}"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>'
                    for w in (9525, 19050, 28575))
    font = '<a:latin typeface="Arial"/><a:ea typeface=""/><a:cs typeface=""/>'
    return (
        _XML_DECL
        + f'<a:theme xmlns:a="{_NS_A}" name="Plain"><a:themeElements>'
        f'<a:clrScheme name="Plain">{scheme}</a:clrScheme>'
        f'<a:fontScheme name="Plain"><a:majorFont>{font}</a:majorFont><a:minorFont>{font}</a:minorFont>'
        "</a:fontScheme>"
        '<a:fmtScheme name="Plain">'
        f"<a:fillStyleLst>{solid * 3}</a:fillStyleLst>"
        f"<a:lnStyleLst>{lines}</a:lnStyleLst>"
        "<a:effectStyleLst>" + "<a:effectStyle><a:effectLst/></a:effectStyle>" * 3 + "</a:effectStyleLst>"
        f"<a:bgFillStyleLst>{solid * 3}</a:bgFillStyleLst>"
        "</a:fmtScheme></a:themeElements><a:objectDefaults/><a:extraClrSchemeLst/></a:theme>"
    )


def _core_xml(title: str) -> str:
    return (
        _XML_DECL
        + '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        f"<dc:title>{_xml_escape(title)}</dc:title><dc:creator>vlmrag synthetic generator</dc:creator>"
        "<cp:revision>1</cp:revision>"
        '<dcterms:created xsi:type="dcterms:W3CDTF">2026-01-01T00:00:00Z</dcterms:created>'
        '<dcterms:modified xsi:type="dcterms:W3CDTF">2026-01-01T00:00:00Z</dcterms:modified>'
        "</cp:coreProperties>"
    )


_APP_XML = (
    _XML_DECL
    + '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
    "<Application>vlmrag synthetic generator</Application></Properties>"
)


def _xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def presentation_skeleton(
    *, title: str = "Synthetic deck", width: int = SLIDE_WIDTH_EMU, height: int = SLIDE_HEIGHT_EMU
) -> bytes:
    """Bytes of an empty, valid PPTX: one master, one "Title Only" layout, a plain theme."""
    parts = [
        ("[Content_Types].xml", _CONTENT_TYPES),
        ("_rels/.rels", _ROOT_RELS),
        ("docProps/core.xml", _core_xml(title)),
        ("docProps/app.xml", _APP_XML),
        ("ppt/presentation.xml", _presentation_xml(width, height)),
        ("ppt/_rels/presentation.xml.rels", _PRESENTATION_RELS),
        ("ppt/slideMasters/slideMaster1.xml", _master_xml(width)),
        ("ppt/slideMasters/_rels/slideMaster1.xml.rels", _MASTER_RELS),
        ("ppt/slideLayouts/slideLayout1.xml", _layout_xml(width)),
        ("ppt/slideLayouts/_rels/slideLayout1.xml.rels", _LAYOUT_RELS),
        ("ppt/theme/theme1.xml", _theme_xml()),
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, text in parts:
            info = zipfile.ZipInfo(name, date_time=_FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, text.encode("utf-8"))
    return buffer.getvalue()


def new_presentation(**kwargs: Any):
    """A ``python-pptx`` Presentation built on :func:`presentation_skeleton` (layout 0 = "Title Only")."""
    from pptx import Presentation

    return Presentation(io.BytesIO(presentation_skeleton(**kwargs)))


def normalize_zip(path: Union[str, Path]) -> None:
    """Rewrite a zip in place with fixed timestamps so identical content gives identical bytes."""
    path = Path(path)
    with zipfile.ZipFile(path, "r") as source:
        entries = [(info.filename, source.read(info.filename)) for info in source.infolist()]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as target:
        for name, data in entries:
            info = zipfile.ZipInfo(name, date_time=_FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            target.writestr(info, data)
    path.write_bytes(buffer.getvalue())


# --------------------------------------------------------------------------- mock screens

_BG = (238, 241, 245)
_BAR = (31, 58, 95)
_MENU = (223, 229, 236)
_TEXT = (30, 34, 40)
_MUTED = (95, 104, 116)
_WHITE = (255, 255, 255)
_BORDER = (160, 170, 182)
_GREEN = (22, 140, 72)
_GRAY_BTN = (206, 212, 220)
_RED_ROW = (250, 214, 214)
_SELECT_ROW = (205, 225, 250)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)  # scalable font, Pillow >= 10.1
    except TypeError:  # older Pillow: fixed-size bitmap font (output differs, still deterministic)
        return ImageFont.load_default()


def _window(width: int, height: int, title: str, code: str, status: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), _BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width - 1, 38), fill=_BAR)
    draw.text((14, 9), f"Sample ERP  |  {title}", font=_font(19), fill=_WHITE)
    draw.text((width - 130, 10), code, font=_font(17), fill=(190, 205, 225))
    draw.rectangle((0, 39, width - 1, 68), fill=_MENU)
    x = 14
    for item in ("File", "Edit", "Purchasing", "Reports", "Help"):
        draw.text((x, 45), item, font=_font(16), fill=_TEXT)
        x += int(draw.textlength(item, font=_font(16))) + 26
    draw.rectangle((0, height - 30, width - 1, height - 1), fill=_MENU)
    draw.text((14, height - 25), status, font=_font(15), fill=_TEXT)
    draw.rectangle((0, 0, width - 1, height - 1), outline=_BORDER, width=2)
    return image, draw


def _field(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, value: str, width: int = 330) -> None:
    draw.text((x, y + 6), label, font=_font(16), fill=_MUTED)
    draw.rectangle((x + 150, y, x + 150 + width, y + 30), fill=_WHITE, outline=_BORDER)
    draw.text((x + 158, y + 6), value, font=_font(16), fill=_TEXT)


def _button(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label: str, fill: tuple, text_fill: tuple) -> None:
    draw.rounded_rectangle(box, radius=6, fill=fill, outline=_BORDER)
    width = draw.textlength(label, font=_font(17))
    x0, y0, x1, y1 = box
    draw.text((x0 + (x1 - x0 - width) / 2, y0 + (y1 - y0 - 18) / 2), label, font=_font(17), fill=text_fill)


def _grid(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    columns: list[tuple[str, int]],
    rows: list[list[str]],
    row_fills: Optional[list[Optional[tuple]]] = None,
    row_h: int = 32,
) -> tuple[int, int, int, int]:
    total_w = sum(w for _, w in columns)
    draw.rectangle((x, y, x + total_w, y + row_h), fill=_MENU, outline=_BORDER)
    cx = x
    for name, width in columns:
        draw.text((cx + 8, y + 8), name, font=_font(15), fill=_TEXT)
        cx += width
    for r, row in enumerate(rows):
        top = y + row_h * (r + 1)
        fill = (row_fills[r] if row_fills and row_fills[r] else _WHITE)
        draw.rectangle((x, top, x + total_w, top + row_h), fill=fill, outline=_BORDER)
        cx = x
        for (name, width), value in zip(columns, row):
            draw.text((cx + 8, top + 8), value, font=_font(15), fill=_TEXT)
            cx += width
    bottom = y + row_h * (len(rows) + 1)
    cx = x
    for _, width in columns[:-1]:
        cx += width
        draw.line((cx, y, cx, bottom), fill=_BORDER)
    return (x, y, x + total_w, bottom)


def _entry_screen(vendor: str, order_date: str, delivery_date: str) -> tuple[Image.Image, dict]:
    image, draw = _window(1200, 700, "Purchase Order Entry", "PUR-110", "Draft - not yet released")
    _field(draw, 40, 100, "Vendor", vendor, 420)
    _field(draw, 40, 145, "Order date", order_date)
    _field(draw, 40, 190, "Delivery date", delivery_date)
    _field(draw, 40, 235, "Plant", "Main Plant")
    _field(draw, 640, 100, "Currency", "USD", 160)
    _field(draw, 640, 145, "Payment terms", "Net 30", 160)
    draw.rectangle((40, 300, 1160, 590), fill=_WHITE, outline=_BORDER)
    draw.text((60, 315), "Order lines are added on the next screen.", font=_font(16), fill=_MUTED)
    buttons = {"Save": (760, 615, 880, 655), "Release": (895, 615, 1015, 655), "Cancel": (1030, 615, 1150, 655)}
    _button(draw, buttons["Save"], "Save", _GRAY_BTN, _TEXT)
    _button(draw, buttons["Release"], "Release", _GREEN, _WHITE)
    _button(draw, buttons["Cancel"], "Cancel", _GRAY_BTN, _TEXT)
    return image, {"buttons": buttons}


def _lines_screen(lines: list[list[str]], total: str) -> tuple[Image.Image, dict]:
    image, draw = _window(1200, 700, "Order Lines", "PUR-120", "4 lines")
    columns = [("Line", 70), ("Item", 130), ("Description", 380), ("Qty", 90), ("Unit", 70),
               ("Unit price", 130), ("Delivery", 140)]
    grid_box = _grid(draw, 40, 100, columns, lines)
    draw.text((40, grid_box[3] + 24), f"Order total: {total} USD", font=_font(18), fill=_TEXT)
    return image, {"grid_box": grid_box}


def _list_screen(orders: list[list[str]]) -> Image.Image:
    image, draw = _window(1200, 700, "Order List", "PUR-100", f"{len(orders)} orders")
    for i, label in enumerate(("New", "Open", "Release", "Print")):
        _button(draw, (40 + i * 130, 84, 160 + i * 130, 118), label, _GRAY_BTN, _TEXT)
    columns = [("Order", 140), ("Vendor", 360), ("Order date", 160), ("Amount", 160), ("Status", 160)]
    fills = [_SELECT_ROW] + [None] * (len(orders) - 1)
    _grid(draw, 40, 140, columns, orders, fills)
    return image


def _dialog_capture(order_no: str) -> Image.Image:
    image = Image.new("RGB", (520, 210), _WHITE)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 519, 34), fill=_BAR)
    draw.text((12, 8), "Release order", font=_font(17), fill=_WHITE)
    draw.text((24, 66), f"Release order {order_no} to the vendor?", font=_font(18), fill=_TEXT)
    _button(draw, (250, 140, 370, 180), "Yes", _GREEN, _WHITE)
    _button(draw, (385, 140, 505, 180), "No", _GRAY_BTN, _TEXT)
    draw.rectangle((0, 0, 519, 209), outline=_BORDER, width=2)
    return image


def _check_icon() -> Image.Image:
    image = Image.new("RGB", (96, 96), _WHITE)
    draw = ImageDraw.Draw(image)
    draw.ellipse((4, 4, 92, 92), fill=_GREEN)
    draw.line((26, 50, 42, 66, 70, 32), fill=_WHITE, width=9, joint="curve")
    return image


def _report_screen(rows: list[list[str]], overdue: list[bool]) -> Image.Image:
    count = sum(overdue)
    image, draw = _window(1200, 700, "Open Orders Report", "PUR-300", "Report generated")
    draw.rectangle((40, 84, 1160, 124), fill=_WHITE, outline=_BORDER)
    draw.text((56, 95), "Plant: Main Plant     Status: Open     Period: current quarter", font=_font(16), fill=_TEXT)
    columns = [("Order", 140), ("Vendor", 340), ("Due date", 160), ("Open qty", 140), ("Flag", 160)]
    fills = [_RED_ROW if late else None for late in overdue]
    grid_box = _grid(draw, 40, 145, columns, rows, fills)
    draw.text((40, grid_box[3] + 24), f"Overdue orders: {count}", font=_font(19), fill=(170, 20, 20))
    return image


def _png(image: Image.Image) -> io.BytesIO:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


# --------------------------------------------------------------------------- deck building


@dataclass
class _SlideTruth:
    slide: int
    title: str
    image_blocks: list[dict] = field(default_factory=list)
    annotation_shapes: list[str] = field(default_factory=list)
    visual_facts: list[str] = field(default_factory=list)
    must_mention: list[str] = field(default_factory=list)
    visual_summary: str = ""
    flow_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "slide": self.slide,
            "title": self.title,
            "expects_visual_call": bool(self.image_blocks),
            "image_blocks": self.image_blocks,
            "annotation_shapes": self.annotation_shapes,
            "visual_facts": self.visual_facts,
            "must_mention": self.must_mention,
            "visual_summary": self.visual_summary,
            "flow_notes": self.flow_notes,
        }


def _inches(value: float) -> int:
    return int(round(value * EMU_PER_INCH))


def _bbox_truth(left: int, top: int, width: int, height: int) -> dict:
    return {
        "x": round(left / SLIDE_WIDTH_EMU, 4),
        "y": round(top / SLIDE_HEIGHT_EMU, 4),
        "w": round(width / SLIDE_WIDTH_EMU, 4),
        "h": round(height / SLIDE_HEIGHT_EMU, 4),
    }


def _add_slide(prs, title: str):
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = title
    return slide


def _add_bullets(slide, left: float, top: float, width: float, height: float, items: list[tuple[int, str]], name: str):
    from pptx.util import Inches, Pt

    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    box.name = name
    frame = box.text_frame
    frame.word_wrap = True
    for i, (level, text) in enumerate(items):
        paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        paragraph.text = text
        paragraph.level = level
        for run in paragraph.runs:
            run.font.size = Pt(16 if level == 0 else 14)
    return box


def _add_picture(container, image: Image.Image, left: float, top: float, width: float, name: str, crop=None):
    from pptx.util import Inches

    picture = container.add_picture(_png(image), Inches(left), Inches(top), width=Inches(width))
    picture.name = name
    if crop:
        picture.crop_left, picture.crop_top, picture.crop_right, picture.crop_bottom = crop
        # Keep the visible region's aspect ratio: python-pptx does not resize on crop.
        visible_w = image.width * (1 - crop[0] - crop[2])
        visible_h = image.height * (1 - crop[1] - crop[3])
        picture.height = int(picture.width * visible_h / visible_w)
    return picture


def _add_highlight(slide, left: int, top: int, width: int, height: int, name: str):
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.util import Pt

    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.name = name
    shape.fill.background()
    shape.line.color.rgb = RGBColor(0xDC, 0x26, 0x26)
    shape.line.width = Pt(2.5)
    return shape


def _add_arrow(container, begin: tuple[int, int], end: tuple[int, int], name: str):
    from lxml import etree
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_CONNECTOR
    from pptx.util import Pt

    connector = container.add_connector(MSO_CONNECTOR.STRAIGHT, begin[0], begin[1], end[0], end[1])
    connector.name = name
    connector.line.color.rgb = RGBColor(0xDC, 0x26, 0x26)
    connector.line.width = Pt(2.25)
    line = connector.line._get_or_add_ln()
    tail = etree.SubElement(line, f"{{{_NS_A}}}tailEnd")
    tail.set("type", "triangle")
    return connector


def _add_marker(slide, left: int, top: int, label: str, name: str):
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.util import Inches, Pt

    marker = slide.shapes.add_shape(MSO_SHAPE.OVAL, left, top, Inches(0.42), Inches(0.42))
    marker.name = name
    marker.fill.solid()
    marker.fill.fore_color.rgb = RGBColor(0xDC, 0x26, 0x26)
    marker.line.fill.background()
    marker.text_frame.text = label
    for run in marker.text_frame.paragraphs[0].runs:
        run.font.size = Pt(14)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    return marker


def _px_to_slide(picture, image: Image.Image, x: float, y: float) -> tuple[int, int]:
    """Slide EMU position of pixel (x, y) of an uncropped picture."""
    return (
        int(picture.left + picture.width * x / image.width),
        int(picture.top + picture.height * y / image.height),
    )


def _build_deck(seed: int) -> tuple[Any, list[_SlideTruth], list[tuple[str, str, int]]]:
    from pptx.util import Inches, Pt

    rng = random.Random(seed)
    vendor_no = f"V-{rng.randint(2000, 2999)}"
    vendor = f"{vendor_no} Example Components Ltd."
    order_no = f"PO-{rng.randint(1001, 1099)}"
    order_date, delivery_date = "2026-03-02", "2026-03-16"
    housing_qty = rng.choice((80, 100, 120, 150))
    housing_price = rng.choice(("12.75", "14.50", "15.20"))
    line_items = [
        ("L300-HSG", "Outdoor Lantern L-300 housing", housing_qty, housing_price),
        ("L300-GLS", "Glass globe", housing_qty, "6.20"),
        ("L300-GSK", "Silicone gasket", housing_qty * 2, "0.35"),
        ("L300-SCR", "Screw set M4", housing_qty * 4, "0.04"),
    ]
    lines = [
        [str(10 * (i + 1)), item, desc, str(qty), "PC", price, delivery_date]
        for i, (item, desc, qty, price) in enumerate(line_items)
    ]
    total = sum(qty * float(price) for _, _, qty, price in line_items)
    total_text = f"{total:,.2f}"
    overdue = [rng.random() < 0.5 for _ in range(6)]
    if sum(overdue) == 0:
        overdue[1] = True
    overdue_count = sum(overdue)

    prs = new_presentation(title="Sample ERP user guide")
    truths: list[_SlideTruth] = []

    # Slide 1: title only, no pictures -> no model call.
    slide = _add_slide(prs, "Sample ERP user guide: purchase orders")
    _add_bullets(slide, 0.6, 1.6, 12.0, 2.0, [
        (0, "Example Manufacturing Co. - procurement training"),
        (0, "Covers creating, checking, releasing and tracking purchase orders."),
        (1, "Audience: new procurement staff"),
    ], "Intro text")
    truths.append(_SlideTruth(1, "Sample ERP user guide: purchase orders"))

    # Slide 2: main screenshot + highlight, marker and arrow (annotation only).
    slide = _add_slide(prs, "1. Create a purchase order")
    _add_bullets(slide, 0.5, 1.3, 4.3, 4.5, [
        (0, "Open Purchasing > Purchase Orders > New."),
        (0, "Enter the supplier and the dates."),
        (1, "Use the search help on the first field."),
        (0, "Confirm the header data with the action marked 1."),
    ], "Steps text")
    entry, entry_geo = _entry_screen(vendor, order_date, delivery_date)
    picture = _add_picture(slide.shapes, entry, 5.0, 1.3, 7.9, "Screen PUR-110")
    bx0, by0, bx1, by1 = entry_geo["buttons"]["Release"]
    hl_left, hl_top = _px_to_slide(picture, entry, bx0 - 6, by0 - 6)
    hl_right, hl_bottom = _px_to_slide(picture, entry, bx1 + 6, by1 + 6)
    _add_highlight(slide, hl_left, hl_top, hl_right - hl_left, hl_bottom - hl_top, "Highlight box")
    target = _px_to_slide(picture, entry, (bx0 + bx1) / 2, by0 - 6)
    _add_arrow(slide.shapes, (target[0] - Inches(1.2), target[1] - Inches(0.9)), target, "Arrow 1")
    _add_marker(slide, target[0] - Inches(1.62), target[1] - Inches(1.3), "1", "Marker 1")
    truths.append(_SlideTruth(
        2,
        "1. Create a purchase order",
        image_blocks=[{
            "shape_name": "Screen PUR-110",
            "image_role": "main_screenshot",
            "bbox_norm": _bbox_truth(picture.left, picture.top, picture.width, picture.height),
            "description": (
                f"Purchase Order Entry screen (PUR-110) of Sample ERP. The header shows vendor {vendor}, "
                f"order date {order_date}, delivery date {delivery_date}, plant Main Plant, currency USD and "
                "payment terms Net 30. Save, a green Release button and Cancel sit at the bottom right; "
                "the status bar reads 'Draft - not yet released'."
            ),
            "description_en": f"Order entry screen with vendor {vendor} filled in and the Release button.",
            "description_ko": f"공급업체 {vendor}가 입력된 구매 오더 입력 화면과 Release 버튼을 보여 줍니다.",
            "related_shapes": ["Marker 1"],
        }],
        annotation_shapes=["Highlight box", "Arrow 1", "Marker 1"],
        visual_facts=[vendor, "Draft - not yet released", "Net 30"],
        must_mention=[vendor_no, "Draft - not yet released"],
        visual_summary="The entry screen header is filled in; the marked action is the green Release button.",
        flow_notes=["An arrow from marker 1 points at the Release button."],
    ))

    # Slide 3: main screenshot + the same image cropped and enlarged (detail_crop).
    slide = _add_slide(prs, "2. Check the order lines")
    _add_bullets(slide, 0.5, 6.35, 12.3, 0.8, [
        (0, "Compare quantities and prices with the request before you continue."),
    ], "Hint text")
    lines_image, lines_geo = _lines_screen(lines, total_text)
    main = _add_picture(slide.shapes, lines_image, 0.5, 1.3, 6.6, "Screen PUR-120")
    gx0, gy0, gx1, gy1 = lines_geo["grid_box"]
    crop = (
        round((gx0 - 10) / lines_image.width, 4),
        round((gy0 - 10) / lines_image.height, 4),
        round(1 - (gx1 + 10) / lines_image.width, 4),
        round(1 - (gy1 + 60) / lines_image.height, 4),
    )
    detail = _add_picture(slide.shapes, lines_image, 7.4, 1.6, 5.5, "Detail PUR-120", crop=crop)
    start = (detail.left, detail.top + detail.height // 2)
    end = _px_to_slide(main, lines_image, gx1 - 40, (gy0 + gy1) / 2)
    _add_arrow(slide.shapes, start, end, "Arrow 2")
    housing = line_items[0]
    truths.append(_SlideTruth(
        3,
        "2. Check the order lines",
        image_blocks=[
            {
                "shape_name": "Screen PUR-120",
                "image_role": "main_screenshot",
                "bbox_norm": _bbox_truth(main.left, main.top, main.width, main.height),
                "description": (
                    "Order Lines screen (PUR-120) listing four lines: lantern housing, glass globe, silicone "
                    f"gasket and screw set; the order total is {total_text} USD."
                ),
                "description_en": "Order lines grid with four items and the order total.",
                "description_ko": "네 개 품목과 오더 총액이 표시된 오더 라인 화면입니다.",
                "related_shapes": ["Detail PUR-120"],
            },
            {
                "shape_name": "Detail PUR-120",
                "image_role": "detail_crop",
                "bbox_norm": _bbox_truth(detail.left, detail.top, detail.width, detail.height),
                "description": (
                    f"Enlarged view of the order line grid: line 10 {housing[1]} has quantity {housing[2]} PC "
                    f"at unit price {housing[3]}."
                ),
                "description_en": f"Zoomed grid: housing line, qty {housing[2]}, unit price {housing[3]}.",
                "description_ko": f"하우징 라인의 수량 {housing[2]}개와 단가 {housing[3]}를 확대해 보여 줍니다.",
                "related_shapes": ["Screen PUR-120"],
            },
        ],
        annotation_shapes=["Arrow 2"],
        visual_facts=[f"Order total: {total_text} USD", housing[3]],
        must_mention=[housing[3], total_text],
        visual_summary="A full order-lines screenshot with an enlarged crop of its grid.",
        flow_notes=["An arrow links the enlarged grid to the grid on the full screen."],
    ))

    # Slide 4: main screenshot, an inline dialog capture and a small icon.
    slide = _add_slide(prs, "3. Release the order")
    _add_bullets(slide, 0.5, 1.3, 5.6, 1.6, [
        (0, "Select the order in the list and choose the action shown by the icon."),
        (0, "Answer the confirmation prompt."),
    ], "Steps text")
    icon = _add_picture(slide.shapes, _check_icon(), 6.25, 1.38, 0.42, "Icon release")
    dialog = _add_picture(slide.shapes, _dialog_capture(order_no), 0.7, 3.2, 4.6, "Dialog capture")
    orders = [
        [order_no, vendor, order_date, f"{total:,.2f}", "Draft"],
        [f"PO-{rng.randint(1100, 1199)}", "Sample Plastics Co.", "2026-02-21", "880.00", "Released"],
        [f"PO-{rng.randint(1200, 1299)}", "Demo Glassworks", "2026-02-14", "1,240.00", "Closed"],
    ]
    listing = _add_picture(slide.shapes, _list_screen(orders), 6.9, 2.1, 6.0, "Screen PUR-100")
    truths.append(_SlideTruth(
        4,
        "3. Release the order",
        image_blocks=[
            {
                "shape_name": "Icon release",
                "image_role": "ui_icon",
                "bbox_norm": _bbox_truth(icon.left, icon.top, icon.width, icon.height),
                "description": "Green round check-mark icon of the release action.",
                "description_en": "Green check-mark release icon.",
                "description_ko": "릴리스 동작을 나타내는 초록색 체크 아이콘입니다.",
                "related_shapes": ["Steps text"],
            },
            {
                "shape_name": "Dialog capture",
                "image_role": "inline_image",
                "bbox_norm": _bbox_truth(dialog.left, dialog.top, dialog.width, dialog.height),
                "description": f"Confirmation dialog 'Release order' asking 'Release order {order_no} to the vendor?' "
                               "with Yes and No buttons.",
                "description_en": f"Dialog asking to release order {order_no} to the vendor.",
                "description_ko": f"오더 {order_no}를 공급업체에 릴리스할지 묻는 확인 대화상자입니다.",
                "related_shapes": ["Steps text"],
            },
            {
                "shape_name": "Screen PUR-100",
                "image_role": "main_screenshot",
                "bbox_norm": _bbox_truth(listing.left, listing.top, listing.width, listing.height),
                "description": f"Order List screen (PUR-100) with New, Open, Release and Print buttons; order "
                               f"{order_no} is selected and has status Draft.",
                "description_en": f"Order list with {order_no} selected.",
                "description_ko": f"{order_no}가 선택된 오더 목록 화면입니다.",
                "related_shapes": [],
            },
        ],
        visual_facts=[f"Release order {order_no} to the vendor?"],
        must_mention=[f"Release order {order_no} to the vendor?"],
        visual_summary="Order list, the release icon and the confirmation dialog.",
    ))

    # Slide 5: native table + a scaled group holding the report screenshot and a callout.
    slide = _add_slide(prs, "4. Track open orders")
    table_shape = slide.shapes.add_table(4, 2, Inches(0.5), Inches(1.3), Inches(4.4), Inches(1.8))
    table_shape.name = "Status table"
    for r, (status, meaning) in enumerate([
        ("Status", "Meaning"),
        ("Draft", "Saved, not yet sent"),
        ("Released", "Sent to the vendor"),
        ("Closed", "Fully received"),
    ]):
        for c, value in enumerate((status, meaning)):
            cell = table_shape.table.cell(r, c)
            cell.text = value
            for run in cell.text_frame.paragraphs[0].runs:
                run.font.size = Pt(14)
    report_rows = []
    for i, late in enumerate(overdue):
        report_rows.append([
            f"PO-{1300 + i * 7}", ("Example Components Ltd.", "Sample Plastics Co.", "Demo Glassworks")[i % 3],
            f"2026-03-{10 + i:02d}", str(20 * (i + 1)), "Overdue" if late else "On time",
        ])
    report = _report_screen(report_rows, overdue)
    group = slide.shapes.add_group_shape()
    group.name = "Report group"
    report_pic = _add_picture(group.shapes, report, 5.3, 1.3, 7.6, "Screen PUR-300")
    note = group.shapes.add_textbox(Inches(5.3), Inches(6.25), Inches(7.6), Inches(0.5))
    note.name = "Filter note"
    note.text_frame.text = "The filter bar at the top limits the report."
    for run in note.text_frame.paragraphs[0].runs:
        run.font.size = Pt(14)
    child_left, child_top, child_w, child_h = report_pic.left, report_pic.top, report_pic.width, report_pic.height
    # Scale the group to 90 % around its offset: children keep their child-space coordinates.
    xfrm = group._element.grpSpPr.find(f"{{{_NS_A}}}xfrm")
    off, ext = xfrm.find(f"{{{_NS_A}}}off"), xfrm.find(f"{{{_NS_A}}}ext")
    ch_off, ch_ext = xfrm.find(f"{{{_NS_A}}}chOff"), xfrm.find(f"{{{_NS_A}}}chExt")
    scale = 0.9
    ext.set("cx", str(int(int(ch_ext.get("cx")) * scale)))
    ext.set("cy", str(int(int(ch_ext.get("cy")) * scale)))
    off.set("x", str(int(off.get("x")) + _inches(0.6)))
    gx, gy = int(off.get("x")), int(off.get("y"))
    cx0, cy0 = int(ch_off.get("x")), int(ch_off.get("y"))
    shown = (
        int(gx + (child_left - cx0) * scale),
        int(gy + (child_top - cy0) * scale),
        int(child_w * scale),
        int(child_h * scale),
    )
    truths.append(_SlideTruth(
        5,
        "4. Track open orders",
        image_blocks=[{
            "shape_name": "Screen PUR-300",
            "image_role": "main_screenshot",
            "bbox_norm": _bbox_truth(*shown),
            "description": (
                "Open Orders Report (PUR-300) filtered by Plant: Main Plant and Status: Open; overdue rows are "
                f"shaded red and the footer reads 'Overdue orders: {overdue_count}'."
            ),
            "description_en": f"Open orders report; {overdue_count} orders are overdue.",
            "description_ko": f"미결 오더 보고서이며 지연 오더는 {overdue_count}건입니다.",
            "related_shapes": ["Filter note"],
        }],
        visual_facts=[f"Overdue orders: {overdue_count}", "Plant: Main Plant"],
        must_mention=[f"Overdue orders: {overdue_count}"],
        visual_summary="Report screenshot next to the status table.",
    ))

    # Slide 6: text only.
    slide = _add_slide(prs, "Troubleshooting")
    _add_bullets(slide, 0.6, 1.4, 12.0, 4.0, [
        (0, "The release action is greyed out"),
        (1, "Check that every order line has a delivery date."),
        (2, "Lines without a date block the release."),
        (0, "The supplier cannot be found"),
        (1, "Ask the master data team to create the supplier record."),
    ], "Help text")
    truths.append(_SlideTruth(6, "Troubleshooting"))

    queries = [
        ("Which vendor is entered on the purchase order entry screen?", vendor_no, 2),
        ("What does the status bar show for a new purchase order before it is released?",
         "Draft - not yet released", 2),
        ("What unit price is shown for the lantern housing order line?", housing[3], 3),
        ("What question does the release confirmation dialog ask?", f"Release order {order_no} to the vendor?", 4),
        ("How many open orders are flagged as overdue in the report?", f"Overdue orders: {overdue_count}", 5),
    ]
    return prs, truths, queries


def generate(out_dir: Union[str, Path], *, seed: int = 0) -> list[Path]:
    """Write ``sample_erp_manual.pptx`` and ``sample_erp_manual.truth.json`` into ``out_dir``.

    Returns the document paths (``[deck]``), like every generator; the truth
    file sits next to the deck as ``<stem>.truth.json``.

    The truth file lists, per slide, the expected image roles (keyed by shape
    name, because block numbers are an output of the technique, not an input),
    annotation-only shapes, facts that are visible only in the pictures,
    must-mention facts, and retrieval queries answerable only from the
    visual content.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prs, truths, queries = _build_deck(seed)
    deck_path = out / f"{DECK_STEM}.pptx"
    prs.save(str(deck_path))
    normalize_zip(deck_path)
    truth = {
        "generator": "vlmrag.synthetic.slides",
        "seed": seed,
        "source": deck_path.name,
        "doc_type": "slides",
        "slide_count": len(truths),
        "slide_size_emu": {"width": SLIDE_WIDTH_EMU, "height": SLIDE_HEIGHT_EMU},
        "slides": [t.to_dict() for t in truths],
        "queries": [
            {"query": text, "answer": answer, "relevant": [{"source": deck_path.name, "page": page}]}
            for text, answer, page in queries
        ],
    }
    truth_path = out / f"{DECK_STEM}.truth.json"
    truth_path.write_text(json.dumps(truth, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    log.info("wrote %s (%d slides) and %s", deck_path.name, len(truths), truth_path.name)
    return [deck_path]
