"""Layout blocks from the PPTX XML: the text side of the slides technique.

A slide manual already contains its text as text: titles, bullets, callout
labels and tables are stored in the PPTX XML. Reading them from there is exact
(no OCR errors, no hallucinated digits), so the model is never asked to
transcribe; it only describes what the *pictures* show. This module turns one
slide into:

* ``text_only`` - shape text and tables as plain lines (retrieval rendering);
* ``text_and_table`` - the same with tables as Markdown (answering-LLM rendering);
* ``blocks`` - an ordered list of layout blocks (text, table, image, shape),
  each with a 1-based ``order`` (the block ID the model refers to) and a
  normalised ``bbox_norm`` that locates it on the rendered slide.

Reading order is "top to bottom, then left to right" within each container,
which matches how a person reads a manual slide; groups are walked
recursively and their children are mapped from the group's child coordinate
space into slide coordinates, so their boxes line up with the render.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from ...text import markdown_table, normalize_ws, plain_table

__all__ = [
    "Block",
    "PictureInfo",
    "SlideLayout",
    "Deck",
    "Transform",
    "VISUAL_SHAPE_TYPES",
    "SlidesExtractionError",
    "read_deck",
    "extract_slide_layout",
    "sort_shapes_by_position",
    "paragraph_lines",
    "table_rows",
    "bbox_norm",
    "picture_crop",
    "visible_crop_area",
    "sniff_vector_ext",
]

log = logging.getLogger(__name__)

_NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
_MULTI_WS_RE = re.compile(r"\s{2,}")
#: Signature of OLE compound files (legacy .ppt, encrypted or rights-protected Office files).
_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


class SlidesExtractionError(RuntimeError):
    """The deck cannot be read at all (hard failure; the CLI records it per file).

    Raised instead of the library's own exception so that the error file says
    what is wrong with the input, as the drawing and docx techniques do.
    """

# Shapes without text that are still kept as "shape" blocks: they are the
# arrows, highlight boxes and connectors that tell the model how the slide's
# pictures relate to each other (and which it must not describe as objects).
VISUAL_SHAPE_TYPES = frozenset({"AUTO_SHAPE", "LINE", "FREEFORM", "TEXT_BOX"})


@dataclass(frozen=True)
class Transform:
    """Affine map from a container's coordinate space to slide space (no rotation).

    ``x_slide = sx * x + dx``; ``y_slide = sy * y + dy``. Group shapes store
    their children in a private child coordinate space (``a:chOff``/``a:chExt``)
    that is scaled and shifted onto the group's frame (``a:off``/``a:ext``);
    reading a child's ``left``/``top`` directly would misplace it whenever a
    group was resized.
    """

    sx: float = 1.0
    dx: float = 0.0
    sy: float = 1.0
    dy: float = 0.0

    def apply(self, left: int, top: int, width: int, height: int) -> tuple[int, int, int, int]:
        return (
            int(round(self.sx * left + self.dx)),
            int(round(self.sy * top + self.dy)),
            int(round(self.sx * width)),
            int(round(self.sy * height)),
        )

    def then(self, outer: "Transform") -> "Transform":
        """Compose: apply ``self`` first, then ``outer``."""
        return Transform(
            sx=outer.sx * self.sx,
            dx=outer.sx * self.dx + outer.dx,
            sy=outer.sy * self.sy,
            dy=outer.sy * self.dy + outer.dy,
        )


IDENTITY = Transform()


@dataclass
class PictureInfo:
    """What the technique knows about one embedded picture.

    ``crop`` is PowerPoint's ``(left, top, right, bottom)`` crop as fractions of
    the source image, clamped to ``[0, 1]``. The same source image is often
    inserted twice - once in full, once cropped and enlarged as a detail view -
    so ``source_sha256`` identifies the shared source while ``crop`` tells the
    two placements apart.
    """

    blob: bytes = field(repr=False)
    ext: str
    source_sha256: str
    crop: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    pixel_size: Optional[tuple[int, int]] = None

    @property
    def source_image_id(self) -> str:
        return self.source_sha256[:12]

    @property
    def cropped(self) -> bool:
        return any(self.crop)

    @property
    def crop_area_norm(self) -> float:
        return visible_crop_area(self.crop)

    @property
    def is_vector(self) -> bool:
        return self.ext in ("wmf", "emf")


@dataclass
class Block:
    order: int
    type: str  # "text" | "table" | "image" | "shape"
    shape_id: str = ""
    shape_name: str = ""
    left: int = 0
    top: int = 0
    width: int = 0
    height: int = 0
    bbox_norm: dict = field(default_factory=dict)
    text_content: str = ""
    markdown_content: str = ""
    shape_type_name: str = ""
    auto_shape_type: str = ""
    picture: Optional[PictureInfo] = None
    image_index: int = 0  # 1-based picture counter within the slide (asset naming)

    def to_dict(self) -> dict:
        """JSON-safe description (no image bytes) used for metadata and sidecars."""
        data: dict[str, Any] = {"order": self.order, "type": self.type}
        if self.shape_id:
            data["shape_id"] = self.shape_id
        if self.shape_name:
            data["shape_name"] = self.shape_name
        data.update(left=self.left, top=self.top, width=self.width, height=self.height)
        if self.bbox_norm:
            data["bbox_norm"] = dict(self.bbox_norm)
        if self.shape_type_name:
            data["shape_type_name"] = self.shape_type_name
        if self.auto_shape_type:
            data["auto_shape_type"] = self.auto_shape_type
        if self.text_content:
            data["text_content"] = self.text_content
        if self.markdown_content and self.markdown_content != self.text_content:
            data["markdown_content"] = self.markdown_content
        if self.picture is not None:
            pic = self.picture
            data["asset_ext"] = pic.ext
            data["source_image_id"] = pic.source_image_id
            data["source_image_sha256"] = pic.source_sha256
            data["crop_area_norm"] = round(pic.crop_area_norm, 6)
            if pic.cropped:
                data["crop_norm"] = _crop_dict(pic.crop)
            if pic.pixel_size:
                data["pixel_size"] = list(pic.pixel_size)
        return data


@dataclass
class SlideLayout:
    number: int  # 1-based slide number in the source deck
    title: str = ""
    hidden: bool = False
    text_only: str = ""
    text_and_table: str = ""
    blocks: list[Block] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def image_blocks(self) -> list[Block]:
        return [b for b in self.blocks if b.type == "image"]

    @property
    def slide_text(self) -> str:
        """The text sent to the model and shown to the answering LLM (tables as Markdown)."""
        return (self.text_and_table or self.text_only).strip()

    def block(self, order: int) -> Optional[Block]:
        for candidate in self.blocks:
            if candidate.order == order:
                return candidate
        return None


@dataclass
class Deck:
    slide_width: int
    slide_height: int
    slides: list[SlideLayout]

    @property
    def slide_count(self) -> int:
        return len(self.slides)


# --------------------------------------------------------------------------- small pure helpers


def _clamp01(value: Any) -> float:
    try:
        number = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def _crop_dict(crop: tuple[float, float, float, float]) -> dict:
    left, top, right, bottom = crop
    return {"l": round(left, 6), "t": round(top, 6), "r": round(right, 6), "b": round(bottom, 6)}


def bbox_norm(left: int, top: int, width: int, height: int, slide_width: int, slide_height: int) -> dict:
    """``{"x", "y", "w", "h"}`` as fractions of the slide size (6 decimals); ``{}`` without a slide size."""
    if not slide_width or not slide_height:
        return {}
    return {
        "x": round(left / slide_width, 6),
        "y": round(top / slide_height, 6),
        "w": round(width / slide_width, 6),
        "h": round(height / slide_height, 6),
    }


def picture_crop(shape: Any) -> tuple[float, float, float, float]:
    """PowerPoint crop fractions ``(left, top, right, bottom)`` clamped to ``[0, 1]``.

    Negative values (a picture stretched beyond its frame) carry no hidden
    content, so they are treated as "no crop".
    """
    return tuple(_clamp01(getattr(shape, name, 0)) for name in ("crop_left", "crop_top", "crop_right", "crop_bottom"))  # type: ignore[return-value]


def visible_crop_area(crop: Iterable[float]) -> float:
    """Visible fraction of the source image after cropping (``1.0`` = uncropped)."""
    left, top, right, bottom = crop
    return max(0.0, 1.0 - left - right) * max(0.0, 1.0 - top - bottom)


def sniff_vector_ext(blob: bytes) -> Optional[str]:
    """``"emf"`` / ``"wmf"`` for Windows metafiles, else ``None``.

    Metafiles are recognised from their bytes because their part names and
    PIL's format name ("WMF" for both kinds) are unreliable.
    """
    if len(blob) >= 44 and blob[:4] == b"\x01\x00\x00\x00" and blob[40:44] == b" EMF":
        return "emf"
    if blob[:4] == b"\xd7\xcd\xc6\x9a":  # placeable WMF
        return "wmf"
    if len(blob) >= 4 and blob[:2] in (b"\x01\x00", b"\x02\x00") and blob[2:4] == b"\x09\x00":
        return "wmf"
    return None


def sort_shapes_by_position(shapes: Iterable[Any]) -> list[Any]:
    """Sort by ``(top, left)``: the reading order of a manual slide.

    Shapes without a position (rare: placeholders with nothing to inherit) are
    kept and placed after the positioned ones instead of being dropped, so
    their text is not lost.
    """
    positioned, unpositioned = [], []
    for shape in shapes:
        top = getattr(shape, "top", None)
        left = getattr(shape, "left", None)
        (positioned if top is not None and left is not None else unpositioned).append(shape)
    positioned.sort(key=lambda s: (s.top or 0, s.left or 0))
    return positioned + unpositioned


def paragraph_lines(text_frame: Any) -> str:
    """Text of a text frame, one line per non-empty paragraph.

    Each line is ``<tabs for the indent level>- <text>``: the dash marks a
    bullet and the tabs keep the outline structure that tells a reader which
    sub-steps belong to which step. Runs of whitespace inside the text and
    soft line breaks are collapsed to one space.
    """
    lines = []
    for paragraph in getattr(text_frame, "paragraphs", []) or []:
        try:
            text = (paragraph.text or "").replace("\v", " ")
            text = _MULTI_WS_RE.sub(" ", text).strip()
            if not text:
                continue
            level = int(getattr(paragraph, "level", 0) or 0)
            lines.append("\t" * level + "- " + text)
        except Exception as exc:  # noqa: BLE001 - one bad paragraph must not lose the shape
            log.warning("skipping unreadable paragraph: %s", exc)
    return "\n".join(lines)


def table_rows(table: Any) -> list[list[str]]:
    """Cell texts of a PPTX table, row by row."""
    rows = []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            try:
                cells.append(cell.text_frame.text if cell.text_frame is not None else "")
            except Exception:  # noqa: BLE001 - unreadable cell -> empty cell
                cells.append("")
        rows.append(cells)
    return rows


def _table_texts(table: Any) -> tuple[str, str]:
    """``(plain, markdown)`` renderings; empty rows are dropped, a blank header falls back to plain."""
    rows = table_rows(table)
    if not rows:
        return "", ""
    plain = plain_table(rows)
    header, body = rows[0], [r for r in rows[1:] if any(normalize_ws(c) for c in r)]
    if not any(normalize_ws(c) for c in header):
        return plain, plain
    return plain, markdown_table([header] + body, header=True)


def _group_transform(group: Any, outer: Transform) -> Transform:
    xfrm = None
    try:
        xfrm = group._element.grpSpPr.find(f"{{{_NS_A}}}xfrm")
    except AttributeError:
        xfrm = None
    if xfrm is None:
        return outer

    def point(tag: str, a: str, b: str) -> Optional[tuple[float, float]]:
        node = xfrm.find(f"{{{_NS_A}}}{tag}")
        if node is None:
            return None
        try:
            return float(node.get(a, 0)), float(node.get(b, 0))
        except (TypeError, ValueError):
            return None

    off, ext = point("off", "x", "y"), point("ext", "cx", "cy")
    ch_off, ch_ext = point("chOff", "x", "y"), point("chExt", "cx", "cy")
    if off is None or ext is None or ch_off is None or ch_ext is None:
        return outer
    sx = ext[0] / ch_ext[0] if ch_ext[0] else 1.0
    sy = ext[1] / ch_ext[1] if ch_ext[1] else 1.0
    inner = Transform(sx=sx, dx=off[0] - ch_off[0] * sx, sy=sy, dy=off[1] - ch_off[1] * sy)
    return inner.then(outer)


def _shape_type_name(shape: Any) -> str:
    shape_type = getattr(shape, "shape_type", None)
    return str(getattr(shape_type, "name", shape_type or "")).strip()


def _auto_shape_name(shape: Any) -> str:
    # python-pptx raises ValueError (not AttributeError) for text boxes and freeforms.
    try:
        auto_shape = shape.auto_shape_type
    except (AttributeError, ValueError, NotImplementedError):
        return ""
    return str(getattr(auto_shape, "name", auto_shape or "")).strip()


def _is_picture(shape: Any) -> bool:
    from pptx.shapes.picture import Picture

    return isinstance(shape, Picture)


def _picture_info(shape: Any) -> PictureInfo:
    image = shape.image  # raises for linked (not embedded) pictures
    blob = image.blob
    ext = sniff_vector_ext(blob)
    if ext is None:
        try:
            ext = image.ext
        except ValueError:
            ext = "bin"
    pixel_size = None
    try:
        width, height = image.size
        if width and height:
            pixel_size = (int(width), int(height))
    except Exception:  # noqa: BLE001 - size is informative only
        pixel_size = None
    return PictureInfo(
        blob=blob,
        ext=(ext or "bin").lower(),
        source_sha256=hashlib.sha256(blob).hexdigest(),
        crop=picture_crop(shape),
        pixel_size=pixel_size,
    )


# --------------------------------------------------------------------------- slide walker


def extract_slide_layout(slide: Any, number: int, slide_width: int, slide_height: int) -> SlideLayout:
    """Walk one slide and build its texts and ordered layout blocks.

    Every shape is handled independently: an unreadable shape is logged and
    skipped so the rest of the slide is still indexed.
    """
    layout = SlideLayout(number=number)
    layout.hidden = str(slide._element.get("show", "1")).lower() in ("0", "false")
    try:
        title_shape = slide.shapes.title
        if title_shape is not None and title_shape.has_text_frame:
            layout.title = normalize_ws(title_shape.text_frame.text)
    except Exception:  # noqa: BLE001 - a missing title is fine
        layout.title = ""

    content_pairs: list[tuple[str, str]] = []
    image_counter = [0]

    def position(shape: Any, transform: Transform) -> dict:
        left = int(getattr(shape, "left", 0) or 0)
        top = int(getattr(shape, "top", 0) or 0)
        width = int(getattr(shape, "width", 0) or 0)
        height = int(getattr(shape, "height", 0) or 0)
        left, top, width, height = transform.apply(left, top, width, height)
        return {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "bbox_norm": bbox_norm(left, top, width, height, slide_width, slide_height),
        }

    def new_block(shape: Any, block_type: str, transform: Transform, **extra: Any) -> Block:
        shape_id = getattr(shape, "shape_id", None)
        block = Block(
            order=len(layout.blocks) + 1,
            type=block_type,
            shape_id="" if shape_id is None else str(shape_id),
            shape_name=str(getattr(shape, "name", "") or "").strip(),
            **position(shape, transform),
            **extra,
        )
        layout.blocks.append(block)
        return block

    def visual_shape_block(shape: Any, transform: Transform) -> None:
        type_name = _shape_type_name(shape)
        auto_name = _auto_shape_name(shape)
        if type_name not in VISUAL_SHAPE_TYPES and "ARROW" not in auto_name:
            return
        new_block(shape, "shape", transform, shape_type_name=type_name, auto_shape_type=auto_name)

    def walk(shapes: Iterable[Any], transform: Transform) -> None:
        for shape in sort_shapes_by_position(shapes):
            try:
                if _shape_type_name(shape) == "GROUP" and hasattr(shape, "shapes"):
                    walk(shape.shapes, _group_transform(shape, transform))
                    continue
                if _is_picture(shape):
                    info = _picture_info(shape)
                    image_counter[0] += 1
                    new_block(shape, "image", transform, picture=info, image_index=image_counter[0])
                    continue
                if getattr(shape, "has_text_frame", False):
                    text = paragraph_lines(shape.text_frame)
                    if text:
                        content_pairs.append((text, text))
                        new_block(shape, "text", transform, text_content=text)
                    else:
                        visual_shape_block(shape, transform)
                    continue
                if getattr(shape, "has_table", False):
                    plain, markdown = _table_texts(shape.table)
                    if plain:
                        content_pairs.append((plain, markdown or plain))
                        new_block(shape, "table", transform, text_content=plain, markdown_content=markdown or plain)
                    continue
                visual_shape_block(shape, transform)
            except Exception as exc:  # noqa: BLE001 - per-shape isolation
                log.warning("slide %d: skipping shape %r: %s", number, getattr(shape, "name", "?"), exc)

    walk(slide.shapes, IDENTITY)
    layout.text_only = "\n".join(p[0] for p in content_pairs if p[0]).strip()
    layout.text_and_table = "\n".join(p[1] for p in content_pairs if p[1]).strip()
    return layout


def _open_failure_message(path: Path, exc: BaseException) -> str:
    try:
        with open(path, "rb") as handle:
            head = handle.read(len(_OLE_MAGIC))
    except OSError:
        head = b""
    if head == _OLE_MAGIC:
        hint = "it is an OLE compound file: a legacy .ppt, or an encrypted or rights-protected deck"
        return f"{path.name} is not a PPTX (zip) package ({hint})"
    if not head.startswith(b"PK"):
        return f"{path.name} is not a PPTX (zip) package"
    return f"{path.name}: the PPTX package could not be opened: {type(exc).__name__}: {exc}"


def read_deck(path: Union[str, Path]) -> Deck:
    """Open a ``.pptx`` and extract every slide's layout.

    A file python-pptx cannot open is a hard failure
    (:class:`SlidesExtractionError`). A slide that fails as a whole yields an
    empty :class:`SlideLayout` with ``error`` set, so slide numbers stay
    aligned with the source deck.
    """
    from pptx import Presentation

    try:
        prs = Presentation(str(path))
    except Exception as exc:  # noqa: BLE001 - any open failure becomes one clear error
        raise SlidesExtractionError(_open_failure_message(Path(path), exc)) from exc
    width = int(prs.slide_width or 0)
    height = int(prs.slide_height or 0)
    slides = []
    for number, slide in enumerate(prs.slides, start=1):
        try:
            slides.append(extract_slide_layout(slide, number, width, height))
        except Exception as exc:  # noqa: BLE001 - keep numbering aligned
            log.warning("slide %d: layout extraction failed: %s", number, exc)
            slides.append(SlideLayout(number=number, error=f"{type(exc).__name__}: {exc}"))
    return Deck(slide_width=width, slide_height=height, slides=slides)
