"""Which regions of a drawing sheet the model sees, at what resolution, under which label.

A title block is a few percent of an A3 sheet and its smallest cells use 4-6 pt
type. Sent as one page image, that text shrinks below legibility once the
provider downsamples the image. The technique therefore sends several "zoom
levels" of the same sheet:

1. the whole page as an overlapping grid of tiles (default 3 x 4, 12 % overlap,
   400 dpi, enlarged x2), so nothing is missed even when a drawing puts its
   title block somewhere unusual and so text crossing a cut appears whole in at
   least one tile;
2. a focus crop of the lower-right corner where title blocks live
   (x >= 0.68, y >= 0.52, 500 dpi, x3);
3. a close-up of the smallest bottom-right table cells
   (x >= 0.82, y >= 0.70, 650 dpi, x4), shared by both passes;
4. for the table pass only, the wider metadata-table region
   (x >= 0.44, y >= 0.62, 650 dpi, x4) as an overview plus a 2 x 3 grid with
   10 % overlap.

Every image is preceded by a position label (kind, grid cell, region name,
normalised box, dpi). The text layer of exactly the same box is sent as a hint
under the same label, so the model can match hints to pixels and the prompt can
tell it to trust a hint only where the image agrees.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

from PIL import Image

from ...render.images import Tile, fit_max_pixels, format_bbox, tile_grid, to_png, upscale
from ...vlm.types import ImageInput
from .config import DrawingConfig

__all__ = [
    "CropSpec",
    "PreparedImage",
    "FULL_PAGE",
    "region_name",
    "corner_box",
    "plan_crops",
    "scale_for_model",
    "render_crop",
    "region_text",
    "hint_section",
    "hint_block",
    "VISION_HINT_INTRO",
    "TABLE_HINT_INTRO",
]

log = logging.getLogger(__name__)

BBox = tuple[float, float, float, float]
FULL_PAGE: BBox = (0.0, 0.0, 1.0, 1.0)

VISION_HINT_INTRO = (
    "Text extracted from the PDF text layer of the same regions follows. Each section is headed by the "
    "position label of the image it belongs to; the FULL_PAGE section covers the whole sheet and has no "
    "image of its own. It may be noisy or incomplete, so use it only where it matches the text visible "
    "in the images."
)
TABLE_HINT_INTRO = (
    "Text extracted from the PDF text layer of the same metadata-table regions follows. Each section is "
    "headed by the position label of the image it belongs to. It may be noisy or incomplete, so use it "
    "only where it matches the table text visible in that image."
)

_ROW_NAMES = {1: ("",), 2: ("top", "bottom"), 3: ("top", "middle", "bottom")}
_COL_NAMES = {
    1: ("",),
    2: ("left", "right"),
    3: ("left", "center", "right"),
    4: ("left", "center-left", "center-right", "right"),
}


def region_name(row: int, col: int, rows: int, cols: int) -> str:
    """Human-readable cell name such as ``"bottom-right"`` (0-based ``row``/``col``).

    Models reason better about "the bottom-right tile" than about indexes;
    grids larger than the named ones fall back to ``"row4-col5"``.
    """
    row_names = _ROW_NAMES.get(rows)
    col_names = _COL_NAMES.get(cols)
    row_part = row_names[row] if row_names else f"row{row + 1}"
    col_part = col_names[col] if col_names else f"col{col + 1}"
    name = "-".join(p for p in (row_part, col_part) if p)
    return name or "whole"


def corner_box(x_start: float, y_start: float) -> Optional[BBox]:
    """``(x_start, y_start)`` to the bottom-right page corner, clamped; ``None`` if empty."""
    x0 = max(0.0, min(1.0, float(x_start)))
    y0 = max(0.0, min(1.0, float(y_start)))
    if x0 >= 1.0 or y0 >= 1.0:
        return None
    return (x0, y0, 1.0, 1.0)


@dataclass(frozen=True)
class CropSpec:
    """One image the model will see: where, how sharp, and which pass(es) use it."""

    kind: str
    label: str
    bbox: BBox
    dpi: float
    upscale: float
    for_vision: bool
    for_table: bool


def _focus_dpi(dpi: float) -> int:
    # Focus and table crops never go below screen resolution.
    return max(72, int(dpi))


def plan_crops(cfg: DrawingConfig) -> list[CropSpec]:
    """All crops of one page, in the order the model receives them.

    Vision pass: page tiles (row by row), title-block focus, table focus.
    Table pass: table focus, metadata-table overview, metadata-table tiles.
    """
    crops: list[CropSpec] = []
    rows, cols = cfg.page_tile_rows, cfg.page_tile_cols
    page_dpi = float(cfg.page_dpi)
    for tile in tile_grid(rows, cols, cfg.page_tile_overlap):
        crops.append(
            CropSpec(
                kind="page_tile",
                label=_grid_label("PAGE_TILE", tile, rows, cols, page_dpi),
                bbox=tile.bbox,
                dpi=page_dpi,
                upscale=cfg.page_tile_upscale,
                for_vision=True,
                for_table=False,
            )
        )
    title_box = corner_box(cfg.title_focus_x_start, cfg.title_focus_y_start)
    if title_box is not None:
        dpi = _focus_dpi(cfg.title_focus_dpi)
        crops.append(
            CropSpec(
                kind="title_block_focus",
                label=f"TITLE_BLOCK_FOCUS lower-right {format_bbox(title_box)} dpi={dpi:g}",
                bbox=title_box,
                dpi=dpi,
                upscale=cfg.title_focus_upscale,
                for_vision=True,
                for_table=False,
            )
        )
    table_box = corner_box(cfg.table_focus_x_start, cfg.table_focus_y_start)
    if table_box is not None:
        dpi = _focus_dpi(cfg.table_focus_dpi)
        crops.append(
            CropSpec(
                kind="table_focus",
                label=f"TABLE_FOCUS bottom-right {format_bbox(table_box)} dpi={dpi:g}",
                bbox=table_box,
                dpi=dpi,
                upscale=cfg.table_focus_upscale,
                for_vision=True,
                for_table=True,
            )
        )
    meta_box = corner_box(cfg.metadata_x_start, cfg.metadata_y_start)
    if meta_box is not None:
        dpi = _focus_dpi(cfg.metadata_dpi)
        crops.append(
            CropSpec(
                kind="table_overview",
                label=f"METADATA_TABLE_OVERVIEW {format_bbox(meta_box)} dpi={dpi:g}",
                bbox=meta_box,
                dpi=dpi,
                upscale=cfg.metadata_upscale,
                for_vision=False,
                for_table=True,
            )
        )
        m_rows, m_cols = cfg.metadata_tile_rows, cfg.metadata_tile_cols
        for tile in tile_grid(m_rows, m_cols, cfg.metadata_tile_overlap, region=meta_box):
            crops.append(
                CropSpec(
                    kind="table_tile",
                    label=_grid_label("METADATA_TABLE_TILE", tile, m_rows, m_cols, dpi),
                    bbox=tile.bbox,
                    dpi=dpi,
                    upscale=cfg.metadata_upscale,
                    for_vision=False,
                    for_table=True,
                )
            )
    return crops


def _grid_label(prefix: str, tile: Tile, rows: int, cols: int, dpi: float) -> str:
    name = region_name(tile.row, tile.col, rows, cols)
    return f"{prefix} r{tile.row + 1}c{tile.col + 1} {name} {format_bbox(tile.bbox)} dpi={dpi:g}"


# --------------------------------------------------------------------------- pixels


def _resize(img: Image.Image, factor: float) -> Image.Image:
    size = (max(1, int(round(img.width * factor))), max(1, int(round(img.height * factor))))
    return img.resize(size, Image.LANCZOS)


def scale_for_model(img: Image.Image, factor: float, max_pixels: int) -> tuple[Image.Image, float, bool]:
    """Apply the configured enlargement, bounded by ``max_pixels`` (0 = unbounded).

    Returns ``(image, applied_factor, capped)``. Enlarging only helps until the
    provider's own downscaling kicks in, while memory and request size grow
    with the square of the factor; the cap keeps a large sheet from producing
    images of hundreds of megapixels.
    """
    width, height = img.size
    factor = float(factor or 1.0)
    capped = False
    if max_pixels and width * height * factor * factor > max_pixels:
        factor = math.sqrt(max_pixels / float(width * height))
        capped = True
    if factor > 1.0:
        out = upscale(img, factor)
    elif factor < 1.0:
        out = _resize(img, factor)
    else:
        out = img
    if max_pixels and out.width * out.height > max_pixels:
        out = fit_max_pixels(out, max_pixels)  # rounding guard
    return out, factor, capped


@dataclass
class PreparedImage:
    spec: CropSpec
    image: ImageInput
    info: dict


def render_crop(pages: Any, index: int, spec: CropSpec, max_pixels: int) -> PreparedImage:
    """Render ``spec`` from page ``index`` (0-based) and encode it as a labelled PNG."""
    rendered = pages.render(index, spec.dpi, bbox=spec.bbox)
    image, factor, capped = scale_for_model(rendered, spec.upscale, max_pixels)
    info = {
        "kind": spec.kind,
        "label": spec.label,
        "rendered_size": [rendered.width, rendered.height],
        "size": [image.width, image.height],
        "upscale": spec.upscale,
        "upscale_applied": round(factor, 4),
        "capped": capped,
    }
    png = to_png(image)
    if image is not rendered:
        image.close()
    rendered.close()
    return PreparedImage(spec=spec, image=ImageInput(png=png, label=spec.label), info=info)


# --------------------------------------------------------------------------- text-layer hints


def region_text(pages: Any, index: int, bbox: BBox) -> str:
    """Text layer inside ``bbox`` (empty for raster-only regions or scanned sheets)."""
    try:
        if tuple(bbox) == FULL_PAGE:
            return (pages.text(index) or "").strip()
        return (pages.text(index, bbox) or "").strip()
    except Exception as exc:  # noqa: BLE001 - hints are optional; a broken text layer must not stop vision
        log.warning("page %d: text layer unreadable in %s: %s", index + 1, format_bbox(bbox), exc)
        return ""


def hint_section(label: str, text: str) -> str:
    text = (text or "").strip()
    return f"[{label}]\n{text}" if text else ""


def hint_block(intro: str, sections: list[str]) -> str:
    """Join non-empty sections under ``intro``; ``""`` when there is nothing to hint."""
    body = "\n\n".join(s for s in sections if s)
    return f"{intro}\n{body}" if body else ""
