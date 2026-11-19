"""Image geometry helpers: overlapping tile grids, upscaling, PNG encoding.

All boxes are normalised ``(x0, y0, x1, y1)`` with a top-left origin on the
*displayed* page, so the same numbers describe a PDF region, a rendered crop and
the position label the model reads.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Optional

from PIL import Image

__all__ = ["Tile", "tile_grid", "upscale", "to_png", "fit_max_pixels", "format_bbox", "tile_label", "clamp_bbox"]

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class Tile:
    """One cell of a tile grid; ``row``/``col`` are 0-based, ``bbox`` is normalised."""

    row: int
    col: int
    bbox: BBox


def clamp_bbox(bbox: BBox) -> BBox:
    """Clamp a normalised box into the unit square and validate it is non-empty."""
    x0, y0, x1, y1 = (min(1.0, max(0.0, float(v))) for v in bbox)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"empty or inverted bbox {bbox!r}")
    return (x0, y0, x1, y1)


def tile_grid(rows: int, cols: int, overlap: float, region: BBox = (0.0, 0.0, 1.0, 1.0)) -> list[Tile]:
    """Split ``region`` into ``rows`` x ``cols`` tiles, each grown by ``overlap``.

    ``overlap`` is a fraction of the tile size added on every side and then
    clamped to the region, so neighbouring tiles share a band of
    ``2 * overlap * tile size``. The overlap exists because small text that
    straddles a cut would otherwise be split into two unreadable halves; with
    the band, every glyph appears whole in at least one tile. Tiles are listed
    row by row, left to right.
    """
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be >= 1")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    rx0, ry0, rx1, ry1 = region
    tile_w = (rx1 - rx0) / cols
    tile_h = (ry1 - ry0) / rows
    tiles = []
    for r in range(rows):
        for c in range(cols):
            x0 = max(rx0, rx0 + c * tile_w - tile_w * overlap)
            y0 = max(ry0, ry0 + r * tile_h - tile_h * overlap)
            x1 = min(rx1, rx0 + (c + 1) * tile_w + tile_w * overlap)
            y1 = min(ry1, ry0 + (r + 1) * tile_h + tile_h * overlap)
            tiles.append(Tile(r, c, (x0, y0, x1, y1)))
    return tiles


def format_bbox(bbox: BBox) -> str:
    """``"x=0.24-0.52 y=0.00-0.37"`` - the coordinate part of a position label."""
    x0, y0, x1, y1 = bbox
    return f"x={x0:.2f}-{x1:.2f} y={y0:.2f}-{y1:.2f}"


def tile_label(tile: Tile, dpi: Optional[float] = None, prefix: str = "TILE") -> str:
    """Position label such as ``"TILE r1c2 x=0.24-0.52 y=0.00-0.37 dpi=400"`` (1-based row/col)."""
    label = f"{prefix} r{tile.row + 1}c{tile.col + 1} {format_bbox(tile.bbox)}"
    if dpi:
        label += f" dpi={dpi:g}"
    return label


def upscale(img: Image.Image, factor: float) -> Image.Image:
    """Resize by ``factor`` with LANCZOS; a no-op when ``factor <= 1``.

    Upscaling a crop does not add information, but vision models tokenise
    images at a fixed patch size: enlarging a dense title block makes 4-6 pt
    glyphs span enough patches to be read reliably.
    """
    if factor is None or factor <= 1:
        return img
    size = (max(1, int(round(img.width * factor))), max(1, int(round(img.height * factor))))
    return img.resize(size, Image.LANCZOS)


def to_png(img: Image.Image) -> bytes:
    buffer = io.BytesIO()
    if img.mode not in ("RGB", "L", "RGBA", "LA", "P", "1"):
        img = img.convert("RGB")
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def fit_max_pixels(img: Image.Image, max_pixels: int) -> Image.Image:
    """Downscale (aspect preserved, LANCZOS) so ``width * height <= max_pixels``.

    Providers reject or silently downsample oversized images; doing it here
    keeps the result deterministic and the payload small.
    """
    if max_pixels is None or max_pixels <= 0:
        return img
    pixels = img.width * img.height
    if pixels <= max_pixels:
        return img
    scale = math.sqrt(max_pixels / pixels)
    size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    while size[0] * size[1] > max_pixels and (size[0] > 1 or size[1] > 1):
        size = (max(1, size[0] - 1), max(1, size[1] - 1))
    return img.resize(size, Image.LANCZOS)
