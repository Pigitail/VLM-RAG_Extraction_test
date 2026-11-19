"""Rendering helpers: PDF pages (pypdfium2), image geometry, LibreOffice conversion.

``vlmrag.render.pdf`` imports pypdfium2, so it is not imported here eagerly;
use ``from vlmrag.render.pdf import PdfPages``.
"""

from __future__ import annotations

from .images import Tile, fit_max_pixels, format_bbox, tile_grid, tile_label, to_png, upscale
from .office import ConversionError, convert, find_soffice

__all__ = [
    "ConversionError",
    "Tile",
    "convert",
    "find_soffice",
    "fit_max_pixels",
    "format_bbox",
    "tile_grid",
    "tile_label",
    "to_png",
    "upscale",
]
