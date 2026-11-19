"""PDF page rendering and text-layer access with pypdfium2 (v5 API).

Everything here speaks *displayed* coordinates: normalised ``(x0, y0, x1, y1)``
with a top-left origin on the page as a viewer shows it, i.e. after the page's
``/Rotate`` has been applied. Engineering drawings are frequently stored as
portrait pages with ``/Rotate 90``; if crops were computed in unrotated
coordinates, "the bottom-right title block" would land in a different corner
for every rotation.

* Rendering: PDFium applies ``/Rotate`` itself and pypdfium2's ``crop`` margins
  are applied after rotation, so a displayed box maps directly onto crop
  margins and only that region is rasterised (no full-page bitmap at 650 dpi).
* Text: ``PdfTextPage.get_text_bounded`` works in *unrotated* PDF user space
  (bottom-left origin, relative to the page box), so displayed boxes are mapped
  back through the inverse rotation first.

PDFium is not thread-safe, so every call into it is serialised with a
module-level lock; VLM calls (the slow part) still run concurrently.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional, Union

import pypdfium2 as pdfium
from PIL import Image

from .images import BBox, clamp_bbox, upscale as _upscale

__all__ = ["PdfPages", "displayed_to_user_space"]

log = logging.getLogger(__name__)

_PDFIUM_LOCK = threading.RLock()


def displayed_to_user_space(bbox: BBox, rotation: int, page_box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Map a normalised displayed box to unrotated PDF user space.

    ``page_box`` is ``(left, bottom, right, top)`` of the unrotated page.
    Returns ``(left, bottom, right, top)`` in points. ``rotation`` is the
    clockwise ``/Rotate`` value (0, 90, 180 or 270).
    """
    left, bottom, right, top = page_box
    w = right - left
    h = top - bottom
    rotation = rotation % 360

    def to_user(u: float, v: float) -> tuple[float, float]:
        if rotation == 0:
            return left + u * w, top - v * h
        if rotation == 90:
            return left + v * w, bottom + u * h
        if rotation == 180:
            return right - u * w, bottom + v * h
        if rotation == 270:
            return right - v * w, top - u * h
        raise ValueError(f"unsupported page rotation {rotation}")

    x0, y0, x1, y1 = bbox
    corners = [to_user(x0, y0), to_user(x1, y1)]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return min(xs), min(ys), max(xs), max(ys)


class PdfPages:
    """Random access to the pages of one PDF (context manager)."""

    def __init__(self, path: Union[str, os.PathLike, bytes]) -> None:
        self.path = None if isinstance(path, bytes) else Path(path)
        with _PDFIUM_LOCK:
            self._doc = pdfium.PdfDocument(path if isinstance(path, bytes) else str(path))
            self._count = len(self._doc)

    def __enter__(self) -> "PdfPages":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        with _PDFIUM_LOCK:
            if self._doc is not None:
                self._doc.close()
                self._doc = None

    def __len__(self) -> int:
        return self._count

    def _page(self, index: int):
        if self._doc is None:
            raise ValueError("PdfPages is closed")
        if not 0 <= index < self._count:
            raise IndexError(f"page index {index} out of range (0..{self._count - 1})")
        return self._doc[index]

    def rotation(self, index: int) -> int:
        with _PDFIUM_LOCK:
            page = self._page(index)
            try:
                return int(page.get_rotation()) % 360
            finally:
                page.close()

    def page_size(self, index: int) -> tuple[float, float]:
        """Displayed ``(width, height)`` in points (``/Rotate`` applied)."""
        with _PDFIUM_LOCK:
            page = self._page(index)
            try:
                width, height = page.get_size()
                return float(width), float(height)
            finally:
                page.close()

    def render(self, index: int, dpi: float, bbox: Optional[BBox] = None, upscale: float = 1.0) -> Image.Image:
        """Render the page (or the displayed region ``bbox``) at ``dpi`` as an RGB image.

        ``upscale`` > 1 enlarges the rendered pixels with LANCZOS afterwards,
        matching how crops are prepared for the model (render at the target DPI,
        then enlarge).
        """
        if dpi <= 0:
            raise ValueError("dpi must be positive")
        scale = float(dpi) / 72.0
        with _PDFIUM_LOCK:
            page = self._page(index)
            try:
                width, height = page.get_size()
                crop = (0.0, 0.0, 0.0, 0.0)
                if bbox is not None:
                    x0, y0, x1, y1 = clamp_bbox(bbox)
                    # Margins (left, bottom, right, top) in points, applied after rotation.
                    crop = (x0 * width, (1.0 - y1) * height, (1.0 - x1) * width, y0 * height)
                bitmap = page.render(scale=scale, crop=crop)
                try:
                    image = bitmap.to_pil().convert("RGB")
                finally:
                    bitmap.close()
            finally:
                page.close()
        return _upscale(image, upscale)

    def text(self, index: int, bbox: Optional[BBox] = None) -> str:
        """Text layer of the page, or of the displayed region ``bbox``.

        Returns ``""`` for pages without a text layer (scans, outlined fonts),
        which callers treat as "vision only".
        """
        with _PDFIUM_LOCK:
            page = self._page(index)
            try:
                textpage = page.get_textpage()
                try:
                    if bbox is None:
                        raw = textpage.get_text_range()
                    else:
                        left, bottom, right, top = displayed_to_user_space(
                            clamp_bbox(bbox), int(page.get_rotation()), tuple(page.get_bbox())
                        )
                        raw = textpage.get_text_bounded(left=left, bottom=bottom, right=right, top=top)
                finally:
                    textpage.close()
            finally:
                page.close()
        return (raw or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").strip()
