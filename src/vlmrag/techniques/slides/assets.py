"""Crop-aware image assets for viewers (optional).

A RAG answer about a manual slide is far more useful when the viewer can show
the screenshot it came from. The picture stored in a PPTX, however, is the
*source* image: when an author cropped a screenshot in PowerPoint to show only
a dialog, the full screenshot is still embedded. Exporting the raw blob would
show the user the wrong thing, so assets are written as the *visible* region:
the crop fractions are applied to the source pixels.

Windows metafiles (WMF/EMF) are common in older manuals and cannot be shown by
browsers; they are converted to PNG with LibreOffice and then cropped.

Every step degrades instead of failing: a crop that cannot be applied keeps
the full image, a metafile that cannot be converted keeps the original bytes,
and each such fallback is reported so it lands in the chunk's provenance.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from PIL import Image, ImageChops

from .layout import Block, PictureInfo

__all__ = ["SavedAsset", "crop_box", "crop_image", "trim_uniform_border", "save_picture", "VECTOR_EXTS"]

log = logging.getLogger(__name__)

VECTOR_EXTS = ("wmf", "emf")
_PIL_FORMATS = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "gif": "GIF", "bmp": "BMP", "tif": "TIFF", "tiff": "TIFF"}

# convert(src, target_ext, out_dir, *, timeout_s, candidates) -> Path ; injectable for tests
Converter = Callable[..., Path]


@dataclass
class SavedAsset:
    """One written asset. ``name`` is relative to the asset root (POSIX separators)."""

    name: str
    path: Path
    ext: str
    cropped: bool = False
    converted: bool = False
    fallbacks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "asset": self.name,
            "ext": self.ext,
            "cropped": self.cropped,
            "converted": self.converted,
            "fallbacks": list(self.fallbacks),
        }


def crop_box(size: tuple[int, int], crop: Sequence[float]) -> tuple[int, int, int, int]:
    """Pixel box ``(left, top, right, bottom)`` of the visible region; raises on an empty box."""
    width, height = size
    left = int(round(width * crop[0]))
    top = int(round(height * crop[1]))
    right = int(round(width * (1 - crop[2])))
    bottom = int(round(height * (1 - crop[3])))
    if right <= left or bottom <= top:
        raise ValueError(f"invalid crop box {(left, top, right, bottom)} for image size {size}")
    return left, top, right, bottom


def crop_image(data: bytes, crop: Sequence[float], ext: str) -> bytes:
    """Apply PowerPoint crop fractions to encoded image bytes, keeping the source format."""
    with Image.open(io.BytesIO(data)) as source:
        source.load()
        fmt = source.format or _PIL_FORMATS.get(ext.lower(), "PNG")
        region = source.crop(crop_box(source.size, crop))
        if fmt == "JPEG" and region.mode not in ("RGB", "L"):
            region = region.convert("RGB")
        buffer = io.BytesIO()
        region.save(buffer, format=fmt)
        return buffer.getvalue()


def trim_uniform_border(data: bytes, tolerance: int = 8) -> bytes:
    """Remove a uniform margin (the colour of the top-left pixel) around a PNG.

    LibreOffice opens a metafile as a drawing and exports the whole *page*,
    with the graphic placed somewhere on it. Without trimming, the asset would
    be a mostly blank page, and applying PowerPoint's crop fractions to it
    would cut the wrong region.
    """
    with Image.open(io.BytesIO(data)) as source:
        image = source.convert("RGB")
    background = Image.new("RGB", image.size, image.getpixel((0, 0)))
    diff = ImageChops.difference(image, background).convert("L").point(lambda v: 255 if v > tolerance else 0)
    box = diff.getbbox()
    if not box or box == (0, 0, image.width, image.height):
        return data
    buffer = io.BytesIO()
    image.crop(box).save(buffer, format="PNG")
    return buffer.getvalue()


def _convert_vector(
    info: PictureInfo,
    stem: str,
    *,
    converter: Converter,
    timeout_s: float,
    candidates: Optional[Sequence[str]],
) -> bytes:
    work = Path(tempfile.mkdtemp(prefix="vlmrag_img_"))
    try:
        source = work / f"{stem}.{info.ext}"
        source.write_bytes(info.blob)
        kwargs: dict = {"timeout_s": timeout_s}
        if candidates is not None:
            kwargs["candidates"] = candidates
        output = converter(source, "png", work / "out", **kwargs)
        return trim_uniform_border(Path(output).read_bytes())
    finally:
        shutil.rmtree(work, ignore_errors=True)


def save_picture(
    block: Block,
    slide_dir: Path,
    asset_root: Path,
    *,
    converter: Optional[Converter] = None,
    timeout_s: float = 60.0,
    candidates: Optional[Sequence[str]] = None,
) -> SavedAsset:
    """Write the visible region of an image block as ``img_NNN.<ext>`` under ``slide_dir``.

    Raster images keep their format; metafiles become PNG. Fallbacks:
    ``crop_failed`` (full image kept), ``vector_conversion_failed`` (original
    metafile bytes kept).
    """
    info = block.picture
    if info is None:
        raise ValueError(f"block {block.order} is not an image block")
    if converter is None:
        from ...render.office import convert as converter  # lazy: LibreOffice only needed for metafiles

    slide_dir.mkdir(parents=True, exist_ok=True)
    stem = f"img_{block.image_index or block.order:03d}"
    fallbacks: list[str] = []
    converted = cropped = False
    data, ext = info.blob, info.ext

    if info.ext in VECTOR_EXTS:
        try:
            data = _convert_vector(info, stem, converter=converter, timeout_s=timeout_s, candidates=candidates)
            ext, converted = "png", True
        except Exception as exc:  # noqa: BLE001 - keep the original metafile
            log.warning("block %d: metafile conversion failed, keeping original bytes: %s", block.order, exc)
            fallbacks.append("vector_conversion_failed")

    if info.cropped and (ext not in VECTOR_EXTS):
        try:
            data = crop_image(data, info.crop, ext)
            cropped = True
        except Exception as exc:  # noqa: BLE001 - keep the full image
            log.warning("block %d: crop failed, keeping the full image: %s", block.order, exc)
            fallbacks.append("crop_failed")

    path = slide_dir / f"{stem}.{ext}"
    path.write_bytes(data)
    name = Path(os.path.relpath(path, asset_root)).as_posix()
    return SavedAsset(name=name, path=path, ext=ext, cropped=cropped, converted=converted, fallbacks=fallbacks)
