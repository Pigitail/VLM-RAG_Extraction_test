from __future__ import annotations

import io
import struct
from pathlib import Path

import pytest
from PIL import Image

from vlmrag.techniques.slides.assets import crop_box, crop_image, save_picture, trim_uniform_border
from vlmrag.techniques.slides.layout import Block, PictureInfo, sniff_vector_ext


def _encoded(size=(200, 100), fmt="PNG", color=(10, 120, 200)) -> bytes:
    image = Image.new("RGB", size, color)
    image.paste((250, 20, 20), (150, 60, 200, 100))  # bottom-right patch
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def _block(blob: bytes, ext: str, crop=(0.0, 0.0, 0.0, 0.0), index=1) -> Block:
    return Block(order=5, type="image", picture=PictureInfo(blob=blob, ext=ext, source_sha256="0" * 64, crop=crop),
                 image_index=index)


def minimal_emf(width: int = 96, height: int = 96) -> bytes:
    """A tiny valid EMF: header, a green solid brush and an ellipse."""
    records = [
        struct.pack("<II", 39, 24) + struct.pack("<IIII", 1, 0, 0x0048A016, 0),  # CREATEBRUSHINDIRECT
        struct.pack("<III", 37, 12, 1),  # SELECTOBJECT
        struct.pack("<II", 42, 24) + struct.pack("<iiii", 4, 4, width - 4, height - 4),  # ELLIPSE
        struct.pack("<II", 14, 20) + struct.pack("<III", 0, 16, 20),  # EOF
    ]
    body = b"".join(records)
    header = struct.pack("<II", 1, 88)
    header += struct.pack("<iiii", 0, 0, width - 1, height - 1)
    header += struct.pack("<iiii", 0, 0, int(width * 26.4583), int(height * 26.4583))
    header += struct.pack("<IIII", 0x464D4520, 0x00010000, 88 + len(body), len(records) + 1)
    header += struct.pack("<HH", 2, 0) + struct.pack("<III", 0, 0, 0)
    header += struct.pack("<ii", 1920, 1080) + struct.pack("<ii", 508, 286)
    return header + body


def test_crop_box():
    assert crop_box((200, 100), (0.25, 0.1, 0.25, 0.2)) == (50, 10, 150, 80)
    with pytest.raises(ValueError):
        crop_box((200, 100), (0.6, 0.0, 0.5, 0.0))


@pytest.mark.parametrize("fmt, ext", [("PNG", "png"), ("JPEG", "jpg")])
def test_crop_image_keeps_format(fmt, ext):
    out = crop_image(_encoded(fmt=fmt), (0.75, 0.6, 0.0, 0.0), ext)
    with Image.open(io.BytesIO(out)) as image:
        assert image.format == fmt and image.size == (50, 40)
        r, g, b = image.convert("RGB").getpixel((25, 20))
        assert r > 200 and g < 80


def test_save_uncropped_raster_writes_blob_unchanged(tmp_path):
    blob = _encoded()
    saved = save_picture(_block(blob, "png"), tmp_path / "deck" / "slide_002", tmp_path)
    assert saved.name == "deck/slide_002/img_001.png"
    assert saved.path.read_bytes() == blob and not saved.cropped and saved.fallbacks == []


def test_save_cropped_raster_writes_visible_region_only(tmp_path):
    saved = save_picture(_block(_encoded(), "png", crop=(0.5, 0.0, 0.0, 0.5), index=2), tmp_path / "s", tmp_path)
    assert saved.cropped and saved.name == "s/img_002.png"
    with Image.open(saved.path) as image:
        assert image.size == (100, 50)


def test_invalid_crop_falls_back_to_full_image(tmp_path):
    blob = _encoded()
    saved = save_picture(_block(blob, "png", crop=(0.7, 0.0, 0.6, 0.0)), tmp_path / "s", tmp_path)
    assert saved.fallbacks == ["crop_failed"] and saved.path.read_bytes() == blob


def test_metafile_is_converted_trimmed_and_cropped_with_injected_converter(tmp_path):
    page = Image.new("RGB", (400, 600), "white")
    page.paste((0, 128, 0), (100, 200, 300, 300))  # graphic placed on a page, like LibreOffice does
    calls = []

    def converter(src, target_ext, out_dir, **kwargs):
        calls.append((Path(src).suffix, target_ext, kwargs))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out = Path(out_dir) / (Path(src).stem + ".png")
        page.save(out)
        return out

    emf = minimal_emf()
    saved = save_picture(_block(emf, "emf", crop=(0.5, 0.0, 0.0, 0.0)), tmp_path / "s", tmp_path,
                         converter=converter, timeout_s=12, candidates=["soffice-test"])
    assert calls == [(".emf", "png", {"timeout_s": 12, "candidates": ["soffice-test"]})]
    assert saved.converted and saved.cropped and saved.ext == "png" and saved.name.endswith(".png")
    with Image.open(saved.path) as image:
        assert image.size == (100, 100)  # trimmed to 200x100, then the right half kept


def test_metafile_conversion_failure_keeps_original_bytes(tmp_path):
    def converter(*args, **kwargs):
        raise RuntimeError("no LibreOffice")

    emf = minimal_emf()
    saved = save_picture(_block(emf, "emf", crop=(0.2, 0.0, 0.0, 0.0)), tmp_path / "s", tmp_path, converter=converter)
    assert saved.fallbacks == ["vector_conversion_failed"]
    assert saved.ext == "emf" and saved.path.read_bytes() == emf and not saved.cropped


def test_trim_uniform_border():
    image = Image.new("RGB", (50, 40), "white")
    image.paste((0, 0, 0), (10, 5, 20, 25))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    with Image.open(io.BytesIO(trim_uniform_border(buffer.getvalue()))) as trimmed:
        assert trimmed.size == (10, 20)
    blank = io.BytesIO()
    Image.new("RGB", (5, 5), "white").save(blank, format="PNG")
    assert trim_uniform_border(blank.getvalue()) == blank.getvalue()


def test_minimal_emf_is_recognised():
    assert sniff_vector_ext(minimal_emf()) == "emf"


@pytest.mark.soffice
def test_real_metafile_conversion(tmp_path, soffice_path):
    saved = save_picture(_block(minimal_emf(), "emf"), tmp_path / "s", tmp_path, candidates=[soffice_path])
    assert saved.converted and saved.fallbacks == []
    with Image.open(saved.path) as image:
        assert image.format == "PNG"
        assert max(image.size) < 600  # trimmed: not a whole exported page
        r, g, b = image.convert("RGB").getpixel((image.width // 2, image.height // 2))
        assert g > 120 and r < 80 and b < 120
