from __future__ import annotations

import io

import pytest
from PIL import Image

from vlmrag.render.images import Tile, clamp_bbox, fit_max_pixels, format_bbox, tile_grid, tile_label, to_png, upscale


def test_tile_grid_without_overlap_partitions_region():
    tiles = tile_grid(2, 2, 0.0)
    assert [(t.row, t.col) for t in tiles] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert tiles[0].bbox == (0.0, 0.0, 0.5, 0.5)
    assert tiles[3].bbox == (0.5, 0.5, 1.0, 1.0)


def test_tile_grid_overlap_is_fraction_of_tile_and_clamped():
    tiles = tile_grid(3, 4, 0.12)
    assert len(tiles) == 12
    tile_w, tile_h = 1 / 4, 1 / 3
    first = tiles[0].bbox
    assert first[0] == 0.0 and first[1] == 0.0  # clamped at the page edge
    assert first[2] == pytest.approx(tile_w + tile_w * 0.12)
    assert first[3] == pytest.approx(tile_h + tile_h * 0.12)
    middle = tiles[5].bbox  # row 1, col 1
    assert middle[0] == pytest.approx(tile_w - tile_w * 0.12)
    assert middle[2] == pytest.approx(2 * tile_w + tile_w * 0.12)
    last = tiles[-1].bbox
    assert last[2] == 1.0 and last[3] == 1.0
    # Neighbours share a band of 2 * overlap * tile width.
    assert tiles[0].bbox[2] - tiles[1].bbox[0] == pytest.approx(2 * 0.12 * tile_w)


def test_tile_grid_inside_region():
    region = (0.44, 0.62, 1.0, 1.0)
    tiles = tile_grid(2, 3, 0.10, region)
    assert len(tiles) == 6
    for tile in tiles:
        x0, y0, x1, y1 = tile.bbox
        assert region[0] <= x0 < x1 <= region[2]
        assert region[1] <= y0 < y1 <= region[3]
    assert tiles[0].bbox[:2] == (0.44, 0.62)
    assert tiles[-1].bbox[2:] == (1.0, 1.0)


def test_tile_grid_validation():
    with pytest.raises(ValueError):
        tile_grid(0, 1, 0.1)
    with pytest.raises(ValueError):
        tile_grid(1, 1, -0.1)


def test_labels():
    tile = Tile(0, 1, (0.24, 0.0, 0.52, 0.37))
    assert format_bbox(tile.bbox) == "x=0.24-0.52 y=0.00-0.37"
    assert tile_label(tile, 400) == "TILE r1c2 x=0.24-0.52 y=0.00-0.37 dpi=400"
    assert tile_label(tile, prefix="META") == "META r1c2 x=0.24-0.52 y=0.00-0.37"


def test_clamp_bbox():
    assert clamp_bbox((-0.1, 0.2, 1.4, 0.9)) == (0.0, 0.2, 1.0, 0.9)
    with pytest.raises(ValueError):
        clamp_bbox((0.5, 0.5, 0.4, 0.9))


def test_upscale_and_noop():
    image = Image.new("RGB", (10, 20), "white")
    assert upscale(image, 1.0) is image
    assert upscale(image, 0.5) is image
    assert upscale(image, 3.0).size == (30, 60)


def test_to_png_round_trip():
    image = Image.new("RGB", (5, 5), (255, 0, 0))
    data = to_png(image)
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert Image.open(io.BytesIO(data)).getpixel((2, 2)) == (255, 0, 0)
    assert to_png(Image.new("CMYK", (2, 2))).startswith(b"\x89PNG")


def test_fit_max_pixels():
    image = Image.new("RGB", (4000, 3000))
    fitted = fit_max_pixels(image, 1_000_000)
    assert fitted.width * fitted.height <= 1_000_000
    assert abs(fitted.width / fitted.height - 4 / 3) < 0.01
    small = Image.new("RGB", (10, 10))
    assert fit_max_pixels(small, 1_000_000) is small
