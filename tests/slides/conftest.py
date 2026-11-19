"""Fixtures for the slides technique tests.

* ``synthetic_deck`` generates the synthetic manual once per session (in a temp
  dir) and returns ``(deck_path, truth)``.
* ``truth_responder(truth)`` builds a ScriptedBackend responder that answers
  like a well-behaved model would: it reads the layout blocks from the request
  (block IDs are an output of the technique, so the truth file names shapes
  instead) and returns the truth descriptions under those IDs.
* ``fake_render`` replaces LibreOffice rendering with small blank PNGs, so the
  orchestration can be tested on machines without LibreOffice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest
from PIL import Image

from vlmrag.render.images import to_png
from vlmrag.synthetic.slides import generate
from vlmrag.techniques.slides import pipeline
from vlmrag.techniques.slides.prompt import IMAGE_BLOCKS_HEADER, LAYOUT_BLOCKS_HEADER
from vlmrag.vlm.types import VLMRequest


@pytest.fixture(scope="session")
def synthetic_deck(tmp_path_factory) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("slides_synthetic")
    [deck_path] = generate(out, seed=0)
    truth_path = deck_path.with_name(f"{deck_path.stem}.truth.json")
    return deck_path, json.loads(truth_path.read_text(encoding="utf-8"))


@pytest.fixture
def deck_path(synthetic_deck) -> Path:
    return synthetic_deck[0]


@pytest.fixture
def truth(synthetic_deck) -> dict:
    return synthetic_deck[1]


def request_blocks(request: VLMRequest, header: str = LAYOUT_BLOCKS_HEADER) -> list[dict]:
    """Decode the JSON block list a request carries under ``header`` (``[]`` if absent)."""
    for part in request.parts:
        if isinstance(part, str) and part.startswith(header + "\n"):
            return json.loads(part.split("\n", 1)[1])
    return []


def answer_from_truth(truth: dict, request: VLMRequest) -> dict:
    slide = next(s for s in truth["slides"] if s["slide"] == request.page)
    by_name = {b["shape_name"]: b["block_order"] for b in request_blocks(request)}
    images = {b["shape_name"] for b in request_blocks(request, IMAGE_BLOCKS_HEADER)}
    items = []
    for block in slide["image_blocks"]:
        assert block["shape_name"] in images, f"{block['shape_name']} missing from the image block list"
        items.append({
            "block_order": by_name[block["shape_name"]],
            "related_block_orders": [by_name[n] for n in block["related_shapes"] if n in by_name],
            "image_role": block["image_role"],
            "description": block["description"],
            "description_en": block["description_en"],
            "description_ko": block["description_ko"],
            "confidence": "high",
        })
    return {
        "visual_summary": slide["visual_summary"],
        "image_descriptions": items,
        "flow_notes": slide["flow_notes"],
        "notes": "",
    }


@pytest.fixture
def truth_responder() -> Callable[[dict], Callable[[VLMRequest], Any]]:
    def factory(truth: dict) -> Callable[[VLMRequest], Any]:
        return lambda request: answer_from_truth(truth, request)

    return factory


@pytest.fixture
def fake_render(monkeypatch) -> list[int]:
    """Patch the renderer; returns the list of slide numbers that were 'rendered'."""
    rendered: list[int] = []
    png = to_png(Image.new("RGB", (64, 36), "white"))

    def fake(path, deck, numbers, settings, *, dpi=pipeline.RENDER_DPI, timeout_s=pipeline.CONVERT_TIMEOUT_S):
        rendered.extend(numbers)
        return {n: (png, n) for n in numbers}, {}

    monkeypatch.setattr(pipeline, "render_slides", fake)
    return rendered
