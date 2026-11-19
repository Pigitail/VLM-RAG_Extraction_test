from __future__ import annotations

import json
import re

import pytest

from vlmrag.techniques.slides.layout import Block, PictureInfo, SlideLayout, read_deck
from vlmrag.techniques.slides.prompt import (
    BLOCK_TEXT_LIMIT,
    IMAGE_BLOCKS_HEADER,
    IMAGE_ROLES,
    LAYOUT_BLOCKS_HEADER,
    PROMPT_VERSION,
    SLIDE_TEXT_HEADER,
    SLIDE_TEXT_LIMIT,
    SLIDE_VISUAL_PROMPT,
    TASK,
    VISUAL_SCHEMA,
    build_request,
    compact_block,
    render_label,
)
from vlmrag.vlm.backends import replay_key
from vlmrag.vlm.types import ImageInput

PNG = b"\x89PNG\r\n\x1a\nfake"


def _slide(**kwargs) -> SlideLayout:
    slide = SlideLayout(number=3, title="Screen", **kwargs)
    return slide


def _image_block(order=2, crop=(0.0, 0.0, 0.0, 0.0)) -> Block:
    return Block(
        order=order,
        type="image",
        shape_id="7",
        shape_name="Screen A",
        bbox_norm={"x": 0.1, "y": 0.2, "w": 0.5, "h": 0.4},
        picture=PictureInfo(blob=b"x", ext="png", source_sha256="ab" * 32, crop=crop),
    )


def _walk_objects(schema):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            yield schema
        for value in schema.values():
            yield from _walk_objects(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from _walk_objects(value)


def test_prompt_identity():
    assert TASK == "slides.visual"
    assert re.fullmatch(r"slides\.visual/\d+", PROMPT_VERSION)


def test_schema_is_strict_mode_compatible():
    objects = list(_walk_objects(VISUAL_SCHEMA))
    assert len(objects) == 2
    for obj in objects:
        assert obj["additionalProperties"] is False
        assert sorted(obj["required"]) == sorted(obj["properties"])
    item = VISUAL_SCHEMA["properties"]["image_descriptions"]["items"]
    assert item["properties"]["image_role"]["enum"] == list(IMAGE_ROLES)
    assert item["properties"]["confidence"]["enum"] == ["low", "medium", "high"]


def test_prompt_keeps_the_output_contract_and_rules():
    for key in ("visual_summary", "image_descriptions", "block_order", "related_block_orders", "image_role",
                "description_en", "description_ko", "confidence", "flow_notes", "notes"):
        assert key in SLIDE_VISUAL_PROMPT
    for role in IMAGE_ROLES:
        assert role in SLIDE_VISUAL_PROMPT
    assert "Do NOT perform full OCR" in SLIDE_VISUAL_PROMPT
    assert "annotation-only" in SLIDE_VISUAL_PROMPT
    assert "bbox_norm" in SLIDE_VISUAL_PROMPT
    # The short captions are display text for a viewer, not summaries "of the UI".
    assert "summary for display in a web UI" in SLIDE_VISUAL_PROMPT
    assert "Korean web UI display summary" in SLIDE_VISUAL_PROMPT


def test_request_parts_order_and_limits():
    long_text = "x" * (SLIDE_TEXT_LIMIT + 500)
    text_block = Block(order=1, type="text", shape_name="Body", text_content="y" * (BLOCK_TEXT_LIMIT + 50),
                       bbox_norm={"x": 0, "y": 0, "w": 1, "h": 0.2})
    slide = _slide(text_only=long_text, text_and_table=long_text, blocks=[text_block, _image_block()])
    request = build_request(slide, 9, PNG, dpi=180, doc_id="abc", source_name="deck.pptx",
                            reasoning_effort="low", timeout_s=240, max_output_tokens=4000)
    parts = request.parts
    assert parts[0] == SLIDE_VISUAL_PROMPT
    assert parts[1] == "Slide context: slide 3 of 9."
    assert parts[2] == f"{SLIDE_TEXT_HEADER}\n" + "x" * SLIDE_TEXT_LIMIT
    layout_blocks = json.loads(parts[3].split("\n", 1)[1])
    assert parts[3].startswith(LAYOUT_BLOCKS_HEADER) and len(layout_blocks) == 2
    assert len(layout_blocks[0]["text_content"]) == BLOCK_TEXT_LIMIT
    image_blocks = json.loads(parts[4].split("\n", 1)[1])
    assert parts[4].startswith(IMAGE_BLOCKS_HEADER) and [b["block_order"] for b in image_blocks] == [2]
    assert isinstance(parts[5], ImageInput) and parts[5].png == PNG
    assert parts[5].label == "SLIDE 3/9 whole-slide render x=0.00-1.00 y=0.00-1.00 dpi=180"
    assert len(parts) == 6
    assert (request.task, request.prompt_version, request.page) == (TASK, PROMPT_VERSION, 3)
    assert (request.timeout_s, request.max_output_tokens, request.reasoning_effort) == (240, 4000, "low")
    assert request.json_schema is VISUAL_SCHEMA and request.source_name == "deck.pptx"


def test_compact_image_block_carries_source_id_and_crop_but_no_paths():
    compact = compact_block(_image_block(crop=(0.1, 0.0, 0.2, 0.3)))
    assert compact == {
        "block_order": 2,
        "type": "image",
        "shape_name": "Screen A",
        "bbox_norm": {"x": 0.1, "y": 0.2, "w": 0.5, "h": 0.4},
        "source_image_id": "abababababab",
        "crop_norm": {"l": 0.1, "t": 0.0, "r": 0.2, "b": 0.3},
        "crop_area_norm": pytest.approx(0.7 * 0.7),
    }
    uncropped = compact_block(_image_block())
    assert "crop_norm" not in uncropped and uncropped["crop_area_norm"] == 1.0


def test_slide_without_text_or_images_sends_only_context_and_render():
    request = build_request(_slide(), 1, PNG, dpi=180)
    assert len(request.parts) == 3 and isinstance(request.parts[-1], ImageInput)


def test_render_label():
    assert render_label(1, 12, 180.0) == "SLIDE 1/12 whole-slide render x=0.00-1.00 y=0.00-1.00 dpi=180"


def test_requests_are_path_free_and_replay_stable(tmp_path, synthetic_deck):
    """Same deck generated in two folders -> identical replay keys; no paths leak into text parts."""
    from vlmrag.synthetic.slides import generate

    [other] = generate(tmp_path / "elsewhere", seed=0)
    keys = []
    for path in (synthetic_deck[0], other):
        deck = read_deck(path)
        slide = deck.slides[1]
        request = build_request(slide, deck.slide_count, PNG, dpi=180, source_name=path.name)
        for part in request.text_parts():
            assert str(tmp_path) not in part and ":\\" not in part and "/tmp" not in part
        keys.append(replay_key(request))
    assert keys[0] == keys[1]
