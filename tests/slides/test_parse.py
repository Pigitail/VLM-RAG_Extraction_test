from __future__ import annotations

import json

import pytest

from vlmrag.techniques.slides.parse import coerce_order, coerce_order_list, normalize_image_role, parse_visual


def _item(**overrides):
    item = {
        "block_order": 3,
        "related_block_orders": [1, 2],
        "image_role": "main_screenshot",
        "description": "Order entry screen with vendor field and Save button.",
        "description_en": "Order entry screen.",
        "description_ko": "오더 입력 화면입니다.",
        "confidence": "high",
    }
    item.update(overrides)
    return item


@pytest.mark.parametrize(
    "value, expected",
    [(3, 3), (3.9, 3), ("4", 4), ("block 5", 5), ("#12a", 12), (None, 0), ("none", 0), (True, 0), (-2, -2)],
)
def test_coerce_order(value, expected):
    assert coerce_order(value) == expected


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_coerce_order_rejects_non_finite_numbers(value):
    assert coerce_order(value) == 0
    assert coerce_order_list([value, 2]) == [2]


def test_parse_survives_infinity_from_json_text():
    visual = parse_visual('{"image_descriptions": [{"block_order": 1e400, "description": "x"},'
                          ' {"block_order": Infinity, "description": "y"}, {"block_order": 2, "description": "z"}]}')
    assert [d["block_order"] for d in visual.image_descriptions] == [2]
    assert "dropped_descriptions:2" in visual.issues


def test_coerce_order_list_handles_scalars_dicts_and_duplicates():
    assert coerce_order_list([3, "3", "block 4", 0, -1, {"order": 6}, {"block_id": "7"}, {"x": 1}]) == [3, 4, 6, 7]
    assert coerce_order_list("5") == [5]
    assert coerce_order_list({"block_order": 2}) == [2]
    assert coerce_order_list(None) == []


@pytest.mark.parametrize(
    "value, expected",
    [
        ("main_screenshot", "main_screenshot"),
        ("Screenshot", "main_screenshot"),
        ("screen capture", "main_screenshot"),
        ("Zoomed", "detail_crop"),
        ("partial-screenshot", "detail_crop"),
        ("supporting image", "inline_image"),
        ("Button icon", "ui_icon"),
        ("ui_icon", "ui_icon"),
        ("logo", ""),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_image_role(value, expected):
    assert normalize_image_role(value) == expected


def test_parse_valid_answer():
    visual = parse_visual({
        "visual_summary": " Entry screen. ",
        "image_descriptions": [_item()],
        "flow_notes": ["Arrow to Save.", "arrow  to save.", ""],
        "notes": "",
    })
    assert visual.visual_summary == "Entry screen."
    assert visual.flow_notes == ["Arrow to Save."]
    assert visual.image_descriptions == [{
        "block_order": 3,
        "related_block_orders": [1, 2],
        "description": "Order entry screen with vendor field and Save button.",
        "description_en": "Order entry screen.",
        "description_ko": "오더 입력 화면입니다.",
        "confidence": "high",
        "image_role": "main_screenshot",
    }]
    assert visual.issues == [] and not visual.is_empty()


def test_parse_raw_text_with_fences_and_prose():
    raw = "Here you go:\n```json\n" + json.dumps({"visual_summary": "S", "image_descriptions": [_item()]}) + "\n```"
    visual = parse_visual(raw)
    assert visual.visual_summary == "S" and len(visual.image_descriptions) == 1


def test_parse_unwraps_a_list_wrapped_object_and_single_dict_items():
    raw = json.dumps([{"visual_summary": "", "image_descriptions": _item(block_order="block 4")}])
    visual = parse_visual(raw)
    assert [d["block_order"] for d in visual.image_descriptions] == [4]


@pytest.mark.parametrize("raw", ["", "not json at all", "{'single': 'quotes'", None, 42])
def test_parse_malformed_returns_none(raw):
    assert parse_visual(raw) is None


def test_parse_drops_items_without_order_or_description():
    visual = parse_visual({
        "image_descriptions": [
            _item(block_order=0),
            _item(block_order="none"),
            _item(description="   "),
            "not an object",
            _item(block_order=None, order=6),
        ],
    })
    assert [d["block_order"] for d in visual.image_descriptions] == [6]
    assert "dropped_descriptions:4" in visual.issues


def test_parse_coerces_invalid_confidence_and_role():
    visual = parse_visual({"image_descriptions": [
        _item(confidence="very sure", image_role="logo"),
        _item(block_order=4, image_role=None, role="zoom", confidence="MEDIUM"),
    ]})
    first, second = visual.image_descriptions
    assert first["confidence"] == "low" and "image_role" not in first
    assert second["confidence"] == "medium" and second["image_role"] == "detail_crop"
    assert "invalid_confidence:3" in visual.issues and "invalid_image_role:3" in visual.issues


def test_parse_empty_object_is_empty_not_none():
    visual = parse_visual("{}")
    assert visual is not None and visual.is_empty()


def test_parse_tolerates_wrong_container_types():
    visual = parse_visual({"visual_summary": None, "image_descriptions": "oops", "flow_notes": "Single note", "notes": 5})
    assert visual.image_descriptions == [] and visual.flow_notes == ["Single note"] and visual.notes == "5"
