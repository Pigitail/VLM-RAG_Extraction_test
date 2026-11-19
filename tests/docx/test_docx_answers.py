"""Strict validation of model answers and the deterministic fallback."""

from __future__ import annotations

import pytest

from vlmrag.techniques.docx.answers import fallback_distribution, parse_image_answer, validate_table_answer
from vlmrag.techniques.docx.xmlsplit import MultiPageTable


@pytest.fixture
def table() -> MultiPageTable:
    return MultiPageTable(
        table_id="t1",
        start_page=2,
        end_page=4,
        rows=[["Item", "Value"], ["a", "1"], ["b", "2"], ["c", "3"], ["d", "4"]],
        row_pages=[2, 2, 3, 3, 4],
        preceding_paragraphs=["4 Results", "Table 2:   Measured  values"],
    )


def answer(dist, **title) -> dict:
    base = {"has_title": False, "title_text": "", "title_source": "none", "title_confidence": "low"}
    base.update(title)
    base["row_distribution"] = dist
    return base


# --------------------------------------------------------------------------- image answers


def test_image_answer_ok_with_notes():
    parsed = parse_image_answer({
        "image_descriptions": [
            {"location_hint": "top diagram", "description": " Flow chart with three steps. "},
            {"location_hint": "x", "description": ""},
            "not a dict",
        ],
        "notes": "Logo in the corner.",
    })
    assert parsed.status == "ok"
    assert parsed.descriptions == [
        {"location_hint": "top diagram", "description": "Flow chart with three steps."},
        {"location_hint": "notes", "description": "Logo in the corner."},
    ]


@pytest.mark.parametrize(
    "value, status",
    [
        (None, "unparseable"),
        ({}, "invalid"),
        ([], "invalid"),
        ({"image_descriptions": "a picture"}, "invalid"),
        ({"image_descriptions": [], "notes": ""}, "no_visuals"),
        ({"image_descriptions": [{"location_hint": "x", "description": "  "}], "notes": ""}, "no_visuals"),
    ],
)
def test_image_answer_statuses(value, status):
    parsed = parse_image_answer(value)
    assert parsed.status == status
    assert parsed.descriptions == []


# --------------------------------------------------------------------------- table answers


def test_valid_answer_is_accepted_and_missing_pages_are_filled(table):
    result, reason = validate_table_answer(answer({"2": [0, 1, 2], "4": ["3", 4]}), table)
    assert reason == ""
    assert result.row_pages == {2: [0, 1, 2], 3: [], 4: [3, 4]}
    assert result.title == ""


@pytest.mark.parametrize(
    "value, reason_prefix",
    [
        (None, "unparseable"),
        ([], "not_an_object"),
        ({"row_distribution": [[0, 1]]}, "row_distribution_not_an_object"),
        (answer({"two": [0, 1, 2, 3, 4]}), "non_integer_page"),
        (answer({"1": [0], "2": [1, 2, 3, 4]}), "page_out_of_range"),
        (answer({"5": [0], "2": [1, 2, 3, 4]}), "page_out_of_range"),
        (answer({"2": "0-4"}), "page_value_not_a_list"),
        (answer({"2": [0, 1, True], "3": [2, 3, 4]}), "non_integer_row"),
        (answer({"2": [0, 1.5], "3": [2, 3, 4]}), "non_integer_row"),
        (answer({"2": [0, 1, 2], "3": [2, 3, 4]}), "duplicate_rows"),
        (answer({"2": [0, 1], "3": [3, 4]}), "row_set_mismatch"),
        (answer({"2": [0, 1, 2], "3": [3, 4, 5]}), "row_set_mismatch"),
        (answer({"2": [0, 3], "3": [1, 2, 4]}), "rows_not_in_page_order"),
        (answer({"2": [0, 1], "2.0": [2, 3, 4]}), "non_integer_page"),
    ],
)
def test_invalid_row_assignments_are_rejected(table, value, reason_prefix):
    result, reason = validate_table_answer(value, table)
    assert result is None
    assert reason.startswith(reason_prefix)


def test_integral_floats_and_digit_strings_are_accepted(table):
    result, _ = validate_table_answer(answer({"2": [0.0, "1"], "3": [2, 3], "4": [4]}), table)
    assert result.row_pages == {2: [0, 1], 3: [2, 3], 4: [4]}


def test_title_from_preceding_paragraph_is_matched_ignoring_case_and_spacing(table):
    value = answer(
        {"2": [0, 1, 2], "3": [3, 4], "4": []},
        has_title=True, title_text="table 2: measured values", title_source="preceding_paragraph",
        title_confidence="High",
    )
    result, _ = validate_table_answer(value, table)
    assert (result.title, result.title_source, result.title_confidence) == (
        "table 2: measured values", "preceding_paragraph", "high",
    )
    assert not result.title_rejected


def test_title_from_first_row(table):
    value = answer(
        {"2": [0, 1, 2, 3, 4]}, has_title=True, title_text="Item", title_source="first_row", title_confidence="medium"
    )
    result, _ = validate_table_answer(value, table)
    assert result.title == "Item" and result.title_source == "first_row"


@pytest.mark.parametrize(
    "title",
    [
        {"has_title": True, "title_text": "Invented caption", "title_source": "preceding_paragraph",
         "title_confidence": "high"},
        {"has_title": True, "title_text": "4 Results", "title_source": "first_row", "title_confidence": "high"},
    ],
)
def test_hallucinated_or_misattributed_title_is_dropped_but_rows_are_kept(table, title):
    result, reason = validate_table_answer(answer({"2": [0, 1, 2, 3, 4]}, **title), table)
    assert reason == ""
    assert result.title == "" and result.title_source == "none" and result.title_rejected
    assert result.row_pages[2] == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    "title",
    [
        {"has_title": True, "title_text": "4 Results", "title_source": "preceding_paragraph", "title_confidence": "low"},
        {"has_title": True, "title_text": "4 Results", "title_source": "somewhere", "title_confidence": "high"},
        {"has_title": True, "title_text": "4 Results", "title_source": "preceding_paragraph", "title_confidence": "sure"},
        {"has_title": False, "title_text": "4 Results", "title_source": "preceding_paragraph",
         "title_confidence": "high"},
        {"has_title": "yes", "title_text": "4 Results", "title_source": "preceding_paragraph",
         "title_confidence": "high"},
    ],
)
def test_low_confidence_or_malformed_titles_are_ignored(table, title):
    result, _ = validate_table_answer(answer({"2": [0, 1, 2, 3, 4]}, **title), table)
    assert result.title == ""
    assert not result.title_rejected


def test_fallback_uses_the_editor_row_markers(table):
    assert fallback_distribution(table) == {2: [0, 1], 3: [2, 3], 4: [4]}
