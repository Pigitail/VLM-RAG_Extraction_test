from __future__ import annotations

import pytest

from vlmrag.vlm.jsonparse import extract_json_object, extract_json_value, iter_json_candidates, strip_code_fences


def test_plain_object_and_array():
    assert extract_json_value('{"a": 1}') == {"a": 1}
    assert extract_json_value("[1, 2]") == [1, 2]


@pytest.mark.parametrize(
    "text",
    [
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        'Here is the result:\n```json\n{"a": 1}\n```\nLet me know if you need more.',
        "```JSON {\"a\": 1}```",
    ],
)
def test_code_fences(text):
    assert extract_json_value(text) == {"a": 1}


def test_truncated_fence_is_stripped():
    assert strip_code_fences('```json\n{"a": 1}') == '{"a": 1}'
    assert extract_json_value('```json\n{"a": 1}') == {"a": 1}


def test_prose_around_object():
    assert extract_json_value('Sure! {"field": "value"} Hope this helps {not json}') == {"field": "value"}


def test_braces_inside_strings_do_not_confuse_the_scanner():
    text = 'noise {"note": "use } and { carefully", "q": "say \\"hi\\" }"} trailing }'
    assert extract_json_value(text) == {"note": "use } and { carefully", "q": 'say "hi" }'}


def test_first_candidate_unparseable_then_valid_one():
    text = "{broken: true} and then {\"ok\": [1, {\"x\": \"]\"}]}"
    assert extract_json_value(text) == {"ok": [1, {"x": "]"}]}


def test_mismatched_closer_skips_candidate():
    candidates = list(iter_json_candidates('{"a": [1, 2}  {"b": 2}'))
    assert candidates[-1] == '{"b": 2}'
    assert extract_json_value('{"a": [1, 2}  {"b": 2}') == {"b": 2}


def test_garbled_and_empty_inputs_return_none():
    for text in ["", "   ", "no json here", "{unterminated", None, "{'single': 'quotes'}"]:
        assert extract_json_value(text) is None


def test_scalar_only_when_whole_payload():
    assert extract_json_value("42") == 42
    assert extract_json_value("the answer is 42") is None


def test_extract_object_skips_leading_array():
    assert extract_json_object('[1, 2] then {"k": "v"}') == {"k": "v"}
    assert extract_json_object('[{"k": "v"}]') == {"k": "v"}
    assert extract_json_object("[1, 2]") is None
    assert extract_json_object("garbage") is None


def test_unicode_content_survives():
    assert extract_json_object('결과: {"제목": "도면 A", "rev": "B"}') == {"제목": "도면 A", "rev": "B"}
