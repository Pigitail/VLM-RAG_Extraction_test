"""Failure paths: malformed, empty and invalid answers, failing passes, and their documented fallbacks."""

from __future__ import annotations

import json

import pypdfium2 as pdfium
import pytest

from vlmrag.config import Settings
from vlmrag.techniques.drawing import DrawingExtractionError, TitleBlockGateError, extract
from vlmrag.vlm.backends import NullBackend, ReplayMiss, ScriptedBackend
from vlmrag.vlm.client import VLMClient

from .conftest import BRACKET, HOUSING, LIGHT, REFLECTOR

SETTINGS = Settings(provider="null")


class BadRequest(Exception):
    """A non-retryable provider error (HTTP 400)."""

    status_code = 400


def _run(path, respond, config=LIGHT):
    client = VLMClient(ScriptedBackend(respond), sleep=lambda s: None)
    return extract(path, client, SETTINGS, config=config), client


def _tasks(client):
    return [r.task for r in client.backend.calls]


def test_vision_plain_text_answer_uses_line_fallback(drawing_docs, responder):
    text = "Sure! Here is what I can read.\nDrawing No.: EX-L300-1007\nRev: A\nTitle\nMOUNTING\nBRACKET\nScale: 1:1"
    chunks, client = _run(drawing_docs[BRACKET], responder({"drawing.vision": text, "drawing.normalize": "{}"}))
    assert "vision_text_fallback" in chunks[0].provenance["fallbacks"]
    assert _tasks(client).count("drawing.vision") == 1  # a non-empty answer is parsed as it is, never re-asked
    fields = chunks[0].metadata["title_block"]
    assert fields["drawing_number"] == "EX-L300-1007" and fields["title"] == "MOUNTING BRACKET"
    assert fields["material"] == "STAINLESS STEEL 1.4301"  # from the table pass


def test_vision_garbage_fails_the_gate(drawing_docs, responder):
    respond = responder({"drawing.vision": "I am unable to read this drawing.", "drawing.table": "{}", "drawing.normalize": "{}"})
    with pytest.raises(TitleBlockGateError) as info:
        _run(drawing_docs[BRACKET], respond)
    assert info.value.pairs_found == 0 and info.value.min_pairs == 3 and info.value.page == 1


def test_vision_empty_answer_is_a_hard_failure(drawing_docs, responder):
    calls = []

    def empty(request, default):
        calls.append(request.task)
        return ""

    with pytest.raises(DrawingExtractionError, match="empty answer after 3 attempt"):
        _run(drawing_docs[BRACKET], responder({"drawing.vision": empty}))
    assert calls == ["drawing.vision"] * 3


def test_empty_answers_are_re_asked_and_not_cached(drawing_docs, truths, responder, tmp_path):
    from vlmrag.vlm.cache import DiskCache

    answers = {"drawing.vision": ["", "  "], "drawing.normalize": [""]}

    def flaky(request, default):
        queue = answers[request.task]
        return queue.pop(0) if queue else default()

    respond = responder({"drawing.vision": flaky, "drawing.normalize": flaky})
    client = VLMClient(ScriptedBackend(respond), cache=DiskCache(tmp_path / "cache"), sleep=lambda s: None)
    chunks = extract(drawing_docs[BRACKET], client, SETTINGS, config=LIGHT)
    assert _tasks(client).count("drawing.vision") == 3 and _tasks(client).count("drawing.normalize") == 2
    assert chunks[0].provenance["fallbacks"] == []
    assert chunks[0].metadata["title_block"]["drawing_number"] == truths[BRACKET]["pages"][0]["title_block"]["drawing_number"]
    # A second run is served from the cache: the empty answers were never stored.
    again = VLMClient(ScriptedBackend(lambda r: ""), cache=DiskCache(tmp_path / "cache"), sleep=lambda s: None)
    assert extract(drawing_docs[BRACKET], again, SETTINGS, config=LIGHT)[0].retrieve_content == chunks[0].retrieve_content
    assert again.backend.calls == []


def test_vision_error_is_wrapped_with_context(drawing_docs, responder):
    def fail(request, default):
        raise BadRequest("content filter")

    with pytest.raises(DrawingExtractionError, match="vision pass failed: BadRequest: content filter") as info:
        _run(drawing_docs[BRACKET], responder({"drawing.vision": fail}))
    assert isinstance(info.value.__cause__, BadRequest)


@pytest.mark.parametrize(
    "answer, fallback",
    [
        ("raise", "table_pass_failed"),
        ("", "table_pass_empty"),
        ("The table is too blurry.", "table_text_fallback"),
    ],
)
def test_table_failures_fall_back_to_vision(drawing_docs, truths, responder, answer, fallback):
    def table(request, default):
        if answer == "raise":
            raise BadRequest("table refused")
        return answer

    chunks, _ = _run(drawing_docs[BRACKET], responder({"drawing.table": table}))
    chunk = chunks[0]
    assert fallback in chunk.provenance["fallbacks"]
    assert chunk.metadata["table_rows"] == [] and chunk.metadata["table_pairs"] == []
    assert chunk.metadata["title_block"]["drawing_number"] == truths[BRACKET]["pages"][0]["title_block"]["drawing_number"]
    table_call = chunk.provenance["calls"][1]
    assert table_call["task"] == "drawing.table"
    assert table_call["status"] == {"raise": "error", "": "empty"}.get(answer, "ok")
    if answer == "raise":
        assert "error" in table_call
    assert "drawing.table" in chunk.provenance["prompt_versions"]


@pytest.mark.parametrize(
    "answer, fallback",
    [
        ("raise", "normalize_failed"),
        ("", "normalize_empty"),
        ("{}", "normalize_no_content"),
        ('{"pairs": [], "title": ""}', "normalize_no_content"),
    ],
)
def test_normalize_failures_fall_back_to_raw_fields(drawing_docs, truths, responder, answer, fallback):
    def normalize(request, default):
        if answer == "raise":
            raise BadRequest("normalize refused")
        return answer

    chunks, _ = _run(drawing_docs[BRACKET], responder({"drawing.normalize": normalize}))
    prov = chunks[0].provenance
    assert fallback in prov["fallbacks"] and prov["responses"]["normalize"] == ""
    assert set(prov["field_sources"].values()) <= {"vision", "table"}
    assert chunks[0].metadata["title_block"]["title"] == truths[BRACKET]["pages"][0]["title_block"]["title"]


def test_normalized_values_win_over_table_and_vision(drawing_docs, responder):
    def vision(request, default):
        answer = default()
        answer.update(material="STEEL", title="BRACKET", scale="1:5")
        return answer

    def table(request, default):
        answer = default()
        answer.update(material="Material: STAINLESS STEEL 1.4301 F", scale="")
        return answer

    def normalize(request, default):
        return {"pairs": [{"topic": "Title", "value": "MOUNTING BRACKET"}], "title": "MOUNTING BRACKET", "material": ""}

    chunks, _ = _run(drawing_docs[BRACKET], responder({"drawing.vision": vision, "drawing.table": table, "drawing.normalize": normalize}))
    fields = chunks[0].metadata["title_block"]
    sources = chunks[0].provenance["field_sources"]
    assert (fields["title"], sources["title"]) == ("MOUNTING BRACKET", "normalize")
    assert (fields["material"], sources["material"]) == ("STAINLESS STEEL 1.4301", "table")  # label + grid letter cleaned
    assert (fields["scale"], sources["scale"]) == ("1:5", "vision")


def test_null_backend_fails_the_gate_with_a_clear_message(drawing_docs):
    client = VLMClient(NullBackend())
    with pytest.raises(TitleBlockGateError, match=r"lantern_reflector_scan\.pdf page 1: title block yielded 0 pair\(s\)"):
        extract(drawing_docs[REFLECTOR], client, SETTINGS, config=LIGHT)
    assert client.usage.summary()["total"]["calls"] == 3  # vision, table and normalize still ran


def test_demo_dry_run_with_null_provider_succeeds(drawing_docs, tmp_path, capsys, monkeypatch):
    import importlib.util
    from pathlib import Path

    for name in ("VLMRAG_CACHE_DIR", "VLMRAG_DRAWING_TITLE_BLOCK_PAGES"):
        monkeypatch.delenv(name, raising=False)
    demo_path = Path(__file__).resolve().parents[2] / "examples" / "drawing_demo.py"
    spec = importlib.util.spec_from_file_location("drawing_demo_under_test", demo_path)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    argv = ["--provider", "null", "--fast", "--out", str(tmp_path), "--env-file", str(tmp_path / "absent.env")]
    assert demo.main(argv + ["--input", *[str(p) for p in drawing_docs.values()]]) == 0
    out = capsys.readouterr().out
    assert out.count("FAILED TitleBlockGateError") == 3 and "as expected for a dry run" in out


def test_min_pairs_is_configurable(drawing_docs, responder):
    answer = {"pairs": [{"topic": "Drawing No.", "value": "EX-L300-1007"}], "drawing_number": "EX-L300-1007"}
    respond = responder({"drawing.vision": answer, "drawing.table": "{}", "drawing.normalize": "{}"})
    with pytest.raises(TitleBlockGateError):
        _run(drawing_docs[BRACKET], respond)
    chunks, _ = _run(drawing_docs[BRACKET], respond, config=LIGHT.replace(min_pairs=1))
    assert chunks[0].metadata["title_block"] == {"drawing_number": "EX-L300-1007"}


def test_replay_miss_is_never_swallowed_by_a_fallback(drawing_docs, responder):
    def miss(request, default):
        raise ReplayMiss("no recorded response")

    with pytest.raises(ReplayMiss):
        _run(drawing_docs[BRACKET], responder({"drawing.table": miss}))


def test_all_pages_mode_fails_the_file_when_one_page_fails(drawing_docs, responder):
    def vision(request, default):
        return "{}" if request.page == 2 else default()

    respond = responder({"drawing.vision": vision, "drawing.normalize": lambda r, d: "{}" if r.page == 2 else d(), "drawing.table": "{}"})
    with pytest.raises(TitleBlockGateError) as info:
        _run(drawing_docs[HOUSING], respond, config=LIGHT.replace(title_block_pages="all"))
    assert info.value.page == 2


def test_pixel_cap_is_recorded(drawing_docs, responder):
    chunks, _ = _run(drawing_docs[BRACKET], responder(), config=LIGHT.replace(max_image_pixels=200_000))
    prov = chunks[0].provenance
    assert "image_pixel_cap" in prov["fallbacks"]
    assert all(img["size"][0] * img["size"][1] <= 200_000 for img in prov["render"]["images"])
    assert any(img["capped"] for img in prov["render"]["images"])


def test_pdf_without_pages_is_rejected(tmp_path):
    path = tmp_path / "empty.pdf"
    pdfium.PdfDocument.new().save(str(path))
    client = VLMClient(NullBackend())
    # PDFium itself refuses to open a page-less document; extract() guards the case as well.
    with pytest.raises(Exception, match="no pages|Failed to load document"):
        extract(path, client, SETTINGS, config=LIGHT)
    assert client.usage.summary()["total"]["calls"] == 0


def test_unparseable_answers_are_not_cached(drawing_docs, responder, tmp_path):
    from vlmrag.vlm.cache import DiskCache

    cache = DiskCache(tmp_path / "cache")
    text = "Drawing No.: EX-L300-1007\nRev: A\nScale: 1:1"
    client = VLMClient(ScriptedBackend(responder({"drawing.vision": text})), cache=cache, sleep=lambda s: None)
    extract(drawing_docs[BRACKET], client, SETTINGS, config=LIGHT)
    cached = [json.loads(p.read_text(encoding="utf-8"))["task"] for p in (tmp_path / "cache").rglob("*.json")]
    assert sorted(cached) == ["drawing.normalize", "drawing.table"]
