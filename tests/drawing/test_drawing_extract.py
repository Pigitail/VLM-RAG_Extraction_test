"""End-to-end ``extract()`` on the synthetic drawings with a scripted model built from the truth files."""

from __future__ import annotations

import functools
import json
import re
import threading

import pytest

from vlmrag import cli
from vlmrag.config import Settings
from vlmrag.render.images import format_bbox
from vlmrag.schema import doc_id_for, read_jsonl, write_jsonl
from vlmrag.synthetic.drawing import generate
from vlmrag.techniques import load_technique
from vlmrag.techniques.drawing import PROMPT_VERSIONS, extract
from vlmrag.techniques.drawing.crops import plan_crops
from vlmrag.text import filtering_text
from vlmrag.vlm.backends import RecordingBackend, ReplayBackend, ScriptedBackend
from vlmrag.vlm.client import VLMClient
from vlmrag.vlm.types import ImageInput

from .conftest import BRACKET, HOUSING, LIGHT, REFLECTOR

SETTINGS = Settings(provider="null")


def _client(respond, **kwargs):
    kwargs.setdefault("sleep", lambda s: None)
    return VLMClient(ScriptedBackend(respond), **kwargs)


def _run(path, respond, config=LIGHT, **client_kwargs):
    client = _client(respond, **client_kwargs)
    return extract(path, client, SETTINGS, config=config), client


def test_registry_resolves_extract():
    assert load_technique("drawing") is extract


@pytest.mark.parametrize("name", [HOUSING, REFLECTOR, BRACKET])
def test_chunks_carry_what_the_model_read(name, drawing_docs, truths, responder):
    path = drawing_docs[name]
    truth = truths[name]
    chunks, client = _run(path, responder())
    assert [c.page for c in chunks] == list(range(1, truth["page_count"] + 1))
    assert [c.chunk_index for c in chunks] == list(range(truth["page_count"]))
    first_page = truth["pages"][0]
    for chunk, page in zip(chunks, truth["pages"]):
        assert chunk.doc_type == "drawing" and chunk.source_name == name and chunk.doc_id == doc_id_for(path)
        # Default mode: the page-1 title block is attached to every page.
        assert chunk.metadata["title_block_page"] == 1
        for key, value in first_page["title_block"].items():
            assert chunk.metadata["title_block"][key] == value
            assert value in chunk.retrieve_content and value in chunk.llm_content
        for row in first_page["table_rows"]:
            assert "ROW " in chunk.retrieve_content and " || ".join(row) in chunk.retrieve_content
        assert chunk.retrieve_content.startswith(f"{name}\npage {chunk.page}\n")
        assert chunk.filtering_content == filtering_text(chunk.retrieve_content)
        assert not re.search(r"\s", chunk.filtering_content)
        assert "[Drawing text layer]" in chunk.retrieve_content and page["notes"][1] in chunk.retrieve_content
        assert "| Field | Value |" in chunk.llm_content
        assert chunk.metadata["rotation"] == page["rotation"]
        assert chunk.metadata["field_values"]["drawing_number"] == [first_page["title_block"]["drawing_number"]]
        prov = chunk.provenance
        assert prov["prompt_versions"] == PROMPT_VERSIONS and prov["model"] == "scripted"
        assert prov["backend"] == "scripted" and prov["technique"] == "drawing"
        assert prov["fallbacks"] == [] and prov["gate"] == {"min_pairs": 3, "pairs_found": len(chunk.metadata["title_block_pairs"])}
        assert [c["task"] for c in prov["calls"]] == ["drawing.vision", "drawing.table", "drawing.normalize"]
        assert all(c["status"] == "ok" and c["usage"]["total_tokens"] > 0 for c in prov["calls"])
        assert len(prov["render"]["images"]) == 21 and prov["render"]["page_tiles"]["rows"] == 3
    for query in truth["queries"]:
        hits = [c for c in chunks if query["evidence"] in c.retrieve_content]
        assert query["relevant"][0]["page"] in [c.page for c in hits]
    assert len(client.backend.calls) == 3


def test_requests_follow_the_contract(drawing_docs, truths, responder):
    _, client = _run(drawing_docs[BRACKET], responder())
    calls = {r.task: r for r in client.backend.calls}
    plan = plan_crops(LIGHT)
    vision, table, normalize = calls["drawing.vision"], calls["drawing.table"], calls["drawing.normalize"]
    for request in (vision, table, normalize):
        assert request.source_name == BRACKET and request.page == 1
        assert request.prompt_version == PROMPT_VERSIONS[request.task]
        assert request.timeout_s == 600 and request.max_output_tokens == 60000
        assert request.reasoning_effort == "low" and request.json_schema["additionalProperties"] is False
        for part in request.parts:
            text = part.label if isinstance(part, ImageInput) else part
            assert "\\" not in text and str(drawing_docs[BRACKET].parent) not in text
    # Vision: prompt, hints, then 12 tiles + title focus + table focus, each with its position label.
    assert isinstance(vision.parts[0], str) and "12 overlapping page tiles" in vision.parts[0]
    assert vision.parts[1].startswith("Text extracted from the PDF text layer")
    assert [i.label for i in vision.images()] == [c.label for c in plan if c.for_vision]
    assert [i.label for i in table.images()] == [c.label for c in plan if c.for_table]
    assert len(vision.images()) == 14 and len(table.images()) == 8
    assert normalize.images() == []
    assert normalize.parts[1].startswith("Raw vision JSON/object:\n[MAIN_VISION_JSON]\n{")
    assert "[TABLE_VISION_JSON]" in normalize.parts[1] and "[TABLE_ROWS]\nROW 0: REV || DESCRIPTION || DATE" in normalize.parts[1]
    assert normalize.parts[2].startswith("Observed raw pairs text:\nCompany: Example Manufacturing Co.")
    # Hints are headed by the same label as the image of the same region.
    hints = vision.parts[1]
    focus_label = next(c.label for c in plan if c.kind == "title_block_focus")
    section = hints.split(f"[{focus_label}]\n", 1)[1].split("\n\n[", 1)[0]
    assert truths[BRACKET]["pages"][0]["title_block"]["drawing_number"] in section
    assert hints.count("[FULL_PAGE x=0.00-1.00 y=0.00-1.00]") == 1
    table_hint = table.parts[1]
    assert table_hint.startswith("Text extracted from the PDF text layer of the same metadata-table regions")
    assert f"METADATA_TABLE_OVERVIEW {format_bbox((0.44, 0.62, 1.0, 1.0))}" in table_hint


def test_raster_title_block_is_read_by_vision_only(drawing_docs, truths, responder):
    """No hint carries the raster values, yet the chunk has them: they came from the vision pass."""
    chunks, client = _run(drawing_docs[REFLECTOR], responder())
    fields = truths[REFLECTOR]["pages"][0]["title_block"]
    vision = next(r for r in client.backend.calls if r.task == "drawing.vision")
    hint_text = "\n".join(p for p in vision.parts if isinstance(p, str))
    for key in ("drawing_number", "material", "part_number"):
        assert fields[key] not in hint_text
        assert chunks[0].metadata["title_block"][key] == fields[key]
    assert chunks[0].provenance["hints"]["page_text_chars"] > 0  # notes are still vector text


def test_all_pages_mode_reads_every_sheet(drawing_docs, truths, responder):
    chunks, client = _run(drawing_docs[HOUSING], responder(), config=LIGHT.replace(title_block_pages="all"))
    assert len(client.backend.calls) == 6
    assert sorted({r.page for r in client.backend.calls}) == [1, 2]
    for chunk, page in zip(chunks, truths[HOUSING]["pages"]):
        assert chunk.metadata["title_block_page"] == chunk.page
        assert chunk.metadata["title_block"]["sheet"] == page["title_block"]["sheet"]
        assert chunk.metadata["title_block"]["title"] == page["title_block"]["title"]
    assert chunks[1].metadata["table_rows"] == []  # sheet 2 has no revision table
    assert "(Title block read from page" not in chunks[1].llm_content


def test_first_page_mode_marks_borrowed_title_block(drawing_docs, responder):
    chunks, _ = _run(drawing_docs[HOUSING], responder())
    assert "(Title block read from page 1 of this file.)" in chunks[1].llm_content
    assert "SECTION A-A: NOMINAL WALL 2.4." in chunks[1].retrieve_content  # page 2's own text layer
    assert "SECTION A-A: NOMINAL WALL 2.4." not in chunks[0].retrieve_content


def test_vision_and_table_passes_run_concurrently(drawing_docs, responder):
    barrier = threading.Barrier(2, timeout=10)

    def meet(request, default):
        barrier.wait()  # breaks (and fails the vision pass) unless both passes are in flight together
        return default()

    chunks, _ = _run(drawing_docs[BRACKET], responder({"drawing.vision": meet, "drawing.table": meet}))
    assert chunks and not barrier.broken


def test_retrieval_text_sections(drawing_docs, responder):
    chunk = _run(drawing_docs[HOUSING], responder())[0][0]
    text = chunk.retrieve_content
    order = [
        "Drawing No.: EX-L300-1001",
        "ROW 1: A || Initial release || 2025-11-02",
        "Title block layout columns: Company | Department",
        "[Structured fields]",
        "Domain synonym: Vent hole = Venthole = Breather hole",
        "[Drawing text layer]",
    ]
    positions = [text.index(marker) for marker in order]
    assert positions == sorted(positions)
    assert "(field=vent_hole / tags=vent_hole)" in text  # a text-layer note lifted into a field entry
    entries = chunk.metadata["field_entries"]
    # Table pairs identical to title-block pairs are merged away, so only these sources remain here.
    assert {e["source"] for e in entries} == {"title_block", "title_block_field", "raw_chunk"}
    finish = next(e for e in entries if e["field_key"] == "surface_finish" and e["source"] == "title_block")
    assert finish["part"] == "housing" and finish["intent_tags"] == ["surface_finish"]


def test_text_layer_can_be_left_out(drawing_docs, responder):
    chunk = _run(drawing_docs[BRACKET], responder(), config=LIGHT.replace(include_text_layer=False))[0][0]
    assert "[Drawing text layer]" not in chunk.retrieve_content and "NOTES:" not in chunk.llm_content


def test_jsonl_round_trip(drawing_docs, responder, tmp_path):
    chunks, _ = _run(drawing_docs[HOUSING], responder())
    write_jsonl(chunks, tmp_path / "out.jsonl")
    assert [c.to_dict() for c in read_jsonl(tmp_path / "out.jsonl")] == [c.to_dict() for c in chunks]
    json.dumps([c.to_dict() for c in chunks])


def test_record_then_replay_in_another_directory(tmp_path, truths, responder):
    """Replay keys hold file names and labels only, so a recording replays against regenerated files elsewhere."""
    first = {p.name: p for p in generate(tmp_path / "a")}
    second = {p.name: p for p in generate(tmp_path / "b")}
    record = tmp_path / "calls.jsonl"
    recording = VLMClient(RecordingBackend(ScriptedBackend(responder()), record))
    expected = extract(first[HOUSING], recording, SETTINGS, config=LIGHT)
    replay = VLMClient(ReplayBackend(record))
    replayed = extract(second[HOUSING], replay, SETTINGS, config=LIGHT)
    assert [c.retrieve_content for c in replayed] == [c.retrieve_content for c in expected]
    assert [c.metadata for c in replayed] == [c.metadata for c in expected]


def test_cli_batch_records_gate_failures_and_continues(drawing_docs, responder, tmp_path):
    def refuse_for_bracket(request, default):
        return "{}" if request.source_name == BRACKET else default()

    respond = responder({"drawing.vision": refuse_for_bracket, "drawing.normalize": refuse_for_bracket, "drawing.table": refuse_for_bracket})
    client = _client(respond)
    paths = [drawing_docs[BRACKET], drawing_docs[HOUSING], drawing_docs[REFLECTOR]]
    out = tmp_path / "out.jsonl"
    result = cli.run_batch("drawing", paths, out, client, SETTINGS, extract=functools.partial(extract, config=LIGHT))
    assert (result.succeeded, result.failed, result.chunks) == (2, 1, 3)
    errors = [json.loads(line) for line in cli.errors_path_for(out).read_text(encoding="utf-8").splitlines()]
    assert errors[0]["source_name"] == BRACKET and errors[0]["error_type"] == "TitleBlockGateError"
    assert [c.source_name for c in read_jsonl(out)] == [HOUSING, HOUSING, REFLECTOR]
