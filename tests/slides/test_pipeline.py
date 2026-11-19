"""End-to-end ``extract()`` with scripted model answers (rendering faked, no LibreOffice needed)."""

from __future__ import annotations

import json
import threading
import time

import pytest
from PIL import Image

from vlmrag.config import Settings
from vlmrag.render.office import ConversionError
from vlmrag.schema import read_jsonl, write_jsonl
from vlmrag.techniques import load_technique
from vlmrag.techniques.slides import PROMPT_VERSION, TASK, SlidesExtractionError, extract, pipeline
from vlmrag.techniques.slides.layout import read_deck
from vlmrag.techniques.slides.prompt import build_request
from vlmrag.vlm.backends import NullBackend
from vlmrag.vlm.client import VLMClient

SETTINGS = Settings(provider="null")
IMAGE_SLIDES = [2, 3, 4, 5]


def _by_page(chunks):
    return {c.page: c for c in chunks}


def test_registry_resolves_the_technique():
    assert load_technique("slides") is extract


def test_unreadable_deck_is_a_clear_hard_failure(tmp_path):
    client = VLMClient(NullBackend())
    broken = tmp_path / "broken.pptx"
    broken.write_bytes(b"not a zip file")
    with pytest.raises(SlidesExtractionError, match=r"broken\.pptx is not a PPTX \(zip\) package$"):
        extract(broken, client, SETTINGS)
    legacy = tmp_path / "legacy.pptx"
    legacy.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + bytes(504))
    with pytest.raises(SlidesExtractionError, match="OLE compound file"):
        extract(legacy, client, SETTINGS)
    assert client.usage.summary()["total"]["calls"] == 0


def test_pdf_page_map():
    assert pipeline.pdf_page_map([False, True, False], 2) == {1: 0, 3: 1}
    assert pipeline.pdf_page_map([False, True, False], 3) == {1: 0, 2: 1, 3: 2}
    assert pipeline.pdf_page_map([False, False], 1) is None
    assert pipeline.pdf_page_map([], 0) == {}


def test_extract_with_truth_answers(deck_path, truth, scripted_client, truth_responder, fake_render):
    client = scripted_client(by_task={TASK: truth_responder(truth)})
    chunks = extract(deck_path, client, SETTINGS)

    assert sorted(fake_render) == IMAGE_SLIDES
    assert sorted(r.page for r in client.backend.calls) == IMAGE_SLIDES  # one request per picture slide
    assert all(r.prompt_version == PROMPT_VERSION and r.source_name == deck_path.name for r in client.backend.calls)
    assert [c.page for c in chunks] == [1, 2, 3, 4, 5, 6]
    assert [c.chunk_index for c in chunks] == list(range(6))
    assert len({c.doc_id for c in chunks}) == 1

    pages = _by_page(chunks)
    for slide_truth in truth["slides"]:
        chunk = pages[slide_truth["slide"]]
        meta = chunk.metadata
        assert meta["slide_title"] == slide_truth["title"]
        roles = {d["shape_name"]: d["image_role"] for d in meta["image_descriptions"]}
        assert roles == {b["shape_name"]: b["image_role"] for b in slide_truth["image_blocks"]}
        for fact in slide_truth["must_mention"]:
            assert fact in chunk.retrieve_content, (slide_truth["slide"], fact)
            assert fact in chunk.llm_content
        described = {d["shape_name"] for d in meta["image_descriptions"]}
        assert not described & set(slide_truth["annotation_shapes"])
        if slide_truth["image_blocks"]:
            assert meta["visual_used"] and "[Visual image retrieval cues]" in chunk.retrieve_content
            assert chunk.provenance["visual_call"]["status"] == "ok"
            assert chunk.provenance["prompt_versions"] == {TASK: PROMPT_VERSION}
            [call] = chunk.provenance["calls"]
            assert call["task"] == TASK and call["prompt_version"] == PROMPT_VERSION and call["status"] == "ok"
            assert chunk.provenance["backend"] == "scripted"
            assert chunk.provenance["render"]["dpi"] == 180
            assert chunk.provenance["request"]["timeout_s"] == 240
            assert chunk.provenance["request"]["max_output_tokens"] == 4000
            assert chunk.provenance["fallbacks"] == []
        else:
            assert chunk.provenance["visual_call"] == {"status": "skipped", "reason": "no_images"}
            assert chunk.provenance["calls"] == [] and chunk.provenance["prompt_versions"] == {}
            assert not meta["visual_used"]

    # Every query is answerable from the retrieval text of its relevant slide.
    for query in truth["queries"]:
        (relevant,) = query["relevant"]
        assert query["answer"] in pages[relevant["page"]].retrieve_content


def test_detail_crop_and_main_share_source_id(deck_path, truth, scripted_client, truth_responder, fake_render):
    chunks = extract(deck_path, scripted_client(by_task={TASK: truth_responder(truth)}), SETTINGS)
    items = {d["shape_name"]: d for d in _by_page(chunks)[3].metadata["image_descriptions"]}
    assert items["Detail PUR-120"]["source_image_id"] == items["Screen PUR-120"]["source_image_id"]
    assert "crop_norm" in items["Detail PUR-120"] and "crop_norm" not in items["Screen PUR-120"]


def test_chunks_round_trip_through_jsonl(deck_path, truth, scripted_client, truth_responder, fake_render, tmp_path):
    chunks = extract(deck_path, scripted_client(by_task={TASK: truth_responder(truth)}), SETTINGS)
    write_jsonl(chunks, tmp_path / "out.jsonl")
    assert [c.to_dict() for c in read_jsonl(tmp_path / "out.jsonl")] == [c.to_dict() for c in chunks]


# --------------------------------------------------------------------------- failure paths


def test_malformed_json_is_re_asked_once_then_falls_back(deck_path, scripted_client, fake_render):
    client = scripted_client("I could not produce JSON {broken")
    chunks = extract(deck_path, client, SETTINGS)
    assert len(client.backend.calls) == 2 * len(IMAGE_SLIDES)  # PARSE_RETRIES = 1 -> two attempts per slide
    for page in IMAGE_SLIDES:
        chunk = _by_page(chunks)[page]
        assert chunk.provenance["fallbacks"] == ["unparseable_json"]
        assert chunk.provenance["visual_call"]["status"] == "unparseable"
        assert "[Visual image retrieval cues]" not in chunk.retrieve_content
        assert chunk.retrieve_content.startswith(f"slide={page} | title=")


def test_empty_answers_fall_back_to_xml_text(deck_path, fake_render):
    client = VLMClient(NullBackend())
    chunks = extract(deck_path, client, SETTINGS)
    for page in IMAGE_SLIDES:
        chunk = _by_page(chunks)[page]
        assert chunk.provenance["fallbacks"] == ["empty_visual_result"]
        assert chunk.metadata["visual_used"] is False and chunk.metadata["image_count"] >= 1
    assert len(chunks) == 6


def test_invalid_answers_are_dropped_or_coerced(deck_path, truth, scripted_client, truth_responder, fake_render):
    good = truth_responder(truth)

    def respond(request):
        answer = good(request)
        if request.page == 3:
            answer["image_descriptions"][0]["block_order"] = 0  # cannot be attached -> dropped
            answer["image_descriptions"][1]["image_role"] = "decorative"  # unknown -> left empty
            answer["image_descriptions"][1]["confidence"] = "certain"  # -> low
        if request.page == 2:
            answer["image_descriptions"][0]["block_order"] = 5  # the arrow shape: kept, flagged
        return answer

    chunks = _by_page(extract(deck_path, scripted_client(by_task={TASK: respond}), SETTINGS))
    slide3 = chunks[3]
    (item,) = slide3.metadata["image_descriptions"]
    assert item["shape_name"] == "Detail PUR-120" and "image_role" not in item and item["confidence"] == "low"
    assert set(slide3.provenance["fallbacks"]) == {"dropped_descriptions", "invalid_image_role", "invalid_confidence"}
    assert "dropped_descriptions:1" in slide3.provenance["visual_issues"]
    slide2 = chunks[2]
    assert "ungrounded_block_order" in slide2.provenance["fallbacks"]
    assert slide2.provenance["ungrounded_block_orders"] == [5]


def test_one_failing_slide_does_not_affect_the_others(deck_path, truth, scripted_client, truth_responder, fake_render):
    good = truth_responder(truth)

    class BadRequest(Exception):
        status_code = 400

    def respond(request):
        if request.page == 4:
            raise BadRequest("content filter")
        return good(request)

    client = scripted_client(by_task={TASK: respond})
    chunks = _by_page(extract(deck_path, client, SETTINGS))
    assert chunks[4].provenance["fallbacks"] == ["vlm_error"]
    assert chunks[4].provenance["visual_call"]["status"] == "error"
    assert "BadRequest" in chunks[4].provenance["visual_call"]["error"]
    assert sum(1 for r in client.backend.calls if r.page == 4) == 1  # 4xx is not retried
    for page in (2, 3, 5):
        assert chunks[page].metadata["visual_used"] and chunks[page].provenance["fallbacks"] == []


def test_transient_errors_are_retried_by_the_client(deck_path, truth, scripted_client, truth_responder, fake_render):
    good = truth_responder(truth)
    failures = {"left": 2}
    lock = threading.Lock()

    def respond(request):
        if request.page == 5:
            with lock:
                if failures["left"]:
                    failures["left"] -= 1
                    raise TimeoutError("slow server")
        return good(request)

    client = scripted_client(by_task={TASK: respond}, max_retries=3)
    chunks = _by_page(extract(deck_path, client, SETTINGS))
    assert chunks[5].metadata["visual_used"] and chunks[5].provenance["fallbacks"] == []
    assert sum(1 for r in client.backend.calls if r.page == 5) == 3


def test_render_unavailable_means_no_model_calls(deck_path, scripted_client, monkeypatch):
    def broken_convert(*args, **kwargs):
        raise ConversionError("LibreOffice was not found")

    monkeypatch.setattr("vlmrag.render.office.convert", broken_convert)
    client = scripted_client()
    chunks = _by_page(extract(deck_path, client, SETTINGS))
    assert client.backend.calls == []
    for page in IMAGE_SLIDES:
        assert chunks[page].provenance["fallbacks"] == ["render_unavailable"]
        assert chunks[page].provenance["visual_call"] == {"status": "skipped", "reason": "render_unavailable"}
    assert chunks[1].provenance["visual_call"]["reason"] == "no_images"


def test_hidden_or_unrendered_slides_are_reported(deck_path, scripted_client, monkeypatch, truth, truth_responder):
    png = pipeline.to_png(Image.new("RGB", (8, 8), "white"))

    def partial(path, deck, numbers, settings, **kwargs):
        return {2: (png, 2)}, {3: ["hidden_slide"], 4: ["render_failed"], 5: ["render_failed"]}

    monkeypatch.setattr(pipeline, "render_slides", partial)
    client = scripted_client(by_task={TASK: truth_responder(truth)})
    chunks = _by_page(extract(deck_path, client, SETTINGS))
    assert [r.page for r in client.backend.calls] == [2]
    assert chunks[3].provenance["fallbacks"] == ["hidden_slide"]
    assert chunks[4].provenance["visual_call"]["reason"] == "render_failed"


def test_requests_run_concurrently_within_the_client_limit(deck_path, truth, scripted_client, truth_responder,
                                                          fake_render):
    good = truth_responder(truth)
    state = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def respond(request):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.15)
        with lock:
            state["now"] -= 1
        return good(request)

    client = scripted_client(by_task={TASK: respond}, max_concurrency=2)
    extract(deck_path, client, Settings(provider="null", max_concurrency=8))
    assert state["peak"] == 2


def test_rejects_non_pptx_and_missing_files(tmp_path, scripted_client):
    with pytest.raises(FileNotFoundError):
        extract(tmp_path / "missing.pptx", scripted_client(), SETTINGS)
    other = tmp_path / "notes.txt"
    other.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        extract(other, scripted_client(), SETTINGS)


def test_corrupt_pptx_is_a_hard_failure(tmp_path, scripted_client):
    bad = tmp_path / "broken.pptx"
    bad.write_bytes(b"not a zip file")
    with pytest.raises(Exception):
        extract(bad, scripted_client(), SETTINGS)


# --------------------------------------------------------------------------- assets


def test_assets_and_sidecars(deck_path, truth, scripted_client, truth_responder, fake_render, tmp_path):
    chunks = _by_page(extract(deck_path, scripted_client(by_task={TASK: truth_responder(truth)}), SETTINGS,
                              asset_dir=tmp_path / "assets"))
    slide3 = chunks[3]
    deck_dir = pipeline.deck_asset_dirname(deck_path.stem, slide3.doc_id)
    assert deck_dir == f"sample_erp_manual-{slide3.doc_id[:8]}"
    names = {d["shape_name"]: d["asset"] for d in slide3.metadata["image_descriptions"]}
    assert names == {"Screen PUR-120": f"{deck_dir}/slide_003/img_001.png",
                     "Detail PUR-120": f"{deck_dir}/slide_003/img_002.png"}
    with Image.open(tmp_path / "assets" / names["Screen PUR-120"]) as full, \
            Image.open(tmp_path / "assets" / names["Detail PUR-120"]) as detail:
        assert detail.width < full.width and detail.height < full.height  # crop applied
    sidecar_text = (tmp_path / "assets" / deck_dir / "slide_003" / "layout.json").read_text(encoding="utf-8")
    sidecar = json.loads(sidecar_text)
    assert sidecar["slide"] == 3 and sidecar["source_name"] == deck_path.name
    # The sidecar carries the same enriched descriptions as the chunk metadata (asset, box, shape name).
    assert sidecar["visual_grounding"]["image_descriptions"] == slide3.metadata["image_descriptions"]
    assert {d["asset"] for d in sidecar["visual_grounding"]["image_descriptions"]} == set(names.values())
    assert tmp_path.name not in sidecar_text and deck_path.parent.name not in sidecar_text  # no local paths
    assert not (tmp_path / "assets" / deck_dir / "slide_001").exists()
    assert "image_assets" not in chunks[1].metadata


def test_unwritable_sidecar_does_not_fail_the_file(deck_path, truth, scripted_client, truth_responder, fake_render,
                                                   tmp_path, monkeypatch):
    real_write_text = pipeline.Path.write_text

    def refuse_sidecar(self, *args, **kwargs):
        if self.name == "layout.json":
            raise PermissionError("read-only share")
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(pipeline.Path, "write_text", refuse_sidecar)
    chunks = _by_page(extract(deck_path, scripted_client(by_task={TASK: truth_responder(truth)}), SETTINGS,
                              asset_dir=tmp_path / "assets"))
    assert len(chunks) == 6
    assert "asset_write_failed" in chunks[3].provenance["fallbacks"]
    assert chunks[3].metadata["visual_used"] and "asset" in chunks[3].metadata["image_descriptions"][0]


def test_deck_asset_dirs_do_not_collide():
    assert pipeline.deck_asset_dirname("manual", "0123456789abcdef") == "manual-01234567"
    assert pipeline.deck_asset_dirname("manual", "fedcba9876543210") == "manual-fedcba98"
    assert pipeline.deck_asset_dirname("Manual (v2)", "aa" * 8) == "Manual_v2-aaaaaaaa"
    assert pipeline.deck_asset_dirname("사용자 매뉴얼", "bb" * 8) == "bbbbbbbb"  # non-ASCII stem -> hash only


def test_pathological_answer_values_never_fail_the_file(deck_path, truth, scripted_client, truth_responder,
                                                         fake_render):
    good = truth_responder(truth)

    def respond(request):
        if request.page == 3:
            # Python's JSON parser turns these into float('inf'); int(inf) raises OverflowError.
            return '{"visual_summary": "S", "image_descriptions": [{"block_order": 1e400, "description": "d"},' \
                   ' {"block_order": 2, "related_block_orders": [Infinity, 3], "description": "kept",' \
                   ' "image_role": "main_screenshot", "confidence": "low"}], "flow_notes": [], "notes": ""}'
        return good(request)

    chunks = _by_page(extract(deck_path, scripted_client(by_task={TASK: respond}), SETTINGS))
    (item,) = chunks[3].metadata["image_descriptions"]
    assert item["description"] == "kept" and item["related_block_orders"] == [3]
    assert "dropped_descriptions" in chunks[3].provenance["fallbacks"]
    assert chunks[2].metadata["visual_used"]


def test_validation_crash_falls_back_for_that_slide_only(deck_path, truth, scripted_client, truth_responder,
                                                        fake_render, monkeypatch):
    real = pipeline.parse_visual

    def flaky(raw):
        if isinstance(raw, dict) and raw.get("visual_summary", "").startswith("A full order-lines"):
            raise RecursionError("too deep")
        return real(raw)

    monkeypatch.setattr(pipeline, "parse_visual", flaky)
    chunks = _by_page(extract(deck_path, scripted_client(by_task={TASK: truth_responder(truth)}), SETTINGS))
    assert chunks[3].provenance["fallbacks"] == ["parse_error"]
    assert chunks[3].provenance["visual_call"]["status"] == "invalid"
    assert "[Visual image retrieval cues]" not in chunks[3].retrieve_content
    assert chunks[4].metadata["visual_used"] and chunks[4].provenance["fallbacks"] == []


# --------------------------------------------------------------------------- render_slides (no LibreOffice)


def _pdf_with_pages(path, widths):
    """A PDF whose pages differ in width, so a render tells which page was used."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument.new()
    try:
        for width in widths:
            pdf.new_page(width, 72).close()
        pdf.save(str(path))
    finally:
        pdf.close()
    return path


def _deck(hidden):
    from vlmrag.techniques.slides.layout import Deck, SlideLayout

    return Deck(slide_width=1, slide_height=1,
                slides=[SlideLayout(number=i + 1, hidden=h) for i, h in enumerate(hidden)])


def _png_width(png: bytes) -> int:
    import io

    with Image.open(io.BytesIO(png)) as image:
        return image.width


def test_render_slides_maps_pages_past_a_hidden_slide(tmp_path, monkeypatch):
    pdf = _pdf_with_pages(tmp_path / "two.pdf", [100, 200])  # slide 2 hidden -> not exported
    monkeypatch.setattr("vlmrag.render.office.convert", lambda *a, **k: pdf)
    pngs, fallbacks = pipeline.render_slides(tmp_path / "deck.pptx", _deck([False, True, False]), [1, 2, 3],
                                             SETTINGS, dpi=72)
    assert fallbacks == {2: ["hidden_slide"]}
    assert {n: page for n, (_, page) in pngs.items()} == {1: 1, 3: 2}
    assert _png_width(pngs[1][0]) == 100 and _png_width(pngs[3][0]) == 200  # slide 3 got PDF page 2


def test_render_slides_skips_grounding_on_page_count_mismatch(tmp_path, monkeypatch):
    pdf = _pdf_with_pages(tmp_path / "one.pdf", [100])
    monkeypatch.setattr("vlmrag.render.office.convert", lambda *a, **k: pdf)
    pngs, fallbacks = pipeline.render_slides(tmp_path / "deck.pptx", _deck([False, False, False]), [2, 3],
                                             SETTINGS, dpi=72)
    assert pngs == {} and fallbacks == {2: ["page_count_mismatch"], 3: ["page_count_mismatch"]}


def test_render_slides_never_raises(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise TypeError("unexpected converter failure")

    monkeypatch.setattr("vlmrag.render.office.convert", unexpected)
    pngs, fallbacks = pipeline.render_slides(tmp_path / "deck.pptx", _deck([False]), [1], SETTINGS)
    assert pngs == {} and fallbacks == {1: ["render_unavailable"]}
    garbage = tmp_path / "garbage.pdf"
    garbage.write_bytes(b"%PDF-1.4 not really")
    monkeypatch.setattr("vlmrag.render.office.convert", lambda *a, **k: garbage)
    assert pipeline.render_slides(tmp_path / "deck.pptx", _deck([False]), [1], SETTINGS) == (
        {}, {1: ["render_unavailable"]})


def test_record_then_replay_on_a_regenerated_deck(deck_path, truth, scripted_backend, truth_responder,
                                                  fake_render, tmp_path):
    from vlmrag.synthetic.slides import generate
    from vlmrag.vlm.backends import RecordingBackend, ReplayBackend

    record = tmp_path / "calls.jsonl"
    recorded = extract(deck_path, VLMClient(RecordingBackend(scripted_backend(truth_responder(truth)), record)),
                       SETTINGS)
    [regenerated] = generate(tmp_path / "again", seed=0)
    replayed = extract(regenerated, VLMClient(ReplayBackend(record)), SETTINGS)
    assert [c.retrieve_content for c in replayed] == [c.retrieve_content for c in recorded]
    assert all(c.provenance["fallbacks"] == [] for c in replayed)


def test_same_request_for_same_slide_content(deck_path):
    """Requests depend only on the deck content, so caches and replays are reusable."""
    deck = read_deck(deck_path)
    first = build_request(deck.slides[3], deck.slide_count, b"png", dpi=180, source_name=deck_path.name)
    second = build_request(read_deck(deck_path).slides[3], deck.slide_count, b"png", dpi=180,
                           source_name=deck_path.name)
    assert first.parts == second.parts
