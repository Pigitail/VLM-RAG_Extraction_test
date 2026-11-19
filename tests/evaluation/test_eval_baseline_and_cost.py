"""Adversarial checks: baseline isolation, the drawing substitute, pixel variants, cost accounting, redaction."""

from __future__ import annotations

import json

import pytest
from PIL import Image

from vlmrag.config import Settings
from vlmrag.evaluation.oracle import oracle_responder
from vlmrag.evaluation.runner import (
    DrawingVariant,
    EvalConfig,
    RequestStats,
    assemble_results,
    drawing_config_for,
    model_free_drawing_chunks,
    png_size,
    run_all,
    run_technique,
    sanitize_message,
)
from vlmrag.render.images import to_png
from vlmrag.techniques.drawing import extract as drawing_extract
from vlmrag.vlm.backends import NullBackend, ScriptedBackend
from vlmrag.vlm.cache import DiskCache
from vlmrag.vlm.client import VLMClient
from vlmrag.vlm.types import ImageInput, VLMRequest

CANARY = "zqcanaryzq"
_TEXT_KEYS = {"description", "description_en", "visual_summary", "value"}


def _with_canary(value):
    """Append CANARY to every free-text answer field (enums, numbers and row lists stay valid)."""
    if isinstance(value, dict):
        return {k: (f"{v} {CANARY}" if k in _TEXT_KEYS and isinstance(v, str) and v else _with_canary(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [_with_canary(v) for v in value]
    return value


def _chunk_text(chunk) -> str:
    return "\n".join([chunk.retrieve_content, chunk.llm_content, json.dumps(chunk.metadata, ensure_ascii=False)])


def test_baseline_never_reaches_the_run_backend_and_holds_no_model_output(eval_docs, fake_renders, light):
    oracle = oracle_responder([d.truth for d in eval_docs if d.truth])
    backend = ScriptedBackend(lambda request: _with_canary(json.loads(json.dumps(oracle(request)))))
    with_baseline = run_all(eval_docs, Settings(provider="null"), EvalConfig(drawing_overrides=light),
                            lambda technique, variant: backend)
    calls_with_baseline = len(backend.calls)
    backend.calls.clear()
    run_all(eval_docs, Settings(provider="null"), EvalConfig(drawing_overrides=light, baseline=False),
            lambda technique, variant: backend)
    # The baseline runs add no call to the model backend: every one of their requests went to NullBackend.
    assert calls_with_baseline == len(backend.calls) > 0

    vlm = [c for r in with_baseline if r.mode == "vlm" for o in r.outcomes for c in o.chunks]
    base_runs = [r for r in with_baseline if r.mode == "baseline"]
    assert all(r.backend == "null" for r in base_runs)
    assert [c for c in vlm if CANARY in _chunk_text(c)], "the canary must reach VLM chunks for this test to mean anything"
    base = [c for r in base_runs for o in r.outcomes for c in o.chunks]
    assert base and not [c for c in base if CANARY in _chunk_text(c)]
    # Same documents in both modes.
    for mode in ("vlm", "baseline"):
        sources = sorted(o.source for r in with_baseline if r.mode == mode for o in r.outcomes)
        assert sources == sorted(d.source for d in eval_docs)

    # The retrieval index of the baseline (with its drawing substitutes) holds no model output either.
    results = assemble_results(with_baseline, eval_docs, EvalConfig(drawing_overrides=light))
    baseline_retrieval = results["metrics"]["retrieval"]["baseline"]
    assert baseline_retrieval["model_free_substitutes"] == sorted(
        d.source for d in eval_docs if d.technique == "drawing")
    assert results["cost"][-1]["input_tokens"] == 0 and all(
        row["input_tokens"] == 0 for row in results["cost"] if row["mode"] == "baseline")


def test_model_free_substitute_is_the_technique_without_a_model(eval_docs, light):
    """The drawing baseline stand-in equals what the technique emits for an empty answer with the gate disabled."""
    cfg = drawing_config_for(DrawingVariant.parse("cap"), light)
    drawings = [d for d in eval_docs if d.technique == "drawing"]
    assert drawings
    for doc in drawings:
        substitute = model_free_drawing_chunks(doc.path, cfg)
        real = drawing_extract(doc.path, VLMClient(NullBackend()), Settings(provider="null"),
                               config=cfg.replace(min_pairs=0))
        assert [c.page for c in substitute] == [c.page for c in real] == list(range(1, len(real) + 1))
        for mine, theirs in zip(substitute, real):
            assert mine.retrieve_content == theirs.retrieve_content
            assert mine.llm_content == theirs.llm_content
            assert mine.metadata["title_block"] == {} and mine.metadata["evaluation_substitute"]
            assert doc.source in mine.retrieve_content and f"page {mine.page}" in mine.retrieve_content


def test_drawing_variants_differ_in_the_pixels_actually_sent(eval_docs, light):
    docs = [d for d in eval_docs if d.source == "lantern_bracket.pdf"]
    oracle = oracle_responder([d.truth for d in docs])
    sent = {}
    for spec in ("nocap", "maxpx=100000"):
        variant = DrawingVariant.parse(spec)
        run = run_technique(docs, "drawing", mode="vlm", backend=ScriptedBackend(oracle),
                            settings=Settings(provider="null"), variant=variant,
                            config=EvalConfig(variants=(variant,), drawing_overrides=light))
        assert run.outcomes[0].ok
        sent[variant.name] = run
    capped, uncapped = sent["maxpx-100000"].requests, sent["nocap"].requests
    assert set(capped) == set(uncapped) == {"drawing.vision", "drawing.table", "drawing.normalize"}
    assert 0 < capped["drawing.vision"]["max_image_pixels"] <= 100_000
    assert uncapped["drawing.vision"]["max_image_pixels"] > 100_000
    assert capped["drawing.vision"]["images"] == uncapped["drawing.vision"]["images"] > 0
    assert capped["drawing.normalize"]["images"] == 0
    assert sent["maxpx-100000"].drawing_config["max_image_pixels"] == 100_000
    assert sent["nocap"].drawing_config["max_image_pixels"] == 0


def test_request_stats_and_png_size():
    png = to_png(Image.new("RGB", (30, 20), "white"))
    assert png_size(png) == (30, 20) and png_size(b"not a png") is None
    stats = RequestStats(ScriptedBackend(lambda r: "{}", model="m", name="s"))
    assert (stats.name, stats.model) == ("s", "m")
    for effort in ("low", None, "low"):
        stats.generate(VLMRequest(task="t", prompt_version="t/1", parts=["x", ImageInput(png), ImageInput(png)],
                                  reasoning_effort=effort))
    assert stats.summary() == {"t": {
        "requests": 3, "images": 6, "max_image_pixels": 600, "image_pixels": 3600, "reasoning_effort": ["low", "none"],
    }}


def test_cached_calls_are_counted_once_and_cost_no_tokens(eval_docs, fake_renders, tmp_path):
    docs = [d for d in eval_docs if d.technique == "slides"]
    oracle = oracle_responder([d.truth for d in docs])
    cache = DiskCache(tmp_path / "cache")
    first_backend = ScriptedBackend(oracle)
    first = run_technique(docs, "slides", mode="vlm", backend=first_backend, settings=Settings(provider="null"),
                          cache=cache)
    second_backend = ScriptedBackend(oracle)
    second = run_technique(docs, "slides", mode="vlm", backend=second_backend, settings=Settings(provider="null"),
                           cache=cache)
    one, two = first.usage["total"], second.usage["total"]
    assert one["calls"] == len(first_backend.calls) > 0 and one["cached"] == 0 and one["input_tokens"] > 0
    # Second run: every answer from the cache, counted once as a call and once as cached, with no tokens.
    assert len(second_backend.calls) == 0 and second.requests == {}
    assert two["calls"] == one["calls"] and two["cached"] == two["calls"]
    assert two["input_tokens"] == two["output_tokens"] == two["total_tokens"] == 0
    config = EvalConfig(techniques=("slides",), baseline=False)
    [row] = assemble_results([second], docs, config)["cost"]
    assert (row["calls"], row["cached"], row["input_tokens"]) == (one["calls"], one["calls"], 0)
    assert sum(d["calls"] for d in row["per_document"]) == row["calls"]
    assert sum(d["cached"] for d in row["per_document"]) == row["cached"]


def test_run_totals_are_the_sum_of_documents_and_variants_stay_separate(eval_docs, light):
    docs = [d for d in eval_docs if d.technique == "drawing"]
    oracle = oracle_responder([d.truth for d in docs])
    config = EvalConfig(techniques=("drawing",), baseline=False, drawing_overrides=light,
                        variants=(DrawingVariant.parse("cap"), DrawingVariant.parse("maxpx=100000")))
    backends = {}

    def backend_for(technique, variant):
        backends[variant] = ScriptedBackend(oracle)
        return backends[variant]

    runs = run_all(docs, Settings(provider="null"), config, backend_for)
    rows = {row["run"]: row for row in assemble_results(runs, docs, config)["cost"]}
    assert set(rows) == {"vlm/drawing@cap", "vlm/drawing@maxpx-100000"}
    for run in runs:
        row = rows[run.run_id]
        assert row["calls"] == len(backends[run.variant].calls) == 9  # 3 passes x 3 drawings, no re-asks
        for key in ("calls", "input_tokens", "output_tokens"):
            assert row[key] == sum(d[key] for d in row["per_document"])
        # By hand, from ScriptedBackend's usage estimate: ceil(prompt chars / 4) + 85 per image, per request.
        expected = sum((sum(len(p) for p in q.text_parts()) + 3) // 4 + 85 * len(q.images())
                       for q in backends[run.variant].calls)
        assert row["input_tokens"] == expected


@pytest.mark.parametrize(
    "raw, leaked",
    [
        ("vision pass failed: APIConnectionError: cannot reach https://example-resource.openai.azure.com/openai/v1/",
         "example-resource"),
        ("BadRequestError: Error code: 400 - {'error': {'message': 'x', 'request_id': 'req_ABCdef1234567890'}}",
         "ABCdef1234567890"),
        ("timeout talking to gateway.corp at 203.0.113.7:8443", "203.0.113.7"),  # documentation range
        ("id 123e4567-e89b-12d3-a456-426614174000 rejected", "123e4567"),
    ],
)
def test_sanitize_message_redacts_endpoints_hosts_and_ids(raw, leaked):
    clean = sanitize_message(raw)
    assert leaked not in clean and ("<url>" in clean or "<id>" in clean or "<ip>" in clean or "<host>" in clean)


def test_sanitize_message_keeps_technique_text():
    text = "lantern_bracket.pdf page 1: title block yielded 0 pair(s), fewer than the required 3 (drawing.vision)"
    assert sanitize_message(text) == text
