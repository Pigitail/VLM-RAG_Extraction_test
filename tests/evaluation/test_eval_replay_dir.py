"""Recording directories: manifest validation and configuration selection."""

from __future__ import annotations

import json

import pytest

from vlmrag.evaluation.replay import MANIFEST_FORMAT, RecordingDir, RecordingError, compare_with_recording
from vlmrag.evaluation.report import render_markdown


def _write(tmp_path, **changes) -> RecordingDir:
    manifest = {
        "format": MANIFEST_FORMAT,
        "model": "m",
        "techniques": ["drawing", "slides"],
        "variants": [{"name": "cap", "max_image_pixels": 16_000_000}, {"name": "nocap", "max_image_pixels": 0}],
        "baseline": True,
        "files": {"drawing@cap": "drawing.cap.jsonl", "drawing@nocap": "drawing.nocap.jsonl", "slides": "slides.jsonl"},
        "documents": {"generated": True, "seed": 3, "docx_layout_pins": {}},
        "drawing_non_defaults": {"page_dpi": 36.0},
        "drawing_overrides": {"page_tile_upscale": 1.0},
    }
    manifest.update(changes)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return RecordingDir(tmp_path)


def test_manifest_is_required_and_versioned(tmp_path):
    with pytest.raises(RecordingError, match="no manifest.json"):
        RecordingDir(tmp_path)
    with pytest.raises(RecordingError, match="unsupported recording format"):
        _write(tmp_path, format="something/2")


def test_config_selects_recorded_subsets(tmp_path):
    recording = _write(tmp_path)
    assert recording.seed == 3 and recording.model == "m" and recording.generated
    full = recording.config()
    assert full.techniques == ("drawing", "slides") and [v.name for v in full.variants] == ["cap", "nocap"]
    assert full.drawing_overrides == {"page_dpi": 36.0, "page_tile_upscale": 1.0} and full.baseline
    part = recording.config(["slides"], ["nocap"], baseline=False)
    assert part.techniques == ("slides",) and [v.name for v in part.variants] == ["nocap"] and not part.baseline
    with pytest.raises(RecordingError, match="docx"):
        recording.config(["docx"])
    with pytest.raises(RecordingError, match="maxpx"):
        recording.config(None, ["maxpx-1"])


def test_missing_answer_files_and_external_documents(tmp_path):
    recording = _write(tmp_path)
    with pytest.raises(RecordingError, match="drawing@cap"):
        recording.check_files(recording.config())
    for name in ("drawing.cap.jsonl", "drawing.nocap.jsonl", "slides.jsonl"):
        (tmp_path / name).write_text("", encoding="utf-8")
    recording.check_files(recording.config())
    assert len(recording.backend("drawing", "nocap")) == 0
    external = _write(tmp_path, documents={"generated": False})
    with pytest.raises(RecordingError, match="--docs"):
        external.generate_documents(tmp_path / "docs", ["slides"])


def test_compare_skips_retrieval_for_a_technique_subset_and_ignores_times():
    recorded = {
        "evaluation": {"techniques": ["slides", "docx"]},
        "metrics": {"slides": {"vlm": {"x": 1}}, "retrieval": {"vlm": {"y": 1}}},
        "cost": [{"run": "vlm/slides", "documents": 1, "failed": 0, "calls": 4, "cached": 0, "input_tokens": 10,
                  "output_tokens": 5, "reasoning_tokens": 0, "wall_time_s": 9.0}],
    }
    replayed = json.loads(json.dumps(recorded))
    replayed["evaluation"]["techniques"] = ["slides"]
    replayed["metrics"]["retrieval"]["vlm"]["y"] = 2  # different index: not comparable, not compared
    replayed["cost"][0]["wall_time_s"] = 1.0
    assert compare_with_recording(recorded, replayed) == []
    replayed["cost"][0]["input_tokens"] = 11
    replayed["metrics"]["docx"] = {"vlm": {}}
    assert compare_with_recording(recorded, replayed) == [
        "metrics.docx: not in the recording", "cost.vlm/slides.input_tokens: 10 != 11",
    ]


def test_markdown_reports_replay_status():
    results = {
        "mode": "replay",
        "model": {"provider": "azure", "model": "m", "backend": "replay"},
        "metrics": {},
        "cost": [{"run": "vlm/slides", "documents": 1, "failed": 0, "calls": 4, "cached": 0, "input_tokens": 10,
                  "output_tokens": 5, "reasoning_tokens": 0, "model_latency_s": 1.0, "wall_time_s": 2.0}],
        "replay": {"served": 4, "misses": 0, "unused": 0, "recorded_failures_served": 1},
        "caveats": ["c"],
        "metric_definitions": {"cost": "d"},
    }
    text = render_markdown(results)
    assert "## Replay" in text and "4 recorded answer(s), 0 miss(es)" in text
    assert "tokens and model latency are the recorded values" in text
