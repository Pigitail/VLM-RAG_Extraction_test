"""End-to-end runs of the harness: oracle (perfect model), NullBackend, failures, record -> replay."""

from __future__ import annotations

import json
import shutil

import pytest

from vlmrag.config import Settings
from vlmrag.evaluation.cli import main
from vlmrag.evaluation.documents import EvalDocument
from vlmrag.evaluation.metrics import compare_metrics
from vlmrag.evaluation.oracle import oracle_responder
from vlmrag.evaluation.replay import RecordingDir, compare_with_recording, replay_technique
from vlmrag.evaluation.report import render_markdown, write_results
from vlmrag.evaluation.runner import DrawingVariant, EvalConfig, assemble_results, run_all
from vlmrag.vlm.backends import NullBackend, ScriptedBackend

from .conftest import LIGHT_ENV

DRAWINGS = ["lantern_bracket.pdf", "lantern_housing.pdf", "lantern_reflector_scan.pdf"]


def _oracle(docs):
    return ScriptedBackend(oracle_responder([d.truth for d in docs if d.truth]), model="oracle", name="oracle")


def test_oracle_scores_at_or_near_100_percent(eval_docs, fake_renders, light, tmp_path):
    backend = _oracle(eval_docs)
    config = EvalConfig(drawing_overrides=light)
    runs = run_all(eval_docs, Settings(provider="null"), config, lambda technique, variant: backend)
    assert [r.run_id for r in runs] == [
        "vlm/drawing@cap", "vlm/slides", "vlm/docx", "baseline/drawing@cap", "baseline/slides", "baseline/docx",
    ]
    results = assemble_results(runs, eval_docs, config)
    m = results["metrics"]

    drawing = m["drawing"]["vlm"]["cap"]
    assert drawing["gate"]["pass_rate"] == 1.0
    assert drawing["groups"]["read"]["exact"] == 1.0 and drawing["groups"]["read"]["sheets"] == 3
    assert drawing["groups"]["raster_title_block"]["normalized"] == 1.0
    # The second housing sheet inherits sheet 1's title block: title, scale and sheet differ by design.
    inherited = drawing["groups"]["inherited"]
    assert inherited["sheets"] == 1 and inherited["fields"] - inherited["normalized_correct"] == 3
    assert 0.9 <= drawing["groups"]["all"]["normalized"] < 1.0

    slides = m["slides"]["vlm"]
    assert slides["image_role"]["accuracy"] == 1.0 and slides["image_role"]["blocks"] == 7
    assert slides["must_mention"]["recall"] == 1.0 and slides["hallucinated_blocks"]["total"] == 0
    assert slides["visual_calls"] == {"expected": 4, "ok": 4}

    docx = m["docx"]["vlm"]
    assert docx["page_count"]["accuracy"] == 1.0 and docx["row_page"]["accuracy"] == 1.0
    assert set(docx["per_table"][0]["assignment"]) == {"vlm"}
    assert docx["image_must_mention"]["recall"] == 1.0

    vlm, base = m["retrieval"]["vlm@cap"]["overall"], m["retrieval"]["baseline"]["overall"]
    assert vlm["queries"] == base["queries"] == 18
    assert vlm["recall@5"] == 1.0 and vlm["mrr"] > base["mrr"] and vlm["recall@1"] > base["recall@1"]

    # Baseline: every model call answers "{}".
    assert m["drawing"]["baseline"]["cap"]["gate"] == {
        "attempted": 3, "passed": 0, "gate_failed": 3, "other_errors": {}, "pass_rate": 0.0,
    }
    assert m["slides"]["baseline"]["must_mention"]["recall"] == 0.0
    assert m["docx"]["baseline"]["row_page"]["accuracy"] == 1.0  # markers are correct by construction
    assert set(m["docx"]["baseline"]["per_table"][0]["assignment"]) == {"xml_row_markers"}
    assert m["retrieval"]["baseline"]["model_free_substitutes"] == DRAWINGS
    assert fake_renders["slides"] and fake_renders["docx"]

    cost = {row["run"]: row for row in results["cost"]}
    assert cost["vlm/drawing@cap"]["calls"] == 9 and cost["vlm/drawing@cap"]["input_tokens"] > 0
    assert cost["baseline/slides"]["input_tokens"] == 0
    assert all(len(row["per_document"]) == row["documents"] for row in results["cost"])

    json_path, md_path = write_results(results, tmp_path / "out")
    assert json.loads(json_path.read_text(encoding="utf-8"))["metrics"] == json.loads(json.dumps(m))
    text = md_path.read_text(encoding="utf-8")
    for heading in ("## Engineering drawings", "## Software-manual slides", "## Specification documents",
                    "## Retrieval", "## Cost and latency", "## Caveats", "## Metric definitions"):
        assert heading in text
    assert render_markdown(results) == text


def test_null_backend_run_equals_the_baseline(eval_docs, fake_renders, light):
    config = EvalConfig(techniques=("slides", "docx"), drawing_overrides=light)
    runs = run_all(eval_docs, Settings(provider="null"), config, lambda technique, variant: NullBackend())
    m = assemble_results(runs, eval_docs, config)["metrics"]
    assert "drawing" not in m and set(m["retrieval"]) == {"vlm", "baseline"}
    for technique in ("slides", "docx"):
        assert compare_metrics(m[technique]["baseline"], m[technique]["vlm"]) == []
    for part in ("overall", "by_doc_type", "per_query"):
        assert compare_metrics(m["retrieval"]["baseline"][part], m["retrieval"]["vlm"][part]) == []
    assert m["retrieval"]["baseline"]["model_free_substitutes"] == []


def test_a_failing_document_is_a_data_point(eval_docs, fake_renders, light, tmp_path):
    broken = tmp_path / "broken_sheet.pdf"
    broken.write_bytes(b"%PDF-1.4 this is not a pdf")
    truth = {"source": broken.name, "doc_type": "drawing", "pages": [{"page": 1, "title_block": {"title": "X"}}],
             "queries": [{"query": "broken sheet title", "relevant": [{"source": broken.name, "page": 1}]}]}
    docs = [d for d in eval_docs if d.technique == "drawing"] + [EvalDocument(broken, "drawing", truth)]

    def respond(request):
        if request.source_name == "lantern_bracket.pdf" and request.task == "drawing.vision":
            raise ValueError("model refused")
        return oracle_responder([d.truth for d in docs])(request)

    backend = ScriptedBackend(respond)
    config = EvalConfig(techniques=("drawing",), variants=(DrawingVariant.parse("cap"),), baseline=False,
                        drawing_overrides=light)
    runs = run_all(docs, Settings(provider="null", max_retries=0), config, lambda t, v: backend)
    outcomes = {o.source: o for o in runs[0].outcomes}
    assert outcomes["lantern_bracket.pdf"].error["type"] == "DrawingExtractionError"
    assert outcomes[broken.name].error is not None and outcomes["lantern_housing.pdf"].ok
    m = assemble_results(runs, docs, config)["metrics"]["drawing"]["vlm"]["cap"]
    assert m["gate"]["passed"] == 2 and m["gate"]["attempted"] == 4 and m["gate"]["gate_failed"] == 0
    assert sum(m["gate"]["other_errors"].values()) == 2
    sheets = {(s["source"], s["page"]): s for s in m["sheets"]}
    assert sheets[("lantern_bracket.pdf", 1)]["normalized"] == 0.0
    assert sheets[(broken.name, 1)]["status"] == "error"
    # No absolute paths leak into published error messages.
    messages = [o.error["message"] for o in runs[0].outcomes if o.error]
    assert messages and not [msg for msg in messages if str(tmp_path) in msg or "Users" in msg]


class HTTPError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def _imperfect(docs):
    """A deliberately imperfect model so that a replayed run has something non-trivial to reproduce."""
    oracle = oracle_responder([d.truth for d in docs if d.truth])

    def respond(request):
        answer = oracle(request)
        if request.task.startswith("drawing.") and request.source_name == "lantern_bracket.pdf":
            answer = json.loads(json.dumps(answer).replace("0.15 kg", "0.51 kg"))  # a misread digit
        if request.task == "slides.visual" and request.page == 3:
            answer = json.loads(json.dumps(answer))
            for item in answer["image_descriptions"]:
                item["image_role"] = "main_screenshot"
        if request.task == "docx.image_description" and request.page == 4:
            raise HTTPError(400)
        if request.task == "docx.table_distribution":
            return {"row_distribution": "not an object"}
        return answer

    return respond


def _files(directory):
    return sorted(p.name for p in directory.iterdir())


def test_record_then_replay_reproduces_results(eval_docs, fake_renders, tmp_path, monkeypatch, capsys):
    import vlmrag.vlm.backends as backends

    for name, value in LIGHT_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("VLMRAG_CACHE_DIR", raising=False)
    monkeypatch.setenv("VLMRAG_REASONING_EFFORT", "medium")
    monkeypatch.setattr(
        backends, "make_backend",
        lambda settings: ScriptedBackend(_imperfect(eval_docs), model=settings.model, name="scripted"),
    )
    rec, out1, out2 = tmp_path / "rec", tmp_path / "out1", tmp_path / "out2"
    no_env = str(tmp_path / "missing.env")
    code = main([
        "--record-dir", str(rec), "--out-dir", str(out1), "--variants", "cap,nocap", "--docx-layout", "estimated",
        "--provider", "null", "--model", "test-model", "--label", "round trip", "--date", "2026-01-01",
        "--env-file", no_env,
    ])
    assert code == 0
    assert _files(rec) == [
        "docx.failures.jsonl", "docx.jsonl", "drawing.cap.jsonl", "drawing.nocap.jsonl", "manifest.json",
        "results.json", "results.md", "slides.jsonl",
    ]
    manifest = json.loads((rec / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"] == "test-model" and manifest["techniques"] == ["drawing", "slides", "docx"]
    assert [v["name"] for v in manifest["variants"]] == ["cap", "nocap"]
    assert manifest["drawing_non_defaults"]["page_dpi"] == 36
    assert list(manifest["documents"]["docx_layout_pins"]) == ["lantern_spec.docx"]
    recorded = json.loads((out1 / "results.json").read_text(encoding="utf-8"))
    assert recorded["mode"] == "record" and recorded["date"] == "2026-01-01"
    m = recorded["metrics"]
    assert m["drawing"]["vlm"]["cap"]["per_field"]["mass"]["normalized"] < 1.0
    assert m["slides"]["vlm"]["image_role"]["accuracy"] < 1.0
    assert set(m["docx"]["vlm"]["per_table"][0]["assignment"]) == {"xml_row_markers"}
    assert m["docx"]["vlm"]["image_must_mention"]["found"] < m["docx"]["vlm"]["image_must_mention"]["facts"]

    sent = {r["run"]: r.get("requests_sent", {}) for r in recorded["runs"]}
    assert sent["vlm/drawing@cap"]["drawing.vision"]["reasoning_effort"] == ["medium"]
    assert json.loads((rec / "drawing.cap.jsonl").read_text(encoding="utf-8").splitlines()[0])["reasoning_effort"] == "medium"

    # Replay with none of the recording-time environment, an unrelated drawing variable that would change the
    # crops if it were applied, and no way to reach a model.
    for name in LIGHT_ENV:
        monkeypatch.delenv(name)
    monkeypatch.delenv("VLMRAG_REASONING_EFFORT")
    monkeypatch.setenv("VLMRAG_DRAWING_PAGE_TILE_ROWS", "2")

    def no_model(settings):
        raise AssertionError("a replay must not build a model backend")

    monkeypatch.setattr(backends, "make_backend", no_model)
    capsys.readouterr()
    assert main(["--replay-dir", str(rec), "--out-dir", str(out2), "--env-file", no_env]) == 0
    assert "matches recording: yes" in capsys.readouterr().out
    replayed = json.loads((out2 / "results.json").read_text(encoding="utf-8"))
    assert replayed["mode"] == "replay" and replayed["model"]["model"] == "test-model"
    # The replay reports the recorded run: label, date, settings and the efforts actually sent.
    assert (replayed["label"], replayed["date"]) == ("round trip", "2026-01-01")
    assert replayed["settings"]["reasoning_effort"] == "medium"
    replay_sent = {r["run"]: r.get("requests_sent", {}) for r in replayed["runs"]}
    assert replay_sent["vlm/drawing@cap"] == sent["vlm/drawing@cap"]
    assert replayed["replay"]["misses"] == 0 and replayed["replay"]["unused"] == 0
    assert replayed["replay"]["recorded_failures_served"] == 1
    assert replayed["replay_check"] == {"recorded_results": True, "matches_recording": True, "differences": []}
    assert compare_with_recording(recorded, replayed) == []
    assert compare_metrics(recorded["metrics"], replayed["metrics"]) == []
    assert not [w for w in replayed["warnings"] if "byte-wise" in w]
    assert [w for w in replayed["warnings"] if "VLMRAG_DRAWING_PAGE_TILE_ROWS" in w and "Ignored" in w]
    assert "Replayed from a recording" in (out2 / "results.md").read_text(encoding="utf-8")
    monkeypatch.delenv("VLMRAG_DRAWING_PAGE_TILE_ROWS")

    # The library path used by tests/replay gives the same numbers technique by technique.
    recording = RecordingDir(rec)
    documents = recording.generate_documents(tmp_path / "regen", ["slides", "docx"])
    assert next(d for d in documents if d.technique == "docx").truth["layout_source"] == "pinned"
    for technique in ("slides", "docx"):
        runs = replay_technique(recording, documents, technique, recording.config([technique]))
        assert all(run.replay["misses"] == 0 for run in runs if run.replay)
        partial = assemble_results(runs, documents, recording.config([technique]))
        assert compare_with_recording(recorded, partial) == []
        assert technique in partial["metrics"] and "retrieval" in partial["metrics"]

    # A missing recorded answer is a miss (even where the technique falls back silently) -> exit 1.
    broken = tmp_path / "broken"
    shutil.copytree(rec, broken)
    lines = (broken / "slides.jsonl").read_text(encoding="utf-8").splitlines()
    (broken / "slides.jsonl").write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
    assert main(["--replay-dir", str(broken), "--out-dir", str(tmp_path / "o3"), "--techniques", "slides",
                 "--env-file", no_env]) == 1
    third = json.loads((tmp_path / "o3" / "results.json").read_text(encoding="utf-8"))
    assert third["replay"]["misses"] == 1 and not third["replay_check"]["matches_recording"]
    assert third["replay"]["miss_details"][0]["reason"] == "not_recorded"

    # A recorded answer the replay never asks for means the replay made fewer calls -> exit 1.
    extra = tmp_path / "extra"
    shutil.copytree(rec, extra)
    lines = (extra / "slides.jsonl").read_text(encoding="utf-8").splitlines()
    (extra / "slides.jsonl").write_text("\n".join(lines + lines[:1]) + "\n", encoding="utf-8")
    assert main(["--replay-dir", str(extra), "--out-dir", str(tmp_path / "o5"), "--techniques", "slides",
                 "--env-file", no_env]) == 1
    fifth = json.loads((tmp_path / "o5" / "results.json").read_text(encoding="utf-8"))
    assert fifth["replay"]["unused"] == 1 and fifth["replay"]["misses"] == 0
    assert not fifth["replay_check"]["matches_recording"]

    # Different recorded numbers are reported as differences -> exit 1.
    tampered = tmp_path / "tampered"
    shutil.copytree(rec, tampered)
    data = json.loads((tampered / "results.json").read_text(encoding="utf-8"))
    data["metrics"]["slides"]["vlm"]["image_role"]["correct"] += 1
    (tampered / "results.json").write_text(json.dumps(data), encoding="utf-8")
    capsys.readouterr()
    assert main(["--replay-dir", str(tampered), "--out-dir", str(tmp_path / "o4"), "--techniques", "slides",
                 "--env-file", no_env]) == 1
    assert "metrics.slides.vlm.image_role.correct" in capsys.readouterr().out


@pytest.mark.soffice
def test_oracle_with_real_libreoffice_renders(tmp_path):
    out = tmp_path / "out"
    assert main(["--oracle", "--techniques", "slides,docx", "--no-baseline", "--out-dir", str(out),
                 "--env-file", str(tmp_path / "none.env")]) == 0
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    m = results["metrics"]
    assert results["environment"]["libreoffice"] is True
    assert m["slides"]["vlm"]["image_role"]["accuracy"] == 1.0 and m["slides"]["vlm"]["must_mention"]["recall"] == 1.0
    assert m["docx"]["vlm"]["row_page"]["accuracy"] == 1.0 and m["docx"]["vlm"]["image_must_mention"]["recall"] == 1.0
    assert results["documents"]["items"][-1]["layout_source"] == "libreoffice"
