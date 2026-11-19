"""Command line: argument validation, small runs, variants and helpers."""

from __future__ import annotations

import json

import pytest

from vlmrag.evaluation.cli import main
from vlmrag.evaluation.runner import DrawingVariant, EvalConfig, default_cap, sanitize_message


def _env(tmp_path) -> list[str]:
    # Never read a developer's .env from the working directory in tests.
    return ["--env-file", str(tmp_path / "missing.env")]


def test_help_and_required_out_dir(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0 and "--record-dir" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "args, message",
    [
        (["--techniques", "drawing,video"], "unknown technique"),
        (["--variants", "half"], "unknown drawing variant"),
        (["--record-dir", "REC", "--cache-dir", "CACHE"], "--cache-dir cannot be combined"),
        (["--replay-dir", "EMPTY"], "no manifest.json"),
        (["--doc-workers", "0"], "--doc-workers must be >= 1"),
    ],
)
def test_usage_errors_exit_with_2(tmp_path, capsys, args, message):
    args = [a.replace("REC", str(tmp_path / "rec")).replace("CACHE", str(tmp_path / "c"))
            .replace("EMPTY", str(tmp_path)) for a in args]
    assert main(args + ["--out-dir", str(tmp_path / "out")] + _env(tmp_path)) == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_mutually_exclusive_sources(tmp_path):
    with pytest.raises(SystemExit):
        main(["--oracle", "--replay-dir", str(tmp_path), "--out-dir", str(tmp_path / "o")])


def test_replay_refuses_unrecorded_techniques(tmp_path, capsys):
    manifest = {"format": "vlmrag-evaluation-recording/1", "techniques": ["slides"], "variants": [], "files": {}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert main(["--replay-dir", str(tmp_path), "--techniques", "docx", "--out-dir", str(tmp_path / "o")]
                + _env(tmp_path)) == 2
    assert "not in the recording" in capsys.readouterr().err
    (tmp_path / "manifest.json").write_text(json.dumps(dict(manifest, format="other/9")), encoding="utf-8")
    assert main(["--replay-dir", str(tmp_path), "--out-dir", str(tmp_path / "o")] + _env(tmp_path)) == 2


def test_oracle_run_of_one_technique(tmp_path, fake_renders, capsys):
    out = tmp_path / "out"
    code = main(["--oracle", "--techniques", "docx", "--docx-layout", "estimated", "--out-dir", str(out),
                 "--label", "unit", "--date", "2026-02-03"] + _env(tmp_path))
    assert code == 0
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["mode"] == "oracle" and results["label"] == "unit" and results["date"] == "2026-02-03"
    assert set(results["metrics"]) == {"docx", "retrieval"}
    assert results["metrics"]["docx"]["vlm"]["row_page"]["accuracy"] == 1.0
    assert results["documents"]["items"][0]["layout_source"] == "estimated"
    text = (out / "results.md").read_text(encoding="utf-8")
    assert "## Specification documents" in text and "## Engineering drawings" not in text
    assert "results ->" in capsys.readouterr().out


def test_live_run_with_the_null_provider_and_label_from_environment(tmp_path, fake_renders, monkeypatch):
    monkeypatch.setenv("VLMRAG_EVAL_LABEL", "from env")
    monkeypatch.delenv("VLMRAG_EVAL_DATE", raising=False)
    out = tmp_path / "out"
    assert main(["--provider", "null", "--techniques", "slides", "--no-baseline", "--out-dir", str(out)]
                + _env(tmp_path)) == 0
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["mode"] == "live" and results["model"]["backend"] == "null"
    assert results["label"] == "from env" and results["date"] == ""
    assert [row["run"] for row in results["cost"]] == ["vlm/slides"]


def test_drawing_variants():
    assert DrawingVariant.parse("cap") == DrawingVariant("cap", default_cap()) and default_cap() == 4_000_000
    assert DrawingVariant.parse(" NoCap ") == DrawingVariant("nocap", 0)
    assert DrawingVariant.parse("maxpx=4000000") == DrawingVariant("maxpx-4000000", 4_000_000)
    with pytest.raises(ValueError):
        DrawingVariant.parse("maxpx=lots")
    with pytest.raises(ValueError):
        EvalConfig(variants=(DrawingVariant.parse("cap"), DrawingVariant.parse("cap")))
    assert EvalConfig().variants == (DrawingVariant("cap", default_cap()),)


def test_drawing_variant_reaches_the_technique_config():
    from vlmrag.evaluation.runner import drawing_config_for, drawing_non_defaults

    cfg = drawing_config_for(DrawingVariant.parse("nocap"), {"page_dpi": 50})
    assert cfg.max_image_pixels == 0 and cfg.page_dpi == 50
    assert drawing_non_defaults(cfg) == {"page_dpi": 50}


def test_sanitize_message_removes_paths():
    raw = (
        r"cannot open C:\Users\someone\AppData\Local\Temp\x\doc.pdf and /home/someone/docs/a.pdf"
        " or /tmp/vlmrag_eval_1/b.docx"
    )
    clean = sanitize_message(raw)
    assert "someone" not in clean and clean.count("<path>") == 3
    assert len(sanitize_message("x" * 1000)) == 300
