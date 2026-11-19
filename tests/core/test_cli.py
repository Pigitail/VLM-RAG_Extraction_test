from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from vlmrag import cli, techniques
from vlmrag import synthetic
from vlmrag.config import Settings
from vlmrag.schema import Chunk, read_jsonl
from vlmrag.synthetic import __main__ as synthetic_main
from vlmrag.vlm.types import VLMRequest

NO_ENV = ["--env-file", "does-not-exist.env"]


def _run_module(*args):
    return subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True, timeout=120)


def test_module_help_works_without_techniques():
    result = _run_module("vlmrag", "--help")
    assert result.returncode == 0
    assert "run" in result.stdout and "techniques" in result.stdout
    run_help = _run_module("vlmrag", "run", "--help")
    assert run_help.returncode == 0
    assert "--replay" in run_help.stdout and "errors.jsonl" in run_help.stdout
    synthetic_help = _run_module("vlmrag.synthetic", "--help")
    assert synthetic_help.returncode == 0
    assert "--only" in synthetic_help.stdout


def test_missing_technique_module_gives_clear_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(techniques.TECHNIQUES, "drawing", "vlmrag.techniques.not_a_real_module:extract")
    code = cli.main(["run", "drawing", str(tmp_path / "a.pdf"), "--out", str(tmp_path / "o.jsonl"), *NO_ENV])
    assert code == 2
    err = capsys.readouterr().err
    assert "technique 'drawing' is not available" in err
    assert "Traceback" not in err


def test_unknown_technique(tmp_path, capsys):
    assert cli.main(["run", "nope", "x", "--out", str(tmp_path / "o.jsonl"), *NO_ENV]) == 2
    assert "unknown technique" in capsys.readouterr().err


def test_techniques_listing(monkeypatch, capsys):
    monkeypatch.setitem(techniques.TECHNIQUES, "stub", lambda path, client, settings: [])
    assert cli.main(["techniques"]) == 0
    out = capsys.readouterr().out
    assert "stub" in out and "available" in out


def test_missing_generator_module_gives_clear_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(synthetic.GENERATORS, "drawing", "vlmrag.synthetic.not_a_real_module:generate")
    assert synthetic_main.main(["--out", str(tmp_path), "--only", "drawing"]) == 2
    assert "generator 'drawing' is not available" in capsys.readouterr().err
    assert synthetic_main.main(["--out", str(tmp_path), "--only", "bogus"]) == 2


def test_synthetic_cli_runs_registered_generators(monkeypatch, tmp_path, capsys):
    def fake_generate(out_dir, *, seed=0):
        path = Path(out_dir) / f"doc-{seed}.txt"
        path.write_text("synthetic", encoding="utf-8")
        return [path]

    monkeypatch.setitem(synthetic.GENERATORS, "fake", fake_generate)
    assert synthetic_main.main(["--out", str(tmp_path), "--only", "fake", "--seed", "7"]) == 0
    assert (tmp_path / "fake" / "doc-7.txt").exists()
    assert "fake:" in capsys.readouterr().out


def test_import_error_inside_technique_is_not_masked(monkeypatch):
    monkeypatch.setitem(techniques.TECHNIQUES, "broken", "vlmrag.techniques:does_not_exist")
    with pytest.raises(techniques.TechniqueUnavailable, match="has no attribute"):
        techniques.load_technique("broken")


def _stub_extract(path, client, settings):
    """A technique stub: fails for files whose name contains 'bad'."""
    if "bad" in path.name:
        raise ValueError(f"cannot read {path.name}")
    if "slow" in path.name:
        time.sleep(0.2)
    request = VLMRequest(task="stub.vision", prompt_version="stub.vision/1", parts=[f"describe {path.name}"], source_name=path.name, page=1)
    value, _ = client.call_json(request)
    note = json.dumps(value, sort_keys=True)
    return [
        Chunk(
            doc_id=path.stem, source_name=path.name, doc_type="stub", page=page, chunk_index=page - 1,
            retrieve_content=f"{path.stem} page {page} {note}", llm_content=f"# {path.stem}", filtering_content=path.stem,
        )
        for page in (1, 2)
    ]


@pytest.fixture
def stub_technique(monkeypatch):
    monkeypatch.setitem(techniques.TECHNIQUES, "stub", _stub_extract)
    monkeypatch.setitem(techniques.SUFFIXES, "stub", (".txt",))


def _inputs(tmp_path, names):
    paths = []
    for name in names:
        path = tmp_path / "in" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
        paths.append(str(path))
    return paths


def test_batch_continues_after_failures(stub_technique, tmp_path, capsys):
    inputs = _inputs(tmp_path, ["a-slow.txt", "b-bad.txt", "c.txt"]) + [str(tmp_path / "missing.txt")]
    out = tmp_path / "out" / "chunks.jsonl"
    stale = cli.errors_path_for(out)
    stale.parent.mkdir(parents=True)
    stale.write_text("stale\n", encoding="utf-8")
    code = cli.main(["run", "stub", *inputs, "--out", str(out), "--provider", "null", "--workers", "3", *NO_ENV])
    assert code == 1  # some files failed, but the batch finished
    chunks = read_jsonl(out)
    assert [(c.source_name, c.page) for c in chunks] == [("a-slow.txt", 1), ("a-slow.txt", 2), ("c.txt", 1), ("c.txt", 2)]
    errors = [json.loads(line) for line in stale.read_text(encoding="utf-8").splitlines()]
    assert {e["source_name"]: e["error_type"] for e in errors} == {"b-bad.txt": "ValueError", "missing.txt": "FileNotFoundError"}
    assert all("traceback" in e and e["technique"] == "stub" for e in errors)
    stdout = capsys.readouterr().out
    assert "4 file(s): 2 ok, 2 failed, 4 chunk(s)" in stdout
    assert "stub.vision" in stdout and "TOTAL" in stdout  # usage summary printed at the end


def test_batch_all_ok_removes_stale_error_file(stub_technique, tmp_path):
    inputs = _inputs(tmp_path, ["a.txt"])
    out = tmp_path / "chunks.jsonl"
    cli.errors_path_for(out).write_text("old\n", encoding="utf-8")
    assert cli.main(["run", "stub", *inputs, "--out", str(out), "--provider", "null", *NO_ENV]) == 0
    assert not cli.errors_path_for(out).exists()
    assert len(read_jsonl(out)) == 2


def test_directory_inputs_are_expanded(stub_technique, tmp_path):
    _inputs(tmp_path, ["one.txt", "two.txt", "~$lock.txt", "skip.md"])
    paths = cli.expand_inputs([str(tmp_path / "in")], "stub")
    assert [p.name for p in paths] == ["one.txt", "two.txt"]


def test_errors_path_naming(tmp_path):
    assert cli.errors_path_for(tmp_path / "out.jsonl").name == "out.errors.jsonl"
    assert cli.errors_path_for(tmp_path / "out").name == "out.errors.jsonl"


def test_record_then_replay_reproduces_output(stub_technique, tmp_path):
    inputs = _inputs(tmp_path, ["a.txt", "b.txt"])
    rec = tmp_path / "calls.jsonl"
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    assert cli.main(["run", "stub", *inputs, "--out", str(first), "--provider", "null", "--record", str(rec), *NO_ENV]) == 0
    assert len(rec.read_text(encoding="utf-8").splitlines()) == 2
    # Replay needs no provider credentials at all.
    assert cli.main(["run", "stub", *inputs, "--out", str(second), "--provider", "openai", "--replay", str(rec), *NO_ENV]) == 0
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_record_and_replay_are_mutually_exclusive(stub_technique, tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["run", "stub", "x", "--out", "o", "--record", "a", "--replay", "b"])


def test_configuration_error_is_reported(stub_technique, tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    inputs = _inputs(tmp_path, ["a.txt"])
    assert cli.main(["run", "stub", *inputs, "--out", str(tmp_path / "o.jsonl"), "--provider", "openai", *NO_ENV]) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_run_batch_applies_summary_when_enabled(stub_technique, tmp_path, scripted_client):
    def responder(request):
        if request.task == "augment.summary":
            return {"summary_ko": "요약", "summary_en": "summary", "keywords_ko": [], "keywords_en": ["kw"], "keywords_source": []}
        return {"seen": request.source_name}

    client = scripted_client(responder)
    settings = Settings(provider="null", retrieval_summary_doc_types=("stub",))
    paths = [Path(p) for p in _inputs(tmp_path, ["a.txt"])]
    result = cli.run_batch("stub", paths, tmp_path / "o.jsonl", client, settings)
    assert result.succeeded == 1 and result.chunks == 2
    chunks = read_jsonl(tmp_path / "o.jsonl")
    assert all(c.retrieve_content.startswith("doc_type: stub") for c in chunks)
    assert all(c.llm_content.startswith("# a") for c in chunks)


def test_run_batch_is_thread_safe_with_many_workers(stub_technique, tmp_path, scripted_client):
    lock = threading.Lock()
    seen = []

    def responder(request):
        with lock:
            seen.append(request.source_name)
        return {}

    names = [f"f{i:02d}.txt" for i in range(12)]
    paths = [Path(p) for p in _inputs(tmp_path, names)]
    result = cli.run_batch("stub", paths, tmp_path / "o.jsonl", scripted_client(responder), Settings(provider="null"), workers=6)
    assert result.succeeded == 12
    assert [c.source_name for c in read_jsonl(tmp_path / "o.jsonl")][::2] == names
    assert sorted(seen) == names
