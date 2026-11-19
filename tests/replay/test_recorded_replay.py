"""Regression: committed model recordings must keep replaying exactly.

``python -m vlmrag.evaluation --record-dir examples/recordings/<model>`` stores
every model exchange of an evaluation run together with ``manifest.json`` and
the run's ``results.json``. For every such directory this test

1. regenerates the synthetic documents with the recorded seed and the recorded
   docx page layout (so the requests are identical on any machine),
2. runs each recorded technique (each drawing variant) and its text-only
   baseline with every model call answered from the recording,
3. asserts that no call missed the recording and that the metrics and token
   counts equal the recorded ``results.json``; a final case rebuilds the
   cross-technique BM25 retrieval numbers.

A failure means a code change altered the requests (prompt text, crops,
labels, page split) or how answers become chunks and metrics - bump the
prompt version and re-record, or fix the regression. Without recordings the
test is skipped. Slides and docx need LibreOffice to render the pages the
recorded requests were made for, so those cases are marked ``soffice``.

Set ``VLMRAG_RECORDINGS_DIR`` to check a fresh recording before committing it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vlmrag.evaluation.replay import RecordingDir, compare_with_recording, replay_technique
from vlmrag.evaluation.runner import assemble_results

DEFAULT_RECORDINGS = Path(__file__).resolve().parents[2] / "examples" / "recordings"
RECORDINGS = Path(os.environ.get("VLMRAG_RECORDINGS_DIR") or DEFAULT_RECORDINGS)
NEEDS_LIBREOFFICE = {"slides", "docx"}


def recording_dirs(root: Path = RECORDINGS) -> list[Path]:
    return sorted(p.parent for p in root.glob("*/manifest.json")) if root.is_dir() else []


def _cases() -> list:
    cases = []
    for directory in recording_dirs():
        techniques = json.loads((directory / "manifest.json").read_text(encoding="utf-8")).get("techniques") or []
        for technique in techniques + ["retrieval"]:
            needs = set(techniques) if technique == "retrieval" else {technique}
            marks = [pytest.mark.soffice] if needs & NEEDS_LIBREOFFICE else []
            cases.append(pytest.param(directory, technique, marks=marks, id=f"{directory.name}-{technique}"))
    if not cases:
        cases.append(pytest.param(
            None, None, id="no-recordings",
            marks=pytest.mark.skip(reason="no recordings under examples/recordings/<model>/"),
        ))
    return cases


class Replays:
    """Session cache: documents per recording, replayed runs per (recording, technique)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._documents: dict[Path, list] = {}
        self._runs: dict[tuple[Path, str], list] = {}

    def documents(self, directory: Path) -> list:
        if directory not in self._documents:
            recording = RecordingDir(directory)
            target = self.root / directory.name
            self._documents[directory] = recording.generate_documents(target, recording.techniques)
        return self._documents[directory]

    def runs(self, directory: Path, technique: str) -> list:
        key = (directory, technique)
        if key not in self._runs:
            recording = RecordingDir(directory)
            self._runs[key] = replay_technique(
                recording, self.documents(directory), technique, recording.config([technique])
            )
        return self._runs[key]


@pytest.fixture(scope="session")
def replays(tmp_path_factory) -> Replays:
    return Replays(tmp_path_factory.mktemp("recorded_replays"))


@pytest.mark.parametrize("directory, technique", _cases())
def test_recording_replays_exactly(replays, directory, technique):
    recording = RecordingDir(directory)
    recorded = recording.recorded_results()
    assert recorded is not None, f"{directory.name}: results.json is missing from the recording"
    techniques = recording.techniques if technique == "retrieval" else [technique]
    documents = [d for d in replays.documents(directory) if d.technique in techniques]
    if recording.docx_pins and "docx" in techniques:
        assert all(d.truth["layout_source"] == "pinned" for d in documents if d.technique == "docx")
    runs = [run for name in techniques for run in replays.runs(directory, name)]

    misses = [m for run in runs if run.replay for m in run.replay["miss_details"]]
    assert sum(run.replay["misses"] for run in runs if run.replay) == 0, misses[:5]
    # Answers the replay never asked for: it made fewer calls than the recorded run.
    unused = {run.run_id: run.replay["unused"] for run in runs if run.replay and run.replay["unused"]}
    assert not unused, unused
    results = assemble_results(runs, documents, recording.config(techniques))
    differences = compare_with_recording(recorded, results)
    assert differences == [], "\n".join(differences[:20])
    if technique == "retrieval":
        assert results["metrics"]["retrieval"], "no retrieval runs were rebuilt"
