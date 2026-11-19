"""Evaluation record/replay backends: usage, failures, misses, compatibility with the core replay."""

from __future__ import annotations

import json

import pytest

from vlmrag.evaluation.recording import (
    EvalRecordingBackend,
    EvalReplayBackend,
    RecordedFailure,
    failures_path_for,
    recording_name,
)
from vlmrag.vlm.backends import ReplayBackend, ReplayMiss, ScriptedBackend
from vlmrag.vlm.client import VLMClient
from vlmrag.vlm.types import ImageInput, VLMRequest


class HTTPError(Exception):
    """Looks like an SDK error: carries a status code and a message that must never be stored."""

    def __init__(self, status_code: int) -> None:
        super().__init__("request rejected by https://secret-host.invalid/path?id=abc")
        self.status_code = status_code


def _raise(status: int):
    def respond(request):
        raise HTTPError(status)

    return respond


def _request(page: int, text: str = "prompt") -> VLMRequest:
    return VLMRequest(
        task="t.task", prompt_version="t.task/1", parts=[text, ImageInput(b"png", label=f"IMG {page}")],
        source_name="doc.pdf", page=page,
    )


def test_names():
    assert recording_name("slides") == "slides.jsonl"
    assert recording_name("drawing", "nocap") == "drawing.nocap.jsonl"
    assert failures_path_for("x/drawing.cap.jsonl").name == "drawing.cap.failures.jsonl"


def test_recording_stores_usage_and_failures_without_messages(tmp_path):
    def respond(request):
        if request.page == 3:
            raise HTTPError(400)
        return {"page": request.page}

    path = tmp_path / "rec.jsonl"
    backend = EvalRecordingBackend(ScriptedBackend(respond, model="m-1"), path)
    client = VLMClient(backend, sleep=lambda s: None)
    client.call(_request(1))
    client.call(_request(2))
    with pytest.raises(HTTPError):
        client.call(_request(3))
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [json.loads(r["response"])["page"] for r in lines] == [1, 2]
    assert set(lines[0]) == {
        "key", "task", "prompt_version", "response", "usage", "latency_s", "model", "reasoning_effort",
    }
    assert lines[0]["usage"]["input_tokens"] > 0 and lines[0]["model"] == "m-1"
    failure_text = failures_path_for(path).read_text(encoding="utf-8")
    [failure] = [json.loads(line) for line in failure_text.splitlines()]
    assert failure["error_type"] == "HTTPError" and failure["status_code"] == 400 and failure["page"] == 3
    assert "secret-host" not in failure_text and "message" not in failure
    assert (backend.recorded, backend.failed) == (2, 1)
    # The core replay backend reads the same file (extra fields are ignored).
    assert len(ReplayBackend(path)) == 2


def test_replay_restores_usage_order_failures_and_counts_misses(tmp_path):
    answers = iter(["first", "second"])

    def respond(request):
        if request.page == 9:
            raise HTTPError(413)
        return next(answers)

    path = tmp_path / "rec.jsonl"
    recorder = EvalRecordingBackend(ScriptedBackend(respond), path)
    recorder.generate(_request(1))
    recorder.generate(_request(1))  # same key twice: served in recording order
    with pytest.raises(HTTPError):
        recorder.generate(_request(9))
    recorded_usage = [json.loads(line)["usage"] for line in path.read_text(encoding="utf-8").splitlines()]

    replay = EvalReplayBackend(path, model="recorded-model")
    one, two = (replay.generate(_request(1)) for _ in range(2))
    assert (one.text, two.text) == ("first", "second")
    assert one.usage.to_dict() == recorded_usage[0] and two.usage.to_dict() == recorded_usage[1]
    assert one.model == "scripted"  # the model the recorded response reported, not the manifest's name
    # A third request for a key recorded twice is a divergence, not a reason to repeat the last answer.
    with pytest.raises(ReplayMiss, match="exhausted"):
        replay.generate(_request(1))
    with pytest.raises(RecordedFailure, match="HTTPError.*413"):
        replay.generate(_request(9))
    with pytest.raises(ReplayMiss):
        replay.generate(_request(5))
    stats = replay.stats()
    assert stats == {
        "file": "rec.jsonl", "recorded": 2, "served": 2, "unused": 0, "misses": 2, "recorded_failures_served": 1,
    }
    assert replay.misses == [
        {"task": "t.task", "prompt_version": "t.task/1", "source_name": "doc.pdf", "page": 1, "reason": "exhausted"},
        {"task": "t.task", "prompt_version": "t.task/1", "source_name": "doc.pdf", "page": 5, "reason": "not_recorded"},
    ]


def test_failure_after_recorded_answers_is_replayed_as_a_failure(tmp_path):
    """An empty answer whose re-ask was rejected: the replay must fail the re-ask too, not repeat the empty answer."""
    outcomes = iter(["", HTTPError(400)])

    def respond(request):
        result = next(outcomes)
        if isinstance(result, Exception):
            raise result
        return result

    path = tmp_path / "rec.jsonl"
    recorder = EvalRecordingBackend(ScriptedBackend(respond), path)
    assert recorder.generate(_request(1)).text == ""
    with pytest.raises(HTTPError):
        recorder.generate(_request(1))
    replay = EvalReplayBackend(path)
    assert replay.generate(_request(1)).text == ""
    with pytest.raises(RecordedFailure):
        replay.generate(_request(1))
    assert replay.stats()["misses"] == 0 and replay.stats()["recorded_failures_served"] == 1
    line = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert line["reasoning_effort"] is None  # stored for readers, not part of the key


def test_recorded_failure_is_not_retried_and_success_wins_over_failure(tmp_path):
    path = tmp_path / "rec.jsonl"
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise HTTPError(503)  # retried by the client, then succeeds
        return "ok"

    client = VLMClient(EvalRecordingBackend(ScriptedBackend(flaky), path), sleep=lambda s: None)
    assert client.call(_request(1)).text == "ok"
    replay = EvalReplayBackend(path)
    assert replay.generate(_request(1)).text == "ok"  # a later success beats the recorded failure
    attempts = {"n": 0}

    class Counting:
        name, model = "c", "c"

        def generate(self, request):
            attempts["n"] += 1
            return replay.generate(request)

    failing = tmp_path / "fail.jsonl"
    with pytest.raises(HTTPError):
        EvalRecordingBackend(ScriptedBackend(_raise(400)), failing).generate(_request(2))
    replay = EvalReplayBackend(failing)
    with pytest.raises(RecordedFailure):
        VLMClient(Counting(), sleep=lambda s: None).call(_request(2))
    assert attempts["n"] == 1


def test_unused_answers_are_counted(tmp_path):
    path = tmp_path / "rec.jsonl"
    recorder = EvalRecordingBackend(ScriptedBackend(lambda r: "x"), path)
    for page in (1, 2, 3):
        recorder.generate(_request(page))
    replay = EvalReplayBackend(path)
    replay.generate(_request(2))
    assert replay.unused() == 2
