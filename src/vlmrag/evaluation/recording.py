"""Record/replay for evaluation runs: usage, failures and miss accounting on top of the core backends.

The core :class:`~vlmrag.vlm.backends.RecordingBackend` stores the answer
text per request key, which is all a technique needs to re-run offline. An
evaluation additionally has to reproduce *its numbers*, so the subclasses here
add three things while keeping the file format readable by the core
:class:`~vlmrag.vlm.backends.ReplayBackend` (and ``vlmrag run --replay``):

* every answer line also stores ``usage``, ``latency_s`` and ``model``, so a
  replayed run reports the tokens the recorded run actually spent;
* a call that *failed* (for example a request rejected as too large) is
  written to a ``<name>.failures.jsonl`` sidecar with its exception type and
  HTTP status only - provider messages can echo endpoints or request ids and
  are never stored. On replay the same request raises
  :class:`RecordedFailure`, so the technique takes the same fallback it took
  while recording instead of reporting a replay miss;
* :class:`EvalReplayBackend` counts served answers, misses and unused
  recordings. Some techniques turn a model error into a fallback, which would
  hide a :class:`~vlmrag.vlm.backends.ReplayMiss`; the counters do not.
  Unlike the core replay (where the last answer of a key repeats forever), a
  request made more often than it was recorded is a miss too: a replay that
  asks for more answers than the recorded run received has diverged from it,
  and repeating an old answer would hide that.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Optional, Union

from ..vlm.backends import RecordingBackend, ReplayBackend, ReplayMiss, replay_key
from ..vlm.types import Usage, VLMError, VLMRequest, VLMResponse

__all__ = ["EvalRecordingBackend", "EvalReplayBackend", "RecordedFailure", "failures_path_for", "recording_name"]

log = logging.getLogger(__name__)


class RecordedFailure(VLMError):
    """Replays a call that failed while recording (never retried by the client)."""


def failures_path_for(path: Union[str, Path]) -> Path:
    """``drawing.cap.jsonl`` -> ``drawing.cap.failures.jsonl``."""
    path = Path(path)
    stem = path.name[: -len(path.suffix)] if path.suffix else path.name
    return path.with_name(f"{stem}.failures.jsonl")


def recording_name(technique: str, variant: Optional[str] = None) -> str:
    """File name of the recording of one technique run: ``slides.jsonl``, ``drawing.nocap.jsonl``."""
    return f"{technique}.{variant}.jsonl" if variant else f"{technique}.jsonl"


def _status_code(exc: BaseException) -> Optional[int]:
    for candidate in (exc, getattr(exc, "response", None)):
        code = getattr(candidate, "status_code", None)
        if isinstance(code, int):
            return code
    return None


class EvalRecordingBackend(RecordingBackend):
    """:class:`RecordingBackend` that also records usage, latency, model and failed calls."""

    def __init__(self, inner: Any, path: Union[str, Path]) -> None:
        super().__init__(inner, path)
        # A run in which no call succeeded (or none was needed) still leaves a
        # recording file, so a replay of it finds the file and simply serves nothing.
        self.path.touch(exist_ok=True)
        self.failures_path = failures_path_for(self.path)
        self._write_lock = threading.Lock()
        self.recorded = 0
        self.failed = 0

    def _append(self, path: Path, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._write_lock:
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")

    def generate(self, request: VLMRequest) -> VLMResponse:
        key = replay_key(request)
        try:
            response = self.inner.generate(request)
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised unchanged
            self._append(self.failures_path, {
                "key": key,
                "task": request.task,
                "prompt_version": request.prompt_version,
                "source_name": request.source_name,
                "page": request.page,
                "error_type": type(exc).__name__,
                "status_code": _status_code(exc),
            })
            with self._write_lock:
                self.failed += 1
            raise
        self._append(self.path, {
            "key": key,
            "task": request.task,
            "prompt_version": request.prompt_version,
            "response": response.text,
            "usage": response.usage.to_dict(),
            "latency_s": round(float(response.latency_s or 0.0), 3),
            "model": response.model,
            # Not part of the replay key; stored so a reader can see the effort each recorded answer was asked with.
            "reasoning_effort": request.reasoning_effort,
        })
        with self._write_lock:
            self.recorded += 1
        return response


class EvalReplayBackend(ReplayBackend):
    """:class:`ReplayBackend` that restores recorded usage and counts misses.

    Answers for a key are served in recording order; the usage of the answer
    actually served is attached to the response. Once the recorded answers of
    a key are used up, a key that failed while recording raises
    :class:`RecordedFailure` (for example an empty answer whose re-ask was
    rejected), and any other key is a miss (reason ``"exhausted"``) instead of
    the core behaviour of repeating the last answer.
    """

    def __init__(self, path: Union[str, Path], *, model: str = "replay") -> None:
        super().__init__(path, model=model)
        self._extras: dict[str, list[dict]] = {}
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    self._extras.setdefault(record["key"], []).append({
                        "usage": record.get("usage"),
                        "latency_s": record.get("latency_s"),
                        "model": record.get("model"),
                    })
        self._failures: dict[str, dict] = {}
        failures = failures_path_for(self.path)
        if failures.is_file():
            with failures.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        self._failures.setdefault(record["key"], record)
        self._lock_eval = threading.Lock()
        self._counts: dict[str, int] = {}
        self.served = 0
        self.failures_served = 0
        self.misses: list[dict] = []

    def _miss(self, request: VLMRequest, reason: str) -> None:
        self.misses.append({
            "task": request.task,
            "prompt_version": request.prompt_version,
            "source_name": request.source_name,
            "page": request.page,
            "reason": reason,
        })

    def generate(self, request: VLMRequest) -> VLMResponse:
        key = replay_key(request)
        with self._lock_eval:
            index = self._counts.get(key, 0)
            extras = self._extras.get(key) or []
            if index >= len(extras):
                if key in self._failures:
                    self._counts[key] = index + 1
                    self.failures_served += 1
                    failure = self._failures[key]
                    code = failure.get("status_code")
                    raise RecordedFailure(
                        f"recorded failure for task={request.task!r} page={request.page!r}: "
                        f"{failure.get('error_type', 'error')}" + (f" (HTTP {code})" if code else "")
                    )
                if extras:
                    # The core backend would repeat the last answer; for an evaluation that hides a divergence.
                    self._miss(request, "exhausted")
                    raise ReplayMiss(
                        f"recorded answers exhausted for task={request.task!r} source={request.source_name!r} "
                        f"page={request.page!r} ({len(extras)} recorded, request #{index + 1})"
                    )
                self._miss(request, "not_recorded")
                raise ReplayMiss(
                    f"no recorded response for task={request.task!r} prompt_version={request.prompt_version!r} "
                    f"source={request.source_name!r} page={request.page!r}"
                )
            # Served from this class's own counter (not the core one) so answer, usage and latency stay aligned.
            self._counts[key] = index + 1
            self.served += 1
            extra = extras[index]
            return VLMResponse(
                text=self._answers[key][index],
                # The model name the provider reported while recording (falls back to the manifest's model).
                model=str(extra.get("model") or self.model),
                backend=self.name,
                usage=Usage.from_dict(extra.get("usage")),
                latency_s=float(extra.get("latency_s") or 0.0),
            )

    def unused(self) -> int:
        """Recorded answers that were never served (the replayed run made fewer calls)."""
        with self._lock_eval:
            return sum(max(0, len(v) - self._counts.get(k, 0)) for k, v in self._extras.items())

    def stats(self) -> dict:
        return {
            "file": self.path.name,
            "recorded": len(self),
            "served": self.served,
            "unused": self.unused(),
            "misses": len(self.misses),
            "recorded_failures_served": self.failures_served,
        }
