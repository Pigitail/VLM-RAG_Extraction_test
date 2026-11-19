from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import openai
import pytest

from vlmrag.vlm.backends import ScriptedBackend
from vlmrag.vlm.cache import DiskCache
from vlmrag.vlm.client import VLMClient, is_retryable, retry_after_seconds
from vlmrag.vlm.types import ImageInput, TransientVLMError, Usage, VLMRequest, VLMResponse
from vlmrag.vlm.usage import UsageLedger


def _status_error(cls, status, headers=None):
    response = SimpleNamespace(request=None, status_code=status, headers=headers or {})
    return cls(f"status {status}", response=response, body=None)


def _request(task="unit.task", page=1, image=b"img", text="prompt"):
    return VLMRequest(task=task, prompt_version=f"{task}/1", parts=[text, ImageInput(image, label="FULL")], page=page)


class FlakyResponder:
    """Raises the given exceptions in order, then answers."""

    def __init__(self, errors, answer='{"ok": true}'):
        self.errors = list(errors)
        self.answer = answer
        self.attempts = 0

    def __call__(self, request):
        self.attempts += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.answer


@pytest.mark.parametrize(
    "exc, expected",
    [
        (_status_error(openai.RateLimitError, 429), True),
        (_status_error(openai.InternalServerError, 500), True),
        (_status_error(openai.APIStatusError, 503), True),
        (_status_error(openai.APIStatusError, 408), True),
        (openai.APITimeoutError(request=None), True),
        (openai.APIConnectionError(request=None), True),
        (TimeoutError("slow"), True),
        (ConnectionResetError("reset"), True),
        (TransientVLMError("odd state"), True),
        (_status_error(openai.BadRequestError, 400), False),
        (_status_error(openai.AuthenticationError, 401), False),
        (_status_error(openai.PermissionDeniedError, 403), False),
        (_status_error(openai.NotFoundError, 404), False),
        (_status_error(openai.UnprocessableEntityError, 422), False),
        (ValueError("bug"), False),
        (KeyError("bug"), False),
    ],
)
def test_retry_classification(exc, expected):
    assert is_retryable(exc) is expected


def test_retries_transient_errors_then_succeeds():
    responder = FlakyResponder(
        [
            _status_error(openai.RateLimitError, 429),
            _status_error(openai.InternalServerError, 502),
            openai.APITimeoutError(request=None),
        ]
    )
    sleeps = []
    client = VLMClient(ScriptedBackend(responder), max_retries=3, sleep=sleeps.append, rng=random.Random(0))
    response = client.call(_request())
    assert response.text == '{"ok": true}'
    assert responder.attempts == 4
    assert len(sleeps) == 3
    # Exponential backoff with equal jitter: delay in [ceiling/2, ceiling], ceiling = base * 2**(n-1).
    for n, delay in enumerate(sleeps, 1):
        ceiling = min(32.0, 1.0 * 2 ** (n - 1))
        assert ceiling / 2 <= delay <= ceiling


def test_gives_up_after_max_retries():
    responder = FlakyResponder([_status_error(openai.RateLimitError, 429)] * 10)
    client = VLMClient(ScriptedBackend(responder), max_retries=2, sleep=lambda s: None)
    with pytest.raises(openai.RateLimitError):
        client.call(_request())
    assert responder.attempts == 3  # first attempt + 2 retries


def test_never_retries_client_errors():
    responder = FlakyResponder([_status_error(openai.BadRequestError, 400)])
    sleeps = []
    client = VLMClient(ScriptedBackend(responder), max_retries=5, sleep=sleeps.append)
    with pytest.raises(openai.BadRequestError):
        client.call(_request())
    assert responder.attempts == 1
    assert sleeps == []


def test_retry_after_header_is_honoured_and_capped():
    exc = _status_error(openai.RateLimitError, 429, headers={"retry-after": "3"})
    assert retry_after_seconds(exc) == 3.0
    sleeps = []
    client = VLMClient(ScriptedBackend(FlakyResponder([exc])), sleep=sleeps.append)
    client.call(_request())
    assert sleeps == [3.0]
    long_wait = _status_error(openai.RateLimitError, 429, headers={"retry-after": "120"})
    sleeps.clear()
    VLMClient(ScriptedBackend(FlakyResponder([long_wait])), max_delay_s=5, sleep=sleeps.append).call(_request())
    assert sleeps == [5]


def test_concurrency_limit_is_enforced():
    lock = threading.Lock()
    state = {"current": 0, "peak": 0}

    def responder(request):
        with lock:
            state["current"] += 1
            state["peak"] = max(state["peak"], state["current"])
        time.sleep(0.05)
        with lock:
            state["current"] -= 1
        return "{}"

    client = VLMClient(ScriptedBackend(responder), max_concurrency=2)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: client.call(_request(page=i)), range(1, 9)))
    assert state["peak"] == 2


def test_semaphore_is_released_while_backing_off():
    """A call sleeping between retries must not block other calls."""
    gate = threading.Event()
    calls = []

    def responder(request):
        calls.append(request.page)
        if request.page == 1 and len([p for p in calls if p == 1]) == 1:
            raise TimeoutError("first attempt of page 1 times out")
        return "{}"

    def slow_sleep(seconds):
        gate.wait(timeout=2)

    client = VLMClient(ScriptedBackend(responder), max_concurrency=1, sleep=slow_sleep)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(client.call, _request(page=1))
        time.sleep(0.05)
        second = pool.submit(client.call, _request(page=2))
        second.result(timeout=1)  # completes although page 1 is still backing off
        gate.set()
        first.result(timeout=2)


def test_cache_hit_path(tmp_path):
    backend = ScriptedBackend(lambda request: '{"answer": 1}')
    cache = DiskCache(tmp_path / "cache")
    client = VLMClient(backend, cache=cache)
    first = client.call(_request())
    second = client.call(_request())
    assert len(backend.calls) == 1
    assert not first.cached and second.cached
    assert second.text == first.text
    assert second.model == first.model
    summary = client.usage.summary()["tasks"]["unit.task"]
    assert summary["calls"] == 2 and summary["cached"] == 1
    assert summary["input_tokens"] == first.usage.input_tokens  # cached call adds no tokens
    # Any change to the request (here: image bytes) misses the cache.
    client.call(_request(image=b"other"))
    assert len(backend.calls) == 2
    # A second client sharing the directory benefits too.
    other = VLMClient(ScriptedBackend(lambda request: "never"), cache=DiskCache(tmp_path / "cache"))
    assert other.call(_request()).cached


def test_usage_accounting_per_task():
    def responder(request):
        return VLMResponse(
            text="{}", model="m", backend="b", usage=Usage(10, 5, 2, 15), latency_s=0.5
        )

    ledger = UsageLedger()
    client = VLMClient(ScriptedBackend(responder), usage=ledger)
    client.call(_request(task="a.vision"))
    client.call(_request(task="a.vision", page=2))
    client.call(_request(task="b.table"))
    summary = client.usage.summary()
    assert client.usage is ledger
    assert summary["tasks"]["a.vision"] == {
        "calls": 2, "cached": 0, "latency_s": 1.0,
        "input_tokens": 20, "output_tokens": 10, "reasoning_tokens": 4, "total_tokens": 30,
    }
    assert summary["total"]["calls"] == 3
    assert summary["total"]["total_tokens"] == 45
    assert "TOTAL" in client.usage.format_table()


def test_call_json_parses_fenced_output():
    client = VLMClient(ScriptedBackend(lambda request: 'Result:\n```json\n{"title": "Bracket"}\n```'))
    value, response = client.call_json(_request())
    assert value == {"title": "Bracket"}
    assert "```" in response.text


def test_call_json_returns_none_on_garbled_output_and_does_not_cache_it(tmp_path):
    backend = ScriptedBackend(lambda request: "I could not read the image {")
    client = VLMClient(backend, cache=DiskCache(tmp_path))
    value, response = client.call_json(_request())
    assert value is None
    assert response.text.startswith("I could not")
    client.call_json(_request())
    assert len(backend.calls) == 2  # the garbled answer was not cached


def test_call_json_parse_retries_reissue_the_request():
    answers = iter(["garbled", '{"ok": 1}'])
    backend = ScriptedBackend(lambda request: next(answers))
    client = VLMClient(backend)
    value, _ = client.call_json(_request(), parse_retries=1)
    assert value == {"ok": 1}
    assert len(backend.calls) == 2


def test_model_property_and_from_settings(tmp_path):
    from vlmrag.config import Settings

    settings = Settings(provider="null", max_concurrency=3, max_retries=1, cache_dir=str(tmp_path / "c"))
    client = VLMClient.from_settings(settings)
    assert client.model == "null"
    assert client.max_concurrency == 3 and client.max_retries == 1
    assert client.cache is not None
    with pytest.raises(ValueError):
        VLMClient(ScriptedBackend(lambda r: "{}"), max_concurrency=0)
