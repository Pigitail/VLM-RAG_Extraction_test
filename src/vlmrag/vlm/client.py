"""The one place where model calls are limited, cached, retried and accounted.

Concurrency
    The client owns a :class:`threading.BoundedSemaphore`. VLM calls are
    I/O-bound, so techniques fan out with a ``ThreadPoolExecutor`` and the
    semaphore caps how many requests are in flight. A module-level
    ``multiprocessing.Semaphore`` is deliberately avoided: it is only shared
    between processes created by ``fork``; under ``spawn``/``forkserver`` (the
    default on Windows and macOS, and on Linux from Python 3.14) every worker
    silently gets its own copy and the "global" limit is multiplied by the
    number of processes.

Retries
    Rate limits (429), server errors (5xx), timeouts and connection errors are
    retried with exponential backoff and jitter, honouring ``Retry-After`` when
    the server sends it. Other 4xx errors (bad request, auth, content filter)
    are never retried: repeating them only burns quota. The semaphore is held
    for the request only, never while sleeping between attempts, so a backing
    off call does not block the others.

Cache and usage
    A :class:`~vlmrag.vlm.cache.DiskCache` short-circuits identical requests,
    and every call (cached or not) is recorded in a
    :class:`~vlmrag.vlm.usage.UsageLedger`.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any, Callable, Optional

from .cache import DiskCache
from .jsonparse import extract_json_value
from .types import TransientVLMError, VLMRequest, VLMResponse
from .usage import UsageLedger

__all__ = ["VLMClient", "is_retryable", "retry_after_seconds"]

log = logging.getLogger(__name__)

_RETRYABLE_CLASS_NAMES = {
    "RateLimitError",
    "APITimeoutError",
    "APIConnectionError",
    "InternalServerError",
    "TimeoutException",
    "ConnectError",
    "ReadTimeout",
    "ConnectTimeout",
    "RemoteProtocolError",
}


def _status_code(exc: BaseException) -> Optional[int]:
    for candidate in (exc, getattr(exc, "response", None)):
        code = getattr(candidate, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def is_retryable(exc: BaseException) -> bool:
    """Classify an exception raised by a backend.

    Works by status code and class name (duck typing) so it covers the OpenAI
    SDK, raw HTTP clients and test doubles without importing any of them.
    """
    if isinstance(exc, (TransientVLMError, TimeoutError, ConnectionError)):
        return True
    code = _status_code(exc)
    if code is not None:
        return code in (408, 409, 429) or code >= 500
    names = {cls.__name__ for cls in type(exc).__mro__}
    return bool(names & _RETRYABLE_CLASS_NAMES)


def retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Return the server's ``Retry-After`` hint (seconds) if the error carries one."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
    except AttributeError:
        return None
    try:
        return max(0.0, float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


class VLMClient:
    """Thread-safe front door for every model call made by the techniques.

    ``max_retries`` counts retries after the first attempt, so the default of 3
    allows up to 4 attempts.
    """

    def __init__(
        self,
        backend: Any,
        *,
        max_concurrency: int = 4,
        max_retries: int = 3,
        base_delay_s: float = 1.0,
        max_delay_s: float = 32.0,
        cache: Optional[DiskCache] = None,
        usage: Optional[UsageLedger] = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: Optional[random.Random] = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.backend = backend
        self.max_concurrency = max_concurrency
        self.max_retries = max(0, int(max_retries))
        self.base_delay_s = base_delay_s
        self.max_delay_s = max_delay_s
        self.cache = cache
        self.usage = usage if usage is not None else UsageLedger()
        self._semaphore = threading.BoundedSemaphore(max_concurrency)
        self._sleep = sleep
        self._rng = rng or random.Random()

    @classmethod
    def from_settings(cls, settings: Any, backend: Any = None, **overrides: Any) -> "VLMClient":
        """Build a client (and, unless given, a backend) from :class:`vlmrag.config.Settings`."""
        if backend is None:
            from .backends import make_backend

            backend = make_backend(settings)
        cache = DiskCache(settings.cache_dir) if getattr(settings, "cache_dir", None) else None
        kwargs = dict(max_concurrency=settings.max_concurrency, max_retries=settings.max_retries, cache=cache)
        kwargs.update(overrides)
        return cls(backend, **kwargs)

    @property
    def model(self) -> str:
        return str(getattr(self.backend, "model", ""))

    # ------------------------------------------------------------------ public API

    def call(self, request: VLMRequest) -> VLMResponse:
        """Run one request: cache lookup, limited + retried backend call, usage."""
        return self._call(request, accept=None)

    def call_json(self, request: VLMRequest, *, parse_retries: int = 0) -> tuple[Optional[Any], VLMResponse]:
        """Run ``request`` and parse its output with :func:`extract_json_value`.

        Returns ``(value, response)`` where ``value`` is ``None`` when the
        output could not be parsed. Unparseable output is never written to the
        cache (a garbled answer must not be replayed forever); with
        ``parse_retries > 0`` the request is re-issued that many extra times.
        """
        attempts = max(0, int(parse_retries)) + 1
        response: Optional[VLMResponse] = None
        for attempt in range(1, attempts + 1):
            response = self._call(request, accept=lambda text: extract_json_value(text) is not None)
            value = extract_json_value(response.text)
            if value is not None:
                return value, response
            log.warning(
                "%s page=%s: unparseable JSON output (attempt %d/%d, %d chars)",
                request.task, request.page, attempt, attempts, len(response.text),
            )
        assert response is not None
        return None, response

    # ------------------------------------------------------------------ internals

    def _backoff_delay(self, attempt: int, exc: BaseException) -> float:
        hinted = retry_after_seconds(exc)
        if hinted is not None:
            return min(hinted, self.max_delay_s)
        ceiling = min(self.max_delay_s, self.base_delay_s * (2 ** (attempt - 1)))
        # "Equal jitter": keep at least half the exponential delay, randomise the rest
        # so parallel workers that failed together do not retry in lockstep.
        return ceiling / 2 + self._rng.uniform(0, ceiling / 2)

    def _call(self, request: VLMRequest, accept: Optional[Callable[[str], bool]]) -> VLMResponse:
        model = self.model
        if self.cache is not None:
            hit = self.cache.get(request, model)
            if hit is not None and (accept is None or accept(hit.text)):
                self.usage.add(request.task, hit.usage, 0.0, cached=True)
                log.debug("%s page=%s: cache hit", request.task, request.page)
                return hit

        attempt = 0
        while True:
            attempt += 1
            started = time.monotonic()
            try:
                with self._semaphore:
                    response = self.backend.generate(request)
            except Exception as exc:  # noqa: BLE001 - classified below
                if not is_retryable(exc) or attempt > self.max_retries:
                    log.error(
                        "%s page=%s: %s after %d attempt(s): %s",
                        request.task, request.page, type(exc).__name__, attempt, exc,
                    )
                    raise
                delay = self._backoff_delay(attempt, exc)
                log.warning(
                    "%s page=%s: %s (attempt %d/%d), retrying in %.1fs",
                    request.task, request.page, type(exc).__name__, attempt, self.max_retries + 1, delay,
                )
                self._sleep(delay)
                continue
            if not response.latency_s:
                response.latency_s = time.monotonic() - started
            break

        self.usage.add(request.task, response.usage, response.latency_s, cached=False)
        log.info(
            "%s page=%s: %s/%s in %.1fs (in=%d out=%d)",
            request.task, request.page, response.backend, response.model, response.latency_s,
            response.usage.input_tokens, response.usage.output_tokens,
        )
        if self.cache is not None and (accept is None or accept(response.text)):
            self.cache.put(request, model, response)
        return response
