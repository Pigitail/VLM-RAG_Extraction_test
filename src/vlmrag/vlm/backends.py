"""Model backends: OpenAI / Azure OpenAI (Responses API), OpenAI-compatible chat
servers, and deterministic backends for tests, dry runs and record/replay.

Backends do exactly one thing: translate a :class:`VLMRequest` into one provider
call and the provider's answer back into a :class:`VLMResponse`. Retries,
concurrency limits, caching and usage accounting live in
:class:`vlmrag.vlm.client.VLMClient`, so they behave the same for every backend.
For the same reason the SDK clients are created with ``max_retries=0``: letting
both the SDK and the client retry would multiply attempts and hide rate limits.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, Union

from .types import ImageInput, Usage, VLMError, VLMRequest, VLMResponse

__all__ = [
    "OpenAIResponsesBackend",
    "ChatCompletionsBackend",
    "ScriptedBackend",
    "NullBackend",
    "RecordingBackend",
    "ReplayBackend",
    "ReplayMiss",
    "replay_key",
    "make_backend",
    "extract_response_text",
]

log = logging.getLogger(__name__)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from an SDK object or a plain dict (proxies and fakes return dicts)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_text(value: Any) -> str:
    return "" if value is None else str(value)


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------- Responses API


def extract_response_text(response: Any) -> str:
    """Collect the assistant text from a Responses API result.

    ``response.output`` is a list of items (reasoning items, messages, tool
    calls); only ``output_text``/``text`` pieces of message content carry the
    answer. Pieces of one message are concatenated as-is (a piece boundary can
    fall inside a JSON string); messages are joined with newlines.
    ``output_text`` is the SDK's convenience concatenation of those same
    pieces, so it is used only as a fallback: combining both would duplicate the
    answer. Identical messages are de-duplicated because some gateways echo the
    final message twice.
    """
    if isinstance(response, str):
        return response.strip()
    chunks: list[str] = []
    seen: set[str] = set()
    for item in _get(response, "output", None) or []:
        pieces = []
        for piece in _get(item, "content", None) or []:
            if _get(piece, "type", "") in ("output_text", "text"):
                pieces.append(_as_text(_get(piece, "text", "")))
        text = "".join(pieces).strip()
        if text and text not in seen:
            seen.add(text)
            chunks.append(text)
    if chunks:
        return "\n".join(chunks).strip()
    for name in ("output_text", "text"):
        value = _get(response, name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _responses_usage(response: Any) -> Usage:
    usage = _get(response, "usage", None)
    if usage is None:
        return Usage()
    details = _get(usage, "output_tokens_details", None)
    input_tokens = _int(_get(usage, "input_tokens", 0))
    output_tokens = _int(_get(usage, "output_tokens", 0))
    total = _int(_get(usage, "total_tokens", 0)) or input_tokens + output_tokens
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=_int(_get(details, "reasoning_tokens", 0)),
        total_tokens=total,
    )


class OpenAIResponsesBackend:
    """OpenAI or Azure OpenAI through the Responses API.

    Images go in as ``input_image`` parts holding a PNG data URL, preceded by
    their position label as an ``input_text`` part. When the request carries a
    JSON schema and structured output is enabled, ``text.format`` asks for a
    strict ``json_schema`` answer; the client still parses defensively because
    gateways and older deployments may ignore it.
    """

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        name: str = "openai",
        structured_output: bool = True,
        image_detail: Optional[str] = None,
        default_timeout_s: Optional[float] = None,
    ) -> None:
        self.client = client
        self.model = model
        self.name = name
        self.structured_output = structured_output
        self.image_detail = image_detail
        self.default_timeout_s = default_timeout_s

    def build_content(self, request: VLMRequest) -> list[dict]:
        content: list[dict] = []
        for part in request.parts:
            if isinstance(part, ImageInput):
                if part.label:
                    content.append({"type": "input_text", "text": part.label})
                image: dict = {"type": "input_image", "image_url": part.data_url()}
                if self.image_detail:
                    image["detail"] = self.image_detail
                content.append(image)
            elif part:
                content.append({"type": "input_text", "text": str(part)})
        return content

    def build_kwargs(self, request: VLMRequest) -> dict:
        kwargs: dict = {
            "model": self.model,
            "input": [{"role": "user", "content": self.build_content(request)}],
        }
        if request.instructions:
            kwargs["instructions"] = request.instructions
        if request.json_schema is not None and self.structured_output:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": request.schema_name or "result",
                    "schema": request.json_schema,
                    "strict": True,
                }
            }
        if request.reasoning_effort:
            kwargs["reasoning"] = {"effort": request.reasoning_effort}
        if request.max_output_tokens:
            kwargs["max_output_tokens"] = int(request.max_output_tokens)
        timeout = request.timeout_s or self.default_timeout_s
        if timeout:
            kwargs["timeout"] = float(timeout)
        return kwargs

    def generate(self, request: VLMRequest) -> VLMResponse:
        started = time.monotonic()
        response = self.client.responses.create(**self.build_kwargs(request))
        latency = time.monotonic() - started
        status = _get(response, "status", None)
        if status == "incomplete":
            reason = _get(_get(response, "incomplete_details", None), "reason", "")
            log.warning("%s: response incomplete (%s); parsing partial output", request.task, reason or "unknown")
        return VLMResponse(
            text=extract_response_text(response),
            model=_as_text(_get(response, "model", "")) or self.model,
            backend=self.name,
            usage=_responses_usage(response),
            latency_s=latency,
        )


# --------------------------------------------------------------------------- Chat Completions


def _chat_text(response: Any) -> str:
    choices = _get(response, "choices", None) or []
    if not choices:
        return ""
    message = _get(choices[0], "message", None)
    content = _get(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):  # some servers return a list of typed parts
        texts = [_as_text(_get(p, "text", "")) for p in content if _get(p, "type", "text") in ("text", "output_text")]
        return "\n".join(t for t in texts if t).strip()
    return ""


def _chat_usage(response: Any) -> Usage:
    usage = _get(response, "usage", None)
    if usage is None:
        return Usage()
    details = _get(usage, "completion_tokens_details", None)
    prompt = _int(_get(usage, "prompt_tokens", 0))
    completion = _int(_get(usage, "completion_tokens", 0))
    return Usage(
        input_tokens=prompt,
        output_tokens=completion,
        reasoning_tokens=_int(_get(details, "reasoning_tokens", 0)),
        total_tokens=_int(_get(usage, "total_tokens", 0)) or prompt + completion,
    )


class ChatCompletionsBackend:
    """Any OpenAI-compatible ``/chat/completions`` server (vLLM, Ollama, LM Studio, ...).

    Uses ``text`` / ``image_url`` content parts and ``max_tokens`` (the field
    every compatible server understands). ``response_format`` with a JSON
    schema is sent when structured output is enabled; turn it off for servers
    that reject it - the JSON recovery in the client still applies.
    ``reasoning_effort`` is only forwarded when ``send_reasoning_effort`` is
    true, because most open-weight servers reject unknown parameters.
    """

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        name: str = "compatible",
        structured_output: bool = True,
        send_reasoning_effort: bool = False,
        image_detail: Optional[str] = None,
        default_timeout_s: Optional[float] = None,
    ) -> None:
        self.client = client
        self.model = model
        self.name = name
        self.structured_output = structured_output
        self.send_reasoning_effort = send_reasoning_effort
        self.image_detail = image_detail
        self.default_timeout_s = default_timeout_s

    def build_messages(self, request: VLMRequest) -> list[dict]:
        content: list[dict] = []
        for part in request.parts:
            if isinstance(part, ImageInput):
                if part.label:
                    content.append({"type": "text", "text": part.label})
                image_url: dict = {"url": part.data_url()}
                if self.image_detail:
                    image_url["detail"] = self.image_detail
                content.append({"type": "image_url", "image_url": image_url})
            elif part:
                content.append({"type": "text", "text": str(part)})
        messages: list[dict] = []
        if request.instructions:
            messages.append({"role": "system", "content": request.instructions})
        messages.append({"role": "user", "content": content})
        return messages

    def build_kwargs(self, request: VLMRequest) -> dict:
        kwargs: dict = {"model": self.model, "messages": self.build_messages(request)}
        if request.json_schema is not None and self.structured_output:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_name or "result",
                    "schema": request.json_schema,
                    "strict": True,
                },
            }
        if request.max_output_tokens:
            kwargs["max_tokens"] = int(request.max_output_tokens)
        if request.reasoning_effort and self.send_reasoning_effort:
            kwargs["reasoning_effort"] = request.reasoning_effort
        timeout = request.timeout_s or self.default_timeout_s
        if timeout:
            kwargs["timeout"] = float(timeout)
        return kwargs

    def generate(self, request: VLMRequest) -> VLMResponse:
        started = time.monotonic()
        response = self.client.chat.completions.create(**self.build_kwargs(request))
        latency = time.monotonic() - started
        return VLMResponse(
            text=_chat_text(response),
            model=_as_text(_get(response, "model", "")) or self.model,
            backend=self.name,
            usage=_chat_usage(response),
            latency_s=latency,
        )


# --------------------------------------------------------------------------- deterministic backends

Responder = Callable[[VLMRequest], Union[str, dict, list, VLMResponse]]


def _approx_tokens(text: str) -> int:
    return (len(text) + 3) // 4


class ScriptedBackend:
    """Test backend: ``responder(request)`` returns the model output.

    A ``dict``/``list`` result is ``json.dumps``'d, a ``str`` is returned as is
    and a :class:`VLMResponse` is passed through untouched. Exceptions raised by
    the responder propagate, which is how tests exercise the retry logic. Usage
    is a rough chars/4 estimate so ledger tests see non-zero numbers. Every
    request is appended to ``calls`` (thread-safe).
    """

    def __init__(self, responder: Responder, *, model: str = "scripted", name: str = "scripted") -> None:
        self.responder = responder
        self.model = model
        self.name = name
        self.calls: list[VLMRequest] = []
        self._lock = threading.Lock()

    def generate(self, request: VLMRequest) -> VLMResponse:
        with self._lock:
            self.calls.append(request)
        result = self.responder(request)
        if isinstance(result, VLMResponse):
            return result
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        prompt_chars = sum(len(p) for p in request.text_parts())
        input_tokens = (prompt_chars + 3) // 4 + 85 * len(request.images())
        output_tokens = _approx_tokens(text)
        return VLMResponse(
            text=text,
            model=self.model,
            backend=self.name,
            usage=Usage(input_tokens, output_tokens, 0, input_tokens + output_tokens),
            latency_s=0.0,
        )


class NullBackend:
    """Returns ``"{}"`` for every request.

    Useful for dry runs: the whole pipeline (rendering, prompting, parsing,
    validation) runs without network access and every technique's fallback path
    is exercised.
    """

    name = "null"

    def __init__(self, model: str = "null") -> None:
        self.model = model

    def generate(self, request: VLMRequest) -> VLMResponse:
        return VLMResponse(text="{}", model=self.model, backend=self.name, usage=Usage(), latency_s=0.0)


# --------------------------------------------------------------------------- record / replay


def replay_key(request: VLMRequest) -> str:
    """Stable key for record/replay.

    Built only from fields that do not depend on the model, the machine or the
    renderer: task, prompt version, source file name, page, the ordered text
    parts and image labels. Image bytes are excluded on purpose - PDFium,
    LibreOffice and font versions change pixels across platforms, and a replay
    file recorded on one machine must still match on another. ``doc_id`` is
    excluded because regenerated fixtures differ byte-wise even when their
    content is identical. Line endings are normalised for the same reason.
    """
    sequence = []
    for part in request.parts:
        if isinstance(part, ImageInput):
            sequence.append(["image", part.label.replace("\r\n", "\n")])
        else:
            sequence.append(["text", str(part).replace("\r\n", "\n")])
    payload = {
        "task": request.task,
        "prompt_version": request.prompt_version,
        "source_name": request.source_name,
        "page": request.page,
        "parts": sequence,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ReplayMiss(KeyError):
    """Raised by :class:`ReplayBackend` when no recorded response matches a request."""

    def __str__(self) -> str:  # KeyError's default str() adds confusing quotes
        return str(self.args[0]) if self.args else "replay miss"


class RecordingBackend:
    """Wraps another backend and appends every exchange to a JSONL file.

    Each line is ``{"key", "task", "prompt_version", "response"}``. Recorded
    files let anyone re-run a technique end to end without credentials (see
    :class:`ReplayBackend`). Failed calls are not recorded.
    """

    def __init__(self, inner: Any, path: Union[str, Path]) -> None:
        self.inner = inner
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.name = getattr(inner, "name", "recording")
        self.model = getattr(inner, "model", "")
        self._lock = threading.Lock()

    def generate(self, request: VLMRequest) -> VLMResponse:
        response = self.inner.generate(request)
        record = {
            "key": replay_key(request),
            "task": request.task,
            "prompt_version": request.prompt_version,
            "response": response.text,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
        return response


class ReplayBackend:
    """Serves responses recorded by :class:`RecordingBackend`.

    When the same key was recorded several times (identical requests), the
    answers are served in recording order and the last one repeats.
    """

    name = "replay"

    def __init__(self, path: Union[str, Path], *, model: str = "replay") -> None:
        self.path = Path(path)
        self.model = model
        self._answers: dict[str, list[str]] = {}
        self._served: dict[str, int] = {}
        self._lock = threading.Lock()
        with self.path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self._answers.setdefault(record["key"], []).append(record["response"])
                except (ValueError, KeyError) as exc:
                    raise VLMError(f"{self.path.name}:{number}: invalid replay record ({exc})") from exc

    def __len__(self) -> int:
        return sum(len(v) for v in self._answers.values())

    def generate(self, request: VLMRequest) -> VLMResponse:
        key = replay_key(request)
        with self._lock:
            answers = self._answers.get(key)
            if not answers:
                raise ReplayMiss(
                    f"no recorded response for task={request.task!r} prompt_version={request.prompt_version!r} "
                    f"source={request.source_name!r} page={request.page!r}"
                )
            index = self._served.get(key, 0)
            self._served[key] = index + 1
            text = answers[min(index, len(answers) - 1)]
        return VLMResponse(text=text, model=self.model, backend=self.name, usage=Usage(), latency_s=0.0)


# --------------------------------------------------------------------------- factory


def _azure_v1_base_url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/openai/v1"):
        return base + "/"
    return base + "/openai/v1/"


def make_backend(settings: Any) -> Any:
    """Build the backend selected by ``settings.provider``.

    * ``openai``: OpenAI Responses API (``OPENAI_API_KEY``, optional ``OPENAI_BASE_URL``).
    * ``azure``: Azure OpenAI Responses API. With ``AZURE_OPENAI_API_VERSION`` set,
      the classic ``AzureOpenAI(api_version=...)`` client is used; without it the
      newer version-less v1 endpoint (``<endpoint>/openai/v1/``) is used through
      the plain OpenAI client. ``VLMRAG_MODEL`` is the deployment name.
    * ``compatible``: Chat Completions against ``OPENAI_BASE_URL``.
    * ``null``: :class:`NullBackend`.
    """
    provider = (settings.provider or "").lower()
    if provider == "null":
        return NullBackend()
    import openai  # imported lazily: keeps CLI start-up and --help fast

    # Retries belong to VLMClient; an unset timeout keeps the SDK default instead of "no timeout".
    common: dict = {"max_retries": 0}
    if settings.timeout_s:
        common["timeout"] = float(settings.timeout_s)

    if provider == "openai":
        if not settings.openai_api_key:
            raise VLMError("provider 'openai' needs OPENAI_API_KEY in the environment")
        client = openai.OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
            **common,
        )
        return OpenAIResponsesBackend(
            client, model=settings.model, name="openai", structured_output=settings.structured_output
        )
    if provider == "azure":
        if not settings.azure_api_key or not settings.azure_endpoint:
            raise VLMError("provider 'azure' needs AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT")
        if settings.azure_api_version:
            client = openai.AzureOpenAI(
                api_key=settings.azure_api_key,
                api_version=settings.azure_api_version,
                azure_endpoint=settings.azure_endpoint,
                **common,
            )
        else:
            client = openai.OpenAI(
                api_key=settings.azure_api_key,
                base_url=_azure_v1_base_url(settings.azure_endpoint),
                **common,
            )
        return OpenAIResponsesBackend(
            client, model=settings.model, name="azure", structured_output=settings.structured_output
        )
    if provider == "compatible":
        if not settings.openai_base_url:
            raise VLMError("provider 'compatible' needs OPENAI_BASE_URL (e.g. http://localhost:8000/v1)")
        client = openai.OpenAI(
            # Local servers usually ignore the key, but the SDK insists on one.
            api_key=settings.openai_api_key or "none",
            base_url=settings.openai_base_url,
            **common,
        )
        return ChatCompletionsBackend(
            client, model=settings.model, name="compatible", structured_output=settings.structured_output
        )
    raise VLMError(f"unknown provider {settings.provider!r}")
