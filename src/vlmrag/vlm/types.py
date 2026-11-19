"""Backend-neutral request/response types for vision-language model calls.

Techniques describe *what* they want from the model with a :class:`VLMRequest`
(ordered text and image parts, optional JSON schema, limits) and never touch a
provider SDK directly. That keeps the extraction logic testable with scripted
backends and lets the same technique run against OpenAI, Azure OpenAI or any
OpenAI-compatible server.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Union, runtime_checkable

__all__ = [
    "ImageInput",
    "Part",
    "VLMRequest",
    "Usage",
    "VLMResponse",
    "VLMBackend",
    "VLMError",
    "TransientVLMError",
]


@dataclass(frozen=True)
class ImageInput:
    """One PNG image sent to the model, with an optional position label.

    The label (for example ``"TILE r1c2 x=0.24-0.52 y=0.00-0.37 dpi=400"``) is
    emitted by every backend as a text part immediately before the image. Models
    cannot know where a crop came from; telling them explicitly lets the prompt
    say "values near the bottom-right label belong to the title block" and lets
    the model reconcile overlapping tiles.
    """

    png: bytes
    label: str = ""

    def sha256(self) -> str:
        return hashlib.sha256(self.png).hexdigest()

    def data_url(self) -> str:
        return "data:image/png;base64," + base64.b64encode(self.png).decode("ascii")

    def __repr__(self) -> str:  # keep logs readable: never dump image bytes
        return f"ImageInput(label={self.label!r}, bytes={len(self.png)})"


Part = Union[str, ImageInput]


@dataclass
class VLMRequest:
    """A single model call, described independently of any provider SDK.

    ``task`` and ``prompt_version`` identify the call site and the prompt
    revision. They feed the response cache, the record/replay key and the usage
    ledger, so bump ``prompt_version`` whenever the prompt text changes.
    """

    task: str
    prompt_version: str
    parts: list[Part]
    instructions: Optional[str] = None
    json_schema: Optional[dict] = None
    schema_name: str = "result"
    max_output_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None
    timeout_s: Optional[float] = None
    doc_id: str = ""
    source_name: str = ""
    page: Optional[int] = None

    def text_parts(self) -> list[str]:
        return [p for p in self.parts if isinstance(p, str)]

    def images(self) -> list[ImageInput]:
        return [p for p in self.parts if isinstance(p, ImageInput)]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Usage":
        data = data or {}
        return cls(
            input_tokens=int(data.get("input_tokens") or 0),
            output_tokens=int(data.get("output_tokens") or 0),
            reasoning_tokens=int(data.get("reasoning_tokens") or 0),
            total_tokens=int(data.get("total_tokens") or 0),
        )


@dataclass
class VLMResponse:
    text: str
    model: str
    backend: str
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    cached: bool = False


@runtime_checkable
class VLMBackend(Protocol):
    """Anything that turns a :class:`VLMRequest` into a :class:`VLMResponse`."""

    name: str
    model: str

    def generate(self, request: VLMRequest) -> VLMResponse: ...


class VLMError(RuntimeError):
    """Base class for errors raised by this package's VLM layer."""


class TransientVLMError(VLMError):
    """A failure the client should retry (backends raise it for odd but transient states)."""


def _json_safe(value: Any) -> Any:
    """Small helper shared by cache/replay code: make values JSON-serialisable."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)
