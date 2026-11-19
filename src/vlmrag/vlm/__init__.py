"""VLM call infrastructure: request types, backends, client, JSON recovery, cache, usage."""

from __future__ import annotations

from .backends import (
    ChatCompletionsBackend,
    NullBackend,
    OpenAIResponsesBackend,
    RecordingBackend,
    ReplayBackend,
    ReplayMiss,
    ScriptedBackend,
    make_backend,
    replay_key,
)
from .cache import DiskCache
from .client import VLMClient, is_retryable
from .jsonparse import extract_json_object, extract_json_value
from .types import ImageInput, Part, TransientVLMError, Usage, VLMBackend, VLMError, VLMRequest, VLMResponse
from .usage import UsageLedger

__all__ = [
    "ChatCompletionsBackend",
    "DiskCache",
    "ImageInput",
    "NullBackend",
    "OpenAIResponsesBackend",
    "Part",
    "RecordingBackend",
    "ReplayBackend",
    "ReplayMiss",
    "ScriptedBackend",
    "TransientVLMError",
    "Usage",
    "UsageLedger",
    "VLMBackend",
    "VLMClient",
    "VLMError",
    "VLMRequest",
    "VLMResponse",
    "extract_json_object",
    "extract_json_value",
    "is_retryable",
    "make_backend",
    "replay_key",
]
