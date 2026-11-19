"""On-disk response cache keyed by everything that can change a model answer.

Extraction runs are long and expensive; re-running after a crash, a code change
in post-processing, or a new evaluation should not pay for identical model calls
again. The key covers the model, the call site (task + prompt version), the
document and page, every text part, the SHA-256 of every image, the JSON schema
and the generation parameters, so changing any of those misses the cache.

Entries are small JSON files written atomically (temp file + ``os.replace``),
which makes the cache safe to share between threads and processes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Union

from .types import ImageInput, Usage, VLMRequest, VLMResponse, _json_safe

__all__ = ["DiskCache", "cache_key"]

log = logging.getLogger(__name__)


def cache_key(request: VLMRequest, model: str) -> str:
    parts = []
    for part in request.parts:
        if isinstance(part, ImageInput):
            parts.append({"image": part.sha256(), "label": part.label})
        else:
            parts.append({"text": part})
    payload = {
        "model": model,
        "task": request.task,
        "prompt_version": request.prompt_version,
        "doc_id": request.doc_id,
        "page": request.page,
        "parts": parts,
        "instructions": request.instructions,
        "schema_name": request.schema_name,
        "json_schema": _json_safe(request.json_schema),
        "max_output_tokens": request.max_output_tokens,
        "reasoning_effort": request.reasoning_effort,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DiskCache:
    """Directory of ``<key[:2]>/<key>.json`` files holding model responses."""

    def __init__(self, directory: Union[str, os.PathLike]) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, request: VLMRequest, model: str) -> Optional[VLMResponse]:
        path = self._path(cache_key(request, model))
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:  # corrupt entry: treat as a miss
            log.warning("ignoring unreadable cache entry %s: %s", path.name, exc)
            return None
        return VLMResponse(
            text=data.get("text", ""),
            model=data.get("model", model),
            backend=data.get("backend", ""),
            usage=Usage.from_dict(data.get("usage")),
            latency_s=0.0,
            cached=True,
        )

    def put(self, request: VLMRequest, model: str, response: VLMResponse) -> None:
        path = self._path(cache_key(request, model))
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "task": request.task,
            "prompt_version": request.prompt_version,
            "page": request.page,
            "text": response.text,
            "model": response.model,
            "backend": response.backend,
            "usage": response.usage.to_dict(),
        }
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False)
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
