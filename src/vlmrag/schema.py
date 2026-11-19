"""The chunk record every technique emits, plus JSONL helpers.

Each chunk carries three renderings of the same content (the "dual
representation", see ``docs/techniques/common.md``):

* ``retrieve_content`` - plain text that is embedded and BM25-indexed; tuned for
  recall (labels spelled out, visual descriptions included, no markup noise);
* ``llm_content`` - Markdown/HTML that the answering LLM reads; tuned for
  faithful presentation (tables stay tables);
* ``filtering_content`` - the retrieval text with all whitespace removed, used
  for exact substring filters that must survive line breaks and spacing
  differences in part numbers, codes and dates.

``page`` is always the 1-based page or slide number of the *source* document:
viewers jump to it directly, so chunks are never emitted for virtual or merged
pages and pages are never renumbered.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Union

__all__ = ["Chunk", "DOC_TYPES", "doc_id_for", "write_jsonl", "read_jsonl"]

DOC_TYPES = ("drawing", "slides", "docx")


@dataclass
class Chunk:
    doc_id: str
    source_name: str
    doc_type: str
    page: int
    chunk_index: int
    retrieve_content: str
    llm_content: str
    filtering_content: str
    metadata: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.page, int) or isinstance(self.page, bool) or self.page < 1:
            raise ValueError(f"Chunk.page must be a 1-based source page number (got {self.page!r})")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


def doc_id_for(path: Union[str, os.PathLike]) -> str:
    """First 16 hex characters of the SHA-256 of the file bytes.

    Content-addressed ids stay stable when a file is renamed or moved and change
    when its content changes, which is what cache keys and de-duplication need.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def write_jsonl(chunks: Iterable[Chunk], path: Union[str, os.PathLike]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def append_jsonl(chunks: Iterable[Chunk], path: Union[str, os.PathLike]) -> None:
    with Path(path).open("a", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: Union[str, os.PathLike]) -> list[Chunk]:
    chunks = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.append(Chunk.from_dict(json.loads(line)))
    return chunks
