"""Optional retrieval-summary prefix for chunks.

Why it exists
    Chunks are often written in one language (or in terse table/OCR fragments)
    while users search in another. A short, source-grounded Korean + English
    summary and a keyword list, prepended to ``retrieve_content``, lets both
    dense and BM25 retrieval match queries that share no surface words with the
    raw chunk. ``llm_content`` is left untouched: the answering LLM must read the
    source, never a model-written paraphrase of it. ``filtering_content`` is
    left untouched as well, so exact-match filters only ever match words that
    really occur in the document.

Why it is opt-in per document type (``Settings.retrieval_summary_doc_types``)
    On numeric tables (specification sheets, measurement tables) the summary
    hurt more than it helped: the model "rounded" or re-combined numbers, so the
    prefix contained values that did not exist in the source and attracted
    wrong matches; and the extra text consumed a large part of the 8192-token
    embedding budget, pushing real table rows out of the embedded window. It was
    therefore disabled for those chunks and is enabled only where the source is
    prose-like or visual.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Optional

from ..schema import Chunk
from ..text import normalize_ws
from ..vlm.client import VLMClient
from ..vlm.types import VLMRequest

__all__ = [
    "RETRIEVAL_SUMMARY_PROMPT",
    "SUMMARY_SCHEMA",
    "SOURCE_CHAR_LIMIT",
    "MAX_OUTPUT_TOKENS",
    "summarize_for_retrieval",
    "apply_retrieval_summary",
    "augment_chunks",
]

log = logging.getLogger(__name__)

TASK = "augment.summary"
PROMPT_VERSION = "augment.summary/1"
SOURCE_CHAR_LIMIT = 7000
MAX_OUTPUT_TOKENS = 2400
TIMEOUT_S = 240.0
KEYWORD_LIMIT = 20

RETRIEVAL_SUMMARY_PROMPT = (
    "You are preparing retrieval text for a RAG embedding pipeline. "
    "The source chunk may be in any language, or a mix of languages, and may come from tables, slides, "
    "drawings or documents (including OCR-like fragments). "
    "Use only information grounded in the source. Do not invent names, numbers, causes, actions, owners or effects. "
    "Create detailed but search-friendly Korean and English retrieval text. "
    "Prioritize distinctive terms that separate this chunk from similar chunks: product or model names, part numbers, "
    "process names, defects or issues, causes, actions taken, before/after states, numeric conditions, dates, "
    "screen/menu/module names, document titles and domain-specific abbreviations. "
    "Avoid generic-only keywords such as quality, improvement, issue, document or guide unless they are paired with "
    "a concrete term from the source. "
    "Return exactly one JSON object and nothing else. "
    "Use this schema exactly: "
    '{"summary_ko":"","summary_en":"","keywords_ko":[],"keywords_en":[],"keywords_source":[]}. '
    "summary_ko must be Korean, 2 to 4 concise but specific retrieval sentences. "
    "summary_en must be English, 1 to 3 concise but specific retrieval sentences. "
    "Keyword arrays must contain short retrieval keywords or phrases only, including original-language terms "
    "when useful. "
    "If the source is empty or has no useful searchable detail, return empty strings and empty arrays."
)

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary_ko": {"type": "string"},
        "summary_en": {"type": "string"},
        "keywords_ko": {"type": "array", "items": {"type": "string"}},
        "keywords_en": {"type": "array", "items": {"type": "string"}},
        "keywords_source": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary_ko", "summary_en", "keywords_ko", "keywords_en", "keywords_source"],
    "additionalProperties": False,
}


def _keywords(value: Any) -> list[str]:
    if isinstance(value, list):
        items = value
    else:
        items = re.split(r"[,;/\n]+", "" if value is None else str(value))
    out: list[str] = []
    for item in items:
        keyword = normalize_ws(item)
        if keyword and keyword not in out:
            out.append(keyword)
    return out[:KEYWORD_LIMIT]


def summarize_for_retrieval(
    client: VLMClient, text: str, *, source_name: str = "", page: Optional[int] = None
) -> Optional[dict]:
    """Ask the model for ``{"summary_ko", "summary_en", "keywords": [...]}``.

    The source is capped at 7000 characters and the answer at 2400 output
    tokens. Returns ``None`` for empty input, unusable output or any call
    failure: the summary is an optional enhancement and must never fail a
    document.
    """
    body = (text or "").strip()
    if not body:
        return None
    header = []
    if source_name:
        header.append(f"source_name: {source_name}")
    if page is not None:
        header.append(f"page: {page}")
    source = "\n".join(header + ([""] if header else []) + [body])[:SOURCE_CHAR_LIMIT]
    request = VLMRequest(
        task=TASK,
        prompt_version=PROMPT_VERSION,
        parts=[RETRIEVAL_SUMMARY_PROMPT, f"[SOURCE_CHUNK]\n{source}"],
        json_schema=SUMMARY_SCHEMA,
        schema_name="retrieval_summary",
        max_output_tokens=MAX_OUTPUT_TOKENS,
        reasoning_effort="low",
        timeout_s=TIMEOUT_S,
        source_name=source_name,
        page=page,
    )
    try:
        value, _ = client.call_json(request, parse_retries=2)
    except Exception as exc:  # noqa: BLE001 - optional step, never fatal
        log.warning("retrieval summary skipped for %s page=%s: %s", source_name, page, exc)
        return None
    if not isinstance(value, dict):
        return None
    summary_ko = normalize_ws(value.get("summary_ko"))
    summary_en = normalize_ws(value.get("summary_en"))
    keywords: list[str] = []
    for key in ("keywords_ko", "keywords_en", "keywords_source", "keywords"):
        for keyword in _keywords(value.get(key)):
            if keyword not in keywords:
                keywords.append(keyword)
    if not summary_ko and not summary_en and not keywords:
        return None
    return {"summary_ko": summary_ko, "summary_en": summary_en, "keywords": keywords}


def build_prefix(chunk: Chunk, summary: dict) -> str:
    summary_ko = normalize_ws(summary.get("summary_ko"))
    summary_en = normalize_ws(summary.get("summary_en"))
    keywords = _keywords(summary.get("keywords"))
    if not (summary_ko or summary_en or keywords):
        return ""
    parts = [
        f"doc_type: {chunk.doc_type}",
        f"source_name: {chunk.source_name}",
        f"page: {chunk.page}",
        f"Korean retrieval summary: {summary_ko}" if summary_ko else "",
        f"English retrieval summary: {summary_en}" if summary_en else "",
        f"retrieval keywords: {' | '.join(keywords)}" if keywords else "",
    ]
    return " | ".join(p for p in parts if p)


def apply_retrieval_summary(chunk: Chunk, summary: Optional[dict]) -> Chunk:
    """Return a copy of ``chunk`` whose ``retrieve_content`` starts with the summary.

    Only ``retrieve_content`` changes; ``llm_content`` and ``filtering_content``
    stay source-only (see the module docstring). Applying the same summary
    twice is a no-op, and empty chunks or empty summaries are returned
    unchanged.
    """
    if not summary:
        return chunk
    base = (chunk.retrieve_content or "").strip()
    prefix = build_prefix(chunk, summary)
    if not base or not prefix or base.startswith(prefix):
        return chunk
    provenance = dict(chunk.provenance)
    provenance["retrieval_summary"] = {"task": TASK, "prompt_version": PROMPT_VERSION}
    # The shared task -> version map lists every prompt that shaped the chunk.
    provenance["prompt_versions"] = {**(provenance.get("prompt_versions") or {}), TASK: PROMPT_VERSION}
    metadata = dict(chunk.metadata)
    metadata["retrieval_summary"] = {
        "summary_ko": normalize_ws(summary.get("summary_ko")),
        "summary_en": normalize_ws(summary.get("summary_en")),
        "keywords": _keywords(summary.get("keywords")),
    }
    return dataclasses.replace(
        chunk, retrieve_content=f"{prefix}\n{base}", metadata=metadata, provenance=provenance
    )


def augment_chunks(client: VLMClient, chunks: Iterable[Chunk], *, workers: int = 4) -> list[Chunk]:
    """Summarise and prefix every chunk (in parallel threads; order preserved)."""
    items = list(chunks)

    def one(chunk: Chunk) -> Chunk:
        source = chunk.retrieve_content or chunk.llm_content
        summary = summarize_for_retrieval(client, source, source_name=chunk.source_name, page=chunk.page)
        return apply_retrieval_summary(chunk, summary)

    if not items:
        return []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(items)))) as pool:
        return list(pool.map(one, items))
