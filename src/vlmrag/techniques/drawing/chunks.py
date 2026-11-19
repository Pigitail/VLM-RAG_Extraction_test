"""Turn a title-block result plus a page's text layer into one :class:`~vlmrag.schema.Chunk`.

``retrieve_content`` is ordered so the most specific, model-read facts come
first (they survive truncation by the embedding model) and the raw text layer
comes last:

1. file name and ``page N``;
2. title-block pairs (``Topic: value``), then table pairs and table rows;
3. the list of title-block labels (layout columns);
4. ``[Structured fields]`` - one line per field entry with field/part/tags;
5. domain synonym lines for terms the page mentions;
6. ``[Drawing text layer]`` - the page's own PDF text.

``llm_content`` holds the same facts as Markdown (title block and metadata
table as tables). ``filtering_content`` is the retrieval text without
whitespace. Structured data (fields, pairs, rows, entries) goes to
``metadata``; how it was produced (model, prompts, crops, fallbacks, usage)
goes to ``provenance``.
"""

from __future__ import annotations

from typing import Any

from ...schema import Chunk
from ...text import dedupe_keep_order, filtering_text, markdown_table
from .config import DrawingConfig
from .entries import (
    domain_synonym_text,
    entries_from_text,
    entries_text,
    field_values,
    layout_columns,
    layout_columns_text,
    merge_entries,
)
from .parsing import table_rows_matrix
from .prompts import PROMPT_VERSIONS
from .title_block import TitleBlockResult

__all__ = ["build_chunk", "DOC_TYPE"]

DOC_TYPE = "drawing"


def build_chunk(
    *,
    doc_id: str,
    source_name: str,
    page_index: int,
    page_count: int,
    page_text: str,
    rotation: int,
    title_block: TitleBlockResult,
    cfg: DrawingConfig,
    model: str = "",
    backend: str = "",
) -> Chunk:
    """Build the chunk of page ``page_index`` (0-based); ``title_block`` may come from another page."""
    page = page_index + 1
    chunk_name = f"page {page}"
    tb = title_block
    domain = cfg.domain

    raw_entries = entries_from_text(
        "\n".join([page_text, source_name, chunk_name]),
        "raw_chunk",
        schema=cfg.schema,
        domain=domain,
        chunk=chunk_name,
        limit=cfg.field_entry_limit,
        source_text_limit=cfg.field_source_text_limit,
    )
    entries = merge_entries(tb.field_entries, raw_entries, limit=cfg.field_entry_limit)
    entry_lines = entries_text(entries)
    columns = layout_columns(tb.pairs)
    columns_line = layout_columns_text(columns)
    synonyms = domain_synonym_text(
        "\n".join([page_text, source_name, chunk_name, tb.pairs_text, tb.table_pairs_text]), domain
    )
    text_layer = page_text if cfg.include_text_layer else ""

    retrieve_parts = [
        source_name,
        chunk_name,
        tb.pairs_text,
        tb.table_pairs_text,
        tb.table_rows_text,
        columns_line,
        f"[Structured fields]\n{entry_lines}" if entry_lines else "",
        synonyms,
        f"[Drawing text layer]\n{text_layer}" if text_layer else "",
    ]
    retrieve_content = "\n".join(dedupe_keep_order(p.strip() for p in retrieve_parts if p and p.strip()))

    llm_lines = [f"## Title block - {source_name}, page {page}"]
    if tb.page != page:
        llm_lines.append(f"(Title block read from page {tb.page} of this file.)")
    llm_lines += ["", markdown_table([["Field", "Value"]] + [[p["topic"], p["value"]] for p in tb.pairs])]
    matrix = table_rows_matrix(tb.table_rows)
    if matrix:
        llm_lines += ["", "### Metadata table", markdown_table(matrix, header=False)]
    if entry_lines:
        llm_lines += ["", "### Structured fields", entry_lines]
    if synonyms:
        llm_lines += ["", "### Domain synonyms", synonyms]
    if text_layer:
        llm_lines += ["", "### Drawing text layer", text_layer]
    llm_content = "\n".join(llm_lines).strip()

    metadata: dict[str, Any] = {
        "title_block": {k: v for k, v in tb.fields.items() if v},
        "title_block_page": tb.page,
        "title_block_pairs": list(tb.pairs),
        "raw_pairs": list(tb.raw_pairs),
        "table_rows": list(tb.table_rows),
        "table_pairs": list(tb.table_pairs),
        "layout_columns": columns,
        "field_entries": entries,
        "field_values": field_values(entries),
        "page_count": page_count,
        "rotation": rotation,
        "has_text_layer": bool(page_text),
    }
    render = cfg.render_summary()
    render["images"] = list(tb.images)
    provenance: dict[str, Any] = {
        "technique": DOC_TYPE,
        "model": model,
        "backend": backend,
        # Only the prompts that actually ran for this title block (task -> version).
        "prompt_versions": {c["task"]: c["prompt_version"] for c in tb.calls if c.get("task") in PROMPT_VERSIONS},
        "schema": cfg.schema.name,
        "title_block_pages": cfg.title_block_pages,
        "title_block_page": tb.page,
        "render": render,
        "hints": dict(tb.hints),
        "fallbacks": list(tb.fallbacks),
        "field_sources": dict(tb.field_sources),
        "gate": {"min_pairs": cfg.min_pairs, "pairs_found": len(tb.pairs)},
        "calls": list(tb.calls),
        "responses": {"vision": tb.raw_text, "table": tb.table_text, "normalize": tb.normalized_text},
    }
    return Chunk(
        doc_id=doc_id,
        source_name=source_name,
        doc_type=DOC_TYPE,
        page=page,
        chunk_index=page_index,
        retrieve_content=retrieve_content,
        llm_content=llm_content,
        filtering_content=filtering_text(retrieve_content),
        metadata=metadata,
        provenance=provenance,
    )
