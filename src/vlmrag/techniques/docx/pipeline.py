"""Orchestration of the docx technique: XML split -> PDF render -> VLM -> chunks.

Pipeline (one chunk per source page)::

    DOCX --XML--> pages (text, tables, image flags, multi-page tables)
         --LibreOffice--> PDF (only if a page has pictures or a table spans pages)
         --render 220 dpi--> page PNGs
         --VLM, concurrently--> image descriptions (pages with pictures)
                               row->page assignments (tables spanning pages)
         --validate / fall back--> per-page Markdown + retrieval text -> Chunk

The model never supplies text that the XML already has. It adds two things the
XML cannot: what the pictures show, and which rendered page each row of a long
table ended up on. Both are optional enrichments with deterministic fallbacks,
so a missing LibreOffice, a dead endpoint or a garbled answer degrades the
output (no visual descriptions, row placement from the file's own markers) but
never loses document text and never shifts page numbers.
"""

from __future__ import annotations

import json
import logging
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from ...render.images import to_png
from ...render.office import convert, soffice_candidates
from ...schema import Chunk, doc_id_for
from ...text import filtering_text
from ...vlm.types import ImageInput, VLMRequest, VLMResponse
from .answers import parse_image_answer, validate_table_answer
from .assemble import TableAssignment, apply_table_assignments, is_table_heavy, render_page
from .prompts import (
    IMAGE_PROMPT,
    IMAGE_PROMPT_VERSION,
    IMAGE_SCHEMA,
    IMAGE_SCHEMA_NAME,
    IMAGE_TASK,
    TABLE_PROMPT,
    TABLE_PROMPT_VERSION,
    TABLE_SCHEMA_NAME,
    TABLE_TASK,
    table_schema,
)
from .xmlsplit import DocxExtractionError, DocxSplit, MultiPageTable, split_docx

__all__ = [
    "DOC_TYPE",
    "RENDER_DPI",
    "CONVERT_TIMEOUT_S",
    "VLM_TIMEOUT_S",
    "MAX_OUTPUT_TOKENS",
    "PARALLELISM",
    "PARSE_RETRIES",
    "IMAGE_REASONING_EFFORT",
    "TABLE_REASONING_EFFORT",
    "extract",
    "convert_to_pdf",
    "build_image_request",
    "build_table_request",
    "table_payload",
]

log = logging.getLogger(__name__)

DOC_TYPE = "docx"
#: Page render resolution for the model (enough for diagram labels, small enough to send many pages).
RENDER_DPI = 220
#: LibreOffice DOCX -> PDF timeout per binary.
CONVERT_TIMEOUT_S = 60.0
#: Per-request timeout for both VLM calls.
VLM_TIMEOUT_S = 600.0
#: Output budget per call; generous because reasoning tokens count against it.
MAX_OUTPUT_TOKENS = 60000
#: Thread-pool width for the VLM fan-out; the client's semaphore still caps in-flight requests.
PARALLELISM = 10
#: Re-ask once when the answer is empty or not JSON (two attempts per call in total).
PARSE_RETRIES = 1
#: Describing pictures needs little deliberation; counting rows across pages needs more.
IMAGE_REASONING_EFFORT = "low"
TABLE_REASONING_EFFORT = "medium"


# --------------------------------------------------------------------------- requests


def _effort(settings: Any, preferred: str) -> Optional[str]:
    # An explicitly disabled effort (models without reasoning) disables it for every call.
    return preferred if getattr(settings, "reasoning_effort", "low") else None


def build_image_request(
    png: bytes,
    page_number: int,
    page_count: int,
    *,
    doc_id: str = "",
    source_name: str = "",
    reasoning_effort: Optional[str] = IMAGE_REASONING_EFFORT,
) -> VLMRequest:
    """One page image plus its page context; the prompt forbids re-reading document text."""
    return VLMRequest(
        task=IMAGE_TASK,
        prompt_version=IMAGE_PROMPT_VERSION,
        parts=[
            IMAGE_PROMPT,
            f"Page context: page {page_number} of {page_count}.",
            ImageInput(png),
        ],
        json_schema=IMAGE_SCHEMA,
        schema_name=IMAGE_SCHEMA_NAME,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        reasoning_effort=reasoning_effort,
        timeout_s=VLM_TIMEOUT_S,
        doc_id=doc_id,
        source_name=source_name,
        page=page_number,
    )


def table_payload(table: MultiPageTable) -> str:
    """The XML rows (ground truth) and title candidates, as indented JSON."""
    payload = {
        "table_id": table.table_id,
        "start_page": table.start_page,
        "end_page": table.end_page,
        "row_count": len(table.rows),
        "rows": [{"index": i, "cells": list(row)} for i, row in enumerate(table.rows)],
        "preceding_paragraphs": list(table.preceding_paragraphs),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_table_request(
    table: MultiPageTable,
    page_images: list[tuple[int, bytes]],
    page_count: int,
    *,
    doc_id: str = "",
    source_name: str = "",
    reasoning_effort: Optional[str] = TABLE_REASONING_EFFORT,
) -> VLMRequest:
    """Rows as JSON plus one labelled image per page of the table's span."""
    parts: list = [
        TABLE_PROMPT,
        f"Document context: table spans pages {table.start_page}-{table.end_page} of {page_count} total.",
        f"Table metadata and XML-extracted rows (ground truth - do NOT re-OCR):\n{table_payload(table)}",
    ]
    for number, png in page_images:
        parts.append(ImageInput(png, label=f"PDF page {number} image follows:"))
    return VLMRequest(
        task=TABLE_TASK,
        prompt_version=TABLE_PROMPT_VERSION,
        parts=parts,
        json_schema=table_schema(table.pages),
        schema_name=TABLE_SCHEMA_NAME,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        reasoning_effort=reasoning_effort,
        timeout_s=VLM_TIMEOUT_S,
        doc_id=doc_id,
        source_name=source_name,
        page=table.start_page,
    )


# --------------------------------------------------------------------------- run bookkeeping


@dataclass
class _Run:
    """Per-document record of what happened, folded into every chunk's provenance."""

    doc_fallbacks: list[str] = field(default_factory=list)
    page_fallbacks: dict[int, list[str]] = field(default_factory=lambda: defaultdict(list))
    page_calls: dict[int, list[dict]] = field(default_factory=lambda: defaultdict(list))
    prompt_versions: dict[int, dict[str, str]] = field(default_factory=lambda: defaultdict(dict))
    pdf_pages: Optional[int] = None

    def call(
        self,
        pages: list[int],
        request: VLMRequest,
        response: Optional[VLMResponse],
        status: str,
        error: Optional[BaseException] = None,
    ) -> None:
        record: dict[str, Any] = {
            "task": request.task,
            "prompt_version": request.prompt_version,
            "status": status,
            "pages": list(pages),
            "cached": bool(response.cached) if response else False,
            "latency_s": round(response.latency_s, 3) if response else None,
            "usage": response.usage.to_dict() if response else None,
        }
        # Call records share one status vocabulary across techniques (ok, error,
        # empty, unparseable, invalid); the detailed reason stays in "reason" and
        # in the fallback codes, which keep their documented names.
        if status == "failed":
            record["status"] = "error"
        elif status.startswith("invalid:"):
            record["status"], record["reason"] = "invalid", status.split(":", 1)[1]
        if error is not None:
            record["error"] = f"{type(error).__name__}: {error}"
        for page in pages:
            self.page_calls[page].append(record)
            self.prompt_versions[page][request.task] = request.prompt_version


def _call_json(client: Any, request: VLMRequest) -> tuple[Any, Optional[VLMResponse], Optional[BaseException]]:
    """Run one call; transport errors become a value so one failed page never aborts the document."""
    try:
        value, response = client.call_json(request, parse_retries=PARSE_RETRIES)
        return value, response, None
    except Exception as exc:  # noqa: BLE001 - every VLM failure has a deterministic fallback
        log.warning(
            "%s %s page=%s failed: %s: %s", request.source_name, request.task, request.page, type(exc).__name__, exc
        )
        return None, None, exc


def convert_to_pdf(path: Path, out_dir: Path, settings: Any) -> Path:
    """DOCX -> PDF with headless LibreOffice (honours ``Settings.soffice``)."""
    return convert(
        path, "pdf", out_dir,
        timeout_s=CONVERT_TIMEOUT_S,
        candidates=soffice_candidates(getattr(settings, "soffice", None)),
    )


def _render_pages(path: Path, numbers: list[int], settings: Any, run: _Run) -> dict[int, bytes]:
    """Convert once and render the requested 1-based pages; ``{}`` (plus a fallback note) on failure."""
    from ...render.pdf import PdfPages  # pypdfium2 is only needed when something must be rendered

    images: dict[int, bytes] = {}
    with tempfile.TemporaryDirectory(prefix="vlmrag_docx_") as tmp:
        try:
            pdf_path = convert_to_pdf(path, Path(tmp), settings)
            with PdfPages(pdf_path) as pdf:
                run.pdf_pages = len(pdf)
                for number in numbers:
                    if 1 <= number <= len(pdf):
                        images[number] = to_png(pdf.render(number - 1, RENDER_DPI))
        except Exception as exc:  # noqa: BLE001 - conversion/rendering problems only cost enrichment
            log.warning("%s: PDF conversion/render failed, continuing without visuals: %s", path.name, exc)
            run.doc_fallbacks.append("pdf_unavailable")
            return {}
    return images


def _vlm_phase(
    path: Path,
    split: DocxSplit,
    client: Any,
    settings: Any,
    doc_id: str,
    run: _Run,
) -> tuple[dict[int, list[dict]], dict[str, TableAssignment]]:
    image_pages = [p.number for p in split.pages if p.has_images]
    visuals: dict[int, list[dict]] = {}
    assignments: dict[str, TableAssignment] = {}
    if not image_pages and not split.tables:
        return visuals, assignments

    needed = sorted(set(image_pages) | {n for table in split.tables for n in table.pages})
    images = _render_pages(path, needed, settings, run)
    if not images and "pdf_unavailable" in run.doc_fallbacks:
        for table in split.tables:
            assignments[table.table_id] = TableAssignment.fallback(table, "pdf_unavailable")
        return visuals, assignments
    if run.pdf_pages is not None and run.pdf_pages != len(split.pages):
        # Rendered page N is assumed to be source page N; say so when the counts disagree.
        log.warning(
            "%s: LibreOffice rendered %d page(s), the XML split has %d; renders may be offset",
            path.name, run.pdf_pages, len(split.pages),
        )
        run.doc_fallbacks.append("pdf_page_count_mismatch")

    page_count = len(split.pages)
    jobs: list[tuple[str, Any, VLMRequest]] = []
    for number in image_pages:
        if number in images:
            request = build_image_request(
                images[number], number, page_count, doc_id=doc_id, source_name=path.name,
                reasoning_effort=_effort(settings, IMAGE_REASONING_EFFORT),
            )
            jobs.append(("image", number, request))
        else:
            run.page_fallbacks[number].append("image_page_not_rendered")
    for table in split.tables:
        page_images = [(n, images[n]) for n in table.pages if n in images]
        if not page_images:
            assignments[table.table_id] = TableAssignment.fallback(table, "table_pages_not_rendered")
            continue
        request = build_table_request(
            table, page_images, page_count, doc_id=doc_id, source_name=path.name,
            reasoning_effort=_effort(settings, TABLE_REASONING_EFFORT),
        )
        jobs.append(("table", table.table_id, request))

    results: dict[tuple[str, Any], tuple[Any, Optional[VLMResponse], Optional[BaseException]]] = {}
    if jobs:
        with ThreadPoolExecutor(max_workers=min(PARALLELISM, len(jobs))) as pool:
            futures = {pool.submit(_call_json, client, request): (kind, key) for kind, key, request in jobs}
            for future in as_completed(futures):
                results[futures[future]] = future.result()

    tables_by_id = {table.table_id: table for table in split.tables}
    for kind, key, request in jobs:  # deterministic order, whatever the completion order was
        value, response, error = results[(kind, key)]
        if kind == "image":
            if error is not None:
                status = "failed"
            else:
                answer = parse_image_answer(value)
                status = answer.status
                if answer.descriptions:
                    visuals[key] = answer.descriptions
            if status in ("failed", "unparseable", "invalid"):
                run.page_fallbacks[key].append(f"image_description_{status}")
            run.call([key], request, response, status, error)
            continue

        table = tables_by_id[key]
        if error is not None:
            status, answer = "failed", None
        else:
            answer, reason = validate_table_answer(value, table)
            status = "ok" if answer is not None else ("unparseable" if reason == "unparseable" else f"invalid:{reason}")
        if answer is None:
            assignment = TableAssignment.fallback(table, f"table_distribution_{status}")
        else:
            assignment = TableAssignment(
                table.table_id, answer.row_pages, "vlm",
                title=answer.title, title_source=answer.title_source, title_confidence=answer.title_confidence,
            )
            if answer.title_rejected:
                for number in table.pages:
                    run.page_fallbacks[number].append(f"table_title_rejected:{table.table_id}")
        assignments[table.table_id] = assignment
        run.call(table.pages, request, response, status, error)
    return visuals, assignments


# --------------------------------------------------------------------------- chunks


def _chunk_for_page(
    page,
    visuals: Optional[list[dict]],
    *,
    path: Path,
    doc_id: str,
    chunk_index: int,
    split: DocxSplit,
    assignments: dict[str, TableAssignment],
    run: _Run,
    client: Any,
) -> Optional[Chunk]:
    markdown, text = render_page(page, visuals)
    if not markdown and not text:
        return None
    header = f"{path.stem} - page {page.number}"
    retrieve = f"{header}\n{text}"
    page_tables = []
    page_fallbacks = list(run.page_fallbacks.get(page.number, []))
    if page.placeholder:
        page_fallbacks.append("app_xml_tail_page")
    for block in page.blocks:
        if block.kind == "table" and block.table_id:
            assignment = assignments[block.table_id]
            page_tables.append({
                "table_id": block.table_id,
                "rows": list(block.row_indexes),
                "assignment": assignment.source,
                "continued_from": block.continued_from,
                "title": assignment.title,
            })
            if assignment.source != "vlm":
                page_fallbacks.append(f"table_rows_from_xml_markers:{block.table_id}:{assignment.reason}")
    metadata = {
        "page_title": page.page_title,
        "section_title": page.section_title,
        "table_titles": list(page.table_titles),
        "table_header_hint": page.table_header_hint,
        "has_images": bool(page.has_images and not page.placeholder),
        "image_count": page.image_count,
        "visual_descriptions": list(visuals or []),
        "is_table_heavy": is_table_heavy(page),
        "placeholder_page": page.placeholder,
        "page_split": split.mode,
        "multi_page_tables": page_tables,
    }
    provenance = {
        "technique": DOC_TYPE,
        "model": client.model,
        "backend": str(getattr(getattr(client, "backend", None), "name", "")),
        "prompt_versions": dict(run.prompt_versions.get(page.number, {})),
        "render": {"dpi": RENDER_DPI, "converter": "libreoffice", "pdf_pages": run.pdf_pages},
        "page_split": split.info(),
        "fallbacks": list(run.doc_fallbacks) + page_fallbacks,
        "calls": list(run.page_calls.get(page.number, [])),
    }
    return Chunk(
        doc_id=doc_id,
        source_name=path.name,
        doc_type=DOC_TYPE,
        page=page.number,
        chunk_index=chunk_index,
        retrieve_content=retrieve,
        llm_content=f"# {header}\n\n{markdown}",
        filtering_content=filtering_text(retrieve),
        metadata=metadata,
        provenance=provenance,
    )


def extract(path: Union[str, Path], client: Any, settings: Any) -> list[Chunk]:
    """Extract one chunk per non-empty source page of a DOCX specification.

    Raises :class:`DocxExtractionError` when the file is not readable
    WordprocessingML or yields no content at all; every model or LibreOffice
    problem is handled by a documented fallback instead (see provenance).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    doc_id = doc_id_for(path)
    split = split_docx(path)
    run = _Run()
    if split.mode != "rendered":
        run.doc_fallbacks.append(f"no_rendered_page_breaks:{split.mode}")
    if split.padded_pages:
        run.doc_fallbacks.append("app_xml_tail_padding")
    elif split.app_pages and len(split.pages) > split.app_pages:
        run.doc_fallbacks.append("page_count_exceeds_app_xml")
    log.info(
        "%s: %d page(s) (%s), %d with pictures, %d multi-page table(s)",
        path.name, len(split.pages), split.mode, sum(1 for p in split.pages if p.has_images), len(split.tables),
    )

    visuals, assignments = _vlm_phase(path, split, client, settings, doc_id, run)
    used = apply_table_assignments(split.pages, split.tables, assignments)

    chunks: list[Chunk] = []
    for page in split.pages:
        chunk = _chunk_for_page(
            page, visuals.get(page.number), path=path, doc_id=doc_id, chunk_index=len(chunks),
            split=split, assignments=used, run=run, client=client,
        )
        if chunk is not None:
            chunks.append(chunk)
    if not chunks:
        raise DocxExtractionError(f"{path.name}: no extractable content (no text and no described visuals)")
    return chunks
