"""docx technique: specification documents (Word ``.docx``) -> one chunk per source page.

* :mod:`.xmlsplit` - XML-first page split (rendered page-break markers,
  cross-checked with ``docProps/app.xml`` ``<Pages>``), tables from the XML,
  picture flags per page, multi-page table events.
* :mod:`.prompts` - the two prompts (image description with OCR suppressed,
  table row -> page assignment) and their strict JSON schemas.
* :mod:`.answers` - parsing and strict validation of model answers, and the
  deterministic row-assignment fallback.
* :mod:`.assemble` - placing table rows on pages and rendering page text.
* :mod:`.pipeline` - orchestration (LibreOffice render, concurrent VLM calls,
  provenance, chunks).

See ``docs/techniques/docx.md`` for the design rationale.
"""

from __future__ import annotations

from .pipeline import (
    CONVERT_TIMEOUT_S,
    DOC_TYPE,
    MAX_OUTPUT_TOKENS,
    PARALLELISM,
    RENDER_DPI,
    VLM_TIMEOUT_S,
    extract,
)
from .xmlsplit import DocxExtractionError, DocxSplit, MultiPageTable, Page, split_docx, split_document

__all__ = [
    "CONVERT_TIMEOUT_S",
    "DOC_TYPE",
    "DocxExtractionError",
    "DocxSplit",
    "MAX_OUTPUT_TOKENS",
    "MultiPageTable",
    "PARALLELISM",
    "Page",
    "RENDER_DPI",
    "VLM_TIMEOUT_S",
    "extract",
    "split_docx",
    "split_document",
]
