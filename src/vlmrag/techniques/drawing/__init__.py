"""Engineering-drawing title-block extraction.

Pipeline per sheet (see ``docs/techniques/drawing.md``):

multi-zoom overlapping page tiles + lower-right ROI crops, each preceded by a
position label, with text-layer hints from the same regions
-> vision pass and table pass (concurrent)
-> text-only normalization pass
-> field merge (normalized > table > raw vision)
-> acceptance gate (``min_pairs``, default 3)
-> one chunk per source page.

Entry point: :func:`extract` (registered as ``vlmrag.techniques.drawing:extract``).
"""

from __future__ import annotations

from .config import DomainTerms, DrawingConfig, FieldSpec, TitleBlockSchema
from .pipeline import extract
from .prompts import PROMPT_VERSIONS
from .title_block import DrawingExtractionError, TitleBlockGateError, TitleBlockResult, read_title_block

__all__ = [
    "DomainTerms",
    "DrawingConfig",
    "DrawingExtractionError",
    "FieldSpec",
    "PROMPT_VERSIONS",
    "TitleBlockGateError",
    "TitleBlockResult",
    "TitleBlockSchema",
    "extract",
    "read_title_block",
]
