"""``slides``: software-manual decks (``.pptx``) -> one grounded chunk per slide.

The text of a manual slide is taken from the PPTX XML, where it is exact; the
model is used only for what the XML cannot say - what the captured screens
show. Each slide with pictures is rendered whole (180 dpi) and sent once,
together with the list of layout blocks (block ID, type, normalised box, text)
so the model can describe each screenshot *by block ID* without OCR, assign it
a role (``main_screenshot``, ``detail_crop``, ``inline_image``, ``ui_icon``)
and give a detailed description plus short English and Korean summaries.
Annotation shapes (callouts, arrows, highlight boxes) are context, never
objects of their own.

See ``docs/techniques/slides.md`` for the design rationale.

Modules: :mod:`.layout` (XML -> blocks), :mod:`.prompt` (request),
:mod:`.parse` (answer validation), :mod:`.chunks` (chunk text),
:mod:`.assets` (crop-aware image export), :mod:`.pipeline` (orchestration).
"""

from __future__ import annotations

from .pipeline import (
    CONVERT_TIMEOUT_S,
    PARSE_RETRIES,
    RENDER_DPI,
    VLM_MAX_OUTPUT_TOKENS,
    VLM_TIMEOUT_S,
    extract,
)
from .layout import SlidesExtractionError
from .prompt import IMAGE_ROLES, PROMPT_VERSION, TASK

__all__ = [
    "extract",
    "SlidesExtractionError",
    "RENDER_DPI",
    "VLM_TIMEOUT_S",
    "VLM_MAX_OUTPUT_TOKENS",
    "PARSE_RETRIES",
    "CONVERT_TIMEOUT_S",
    "IMAGE_ROLES",
    "PROMPT_VERSION",
    "TASK",
]
