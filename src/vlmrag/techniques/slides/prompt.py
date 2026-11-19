"""The single model call of the slides technique: prompt, schema and request.

One request per slide carries:

1. the instructions (the visual-grounding prompt below);
2. ``Slide context: slide N of M.``;
3. the slide's text and tables, read from the PPTX XML (capped at 8,000
   characters), labelled as the source of truth for any textual value;
4. the compact layout blocks - every block's ``block_order``, type, shape name
   and normalised ``bbox_norm`` (plus up to 600 characters of text for text
   and table blocks) - and, separately, the image blocks that need a
   description;
5. the whole-slide render, preceded by its position label.

Why the model gets block IDs and boxes instead of crops
    A manual slide is a composition: a screenshot, a callout pointing at one
    of its buttons, an enlarged detail of another region. Only the whole slide
    shows those relations, so the model sees the whole render and refers to
    pictures by ``block_order``; the normalised boxes tell it which region of
    the render each ID is. Its answers can then be attached to the right image
    block (and its asset) without any coordinate matching on our side.

Why "no OCR"
    Every textual value is already available, exactly, from the XML. Asking the
    model to transcribe the screenshot would add tokens and invite misread
    digits; it is asked instead for what the XML cannot provide - what the
    captured screen *is*, what it is for, and short UI cues (screen title,
    button, field, status) that a user might search for.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from ...render.images import format_bbox
from ...vlm.types import ImageInput, VLMRequest
from .layout import Block, SlideLayout

__all__ = [
    "TASK",
    "PROMPT_VERSION",
    "SLIDE_VISUAL_PROMPT",
    "VISUAL_SCHEMA",
    "IMAGE_ROLES",
    "CONFIDENCE_LEVELS",
    "SLIDE_TEXT_LIMIT",
    "BLOCK_TEXT_LIMIT",
    "IMAGE_BLOCKS_HEADER",
    "LAYOUT_BLOCKS_HEADER",
    "SLIDE_TEXT_HEADER",
    "compact_block",
    "compact_blocks",
    "render_label",
    "build_request",
]

TASK = "slides.visual"
PROMPT_VERSION = "slides.visual/1"

SLIDE_TEXT_LIMIT = 8000
BLOCK_TEXT_LIMIT = 600

IMAGE_ROLES = ("main_screenshot", "detail_crop", "inline_image", "ui_icon")
CONFIDENCE_LEVELS = ("low", "medium", "high")

SLIDE_TEXT_HEADER = "Loaded slide text and table content (trust this over image OCR):"
LAYOUT_BLOCKS_HEADER = "Compact layout blocks:"
IMAGE_BLOCKS_HEADER = "Image blocks that need object-level descriptions:"

_SCHEMA_EXAMPLE = (
    '{"visual_summary":"","image_descriptions":[{"block_order":0,"related_block_orders":[],'
    '"image_role":"main_screenshot","description":"","description_en":"","description_ko":"",'
    '"confidence":"low"}],"flow_notes":[],"notes":""}'
)

SLIDE_VISUAL_PROMPT = (
    "You are grounding visual descriptions for one slide of a software user manual. "
    "You will receive the full slide image plus extracted slide text and compact layout blocks from the source "
    "presentation file. "
    "Treat the provided extracted text and layout as the primary source of truth for textual values such as "
    "titles, menu paths, screen names, labels, and table content. "
    "Do NOT perform full OCR, do NOT rewrite long text from the image, and do NOT invent missing text. "
    "Use the slide image only to identify original captured screen image objects and to describe their visible "
    "purpose at a high level. "
    "Each layout block has a block_order and a bbox_norm (x, y, w, h as fractions of the slide width and height, "
    "origin at the top-left corner of the slide image) that locates it on the slide image. "
    "For each meaningful original screenshot or captured UI image block, add one image_descriptions item with "
    "that exact block_order. "
    "Set image_role to main_screenshot for a primary/full application screen, detail_crop for an "
    "enlarged/cropped part of another screen, inline_image for a supporting explanation-area capture that should "
    "be shown near the relevant answer text, and ui_icon for a small meaningful UI/app icon. "
    "Multiple image blocks on one slide may all be main_screenshot when they are independent process screens or "
    "separate steps. "
    "Do not label an image as detail_crop only because it is smaller, cropped in the presentation, or shares the "
    "same source_image_id; use detail_crop only when the slide visually uses it as a partial/enlarged view of "
    "another visible screenshot. "
    "When uncertain between main_screenshot and detail_crop, choose main_screenshot and explain the visible "
    "purpose in description. "
    "Do not create image_descriptions for standalone number markers, arrows, callout icons, red boxes, highlight "
    "rectangles, decorative logos/backgrounds, or other annotation-only objects. "
    "Do not mention annotation-only presentation objects such as red boxes, highlight rectangles, arrows, "
    "numbered markers, or callout labels in description, description_en, or description_ko; describe the "
    "underlying captured application screen instead. "
    "If an image block contains UI text, do not transcribe long text; describe the captured screen object, "
    "purpose, action, or relevant screen area using the extracted slide text as grounding context. "
    "When an application screenshot clearly shows short UI cues such as screen title, screen code, menu, button, "
    "tab, field, status, or action label, include those cues in description and only the most useful cues in "
    "description_en and description_ko. "
    "Do not leave image_descriptions empty when original screenshot/captured UI image blocks are present. "
    "related_block_orders may include connected text/table/shape/image block orders when the relationship is "
    "visually supported, but only the original captured image block itself should be described. "
    "If something is uncertain, keep both descriptions conservative and lower confidence. "
    "Return exactly one JSON object and nothing else. "
    f"Use this schema exactly: {_SCHEMA_EXAMPLE}. "
    "description is the detailed LLM/retrieval description; make it rich and useful for answering questions "
    "about the captured application screen without inventing unreadable text. "
    "description_en is a short English summary for display in a web UI: one sentence for a simple image, at most "
    "two sentences for a complex image. "
    "description_ko is the Korean web UI display summary with the same concise meaning, using natural Korean "
    "wording and accurate technical terms when needed. "
    "visual_summary should be a short grounded summary that combines the provided extracted content with what is "
    "visually shown. "
    "flow_notes should describe visual flow such as arrows, numbered progression, or before/after movement only "
    "when visually supported. "
    "Do not add explanations or markdown."
)

# Strict-mode schema (every property required, no additional properties). Servers
# that ignore it still get the inline schema above, and the parser never trusts it.
VISUAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "visual_summary": {"type": "string"},
        "image_descriptions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "block_order": {"type": "integer"},
                    "related_block_orders": {"type": "array", "items": {"type": "integer"}},
                    "image_role": {"type": "string", "enum": list(IMAGE_ROLES)},
                    "description": {"type": "string"},
                    "description_en": {"type": "string"},
                    "description_ko": {"type": "string"},
                    "confidence": {"type": "string", "enum": list(CONFIDENCE_LEVELS)},
                },
                "required": [
                    "block_order",
                    "related_block_orders",
                    "image_role",
                    "description",
                    "description_en",
                    "description_ko",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        },
        "flow_notes": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
    "required": ["visual_summary", "image_descriptions", "flow_notes", "notes"],
    "additionalProperties": False,
}


def compact_block(block: Block) -> dict:
    """The model's view of one block: ID, type, name, box and (truncated) text.

    Image blocks also carry ``source_image_id`` and the crop, which is how the
    model can tell that a small picture is a cropped copy of a larger one.
    No file paths are sent: they carry no visual information and would make
    cache and replay keys machine-specific.
    """
    compact: dict[str, Any] = {
        "block_order": int(block.order),
        "type": block.type,
        "shape_name": block.shape_name,
        "bbox_norm": dict(block.bbox_norm or {}),
    }
    if block.shape_type_name:
        compact["shape_type_name"] = block.shape_type_name
    if block.auto_shape_type:
        compact["auto_shape_type"] = block.auto_shape_type
    if block.type in ("text", "table") and block.text_content.strip():
        compact["text_content"] = block.text_content.strip()[:BLOCK_TEXT_LIMIT]
    if block.type == "image" and block.picture is not None:
        compact["source_image_id"] = block.picture.source_image_id
        if block.picture.cropped:
            left, top, right, bottom = block.picture.crop
            compact["crop_norm"] = {"l": round(left, 6), "t": round(top, 6), "r": round(right, 6), "b": round(bottom, 6)}
        compact["crop_area_norm"] = round(block.picture.crop_area_norm, 6)
    return compact


def compact_blocks(blocks: list[Block]) -> list[dict]:
    return [compact_block(b) for b in blocks]


def render_label(slide_number: int, slide_count: int, dpi: float) -> str:
    """Position label of the whole-slide render: the frame every ``bbox_norm`` refers to."""
    return f"SLIDE {slide_number}/{slide_count} whole-slide render {format_bbox((0.0, 0.0, 1.0, 1.0))} dpi={dpi:g}"


def build_request(
    slide: SlideLayout,
    slide_count: int,
    png: bytes,
    *,
    dpi: float,
    doc_id: str = "",
    source_name: str = "",
    reasoning_effort: Optional[str] = None,
    timeout_s: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
) -> VLMRequest:
    """Assemble the one-slide request (text parts first, the render last)."""
    parts: list[Any] = [SLIDE_VISUAL_PROMPT, f"Slide context: slide {slide.number} of {slide_count}."]
    slide_text = slide.slide_text
    if slide_text:
        parts.append(f"{SLIDE_TEXT_HEADER}\n{slide_text[:SLIDE_TEXT_LIMIT]}")
    blocks = compact_blocks(slide.blocks)
    if blocks:
        parts.append(f"{LAYOUT_BLOCKS_HEADER}\n" + json.dumps(blocks, ensure_ascii=False))
        image_blocks = [b for b in blocks if b.get("type") == "image"]
        if image_blocks:
            parts.append(f"{IMAGE_BLOCKS_HEADER}\n" + json.dumps(image_blocks, ensure_ascii=False))
    parts.append(ImageInput(png=png, label=render_label(slide.number, slide_count, dpi)))
    return VLMRequest(
        task=TASK,
        prompt_version=PROMPT_VERSION,
        parts=parts,
        json_schema=VISUAL_SCHEMA,
        schema_name="slide_visual",
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        timeout_s=timeout_s,
        doc_id=doc_id,
        source_name=source_name,
        page=slide.number,
    )
