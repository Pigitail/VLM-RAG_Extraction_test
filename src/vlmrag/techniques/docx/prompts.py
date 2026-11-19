"""Prompts, prompt versions and strict JSON schemas for the docx technique.

Both prompts share one idea: the XML already gave us the text, so the model is
only asked for what the XML cannot know.

* The image prompt suppresses OCR of document text. Paragraphs and tables are
  already exact; letting the model transcribe them again would only add a
  second, noisier copy that competes with the real one at retrieval time. Text
  that exists *only inside a picture* (diagram labels, callouts) is still
  wanted, because the XML does not contain it.
* The table prompt asks for row *indexes* per page and a title choice, never
  for cell text. An index-only answer can be checked exactly (every row once,
  pages in range), so a bad answer is detected and replaced by a deterministic
  fallback instead of silently corrupting the table.

Bump the matching ``*_PROMPT_VERSION`` whenever a prompt changes: the version
is part of the cache key and of the record/replay key.
"""

from __future__ import annotations

from typing import Iterable

__all__ = [
    "IMAGE_TASK",
    "IMAGE_PROMPT_VERSION",
    "IMAGE_PROMPT",
    "IMAGE_SCHEMA",
    "IMAGE_SCHEMA_NAME",
    "TABLE_TASK",
    "TABLE_PROMPT_VERSION",
    "TABLE_PROMPT",
    "TABLE_SCHEMA_NAME",
    "TITLE_SOURCES",
    "TITLE_CONFIDENCES",
    "table_schema",
]

IMAGE_TASK = "docx.image_description"
IMAGE_PROMPT_VERSION = "docx.image_description/1"
IMAGE_SCHEMA_NAME = "docx_image_descriptions"

TABLE_TASK = "docx.table_distribution"
TABLE_PROMPT_VERSION = "docx.table_distribution/1"
TABLE_SCHEMA_NAME = "docx_table_distribution"

TITLE_SOURCES = ("preceding_paragraph", "first_row", "none")
TITLE_CONFIDENCES = ("high", "medium", "low")

IMAGE_PROMPT = (
    "Your job is to describe the non-text visual content of a single page taken from a specification document. "
    "All running text of the page (headings, paragraphs and tables) was already extracted from the document's XML "
    "and is exact, so do NOT transcribe or re-OCR it. "
    "Look only at pictures: diagrams, photographs, flowcharts, schematic or technical drawings, charts, icons and "
    "previews of embedded objects. "
    "For every visual, write a description that would help someone find this page by searching: what the subject "
    "is, the labels and callouts drawn inside the visual, the direction of arrows and flows, colours, and other "
    "distinctive visual features. "
    "Words that appear only inside a picture (for example a label on a diagram) are part of the visual and may be "
    "quoted; words that belong to the page's normal paragraphs or tables must not be copied. "
    "Answer with exactly one JSON object and nothing else, following this schema: "
    '{"image_descriptions":[{"location_hint":"","description":""}],"notes":""}. '
    "location_hint is a short positional cue such as 'upper-left photo', 'centre diagram' or 'right-hand flowchart'. "
    "Use notes only for a page-level remark that does not belong to a single visual; otherwise leave it empty. "
    'When the page contains no non-text visual content, answer {"image_descriptions":[],"notes":""}. '
    "No explanations, no Markdown."
)

TABLE_PROMPT = (
    "You are given one table from a word-processing document that continues over several rendered pages. "
    "The cell values of every row were already extracted from the document's XML; they are ground truth. "
    "You MUST NOT read, re-OCR, correct or rewrite any cell text. "
    "You have exactly two jobs:\n"
    "(A) Decide whether the table has a distinct TITLE and, if it does, which of the provided texts is that title.\n"
    "(B) Assign every table ROW INDEX to the rendered page on which that row appears.\n"
    "\n"
    "TITLE RULES:\n"
    "- A title can only be one of the provided `preceding_paragraphs`, or the merged first row of the table "
    "(rows[0]).\n"
    "- Never invent a title and never read a title from the page images.\n"
    '- `title_source` is one of "preceding_paragraph", "first_row" or "none".\n'
    "- `title_confidence`:\n"
    '    "high"   = clearly the table\'s title (for example a caption such as \'Table 2: Fastener torque\' or '
    "'<Table 3>', or a bold centred line naming the table's subject).\n"
    '    "medium" = plausible, but it might also be a section heading that happens to sit above the table.\n'
    '    "low"    = unsure. Prefer `has_title=false` in that case.\n'
    '- When `has_title` is false, `title_text` must be "" and `title_source` must be "none".\n'
    "\n"
    "ROW ASSIGNMENT RULES:\n"
    "- The keys of `row_distribution` are page numbers written as strings, from `start_page` to `end_page` "
    "inclusive.\n"
    "- Every row index from 0 to N-1 must appear EXACTLY ONCE over all pages together: no duplicates, no gaps.\n"
    "- A page that shows none of the rows must still be present with an empty list [].\n"
    "- Decide from the page images: count how many table rows are drawn on each page, then map them in order onto "
    "the XML rows (XML rows are ordered top to bottom, the same order as on the pages).\n"
    "- Use the visual position of rows only; do not use the cell text to decide.\n"
    "\n"
    "OUTPUT:\n"
    "Return exactly one JSON object and nothing else: no prose, no Markdown fences, no comments.\n"
    "Schema (keys and types must match exactly):\n"
    '{"has_title": false, "title_text": "", "title_source": "none", "title_confidence": "low", '
    '"row_distribution": {}}'
)

IMAGE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "image_descriptions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "location_hint": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["location_hint", "description"],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["image_descriptions", "notes"],
    "additionalProperties": False,
}

#: Strict structured output caps the number of object properties in one schema (the cap was
#: 100 when the feature launched and differs between servers); 90 page keys plus the five
#: top-level fields stay under the strictest cap. Wider spans are sent without a schema - the
#: validator checks the answer either way.
MAX_SCHEMA_PAGES = 90


def table_schema(pages: Iterable[int]) -> dict | None:
    """Strict JSON schema for one table's answer.

    Strict structured output forbids free-form object keys, so the page keys
    of ``row_distribution`` are spelled out for this table's span. That also
    enforces the "every page present, even when empty" rule at decode time.
    Returns ``None`` for spans too wide for a strict schema.
    """
    keys = [str(int(p)) for p in pages]
    if not keys or len(keys) > MAX_SCHEMA_PAGES:
        return None
    return {
        "type": "object",
        "properties": {
            "has_title": {"type": "boolean"},
            "title_text": {"type": "string"},
            "title_source": {"type": "string", "enum": list(TITLE_SOURCES)},
            "title_confidence": {"type": "string", "enum": list(TITLE_CONFIDENCES)},
            "row_distribution": {
                "type": "object",
                "properties": {key: {"type": "array", "items": {"type": "integer"}} for key in keys},
                "required": keys,
                "additionalProperties": False,
            },
        },
        "required": ["has_title", "title_text", "title_source", "title_confidence", "row_distribution"],
        "additionalProperties": False,
    }
