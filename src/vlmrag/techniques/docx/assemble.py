"""Turn split pages plus model answers into per-page Markdown and retrieval text.

Everything here is deterministic and model-free, which is what makes the
fallbacks trustworthy: given the same split and the same (or no) answers, the
output is byte-identical.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Optional

from ...text import markdown_table, plain_table
from .answers import fallback_distribution
from .xmlsplit import Block, MultiPageTable, Page

__all__ = [
    "TableAssignment",
    "VISUAL_HEADING_MD",
    "VISUAL_HEADING_TEXT",
    "apply_table_assignments",
    "is_table_heavy",
    "render_page",
]

log = logging.getLogger(__name__)

VISUAL_HEADING_MD = "### Visual content (image description)"
VISUAL_HEADING_TEXT = "[Visual content]"
TITLE_HINT_LIMIT = 200


@dataclass
class TableAssignment:
    """How the rows of one multi-page table were placed on pages.

    ``source`` is ``"vlm"`` for a validated model answer or
    ``"xml_row_markers"`` for the deterministic fallback; ``reason`` says why
    the fallback was used.
    """

    table_id: str
    row_pages: dict[int, list[int]]
    source: str
    title: str = ""
    title_source: str = "none"
    title_confidence: str = "low"
    reason: str = ""

    @classmethod
    def fallback(cls, table: MultiPageTable, reason: str) -> "TableAssignment":
        return cls(table.table_id, fallback_distribution(table), "xml_row_markers", reason=reason)


def _continuation_note(first_page: int, title: str, markdown: bool) -> str:
    suffix = f": {title}" if title else ""
    note = f"(Table continued from page {first_page}{suffix})"
    return f"_{note}_" if markdown else note


def apply_table_assignments(
    pages: list[Page],
    tables: list[MultiPageTable],
    assignments: Mapping[str, TableAssignment],
) -> dict[str, TableAssignment]:
    """Place each multi-page table's rows on its pages, in reading order.

    * The slice for the table's end page replaces the ``table_slot`` block in
      place, so text that follows the table on that page stays after it.
    * Slices for earlier pages are appended to those pages: the table began
      after everything else on its first page, and the pages in between hold
      nothing but table rows.
    * A slice that does not start with row 0 is a continuation: it gets a note
      naming the page where the table starts (and its validated title), and
      it is rendered without promoting its first data row to a Markdown header.
    * A validated title (confidence above ``"low"``) is added to
      ``table_titles`` of every page holding rows of the table and fills an
      empty ``table_header_hint``.

    Tables without an assignment use the row-marker fallback. Returns the
    assignment actually used per table.
    """
    by_number = {page.number: page for page in pages}
    used: dict[str, TableAssignment] = {}
    for table in tables:
        assignment = assignments.get(table.table_id) or TableAssignment.fallback(table, "not_attempted")
        used[table.table_id] = assignment
        title = assignment.title if assignment.title and assignment.title_confidence != "low" else ""
        pages_with_rows = [p for p in table.pages if assignment.row_pages.get(p)]
        first_page = pages_with_rows[0] if pages_with_rows else table.start_page
        for number in table.pages:
            page = by_number.get(number)
            if page is None:
                continue
            indexes = [i for i in assignment.row_pages.get(number) or [] if 0 <= i < len(table.rows)]
            block: Optional[Block] = None
            if indexes:
                rows = [table.rows[i] for i in indexes]
                starts_table = indexes[0] == 0
                markdown = markdown_table(rows, header=starts_table)
                text = plain_table(rows)
                continued_from = None
                if not starts_table:
                    continued_from = first_page
                    markdown = _continuation_note(first_page, title, True) + "\n\n" + markdown
                    text = _continuation_note(first_page, title, False) + "\n" + text
                block = Block(
                    "table", markdown, text, table_id=table.table_id,
                    continued_from=continued_from, row_indexes=list(indexes),
                )
                if title:
                    if not page.table_header_hint:
                        page.table_header_hint = title[:TITLE_HINT_LIMIT]
                    if title not in page.table_titles:
                        page.table_titles.append(title)
            slot = next(
                (i for i, b in enumerate(page.blocks) if b.kind == "table_slot" and b.table_id == table.table_id),
                None,
            )
            if slot is not None:
                if block is None:
                    del page.blocks[slot]
                else:
                    page.blocks[slot] = block
            elif block is not None:
                page.blocks.append(block)
        log.info(
            "table %s pages %d-%d rows=%d placed via %s%s",
            table.table_id, table.start_page, table.end_page, len(table.rows), assignment.source,
            f" ({assignment.reason})" if assignment.reason else "",
        )
    return used


def render_page(page: Page, visuals: Optional[list[dict]] = None) -> tuple[str, str]:
    """``(markdown, text)`` for one page: XML blocks in order, then the visual descriptions."""
    markdown_parts = [b.markdown for b in page.blocks if b.kind != "table_slot" and b.markdown.strip()]
    text_parts = [b.text for b in page.blocks if b.kind != "table_slot" and b.text.strip()]
    if visuals:
        md_lines = [VISUAL_HEADING_MD]
        text_lines = [VISUAL_HEADING_TEXT]
        for item in visuals:
            location = item.get("location_hint") or ""
            description = item.get("description") or ""
            if location:
                md_lines.append(f"- **{location}**: {description}")
                text_lines.append(f"- {location}: {description}")
            else:
                md_lines.append(f"- {description}")
                text_lines.append(f"- {description}")
        markdown_parts.append("\n".join(md_lines))
        text_parts.append("\n".join(text_lines))
    return "\n\n".join(markdown_parts).strip(), "\n\n".join(text_parts).strip()


def is_table_heavy(page: Page) -> bool:
    """At least half of the page's blocks are tables (useful to switch summaries off)."""
    blocks = [b for b in page.blocks if b.kind != "table_slot"]
    tables = sum(1 for b in blocks if b.kind == "table")
    return bool(blocks) and tables * 2 >= len(blocks)
