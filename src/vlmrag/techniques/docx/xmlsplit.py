"""XML-first page split for DOCX specification documents.

Why the XML and not a rendered PDF
    A DOCX has no fixed pages: pagination is computed by whatever program
    opens it. Converting with LibreOffice and reading the PDF gives
    LibreOffice's pagination, which drifts from the author's word processor
    (different fonts, kerning, widow/orphan rules), and reading table cells
    back from pixels introduces OCR errors in exactly the values people search
    for. The XML already holds the exact text, the table structure and - when
    the file was last saved by a paginating editor - the editor's own page
    boundaries:

    * ``<w:lastRenderedPageBreak/>`` marks the point where the editor's last
      layout started a new page (natural overflow *and* manual breaks);
    * ``<w:br w:type="page"/>`` is an explicit (manual) page break;
    * ``docProps/app.xml`` ``<Pages>`` is the page count the editor stored.

    So text and tables come from the XML, pages come from the rendered break
    markers, and ``<Pages>`` is the cross-check: when the split comes up short
    (typically a last page holding only a picture, where no marker is
    written), visual-only tail pages are appended so the image pass can still
    describe them and every chunk keeps a real source page number.

What this module does not do
    It never calls a model and never renders anything. Tables that cross a
    page boundary are reported as :class:`MultiPageTable` events and leave a
    ``table_slot`` block on the page where they end; the pipeline later decides
    which rows belong to which page (VLM answer or deterministic fallback) and
    fills the slot.
"""

from __future__ import annotations

import io
import logging
import posixpath
import re
import zipfile
import zlib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from lxml import etree

from ...text import markdown_table, plain_table

__all__ = [
    "DocxExtractionError",
    "DocxParts",
    "Block",
    "Page",
    "MultiPageTable",
    "DocxSplit",
    "read_docx_parts",
    "app_page_count",
    "style_names",
    "heading_level",
    "split_document",
    "split_docx",
]

log = logging.getLogger(__name__)

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"
_W = "{%s}" % W_NS
_MC_FALLBACK = "{%s}Fallback" % MC_NS

OFFICE_DOCUMENT_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
DEFAULT_MAIN_PART = "word/document.xml"

#: How many preceding paragraphs are kept as title candidates for a table.
PRECEDING_PARAGRAPHS = 5
#: Maximum length of the per-page ``table_header_hint``.
HEADER_HINT_LIMIT = 200
#: Refuse to inflate XML parts larger than this (zip-bomb guard).
MAX_PART_BYTES = 256 * 1024 * 1024
#: Signature of OLE compound files (legacy .doc, encrypted or rights-protected Office files).
_CFB_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

_PAGES_RE = re.compile(rb"<(?:[A-Za-z_][\w.-]*:)?Pages>\s*(\d+)\s*</(?:[A-Za-z_][\w.-]*:)?Pages>")
_HEADING_NUM_RE = re.compile(r"heading\s*(\d+)")
_INLINE_WS_RE = re.compile(r"[ \t]+")
_LINE_BREAK_RE = re.compile(r"[ \t]*\n[ \t\n]*")
_ANY_WS_RE = re.compile(r"\s+")

# Elements that put a picture, chart, shape or embedded-object preview on the page.
_VISUAL_TAGS = {"drawing", "pict", "object"}


class DocxExtractionError(RuntimeError):
    """The DOCX could not be read as WordprocessingML (hard failure for this file)."""


# --------------------------------------------------------------------------- data model


@dataclass
class Block:
    """One unit of page content in reading order.

    ``kind`` is ``"heading"``, ``"paragraph"``, ``"table"`` or ``"table_slot"``
    (the placeholder a multi-page table leaves on its end page). ``markdown``
    feeds ``llm_content``; ``text`` feeds ``retrieve_content``.
    """

    kind: str
    markdown: str = ""
    text: str = ""
    table_id: str = ""
    continued_from: Optional[int] = None
    row_indexes: list[int] = field(default_factory=list)


@dataclass
class Page:
    """Content of one source page as found in the XML."""

    number: int
    blocks: list[Block] = field(default_factory=list)
    page_title: str = ""
    section_title: str = ""
    table_header_hint: str = ""
    table_titles: list[str] = field(default_factory=list)
    has_images: bool = False
    image_count: int = 0
    placeholder: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.blocks and not self.has_images


@dataclass
class MultiPageTable:
    """A table whose rows cross at least one page boundary.

    ``rows`` are the non-empty rows exactly as found in the XML (ground truth,
    never re-read from pixels). ``start_page`` is the page on which the text of
    the first row was rendered. ``row_pages[i]`` is the page on which the
    editor last rendered row ``i`` (taken from rendered break markers inside
    the table); it is the deterministic fallback when the model's row
    assignment is missing or invalid. ``preceding_paragraphs`` are the last few
    paragraphs before the table: the only text a title may be taken from.
    """

    table_id: str
    start_page: int
    end_page: int
    rows: list[list[str]]
    row_pages: list[int]
    preceding_paragraphs: list[str]

    @property
    def pages(self) -> list[int]:
        return list(range(self.start_page, self.end_page + 1))


@dataclass
class DocxSplit:
    """Result of the split.

    ``rendered_breaks`` counts every ``w:lastRenderedPageBreak`` seen;
    ``merged_breaks`` counts those that did not start a page because the
    boundary was already taken (right after a manual break with nothing in
    between, or by another cell of the same table row).
    """

    pages: list[Page]
    tables: list[MultiPageTable]
    rendered_breaks: int = 0
    explicit_breaks: int = 0
    merged_breaks: int = 0
    app_pages: int = 0
    padded_pages: int = 0

    @property
    def mode(self) -> str:
        """Where the page boundaries came from.

        ``"rendered"``: the editor's rendered break markers (reliable);
        ``"explicit"``: only manual page breaks were found, so natural page
        overflows are unknown; ``"none"``: no page information at all - the
        whole body is page 1 (plus app.xml tail padding).
        """
        if self.rendered_breaks:
            return "rendered"
        if self.explicit_breaks:
            return "explicit"
        return "none"

    def info(self) -> dict:
        return {
            "mode": self.mode,
            "rendered_breaks": self.rendered_breaks,
            "explicit_breaks": self.explicit_breaks,
            "merged_breaks": self.merged_breaks,
            "app_xml_pages": self.app_pages,
            "xml_pages": len(self.pages) - self.padded_pages,
            "padded_pages": self.padded_pages,
        }


@dataclass
class DocxParts:
    document_xml: bytes
    app_xml: bytes = b""
    styles_xml: bytes = b""
    main_part: str = DEFAULT_MAIN_PART


# --------------------------------------------------------------------------- package access


def _read_member(archive: zipfile.ZipFile, name: str) -> bytes:
    info = archive.getinfo(name)
    if info.file_size > MAX_PART_BYTES:
        raise DocxExtractionError(f"{name} is too large ({info.file_size} bytes)")
    return archive.read(info)


def _main_part_name(archive: zipfile.ZipFile) -> str:
    """Resolve the main document part from ``_rels/.rels`` (it is not always ``word/document.xml``)."""
    try:
        root = etree.fromstring(_read_member(archive, "_rels/.rels"), parser=_safe_parser())
    except (KeyError, etree.XMLSyntaxError, DocxExtractionError):
        return DEFAULT_MAIN_PART
    for rel in root:
        if rel.get("Type") == OFFICE_DOCUMENT_REL and rel.get("Target"):
            target = rel.get("Target").lstrip("/")
            return posixpath.normpath(target)
    return DEFAULT_MAIN_PART


def _safe_parser() -> etree.XMLParser:
    # No entity resolution, no network: document parts are untrusted input.
    return etree.XMLParser(resolve_entities=False, no_network=True)


def _container_hint(path: Union[str, Path]) -> str:
    try:
        with open(path, "rb") as handle:
            if handle.read(len(_CFB_MAGIC)) == _CFB_MAGIC:
                return " (it is an OLE compound file: a legacy .doc, or an encrypted or rights-protected document)"
    except OSError:
        pass
    return ""


def read_docx_parts(path: Union[str, Path]) -> DocxParts:
    """Read the parts the split needs; raise :class:`DocxExtractionError` if the file is not a DOCX."""
    name = Path(path).name
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            main = _main_part_name(archive)
            if main not in names:
                raise DocxExtractionError(f"{name}: main document part {main!r} is missing")
            document_xml = _read_member(archive, main)
            app_xml = _read_member(archive, "docProps/app.xml") if "docProps/app.xml" in names else b""
            styles_name = posixpath.join(posixpath.dirname(main), "styles.xml")
            styles_xml = _read_member(archive, styles_name) if styles_name in names else b""
    except DocxExtractionError:
        raise
    except zipfile.BadZipFile as exc:
        raise DocxExtractionError(f"{name} is not a DOCX (zip) package{_container_hint(path)}: {exc}") from exc
    except (RuntimeError, NotImplementedError, EOFError, zlib.error) as exc:
        # Encrypted zip members, unsupported compression, truncated or corrupt data.
        raise DocxExtractionError(f"{name}: the package cannot be read ({type(exc).__name__}: {exc})") from exc
    return DocxParts(document_xml=document_xml, app_xml=app_xml, styles_xml=styles_xml, main_part=main)


def app_page_count(app_xml: Union[bytes, str, None]) -> int:
    """``<Pages>`` from ``docProps/app.xml``, or 0 when absent/unreadable."""
    if not app_xml:
        return 0
    data = app_xml.encode("utf-8") if isinstance(app_xml, str) else app_xml
    match = _PAGES_RE.search(data)
    return int(match.group(1)) if match else 0


def style_names(styles_xml: Union[bytes, None]) -> dict[str, str]:
    """Map style id -> style name (lower case).

    Word writes localized style *ids* (a heading style id is not "Heading1" in
    every language) but the built-in style *names* stay English
    (``"heading 1"``), so heading detection checks both.
    """
    if not styles_xml:
        return {}
    try:
        root = etree.fromstring(styles_xml, parser=_safe_parser())
    except etree.XMLSyntaxError:
        return {}
    names: dict[str, str] = {}
    for style in root.iter(_W + "style"):
        style_id = style.get(_W + "styleId")
        name = style.find(_W + "name")
        if style_id and name is not None and name.get(_W + "val"):
            names[style_id] = name.get(_W + "val").lower()
    return names


def heading_level(style_id: str, style_name: str = "") -> Optional[int]:
    """Heading level for a paragraph style, or ``None`` for body text.

    ``Title``/``Subtitle`` count as level 1, ``heading N`` as level N and a
    heading style without a number as level 2.
    """
    for candidate in (style_id, style_name):
        low = (candidate or "").strip().lower()
        if not low:
            continue
        if low in ("title", "subtitle"):
            return 1
        if low.startswith("heading"):
            match = _HEADING_NUM_RE.search(low)
            # "heading 0" is not a real level; treat it like level 1 so it never takes the "#" chunk header level.
            return max(1, int(match.group(1))) if match else 2
    return None


def _heading_markdown(text: str, level: int) -> str:
    # Page headings start at "##": "#" is reserved for the chunk header line.
    return "#" * max(1, min(6, level + 1)) + " " + text


# --------------------------------------------------------------------------- the splitter


@dataclass
class _Paragraph:
    parts: list[str] = field(default_factory=list)
    style_id: str = ""
    level: Optional[int] = None
    seen_text: bool = False
    images: int = 0
    nested: int = 0  # paragraphs nested inside this one (text boxes)


@dataclass
class _Table:
    preceding: list[str]
    rows: list[list[str]] = field(default_factory=list)
    row_pages: list[int] = field(default_factory=list)  # page at the end of each row (fallback assignment)
    row_first_pages: list[int] = field(default_factory=list)  # page of each row's first text
    depth: int = 1  # nesting of w:tbl while this top-level table is open
    row: Optional[list[str]] = None
    cell: Optional[list[str]] = None
    row_first_page: Optional[int] = None
    cell_marks: dict[int, int] = field(default_factory=dict)  # rendered markers per cell of the open row
    row_breaks: int = 0  # page boundaries already taken inside the open row


class _Splitter:
    """Streaming state machine over ``word/document.xml`` events."""

    def __init__(self, styles: dict[str, str]) -> None:
        self.styles = styles
        self.pages: list[Page] = []
        self.page = Page(number=1)
        self.tables: list[MultiPageTable] = []
        self.preceding: deque[str] = deque(maxlen=PRECEDING_PARAGRAPHS)
        self.para: Optional[_Paragraph] = None
        self.para_tables = 0  # tables nested inside the open paragraph (text boxes)
        self.table: Optional[_Table] = None
        self.skip = 0  # depth inside mc:Fallback (duplicate legacy rendering of mc:Choice)
        self.rendered_breaks = 0
        self.explicit_breaks = 0
        self.merged_breaks = 0
        self._last_boundary_explicit = False
        self._touched = True  # anything seen since the last page boundary

    # ---------------------------------------------------------------- page handling

    def _flush_page(self) -> None:
        self.pages.append(self.page)
        self.page = Page(number=len(self.pages) + 1)

    def _add_images(self, count: int) -> None:
        if count:
            self.page.has_images = True
            self.page.image_count += count

    def _row_boundary_already_taken(self) -> bool:
        """True when a rendered marker repeats a page boundary already taken in this table row.

        The cells of a row are laid out side by side, so when a row starts (or
        continues) on a new page every cell's text starts there, and each cell
        can carry its own marker. The row crosses as many boundaries as its
        most-broken cell, not the sum over its cells.
        """
        table = self.table
        if table is None or table.row is None:
            return False
        cell = len(table.row)  # index of the open top-level cell (nested tables count for it)
        seen = table.cell_marks.get(cell, 0) + 1
        table.cell_marks[cell] = seen
        if seen <= table.row_breaks:
            return True
        table.row_breaks = seen
        return False

    def _page_break(self, explicit: bool) -> None:
        if explicit:
            self.explicit_breaks += 1
        else:
            self.rendered_breaks += 1
            if self._row_boundary_already_taken():
                self.merged_breaks += 1
                return
            if self._last_boundary_explicit and not self._touched:
                # Editors also record a rendered break at the top of the page that a
                # manual break started; it is the same boundary, not a blank page.
                self.merged_breaks += 1
                return
        if self.para is not None:
            # Pictures met before the break were laid out on the current page.
            self._add_images(self.para.images)
            self.para.images = 0
            if self.para.seen_text:
                # Break in the middle of a paragraph: the text so far stays on this page.
                self._commit_paragraph_text(final=False)
            # Otherwise the paragraph simply starts on the next page.
        self._flush_page()
        self._last_boundary_explicit = explicit
        self._touched = False

    # ---------------------------------------------------------------- paragraphs

    def _commit_paragraph_text(self, final: bool) -> None:
        para = self.para
        assert para is not None
        text = _LINE_BREAK_RE.sub("\n", _INLINE_WS_RE.sub(" ", "".join(para.parts))).strip()
        para.parts = []
        para.seen_text = False
        if not text:
            return
        if para.level is not None:
            if not self.page.page_title:
                self.page.page_title = text
            elif not self.page.section_title:
                self.page.section_title = text
            self.page.blocks.append(Block("heading", _heading_markdown(text, para.level), text))
        else:
            self.page.blocks.append(Block("paragraph", text, text))
        if final:
            # Only complete paragraphs are title candidates for a following table.
            self.preceding.append(text)

    def _finish_paragraph(self) -> None:
        self._add_images(self.para.images)
        self._commit_paragraph_text(final=True)
        self.para = None
        self.para_tables = 0

    # ---------------------------------------------------------------- tables

    def _finish_table(self) -> None:
        table = self.table
        assert table is not None
        self.table = None
        kept = [
            (row, page, first)
            for row, page, first in zip(table.rows, table.row_pages, table.row_first_pages)
            if any(c.strip() for c in row)
        ]
        if not kept:
            return
        rows = [row for row, _, _ in kept]
        row_pages = [page for _, page, _ in kept]
        # The table starts where its first row's text was rendered, not where <w:tbl> opened: a
        # table that begins at the top of a page has its marker inside the first cell, after
        # <w:tbl>, and must not be mistaken for a table that crosses a page.
        start_page = kept[0][2]
        end_page = self.page.number
        if end_page != start_page:
            table_id = f"t{len(self.tables) + 1}"
            self.tables.append(
                MultiPageTable(
                    table_id=table_id,
                    start_page=start_page,
                    end_page=end_page,
                    rows=rows,
                    row_pages=row_pages,
                    preceding_paragraphs=list(table.preceding),
                )
            )
            self.page.blocks.append(Block("table_slot", table_id=table_id))
        else:
            self.page.blocks.append(
                Block("table", markdown_table(rows, header=True), plain_table(rows), row_indexes=list(range(len(rows))))
            )
            if not self.page.table_header_hint:
                self.page.table_header_hint = " | ".join(c.strip() for c in rows[0])[:HEADER_HINT_LIMIT]

    # ---------------------------------------------------------------- event dispatch

    def start(self, tag: str, elem) -> None:
        if tag == "p":
            if self.table is not None:
                return
            if self.para is None:
                self.para = _Paragraph()
            else:
                self.para.nested += 1
                self.para.parts.append("\n")
        elif tag == "pStyle":
            if self.para is not None and self.table is None and self.para.nested == 0 and not self.para_tables:
                style_id = elem.get(_W + "val") or ""
                self.para.style_id = style_id
                self.para.level = heading_level(style_id, self.styles.get(style_id, ""))
        elif tag == "tbl":
            if self.para is not None:
                self.para_tables += 1  # a table inside a text box: flattened into the paragraph
            elif self.table is not None:
                self.table.depth += 1  # nested table: flattened into the current cell
            else:
                self.table = _Table(preceding=list(self.preceding))
        elif tag == "tr":
            if self.table is not None and self.table.depth == 1:
                self.table.row = []
                self.table.row_first_page = None
                self.table.cell_marks = {}
                self.table.row_breaks = 0
        elif tag == "tc":
            if self.table is not None and self.table.depth == 1 and self.table.row is not None:
                self.table.cell = []
        elif tag in ("br", "cr"):
            if tag == "br" and (elem.get(_W + "type") or "") == "page":
                self._page_break(explicit=True)
            elif self.table is not None:
                if self.table.cell is not None:
                    self.table.cell.append(" ")
            elif self.para is not None:
                self.para.parts.append("\n")
        elif tag == "lastRenderedPageBreak":
            self._page_break(explicit=False)
        elif tag in _VISUAL_TAGS:
            self._touched = True
            if self.table is not None:
                self._add_images(1)  # pictures in cells: flag the page directly
            elif self.para is not None:
                self.para.images += 1
            else:
                self._add_images(1)
        elif tag == "tab":
            if self.table is not None:
                if self.table.cell is not None:
                    self.table.cell.append(" ")
            elif self.para is not None:
                self.para.parts.append("\t")

    def end(self, tag: str, elem) -> None:
        if tag == "t":
            text = elem.text or ""
            if not text:
                return
            self._touched = True
            if self.table is not None:
                if self.table.cell is not None:
                    self.table.cell.append(text)
                if self.table.row is not None and self.table.row_first_page is None and text.strip():
                    self.table.row_first_page = self.page.number
            elif self.para is not None:
                self.para.parts.append(text)
                self.para.seen_text = True
        elif tag == "p":
            if self.table is not None:
                if self.table.cell is not None:
                    self.table.cell.append(" ")
            elif self.para is not None:
                if self.para.nested:
                    self.para.nested -= 1
                    self.para.parts.append("\n")
                else:
                    self._finish_paragraph()
        elif tag == "tc":
            if self.para is not None and self.para_tables:
                self.para.parts.append(" ")
            elif self.table is not None and self.table.depth == 1 and self.table.cell is not None:
                if self.table.row is not None:
                    self.table.row.append(_ANY_WS_RE.sub(" ", "".join(self.table.cell)).strip())
                self.table.cell = None
        elif tag == "tr":
            if self.table is not None and self.table.depth == 1 and self.table.row is not None:
                self.table.rows.append(self.table.row)
                # The page this row was last rendered on: the fallback row assignment.
                self.table.row_pages.append(self.page.number)
                first = self.table.row_first_page
                self.table.row_first_pages.append(first if first is not None else self.page.number)
                self.table.row = None
                self._touched = True
        elif tag == "tbl":
            if self.para is not None and self.para_tables:
                self.para_tables -= 1
            elif self.table is not None:
                if self.table.depth > 1:
                    self.table.depth -= 1
                else:
                    self._finish_table()

    def finish(self) -> None:
        if self.para is not None:
            self._finish_paragraph()
        if not self.page.is_empty:
            self._flush_page()


def split_document(
    document_xml: bytes,
    *,
    app_pages: int = 0,
    styles: Optional[dict[str, str]] = None,
    source_name: str = "",
) -> DocxSplit:
    """Split ``word/document.xml`` into source pages.

    ``app_pages`` is the ``<Pages>`` value of ``docProps/app.xml`` (0 if
    unknown). When the split yields fewer pages, visual-only placeholder pages
    are appended (``placeholder=True``, ``has_images=True``) so that the image
    pass can recover their content from the rendered PDF; when it yields more,
    the extra pages are kept and the mismatch is reported in
    :meth:`DocxSplit.info`.
    """
    splitter = _Splitter(styles or {})
    try:
        for event, elem in etree.iterparse(
            io.BytesIO(document_xml), events=("start", "end"), resolve_entities=False, no_network=True
        ):
            tag = elem.tag
            if not isinstance(tag, str):
                continue
            if tag == _MC_FALLBACK:
                splitter.skip += 1 if event == "start" else -1
                if event == "end":
                    elem.clear()
                continue
            if splitter.skip:
                if event == "end":
                    elem.clear()
                continue
            local = tag.rsplit("}", 1)[-1]
            if event == "start":
                splitter.start(local, elem)
            else:
                splitter.end(local, elem)
                elem.clear()  # keep memory flat on large documents
    except etree.XMLSyntaxError as exc:
        raise DocxExtractionError(f"{source_name or 'document'}: invalid WordprocessingML ({exc})") from exc
    splitter.finish()

    pages = splitter.pages
    padded = 0
    if app_pages and len(pages) < app_pages:
        padded = app_pages - len(pages)
        log.warning(
            "%s: XML split found %d page(s) but app.xml reports %d; adding %d visual-only tail page(s)",
            source_name or "document", len(pages), app_pages, padded,
        )
        for _ in range(padded):
            pages.append(Page(number=len(pages) + 1, has_images=True, placeholder=True))
    elif app_pages and len(pages) > app_pages:
        log.warning(
            "%s: XML split found %d page(s) but app.xml reports %d; keeping the XML split",
            source_name or "document", len(pages), app_pages,
        )
    for number, page in enumerate(pages, start=1):
        page.number = number  # already sequential; kept as a cheap invariant

    split = DocxSplit(
        pages=pages,
        tables=splitter.tables,
        rendered_breaks=splitter.rendered_breaks,
        explicit_breaks=splitter.explicit_breaks,
        merged_breaks=splitter.merged_breaks,
        app_pages=app_pages,
        padded_pages=padded,
    )
    if split.mode != "rendered":
        log.warning(
            "%s: no rendered page breaks in the XML (mode=%s); text page numbers follow %s only",
            source_name or "document", split.mode,
            "manual page breaks" if split.mode == "explicit" else "the start of the document",
        )
    return split


def split_docx(path: Union[str, Path]) -> DocxSplit:
    """Read ``path`` and run :func:`split_document` with its app.xml page count and style names."""
    parts = read_docx_parts(path)
    return split_document(
        parts.document_xml,
        app_pages=app_page_count(parts.app_xml),
        styles=style_names(parts.styles_xml),
        source_name=Path(path).name,
    )
