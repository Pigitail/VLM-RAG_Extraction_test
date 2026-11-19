"""Read the title block of one page: crops -> vision + table passes -> normalize -> merge -> gate.

Stage by stage, and why:

1. **Render** every crop of :func:`~.crops.plan_crops` and read the text layer
   of the very same boxes as hints.
2. **Vision pass** (all page tiles + focus crops) and **table pass**
   (metadata-table crops) run concurrently: they are independent, and the
   client's semaphore keeps the overall number of in-flight calls bounded.
   The vision pass is required; if it fails or answers nothing, the page
   cannot be read and the error propagates. The table pass is an enhancement:
   any failure falls back to the vision answer alone.
3. **Normalize pass** (text only) maps the raw labels of both passes onto the
   canonical fields. Labels vary too much between organisations for an alias
   table alone; a text-only pass is cheap compared with the image passes. A
   failed or empty normalization falls back to the raw vision fields.
4. **Merge** each field with priority *normalized > table > raw vision*, then
   clean it (table fields lose echoed labels and grid letters).
5. **Gate**: canonical pairs plus raw pairs must reach ``min_pairs`` (default
   3). Fewer means the model did not really read a title block - typically a
   wrong page, a blank answer or a refusal - and indexing an almost empty
   record would only produce chunks that can never be found, so the page
   raises :class:`TitleBlockGateError`.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

from ...vlm.backends import ReplayMiss
from ...vlm.types import VLMRequest, VLMResponse
from .config import DrawingConfig, TitleBlockSchema
from .crops import (
    FULL_PAGE,
    TABLE_HINT_INTRO,
    VISION_HINT_INTRO,
    hint_block,
    hint_section,
    plan_crops,
    region_text,
    render_crop,
)
from .entries import build_field_entries
from .parsing import (
    ParsedTitleBlock,
    clean_field_value,
    dedupe_pairs,
    first_json_object_text,
    merge_pairs,
    pairs_text,
    parse_table_rows,
    parse_title_block,
    table_rows_text,
)
from .prompts import (
    NORMALIZE_PAIRS_HEADER,
    NORMALIZE_PAYLOAD_HEADER,
    NORMALIZE_PROMPT_VERSION,
    NORMALIZE_TASK,
    TABLE_PROMPT_VERSION,
    TABLE_TASK,
    VISION_PROMPT_VERSION,
    VISION_TASK,
    build_normalize_prompt,
    build_table_prompt,
    build_vision_prompt,
    fields_json_schema,
    table_json_schema,
)

__all__ = [
    "DrawingExtractionError",
    "TitleBlockGateError",
    "TitleBlockResult",
    "merge_fields",
    "canonical_pairs",
    "apply_gate",
    "read_title_block",
    "FULL_PAGE_LABEL",
]

log = logging.getLogger(__name__)

FULL_PAGE_LABEL = "FULL_PAGE x=0.00-1.00 y=0.00-1.00"


class DrawingExtractionError(RuntimeError):
    """A drawing page could not be read (the CLI records it per file and moves on)."""


class TitleBlockGateError(DrawingExtractionError):
    """Fewer title-block pairs than ``min_pairs`` were found on a page."""

    # Keyword defaults keep the error picklable (unpickling calls ``cls(message)`` and then
    # restores the attributes), so it survives a process-pool batch runner.
    def __init__(self, message: str, *, page: int = 0, pairs_found: int = 0, min_pairs: int = 0) -> None:
        super().__init__(message)
        self.page = page
        self.pairs_found = pairs_found
        self.min_pairs = min_pairs


@dataclass
class TitleBlockResult:
    """Everything read from one page's title block, plus how it was obtained."""

    page: int
    fields: dict[str, str]
    pairs: list[dict]
    raw_pairs: list[dict] = field(default_factory=list)
    table_rows: list[dict] = field(default_factory=list)
    table_pairs: list[dict] = field(default_factory=list)
    field_entries: list[dict] = field(default_factory=list)
    field_sources: dict[str, str] = field(default_factory=dict)
    raw_text: str = ""
    table_text: str = ""
    normalized_text: str = ""
    fallbacks: list[str] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    hints: dict = field(default_factory=dict)

    @property
    def pairs_text(self) -> str:
        return pairs_text(self.pairs)

    @property
    def raw_pairs_text(self) -> str:
        return pairs_text(self.raw_pairs)

    @property
    def table_pairs_text(self) -> str:
        return pairs_text(self.table_pairs)

    @property
    def table_rows_text(self) -> str:
        return table_rows_text(self.table_rows)


# --------------------------------------------------------------------------- pure steps


def merge_fields(
    primary: dict[str, str],
    table: dict[str, str],
    raw: dict[str, str],
    schema: TitleBlockSchema,
    *,
    primary_name: str = "normalize",
) -> tuple[dict[str, str], dict[str, str]]:
    """Per field, the first non-empty cleaned value of ``primary`` > ``table`` > ``raw``.

    Returns ``(fields, sources)`` where ``sources[key]`` names the pass that won.
    The table pass outranks the raw vision answer because it read the ruled
    cells with dedicated close-ups; the normalized answer outranks both because
    it saw both.
    """
    fields: dict[str, str] = {}
    sources: dict[str, str] = {}
    for key in schema.keys():
        fields[key] = ""
        for name, candidate in ((primary_name, primary), ("table", table), ("vision", raw)):
            value = clean_field_value(key, (candidate or {}).get(key, ""), schema)
            if value:
                fields[key] = value
                sources[key] = name
                break
    # A second pass keeps cleaning idempotent for values that only became table values above.
    return {k: clean_field_value(k, v, schema) for k, v in fields.items()}, sources


def canonical_pairs(fields: dict[str, str], schema: TitleBlockSchema) -> list[dict[str, str]]:
    """Filled canonical fields as ``{"topic": display label, "value": ...}`` in schema order."""
    return [{"topic": spec.label, "value": fields[spec.key]} for spec in schema.fields if fields.get(spec.key)]


def apply_gate(
    final_pairs: list[dict],
    min_pairs: int,
    *,
    page: int,
    source_name: str = "",
    raw_text: str = "",
    preview_chars: int = 4000,
) -> None:
    """Raise :class:`TitleBlockGateError` when fewer than ``min_pairs`` pairs were found."""
    found = len(final_pairs)
    if found >= min_pairs:
        return
    preview = (raw_text or "").strip()[:preview_chars]
    log.warning("%s page %d: vision answer preview (gate failed): %s", source_name, page, preview)
    raise TitleBlockGateError(
        f"{source_name or 'drawing'} page {page}: title block yielded {found} pair(s), "
        f"fewer than the required {min_pairs}",
        page=page,
        pairs_found=found,
        min_pairs=min_pairs,
    )


# --------------------------------------------------------------------------- model calls


@dataclass
class _PassOutput:
    text: str = ""
    response: Optional[VLMResponse] = None
    error: Optional[BaseException] = None


def _run_pass(client: Any, request: VLMRequest, empty_retries: int) -> _PassOutput:
    """One pass; errors are captured so a concurrent sibling pass can finish first.

    An *empty* answer is re-asked up to ``empty_retries`` more times: it carries
    nothing a fallback could use. A non-empty answer that is not JSON is kept as
    it is and goes to the line-based parser, as re-asking it would mostly buy
    the same prose again at the price of another dozen large images.
    ``call_json`` is used (without its own parse retries) because the client
    never caches output it cannot parse, so an empty answer is not replayed
    from the cache and each re-ask is a real new attempt.
    """
    attempts = max(0, int(empty_retries)) + 1
    response: Optional[VLMResponse] = None
    try:
        for attempt in range(1, attempts + 1):
            _, response = client.call_json(request)
            if (response.text or "").strip():
                break
            log.warning("%s page=%s: empty answer (attempt %d/%d)", request.task, request.page, attempt, attempts)
    except Exception as exc:  # noqa: BLE001 - classified by the caller (required vs optional pass)
        return _PassOutput(error=exc)
    assert response is not None
    return _PassOutput(text=(response.text or "").strip(), response=response)


def _call_record(request: VLMRequest, output: _PassOutput) -> dict:
    # Same record shape as the other techniques: task, prompt_version, status, then
    # the response details. A non-JSON answer is still "ok" here: the line parser
    # reads it, and that fallback is recorded in provenance["fallbacks"].
    if output.error is not None:
        status = "error"
    elif not output.text:
        status = "empty"
    else:
        status = "ok"
    record: dict[str, Any] = {"task": request.task, "prompt_version": request.prompt_version, "status": status}
    if output.response is not None:
        response = output.response
        record.update(
            model=response.model,
            backend=response.backend,
            cached=response.cached,
            latency_s=round(float(response.latency_s or 0.0), 3),
            usage=response.usage.to_dict(),
        )
    if output.error is not None:
        record["error"] = f"{type(output.error).__name__}: {output.error}"
    return record


def _reraise_replay_miss(output: _PassOutput) -> None:
    # A replay miss means the recording does not match this run; hiding it behind a
    # fallback would silently produce different chunks than the recorded run.
    if isinstance(output.error, ReplayMiss):
        raise output.error


# --------------------------------------------------------------------------- orchestration


def read_title_block(
    pages: Any,
    index: int,
    client: Any,
    cfg: DrawingConfig,
    *,
    doc_id: str = "",
    source_name: str = "",
    reasoning_effort: Optional[str] = "low",
) -> TitleBlockResult:
    """Run the full title-block pipeline on page ``index`` (0-based) of an open :class:`PdfPages`."""
    page_no = index + 1
    schema = cfg.schema
    fallbacks: list[str] = []
    calls: list[dict] = []

    crops = plan_crops(cfg)
    prepared = [render_crop(pages, index, spec, cfg.max_image_pixels) for spec in crops]
    page_text = region_text(pages, index, FULL_PAGE)
    region_texts = {p.spec.label: region_text(pages, index, p.spec.bbox) for p in prepared}
    vision_items = [p for p in prepared if p.spec.for_vision]
    table_items = [p for p in prepared if p.spec.for_table]

    vision_sections = [hint_section(FULL_PAGE_LABEL, page_text)]
    vision_sections += [hint_section(p.spec.label, region_texts[p.spec.label]) for p in vision_items]
    table_sections = [hint_section(p.spec.label, region_texts[p.spec.label]) for p in table_items]
    vision_hints = hint_block(VISION_HINT_INTRO, vision_sections)
    table_hints = hint_block(TABLE_HINT_INTRO, table_sections)
    hints = {
        "page_text_chars": len(page_text),
        "vision_sections": sum(1 for s in vision_sections if s),
        "table_sections": sum(1 for s in table_sections if s),
    }
    if not page_text:
        fallbacks.append("no_text_layer")
    if any(p.info["capped"] for p in prepared):
        fallbacks.append("image_pixel_cap")

    def request(task: str, version: str, parts: list, json_schema: dict) -> VLMRequest:
        return VLMRequest(
            task=task,
            prompt_version=version,
            parts=parts,
            json_schema=json_schema,
            schema_name=task.replace(".", "_"),
            max_output_tokens=cfg.max_output_tokens,
            reasoning_effort=reasoning_effort,
            timeout_s=cfg.timeout_s,
            doc_id=doc_id,
            source_name=source_name,
            page=page_no,
        )

    vision_parts: list = [build_vision_prompt(cfg)]
    if vision_hints:
        vision_parts.append(vision_hints)
    vision_parts += [p.image for p in vision_items]
    vision_request = request(VISION_TASK, VISION_PROMPT_VERSION, vision_parts, fields_json_schema(schema))

    table_request = None
    if table_items:
        table_parts: list = [build_table_prompt(cfg)]
        if table_hints:
            table_parts.append(table_hints)
        table_parts += [p.image for p in table_items]
        table_request = request(TABLE_TASK, TABLE_PROMPT_VERSION, table_parts, table_json_schema(schema))

    retries = max(0, int(cfg.empty_output_retries))
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="drawing-pass") as pool:
        vision_future = pool.submit(_run_pass, client, vision_request, retries)
        table_future = pool.submit(_run_pass, client, table_request, retries) if table_request else None
        vision = vision_future.result()
        table = table_future.result() if table_future is not None else None

    # ---- vision (required)
    calls.append(_call_record(vision_request, vision))
    _reraise_replay_miss(vision)
    if vision.error is not None:
        raise DrawingExtractionError(
            f"{source_name} page {page_no}: vision pass failed: {type(vision.error).__name__}: {vision.error}"
        ) from vision.error
    if not vision.text:
        raise DrawingExtractionError(
            f"{source_name} page {page_no}: vision pass returned an empty answer after {retries + 1} attempt(s)"
        )
    raw = parse_title_block(vision.text, schema)
    if not raw.from_json:
        fallbacks.append("vision_text_fallback")
    raw_pairs = dedupe_pairs(raw.pairs)

    # ---- table (optional)
    table_text = ""
    table_rows: list[dict] = []
    table_pairs: list[dict] = []
    table_parsed = ParsedTitleBlock(fields={k: "" for k in schema.keys()})
    if table is None or table_request is None:
        fallbacks.append("table_pass_skipped")
    else:
        calls.append(_call_record(table_request, table))
        _reraise_replay_miss(table)
        if table.error is not None:
            log.warning(
                "%s page %d: table pass failed, using the vision answer only: %s: %s",
                source_name, page_no, type(table.error).__name__, table.error,
            )
            fallbacks.append("table_pass_failed")
        elif not table.text:
            fallbacks.append("table_pass_empty")
        else:
            table_text = table.text
            table_parsed = parse_title_block(table_text, schema)
            table_rows = parse_table_rows(table_text)
            table_pairs = dedupe_pairs(table_parsed.pairs)
            if not table_parsed.from_json:
                fallbacks.append("table_text_fallback")

    raw_pairs = merge_pairs(raw_pairs, table_pairs)
    raw_pairs_text = pairs_text(raw_pairs)

    # ---- normalize (optional, text only)
    payload_parts = []
    main_json = first_json_object_text(vision.text)
    if main_json:
        payload_parts.append(f"[MAIN_VISION_JSON]\n{main_json}")
    if table_text:
        payload_parts.append(f"[TABLE_VISION_JSON]\n{first_json_object_text(table_text)}")
    rows_text = table_rows_text(table_rows)
    if rows_text:
        payload_parts.append(f"[TABLE_ROWS]\n{rows_text}")
    normalize_parts: list = [build_normalize_prompt(cfg)]
    payload = "\n\n".join(payload_parts)
    if payload:
        normalize_parts.append(f"{NORMALIZE_PAYLOAD_HEADER}\n{payload}")
    if raw_pairs_text:
        normalize_parts.append(f"{NORMALIZE_PAIRS_HEADER}\n{raw_pairs_text}")
    normalize_request = request(NORMALIZE_TASK, NORMALIZE_PROMPT_VERSION, normalize_parts, fields_json_schema(schema))
    normalized = _run_pass(client, normalize_request, retries)
    calls.append(_call_record(normalize_request, normalized))
    _reraise_replay_miss(normalized)

    primary, primary_name, normalized_text = raw, "vision", ""
    if normalized.error is not None:
        log.warning(
            "%s page %d: normalize pass failed, using the raw vision fields: %s: %s",
            source_name, page_no, type(normalized.error).__name__, normalized.error,
        )
        fallbacks.append("normalize_failed")
    elif not normalized.text:
        fallbacks.append("normalize_empty")
    else:
        candidate = parse_title_block(normalized.text, schema)
        if candidate.has_content():
            primary, primary_name, normalized_text = candidate, "normalize", normalized.text
        else:
            fallbacks.append("normalize_no_content")

    # ---- merge + gate
    fields, sources = merge_fields(primary.fields, table_parsed.fields, raw.fields, schema, primary_name=primary_name)
    final_pairs = merge_pairs(canonical_pairs(fields, schema), raw_pairs)
    apply_gate(
        final_pairs,
        cfg.min_pairs,
        page=page_no,
        source_name=source_name,
        raw_text=vision.text,
        preview_chars=cfg.error_preview_chars,
    )
    entries = build_field_entries(
        fields=fields,
        pairs=final_pairs,
        table_pairs=table_pairs,
        schema=schema,
        domain=cfg.domain,
        limit=cfg.field_entry_limit,
        source_text_limit=cfg.field_source_text_limit,
    )
    log.info(
        "%s page %d: title block read (%d pairs, %d table rows, fallbacks=%s)",
        source_name, page_no, len(final_pairs), len(table_rows), ",".join(fallbacks) or "none",
    )
    return TitleBlockResult(
        page=page_no,
        fields=fields,
        pairs=final_pairs,
        raw_pairs=raw_pairs,
        table_rows=table_rows,
        table_pairs=table_pairs,
        field_entries=entries,
        field_sources=sources,
        raw_text=vision.text,
        table_text=table_text,
        normalized_text=normalized_text,
        fallbacks=fallbacks,
        calls=calls,
        images=[p.info for p in prepared],
        hints=hints,
    )
