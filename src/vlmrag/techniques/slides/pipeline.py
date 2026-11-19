"""Orchestration: PPTX -> layout blocks -> slide renders -> one VLM call per slide -> chunks.

Steps
    1. Read every slide's text, tables and layout blocks from the PPTX XML
       (:mod:`.layout`). This never needs a model and never fails the file
       because of one bad shape.
    2. Only slides that contain at least one picture need visual grounding.
       For those, the deck is converted to PDF once with LibreOffice and the
       needed pages are rendered at 180 dpi.
    3. One request per slide (:mod:`.prompt`): whole-slide render + block IDs
       and boxes + XML text. Slides are never batched into one request - the
       "one slide, one answer" contract keeps block IDs unambiguous and lets a
       failure affect a single slide only. Requests run concurrently in a
       thread pool; :class:`~vlmrag.vlm.client.VLMClient` enforces the
       concurrency limit.
    4. Answers are validated (:mod:`.parse`), attached to their image blocks
       and folded into the chunk text (:mod:`.chunks`).
    5. Optionally, crop-aware image assets and a ``layout.json`` sidecar per
       slide are written for viewers (:mod:`.assets`).

Fallbacks never fail the file: whatever goes wrong in steps 2-5 (no
LibreOffice, a page that cannot be rendered, an API error after retries,
unparseable or empty answers) leaves that slide with its XML text only, and
the reason is recorded in ``Chunk.provenance["fallbacks"]``. Only a file that
cannot be opened as a presentation raises.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from ...config import Settings
from ...render.images import to_png
from ...schema import Chunk, doc_id_for
from ...vlm.types import VLMRequest
from .assets import save_picture
from .chunks import build_slide_chunk, has_content, payload_is_empty, visual_payload
from .layout import Deck, SlideLayout, read_deck
from .parse import SlideVisual, parse_visual
from .prompt import BLOCK_TEXT_LIMIT, PROMPT_VERSION, SLIDE_TEXT_LIMIT, TASK, build_request

__all__ = [
    "RENDER_DPI",
    "VLM_TIMEOUT_S",
    "VLM_MAX_OUTPUT_TOKENS",
    "PARSE_RETRIES",
    "CONVERT_TIMEOUT_S",
    "SlideRun",
    "pdf_page_map",
    "render_slides",
    "describe_slide",
    "deck_asset_dirname",
    "extract",
]

log = logging.getLogger(__name__)

RENDER_DPI = 180
VLM_TIMEOUT_S = 240.0
VLM_MAX_OUTPUT_TOKENS = 4000
# One re-ask when the answer is empty or not JSON: two attempts in total, like the
# original per-slide loop. Transport errors are retried separately by VLMClient.
PARSE_RETRIES = 1
CONVERT_TIMEOUT_S = 60.0
_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class SlideRun:
    """Mutable per-slide state collected while the pipeline runs."""

    slide: SlideLayout
    png: Optional[bytes] = field(default=None, repr=False)
    pdf_page: Optional[int] = None  # 1-based page of the intermediate PDF
    visual: Optional[SlideVisual] = None
    call: dict = field(default_factory=dict)
    fallbacks: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    assets: dict[int, dict] = field(default_factory=dict)

    def add_fallback(self, code: str) -> None:
        if code not in self.fallbacks:
            self.fallbacks.append(code)


# --------------------------------------------------------------------------- rendering


def pdf_page_map(hidden: list[bool], pdf_page_count: int) -> Optional[dict[int, int]]:
    """Map 1-based slide numbers to 0-based PDF page indexes, or ``None`` if ambiguous.

    LibreOffice does not export hidden slides, so after a hidden slide the PDF
    page index no longer equals ``slide - 1``. If the page count matches the
    number of visible slides, visible slides are mapped in order; if it
    matches the total (hidden slides were exported), the mapping is the
    identity; anything else is ambiguous and visual grounding is skipped
    rather than describing the wrong slide.
    """
    visible = [i + 1 for i, is_hidden in enumerate(hidden) if not is_hidden]
    if pdf_page_count == len(visible):
        return {number: index for index, number in enumerate(visible)}
    if pdf_page_count == len(hidden):
        return {i + 1: i for i in range(len(hidden))}
    return None


def render_slides(
    path: Path,
    deck: Deck,
    numbers: list[int],
    settings: Settings,
    *,
    dpi: float = RENDER_DPI,
    timeout_s: float = CONVERT_TIMEOUT_S,
) -> tuple[dict[int, tuple[bytes, int]], dict[int, list[str]]]:
    """Render the requested slides as PNG via one LibreOffice PDF conversion.

    Returns ``({slide: (png, pdf_page_1_based)}, {slide: [fallback codes]})``.
    Never raises: conversion and render problems become fallback codes.
    """
    from ...render.office import ConversionError, convert, soffice_candidates
    from ...render.pdf import PdfPages

    pngs: dict[int, tuple[bytes, int]] = {}
    fallbacks: dict[int, list[str]] = {}
    if not numbers:
        return pngs, fallbacks
    work = Path(tempfile.mkdtemp(prefix="vlmrag_slides_"))
    try:
        try:
            pdf_path = convert(
                path, "pdf", work, timeout_s=timeout_s, candidates=soffice_candidates(settings.soffice)
            )
            pdf = PdfPages(pdf_path)
        except Exception as exc:  # noqa: BLE001 - ConversionError, PdfiumError, OSError, ...: never fail the file
            kind = "" if isinstance(exc, ConversionError) else f"{type(exc).__name__}: "
            log.warning("%s: slide render unavailable, using XML text only: %s%s", path.name, kind, exc)
            return pngs, {n: ["render_unavailable"] for n in numbers}
        with pdf:
            mapping = pdf_page_map([s.hidden for s in deck.slides], len(pdf))
            if mapping is None:
                log.warning(
                    "%s: PDF has %d pages for %d slides; skipping visual grounding", path.name, len(pdf), deck.slide_count
                )
                return pngs, {n: ["page_count_mismatch"] for n in numbers}
            for number in numbers:
                index = mapping.get(number)
                if index is None:
                    fallbacks[number] = ["hidden_slide"]
                    continue
                try:
                    pngs[number] = (to_png(pdf.render(index, dpi)), index + 1)
                except Exception as exc:  # noqa: BLE001 - one page must not stop the others
                    log.warning("%s: slide %d render failed: %s", path.name, number, exc)
                    fallbacks[number] = ["render_failed"]
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return pngs, fallbacks


# --------------------------------------------------------------------------- model call


def describe_slide(client: Any, request: VLMRequest) -> tuple[Optional[SlideVisual], dict, list[str]]:
    """Run one slide request; returns ``(visual or None, call info, fallback codes)``. Never raises."""
    try:
        value, response = client.call_json(request, parse_retries=PARSE_RETRIES)
    except Exception as exc:  # noqa: BLE001 - after the client's retries: fall back to XML text
        log.warning("%s slide %s: visual grounding failed: %s: %s", request.source_name, request.page,
                    type(exc).__name__, exc)
        return None, {"status": "error", "error": f"{type(exc).__name__}: {exc}"}, ["vlm_error"]
    info = {
        "status": "ok",
        "backend": response.backend,
        "model": response.model,
        "cached": bool(response.cached),
        "latency_s": round(float(response.latency_s or 0.0), 3),
        "usage": response.usage.to_dict(),
    }
    if value is None:
        info["status"] = "unparseable"
        return None, info, ["unparseable_json"]
    try:
        visual = parse_visual(value if isinstance(value, dict) else response.text)
    except Exception as exc:  # noqa: BLE001 - a pathological answer must not fail the file
        log.warning("%s slide %s: visual answer could not be validated: %s: %s", request.source_name,
                    request.page, type(exc).__name__, exc)
        info["status"] = "invalid"
        info["error"] = f"{type(exc).__name__}: {exc}"
        return None, info, ["parse_error"]
    if visual is None:
        info["status"] = "invalid"
        return None, info, ["invalid_json_shape"]
    fallbacks = []
    for issue in visual.issues:
        code = issue.split(":", 1)[0]
        if code not in fallbacks:
            fallbacks.append(code)
    if visual.is_empty():
        info["status"] = "empty"
        fallbacks.append("empty_visual_result")
        log.warning("%s slide %s: visual grounding returned an empty result (%d chars)",
                    request.source_name, request.page, len(response.text or ""))
    return visual, info, fallbacks


# --------------------------------------------------------------------------- assets


def _safe_stem(stem: str, fallback: str) -> str:
    return _SAFE_STEM_RE.sub("_", stem).strip("._-") or fallback


def deck_asset_dirname(stem: str, doc_id: str) -> str:
    """``<ascii-safe stem>-<first 8 hex of doc_id>``: readable, portable and collision-free.

    Two different decks often share a file name (``manual.pptx`` in two
    folders) or reduce to the same safe stem; without the content hash their
    assets would overwrite each other in a shared ``asset_dir`` and the first
    deck's chunks would point at the second deck's pictures.
    """
    short = (doc_id or "deck")[:8]
    safe = _safe_stem(stem, "")
    return f"{safe}-{short}" if safe else short


def _write_assets(run: SlideRun, deck: Deck, source_name: str, asset_root: Path, deck_dir: Path,
                  settings: Settings) -> None:
    from ...render.office import soffice_candidates

    slide = run.slide
    slide_dir = deck_dir / f"slide_{slide.number:03d}"
    for block in slide.image_blocks:
        try:
            saved = save_picture(block, slide_dir, asset_root, timeout_s=CONVERT_TIMEOUT_S,
                                 candidates=soffice_candidates(settings.soffice))
        except Exception as exc:  # noqa: BLE001 - assets are optional
            log.warning("%s slide %d: asset for block %d not written: %s", source_name, slide.number, block.order, exc)
            run.add_fallback("asset_write_failed")
            continue
        run.assets[block.order] = saved.to_dict()
        for code in saved.fallbacks:
            run.add_fallback(f"asset_{code}")
    sidecar = {
        "source_name": source_name,
        "slide": slide.number,
        "slide_size": {"width": deck.slide_width, "height": deck.slide_height},
        "blocks": [b.to_dict() for b in slide.blocks],
        "assets": [dict(v, block_order=k) for k, v in sorted(run.assets.items())],
    }
    if run.visual is not None:
        # The same normalised, block-enriched answer as the chunk metadata, so a
        # viewer reading only the sidecar gets each description with its asset.
        payload, _ = visual_payload(slide, run.visual, run.assets)
        if not payload_is_empty(payload):
            sidecar["visual_grounding"] = payload
    slide_dir.mkdir(parents=True, exist_ok=True)
    (slide_dir / "layout.json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


# --------------------------------------------------------------------------- entry point


def extract(
    path: Union[str, Path],
    client: Any,
    settings: Optional[Settings] = None,
    *,
    asset_dir: Optional[Union[str, Path]] = None,
    dpi: float = RENDER_DPI,
) -> list[Chunk]:
    """Extract one chunk per non-empty slide of a ``.pptx`` software manual.

    ``asset_dir`` (optional) receives ``<deck stem>-<doc_id[:8]>/slide_NNN/img_NNN.<ext>``
    crop-aware image assets plus a ``layout.json`` per slide with pictures.
    """
    path = Path(path)
    settings = settings or Settings()
    if not path.is_file():
        raise FileNotFoundError(f"input not found: {path}")
    if path.suffix.lower() != ".pptx":
        raise ValueError(f"the slides technique reads .pptx files (got {path.name})")

    deck = read_deck(path)
    doc_id = doc_id_for(path)
    source_name = path.name
    runs = {s.number: SlideRun(slide=s) for s in deck.slides}
    for run in runs.values():
        if run.slide.error:
            run.add_fallback("slide_parse_error")

    numbers = [s.number for s in deck.slides if s.image_blocks]
    if numbers:
        pngs, render_fallbacks = render_slides(path, deck, numbers, settings, dpi=dpi)
        for number, codes in render_fallbacks.items():
            for code in codes:
                runs[number].add_fallback(code)
        for number, (png, page) in pngs.items():
            runs[number].png, runs[number].pdf_page = png, page

    jobs = [run for run in runs.values() if run.png is not None]
    if jobs:
        requests = {
            run.slide.number: build_request(
                run.slide,
                deck.slide_count,
                run.png,  # type: ignore[arg-type]
                dpi=dpi,
                doc_id=doc_id,
                source_name=source_name,
                reasoning_effort=settings.reasoning_effort,
                timeout_s=VLM_TIMEOUT_S,
                max_output_tokens=VLM_MAX_OUTPUT_TOKENS,
            )
            for run in jobs
        }
        workers = min(len(jobs), max(1, int(getattr(client, "max_concurrency", settings.max_concurrency) or 1)))
        log.info("%s: visual grounding for %d slide(s) with %d worker(s)", source_name, len(jobs), workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {n: pool.submit(describe_slide, client, req) for n, req in requests.items()}
            for number, future in futures.items():
                visual, info, codes = future.result()
                run = runs[number]
                run.visual, run.call = visual, info
                run.issues = list(visual.issues) if visual is not None else []
                for code in codes:
                    run.add_fallback(code)
        for run in jobs:
            run.png = None  # release the renders early

    if asset_dir is not None:
        asset_root = Path(asset_dir)
        deck_dir = asset_root / deck_asset_dirname(path.stem, doc_id)
        for run in runs.values():
            if not run.slide.image_blocks:
                continue
            try:
                _write_assets(run, deck, source_name, asset_root, deck_dir, settings)
            except Exception as exc:  # noqa: BLE001 - assets are optional; e.g. an unwritable sidecar
                log.warning("%s slide %d: assets/sidecar not written: %s", source_name, run.slide.number, exc)
                run.add_fallback("asset_write_failed")

    chunks: list[Chunk] = []
    slide_size = {"width": deck.slide_width, "height": deck.slide_height}
    for number in sorted(runs):
        run = runs[number]
        if not has_content(run.slide, run.visual):
            continue
        chunks.append(build_slide_chunk(
            doc_id=doc_id,
            source_name=source_name,
            slide=run.slide,
            slide_count=deck.slide_count,
            chunk_index=len(chunks),
            visual=run.visual,
            assets=run.assets or None,
            slide_size=slide_size,
            provenance=_provenance(run, client, settings, dpi),
        ))
    log.info("%s: %d slide(s) -> %d chunk(s), %d visual call(s)", source_name, deck.slide_count, len(chunks), len(jobs))
    return chunks


def _provenance(run: SlideRun, client: Any, settings: Settings, dpi: float) -> dict:
    prov: dict[str, Any] = {
        "technique": "slides",
        "model": str(getattr(client, "model", "") or ""),
        "backend": str(getattr(getattr(client, "backend", None), "name", "") or ""),
        # The keys every technique shares (see docs/techniques/common.md): the
        # prompts that ran for this chunk and one record per model call.
        "prompt_versions": {},
        "calls": [],
        "fallbacks": list(run.fallbacks),
    }
    if not run.slide.image_blocks:
        prov["visual_call"] = {"status": "skipped", "reason": "no_images"}
        return prov
    prov["render"] = {"dpi": dpi, "converter": "libreoffice", "pdf_page": run.pdf_page}
    if run.call:
        prov["visual_call"] = dict(run.call)
        prov["prompt_versions"] = {TASK: PROMPT_VERSION}
        prov["calls"] = [{"task": TASK, "prompt_version": PROMPT_VERSION, **run.call}]
        prov["request"] = {
            "timeout_s": VLM_TIMEOUT_S,
            "max_output_tokens": VLM_MAX_OUTPUT_TOKENS,
            "reasoning_effort": settings.reasoning_effort,
            "parse_retries": PARSE_RETRIES,
            "slide_text_limit": SLIDE_TEXT_LIMIT,
            "block_text_limit": BLOCK_TEXT_LIMIT,
        }
    else:
        prov["visual_call"] = {"status": "skipped", "reason": run.fallbacks[0] if run.fallbacks else "not_rendered"}
    if run.issues:
        prov["visual_issues"] = list(run.issues)
    return prov
