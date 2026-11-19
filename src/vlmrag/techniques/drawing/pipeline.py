"""Entry point of the drawing technique: one PDF in, one chunk per page out.

By default (``title_block_pages="first"``) the title block is read once, on
page 1, and attached to every page's chunk together with that page's own text
layer: a drawing set shares one identity (number, title, revision), and
reading a title block costs three model calls with a dozen large images. With
``"all"`` every page is read separately (for files that concatenate unrelated
sheets); pages then run concurrently and any page that fails makes the whole
file fail, so the CLI's error file names it instead of silently dropping it.

Chunk pages are always the 1-based page numbers of the source PDF.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, Union

from ...render.pdf import PdfPages
from ...schema import Chunk, doc_id_for
from .chunks import build_chunk
from .config import DrawingConfig
from .crops import FULL_PAGE, region_text
from .title_block import DrawingExtractionError, TitleBlockResult, read_title_block

__all__ = ["extract"]

log = logging.getLogger(__name__)


def _read_targets(
    pages: PdfPages,
    targets: list[int],
    client: Any,
    cfg: DrawingConfig,
    **kwargs: Any,
) -> dict[int, TitleBlockResult]:
    if len(targets) == 1:
        return {targets[0]: read_title_block(pages, targets[0], client, cfg, **kwargs)}
    workers = max(1, min(len(targets), int(getattr(client, "max_concurrency", 4) or 4)))
    results: dict[int, TitleBlockResult] = {}
    errors: list[tuple[int, BaseException]] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="drawing-page") as pool:
        futures = {index: pool.submit(read_title_block, pages, index, client, cfg, **kwargs) for index in targets}
        for index, future in futures.items():
            try:
                results[index] = future.result()
            except Exception as exc:  # noqa: BLE001 - collected and re-raised below
                errors.append((index, exc))
    if errors:
        for index, exc in errors[1:]:
            log.error("page %d: %s: %s", index + 1, type(exc).__name__, exc)
        raise errors[0][1]
    return results


def extract(
    path: Union[str, Path],
    client: Any,
    settings: Any = None,
    *,
    config: Optional[DrawingConfig] = None,
) -> list[Chunk]:
    """Extract one chunk per page of the drawing PDF at ``path``.

    ``config`` defaults to :meth:`DrawingConfig.from_env`. ``settings`` supplies
    the reasoning effort (``Settings.reasoning_effort``). Raises
    :class:`~.title_block.TitleBlockGateError` when a title block yields fewer
    than ``config.min_pairs`` pairs, and :class:`~.title_block.DrawingExtractionError`
    when the vision pass fails or the PDF has no pages.
    """
    cfg = config if config is not None else DrawingConfig.from_env()
    path = Path(path)
    doc_id = doc_id_for(path)
    source_name = path.name
    effort = getattr(settings, "reasoning_effort", "low") if settings is not None else "low"

    with PdfPages(path) as pages:
        count = len(pages)
        if count == 0:
            raise DrawingExtractionError(f"{source_name}: the PDF has no pages")
        targets = [0] if cfg.title_block_pages == "first" else list(range(count))
        results = _read_targets(
            pages, targets, client, cfg, doc_id=doc_id, source_name=source_name, reasoning_effort=effort
        )
        page_texts = [region_text(pages, i, FULL_PAGE) for i in range(count)]
        rotations = [pages.rotation(i) for i in range(count)]

    model = str(getattr(client, "model", "") or "")
    backend = str(getattr(getattr(client, "backend", None), "name", "") or "")
    chunks = []
    for index in range(count):
        title_block = results.get(index) or results[targets[0]]
        chunks.append(
            build_chunk(
                doc_id=doc_id,
                source_name=source_name,
                page_index=index,
                page_count=count,
                page_text=page_texts[index],
                rotation=rotations[index],
                title_block=title_block,
                cfg=cfg,
                model=model,
                backend=backend,
            )
        )
    log.info("%s: %d chunk(s), title block from page(s) %s", source_name, len(chunks), [t + 1 for t in targets])
    return chunks
