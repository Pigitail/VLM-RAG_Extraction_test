"""A "perfect model" built from truth files, for checking the harness itself.

``oracle_responder(truths)`` answers every request of every technique the way
a flawless model would, using only the ground truth of the synthetic
documents. Running the evaluation with it (``--oracle``) must score at or
near 100 %: anything less is a harness or technique bug, not a model
weakness. The remaining gap is by design and documented in the report (for
example the ``inherited`` drawing sheets: with ``title_block_pages="first"``
the second sheet of a set carries the first sheet's title block).

The oracle never looks at pixels, so it also works with stubbed renderers.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable, Mapping

__all__ = ["oracle_responder", "slides_answer", "docx_image_answer", "docx_table_answer"]


def _blocks(request: Any, header: str) -> list[dict]:
    for part in request.parts:
        if isinstance(part, str) and part.startswith(header + "\n"):
            try:
                return json.loads(part.split("\n", 1)[1])
            except ValueError:
                return []
    return []


def slides_answer(truth: Mapping[str, Any], request: Any) -> dict:
    """The truth descriptions of one slide, under the block numbers the request assigned."""
    from ..techniques.slides.prompt import IMAGE_BLOCKS_HEADER, LAYOUT_BLOCKS_HEADER

    slide = next((s for s in truth.get("slides") or [] if int(s["slide"]) == int(request.page or 0)), None)
    if slide is None:
        return {"visual_summary": "", "image_descriptions": [], "flow_notes": [], "notes": ""}
    order_by_name = {b.get("shape_name"): b.get("block_order") for b in _blocks(request, LAYOUT_BLOCKS_HEADER)}
    image_names = {b.get("shape_name") for b in _blocks(request, IMAGE_BLOCKS_HEADER)}
    items = []
    for block in slide.get("image_blocks") or []:
        name = block["shape_name"]
        if name not in image_names or order_by_name.get(name) is None:
            continue
        items.append({
            "block_order": order_by_name[name],
            "related_block_orders": [
                order_by_name[n] for n in block.get("related_shapes") or [] if order_by_name.get(n) is not None
            ],
            "image_role": block["image_role"],
            "description": block["description"],
            "description_en": block.get("description_en", ""),
            "description_ko": block.get("description_ko", ""),
            "confidence": "high",
        })
    return {
        "visual_summary": slide.get("visual_summary", ""),
        "image_descriptions": items,
        "flow_notes": list(slide.get("flow_notes") or []),
        "notes": "",
    }


def docx_image_answer(truth: Mapping[str, Any], page: Any) -> dict:
    return {
        "image_descriptions": [
            {"location_hint": image.get("location_hint", ""), "description": image["description"]}
            for image in truth.get("images") or []
            if int(image["page"]) == int(page or 0)
        ],
        "notes": "",
    }


def docx_table_answer(truth: Mapping[str, Any], request: Any) -> dict:
    """Row -> page distribution of the (first) multi-page truth table covering ``request.page``."""
    tables = [t for t in truth.get("tables") or [] if t.get("multi_page") and t.get("row_pages")]
    table = next(
        (t for t in tables if int(t["start_page"]) == int(request.page or 0)),
        tables[0] if tables else None,
    )
    if table is None:
        return {}
    pages = range(int(table["start_page"]), int(table["end_page"]) + 1)
    return {
        "has_title": bool(table.get("title")),
        "title_text": table.get("title", ""),
        "title_source": table.get("title_source", "none"),
        "title_confidence": "high",
        "row_distribution": {
            str(p): [i for i, rp in enumerate(table["row_pages"]) if int(rp) == p] for p in pages
        },
    }


def oracle_responder(truths: Iterable[Mapping[str, Any]]) -> Callable[[Any], Any]:
    """A :class:`~vlmrag.vlm.backends.ScriptedBackend` responder answering from ``truths``.

    Requests are routed by ``request.source_name`` (the truth's ``source``) and
    the task prefix (``drawing.``, ``slides.``, ``docx.``). Unknown documents or
    tasks get ``"{}"``, which every technique treats as an empty answer.
    """
    by_source = {t.get("source"): t for t in truths if t.get("source")}
    drawing_truths = [t for t in by_source.values() if t.get("doc_type") == "drawing"]
    drawing = None
    if drawing_truths:
        from ..synthetic.drawing import truth_responder

        drawing = truth_responder(drawing_truths)

    def respond(request: Any) -> Any:
        truth = by_source.get(request.source_name)
        if truth is None:
            return "{}"
        task = str(request.task)
        if task.startswith("drawing.") and drawing is not None and truth.get("doc_type") == "drawing":
            try:
                return drawing(request)
            except KeyError:
                return "{}"
        if task.startswith("slides."):
            return slides_answer(truth, request)
        if task == "docx.image_description":
            return docx_image_answer(truth, request.page)
        if task == "docx.table_distribution":
            return docx_table_answer(truth, request)
        return "{}"

    return respond
