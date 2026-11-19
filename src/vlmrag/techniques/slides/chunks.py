"""Fold a slide's XML text and its visual grounding into one chunk.

One slide becomes one chunk (``Chunk.page`` = slide number). The three
renderings differ on purpose:

``retrieve_content``
    ``slide=N | title=...``, the slide text with tables as plain rows, then a
    ``[Visual image retrieval cues]`` section with the visual summary and, per
    described image, its detailed description, role and the short English and
    Korean summaries. The visual cues **must** be here: metadata is not
    embedded, so a description that lived only in metadata could never be
    found. Screenshots are often the only place where a screen title, a field
    value or a button label appears.
``llm_content``
    A ``[Source]`` header, the slide text with Markdown tables, and a
    ``[Grounded visual description]`` section that lists every described
    image block with its block number, related blocks, role and confidence,
    so the answering LLM can weigh uncertain descriptions.
``filtering_content``
    ``retrieve_content`` without whitespace.

Descriptions are enriched with the block they refer to (shape name, box,
crop, source image id, asset) so that a viewer can show the right picture.

Model text (summary, descriptions, captions, notes) is Unicode-normalised
(NFKC, straight quotes, one dash, collapsed whitespace) before it is used:
models mix full-width digits and typographic punctuation, and the exact-match
``filtering_content`` should not depend on which variant the model chose.
"""

from __future__ import annotations

from typing import Any, Optional

from ...schema import Chunk
from ...text import filtering_text, normalize_unicode
from .layout import SlideLayout
from .parse import SlideVisual, coerce_order_list, normalize_image_role

__all__ = ["DOC_TYPE", "enrich_descriptions", "visual_payload", "payload_is_empty", "build_slide_chunk", "has_content"]

DOC_TYPE = "slides"
_CONFIDENCE = ("low", "medium", "high")


def _clean_flow_notes(notes: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in notes or []:
        note = normalize_unicode(raw)
        if note and note.lower() not in seen:
            seen.add(note.lower())
            out.append(note)
    return out


def enrich_descriptions(
    slide: SlideLayout, visual: SlideVisual, assets: Optional[dict[int, dict]] = None
) -> tuple[list[dict], list[int]]:
    """Normalise descriptions and attach image-block data.

    Returns ``(descriptions, ungrounded_orders)``. A description whose
    ``block_order`` is not an image block on this slide is kept (the model may
    still be right about what is shown) but gets no block data, and its order
    is reported so provenance can flag it.
    """
    image_blocks = {b.order: b for b in slide.image_blocks}
    assets = assets or {}
    enriched: list[dict] = []
    ungrounded: list[int] = []
    for item in visual.image_descriptions:
        order = int(item.get("block_order") or 0)
        description = normalize_unicode(item.get("description"))
        if order <= 0 or not description:
            continue
        confidence = str(item.get("confidence") or "low").lower()
        entry: dict[str, Any] = {
            "block_order": order,
            "related_block_orders": coerce_order_list(item.get("related_block_orders") or []),
            "description": description,
            "description_en": normalize_unicode(item.get("description_en")),
            "description_ko": normalize_unicode(item.get("description_ko")),
            "confidence": confidence if confidence in _CONFIDENCE else "low",
        }
        role = normalize_image_role(item.get("image_role"))
        if role:
            entry["image_role"] = role
        block = image_blocks.get(order)
        if block is None:
            ungrounded.append(order)
        else:
            data = block.to_dict()
            for key in ("shape_id", "shape_name", "bbox_norm", "crop_norm", "crop_area_norm", "source_image_id"):
                if data.get(key):
                    entry[key] = data[key]
            asset = assets.get(order)
            if asset and asset.get("asset"):
                entry["asset"] = asset["asset"]
        enriched.append(entry)
    return enriched, ungrounded


def visual_payload(
    slide: SlideLayout, visual: Optional[SlideVisual], assets: Optional[dict[int, dict]] = None
) -> tuple[dict, list[int]]:
    """The normalised, block-enriched visual answer, as stored in metadata and in ``layout.json``.

    Returns ``(payload, ungrounded_orders)`` with payload keys
    ``visual_summary``, ``image_descriptions``, ``flow_notes`` and ``notes``.
    Chunk text and the viewer sidecar are built from this one function so
    they can never disagree about what the model said.
    """
    visual = visual or SlideVisual()
    descriptions, ungrounded = enrich_descriptions(slide, visual, assets)
    payload = {
        "visual_summary": normalize_unicode(visual.visual_summary),
        "image_descriptions": descriptions,
        "flow_notes": _clean_flow_notes(visual.flow_notes),
        "notes": normalize_unicode(visual.notes),
    }
    return payload, ungrounded


def payload_is_empty(payload: dict) -> bool:
    return not any(payload.get(k) for k in ("visual_summary", "image_descriptions", "flow_notes", "notes"))


def has_content(slide: SlideLayout, visual: Optional[SlideVisual]) -> bool:
    """A slide is emitted when it has text, pictures or a non-empty visual result."""
    return bool(slide.text_only or slide.text_and_table or slide.image_blocks or (visual and not visual.is_empty()))


def build_slide_chunk(
    *,
    doc_id: str,
    source_name: str,
    slide: SlideLayout,
    slide_count: int,
    chunk_index: int,
    visual: Optional[SlideVisual] = None,
    assets: Optional[dict[int, dict]] = None,
    slide_size: Optional[dict] = None,
    provenance: Optional[dict] = None,
) -> Chunk:
    plain_text = (slide.text_only or slide.text_and_table).strip()
    rich_text = slide.slide_text
    title = slide.title

    prefix = " | ".join(p for p in (f"slide={slide.number}", f"title={title}" if title else "") if p)
    retrieve_parts = [prefix, plain_text]

    llm_lines = ["[Source]", f"source: {source_name}" if source_name else "", f"slide: {slide.number}"]
    if title:
        llm_lines.append(f"title: {title}")
    llm_content = "\n".join(line for line in llm_lines if line)
    if rich_text:
        llm_content = f"{llm_content}\n\n{rich_text}"

    payload, ungrounded = visual_payload(slide, visual, assets)
    descriptions = payload["image_descriptions"]
    summary = payload["visual_summary"]
    notes = payload["notes"]
    flow_notes = payload["flow_notes"]
    visual_used = not payload_is_empty(payload)

    if visual_used:
        visual_lines = ["[Grounded visual description]"]
        cue_lines: list[str] = []
        if summary:
            visual_lines.append(summary)
            cue_lines.append(f"Visual summary: {summary}")
        for item in descriptions:
            related = ""
            if item["related_block_orders"]:
                related = f" (related blocks: {', '.join(str(v) for v in item['related_block_orders'])})"
            text = f"description={item['description']}"
            if item["description_en"]:
                text += f"; summary_en={item['description_en']}"
            if item["description_ko"]:
                text += f"; summary_ko={item['description_ko']}"
            if item.get("image_role"):
                text += f"; image_role={item['image_role']}"
            visual_lines.append(f"- image block {item['block_order']}{related}: {text} [confidence={item['confidence']}]")
            cue_lines.append(f"Image detailed description: {item['description']}")
            if item.get("image_role"):
                cue_lines.append(f"Image role: {item['image_role']}")
            if item["description_en"]:
                cue_lines.append(f"Image display summary EN: {item['description_en']}")
            if item["description_ko"]:
                cue_lines.append(f"Image display summary KO: {item['description_ko']}")
        for note in flow_notes:
            visual_lines.append(f"- flow: {note}")
        if notes:
            visual_lines.append(f"- notes: {notes}")
        llm_content = f"{llm_content}\n\n" + "\n".join(visual_lines)
        if cue_lines:
            retrieve_parts += ["[Visual image retrieval cues]", *cue_lines]

    retrieve_content = "\n".join(p for p in retrieve_parts if p).strip()

    image_blocks = slide.image_blocks
    metadata: dict[str, Any] = {
        "slide_title": title,
        "slide_count": slide_count,
        "hidden": slide.hidden,
        "has_images": bool(image_blocks),
        "image_count": len(image_blocks),
        "visual_used": visual_used,
        "visual_summary": summary,
        "image_descriptions": descriptions,
        "flow_notes": flow_notes,
        "visual_notes": notes,
        "layout": {"slide_size": dict(slide_size or {}), "blocks": [b.to_dict() for b in slide.blocks]},
    }
    if assets:
        metadata["image_assets"] = [dict(v, block_order=k) for k, v in sorted(assets.items())]

    prov = dict(provenance or {})
    if ungrounded:
        prov["fallbacks"] = list(prov.get("fallbacks") or []) + ["ungrounded_block_order"]
        prov["ungrounded_block_orders"] = ungrounded

    return Chunk(
        doc_id=doc_id,
        source_name=source_name,
        doc_type=DOC_TYPE,
        page=slide.number,
        chunk_index=chunk_index,
        retrieve_content=retrieve_content,
        llm_content=llm_content.strip(),
        filtering_content=filtering_text(retrieve_content),
        metadata=metadata,
        provenance=prov,
    )
