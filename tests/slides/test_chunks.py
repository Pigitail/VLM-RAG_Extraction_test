from __future__ import annotations

import re

from vlmrag.techniques.slides.chunks import build_slide_chunk, enrich_descriptions, has_content
from vlmrag.techniques.slides.layout import Block, PictureInfo, SlideLayout
from vlmrag.techniques.slides.parse import SlideVisual


def _slide() -> SlideLayout:
    return SlideLayout(
        number=4,
        title="Release the order",
        text_only="- Release the order\nStatus | Meaning\nDraft | Saved",
        text_and_table="- Release the order\n| Status | Meaning |\n| --- | --- |\n| Draft | Saved |",
        blocks=[
            Block(order=1, type="text", shape_name="Title 1", text_content="- Release the order"),
            Block(
                order=2,
                type="image",
                shape_id="9",
                shape_name="Screen B",
                bbox_norm={"x": 0.5, "y": 0.2, "w": 0.4, "h": 0.5},
                picture=PictureInfo(blob=b"x", ext="png", source_sha256="cd" * 32, crop=(0.1, 0.0, 0.0, 0.0)),
                image_index=1,
            ),
            Block(order=3, type="shape", shape_name="Arrow", shape_type_name="LINE"),
        ],
    )


def _visual(**overrides) -> SlideVisual:
    visual = SlideVisual(
        visual_summary="List screen with the release dialog.",
        image_descriptions=[{
            "block_order": 2,
            "related_block_orders": [1, 3],
            "description": "Order  list screen;\n order PO-1001 is selected.",
            "description_en": "Order list with PO-1001 selected.",
            "description_ko": "PO-1001이 선택된 오더 목록입니다.",
            "confidence": "medium",
            "image_role": "main_screenshot",
        }],
        flow_notes=["Arrow from the step text to the list.", "arrow from the step text to the list."],
        notes="Dialog partly covered.",
    )
    for key, value in overrides.items():
        setattr(visual, key, value)
    return visual


def test_chunk_with_visual_grounding():
    chunk = build_slide_chunk(
        doc_id="d1", source_name="manual.pptx", slide=_slide(), slide_count=6, chunk_index=2, visual=_visual(),
        assets={2: {"asset": "manual/slide_004/img_001.png", "ext": "png", "cropped": True}},
        slide_size={"width": 100, "height": 50}, provenance={"fallbacks": ["x"]},
    )
    assert (chunk.page, chunk.chunk_index, chunk.doc_type) == (4, 2, "slides")
    retrieve = chunk.retrieve_content.splitlines()
    assert retrieve[0] == "slide=4 | title=Release the order"
    assert "Draft | Saved" in chunk.retrieve_content and "| --- |" not in chunk.retrieve_content
    cues = retrieve[retrieve.index("[Visual image retrieval cues]") + 1:]
    assert cues == [
        "Visual summary: List screen with the release dialog.",
        "Image detailed description: Order list screen; order PO-1001 is selected.",
        "Image role: main_screenshot",
        "Image display summary EN: Order list with PO-1001 selected.",
        "Image display summary KO: PO-1001이 선택된 오더 목록입니다.",
    ]
    llm = chunk.llm_content
    assert llm.startswith("[Source]\nsource: manual.pptx\nslide: 4\ntitle: Release the order\n\n")
    assert "| --- | --- |" in llm
    assert (
        "- image block 2 (related blocks: 1, 3): description=Order list screen; order PO-1001 is selected.; "
        "summary_en=Order list with PO-1001 selected.; summary_ko=PO-1001이 선택된 오더 목록입니다.; "
        "image_role=main_screenshot [confidence=medium]"
    ) in llm
    assert llm.count("- flow: ") == 1 and "- notes: Dialog partly covered." in llm
    assert chunk.filtering_content == re.sub(r"\s+", "", chunk.retrieve_content)


def test_metadata_is_enriched_with_block_data():
    chunk = build_slide_chunk(
        doc_id="d1", source_name="manual.pptx", slide=_slide(), slide_count=6, chunk_index=0, visual=_visual(),
        assets={2: {"asset": "manual/slide_004/img_001.png"}},
    )
    meta = chunk.metadata
    assert meta["visual_used"] and meta["has_images"] and meta["image_count"] == 1
    (item,) = meta["image_descriptions"]
    assert item["shape_name"] == "Screen B" and item["shape_id"] == "9"
    assert item["bbox_norm"] == {"x": 0.5, "y": 0.2, "w": 0.4, "h": 0.5}
    assert item["crop_norm"] == {"l": 0.1, "t": 0.0, "r": 0.0, "b": 0.0}
    assert item["source_image_id"] == "cdcdcdcdcdcd" and item["asset"] == "manual/slide_004/img_001.png"
    assert meta["image_assets"] == [{"asset": "manual/slide_004/img_001.png", "block_order": 2}]
    assert [b["order"] for b in meta["layout"]["blocks"]] == [1, 2, 3]
    assert "fallbacks" not in chunk.provenance


def test_ungrounded_block_order_is_kept_but_flagged():
    visual = _visual()
    visual.image_descriptions[0]["block_order"] = 3  # a shape block, not an image block
    descriptions, ungrounded = enrich_descriptions(_slide(), visual)
    assert ungrounded == [3] and "shape_name" not in descriptions[0]
    chunk = build_slide_chunk(doc_id="d", source_name="m.pptx", slide=_slide(), slide_count=6, chunk_index=0,
                              visual=visual, provenance={"fallbacks": ["vlm_error"]})
    assert chunk.provenance["fallbacks"] == ["vlm_error", "ungrounded_block_order"]
    assert chunk.provenance["ungrounded_block_orders"] == [3]


def test_text_only_chunk_without_visual():
    chunk = build_slide_chunk(doc_id="d", source_name="m.pptx", slide=_slide(), slide_count=6, chunk_index=0)
    assert "[Visual image retrieval cues]" not in chunk.retrieve_content
    assert "[Grounded visual description]" not in chunk.llm_content
    assert chunk.metadata["visual_used"] is False and chunk.metadata["image_descriptions"] == []


def test_summary_only_answer_still_counts_as_visual():
    chunk = build_slide_chunk(doc_id="d", source_name="m.pptx", slide=_slide(), slide_count=6, chunk_index=0,
                              visual=SlideVisual(visual_summary="Only a summary."))
    assert "Visual summary: Only a summary." in chunk.retrieve_content
    assert chunk.metadata["visual_used"] is True


def test_model_text_is_unicode_normalised():
    visual = _visual(visual_summary="Ｏｒｄｅｒ　list — “draft” rows")
    visual.image_descriptions[0]["description"] = "Order ＰＯ‐１００１ is selected"
    visual.flow_notes = ["Arrow – step 1", "arrow - step 1"]
    chunk = build_slide_chunk(doc_id="d", source_name="m.pptx", slide=_slide(), slide_count=6, chunk_index=0,
                              visual=visual)
    meta = chunk.metadata
    assert meta["visual_summary"] == 'Order list - "draft" rows'
    assert meta["image_descriptions"][0]["description"] == "Order PO-1001 is selected"
    assert meta["flow_notes"] == ["Arrow - step 1"]  # variants collapse after normalisation
    assert "PO-1001" in chunk.filtering_content


def test_visual_payload_matches_chunk_metadata():
    from vlmrag.techniques.slides.chunks import payload_is_empty, visual_payload

    assets = {2: {"asset": "m-0000/slide_004/img_001.png"}}
    payload, ungrounded = visual_payload(_slide(), _visual(), assets)
    chunk = build_slide_chunk(doc_id="d", source_name="m.pptx", slide=_slide(), slide_count=6, chunk_index=0,
                              visual=_visual(), assets=assets)
    assert ungrounded == []
    assert payload["image_descriptions"] == chunk.metadata["image_descriptions"]
    assert payload["visual_summary"] == chunk.metadata["visual_summary"]
    assert payload["flow_notes"] == chunk.metadata["flow_notes"] and payload["notes"] == chunk.metadata["visual_notes"]
    assert payload_is_empty(visual_payload(_slide(), SlideVisual(visual_summary="  "))[0])


def test_has_content():
    assert not has_content(SlideLayout(number=1), None)
    assert has_content(SlideLayout(number=1, text_only="- x"), None)
    assert has_content(_slide(), SlideVisual())
    assert has_content(SlideLayout(number=1), SlideVisual(notes="n"))
