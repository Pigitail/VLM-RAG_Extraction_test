"""Hand-computed checks of every metric, including the edge cases."""

from __future__ import annotations

import pytest

from vlmrag.evaluation.metrics import (
    DocOutcome,
    collect_queries,
    compare_metrics,
    docx_metrics,
    drawing_metrics,
    mean,
    rank_pages,
    ratio,
    retrieval_metrics,
    slides_metrics,
)

from .conftest import make_chunk


def test_ratio_and_mean_conventions():
    assert ratio(1, 3) == 0.333333 and ratio(0, 0) is None and ratio(2, 2) == 1.0
    assert mean([None, 0.5, 1.0]) == 0.75 and mean([None]) is None and mean([]) is None


# --------------------------------------------------------------------------- drawing


def _drawing_outcomes() -> list[DocOutcome]:
    truth = {
        "source": "d.pdf",
        "doc_type": "drawing",
        "pages": [
            {
                "page": 1,
                "title_block": {
                    "title": "LANTERN HOUSING",
                    "revision": "C",
                    "material": "PC-UV STABILISED",
                    "general_tolerance": "ISO 2768-m",
                    "mass": "",  # empty truth: not scored
                },
                "raster_regions": ["revision_table"],
                "text_layer": {"title_block": True},
            },
            {
                "page": 2,
                "title_block": {
                    "title": "SECTIONS",
                    "revision": "C",
                    "material": "PC-UV STABILISED",
                    "general_tolerance": "ISO 2768-m",
                },
                "raster_regions": [],
                "text_layer": {"title_block": True},
            },
        ],
    }
    block = {
        "title": "Lantern  Housing",
        "revision": "C",
        "material": "PC-UV STABILISED",
        "general_tolerance": "ISO 2768–m",
    }
    chunks = [
        make_chunk("d.pdf", 1, metadata={"title_block": block, "title_block_page": 1}),
        make_chunk("d.pdf", 2, index=1, metadata={"title_block": block, "title_block_page": 1}),
    ]
    raster_truth = {
        "source": "r.pdf",
        "doc_type": "drawing",
        "pages": [{
            "page": 1,
            "title_block": {"title": "REFLECTOR", "revision": "B"},
            "raster_regions": ["revision_table", "title_block"],
            "text_layer": {"title_block": False},
        }],
    }
    return [
        DocOutcome("d.pdf", "drawing", truth, chunks),
        DocOutcome("r.pdf", "drawing", raster_truth, [], {"type": "TitleBlockGateError", "message": "0 pairs"}),
        DocOutcome("x.pdf", "drawing", None, [], {"type": "ReplayMiss", "message": "no recorded response"}),
    ]


def test_drawing_fields_groups_and_gate():
    m = drawing_metrics(_drawing_outcomes())
    sheets = {(s["source"], s["page"]): s for s in m["sheets"]}
    p1, p2, r1 = sheets[("d.pdf", 1)], sheets[("d.pdf", 2)], sheets[("r.pdf", 1)]
    assert p1["fields_scored"] == 4 and "mass" not in p1["fields"]
    assert (p1["exact"], p1["normalized"]) == (0.5, 1.0)  # title and tolerance only match normalised
    assert (p2["exact"], p2["normalized"]) == (0.5, 0.75)  # title of sheet 1 inherited
    assert p2["inherited"] and not p1["inherited"] and p1["title_block_page"] == 1
    assert r1["status"] == "error" and r1["error_type"] == "TitleBlockGateError" and r1["raster_title_block"]
    assert (r1["exact"], r1["normalized"]) == (0.0, 0.0)

    groups = m["groups"]
    assert groups["all"] == {
        "sheets": 3, "fields": 10, "exact_correct": 4, "normalized_correct": 7,
        "exact": 0.4, "normalized": 0.7, "sheet_mean_normalized": 0.583333,
    }
    assert (groups["read"]["sheets"], groups["read"]["normalized"]) == (1, 1.0)
    assert (groups["inherited"]["sheets"], groups["inherited"]["normalized"]) == (1, 0.75)
    assert (groups["raster_title_block"]["fields"], groups["raster_title_block"]["normalized"]) == (2, 0.0)
    assert groups["text_layer_title_block"]["normalized"] == 0.875

    assert m["per_field"]["title"] == {
        "sheets": 3, "exact_correct": 0, "normalized_correct": 1, "exact": 0.0, "normalized": 0.333333,
    }
    assert m["per_field"]["general_tolerance"]["exact"] == 0.0
    assert m["per_field"]["general_tolerance"]["normalized"] == 1.0
    assert m["gate"] == {
        "attempted": 3, "passed": 1, "gate_failed": 1, "other_errors": {"ReplayMiss": 1}, "pass_rate": 0.333333,
    }
    assert m["documents_with_truth"] == 2


def test_drawing_empty_inputs_give_na_not_zero():
    m = drawing_metrics([])
    assert m["groups"]["all"]["exact"] is None and m["gate"]["pass_rate"] is None
    empty = DocOutcome("e.pdf", "drawing", {"source": "e.pdf", "pages": []}, [])
    m = drawing_metrics([empty])
    assert m["sheets"] == [] and m["gate"]["pass_rate"] == 1.0


def test_drawing_missing_chunk_for_a_page_counts_as_wrong():
    truth = {"pages": [{"page": 1, "title_block": {"title": "A"}}, {"page": 2, "title_block": {"title": "B"}}]}
    outcome = DocOutcome("t.pdf", "drawing", truth, [make_chunk("t.pdf", 1, metadata={"title_block": {"title": "A"}})])
    m = drawing_metrics([outcome])
    statuses = [s["status"] for s in m["sheets"]]
    assert statuses == ["ok", "no_chunk"] and m["groups"]["all"]["normalized"] == 0.5


# --------------------------------------------------------------------------- slides


def _slides_outcome() -> DocOutcome:
    truth = {
        "source": "deck.pptx",
        "slides": [
            {"slide": 1, "expects_visual_call": False, "image_blocks": [], "must_mention": []},
            {
                "slide": 2,
                "expects_visual_call": True,
                "image_blocks": [
                    {"shape_name": "Screen A", "image_role": "main_screenshot"},
                    {"shape_name": "Detail A", "image_role": "detail_crop"},
                ],
                "must_mention": ["V-2864", "Draft – not yet released"],
            },
            {
                "slide": 3,
                "expects_visual_call": True,
                "image_blocks": [{"shape_name": "Icon", "image_role": "ui_icon"}],
                "must_mention": ["Overdue orders: 4"],
            },
        ],
    }
    chunks = [
        make_chunk("deck.pptx", 1, "intro", doc_type="slides"),
        make_chunk("deck.pptx", 2, "vendor V-2864 entered", doc_type="slides", index=1, metadata={
            "image_descriptions": [
                {"block_order": 3, "shape_name": "Screen A", "image_role": "main_screenshot"},
                {"block_order": 7, "image_role": "inline_image"},  # ungrounded
            ]}, provenance={"visual_call": {"status": "ok"}}),
        make_chunk("deck.pptx", 2, "status: DRAFT - not yet  released", doc_type="slides", index=2, metadata={
            "image_descriptions": [
                {"block_order": 3, "shape_name": "Screen A", "image_role": "inline_image"},  # duplicate
                {"block_order": 4, "shape_name": "Detail A", "image_role": "main_screenshot"},  # wrong role
                {"block_order": 5, "shape_name": "Arrow 1", "image_role": "main_screenshot"},  # not in truth
            ]}),
        make_chunk("deck.pptx", 3, "Overdue orders 4", doc_type="slides", index=3,
                   provenance={"visual_call": {"status": "empty"}}),
    ]
    return DocOutcome("deck.pptx", "slides", truth, chunks)


def test_slides_roles_mentions_and_hallucinations():
    m = slides_metrics([_slides_outcome()])
    assert m["image_role"] == {
        "blocks": 3,
        "described": 2,
        "correct": 1,
        "accuracy": 0.333333,
        "confusion": {
            "detail_crop": {"main_screenshot": 1},
            "main_screenshot": {"main_screenshot": 1},
            "ui_icon": {"missing": 1},
        },
    }
    # Slide 2: both facts found across its two chunks; slide 3: the colon is missing -> not found.
    assert m["must_mention"] == {"facts": 3, "found": 2, "recall": 0.666667, "slide_mean_recall": 0.5}
    assert m["hallucinated_blocks"] == {"total": 3, "ungrounded": 1, "not_in_truth": 1, "duplicate": 1}
    assert m["visual_calls"] == {"expected": 2, "ok": 1}
    slide2 = next(r for r in m["per_slide"] if r["slide"] == 2)
    assert slide2["role_correct"] == 1 and slide2["hallucinated"]["total"] == 3


def test_slides_failed_document_and_empty_truth():
    outcome = _slides_outcome()
    failed = DocOutcome(outcome.source, "slides", outcome.truth, [], {"type": "ValueError", "message": "bad"})
    m = slides_metrics([failed])
    assert m["failed_documents"] == 1
    assert m["image_role"]["accuracy"] == 0.0 and m["must_mention"]["recall"] == 0.0
    assert m["image_role"]["confusion"]["ui_icon"] == {"missing": 1}
    m = slides_metrics([DocOutcome("x.pptx", "slides", {"slides": []}, [])])
    assert m["image_role"]["accuracy"] is None and m["must_mention"]["recall"] is None


# --------------------------------------------------------------------------- docx


def _docx_outcomes() -> list[DocOutcome]:
    truth = {
        "source": "s.docx",
        "page_count": 3,
        "pages": [
            {"page": 1, "figures": ["fig1"], "must_mention": ["glass globe", "LED module"]},
            {"page": 2, "figures": [], "must_mention": []},
            {"page": 3, "figures": ["fig2"], "must_mention": ["5 V / 2 A"]},
        ],
        "tables": [
            {"table_id": "t1", "multi_page": True, "row_pages": [1, 1, 2, 2, 2]},
            {"table_id": None, "multi_page": False, "row_pages": [3, 3]},
        ],
    }
    split = {"page_split": {"xml_pages": 3, "padded_pages": 0}}
    chunks = [
        make_chunk("s.docx", 1, "Glass  Globe and LED-module", doc_type="docx", provenance=split, metadata={
            "multi_page_tables": [{"table_id": "t1", "rows": [0, 1, 2], "assignment": "vlm"}]}),
        make_chunk("s.docx", 2, "rows", doc_type="docx", index=1, provenance=split, metadata={
            "multi_page_tables": [{"table_id": "t1", "rows": [2, 3, 9], "assignment": "vlm"}]}),
        make_chunk("s.docx", 3, "MAX 5 V / 2 A", doc_type="docx", index=2, provenance=split),
    ]
    failed_truth = {
        "source": "e.docx",
        "page_count": 2,
        "pages": [{"page": 1, "must_mention": ["carry hook"]}],
        "tables": [{"table_id": "t1", "multi_page": True, "row_pages": [1, 2]}],
    }
    return [
        DocOutcome("s.docx", "docx", truth, chunks),
        DocOutcome("e.docx", "docx", failed_truth, [], {"type": "DocxExtractionError", "message": "x"}),
    ]


def test_docx_pages_rows_and_mentions():
    m = docx_metrics(_docx_outcomes())
    assert m["page_count"] == {"documents": 2, "correct": 1, "accuracy": 0.5}
    first = m["per_table"][0]
    # Row 0, 1, 3 on their page; row 2 listed on two pages; row 4 missing; row 9 does not exist.
    assert first == {
        "source": "s.docx", "table_id": "t1", "rows": 5, "correct": 3, "accuracy": 0.6,
        "missing": 1, "on_several_pages": 1, "unexpected_rows": 1, "assignment": {"vlm": 2},
    }
    # The failed document's two rows are all wrong; the one-page table is not scored.
    assert len(m["per_table"]) == 2
    assert m["row_page"] == {"tables": 2, "rows": 7, "correct": 3, "accuracy": 0.428571}
    # "LED module" is not "LED-module"; the failed document's fact is not found.
    assert m["image_must_mention"] == {
        "pages": 3, "facts": 4, "found": 2, "recall": 0.5, "page_mean_recall": 0.5,
    }
    assert [d["predicted_pages"] for d in m["per_document"]] == [3, None]


def test_docx_page_count_falls_back_to_the_highest_chunk_page():
    truth = {"page_count": 4, "pages": [], "tables": []}
    chunks = [make_chunk("n.docx", 1, "a", doc_type="docx"), make_chunk("n.docx", 4, "b", doc_type="docx", index=1)]
    m = docx_metrics([DocOutcome("n.docx", "docx", truth, chunks)])
    assert m["page_count"]["accuracy"] == 1.0 and m["row_page"]["accuracy"] is None


# --------------------------------------------------------------------------- retrieval


def test_rank_pages_collapses_duplicates_drops_zero_and_breaks_ties_pessimistically():
    keys = [("a", 1), ("a", 1), ("b", 1), ("c", 1)]
    assert rank_pages([1.0, 3.0, 0.0, 3.0], keys, {("a", 1)}) == [("c", 1), ("a", 1)]
    assert rank_pages([1.0, 3.0, 0.0, 3.0], keys, {("c", 1)}) == [("a", 1), ("c", 1)]
    assert rank_pages([0.0, 0.0, 0.0, 0.0], keys, set()) == []


def _query(text: str, *relevant: tuple[str, int], doc_type: str = "drawing") -> dict:
    return {"query": text, "relevant": [{"source": s, "page": p} for s, p in relevant], "doc_type": doc_type}


def test_retrieval_recall_mrr_duplicates_and_several_relevant_pages():
    chunks = [
        make_chunk("A.pdf", 1, "red lantern housing"),
        make_chunk("A.pdf", 1, "red lantern", index=1),  # duplicate page
        make_chunk("B.pdf", 2, "blue reflector cone"),
        make_chunk("C.pdf", 1, "unrelated words here"),
    ]
    queries = [
        _query("red lantern", ("A.pdf", 1)),
        _query("reflector housing", ("A.pdf", 1), ("B.pdf", 2)),
        _query("nothing matches", ("C.pdf", 1), doc_type="slides"),
        _query("no relevance given"),
    ]
    m = retrieval_metrics(chunks, queries)
    rows = {r["query"]: r for r in m["per_query"]}
    first = rows["red lantern"]
    assert first["first_relevant_rank"] == 1 and first["recall"] == {"@1": 1.0, "@3": 1.0, "@5": 1.0}
    assert first["top"][0] == "A.pdf#p1" and len(set(first["top"])) == len(first["top"])
    both = rows["reflector housing"]
    assert both["recall"]["@1"] == 0.5 and both["recall"]["@3"] == 1.0 and both["reciprocal_rank"] == 1.0
    none = rows["nothing matches"]
    assert none["first_relevant_rank"] is None and none["reciprocal_rank"] == 0.0 and none["recall"]["@5"] == 0.0
    assert m["skipped_queries"] == ["no relevance given"]
    assert m["overall"] == {"queries": 3, "recall@1": 0.5, "recall@3": 0.666667, "recall@5": 0.666667, "mrr": 0.666667}
    assert m["by_doc_type"]["slides"]["queries"] == 1 and m["by_doc_type"]["drawing"]["mrr"] == 1.0
    assert m["index"]["chunks"] == 4 and m["index"]["pages"] == 3


def test_retrieval_ties_do_not_depend_on_chunk_order():
    x = make_chunk("X.pdf", 1, "alpha beta")
    y = make_chunk("Y.pdf", 1, "alpha beta")
    query = [_query("alpha", ("Y.pdf", 1))]
    for chunks in ([x, y], [y, x]):
        row = retrieval_metrics(chunks, query)["per_query"][0]
        assert row["first_relevant_rank"] == 2 and row["recall"]["@1"] == 0.0 and row["reciprocal_rank"] == 0.5


def test_retrieval_hand_computed_scores_duplicate_relevance_and_short_rankings():
    # N = 4, lengths 2, 1, 3, 1 -> avgdl 1.75; idf(alpha) = ln 2 (df 2), idf(gamma) = ln(10/3) (df 1).
    # "alpha gamma": C = ln(10/3) * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 3 / 1.75)) = 1.3785
    #                B = ln 2 * 2.2 / (1 + 1.2 * (0.25 + 0.75 / 1.75))            = 0.8405
    #                A = ln 2 * 2.2 / (1 + 1.2 * (0.25 + 0.75 * 2 / 1.75))        = 0.6549
    #                D = 0 (not retrieved)
    chunks = [
        make_chunk("A.pdf", 1, "alpha beta"),
        make_chunk("B.pdf", 1, "alpha"),
        make_chunk("C.pdf", 2, "beta gamma gamma"),
        make_chunk("D.pdf", 1, "delta"),
    ]
    queries = [
        _query("alpha gamma", ("C.pdf", 2), ("D.pdf", 1), ("D.pdf", 1)),  # duplicate relevance entry counts once
        _query("alpha", ("A.pdf", 1)),
    ]
    m = retrieval_metrics(chunks, queries)
    first, second = m["per_query"]
    assert first["top"] == ["C.pdf#p2", "B.pdf#p1", "A.pdf#p1"]  # only 3 pages score > 0
    assert first["relevant"] == ["C.pdf#p2", "D.pdf#p1"]
    assert first["recall"] == {"@1": 0.5, "@3": 0.5, "@5": 0.5} and first["reciprocal_rank"] == 1.0
    assert second["top"] == ["B.pdf#p1", "A.pdf#p1"] and second["first_relevant_rank"] == 2
    assert second["recall"] == {"@1": 0.0, "@3": 1.0, "@5": 1.0} and second["reciprocal_rank"] == 0.5
    assert m["overall"] == {"queries": 2, "recall@1": 0.25, "recall@3": 0.75, "recall@5": 0.75, "mrr": 0.75}


def test_retrieval_relevance_is_file_name_and_source_page():
    # The matching text sits on page 2 of A.pdf: page numbers are not interchangeable between pages or files.
    chunks = [make_chunk("A.pdf", 1, "intro"), make_chunk("A.pdf", 2, "vent hole"), make_chunk("B.pdf", 1, "vent")]
    row = retrieval_metrics(chunks, [_query("vent hole", ("A.pdf", 2))])["per_query"][0]
    assert row["first_relevant_rank"] == 1
    row = retrieval_metrics(chunks, [_query("vent hole", ("A.pdf", 1))])["per_query"][0]
    assert row["first_relevant_rank"] is None and row["recall"]["@5"] == 0.0  # page 1 does not mention it


def test_drawing_uses_the_first_chunk_of_a_page():
    truth = {"pages": [{"page": 1, "title_block": {"title": "B"}}]}
    chunks = [make_chunk("t.pdf", 1, metadata={"title_block": {"title": "A"}}),
              make_chunk("t.pdf", 1, index=1, metadata={"title_block": {"title": "B"}})]
    m = drawing_metrics([DocOutcome("t.pdf", "drawing", truth, chunks)])
    assert m["sheets"][0]["fields"]["title"]["predicted"] == "A" and m["groups"]["all"]["normalized"] == 0.0


def test_retrieval_without_chunks_or_queries():
    m = retrieval_metrics([], [_query("anything", ("A.pdf", 1))])
    assert m["overall"]["recall@1"] == 0.0 and m["overall"]["mrr"] == 0.0
    m = retrieval_metrics([make_chunk("A.pdf", 1, "x")], [])
    assert m["overall"] == {"queries": 0, "recall@1": None, "recall@3": None, "recall@5": None, "mrr": None}


def test_collect_queries_tags_doc_type():
    truth = {"queries": [{"query": "q", "relevant": [{"source": "a.pdf", "page": "2"}]}]}
    [query] = collect_queries([DocOutcome("a.pdf", "drawing", truth)])
    assert query == {"query": "q", "relevant": [{"source": "a.pdf", "page": 2}], "doc_type": "drawing", "source": "a.pdf"}


# --------------------------------------------------------------------------- comparison


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ({"x": 1.0, "y": [1, 2]}, {"x": 1.0 + 1e-12, "y": [1, 2]}, []),
        ({"x": 0.5}, {"x": 0.6}, ["x: 0.5 != 0.6"]),
        ({"x": {"y": True}}, {"x": {"y": 1}}, ["x.y: True != 1"]),
        ({"x": [1]}, {"x": [1, 2]}, ["x: 1 item(s) != 2 item(s)"]),
        ({"x": 1}, {"z": 1}, ["x: missing in actual", "z: not in expected"]),
        ({"s": None}, {"s": "a"}, ["s: None != 'a'"]),
    ],
)
def test_compare_metrics(a, b, expected):
    assert compare_metrics(a, b) == expected
