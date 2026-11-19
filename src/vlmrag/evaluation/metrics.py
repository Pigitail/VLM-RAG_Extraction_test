"""Pure metric functions: ground truth + emitted chunks -> numbers.

Every function here is deterministic and model-free; it reads only the truth
files written by :mod:`vlmrag.synthetic` and the :class:`~vlmrag.schema.Chunk`
objects a technique returned. The exact definitions are in
:data:`METRIC_DEFINITIONS` (copied into ``results.json`` and
``docs/evaluation.md``), so a number in a README can always be traced back to
the sentence that defines it.

Conventions shared by all metrics

* A document that failed is a data point, not a gap: its truth items are
  scored as wrong / not found (with an empty prediction), and the failure is
  listed with its exception type.
* Ratios are rounded to 6 decimals; a ratio with an empty denominator is
  ``None`` (shown as ``n/a``), never 0 or 1.
* "Normalised" comparisons use :func:`vlmrag.evaluation.textnorm.normalize_value`.
* Pages are 1-based source pages, as in ``Chunk.page``. When several chunks
  share a page, their texts and descriptions are combined for that page.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from .bm25 import B, BM25, K1
from .textnorm import NORMALIZATION, exact_match, mentions, normalize_value, normalized_match

__all__ = [
    "DocOutcome",
    "METRIC_DEFINITIONS",
    "RETRIEVAL_KS",
    "ratio",
    "mean",
    "drawing_sheets",
    "drawing_metrics",
    "slides_metrics",
    "docx_metrics",
    "collect_queries",
    "rank_pages",
    "retrieval_metrics",
    "compare_metrics",
]

RETRIEVAL_KS = (1, 3, 5)
GATE_ERROR = "TitleBlockGateError"


@dataclass
class DocOutcome:
    """What one technique run produced for one document.

    ``error`` is ``None`` on success, else ``{"type": <exception class>, "message": ...}``;
    ``chunks`` is then empty. ``truth`` is the parsed ``<stem>.truth.json`` or
    ``None`` (documents without truth still take part in retrieval as distractors).
    """

    source: str
    technique: str
    truth: Optional[dict] = None
    chunks: list = field(default_factory=list)
    error: Optional[dict] = None
    wall_time_s: float = 0.0
    usage: Optional[dict] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# --------------------------------------------------------------------------- helpers


def ratio(numerator: float, denominator: float) -> Optional[float]:
    """``numerator / denominator`` rounded to 6 decimals, ``None`` for an empty denominator."""
    if not denominator:
        return None
    return round(numerator / denominator, 6)


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    """Mean of the non-``None`` values, rounded to 6 decimals; ``None`` if there are none."""
    items = [float(v) for v in values if v is not None]
    if not items:
        return None
    return round(math.fsum(items) / len(items), 6)


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _chunks_by_page(chunks: Iterable[Any]) -> dict[int, list]:
    pages: dict[int, list] = defaultdict(list)
    for chunk in chunks:
        pages[int(chunk.page)].append(chunk)
    return pages


def _page_text(chunks: Sequence[Any]) -> str:
    return "\n".join(_text(getattr(c, "retrieve_content", "")) for c in chunks)


def _error_type(outcome: DocOutcome) -> Optional[str]:
    return (outcome.error or {}).get("type") if outcome.error else None


# --------------------------------------------------------------------------- drawing


def drawing_sheets(outcomes: Sequence[DocOutcome]) -> list[dict]:
    """One record per truth sheet (page) of every drawing document with truth.

    The prediction for a sheet is ``metadata["title_block"]`` of the first chunk
    on that page (the merged canonical fields). ``inherited`` marks sheets whose
    chunk carries a title block read on another page (the technique's default
    ``title_block_pages="first"`` mode); ``raster_title_block`` marks sheets whose
    truth says the title block exists only as pixels.
    """
    sheets: list[dict] = []
    for outcome in outcomes:
        if outcome.technique != "drawing" or not outcome.truth:
            continue
        by_page = _chunks_by_page(outcome.chunks)
        for truth_page in outcome.truth.get("pages") or []:
            page = int(truth_page["page"])
            chunk = (by_page.get(page) or [None])[0]
            metadata = (chunk.metadata or {}) if chunk is not None else {}
            predicted = metadata.get("title_block") or {}
            read_from = metadata.get("title_block_page")
            raster = sorted(truth_page.get("raster_regions") or [])
            text_layer = truth_page.get("text_layer") or {}
            fields: dict[str, dict] = {}
            for key, truth_value in (truth_page.get("title_block") or {}).items():
                if not _text(truth_value).strip():
                    continue  # nothing to read: not scored
                value = _text(predicted.get(key))
                fields[key] = {
                    "truth": truth_value,
                    "predicted": value,
                    "exact": exact_match(value, truth_value),
                    "normalized": normalized_match(value, truth_value),
                }
            if chunk is not None:
                status = "ok"
            else:
                status = "error" if outcome.error else "no_chunk"
            count = len(fields)
            sheets.append({
                "source": outcome.source,
                "page": page,
                "status": status,
                "error_type": _error_type(outcome),
                "title_block_page": read_from,
                "inherited": chunk is not None and read_from is not None and int(read_from) != page,
                "raster_regions": raster,
                "raster_title_block": "title_block" in raster or text_layer.get("title_block") is False,
                "fields_scored": count,
                "exact": ratio(sum(f["exact"] for f in fields.values()), count),
                "normalized": ratio(sum(f["normalized"] for f in fields.values()), count),
                "fields": fields,
            })
    return sheets


def _sheet_group(sheets: Sequence[dict]) -> dict:
    fields = [f for sheet in sheets for f in sheet["fields"].values()]
    exact = sum(f["exact"] for f in fields)
    normalized = sum(f["normalized"] for f in fields)
    return {
        "sheets": len(sheets),
        "fields": len(fields),
        "exact_correct": exact,
        "normalized_correct": normalized,
        "exact": ratio(exact, len(fields)),
        "normalized": ratio(normalized, len(fields)),
        "sheet_mean_normalized": mean(s["normalized"] for s in sheets),
    }


def drawing_metrics(outcomes: Sequence[DocOutcome]) -> dict:
    """Title-block field accuracy, per-sheet accuracy and gate pass rate of one drawing run."""
    docs = [o for o in outcomes if o.technique == "drawing"]
    sheets = drawing_sheets(docs)
    groups = {
        "all": sheets,
        "read": [s for s in sheets if s["status"] == "ok" and not s["inherited"]],
        "inherited": [s for s in sheets if s["inherited"]],
        "raster_title_block": [s for s in sheets if s["raster_title_block"]],
        "text_layer_title_block": [s for s in sheets if not s["raster_title_block"]],
    }
    keys: list[str] = []
    for sheet in sheets:
        for key in sheet["fields"]:
            if key not in keys:
                keys.append(key)
    per_field = {}
    for key in keys:
        scored = [s["fields"][key] for s in sheets if key in s["fields"]]
        per_field[key] = {
            "sheets": len(scored),
            "exact_correct": sum(f["exact"] for f in scored),
            "normalized_correct": sum(f["normalized"] for f in scored),
            "exact": ratio(sum(f["exact"] for f in scored), len(scored)),
            "normalized": ratio(sum(f["normalized"] for f in scored), len(scored)),
        }
    errors = Counter(_error_type(o) for o in docs if o.error)
    gate_failed = errors.pop(GATE_ERROR, 0)
    passed = sum(1 for o in docs if o.ok)
    return {
        "documents": len(docs),
        "documents_with_truth": sum(1 for o in docs if o.truth),
        "gate": {
            "attempted": len(docs),
            "passed": passed,
            "gate_failed": gate_failed,
            "other_errors": dict(sorted(errors.items())),
            "pass_rate": ratio(passed, len(docs)),
        },
        "groups": {name: _sheet_group(group) for name, group in groups.items()},
        "per_field": per_field,
        "sheets": sheets,
    }


# --------------------------------------------------------------------------- slides


def slides_metrics(outcomes: Sequence[DocOutcome]) -> dict:
    """Image-role accuracy, must-mention recall and hallucinated blocks of one slides run."""
    docs = [o for o in outcomes if o.technique == "slides"]
    rows: list[dict] = []
    confusion: dict[str, Counter] = defaultdict(Counter)
    for outcome in docs:
        if not outcome.truth:
            continue
        by_page = _chunks_by_page(outcome.chunks)
        for truth_slide in outcome.truth.get("slides") or []:
            number = int(truth_slide["slide"])
            here = by_page.get(number, [])
            descriptions = [d for c in here for d in ((c.metadata or {}).get("image_descriptions") or [])]
            text = _page_text(here)
            truth_roles = {b["shape_name"]: b["image_role"] for b in truth_slide.get("image_blocks") or []}
            matched: dict[str, dict] = {}
            ungrounded = not_in_truth = duplicate = 0
            for item in descriptions:
                name = item.get("shape_name")
                if not name:
                    ungrounded += 1
                elif name not in truth_roles:
                    not_in_truth += 1
                elif name in matched:
                    duplicate += 1
                else:
                    matched[name] = item
            blocks = []
            for name, role in truth_roles.items():
                predicted = (matched.get(name) or {}).get("image_role") or None
                blocks.append({
                    "shape_name": name,
                    "truth_role": role,
                    "predicted_role": predicted,
                    "described": name in matched,
                    "correct": predicted == role,
                })
                confusion[role][predicted or "missing"] += 1
            facts = [
                {"fact": fact, "found": mentions(text, fact)}
                for fact in truth_slide.get("must_mention") or []
                if normalize_value(fact)
            ]
            visual = [((c.provenance or {}).get("visual_call") or {}).get("status") for c in here]
            rows.append({
                "source": outcome.source,
                "slide": number,
                "chunk": bool(here),
                "error_type": _error_type(outcome),
                "expects_visual_call": bool(truth_slide.get("expects_visual_call")),
                "visual_call_status": next((s for s in visual if s), None),
                "image_blocks": blocks,
                "role_correct": sum(b["correct"] for b in blocks),
                "must_mention": facts,
                "must_mention_recall": ratio(sum(f["found"] for f in facts), len(facts)),
                "hallucinated": {
                    "total": ungrounded + not_in_truth + duplicate,
                    "ungrounded": ungrounded,
                    "not_in_truth": not_in_truth,
                    "duplicate": duplicate,
                },
            })
    blocks_total = sum(len(r["image_blocks"]) for r in rows)
    facts_all = [f for r in rows for f in r["must_mention"]]
    expected = [r for r in rows if r["expects_visual_call"]]
    hallucinated = Counter()
    for row in rows:
        hallucinated.update(row["hallucinated"])
    return {
        "documents": len(docs),
        "failed_documents": sum(1 for o in docs if not o.ok),
        "slides": len(rows),
        "image_role": {
            "blocks": blocks_total,
            "described": sum(b["described"] for r in rows for b in r["image_blocks"]),
            "correct": sum(r["role_correct"] for r in rows),
            "accuracy": ratio(sum(r["role_correct"] for r in rows), blocks_total),
            "confusion": {role: dict(sorted(counts.items())) for role, counts in sorted(confusion.items())},
        },
        "must_mention": {
            "facts": len(facts_all),
            "found": sum(f["found"] for f in facts_all),
            "recall": ratio(sum(f["found"] for f in facts_all), len(facts_all)),
            "slide_mean_recall": mean(r["must_mention_recall"] for r in rows),
        },
        "hallucinated_blocks": {k: int(hallucinated.get(k, 0)) for k in ("total", "ungrounded", "not_in_truth", "duplicate")},
        "visual_calls": {
            "expected": len(expected),
            "ok": sum(1 for r in expected if r["visual_call_status"] == "ok"),
        },
        "per_slide": rows,
    }


# --------------------------------------------------------------------------- docx


def _docx_page_count(chunks: Sequence[Any]) -> Optional[int]:
    """Pages the technique split the document into (``provenance.page_split``), else the highest chunk page."""
    for chunk in chunks:
        split = (chunk.provenance or {}).get("page_split") or {}
        if "xml_pages" in split:
            return int(split.get("xml_pages") or 0) + int(split.get("padded_pages") or 0)
    pages = [int(c.page) for c in chunks]
    return max(pages) if pages else None


def docx_metrics(outcomes: Sequence[DocOutcome]) -> dict:
    """Page-count correctness, row->page accuracy and image must-mention recall of one docx run."""
    docs = [o for o in outcomes if o.technique == "docx"]
    documents: list[dict] = []
    tables: list[dict] = []
    pages: list[dict] = []
    for outcome in docs:
        if not outcome.truth:
            continue
        truth = outcome.truth
        predicted = _docx_page_count(outcome.chunks)
        documents.append({
            "source": outcome.source,
            "error_type": _error_type(outcome),
            "truth_pages": truth.get("page_count"),
            "predicted_pages": predicted,
            "page_count_correct": predicted is not None and predicted == truth.get("page_count"),
        })
        for table in truth.get("tables") or []:
            row_pages = table.get("row_pages") or []
            if not table.get("multi_page") or not row_pages:
                continue  # rows of a one-page table cannot be misplaced
            listed: dict[int, set] = defaultdict(set)
            sources: Counter = Counter()
            for chunk in outcome.chunks:
                for entry in (chunk.metadata or {}).get("multi_page_tables") or []:
                    if entry.get("table_id") != table.get("table_id"):
                        continue
                    sources[entry.get("assignment") or "unknown"] += 1
                    for index in entry.get("rows") or []:
                        listed[int(index)].add(int(chunk.page))
            correct = sum(1 for i, page in enumerate(row_pages) if listed.get(i) == {int(page)})
            tables.append({
                "source": outcome.source,
                "table_id": table.get("table_id"),
                "rows": len(row_pages),
                "correct": correct,
                "accuracy": ratio(correct, len(row_pages)),
                "missing": sum(1 for i in range(len(row_pages)) if i not in listed),
                "on_several_pages": sum(1 for i in range(len(row_pages)) if len(listed.get(i, ())) > 1),
                "unexpected_rows": sum(1 for i in listed if i < 0 or i >= len(row_pages)),
                "assignment": dict(sorted(sources.items())),
            })
        by_page = _chunks_by_page(outcome.chunks)
        for truth_page in truth.get("pages") or []:
            facts_raw = [f for f in truth_page.get("must_mention") or [] if normalize_value(f)]
            if not facts_raw:
                continue
            text = _page_text(by_page.get(int(truth_page["page"]), []))
            facts = [{"fact": fact, "found": mentions(text, fact)} for fact in facts_raw]
            pages.append({
                "source": outcome.source,
                "page": int(truth_page["page"]),
                "figures": list(truth_page.get("figures") or []),
                "must_mention": facts,
                "recall": ratio(sum(f["found"] for f in facts), len(facts)),
            })
    rows_total = sum(t["rows"] for t in tables)
    facts_all = [f for p in pages for f in p["must_mention"]]
    return {
        "documents": len(docs),
        "failed_documents": sum(1 for o in docs if not o.ok),
        "page_count": {
            "documents": len(documents),
            "correct": sum(1 for d in documents if d["page_count_correct"]),
            "accuracy": ratio(sum(1 for d in documents if d["page_count_correct"]), len(documents)),
        },
        "row_page": {
            "tables": len(tables),
            "rows": rows_total,
            "correct": sum(t["correct"] for t in tables),
            "accuracy": ratio(sum(t["correct"] for t in tables), rows_total),
        },
        "image_must_mention": {
            "pages": len(pages),
            "facts": len(facts_all),
            "found": sum(f["found"] for f in facts_all),
            "recall": ratio(sum(f["found"] for f in facts_all), len(facts_all)),
            "page_mean_recall": mean(p["recall"] for p in pages),
        },
        "per_document": documents,
        "per_table": tables,
        "per_page": pages,
    }


# --------------------------------------------------------------------------- retrieval


def collect_queries(outcomes: Sequence[DocOutcome]) -> list[dict]:
    """The ``queries`` of every truth file in ``outcomes``, tagged with their document type."""
    queries = []
    for outcome in outcomes:
        for query in (outcome.truth or {}).get("queries") or []:
            queries.append({
                "query": query.get("query", ""),
                "relevant": [
                    {"source": r["source"], "page": int(r["page"])} for r in query.get("relevant") or []
                ],
                "doc_type": outcome.technique,
                "source": outcome.source,
            })
    return queries


def rank_pages(scores: Sequence[float], keys: Sequence[tuple[str, int]], relevant: set) -> list[tuple[str, int]]:
    """Collapse chunk scores to pages and rank them, breaking ties *against* relevance.

    A page's score is the best score of its chunks; pages with score 0 (no
    query term in common) are not retrieved at all. Among pages with equal
    scores (compared at 9 decimals) non-relevant pages come first, then file
    name and page: the reported rank is therefore a lower bound that does not
    depend on chunk order or on how documents are named.
    """
    best: dict[tuple[str, int], float] = {}
    for key, score in zip(keys, scores):
        if score > 0 and score > best.get(key, 0.0):
            best[key] = score
    ordered = sorted(best.items(), key=lambda kv: (-round(kv[1], 9), kv[0] in relevant, kv[0]))
    return [key for key, _ in ordered]


def _retrieval_summary(rows: Sequence[dict], ks: Sequence[int]) -> dict:
    summary: dict[str, Any] = {"queries": len(rows)}
    for k in ks:
        summary[f"recall@{k}"] = mean(r["recall"][f"@{k}"] for r in rows)
    summary["mrr"] = mean(r["reciprocal_rank"] for r in rows)
    return summary


def retrieval_metrics(chunks: Sequence[Any], queries: Sequence[dict], *, ks: Sequence[int] = RETRIEVAL_KS) -> dict:
    """BM25 over ``chunks`` (one index), each query scored against its relevant (source, page) set."""
    index = BM25([_text(getattr(c, "retrieve_content", "")) for c in chunks])
    keys = [(str(c.source_name), int(c.page)) for c in chunks]
    rows: list[dict] = []
    skipped: list[str] = []
    for query in queries:
        relevant = {(r["source"], int(r["page"])) for r in query.get("relevant") or []}
        if not relevant:
            skipped.append(query.get("query", ""))
            continue
        ranked = rank_pages(index.scores(query.get("query", "")), keys, relevant)
        first = next((i + 1 for i, key in enumerate(ranked) if key in relevant), None)
        rows.append({
            "query": query.get("query", ""),
            "doc_type": query.get("doc_type", ""),
            "relevant": [f"{s}#p{p}" for s, p in sorted(relevant)],
            "first_relevant_rank": first,
            "reciprocal_rank": round(1.0 / first, 6) if first else 0.0,
            "recall": {f"@{k}": ratio(len(relevant & set(ranked[:k])), len(relevant)) for k in ks},
            "top": [f"{s}#p{p}" for s, p in ranked[:max(ks)]],
        })
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[row["doc_type"]].append(row)
    return {
        "index": {"chunks": len(chunks), "pages": len(set(keys)), "k1": K1, "b": B},
        "overall": _retrieval_summary(rows, ks),
        "by_doc_type": {name: _retrieval_summary(group, ks) for name, group in sorted(by_type.items())},
        "skipped_queries": skipped,
        "per_query": rows,
    }


# --------------------------------------------------------------------------- comparison


def compare_metrics(expected: Any, actual: Any, path: str = "", *, tolerance: float = 1e-9) -> list[str]:
    """Differences between two metric trees as ``"path: expected != actual"`` lines (empty when equal)."""
    diffs: list[str] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual), key=str):
            where = f"{path}.{key}" if path else str(key)
            if key not in actual:
                diffs.append(f"{where}: missing in actual")
            elif key not in expected:
                diffs.append(f"{where}: not in expected")
            else:
                diffs.extend(compare_metrics(expected[key], actual[key], where, tolerance=tolerance))
        return diffs
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: {len(expected)} item(s) != {len(actual)} item(s)"]
        for i, (a, b) in enumerate(zip(expected, actual)):
            diffs.extend(compare_metrics(a, b, f"{path}[{i}]", tolerance=tolerance))
        return diffs
    if isinstance(expected, bool) or isinstance(actual, bool):
        return [] if expected is actual else [f"{path}: {expected!r} != {actual!r}"]
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return [] if abs(float(expected) - float(actual)) <= tolerance else [f"{path}: {expected!r} != {actual!r}"]
    return [] if expected == actual else [f"{path}: {expected!r} != {actual!r}"]


# --------------------------------------------------------------------------- definitions


METRIC_DEFINITIONS: dict[str, str] = {
    "normalization": (
        "Normalised comparisons first " + NORMALIZATION + ". Exact comparisons compare the raw strings."
    ),
    "failures": (
        "A document whose extraction raised is scored, not skipped: every truth item of it counts as wrong or "
        "not found. Ratios with an empty denominator are reported as n/a."
    ),
    "drawing.field_accuracy": (
        "Per title-block field: fraction of truth sheets whose emitted metadata.title_block[field] equals the truth "
        "value (exact and normalised). Fields with an empty truth value are not scored."
    ),
    "drawing.sheet_accuracy": (
        "Per sheet (source page): fraction of that sheet's scored fields that match. Group rows report "
        "micro accuracy over all fields of the group and sheet_mean_normalized, the mean of per-sheet accuracies."
    ),
    "drawing.groups": (
        "all = every truth sheet; read = sheets whose chunk carries a title block read on that same sheet "
        "(conditional on success: sheets of failed documents are not in it, so compare it with 'all'); "
        "inherited = sheets that received the title block of another page (title_block_pages='first'); "
        "raster_title_block = sheets whose truth marks the title block as raster-only (no text layer); "
        "text_layer_title_block = the other sheets. all, raster_title_block and text_layer_title_block include "
        "the sheets of failed documents, scored as wrong."
    ),
    "drawing.gate_pass_rate": (
        "Documents that returned chunks / documents attempted. gate_failed counts TitleBlockGateError (fewer "
        "than min_pairs pairs); other_errors counts failures before the gate (transport, empty answer, replay miss)."
    ),
    "drawing.variants": (
        "Each drawing variant is a separate run of the same documents that differs only in "
        "DrawingConfig.max_image_pixels (cap = packaged default, nocap = 0 = no cap, the original behaviour)."
    ),
    "slides.image_role_accuracy": (
        "Over all truth image blocks: the first emitted description attached to the same shape (matched by "
        "shape name) has image_role equal to the truth role. A block without a description counts as wrong."
    ),
    "slides.must_mention_recall": (
        "Over all must-mention facts of all slides: the normalised fact is a substring of the normalised "
        "retrieve_content of that slide's chunk(s). slide_mean_recall averages per-slide recall over slides "
        "with at least one fact."
    ),
    "slides.hallucinated_blocks": (
        "Emitted descriptions that do not describe a distinct truth image block: not tied to any image block "
        "(ungrounded), tied to an image not listed in the truth (not_in_truth), or a second description of a "
        "block already described (duplicate)."
    ),
    "docx.page_count_correct": (
        "The page count of the technique's page split (provenance.page_split xml_pages + padded_pages, else "
        "the highest chunk page) equals truth page_count."
    ),
    "docx.row_page_accuracy": (
        "Over all rows (header included) of every truth table that spans pages: the set of pages whose chunk "
        "metadata.multi_page_tables lists the row index is exactly {truth page}. Rows missing or listed on "
        "several pages count as wrong. assignment counts page entries by source (vlm or xml_row_markers)."
    ),
    "docx.image_must_mention_recall": (
        "Over all must-mention facts of pages with pictures: the normalised fact is a substring of the "
        "normalised retrieve_content of that page's chunk(s)."
    ),
    "retrieval.index": (
        "One BM25 index (k1=1.2, b=0.75, NFKC+casefold word tokens, distinct query terms) per run over the "
        "retrieve_content of ALL chunks of ALL documents of that run. Queries are the truth-file queries of "
        "the run's documents; relevance is the (source file name, 1-based page) pairs listed in the truth."
    ),
    "retrieval.ranking": (
        "Chunk scores are collapsed to pages (a page scores its best chunk; duplicate chunks of a page count "
        "once); pages scoring 0 are not retrieved. Ties are broken against relevant pages, so ranks are "
        "lower bounds."
    ),
    "retrieval.recall_at_k": "|relevant pages among the top k ranked pages| / |relevant pages|, averaged over queries.",
    "retrieval.mrr": "Mean over queries of 1 / rank of the first relevant page (0 when none is retrieved).",
    "retrieval.baseline": (
        "The text-only baseline indexes the chunks of the NullBackend runs. A drawing whose baseline run "
        "raised (the acceptance gate rejects an empty answer by design) is represented by the drawing "
        "technique's own chunks built with an empty title block - file name, page, text layer, fields parsed "
        "from the text layer, domain synonyms - so it differs from a VLM chunk only by what the model read "
        "(listed under model_free_substitutes). A drawing whose VLM run failed contributes no chunks to the "
        "VLM index, as it would not be indexed in production."
    ),
    "cost": (
        "Per technique run (and drawing variant), from the VLM client's usage ledger: calls = every answer "
        "the technique received (re-asks included, cached answers included), cached = the subset served from "
        "--cache-dir with no tokens or latency, input/output/reasoning tokens as reported by the provider, "
        "model latency (sum of per-call latencies, so concurrent calls overlap) and wall time of the whole "
        "run. requests_sent (per run) and max_image_pixels count what was actually sent to the backend, "
        "every attempt included; reasoning_effort lists the efforts sent per task. Replayed runs report the "
        "usage stored in the recording. Baseline runs answer locally with NullBackend: zero tokens."
    ),
}
