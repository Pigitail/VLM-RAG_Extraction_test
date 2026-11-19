"""Evaluation harness: run the techniques on documents with ground truth and score them.

``python -m vlmrag.evaluation --out-dir DIR`` generates the synthetic
documents, runs every technique with the configured model and with a
text-only baseline (``NullBackend``), and writes ``results.json`` plus
``results.md``. ``--record-dir`` stores every model exchange so that
``--replay-dir`` reproduces the numbers without credentials. See
``docs/evaluation.md`` for the metric definitions.

Modules:

* :mod:`.textnorm` - the string normalisation every comparison uses;
* :mod:`.bm25` - a dependency-free BM25 index;
* :mod:`.metrics` - pure metric functions and their written definitions;
* :mod:`.documents` - document discovery / generation, docx layout pins;
* :mod:`.recording` - record/replay with usage, failures and miss counting;
* :mod:`.oracle` - a perfect model built from truth (harness self-check);
* :mod:`.runner` - technique runs with per-document error capture, results assembly;
* :mod:`.report` - ``results.json`` / ``results.md``;
* :mod:`.cli` - the command line.
"""

from __future__ import annotations

from .bm25 import BM25, tokenize
from .metrics import (
    METRIC_DEFINITIONS,
    DocOutcome,
    compare_metrics,
    docx_metrics,
    drawing_metrics,
    rank_pages,
    retrieval_metrics,
    slides_metrics,
)
from .runner import DrawingVariant, EvalConfig, TechniqueRun, assemble_results, run_all, run_technique
from .textnorm import normalize_value

__all__ = [
    "BM25",
    "DocOutcome",
    "DrawingVariant",
    "EvalConfig",
    "METRIC_DEFINITIONS",
    "TechniqueRun",
    "assemble_results",
    "compare_metrics",
    "docx_metrics",
    "drawing_metrics",
    "normalize_value",
    "rank_pages",
    "retrieval_metrics",
    "run_all",
    "run_technique",
    "slides_metrics",
    "tokenize",
]
