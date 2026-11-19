"""Which documents an evaluation runs on, and how to rebuild them for a replay.

Documents are either generated (the default: the synthetic fixtures, written
to a temporary directory - never into the repository) or taken from a
directory the user names. Each document is paired with its
``<stem>.truth.json`` when one exists; documents without truth are still
processed and indexed (they act as retrieval distractors) but contribute no
quality numbers.

Portable replay
    Replay keys contain the request *text*, and the ``docx`` generator places
    its rendered page-break markers where the local LibreOffice breaks pages.
    A different LibreOffice version (or none) can therefore produce a
    different document, different requests and replay misses. A recording run
    stores every docx truth's ``anchor_pages`` in its manifest, and a replay
    regenerates the documents with exactly that layout
    (``generate(..., layout=anchor_pages)``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

from ..schema import doc_id_for
from ..synthetic import accepted_options, load_generator
from ..techniques import SUFFIXES

__all__ = [
    "EvalDocument",
    "DOCX_LAYOUTS",
    "discover_documents",
    "generate_documents",
    "docx_layout_pins",
    "load_truth",
    "document_summary",
]

log = logging.getLogger(__name__)

#: ``auto``: LibreOffice layout when installed (renders match the markers); ``estimated``: machine-independent.
DOCX_LAYOUTS = ("auto", "estimated")


@dataclass
class EvalDocument:
    path: Path
    technique: str
    truth: Optional[dict] = None

    @property
    def source(self) -> str:
        return self.path.name


def load_truth(path: Union[str, Path]) -> Optional[dict]:
    """``<stem>.truth.json`` next to ``path``, or ``None``."""
    path = Path(path)
    truth_path = path.with_name(f"{path.stem}.truth.json")
    if not truth_path.is_file():
        return None
    return json.loads(truth_path.read_text(encoding="utf-8"))


def _check_unique(documents: Sequence[EvalDocument]) -> None:
    seen: dict[str, Path] = {}
    for doc in documents:
        if doc.source in seen:
            raise ValueError(
                f"two documents are named {doc.source!r} ({seen[doc.source]} and {doc.path}); relevance is keyed by "
                "file name, so document names must be unique"
            )
        seen[doc.source] = doc.path


def discover_documents(root: Union[str, Path], techniques: Iterable[str]) -> list[EvalDocument]:
    """Every file under ``root`` whose suffix one of ``techniques`` reads (Office lock files skipped)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"document directory not found: {root}")
    documents: list[EvalDocument] = []
    for technique in techniques:
        suffixes = SUFFIXES.get(technique, ())
        for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes):
            if path.name.startswith("~$"):
                continue
            documents.append(EvalDocument(path=path, technique=technique, truth=load_truth(path)))
    _check_unique(documents)
    return documents


def generate_documents(
    out_dir: Union[str, Path],
    techniques: Iterable[str],
    *,
    seed: int = 0,
    docx_layout: str = "auto",
    docx_pins: Optional[Mapping[str, Sequence[int]]] = None,
) -> list[EvalDocument]:
    """Generate the synthetic fixtures of ``techniques`` into ``out_dir/<technique>/``.

    ``docx_pins`` maps a docx file name to the ``anchor_pages`` of a recorded
    run and wins over ``docx_layout``.
    """
    if docx_layout not in DOCX_LAYOUTS:
        raise ValueError(f"docx_layout must be one of {DOCX_LAYOUTS} (got {docx_layout!r})")
    root = Path(out_dir)
    documents: list[EvalDocument] = []
    for technique in techniques:
        generate = load_generator(technique)
        target = root / technique
        target.mkdir(parents=True, exist_ok=True)
        options: dict = {}
        if technique == "docx":
            pins = dict(docx_pins or {})
            if pins:
                # The synthetic docx generator writes a single document.
                (name, pages), *rest = sorted(pins.items())
                if rest:
                    log.warning("only one docx layout pin is used (%s); ignoring %d more", name, len(rest))
                options["layout"] = list(pages)
            elif docx_layout == "estimated":
                options["use_libreoffice"] = False
        kwargs = accepted_options(generate, options)
        if set(kwargs) != set(options):
            raise ValueError(f"the {technique} generator does not accept {sorted(set(options) - set(kwargs))}")
        for path in generate(target, seed=seed, **kwargs):
            path = Path(path)
            documents.append(EvalDocument(path=path, technique=technique, truth=load_truth(path)))
    _check_unique(documents)
    return documents


def docx_layout_pins(documents: Iterable[EvalDocument]) -> dict[str, list[int]]:
    """``{file name: anchor_pages}`` of every docx document whose truth records its layout."""
    pins = {}
    for doc in documents:
        if doc.technique == "docx" and doc.truth and doc.truth.get("anchor_pages"):
            pins[doc.source] = [int(p) for p in doc.truth["anchor_pages"]]
    return pins


def document_summary(doc: EvalDocument) -> dict:
    """What results.json records about a document (no paths: they are machine specific)."""
    truth = doc.truth or {}
    summary = {
        "source": doc.source,
        "technique": doc.technique,
        "has_truth": doc.truth is not None,
        "doc_id": doc_id_for(doc.path),
        "queries": len(truth.get("queries") or []),
    }
    if "layout_source" in truth:
        summary["layout_source"] = truth["layout_source"]
    pages = truth.get("page_count") or truth.get("slide_count") or len(truth.get("pages") or []) or None
    if pages:
        summary["truth_pages"] = pages
    return summary
