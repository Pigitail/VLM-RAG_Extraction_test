"""Run the techniques over the evaluation documents and assemble ``results.json``.

Shape of an evaluation
    * one *technique run* per (mode, technique[, drawing variant]):
      ``vlm/drawing@cap``, ``vlm/drawing@nocap``, ``vlm/slides``, ``vlm/docx``,
      and the text-only baseline ``baseline/drawing@cap``, ``baseline/slides``,
      ``baseline/docx`` (NullBackend: every model call answers ``{}``, so every
      technique takes its documented fallback);
    * inside a run every document is isolated: an exception becomes a
      :class:`~vlmrag.evaluation.metrics.DocOutcome` with ``error`` set and the
      run moves on - a failure is a data point;
    * every document gets its own :class:`~vlmrag.vlm.client.VLMClient` on the
      run's backend, so usage is known per document and per run;
    * retrieval is scored once per *retrieval run*: all chunks of all documents
      of ``vlm@<variant>`` (the drawing variant plus the slides and docx runs)
      and of ``baseline`` go into one BM25 index each;
    * every run also records what it actually sent (:class:`RequestStats`):
      requests, images, the largest image in pixels and the reasoning effort
      per task, so "cap" vs "nocap" and per-task efforts are visible in the
      results rather than assumed from the configuration.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from .. import __version__
from ..schema import Chunk, doc_id_for
from ..techniques import load_technique
from ..vlm.backends import NullBackend
from ..vlm.client import VLMClient
from ..vlm.usage import UsageLedger
from .documents import EvalDocument, document_summary
from .metrics import (
    METRIC_DEFINITIONS,
    RETRIEVAL_KS,
    DocOutcome,
    collect_queries,
    docx_metrics,
    drawing_metrics,
    retrieval_metrics,
    slides_metrics,
)
from .textnorm import NORMALIZATION

__all__ = [
    "DrawingVariant",
    "EvalConfig",
    "TechniqueRun",
    "RESULTS_FORMAT",
    "default_cap",
    "drawing_config_for",
    "drawing_non_defaults",
    "run_technique",
    "run_all",
    "assemble_results",
    "model_free_drawing_chunks",
    "RequestStats",
    "png_size",
    "cost_fingerprint",
    "sanitize_message",
    "attach_backend_stats",
]

log = logging.getLogger(__name__)

RESULTS_FORMAT = "vlmrag-evaluation/1"
MODES = ("vlm", "baseline")
_PATH_FIELDS = {"schema_path", "domain_terms_path"}
_WIN_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s'\"<>|]*")
_POSIX_PATH_RE = re.compile(r"(?<![\w.])/(?:home|Users|tmp|var|private|mnt|opt|root)/[^\s'\"<>|]*")
# Provider error texts can carry endpoints, hosts and request ids; none of that belongs in a published file.
_URL_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s'\"<>]+")
_HOST_RE = re.compile(
    r"\b(?:[a-zA-Z0-9-]+\.)+(?:com|net|org|io|ai|dev|cloud|internal|local|corp|lan|intra|kr|cn|jp|us|eu|de|uk)\b"
)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b")
_UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_REQUEST_ID_RE = re.compile(r"\b(?:req|chatcmpl|resp|run|msg)_[A-Za-z0-9]{8,}\b")
_VARIANT_RE = re.compile(r"maxpx[=-](\d+)")


def default_cap() -> int:
    """The packaged ``DrawingConfig.max_image_pixels`` default (4 MP)."""
    from ..techniques.drawing import DrawingConfig

    return int(next(f.default for f in dataclasses.fields(DrawingConfig) if f.name == "max_image_pixels"))


@dataclass(frozen=True)
class DrawingVariant:
    """A drawing run setting compared side by side: the per-image pixel cap.

    ``cap`` is the packaged default (4 MP), ``nocap`` disables the cap (the
    original behaviour: every crop keeps its configured enlargement) and
    ``maxpx=N`` sets any other cap.
    """

    name: str
    max_image_pixels: int

    @classmethod
    def parse(cls, spec: str) -> "DrawingVariant":
        text = spec.strip().lower()
        if text == "cap":
            return cls("cap", default_cap())
        if text == "nocap":
            return cls("nocap", 0)
        match = _VARIANT_RE.fullmatch(text)
        if match:
            pixels = int(match.group(1))
            return cls(f"maxpx-{pixels}", pixels)
        raise ValueError(f"unknown drawing variant {spec!r}; use cap, nocap or maxpx=N")

    def to_dict(self) -> dict:
        return {"name": self.name, "max_image_pixels": self.max_image_pixels}


@dataclass
class EvalConfig:
    """What to run. ``drawing_overrides`` are extra :class:`DrawingConfig` fields (tests use low DPI).

    ``drawing_env=False`` ignores ``VLMRAG_DRAWING_*`` variables: a replay
    rebuilds the drawing configuration from the recording's manifest alone, so
    an unrelated variable on the replaying machine cannot change the requests.
    """

    techniques: tuple[str, ...] = ("drawing", "slides", "docx")
    variants: tuple[DrawingVariant, ...] = ()
    baseline: bool = True
    drawing_overrides: dict = field(default_factory=dict)
    doc_workers: int = 1
    drawing_env: bool = True

    def __post_init__(self) -> None:
        if not self.variants:
            self.variants = (DrawingVariant.parse("cap"),)
        names = [v.name for v in self.variants]
        if len(set(names)) != len(names):
            raise ValueError("drawing variants must be distinct")
        if self.doc_workers < 1:
            raise ValueError("doc_workers must be >= 1")


@dataclass
class TechniqueRun:
    run_id: str
    mode: str
    technique: str
    variant: Optional[str]
    outcomes: list[DocOutcome]
    usage: dict
    wall_time_s: float
    backend: str
    model: str
    drawing_config: Optional[dict] = None
    replay: Optional[dict] = None
    recording: Optional[dict] = None
    requests: Optional[dict] = None
    models_reported: Optional[list] = None


# --------------------------------------------------------------------------- what was sent


def png_size(data: bytes) -> Optional[tuple[int, int]]:
    """``(width, height)`` from a PNG header, without decoding the image (``None`` if not a PNG)."""
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        width, height = struct.unpack(">II", data[16:24])
        return int(width), int(height)
    return None


class RequestStats:
    """Transparent backend wrapper that tallies what each task actually sent.

    Configuration says what *should* be sent; this records what was: request
    and image counts, the largest and total image pixels, and the reasoning
    efforts, per task. It sits between the client and the run's backend, so it
    sees every attempt (retries included) of uncached calls; cached answers
    never reach it.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.name = getattr(inner, "name", "")
        self.model = getattr(inner, "model", "")
        self._lock = threading.Lock()
        self._tasks: dict[str, dict] = {}
        # On Azure the configured "model" is a deployment name; the response says which model answered.
        self._reported: set[str] = set()

    def generate(self, request: Any) -> Any:
        sizes = [png_size(image.png) for image in request.images()]
        pixels = [w * h for w, h in (s for s in sizes if s)]
        with self._lock:
            row = self._tasks.setdefault(request.task or "unknown", {
                "requests": 0, "images": 0, "max_image_pixels": 0, "image_pixels": 0, "reasoning_effort": [],
            })
            row["requests"] += 1
            row["images"] += len(sizes)
            row["image_pixels"] += sum(pixels)
            row["max_image_pixels"] = max([row["max_image_pixels"]] + pixels)
            effort = request.reasoning_effort or "none"
            if effort not in row["reasoning_effort"]:
                row["reasoning_effort"].append(effort)
        response = self.inner.generate(request)
        model = getattr(response, "model", None)
        if isinstance(model, str) and model:
            with self._lock:
                self._reported.add(model)
        return response

    def summary(self) -> dict:
        with self._lock:
            return {task: dict(row, reasoning_effort=sorted(row["reasoning_effort"]))
                    for task, row in sorted(self._tasks.items())}

    def reported_models(self) -> list[str]:
        """Model names the backend's responses reported (what actually answered)."""
        with self._lock:
            return sorted(self._reported)


# --------------------------------------------------------------------------- drawing configuration


def drawing_config_for(variant: DrawingVariant, overrides: Optional[dict] = None, *, use_env: bool = True) -> Any:
    """``DrawingConfig.from_env`` + ``overrides`` + the variant's pixel cap (the variant always wins).

    ``use_env=False`` starts from the packaged defaults instead of the
    ``VLMRAG_DRAWING_*`` environment (replays).
    """
    from ..techniques.drawing import DrawingConfig

    values = dict(overrides or {})
    values["max_image_pixels"] = variant.max_image_pixels
    return DrawingConfig.from_env(None if use_env else {}, **values)


def drawing_non_defaults(cfg: Any) -> dict:
    """Plain-valued fields of ``cfg`` that differ from the packaged defaults (cap and file paths excluded).

    Recorded in the manifest so a replay rebuilds the same crops and prompts
    even if the replaying machine has different ``VLMRAG_DRAWING_*`` variables.
    """
    out = {}
    for f in dataclasses.fields(cfg):
        if f.name in _PATH_FIELDS or f.name == "max_image_pixels" or f.default is dataclasses.MISSING:
            continue
        value = getattr(cfg, f.name)
        if isinstance(value, (bool, int, float, str)) and value != f.default:
            out[f.name] = value
    return out


def _drawing_summary(cfg: Any) -> dict:
    summary = cfg.render_summary()
    summary.update(
        min_pairs=cfg.min_pairs,
        title_block_pages=cfg.title_block_pages,
        schema=cfg.schema.name,
        custom_schema_file=bool(cfg.schema_path),
        domain_terms=("custom" if cfg.domain_terms_path else "packaged") if not cfg.domain.is_empty() else "none",
        include_text_layer=cfg.include_text_layer,
        non_defaults=drawing_non_defaults(cfg),
    )
    return summary


# --------------------------------------------------------------------------- running


def sanitize_message(message: str, limit: int = 300) -> str:
    """Error text safe to publish: no absolute paths, URLs, host names, IP addresses or request ids; length capped.

    Technique errors embed the provider's message (for example "vision pass
    failed: BadRequestError: ..."), which can echo an endpoint or a request
    id. The recording's failure sidecar never stores messages at all;
    ``results.json`` keeps this redacted form because the technique's own text
    says which step failed.
    """
    text = _URL_RE.sub("<url>", str(message))
    text = _WIN_PATH_RE.sub("<path>", text)
    text = _POSIX_PATH_RE.sub("<path>", text)
    text = _HOST_RE.sub("<host>", text)
    text = _IPV4_RE.sub("<ip>", text)
    text = _UUID_RE.sub("<id>", text)
    text = _REQUEST_ID_RE.sub("<id>", text)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def run_technique(
    documents: Sequence[EvalDocument],
    technique: str,
    *,
    mode: str,
    backend: Any,
    settings: Any,
    variant: Optional[DrawingVariant] = None,
    config: Optional[EvalConfig] = None,
    cache: Any = None,
    extract: Optional[Callable[..., list]] = None,
) -> TechniqueRun:
    """Run one technique over its documents with per-document error capture."""
    config = config or EvalConfig()
    docs = [d for d in documents if d.technique == technique]
    extract_fn = extract or load_technique(technique)
    kwargs: dict = {}
    drawing_summary = None
    if technique == "drawing":
        variant = variant or config.variants[0]
        cfg = drawing_config_for(variant, config.drawing_overrides, use_env=config.drawing_env)
        kwargs["config"] = cfg
        drawing_summary = _drawing_summary(cfg)
    else:
        variant = None
    run_id = f"{mode}/{technique}" + (f"@{variant.name}" if variant else "")
    ledger = UsageLedger()
    sent = RequestStats(backend)

    def one(doc: EvalDocument) -> DocOutcome:
        client = VLMClient(
            sent,
            max_concurrency=int(getattr(settings, "max_concurrency", 4) or 4),
            max_retries=int(getattr(settings, "max_retries", 3) or 0),
            cache=cache,
        )
        started = time.perf_counter()
        error = None
        chunks: list = []
        try:
            chunks = list(extract_fn(doc.path, client, settings, **kwargs))
        except Exception as exc:  # noqa: BLE001 - per-document isolation: a failure is a data point
            error = {"type": type(exc).__name__, "message": sanitize_message(str(exc))}
            log.warning("%s %s: %s: %s", run_id, doc.source, type(exc).__name__, exc)
        wall = round(time.perf_counter() - started, 3)
        ledger.merge(client.usage)
        return DocOutcome(
            source=doc.source,
            technique=technique,
            truth=doc.truth,
            chunks=chunks,
            error=error,
            wall_time_s=wall,
            usage=client.usage.summary()["total"],
        )

    started = time.perf_counter()
    workers = max(1, min(config.doc_workers, len(docs) or 1))
    if workers == 1:
        outcomes = [one(doc) for doc in docs]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="eval-doc") as pool:
            outcomes = list(pool.map(one, docs))
    wall_total = round(time.perf_counter() - started, 3)
    failed = sum(1 for o in outcomes if not o.ok)
    log.info("%s: %d document(s), %d failed, %.1fs", run_id, len(outcomes), failed, wall_total)
    return TechniqueRun(
        run_id=run_id,
        mode=mode,
        technique=technique,
        variant=variant.name if variant else None,
        outcomes=outcomes,
        usage=ledger.summary(),
        wall_time_s=wall_total,
        backend=str(getattr(backend, "name", "") or ""),
        model=str(getattr(backend, "model", "") or ""),
        drawing_config=drawing_summary,
        requests=sent.summary(),
        models_reported=sent.reported_models(),
    )


BackendFor = Callable[[str, Optional[str]], Any]  # (technique, variant name or None) -> backend


def attach_backend_stats(run: TechniqueRun, backend: Any) -> None:
    """Copy replay / recording counters of the run's backend into the run record."""
    from .recording import EvalRecordingBackend, EvalReplayBackend

    if isinstance(backend, EvalReplayBackend):
        run.replay = dict(backend.stats(), miss_details=list(backend.misses[:20]))
    elif isinstance(backend, EvalRecordingBackend):
        run.recording = {"file": backend.path.name, "recorded": backend.recorded, "failed_calls": backend.failed}


def run_all(
    documents: Sequence[EvalDocument],
    settings: Any,
    config: EvalConfig,
    backend_for: BackendFor,
    *,
    cache: Any = None,
) -> list[TechniqueRun]:
    """Every VLM technique run (each drawing variant separately), then the NullBackend baseline runs."""
    runs: list[TechniqueRun] = []
    for technique in config.techniques:
        variants: Iterable[Optional[DrawingVariant]] = config.variants if technique == "drawing" else (None,)
        for variant in variants:
            backend = backend_for(technique, variant.name if variant else None)
            run = run_technique(
                documents, technique, mode="vlm", backend=backend, settings=settings,
                variant=variant, config=config, cache=cache,
            )
            attach_backend_stats(run, backend)
            runs.append(run)
    if config.baseline:
        for technique in config.techniques:
            runs.append(run_technique(
                documents, technique, mode="baseline", backend=NullBackend(), settings=settings,
                variant=config.variants[0] if technique == "drawing" else None, config=config,
            ))
    return runs


# --------------------------------------------------------------------------- baseline substitute

SUBSTITUTE = "drawing_chunk_without_model_output"


def model_free_drawing_chunks(path: Path, cfg: Any) -> list[Chunk]:
    """The drawing technique's own chunks with an empty title block: everything it adds *without* a model.

    Stands in, for retrieval only, for a drawing whose baseline run raised
    (the title-block gate rejects an empty answer by design). It is built with
    the technique's chunk builder, so it keeps every deterministic part of a
    VLM chunk - file name, ``page N``, the text layer, structured fields
    parsed from that text and domain synonyms - and lacks only what the model
    read. The baseline therefore differs from the VLM run by the model output
    alone; a plain text-layer stand-in would also lack the deterministic
    enrichment and credit it to the model. A test pins this to the
    technique's own output for an empty answer with the gate disabled.
    """
    from ..render.pdf import PdfPages
    from ..techniques.drawing.chunks import build_chunk
    from ..techniques.drawing.crops import FULL_PAGE, region_text
    from ..techniques.drawing.title_block import TitleBlockResult

    doc_id = doc_id_for(path)
    with PdfPages(path) as pages:
        count = len(pages)
        texts = [region_text(pages, index, FULL_PAGE) for index in range(count)]
        rotations = [pages.rotation(index) for index in range(count)]
    chunks: list[Chunk] = []
    for index in range(count):
        # Same page attribution as the technique: with title_block_pages="first" every page carries page 1's block.
        read_on = 1 if cfg.title_block_pages == "first" else index + 1
        empty = TitleBlockResult(page=read_on, fields={key: "" for key in cfg.schema.keys()}, pairs=[])
        chunk = build_chunk(
            doc_id=doc_id,
            source_name=path.name,
            page_index=index,
            page_count=count,
            page_text=texts[index],
            rotation=rotations[index],
            title_block=empty,
            cfg=cfg,
        )
        chunk.metadata["evaluation_substitute"] = SUBSTITUTE
        chunk.provenance["evaluation_substitute"] = SUBSTITUTE
        chunks.append(chunk)
    return chunks


# --------------------------------------------------------------------------- results


def _retrieval_sets(
    runs: Sequence[TechniqueRun], documents: Sequence[EvalDocument], config: EvalConfig
) -> tuple[dict[str, list[DocOutcome]], dict[str, list[str]]]:
    """Outcomes per retrieval run name, and the drawings replaced by their model-free chunks (baseline only).

    A drawing whose *VLM* run failed contributes no chunks to that VLM index,
    as in production, where the file is not indexed.
    """
    sets: dict[str, list[DocOutcome]] = {}
    substitutes: dict[str, list[str]] = {}
    vlm = [r for r in runs if r.mode == "vlm"]
    drawing_runs = [r for r in vlm if r.technique == "drawing"]
    others = [o for r in vlm if r.technique != "drawing" for o in r.outcomes]
    if drawing_runs:
        for run in drawing_runs:
            sets[f"vlm@{run.variant}"] = list(run.outcomes) + others
    elif vlm:
        sets["vlm"] = others
    baseline = [r for r in runs if r.mode == "baseline"]
    if baseline:
        by_source = {d.source: d for d in documents}
        outcomes: list[DocOutcome] = []
        replaced: list[str] = []
        cfg = None
        for run in baseline:
            for outcome in run.outcomes:
                doc = by_source.get(outcome.source)
                if outcome.technique == "drawing" and not outcome.ok and doc is not None:
                    try:
                        if cfg is None:
                            variant = next((v for v in config.variants if v.name == run.variant), config.variants[0])
                            cfg = drawing_config_for(variant, config.drawing_overrides, use_env=config.drawing_env)
                        outcome = dataclasses.replace(outcome, chunks=model_free_drawing_chunks(doc.path, cfg))
                        replaced.append(outcome.source)
                    except Exception as exc:  # noqa: BLE001 - an unreadable PDF simply contributes nothing
                        log.warning("model-free chunks of %s unavailable: %s", outcome.source, exc)
                outcomes.append(outcome)
        sets["baseline"] = outcomes
        substitutes["baseline"] = replaced
    return sets, substitutes


def _cost_row(run: TechniqueRun) -> dict:
    total = run.usage.get("total", {})
    requests = run.requests or {}
    return {
        "run": run.run_id,
        "mode": run.mode,
        "technique": run.technique,
        "variant": run.variant,
        "backend": run.backend,
        "documents": len(run.outcomes),
        "failed": sum(1 for o in run.outcomes if not o.ok),
        # calls = every answer the technique received, cached ones included; cached = the subset served
        # from --cache-dir (no tokens, no latency). They are never added together.
        "calls": int(total.get("calls", 0)),
        "cached": int(total.get("cached", 0)),
        "max_image_pixels": max([0] + [int(r.get("max_image_pixels", 0)) for r in requests.values()]),
        "reasoning_effort": {task: r.get("reasoning_effort", []) for task, r in requests.items()},
        "input_tokens": int(total.get("input_tokens", 0)),
        "output_tokens": int(total.get("output_tokens", 0)),
        "reasoning_tokens": int(total.get("reasoning_tokens", 0)),
        "total_tokens": int(total.get("total_tokens", 0)),
        "model_latency_s": float(total.get("latency_s", 0.0)),
        "wall_time_s": run.wall_time_s,
        "per_task": run.usage.get("tasks", {}),
        "per_document": [
            {
                "source": o.source,
                "error_type": (o.error or {}).get("type"),
                "wall_time_s": o.wall_time_s,
                "calls": int((o.usage or {}).get("calls", 0)),
                "cached": int((o.usage or {}).get("cached", 0)),
                "input_tokens": int((o.usage or {}).get("input_tokens", 0)),
                "output_tokens": int((o.usage or {}).get("output_tokens", 0)),
                "reasoning_tokens": int((o.usage or {}).get("reasoning_tokens", 0)),
            }
            for o in run.outcomes
        ],
    }


def cost_fingerprint(results: dict) -> dict:
    """The parts of ``results["cost"]`` a replay must reproduce exactly (no times)."""
    keys = ("documents", "failed", "calls", "cached", "input_tokens", "output_tokens", "reasoning_tokens")
    return {row["run"]: {k: row[k] for k in keys} for row in results.get("cost", [])}


def _run_record(run: TechniqueRun) -> dict:
    record = {
        "run": run.run_id,
        "mode": run.mode,
        "technique": run.technique,
        "variant": run.variant,
        "backend": run.backend,
        "model": run.model,
        "wall_time_s": run.wall_time_s,
        "documents": [
            {
                "source": o.source,
                "ok": o.ok,
                "chunks": len(o.chunks),
                "pages": sorted({int(c.page) for c in o.chunks}),
                "error": o.error,
                "fallbacks": sorted({
                    str(code) for c in o.chunks for code in ((c.provenance or {}).get("fallbacks") or [])
                }),
            }
            for o in run.outcomes
        ],
    }
    if run.drawing_config is not None:
        record["drawing_config"] = run.drawing_config
    if run.requests is not None:
        record["requests_sent"] = run.requests
    if run.models_reported is not None:
        record["models_reported"] = run.models_reported
    if run.replay is not None:
        record["replay"] = run.replay
    if run.recording is not None:
        record["recording"] = run.recording
    return record


def _caveats(documents: Sequence[EvalDocument]) -> list[str]:
    counts: dict[str, int] = {}
    for doc in documents:
        counts[doc.technique] = counts.get(doc.technique, 0) + 1
    listed = ", ".join(f"{n} {t}" for t, n in counts.items())
    caveats = [
        f"Synthetic documents only ({listed}); invented content generated by vlmrag.synthetic. "
        "Numbers describe these fixtures, not real document collections.",
        "Small N: every ratio comes with its count; one field, block or query moves a ratio by several points.",
        "Single run: model answers vary between runs and no confidence intervals are reported.",
        "Retrieval is lexical BM25 over retrieve_content; an embedding retriever would also match paraphrases. "
        "Must-mention facts are matched as normalised substrings, so a correct paraphrase counts as not found.",
        "The text-only baseline never contacts a model: every request it builds is answered locally with '{}' "
        "(NullBackend), so its calls are stub answers with zero tokens.",
    ]
    if "drawing" in counts:
        caveats.append(
            "Drawings use the technique default title_block_pages='first': later sheets carry the title block read "
            "on sheet 1 ('inherited' group), so sheet-specific fields (title, scale, sheet) differ there by design."
        )
        caveats.append(
            "The text-only drawing baseline fails the title-block gate by design (an empty answer yields no pairs); "
            "for retrieval it is represented by the technique's own chunks with an empty title block (text layer, "
            "fields parsed from it, domain synonyms), so the retrieval difference is the model output alone. A drawing "
            "whose VLM run fails contributes nothing to the VLM index."
        )
    if "docx" in counts:
        caveats.append(
            "The synthetic DOCX carries correct rendered-page markers, so the deterministic row fallback is exact by "
            "construction: row->page accuracy shows whether the model path keeps rows on their pages, not whether it "
            "repairs stale markers."
        )
    return caveats


def assemble_results(
    runs: Sequence[TechniqueRun],
    documents: Sequence[EvalDocument],
    config: EvalConfig,
    *,
    header: Optional[dict] = None,
    warnings: Optional[list[str]] = None,
) -> dict:
    """Compute every metric and build the ``results.json`` document."""
    metrics: dict[str, Any] = {}
    by_key = {(r.mode, r.technique, r.variant): r for r in runs}
    for technique, fn in (("drawing", drawing_metrics), ("slides", slides_metrics), ("docx", docx_metrics)):
        if technique not in config.techniques:
            continue
        section: dict[str, Any] = {}
        for mode in MODES:
            mode_runs = [r for (m, t, _), r in by_key.items() if m == mode and t == technique]
            if not mode_runs:
                continue
            if technique == "drawing":
                section[mode] = {r.variant: fn(r.outcomes) for r in mode_runs}
            else:
                section[mode] = fn(mode_runs[0].outcomes)
        metrics[technique] = section
    retrieval_sets, substitutes = _retrieval_sets(runs, documents, config)
    metrics["retrieval"] = {}
    for name, outcomes in retrieval_sets.items():
        chunks = [c for o in outcomes for c in o.chunks]
        result = retrieval_metrics(chunks, collect_queries(outcomes), ks=RETRIEVAL_KS)
        if name in substitutes:
            result["model_free_substitutes"] = sorted(substitutes[name])
        metrics["retrieval"][name] = result

    results: dict[str, Any] = {"format": RESULTS_FORMAT, "package_version": __version__}
    results.update(header or {})
    reported = sorted({m for r in runs if r.mode == "vlm" for m in (r.models_reported or [])})
    if reported:
        results["model"] = dict(results.get("model") or {}, reported_by_responses=reported)
    results["evaluation"] = {
        "techniques": list(config.techniques),
        "drawing_variants": [v.to_dict() for v in config.variants] if "drawing" in config.techniques else [],
        "baseline": config.baseline,
        "doc_workers": config.doc_workers,
        "drawing_overrides": dict(config.drawing_overrides),
        "retrieval": {"model": "bm25", "ks": list(RETRIEVAL_KS)},
        "normalization": NORMALIZATION,
    }
    results["documents"] = dict(results.get("documents") or {})
    results["documents"]["items"] = [document_summary(d) for d in documents]
    results["metrics"] = metrics
    results["cost"] = [_cost_row(r) for r in runs]
    results["runs"] = [_run_record(r) for r in runs]
    replay = [r.replay for r in runs if r.replay is not None]
    if replay:
        results["replay"] = {
            "misses": sum(int(r.get("misses", 0)) for r in replay),
            "served": sum(int(r.get("served", 0)) for r in replay),
            "unused": sum(int(r.get("unused", 0)) for r in replay),
            "recorded_failures_served": sum(int(r.get("recorded_failures_served", 0)) for r in replay),
            "miss_details": [m for r in runs if r.replay for m in r.replay.get("miss_details", [])][:20],
        }
    results["warnings"] = list(warnings or [])
    results["caveats"] = _caveats(documents)
    results["metric_definitions"] = dict(METRIC_DEFINITIONS)
    return results
