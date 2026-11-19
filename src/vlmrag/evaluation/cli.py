"""``python -m vlmrag.evaluation``: one command from documents to ``results.json`` / ``results.md``.

Modes (mutually exclusive)
    live (default)   the backend from ``--provider``/``--model`` or the environment;
    ``--record-dir``  live, and every model exchange is written to
                      ``<dir>/<technique>[.<variant>].jsonl`` plus ``manifest.json``,
                      ``results.json`` and ``results.md`` - the files a reader
                      needs to reproduce the numbers without a key;
    ``--replay-dir``  no network: answers come from a recording directory, the
                      documents are regenerated exactly as recorded and the
                      metrics are compared with the recorded ``results.json``;
    ``--oracle``      a perfect model built from the truth files (checks the
                      harness, not a model).

Exit status: 0 on success (document failures are data points, not errors),
1 when a replay misses recorded answers or does not reproduce the recorded
metrics, 2 on usage or configuration errors.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

from .. import __version__
from ..config import PROVIDERS, Settings, load_dotenv
from ..techniques import TECHNIQUES
from .documents import (
    DOCX_LAYOUTS,
    discover_documents,
    docx_layout_pins,
    document_summary,
    generate_documents,
)
from .recording import EvalRecordingBackend, failures_path_for, recording_name
from .replay import MANIFEST_FORMAT, RecordingDir, RecordingError, compare_with_recording, recording_key
from .report import write_results
from .runner import (
    DrawingVariant,
    EvalConfig,
    assemble_results,
    drawing_config_for,
    drawing_non_defaults,
    run_all,
)

__all__ = ["main", "build_parser"]

log = logging.getLogger(__name__)


class UsageError(Exception):
    """A configuration problem reported as ``error: ...`` with exit status 2."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vlmrag.evaluation",
        description="Score the extraction techniques against ground truth: field accuracy, image roles, "
        "row->page placement, must-mention recall, BM25 retrieval, cost and latency.",
    )
    parser.add_argument("--version", action="version", version=f"vlmrag {__version__}")
    parser.add_argument("--out-dir", required=True, help="where results.json and results.md are written")
    parser.add_argument(
        "--docs",
        help="directory of documents with <stem>.truth.json files (default: generate the synthetic set into a "
        "temporary directory that is deleted afterwards)",
    )
    parser.add_argument(
        "--techniques",
        help=f"comma-separated subset of {','.join(TECHNIQUES)} (default: all; on replay: the recorded ones)",
    )
    parser.add_argument(
        "--variants",
        help="drawing pixel-cap variants, comma-separated: cap (packaged 4 MP cap), nocap (no cap, the original "
        "behaviour), maxpx=N (default: cap; on replay: the recorded ones)",
    )
    parser.add_argument("--provider", choices=PROVIDERS, help="override VLMRAG_PROVIDER")
    parser.add_argument("--model", help="override VLMRAG_MODEL (the deployment name on Azure)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--record-dir", help="record every model exchange (and the run manifest) into this directory")
    source.add_argument("--replay-dir", help="replay a recording directory instead of calling a model")
    source.add_argument("--oracle", action="store_true", help="answer from the truth files (harness self-check)")
    parser.add_argument("--no-baseline", action="store_true", help="skip the text-only (NullBackend) baseline runs")
    parser.add_argument("--seed", type=int, default=0, help="synthetic generator seed (default: 0)")
    parser.add_argument(
        "--docx-layout",
        choices=DOCX_LAYOUTS,
        default="auto",
        help="synthetic docx page breaks: auto = LibreOffice layout when installed (use this for model runs: the "
        "renders the model sees then match the truth), estimated = machine-independent estimate",
    )
    parser.add_argument("--label", help="free-text label stored in the results (default: $VLMRAG_EVAL_LABEL)")
    parser.add_argument("--date", help="run date stored in the results (default: $VLMRAG_EVAL_DATE)")
    parser.add_argument("--doc-workers", type=int, default=1, help="documents processed in parallel per run (default: 1)")
    parser.add_argument("--cache-dir", help="response cache for live runs (not allowed with --record-dir/--replay-dir)")
    parser.add_argument("--env-file", default=".env", help="environment file to load if present (default: .env)")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    return parser


# --------------------------------------------------------------------------- helpers


def _csv(value: Optional[str]) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _settings_summary(settings: Settings, cache: bool) -> dict:
    return {
        "reasoning_effort": settings.reasoning_effort,
        "structured_output": settings.structured_output,
        "max_retries": settings.max_retries,
        "timeout_s": settings.timeout_s,
        "max_concurrency": settings.max_concurrency,
        "cache": cache,
    }


# --------------------------------------------------------------------------- main flow


def _run(args: argparse.Namespace) -> int:
    if args.doc_workers < 1:
        raise UsageError("--doc-workers must be >= 1")
    if args.cache_dir and (args.record_dir or args.replay_dir):
        raise UsageError("--cache-dir cannot be combined with --record-dir or --replay-dir: cached answers are "
                         "never sent to the model, so they would be missing from the recording")
    load_dotenv(args.env_file)

    recording: Optional[RecordingDir] = None
    if args.replay_dir:
        try:
            recording = RecordingDir(args.replay_dir)
            config = recording.config(
                _csv(args.techniques), _csv(args.variants),
                baseline=False if args.no_baseline else None, doc_workers=args.doc_workers,
            )
            recording.check_files(config)
        except RecordingError as exc:
            raise UsageError(str(exc)) from exc
    else:
        techniques = _csv(args.techniques) or list(TECHNIQUES)
        unknown = [t for t in techniques if t not in TECHNIQUES]
        if unknown:
            raise UsageError(f"unknown technique(s): {', '.join(unknown)}")
        try:
            variants = tuple(DrawingVariant.parse(v) for v in (_csv(args.variants) or ["cap"]))
        except ValueError as exc:
            raise UsageError(str(exc)) from exc
        config = EvalConfig(
            techniques=tuple(techniques),
            variants=variants,
            baseline=not args.no_baseline,
            doc_workers=args.doc_workers,
        )

    # A replay reports the recorded run: its label and date unless overridden by a flag.
    recorded_header = recording.manifest if recording is not None else {}
    label = args.label or recorded_header.get("label") or os.environ.get("VLMRAG_EVAL_LABEL") or ""
    date = args.date or recorded_header.get("date") or os.environ.get("VLMRAG_EVAL_DATE") or ""

    try:
        settings = Settings.from_env().replace(provider=args.provider, model=args.model)
        if recording is not None:
            settings = recording.settings(settings)
    except ValueError as exc:
        raise UsageError(str(exc)) from exc

    work: Optional[Path] = None
    try:
        if args.docs:
            documents = discover_documents(args.docs, config.techniques)
            generated, seed = False, None
        else:
            work = Path(tempfile.mkdtemp(prefix="vlmrag_eval_"))
            if recording is not None:
                try:
                    documents = recording.generate_documents(work, config.techniques)
                except RecordingError as exc:
                    raise UsageError(str(exc)) from exc
                seed = recording.seed
            else:
                documents = generate_documents(work, config.techniques, seed=args.seed, docx_layout=args.docx_layout)
                seed = args.seed
            generated = True
        if not documents:
            raise UsageError("no documents to evaluate")
        return _evaluate(args, config, settings, documents, generated, seed, recording, label, date)
    finally:
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)


def _evaluate(
    args: argparse.Namespace,
    config: EvalConfig,
    settings: Settings,
    documents: list,
    generated: bool,
    seed: Optional[int],
    recording: Optional[RecordingDir],
    label: str,
    date: str,
) -> int:
    from ..render.office import find_soffice
    from ..vlm.backends import ScriptedBackend, make_backend
    from ..vlm.cache import DiskCache
    from .oracle import oracle_responder

    warnings: list[str] = []
    soffice = find_soffice() is not None
    if not soffice and {"slides", "docx"} & set(config.techniques):
        warnings.append(
            "LibreOffice was not found: slides and docx render no pages, make no model calls and keep their "
            "text-only fallbacks, so their VLM numbers equal the baseline."
        )
    layouts = {d.truth.get("layout_source") for d in documents if d.technique == "docx" and d.truth}
    record_dir = Path(args.record_dir) if args.record_dir else None

    if args.oracle:
        mode = "oracle"
        truths = [d.truth for d in documents if d.truth]
        backend = ScriptedBackend(oracle_responder(truths), model="oracle", name="oracle")
        model_info = {"provider": "oracle", "model": "oracle", "backend": "oracle"}

        def backend_for(technique: str, variant: Optional[str]) -> Any:
            return backend
    elif recording is not None:
        mode = "replay"
        replay = recording
        model_info = {"provider": replay.manifest.get("provider"), "model": replay.model, "backend": "replay"}
        recorded_ids = replay.recorded_doc_ids()
        changed = sorted(
            d.source for d in documents
            if d.source in recorded_ids and recorded_ids[d.source] != document_summary(d)["doc_id"]
        )
        if changed:
            warnings.append(
                "Regenerated documents differ byte-wise from the recorded ones (" + ", ".join(changed) + "); replay "
                "keys ignore bytes, but numbers can change if the content changed."
            )
        if "drawing" in config.techniques:
            local = sorted(name for name, value in os.environ.items()
                           if name.startswith("VLMRAG_DRAWING_") and str(value).strip())
            if local:
                warnings.append(
                    "Ignored during the replay (the recording's drawing settings apply): " + ", ".join(local) + "."
                )
            recorded_cfg = replay.manifest.get("drawing_config") or {}
            if recorded_cfg.get("custom_schema_file") or recorded_cfg.get("domain_terms") == "custom":
                warnings.append(
                    "The recording used a custom drawing schema or domain-terms file, which a replay cannot rebuild; "
                    "expect replay misses."
                )

        def backend_for(technique: str, variant: Optional[str]) -> Any:
            return replay.backend(technique, variant)
    else:
        mode = "record" if record_dir is not None else "live"
        try:
            real = make_backend(settings)
        except Exception as exc:  # noqa: BLE001 - missing credentials etc. end the run with a clear message
            raise UsageError(str(exc)) from exc
        model_info = {"provider": settings.provider, "model": settings.model, "backend": getattr(real, "name", "")}
        if "estimated" in layouts and soffice:
            warnings.append(
                "The docx page-break markers come from the layout estimate while LibreOffice renders the pages the "
                "model sees; rows can sit on other pages in the renders than in the truth. Use --docx-layout auto."
            )
        if record_dir is not None:
            record_dir.mkdir(parents=True, exist_ok=True)
            for technique in config.techniques:
                for variant in (config.variants if technique == "drawing" else (None,)):
                    target = record_dir / recording_name(technique, variant.name if variant else None)
                    for stale in (target, failures_path_for(target)):
                        if stale.exists():
                            stale.unlink()

            def backend_for(technique: str, variant: Optional[str]) -> Any:
                return EvalRecordingBackend(real, record_dir / recording_name(technique, variant))
        else:
            def backend_for(technique: str, variant: Optional[str]) -> Any:
                return real

    cache = DiskCache(args.cache_dir) if args.cache_dir else None
    runs = run_all(documents, settings, config, backend_for, cache=cache)

    pins = docx_layout_pins(documents)
    recorded_settings = dict(recording.manifest.get("settings") or {}) if recording is not None else {}
    header = {
        "label": label,
        "date": date,
        "mode": mode,
        "model": model_info,
        # A replay reports the settings the recorded answers were obtained with, not the replaying machine's.
        "settings": recorded_settings or _settings_summary(settings, cache is not None),
        "environment": {"libreoffice": soffice},
        "documents": {
            "generated": generated,
            "seed": seed,
            "docx_layout": ("pinned" if recording is not None and pins else args.docx_layout) if generated else None,
        },
    }
    results = assemble_results(runs, documents, config, header=header, warnings=warnings)

    status = 0
    if recording is not None:
        recorded = recording.recorded_results()
        diffs = compare_with_recording(recorded, results) if recorded is not None else []
        misses = results.get("replay", {}).get("misses", 0)
        # Recorded answers that were never asked for mean the replay made fewer calls than the recorded run.
        unused = results.get("replay", {}).get("unused", 0)
        results["replay_check"] = {
            "recorded_results": recorded is not None,
            "matches_recording": recorded is not None and not diffs and not misses and not unused,
            "differences": diffs[:50],
        }
        if misses or unused or diffs or recorded is None:
            status = 1

    json_path, md_path = write_results(results, args.out_dir)
    if mode == "record":
        assert record_dir is not None
        _write_manifest(record_dir, config, settings, generated, header, pins, results)
        write_results(results, record_dir)

    print(f"results -> {json_path}")
    print(f"summary -> {md_path}")
    for row in results["cost"]:
        print(f"{row['run']:<28} docs={row['documents']} failed={row['failed']} calls={row['calls']} "
              f"in={row['input_tokens']} out={row['output_tokens']} wall={row['wall_time_s']:.1f}s")
    if mode == "record":
        print(f"recording -> {record_dir}")
    if mode == "replay":
        check = results["replay_check"]
        stats = results.get("replay", {})
        print(f"replay: {stats.get('served', 0)} served, {stats.get('misses', 0)} miss(es), "
              f"{stats.get('unused', 0)} unused; matches recording: {'yes' if check['matches_recording'] else 'no'}")
        for line in check["differences"][:20]:
            print(f"  {line}")
        if not check["recorded_results"]:
            print("  (the recording directory has no results.json to compare with)")
    for warning in warnings:
        print(f"warning: {warning}")
    return status


def _write_manifest(
    record_dir: Path,
    config: EvalConfig,
    settings: Settings,
    generated: bool,
    header: dict,
    pins: dict,
    results: dict,
) -> None:
    files = {}
    for technique in config.techniques:
        for variant in (config.variants if technique == "drawing" else (None,)):
            name = variant.name if variant else None
            files[recording_key(technique, name)] = recording_name(technique, name)
    non_defaults = {}
    drawing_files: dict = {}
    if "drawing" in config.techniques:
        cfg = drawing_config_for(config.variants[0], config.drawing_overrides)
        non_defaults = drawing_non_defaults(cfg)
        # File-based settings cannot be rebuilt from the manifest; recorded so a replay can say so.
        drawing_files = {
            "custom_schema_file": bool(cfg.schema_path),
            "domain_terms": "custom" if cfg.domain_terms_path else "packaged",
        }
    manifest = {
        "format": MANIFEST_FORMAT,
        "package_version": __version__,
        "label": header.get("label", ""),
        "date": header.get("date", ""),
        "provider": settings.provider,
        "model": settings.model,
        "techniques": list(config.techniques),
        "variants": [v.to_dict() for v in config.variants] if "drawing" in config.techniques else [],
        "baseline": config.baseline,
        "files": files,
        "documents": {
            "generated": generated,
            "seed": header["documents"].get("seed"),
            "docx_layout_pins": pins,
            "items": results["documents"]["items"],
        },
        "drawing_non_defaults": non_defaults,
        "drawing_config": drawing_files,
        "drawing_overrides": dict(config.drawing_overrides),
        "settings": header.get("settings", {}),
    }
    (record_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return _run(args)
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
