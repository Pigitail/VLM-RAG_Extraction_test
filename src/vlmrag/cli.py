"""Command-line interface.

::

    vlmrag run <technique> INPUT... --out out.jsonl [--provider P] [--model M]
               [--record F | --replay F] [--cache-dir D] [--summary] [--workers N]
    vlmrag techniques

The batch runner isolates every file: an exception while processing one input
is written to ``<out stem>.errors.jsonl`` and the batch moves on. (A pipeline
that collects results from a worker pool inside one ``try`` block loses every
remaining result as soon as one file fails; this runner never does.) Chunks are
written in input order as soon as all earlier files are done, so an interrupted
run keeps its completed prefix.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from . import __version__
from .config import PROVIDERS, Settings, load_dotenv
from .schema import Chunk, append_jsonl
from .techniques import SUFFIXES, TECHNIQUES, TechniqueUnavailable, load_technique

__all__ = ["main", "build_parser", "run_batch", "BatchResult", "errors_path_for", "expand_inputs"]

log = logging.getLogger(__name__)


def errors_path_for(out: Path) -> Path:
    """``out.jsonl`` -> ``out.errors.jsonl`` (``out`` -> ``out.errors.jsonl``)."""
    out = Path(out)
    stem = out.name[: -len(out.suffix)] if out.suffix else out.name
    return out.with_name(f"{stem}.errors.jsonl")


def expand_inputs(inputs: Sequence[str], technique: str) -> list[Path]:
    """Files are kept as given; directories are searched recursively for the technique's suffixes.

    Office lock files (``~$name.docx``) are skipped: they are not documents.
    Missing paths are kept so the runner reports them as per-file errors.
    """
    suffixes = SUFFIXES.get(technique, ())
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            found = sorted(
                p for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in suffixes and not p.name.startswith("~$")
            )
            paths.extend(found)
        else:
            paths.append(path)
    return paths


@dataclass
class BatchResult:
    files: int = 0
    succeeded: int = 0
    failed: int = 0
    chunks: int = 0
    errors: list[dict] = field(default_factory=list)


def _error_record(path: Path, technique: str, exc: BaseException) -> dict:
    return {
        "source": str(path),
        "source_name": path.name,
        "technique": technique,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def run_batch(
    technique: str,
    paths: Sequence[Path],
    out: Path,
    client: Any,
    settings: Settings,
    *,
    workers: int = 1,
    extract: Optional[Callable[..., list]] = None,
) -> BatchResult:
    """Run ``technique`` over ``paths``; never stops on a per-file failure.

    ``out`` is truncated first; its error file (:func:`errors_path_for`) is
    removed and rewritten only when there are errors. When the technique's doc
    type is enabled in ``settings.retrieval_summary_doc_types``, chunks get the
    optional retrieval-summary prefix.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("", encoding="utf-8")
    errors_path = errors_path_for(out)
    if errors_path.exists():
        errors_path.unlink()
    result = BatchResult(files=len(paths))
    extract_fn = extract or load_technique(technique)
    summary_on = settings.summary_enabled_for(technique)
    lock = threading.Lock()
    pending: dict[int, list[Chunk]] = {}
    next_index = [0]

    def process(path: Path) -> list[Chunk]:
        if not path.is_file():
            raise FileNotFoundError(f"input not found: {path}")
        chunks = list(extract_fn(path, client, settings))
        if summary_on and chunks:
            from .augment.summary import augment_chunks

            chunks = augment_chunks(client, chunks, workers=settings.max_concurrency)
        return chunks

    def flush() -> None:
        # Write finished results strictly in input order.
        while next_index[0] in pending:
            chunks = pending.pop(next_index[0])
            if chunks:
                append_jsonl(chunks, out)
            next_index[0] += 1

    def record_error(path: Path, exc: BaseException) -> None:
        record = _error_record(path, technique, exc)
        result.errors.append(record)
        result.failed += 1
        with errors_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.error("%s: %s: %s", path.name, type(exc).__name__, exc)

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {pool.submit(process, path): (index, path) for index, path in enumerate(paths)}
        for future in as_completed(futures):
            index, path = futures[future]
            try:
                chunks = future.result()
            except Exception as exc:  # noqa: BLE001 - per-file isolation is the point
                with lock:
                    record_error(path, exc)
                    pending[index] = []
                    flush()
                continue
            with lock:
                result.succeeded += 1
                result.chunks += len(chunks)
                pending[index] = chunks
                flush()
                log.info("%s: %d chunk(s)", path.name, len(chunks))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vlmrag",
        description="VLM-assisted document extraction for retrieval-augmented generation.",
    )
    parser.add_argument("--version", action="version", version=f"vlmrag {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    run = sub.add_parser(
        "run",
        help="extract chunks from documents with one technique",
        description="Extract chunks from INPUT files (or directories) and write them as JSONL. "
        "Per-file failures are written to <out stem>.errors.jsonl and never stop the batch.",
    )
    run.add_argument("technique", help=f"one of: {', '.join(TECHNIQUES)}")
    run.add_argument("inputs", nargs="+", metavar="INPUT", help="input files or directories")
    run.add_argument("--out", required=True, help="output JSONL file")
    run.add_argument("--provider", choices=PROVIDERS, help="override VLMRAG_PROVIDER")
    run.add_argument("--model", help="override VLMRAG_MODEL (deployment name on Azure)")
    replay = run.add_mutually_exclusive_group()
    replay.add_argument("--record", metavar="FILE", help="append every model exchange to this JSONL file")
    replay.add_argument("--replay", metavar="FILE", help="answer model calls from a recorded JSONL file (no network)")
    run.add_argument("--cache-dir", help="on-disk response cache directory (overrides VLMRAG_CACHE_DIR)")
    run.add_argument("--summary", action="store_true", help="add the optional retrieval-summary prefix")
    run.add_argument("--workers", type=int, default=1, help="files processed in parallel (default: 1)")
    run.add_argument("--env-file", default=".env", help="environment file to load if present (default: .env)")
    run.add_argument("-v", "--verbose", action="store_true", help="verbose logging")

    sub.add_parser("techniques", help="list registered techniques and whether they are available")
    return parser


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _cmd_techniques() -> int:
    for name in TECHNIQUES:
        try:
            load_technique(name)
            status = "available"
        except TechniqueUnavailable as exc:
            status = f"unavailable ({exc})"
        print(f"{name:<10} {status}")
    return 0


def _build_client(args: argparse.Namespace, settings: Settings):
    from .vlm.backends import RecordingBackend, ReplayBackend, make_backend
    from .vlm.client import VLMClient

    if args.replay:
        backend = ReplayBackend(args.replay)
    else:
        backend = make_backend(settings)
        if args.record:
            backend = RecordingBackend(backend, args.record)
    return VLMClient.from_settings(settings, backend=backend)


def _cmd_run(args: argparse.Namespace) -> int:
    if args.technique not in TECHNIQUES:
        print(f"error: unknown technique {args.technique!r}; available: {', '.join(TECHNIQUES)}", file=sys.stderr)
        return 2
    try:
        extract = load_technique(args.technique)
    except TechniqueUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    load_dotenv(args.env_file)
    try:
        settings = Settings.from_env().replace(provider=args.provider, model=args.model, cache_dir=args.cache_dir)
        if args.summary and args.technique not in settings.retrieval_summary_doc_types:
            settings = dataclasses.replace(
                settings, retrieval_summary_doc_types=settings.retrieval_summary_doc_types + (args.technique,)
            )
        client = _build_client(args, settings)
    except Exception as exc:  # noqa: BLE001 - configuration problems end the run with a clear message
        print(f"error: {exc}", file=sys.stderr)
        return 2
    paths = expand_inputs(args.inputs, args.technique)
    if not paths:
        print("error: no input files found", file=sys.stderr)
        return 2
    out = Path(args.out)
    result = run_batch(args.technique, paths, out, client, settings, workers=args.workers, extract=extract)
    print(
        f"{result.files} file(s): {result.succeeded} ok, {result.failed} failed, "
        f"{result.chunks} chunk(s) -> {out}"
    )
    if result.failed:
        print(f"errors -> {errors_path_for(out)}")
    print(client.usage.format_table())
    return 1 if result.failed else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2
    _configure_logging(getattr(args, "verbose", False))
    if args.command == "techniques":
        return _cmd_techniques()
    return _cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
