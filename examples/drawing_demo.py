"""Run the ``drawing`` technique on engineering-drawing PDFs and print what it produced.

Without ``--input`` the synthetic A3 drawings are generated into a temporary
directory first, so the demo works out of the box::

    python examples/drawing_demo.py --simulate               # offline, scripted "perfect" model from ground truth
    python examples/drawing_demo.py --provider null --fast   # offline dry run: every sheet fails the gate (by design, exit 0)
    python examples/drawing_demo.py --record calls.jsonl     # real model, record the exchanges
    python examples/drawing_demo.py --replay calls.jsonl     # replay them, no credentials needed
    python examples/drawing_demo.py --input sheet.pdf --out out --pages all

Credentials and the provider come from the environment (or a local ``.env``),
see ``.env.example``. Drawing options (crop geometry, schema, domain terms)
come from ``VLMRAG_DRAWING_*`` variables, see ``docs/techniques/drawing.md``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from vlmrag.config import PROVIDERS, Settings, load_dotenv
from vlmrag.schema import write_jsonl
from vlmrag.techniques.drawing import DrawingConfig, TitleBlockGateError, extract
from vlmrag.vlm.backends import RecordingBackend, ReplayBackend, ScriptedBackend, make_backend
from vlmrag.vlm.client import VLMClient

# Same crops and labels as the defaults, at a fraction of the pixels: for dry runs, not for real models.
FAST = dict(
    page_dpi=72,
    page_tile_upscale=1.0,
    title_focus_dpi=150,
    title_focus_upscale=1.0,
    table_focus_dpi=150,
    table_focus_upscale=1.0,
    metadata_dpi=150,
    metadata_upscale=1.0,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Engineering-drawing PDFs -> title-block retrieval chunks.")
    parser.add_argument("--input", nargs="+", help="drawing PDF(s) (default: generate the synthetic drawings)")
    parser.add_argument("--out", help="output directory for the JSONL (default: a temporary directory)")
    parser.add_argument("--provider", choices=PROVIDERS, help="override VLMRAG_PROVIDER")
    parser.add_argument("--model", help="override VLMRAG_MODEL")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--record", metavar="FILE", help="record model exchanges to this JSONL file")
    group.add_argument("--replay", metavar="FILE", help="answer model calls from a recorded JSONL file")
    group.add_argument("--simulate", action="store_true", help="answer from the synthetic ground truth (offline)")
    parser.add_argument("--pages", choices=("first", "all"), help="read the title block on page 1 only, or on every page")
    parser.add_argument("--fast", action="store_true", help="low-resolution crops (dry runs only)")
    parser.add_argument("--seed", type=int, default=0, help="seed for the synthetic drawings (default: 0)")
    parser.add_argument("--env-file", default=".env", help="environment file to load if present")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    return parser


def _client(args: argparse.Namespace, settings: Settings, truths: list[dict]) -> VLMClient:
    if args.simulate:
        from vlmrag.synthetic.drawing import truth_responder

        if not truths:
            raise ValueError("--simulate needs the synthetic drawings (omit --input)")
        return VLMClient(ScriptedBackend(truth_responder(truths), model="simulated"), max_concurrency=settings.max_concurrency)
    if args.replay:
        backend = ReplayBackend(args.replay)
    else:
        backend = make_backend(settings)
        if args.record:
            backend = RecordingBackend(backend, args.record)
    return VLMClient.from_settings(settings, backend=backend)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    load_dotenv(args.env_file)

    truths: dict[str, dict] = {}
    if args.input:
        documents = [Path(p) for p in args.input]
    else:
        from vlmrag.synthetic.drawing import generate, load_truth

        work = Path(tempfile.mkdtemp(prefix="vlmrag_drawing_demo_"))
        documents = generate(work, seed=args.seed)
        truths = {p.name: load_truth(p) for p in documents}
        print(f"synthetic drawings: {work}")

    try:
        settings = Settings.from_env().replace(provider=args.provider, model=args.model)
        overrides = dict(FAST) if args.fast else {}
        if args.pages:
            overrides["title_block_pages"] = args.pages
        config = DrawingConfig.from_env(**overrides)
        client = _client(args, settings, list(truths.values()))
    except Exception as exc:  # noqa: BLE001 - configuration problems: explain and stop
        print(f"error: {exc}\nhint: use --simulate or --provider null for an offline run", file=sys.stderr)
        return 2

    chunks, failures = [], []
    gate_failures = 0
    for document in documents:
        try:
            chunks.extend(extract(document, client, settings, config=config))
        except Exception as exc:  # noqa: BLE001 - like the CLI: record the failure and continue
            gate_failures += isinstance(exc, TitleBlockGateError)
            failures.append((document.name, f"{type(exc).__name__}: {exc}"))

    out_dir = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="vlmrag_drawing_out_"))
    out_path = out_dir / "drawing_chunks.jsonl"
    write_jsonl(chunks, out_path)

    print(f"\n{len(documents)} file(s): {len(chunks)} chunk(s) -> {out_path}")
    print(f"{'file':<28} {'page':>4} {'tb':>3} {'drawing no.':<14} {'pairs':>5} {'rows':>4} {'truth':<6} fallbacks")
    for chunk in chunks:
        meta = chunk.metadata
        truth = truths.get(chunk.source_name)
        expected = truth["pages"][meta["title_block_page"] - 1]["title_block"] if truth else {}
        found = f"{sum(meta['title_block'].get(k) == v for k, v in expected.items())}/{len(expected)}" if expected else "-"
        print(
            f"{chunk.source_name[:28]:<28} {chunk.page:>4} {meta['title_block_page']:>3} "
            f"{meta['title_block'].get('drawing_number', '-')[:14]:<14} {len(meta['title_block_pairs']):>5} "
            f"{len(meta['table_rows']):>4} {found:<6} {','.join(chunk.provenance['fallbacks']) or '-'}"
        )
    for name, error in failures:
        print(f"{name[:28]:<28} FAILED {error}")
    if truths:
        queries = [q for t in truths.values() for q in t["queries"]]
        answered = sum(
            any(c.source_name == r["source"] and c.page == r["page"] and q["evidence"] in c.retrieve_content
                for c in chunks for r in q["relevant"])
            for q in queries
        )
        print(f"\nvisual-only facts present in the relevant chunk: {answered}/{len(queries)}")
    print()
    print(client.usage.format_table())
    # The null backend answers "{}", so every sheet is expected to fail the gate: that dry run
    # exercises rendering, requests and the failure path, and counts as a success for this demo.
    dry_run = settings.provider == "null" and not (args.simulate or args.replay)
    if dry_run and failures and gate_failures == len(failures):
        print("\nnull provider: every sheet failed the title-block gate, as expected for a dry run")
        return 0
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
