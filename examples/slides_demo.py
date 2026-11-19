"""Run the ``slides`` technique on a deck and print what it produced.

Without ``--input`` the synthetic software-manual deck is generated into a
temporary directory first, so the demo works out of the box::

    python examples/slides_demo.py --provider null            # offline dry run (fallback paths)
    python examples/slides_demo.py --record calls.jsonl       # real model, record the exchanges
    python examples/slides_demo.py --replay calls.jsonl       # replay them, no credentials needed
    python examples/slides_demo.py --input manual.pptx --out out --assets out/assets

Credentials and the provider come from the environment (or a local ``.env``),
see ``.env.example``. LibreOffice is needed to render slides with pictures;
without it every slide falls back to its XML text.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from vlmrag.config import PROVIDERS, Settings, load_dotenv
from vlmrag.schema import write_jsonl
from vlmrag.techniques.slides import extract
from vlmrag.vlm.backends import RecordingBackend, ReplayBackend, make_backend
from vlmrag.vlm.client import VLMClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Software-manual slides -> grounded retrieval chunks.")
    parser.add_argument("--input", help="a .pptx deck (default: generate the synthetic deck)")
    parser.add_argument("--out", help="output directory for the JSONL (default: a temporary directory)")
    parser.add_argument("--assets", help="also write crop-aware image assets and layout.json sidecars here")
    parser.add_argument("--provider", choices=PROVIDERS, help="override VLMRAG_PROVIDER")
    parser.add_argument("--model", help="override VLMRAG_MODEL")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--record", metavar="FILE", help="record model exchanges to this JSONL file")
    group.add_argument("--replay", metavar="FILE", help="answer model calls from a recorded JSONL file")
    parser.add_argument("--seed", type=int, default=0, help="seed for the synthetic deck (default: 0)")
    parser.add_argument("--env-file", default=".env", help="environment file to load if present")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    return parser


def _client(args: argparse.Namespace, settings: Settings) -> VLMClient:
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

    truth = None
    if args.input:
        deck = Path(args.input)
    else:
        from vlmrag.synthetic.slides import generate

        work = Path(tempfile.mkdtemp(prefix="vlmrag_slides_demo_"))
        [deck] = generate(work, seed=args.seed)
        truth = json.loads((work / f"{deck.stem}.truth.json").read_text(encoding="utf-8"))
        print(f"synthetic deck: {deck}")

    try:
        settings = Settings.from_env().replace(provider=args.provider, model=args.model)
        client = _client(args, settings)
    except Exception as exc:  # noqa: BLE001 - configuration problems: explain and stop
        print(f"error: {exc}\nhint: use --provider null for an offline dry run", file=sys.stderr)
        return 2

    chunks = extract(deck, client, settings, asset_dir=args.assets)

    out_dir = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="vlmrag_slides_out_"))
    out_path = out_dir / f"{deck.stem}.jsonl"
    write_jsonl(chunks, out_path)

    facts = {s["slide"]: s["must_mention"] for s in truth["slides"]} if truth else {}
    print(f"\n{deck.name}: {len(chunks)} chunk(s) -> {out_path}")
    print(f"{'slide':>5}  {'visual':<10} {'roles':<44} {'facts':<6} fallbacks")
    for chunk in chunks:
        call = chunk.provenance.get("visual_call", {})
        status = call.get("status", "-")
        roles = ",".join(d.get("image_role", "?") for d in chunk.metadata.get("image_descriptions", [])) or "-"
        expected = facts.get(chunk.page, [])
        found = f"{sum(f in chunk.retrieve_content for f in expected)}/{len(expected)}" if expected else "-"
        fallbacks = ",".join(chunk.provenance.get("fallbacks", [])) or "-"
        print(f"{chunk.page:>5}  {status:<10} {roles[:44]:<44} {found:<6} {fallbacks}")
    if args.assets:
        print(f"assets -> {Path(args.assets).resolve()}")
    print()
    print(client.usage.format_table())
    return 0


if __name__ == "__main__":
    sys.exit(main())
