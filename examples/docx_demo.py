"""Run the ``docx`` technique on a specification document and print what it produced.

Without ``--input`` the synthetic product specification is generated into a
temporary directory first, so the demo works out of the box::

    python examples/docx_demo.py --provider null            # offline dry run (fallback paths)
    python examples/docx_demo.py --record calls.jsonl       # real model, record the exchanges
    python examples/docx_demo.py --replay calls.jsonl       # replay them, no credentials needed
    python examples/docx_demo.py --input spec.docx --out out

Credentials and the provider come from the environment (or a local ``.env``),
see ``.env.example``. LibreOffice is needed for the model steps (page renders);
without it the pictures are not described and rows of tables that cross pages
are placed from the file's own page-break markers.
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
from vlmrag.techniques.docx import extract
from vlmrag.vlm.backends import RecordingBackend, ReplayBackend, make_backend
from vlmrag.vlm.client import VLMClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Specification DOCX -> one retrieval chunk per source page.")
    parser.add_argument("--input", help="a .docx file (default: generate the synthetic specification)")
    parser.add_argument("--out", help="output directory for the JSONL (default: a temporary directory)")
    parser.add_argument("--provider", choices=PROVIDERS, help="override VLMRAG_PROVIDER")
    parser.add_argument("--model", help="override VLMRAG_MODEL")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--record", metavar="FILE", help="record model exchanges to this JSONL file")
    group.add_argument("--replay", metavar="FILE", help="answer model calls from a recorded JSONL file")
    parser.add_argument("--seed", type=int, default=0, help="seed for the synthetic document (default: 0)")
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
        document = Path(args.input)
    else:
        from vlmrag.synthetic.docx import generate

        work = Path(tempfile.mkdtemp(prefix="vlmrag_docx_demo_"))
        [document] = generate(work, seed=args.seed)
        truth = json.loads((work / f"{document.stem}.truth.json").read_text(encoding="utf-8"))
        print(f"synthetic document: {document} (layout from {truth['layout_source']})")

    try:
        settings = Settings.from_env().replace(provider=args.provider, model=args.model)
        client = _client(args, settings)
    except Exception as exc:  # noqa: BLE001 - configuration problems: explain and stop
        print(f"error: {exc}\nhint: use --provider null for an offline dry run", file=sys.stderr)
        return 2

    chunks = extract(document, client, settings)

    out_dir = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="vlmrag_docx_out_"))
    out_path = out_dir / f"{document.stem}.jsonl"
    write_jsonl(chunks, out_path)

    facts = {p["page"]: p["must_mention"] for p in truth["pages"]} if truth else {}
    split = chunks[0].provenance["page_split"] if chunks else {}
    print(f"\n{document.name}: {len(chunks)} chunk(s) -> {out_path}")
    print(f"page split: {split.get('mode')} (app.xml pages={split.get('app_xml_pages')}, "
          f"rendered breaks={split.get('rendered_breaks')}, padded={split.get('padded_pages')})")
    print(f"{'page':>4}  {'chars':>6}  {'visuals':>7}  {'table rows (source)':<26} {'facts':<6} fallbacks")
    for chunk in chunks:
        tables = ", ".join(
            f"{t['table_id']}:{len(t['rows'])} ({t['assignment']})" for t in chunk.metadata["multi_page_tables"]
        ) or "-"
        expected = facts.get(chunk.page, [])
        text = chunk.retrieve_content.lower()
        found = f"{sum(f.lower() in text for f in expected)}/{len(expected)}" if expected else "-"
        fallbacks = ",".join(chunk.provenance["fallbacks"]) or "-"
        print(f"{chunk.page:>4}  {len(chunk.retrieve_content):>6}  {len(chunk.metadata['visual_descriptions']):>7}  "
              f"{tables[:26]:<26} {found:<6} {fallbacks}")
    print()
    print(client.usage.format_table())
    return 0


if __name__ == "__main__":
    sys.exit(main())
