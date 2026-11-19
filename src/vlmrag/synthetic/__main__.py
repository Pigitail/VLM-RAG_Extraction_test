"""``python -m vlmrag.synthetic --out DIR [--only drawing,slides,docx] [--seed N] [--no-libreoffice]``."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional, Sequence

from . import GENERATORS, generate_all
from ..techniques import TechniqueUnavailable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vlmrag.synthetic",
        description="Write deterministic synthetic documents plus <stem>.truth.json ground truth.",
    )
    parser.add_argument("--out", required=True, help="output directory (one sub-directory per generator)")
    parser.add_argument(
        "--only",
        default="",
        help=f"comma-separated subset of: {', '.join(GENERATORS)} (default: all)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument(
        "--no-libreoffice",
        action="store_true",
        help="never ask LibreOffice for page layout (docx: use the deterministic, machine-independent estimate)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    only = [n.strip() for n in args.only.split(",") if n.strip()] or None
    unknown = [n for n in (only or []) if n not in GENERATORS]
    if unknown:
        print(f"error: unknown generator(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    try:
        options = {"use_libreoffice": False} if args.no_libreoffice else {}
        written = generate_all(args.out, only=only, seed=args.seed, **options)
    except TechniqueUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for name, paths in written.items():
        for path in paths:
            print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
