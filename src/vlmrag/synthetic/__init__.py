"""Deterministic synthetic fixtures (registry + CLI).

Real engineering drawings, manuals and specifications cannot be published, so
every technique ships a generator that writes look-alike documents with
invented content and a ``<stem>.truth.json`` ground-truth file next to each
document. Generators are deterministic for a given ``seed``.

Generator contract::

    generate(out_dir: Path, *, seed: int = 0, **options) -> list[Path]

Options are generator specific (for example ``use_libreoffice=False`` makes
the ``docx`` generator use its LibreOffice-independent layout estimate);
:func:`generate_all` passes each option only to the generators that accept it.

Run ``python -m vlmrag.synthetic --out DIR [--only drawing,slides,docx] [--no-libreoffice]``.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

from ..techniques import TechniqueUnavailable, resolve_target

__all__ = ["GENERATORS", "load_generator", "generate_all", "accepted_options"]

GENERATORS: dict[str, Union[str, Callable[..., Any]]] = {
    "drawing": "vlmrag.synthetic.drawing:generate",
    "slides": "vlmrag.synthetic.slides:generate",
    "docx": "vlmrag.synthetic.docx:generate",
}


def load_generator(name: str) -> Callable[..., Any]:
    if name not in GENERATORS:
        raise TechniqueUnavailable(f"unknown generator {name!r}; available: {', '.join(sorted(GENERATORS))}")
    return resolve_target(GENERATORS[name], "generator", name)


def accepted_options(generate: Callable[..., Any], options: dict[str, Any]) -> dict[str, Any]:
    """The subset of ``options`` that ``generate`` declares as keyword parameters."""
    try:
        params = inspect.signature(generate).parameters
    except (TypeError, ValueError):
        return {}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(options)
    return {k: v for k, v in options.items() if k in params}


def generate_all(
    out_dir: Union[str, Path],
    *,
    only: Optional[Iterable[str]] = None,
    seed: int = 0,
    **options: Any,
) -> dict[str, list[Path]]:
    """Run the selected generators into ``out_dir/<name>/``; returns written paths per generator.

    ``options`` (for example ``use_libreoffice=False``) reach only the
    generators whose ``generate`` accepts them, so one call can configure a
    single generator without breaking the others.
    """
    root = Path(out_dir)
    names = list(only) if only else list(GENERATORS)
    written: dict[str, list[Path]] = {}
    for name in names:
        generate = load_generator(name)
        target = root / name
        target.mkdir(parents=True, exist_ok=True)
        written[name] = [Path(p) for p in generate(target, seed=seed, **accepted_options(generate, options))]
    return written
