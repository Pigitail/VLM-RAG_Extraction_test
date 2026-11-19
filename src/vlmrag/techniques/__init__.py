"""Technique registry.

Every technique module exposes::

    extract(path: Path, client: VLMClient, settings: Settings) -> list[Chunk]

and raises on hard failure (the CLI records the error per file and carries on).
Techniques are imported lazily so that ``vlmrag --help`` and the other
techniques keep working when one of them, or its optional dependency, is
missing.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Union

__all__ = ["TECHNIQUES", "SUFFIXES", "TechniqueUnavailable", "load_technique", "resolve_target"]

TECHNIQUES: dict[str, Union[str, Callable[..., Any]]] = {
    "drawing": "vlmrag.techniques.drawing:extract",
    "slides": "vlmrag.techniques.slides:extract",
    "docx": "vlmrag.techniques.docx:extract",
}

# File suffixes each technique accepts when the CLI expands a directory.
SUFFIXES: dict[str, tuple[str, ...]] = {
    "drawing": (".pdf",),
    "slides": (".pptx",),
    "docx": (".docx",),
}


class TechniqueUnavailable(LookupError):
    """The technique name is unknown or its module is not available."""


def resolve_target(target: Union[str, Callable[..., Any]], kind: str, name: str) -> Callable[..., Any]:
    """Import ``"package.module:attr"`` lazily (or return a callable unchanged).

    A missing *target module* becomes a clear :class:`TechniqueUnavailable`;
    an ``ImportError`` raised from inside the module (a real bug or a missing
    dependency of it) is re-raised unchanged so it is not masked.
    """
    if callable(target):
        return target
    module_name, _, attr = str(target).partition(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name and (module_name == exc.name or module_name.startswith(exc.name + ".")):
            raise TechniqueUnavailable(
                f"{kind} {name!r} is not available: module {module_name} is not installed in this build"
            ) from exc
        raise
    try:
        return getattr(module, attr or "extract")
    except AttributeError as exc:
        raise TechniqueUnavailable(f"{kind} {name!r}: {module_name} has no attribute {attr!r}") from exc


def load_technique(name: str) -> Callable[..., Any]:
    """Return the ``extract`` callable registered under ``name``."""
    if name not in TECHNIQUES:
        raise TechniqueUnavailable(f"unknown technique {name!r}; available: {', '.join(sorted(TECHNIQUES))}")
    return resolve_target(TECHNIQUES[name], "technique", name)
