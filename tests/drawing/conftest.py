"""Fixtures for the drawing technique tests.

The synthetic drawings are generated once per session into a pytest temp
directory (never inside the repository). End-to-end tests use ``LIGHT``, a
low-resolution variant of the default geometry: crops, labels and hints are
identical, only the pixel counts are small enough for a fast test run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import pytest

from vlmrag.synthetic.drawing import generate, load_truth, truth_responder
from vlmrag.techniques.drawing import DrawingConfig

LIGHT = DrawingConfig(
    page_dpi=36,
    page_tile_upscale=1.0,
    title_focus_dpi=72,
    title_focus_upscale=1.0,
    table_focus_dpi=72,
    table_focus_upscale=1.0,
    metadata_dpi=72,
    metadata_upscale=1.0,
)

HOUSING = "lantern_housing.pdf"
REFLECTOR = "lantern_reflector_scan.pdf"
BRACKET = "lantern_bracket.pdf"


@pytest.fixture(scope="session")
def drawing_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("drawings")
    generate(out, seed=0)
    return out


@pytest.fixture(scope="session")
def drawing_docs(drawing_dir) -> dict[str, Path]:
    return {p.name: p for p in sorted(drawing_dir.glob("*.pdf"))}


@pytest.fixture(scope="session")
def truths(drawing_docs) -> dict[str, dict]:
    return {name: load_truth(path) for name, path in drawing_docs.items()}


@pytest.fixture
def light_config() -> DrawingConfig:
    return LIGHT


@pytest.fixture
def responder(truths) -> Callable[..., Callable]:
    """``responder(overrides=None)``: the truth responder with per-task overrides.

    An override is an answer (str/dict) or a callable ``(request, default_answer) -> answer``.
    """

    def factory(overrides: Optional[Mapping[str, Any]] = None) -> Callable:
        base = truth_responder(truths.values())
        table = dict(overrides or {})

        def respond(request):
            if request.task in table:
                override = table[request.task]
                if callable(override):
                    return override(request, lambda: base(request))
                return override
            return base(request)

        return respond

    return factory
