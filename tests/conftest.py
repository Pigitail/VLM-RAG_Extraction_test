"""Shared pytest configuration and fixtures for every test area.

* ``@pytest.mark.soffice`` tests are skipped when LibreOffice is not installed
  (``vlmrag.render.office.find_soffice()`` returns ``None``).
* ``@pytest.mark.network`` tests are skipped unless ``VLMRAG_RUN_NETWORK_TESTS=1``;
  everything else must run offline.
* ``scripted_backend`` / ``scripted_client`` build deterministic model doubles.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Mapping, Optional

import pytest

from vlmrag.render.office import find_soffice
from vlmrag.vlm.backends import ScriptedBackend
from vlmrag.vlm.client import VLMClient
from vlmrag.vlm.types import VLMRequest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    soffice_missing = find_soffice() is None
    network_enabled = os.environ.get("VLMRAG_RUN_NETWORK_TESTS", "").strip() == "1"
    skip_soffice = pytest.mark.skip(reason="LibreOffice (soffice) is not installed")
    skip_network = pytest.mark.skip(reason="network tests are opt-in: set VLMRAG_RUN_NETWORK_TESTS=1")
    for item in items:
        if soffice_missing and "soffice" in item.keywords:
            item.add_marker(skip_soffice)
        if not network_enabled and "network" in item.keywords:
            item.add_marker(skip_network)


@pytest.fixture
def soffice_path() -> str:
    """Path of the LibreOffice binary (skips the test when it is missing)."""
    path = find_soffice()
    if path is None:
        pytest.skip("LibreOffice (soffice) is not installed")
    return path


def _responder_from(spec: Any, by_task: Optional[Mapping[str, Any]]) -> Callable[[VLMRequest], Any]:
    """Build a responder from a callable, a fixed answer, or a ``{task: answer-or-callable}`` table."""
    if by_task is not None:
        table = dict(by_task)

        def respond(request: VLMRequest) -> Any:
            if request.task not in table:
                if spec is not None:
                    return spec(request) if callable(spec) else spec
                raise AssertionError(f"unexpected task {request.task!r}; scripted: {sorted(table)}")
            answer = table[request.task]
            return answer(request) if callable(answer) else answer

        return respond
    if spec is None:
        return lambda request: "{}"
    if callable(spec):
        return spec
    return lambda request: spec


@pytest.fixture
def scripted_backend() -> Callable[..., ScriptedBackend]:
    """Factory: ``scripted_backend(spec=None, *, by_task=None, model="scripted")``.

    ``spec`` is a responder callable or one fixed answer (str/dict/list; dicts
    and lists are JSON-encoded). ``by_task`` maps task names such as
    ``"drawing.vision"`` to an answer or a callable; unknown tasks fall back to
    ``spec`` or fail the test. Inspect ``backend.calls`` afterwards.
    """

    def factory(spec: Any = None, *, by_task: Optional[Mapping[str, Any]] = None, model: str = "scripted") -> ScriptedBackend:
        return ScriptedBackend(_responder_from(spec, by_task), model=model)

    return factory


@pytest.fixture
def scripted_client(scripted_backend) -> Callable[..., VLMClient]:
    """Factory: ``scripted_client(spec=None, *, by_task=None, **client_kwargs)`` -> VLMClient.

    Retry sleeps are no-ops. The backend is ``client.backend`` (with ``.calls``).
    """

    def factory(
        spec: Any = None, *, by_task: Optional[Mapping[str, Any]] = None, model: str = "scripted", **kwargs: Any
    ) -> VLMClient:
        kwargs.setdefault("sleep", lambda seconds: None)
        return VLMClient(scripted_backend(spec, by_task=by_task, model=model), **kwargs)

    return factory


@pytest.fixture
def network_settings():
    """Settings from the real environment, for opt-in network tests only."""
    from vlmrag.config import Settings

    return Settings.from_env()
