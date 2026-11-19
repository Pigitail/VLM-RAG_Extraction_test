"""Runtime settings, read from environment variables only.

Credentials never appear in code, config files inside the repository, logs or
``repr`` output: they are read from the environment (optionally populated from
a local ``.env`` file that is git-ignored) and the secret fields are excluded
from the dataclass ``repr``.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Union

__all__ = ["Settings", "load_dotenv", "PROVIDERS"]

log = logging.getLogger(__name__)

PROVIDERS = ("openai", "azure", "compatible", "null")
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def _parse_bool(value: str, name: str) -> bool:
    low = value.strip().lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    raise ValueError(f"{name} must be 0/1 (got {value!r})")


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass
class Settings:
    """Everything a technique or the CLI needs to know about the runtime.

    ``provider`` selects the backend family (``openai``, ``azure``,
    ``compatible`` for any OpenAI-compatible server such as vLLM or Ollama, or
    ``null`` for dry runs). ``model`` is the model name, or the deployment name
    on Azure.
    """

    provider: str = "openai"
    model: str = "gpt-6-luna"
    openai_api_key: Optional[str] = field(default=None, repr=False)
    openai_base_url: Optional[str] = None
    azure_api_key: Optional[str] = field(default=None, repr=False)
    azure_endpoint: Optional[str] = None
    azure_api_version: Optional[str] = None
    max_concurrency: int = 4
    timeout_s: float = 600.0
    max_retries: int = 3
    reasoning_effort: Optional[str] = "low"
    structured_output: bool = True
    cache_dir: Optional[str] = None
    soffice: Optional[str] = None
    retrieval_summary_doc_types: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "Settings":
        env = os.environ if environ is None else environ

        def get(name: str) -> Optional[str]:
            value = env.get(name)
            return value.strip() if isinstance(value, str) and value.strip() else None

        settings = cls()
        provider = get("VLMRAG_PROVIDER")
        if provider:
            settings.provider = provider.lower()
        settings.model = get("VLMRAG_MODEL") or settings.model
        settings.openai_api_key = get("OPENAI_API_KEY")
        settings.openai_base_url = get("OPENAI_BASE_URL")
        settings.azure_api_key = get("AZURE_OPENAI_API_KEY")
        settings.azure_endpoint = get("AZURE_OPENAI_ENDPOINT")
        settings.azure_api_version = get("AZURE_OPENAI_API_VERSION")
        if get("VLMRAG_MAX_CONCURRENCY"):
            settings.max_concurrency = max(1, int(get("VLMRAG_MAX_CONCURRENCY")))
        if get("VLMRAG_TIMEOUT_S"):
            settings.timeout_s = float(get("VLMRAG_TIMEOUT_S"))
        if get("VLMRAG_MAX_RETRIES"):
            settings.max_retries = max(0, int(get("VLMRAG_MAX_RETRIES")))
        effort = env.get("VLMRAG_REASONING_EFFORT")
        if effort is not None:
            # An explicitly empty value disables the parameter for models without reasoning.
            settings.reasoning_effort = effort.strip().lower() or None
        if get("VLMRAG_STRUCTURED_OUTPUT") is not None:
            settings.structured_output = _parse_bool(get("VLMRAG_STRUCTURED_OUTPUT"), "VLMRAG_STRUCTURED_OUTPUT")
        settings.cache_dir = get("VLMRAG_CACHE_DIR")
        settings.soffice = get("VLMRAG_SOFFICE")
        if get("VLMRAG_RETRIEVAL_SUMMARY"):
            settings.retrieval_summary_doc_types = _parse_csv(get("VLMRAG_RETRIEVAL_SUMMARY"))
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValueError(f"unknown provider {self.provider!r}; expected one of {', '.join(PROVIDERS)}")
        if self.reasoning_effort not in (None, "minimal", "low", "medium", "high"):
            raise ValueError(f"invalid reasoning effort {self.reasoning_effort!r}")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")

    def replace(self, **changes) -> "Settings":
        """Return a copy with ``changes`` applied (``None`` values are ignored)."""
        changes = {k: v for k, v in changes.items() if v is not None}
        updated = dataclasses.replace(self, **changes)
        updated.validate()
        return updated

    def summary_enabled_for(self, doc_type: str) -> bool:
        return doc_type in self.retrieval_summary_doc_types or "all" in self.retrieval_summary_doc_types


def load_dotenv(path: Union[str, os.PathLike] = ".env", *, override: bool = False) -> bool:
    """Minimal ``.env`` loader (``KEY=VALUE`` lines, ``#`` comments, optional quotes).

    Exists so the CLI can read a local, git-ignored ``.env`` without adding a
    dependency. Values already present in the environment win unless
    ``override`` is true. Returns whether the file existed. Values are never
    logged.
    """
    env_path = Path(path)
    if not env_path.is_file():
        return False
    for raw in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key and (override or key not in os.environ):
            os.environ[key] = value
    log.debug("loaded environment file %s", env_path.name)
    return True
