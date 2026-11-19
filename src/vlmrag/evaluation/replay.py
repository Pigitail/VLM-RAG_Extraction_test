"""Recording directories: what a recording run leaves behind and how a replay reads it back.

A recording directory (``--record-dir``) holds

* ``<technique>[.<variant>].jsonl`` - the model exchanges of one technique run
  (plus ``.failures.jsonl`` for calls that failed);
* ``manifest.json`` - everything needed to rebuild the *requests*: techniques,
  drawing variants, generator seed, the docx layout pins, drawing settings
  that differ from the defaults, the recorded model name;
* ``results.json`` / ``results.md`` - the numbers of the recorded run.

:class:`RecordingDir` turns a directory back into documents, configuration
and replay backends; :func:`compare_with_recording` checks that a replayed
run reproduced the recorded numbers. The command line and the regression
test under ``tests/replay`` both use these, so they cannot drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

from .documents import EvalDocument, generate_documents
from .metrics import compare_metrics
from .recording import EvalReplayBackend
from .runner import (
    DrawingVariant,
    EvalConfig,
    TechniqueRun,
    attach_backend_stats,
    cost_fingerprint,
    run_technique,
)

__all__ = [
    "MANIFEST_FORMAT",
    "RecordingDir",
    "RecordingError",
    "compare_with_recording",
    "recording_key",
    "replay_technique",
]

MANIFEST_FORMAT = "vlmrag-evaluation-recording/1"


class RecordingError(ValueError):
    """The recording directory is missing, incomplete or of an unknown format."""


def recording_key(technique: str, variant: Optional[str] = None) -> str:
    """Key of a technique run in ``manifest["files"]``: ``slides``, ``drawing@nocap``."""
    return f"{technique}@{variant}" if variant else technique


class RecordingDir:
    def __init__(self, directory: Union[str, Path]) -> None:
        self.directory = Path(directory)
        path = self.directory / "manifest.json"
        if not path.is_file():
            raise RecordingError(f"{self.directory} has no manifest.json; record with --record-dir first")
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != MANIFEST_FORMAT:
            raise RecordingError(
                f"unsupported recording format {self.manifest.get('format')!r} (expected {MANIFEST_FORMAT})"
            )

    # ------------------------------------------------------------------ manifest views

    @property
    def techniques(self) -> list[str]:
        return list(self.manifest.get("techniques") or [])

    @property
    def variants(self) -> tuple[DrawingVariant, ...]:
        return tuple(
            DrawingVariant(str(v["name"]), int(v["max_image_pixels"])) for v in self.manifest.get("variants") or []
        )

    @property
    def model(self) -> str:
        return str(self.manifest.get("model") or "replay")

    @property
    def generated(self) -> bool:
        return bool((self.manifest.get("documents") or {}).get("generated", True))

    @property
    def seed(self) -> int:
        return int((self.manifest.get("documents") or {}).get("seed") or 0)

    @property
    def docx_pins(self) -> dict:
        return dict((self.manifest.get("documents") or {}).get("docx_layout_pins") or {})

    @property
    def drawing_overrides(self) -> dict:
        overrides = dict(self.manifest.get("drawing_non_defaults") or {})
        overrides.update(self.manifest.get("drawing_overrides") or {})
        return overrides

    def config(
        self,
        techniques: Optional[Iterable[str]] = None,
        variants: Optional[Iterable[str]] = None,
        *,
        baseline: Optional[bool] = None,
        doc_workers: int = 1,
    ) -> EvalConfig:
        """An :class:`EvalConfig` for (a subset of) what was recorded; unknown names raise."""
        chosen = list(techniques or []) or self.techniques
        missing = [t for t in chosen if t not in self.techniques]
        if missing:
            raise RecordingError(
                f"technique(s) {', '.join(missing)} not in the recording (recorded: {', '.join(self.techniques)})"
            )
        recorded = self.variants
        names = list(variants or []) or [v.name for v in recorded]
        unknown = [n for n in names if n not in {v.name for v in recorded}]
        if unknown and recorded:
            raise RecordingError(
                f"drawing variant(s) {', '.join(unknown)} not in the recording "
                f"(recorded: {', '.join(v.name for v in recorded)})"
            )
        selected = tuple(v for v in recorded if v.name in names)
        return EvalConfig(
            techniques=tuple(chosen),
            variants=selected or (DrawingVariant.parse("cap"),),
            baseline=bool(self.manifest.get("baseline", True)) if baseline is None else baseline,
            drawing_overrides=self.drawing_overrides,
            doc_workers=doc_workers,
            # The manifest alone defines the drawing configuration of a replay (see EvalConfig).
            drawing_env=False,
        )

    def settings(self, base: Any = None) -> Any:
        """Settings for a replay: request-shaping values from the recording, machine-specific ones from ``base``.

        The reasoning effort is not part of the replay key, so a replay would
        otherwise send (and report) whatever the replaying machine's
        environment says instead of what the recorded run used.
        ``base`` (default: packaged defaults) still provides machine-specific
        values such as the LibreOffice path.
        """
        import dataclasses

        from ..config import Settings

        base = base if base is not None else Settings(provider="null")
        recorded = self.manifest.get("settings") or {}
        changes: dict = {"provider": "null", "model": self.model, "cache_dir": None}
        if "reasoning_effort" in recorded:
            changes["reasoning_effort"] = recorded["reasoning_effort"] or None
        if "structured_output" in recorded:
            changes["structured_output"] = bool(recorded["structured_output"])
        settings = dataclasses.replace(base, **changes)
        settings.validate()
        return settings

    def check_files(self, config: EvalConfig) -> None:
        files = self.manifest.get("files") or {}
        for technique in config.techniques:
            for variant in (config.variants if technique == "drawing" else (None,)):
                key = recording_key(technique, variant.name if variant else None)
                if key not in files or not (self.directory / files[key]).is_file():
                    raise RecordingError(f"recording for {key} not found in {self.directory}")

    # ------------------------------------------------------------------ replay inputs

    def generate_documents(self, out_dir: Union[str, Path], techniques: Iterable[str]) -> list[EvalDocument]:
        """Rebuild the recorded synthetic documents (same seed, same docx page layout)."""
        if not self.generated:
            raise RecordingError("this recording used user-supplied documents; pass the same ones with --docs")
        return generate_documents(out_dir, list(techniques), seed=self.seed, docx_pins=self.docx_pins or None)

    def backend(self, technique: str, variant: Optional[str] = None) -> EvalReplayBackend:
        files = self.manifest.get("files") or {}
        key = recording_key(technique, variant)
        if key not in files:
            raise RecordingError(f"recording for {key} not found in {self.directory}")
        return EvalReplayBackend(self.directory / files[key], model=self.model)

    def recorded_results(self) -> Optional[dict]:
        path = self.directory / "results.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def recorded_doc_ids(self) -> dict[str, str]:
        items = (self.manifest.get("documents") or {}).get("items") or []
        return {item["source"]: item.get("doc_id") for item in items}


def compare_with_recording(recorded: dict, replayed: dict) -> list[str]:
    """Differences between a recorded and a replayed ``results.json`` (quality metrics and token counts).

    Only what both runs evaluated is compared: technique sections and modes
    present in the replay, and retrieval only when both ran the same
    techniques (the index contains every technique's chunks). Wall times are
    never compared.
    """
    diffs: list[str] = []
    rec_metrics, rep_metrics = recorded.get("metrics", {}), replayed.get("metrics", {})
    for technique, section in rep_metrics.items():
        if technique == "retrieval":
            continue
        rec_section = rec_metrics.get(technique)
        if rec_section is None:
            diffs.append(f"metrics.{technique}: not in the recording")
            continue
        for mode, data in section.items():
            if mode not in rec_section:
                continue
            if technique == "drawing":
                for variant, values in data.items():
                    if variant in rec_section[mode]:
                        diffs += compare_metrics(rec_section[mode][variant], values, f"metrics.drawing.{mode}.{variant}")
            else:
                diffs += compare_metrics(rec_section[mode], data, f"metrics.{technique}.{mode}")
    same_techniques = set(recorded.get("evaluation", {}).get("techniques", [])) == set(
        replayed.get("evaluation", {}).get("techniques", [])
    )
    if same_techniques:
        for name, data in rep_metrics.get("retrieval", {}).items():
            if name in rec_metrics.get("retrieval", {}):
                diffs += compare_metrics(rec_metrics["retrieval"][name], data, f"metrics.retrieval.{name}")
    rec_cost, rep_cost = cost_fingerprint(recorded), cost_fingerprint(replayed)
    for run, values in rep_cost.items():
        if run in rec_cost:
            diffs += compare_metrics(rec_cost[run], values, f"cost.{run}")
    return diffs


def replay_technique(
    recording: RecordingDir,
    documents: Sequence[EvalDocument],
    technique: str,
    config: EvalConfig,
    settings: Any = None,
) -> list[TechniqueRun]:
    """Replay every recorded run of ``technique`` (each drawing variant) plus its baseline run.

    Each returned VLM run carries ``replay`` statistics (served answers,
    misses with their task/source/page, unused answers, replayed failures).
    """
    from ..vlm.backends import NullBackend

    settings = settings if settings is not None else recording.settings()
    runs: list[TechniqueRun] = []
    for variant in (config.variants if technique == "drawing" else (None,)):
        backend = recording.backend(technique, variant.name if variant else None)
        run = run_technique(
            documents, technique, mode="vlm", backend=backend, settings=settings, variant=variant, config=config
        )
        attach_backend_stats(run, backend)
        runs.append(run)
    if config.baseline:
        runs.append(run_technique(
            documents, technique, mode="baseline", backend=NullBackend(), settings=settings,
            variant=config.variants[0] if technique == "drawing" else None, config=config,
        ))
    return runs
