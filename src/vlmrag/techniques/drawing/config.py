"""Settings of the drawing technique: render geometry, limits, title-block schema, domain terms.

Everything that decides *what the model sees* lives here as data, so it can be
reviewed in one place and changed without touching the pipeline:

* :class:`DrawingConfig` - crop geometry, resolutions, call limits and the
  acceptance gate. The defaults are the values the technique was tuned with on
  real drawing sets; every field can be overridden with an environment variable
  ``VLMRAG_DRAWING_<FIELD NAME IN UPPER CASE>``.
* :class:`TitleBlockSchema` - the canonical fields the model is asked for, their
  display labels and the printed label variants that map onto them. Title-block
  vocabulary differs between organisations, so it is a JSON file
  (``configs/title_block_schema.example.json``) rather than code.
* :class:`DomainTerms` - retrieval-critical vocabulary (terms whose rows must be
  kept separately, part names, sub-part tokens, synonyms). It shapes the prompts
  and the retrieval text and is empty-able for domains that need none.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Optional, Union

__all__ = [
    "DrawingConfig",
    "FieldSpec",
    "TitleBlockSchema",
    "TermGroup",
    "PartSpec",
    "DomainTerms",
    "normalize_label",
    "TITLE_BLOCK_PAGE_MODES",
]

log = logging.getLogger(__name__)

TITLE_BLOCK_PAGE_MODES = ("first", "all")
_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_RESERVED_KEYS = {"pairs", "table_rows"}
_SCHEMA_RESOURCE = "title_block_schema.json"
_DOMAIN_RESOURCE = "domain_terms.json"


def normalize_label(text: Any) -> str:
    """Canonical form of a printed label, used for every label comparison.

    Printed labels vary in case, separators and trailing colons
    (``"Drawing_No:"``, ``"drawing no"``, ``"DRAWING  NO"``); comparing this
    form lets one alias table cover all of them. Slashes get single spaces on
    both sides so ``"P/N"`` and ``"P / N"`` meet.
    """
    value = "" if text is None else str(text)
    value = value.strip().lower()
    value = value.replace("’", "'").replace("`", "'")
    value = value.replace("_", " ")
    value = re.sub(r"\s*/\s*", " / ", value)
    value = re.sub(r"\s+", " ", value)
    return value.rstrip(":")


def _alias_pattern(alias: str) -> str:
    """Regex for an alias as it may be echoed inside a value (``"gen. tol"`` -> ``gen\\.?\\s*tol``)."""
    tokens = [t for t in re.split(r"\s+", alias.strip().lower()) if t]
    parts = []
    for token in tokens:
        if token.endswith("."):
            parts.append(re.escape(token[:-1]) + r"\.?")
        else:
            parts.append(re.escape(token))
    return r"\s*".join(parts)


# --------------------------------------------------------------------------- title-block schema


@dataclass(frozen=True)
class FieldSpec:
    """One canonical title-block field.

    ``table_field`` marks values that usually sit in the small ruled table of
    the title block: models tend to copy neighbouring labels and grid letters
    into those values, so they get an extra cleaning pass. ``row_field`` marks
    fields that legitimately hold one value per row (for example a finish per
    part); their cleaned segments are kept apart with ``" | "``. ``join`` is the
    separator used when a value continues over several lines of a free-text
    answer.
    """

    key: str
    label: str
    aliases: tuple[str, ...] = ()
    table_field: bool = False
    row_field: bool = False
    join: str = " "

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FieldSpec":
        key = str(data.get("key", "")).strip()
        if not _KEY_RE.match(key) or key in _RESERVED_KEYS:
            raise ValueError(f"invalid title-block field key {key!r}")
        label = str(data.get("label") or key.replace("_", " ").title()).strip()
        aliases = tuple(str(a).strip() for a in data.get("aliases", ()) if str(a).strip())
        return cls(
            key=key,
            label=label,
            aliases=aliases,
            table_field=bool(data.get("table_field", False)),
            row_field=bool(data.get("row_field", False)),
            join=str(data.get("join", " ")),
        )


@dataclass(frozen=True)
class TitleBlockSchema:
    """Ordered canonical fields plus extra rules for the normalization prompt.

    Field order is the display order of canonical pairs and of the title-block
    text. ``normalize_rules`` are sentences appended to the normalization prompt;
    they encode schema-specific confusions (two numbers that look alike, two
    organisation fields) that a generic prompt cannot know about.
    """

    name: str
    fields: tuple[FieldSpec, ...]
    normalize_rules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("a title-block schema needs at least one field")
        keys = [f.key for f in self.fields]
        if len(set(keys)) != len(keys):
            raise ValueError("title-block field keys must be unique")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TitleBlockSchema":
        fields = tuple(FieldSpec.from_dict(item) for item in data.get("fields", ()))
        rules = tuple(str(r).strip() for r in data.get("normalize_rules", ()) if str(r).strip())
        return cls(name=str(data.get("name", "custom")), fields=fields, normalize_rules=rules)

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "TitleBlockSchema":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def default(cls) -> "TitleBlockSchema":
        """The packaged neutral schema (ISO 7200-like, without person fields)."""
        text = resources.files(__package__).joinpath(_SCHEMA_RESOURCE).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))

    def keys(self) -> list[str]:
        return [f.key for f in self.fields]

    def get(self, key: str) -> Optional[FieldSpec]:
        for spec in self.fields:
            if spec.key == key:
                return spec
        return None

    def label_map(self) -> dict[str, str]:
        """Normalised printed label -> field key (aliases, display label and the key itself).

        On a clash the earlier field wins, so put the more specific field first.
        """
        mapping: dict[str, str] = {}
        for spec in self.fields:
            for raw in (*spec.aliases, spec.label, spec.key.replace("_", " ")):
                norm = normalize_label(raw)
                if norm and norm not in mapping:
                    mapping[norm] = spec.key
        return mapping

    def table_label_patterns(self) -> list[str]:
        """Regexes of every table-field label, stripped from table-field values."""
        patterns = []
        for spec in self.fields:
            if not spec.table_field:
                continue
            for raw in (*spec.aliases, spec.label):
                pattern = _alias_pattern(raw)
                if pattern and pattern not in patterns:
                    patterns.append(pattern)
        # Longest first, so "surface finish" is removed before "finish".
        return sorted(patterns, key=len, reverse=True)

    def field_label_patterns(self, key: str) -> list[str]:
        """Regexes of one field's own labels (longest first); empty for unknown keys."""
        spec = self.get(key)
        if spec is None:
            return []
        patterns: list[str] = []
        for raw in (*spec.aliases, spec.label):
            pattern = _alias_pattern(raw)
            if pattern and pattern not in patterns:
                patterns.append(pattern)
        return sorted(patterns, key=len, reverse=True)

    def table_label_aliases(self) -> set[str]:
        return {
            normalize_label(raw)
            for spec in self.fields
            if spec.table_field
            for raw in (*spec.aliases, spec.label)
            if normalize_label(raw)
        }


# --------------------------------------------------------------------------- domain terms


@dataclass(frozen=True)
class TermGroup:
    """A retrieval-critical term and its spelling variants.

    ``terms`` are named in the prompts (rows mentioning them must be kept row
    by row) and form the synonym line added to the retrieval text, so a query
    using one variant finds a chunk that prints another. ``patterns`` detect the
    term in labels and text (case-insensitive). A label that mentions the term
    is filed under ``field_key`` and every entry mentioning it gets ``tag``.
    """

    tag: str
    terms: tuple[str, ...]
    patterns: tuple[str, ...]
    field_key: str = ""
    synonym_text: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TermGroup":
        tag = str(data.get("tag", "")).strip()
        if not _KEY_RE.match(tag):
            raise ValueError(f"invalid term-group tag {tag!r}")
        terms = tuple(str(t).strip() for t in data.get("terms", ()) if str(t).strip())
        patterns = tuple(str(p) for p in data.get("patterns", ()) if str(p).strip())
        if not patterns:
            patterns = tuple(_alias_pattern(t) for t in terms)
        for pattern in patterns:
            re.compile(pattern)
        return cls(
            tag=tag,
            terms=terms,
            patterns=patterns,
            field_key=str(data.get("field_key", "")).strip(),
            synonym_text=str(data.get("synonym_text", "")).strip(),
        )

    def search(self, text: str) -> bool:
        return any(re.search(p, text, re.IGNORECASE) for p in self.patterns)

    def synonym_line(self) -> str:
        if self.synonym_text:
            return self.synonym_text
        if len(self.terms) < 2:
            return ""
        return "Domain synonym: " + " = ".join(self.terms)


@dataclass(frozen=True)
class PartSpec:
    """A part name and the regexes (abbreviations included) that identify it in a label or value."""

    name: str
    patterns: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PartSpec":
        name = str(data.get("name", "")).strip()
        if not name:
            raise ValueError("a part needs a name")
        patterns = tuple(str(p) for p in data.get("patterns", ()) if str(p).strip()) or (rf"\b{re.escape(name)}\b",)
        for pattern in patterns:
            re.compile(pattern)
        return cls(name=name, patterns=patterns)


@dataclass(frozen=True)
class DomainTerms:
    """Domain vocabulary that shapes prompts and retrieval text (all of it optional)."""

    groups: tuple[TermGroup, ...] = ()
    parts: tuple[PartSpec, ...] = ()
    sub_part_tokens: tuple[str, ...] = ()
    abbreviations: tuple[tuple[str, str], ...] = ()
    repeated_header_example: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DomainTerms":
        abbreviations = []
        for item in data.get("abbreviations", ()):
            short, meaning = str(item.get("short", "")).strip(), str(item.get("meaning", "")).strip()
            if short and meaning:
                abbreviations.append((short, meaning))
        tokens = tuple(str(t).strip() for t in data.get("sub_part_tokens", ()) if str(t).strip())
        for token in tokens:
            if not re.fullmatch(r"[A-Za-z0-9]+", token):
                raise ValueError(f"sub-part tokens must be alphanumeric (got {token!r})")
        return cls(
            groups=tuple(TermGroup.from_dict(g) for g in data.get("groups", ())),
            parts=tuple(PartSpec.from_dict(p) for p in data.get("parts", ())),
            sub_part_tokens=tokens,
            abbreviations=tuple(abbreviations),
            repeated_header_example=str(data.get("repeated_header_example", "")).strip(),
        )

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "DomainTerms":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def default(cls) -> "DomainTerms":
        """The packaged neutral example vocabulary (see ``configs/domain_terms.example.json``)."""
        text = resources.files(__package__).joinpath(_DOMAIN_RESOURCE).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))

    @classmethod
    def empty(cls) -> "DomainTerms":
        return cls()

    def is_empty(self) -> bool:
        return not (self.groups or self.parts or self.sub_part_tokens)

    def critical_terms(self) -> list[str]:
        """Every term the prompts call retrieval-critical, in configuration order."""
        terms: list[str] = []
        for group in self.groups:
            terms.extend(group.terms)
        terms.extend(part.name for part in self.parts)
        terms.extend(self.sub_part_tokens)
        seen: set[str] = set()
        ordered = []
        for term in terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                ordered.append(term)
        return ordered

    def signal_regex(self) -> Optional["re.Pattern[str]"]:
        """One case-insensitive regex matching any term group (``None`` when there are none)."""
        patterns = [p for group in self.groups for p in group.patterns]
        if not patterns:
            return None
        return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


# --------------------------------------------------------------------------- technique settings


def _parse_env_value(raw: str, annotation: Any, name: str) -> Any:
    """Convert an environment string using the field's annotation (``"int"``, ``"float"``, ...)."""
    raw = raw.strip()
    kind = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "str")
    if kind == "bool":
        low = raw.lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{name} must be 0/1 (got {raw!r})")
    try:
        if kind == "int":
            return int(raw)
        if kind == "float":
            return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name}: {exc}") from exc
    return raw


@dataclass(frozen=True)
class DrawingConfig:
    """Crop geometry, resolutions, call limits and the acceptance gate.

    Geometry is expressed in normalised displayed-page coordinates (top-left
    origin, ``/Rotate`` applied). Each region is rendered at its own DPI and
    then enlarged with LANCZOS by its ``*_upscale`` factor: the farther a
    region is from "whole page", the smaller its text and the more pixels per
    glyph the model needs.
    """

    # Whole page, cut into an overlapping grid so 4-6 pt text survives the model's image downscaling.
    page_dpi: float = 400
    page_tile_rows: int = 3
    page_tile_cols: int = 4
    page_tile_overlap: float = 0.12
    page_tile_upscale: float = 2.0
    # Lower-right corner where title blocks live (x >= 0.68, y >= 0.52).
    title_focus_x_start: float = 0.68
    title_focus_y_start: float = 0.52
    title_focus_dpi: float = 500
    title_focus_upscale: float = 3.0
    # Smallest bottom-right table cells (x >= 0.82, y >= 0.70); sent to both passes.
    table_focus_x_start: float = 0.82
    table_focus_y_start: float = 0.70
    table_focus_dpi: float = 650
    table_focus_upscale: float = 4.0
    # Wider metadata-table region (x >= 0.44, y >= 0.62): an overview plus a 2 x 3 overlapping grid.
    metadata_x_start: float = 0.44
    metadata_y_start: float = 0.62
    metadata_dpi: float = 650
    metadata_upscale: float = 4.0
    metadata_tile_rows: int = 2
    metadata_tile_cols: int = 3
    metadata_tile_overlap: float = 0.10
    # Upper bound per image after enlargement; 0 disables the cap (see docs/techniques/drawing.md).
    # 4 MP: same field accuracy as 8/16 MP on the synthetic set at ~1/4 of the 16 MP input tokens;
    # uncapped requests were rejected by the provider (image patch limit).
    max_image_pixels: int = 4_000_000
    # Model calls.
    timeout_s: float = 600.0
    max_output_tokens: int = 60000
    empty_output_retries: int = 2
    # Acceptance gate and retrieval-text limits.
    min_pairs: int = 3
    field_entry_limit: int = 300
    field_source_text_limit: int = 500
    error_preview_chars: int = 4000
    # "first": read the title block on page 1 and attach it to every page; "all": read every page.
    title_block_pages: str = "first"
    include_text_layer: bool = True
    # Optional JSON files overriding the packaged schema / domain terms ("none" disables domain terms).
    schema_path: str = ""
    domain_terms_path: str = ""
    schema: TitleBlockSchema = field(default_factory=TitleBlockSchema.default)
    domain: DomainTerms = field(default_factory=DomainTerms.default)

    def __post_init__(self) -> None:
        if self.title_block_pages not in TITLE_BLOCK_PAGE_MODES:
            raise ValueError(f"title_block_pages must be one of {TITLE_BLOCK_PAGE_MODES} (got {self.title_block_pages!r})")
        for name in ("page_tile_rows", "page_tile_cols", "metadata_tile_rows", "metadata_tile_cols"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        for name in ("page_dpi", "title_focus_dpi", "table_focus_dpi", "metadata_dpi"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_pairs < 0:
            raise ValueError("min_pairs must be >= 0")
        if self.max_image_pixels < 0:
            raise ValueError("max_image_pixels must be >= 0 (0 disables the cap)")

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None, **overrides: Any) -> "DrawingConfig":
        """Defaults, then ``VLMRAG_DRAWING_*`` variables, then keyword ``overrides``.

        ``VLMRAG_DRAWING_SCHEMA_PATH`` / ``VLMRAG_DRAWING_DOMAIN_TERMS_PATH``
        point at JSON files shaped like the examples in ``configs/``.
        """
        env = os.environ if environ is None else environ
        values: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            if f.default is dataclasses.MISSING:  # schema / domain come from the *_path fields
                continue
            name = "VLMRAG_DRAWING_" + f.name.upper()
            raw = env.get(name)
            if raw is None or not str(raw).strip():
                continue
            values[f.name] = _parse_env_value(str(raw), f.type, name)
        values.update(overrides)
        return cls.with_files(**values)

    @classmethod
    def with_files(cls, **values: Any) -> "DrawingConfig":
        """Build a config, loading ``schema_path`` / ``domain_terms_path`` when given."""
        schema_path = str(values.get("schema_path") or "")
        domain_path = str(values.get("domain_terms_path") or "")
        if "schema" not in values and schema_path:
            values["schema"] = TitleBlockSchema.load(schema_path)
        if "domain" not in values and domain_path:
            values["domain"] = DomainTerms.empty() if domain_path.lower() == "none" else DomainTerms.load(domain_path)
        return cls(**values)

    def replace(self, **changes: Any) -> "DrawingConfig":
        return dataclasses.replace(self, **changes)

    def render_summary(self) -> dict:
        """Render parameters recorded in every chunk's provenance."""
        return {
            "page_tiles": {
                "rows": self.page_tile_rows,
                "cols": self.page_tile_cols,
                "overlap": self.page_tile_overlap,
                "dpi": self.page_dpi,
                "upscale": self.page_tile_upscale,
            },
            "title_block_focus": {
                "bbox": [self.title_focus_x_start, self.title_focus_y_start, 1.0, 1.0],
                "dpi": self.title_focus_dpi,
                "upscale": self.title_focus_upscale,
            },
            "table_focus": {
                "bbox": [self.table_focus_x_start, self.table_focus_y_start, 1.0, 1.0],
                "dpi": self.table_focus_dpi,
                "upscale": self.table_focus_upscale,
            },
            "metadata_table": {
                "bbox": [self.metadata_x_start, self.metadata_y_start, 1.0, 1.0],
                "dpi": self.metadata_dpi,
                "upscale": self.metadata_upscale,
                "tile_rows": self.metadata_tile_rows,
                "tile_cols": self.metadata_tile_cols,
                "tile_overlap": self.metadata_tile_overlap,
            },
            "max_image_pixels": self.max_image_pixels,
        }
