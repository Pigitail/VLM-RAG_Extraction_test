"""Turn the model's answer into a validated :class:`SlideVisual`.

The parser is deliberately forgiving about *shape* and strict about
*content*: models (and servers that ignore the JSON schema) return block IDs
as ``"3"``, ``"block 3"`` or ``{"block_order": 3}``, roles as ``"screenshot"``
or ``"Zoomed"``, a single object instead of a list. All of those are coerced.
What is never accepted is an item that cannot be tied to a block (order
``<= 0``) or that has no description - those are dropped, because a
description that cannot be attached to an image block is unusable for the
chunk and for the viewer.

Every coercion that changed the meaning of an answer is recorded in
``SlideVisual.issues`` so it ends up in the chunk's provenance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ...vlm.jsonparse import extract_json_object
from .prompt import CONFIDENCE_LEVELS, IMAGE_ROLES

__all__ = [
    "ROLE_ALIASES",
    "SlideVisual",
    "coerce_order",
    "coerce_order_list",
    "normalize_image_role",
    "parse_visual",
]

_ORDER_KEYS = ("block_order", "order", "block", "block_id")
_RELATED_KEYS = ("block_order", "order", "block", "block_id", "slide_no", "slide")
_ROLE_KEYS = ("image_role", "visual_role", "role")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_DIGITS_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")

ROLE_ALIASES = {
    "main": "main_screenshot",
    "image": "main_screenshot",
    "screenshot": "main_screenshot",
    "full_screenshot": "main_screenshot",
    "primary_screenshot": "main_screenshot",
    "screen_capture": "main_screenshot",
    "capture": "main_screenshot",
    "detail": "detail_crop",
    "crop": "detail_crop",
    "cropped": "detail_crop",
    "zoom": "detail_crop",
    "zoomed": "detail_crop",
    "enlarged": "detail_crop",
    "partial": "detail_crop",
    "partial_screenshot": "detail_crop",
    "inline": "inline_image",
    "supporting_image": "inline_image",
    "supporting_screenshot": "inline_image",
    "explanation_image": "inline_image",
    "icon": "ui_icon",
    "button_icon": "ui_icon",
    "app_icon": "ui_icon",
}


@dataclass
class SlideVisual:
    visual_summary: str = ""
    image_descriptions: list[dict] = field(default_factory=list)
    flow_notes: list[str] = field(default_factory=list)
    notes: str = ""
    issues: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.visual_summary or self.image_descriptions or self.flow_notes or self.notes)

    def to_dict(self) -> dict:
        return {
            "visual_summary": self.visual_summary,
            "image_descriptions": [dict(d) for d in self.image_descriptions],
            "flow_notes": list(self.flow_notes),
            "notes": self.notes,
        }


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def coerce_order(value: Any) -> int:
    """Best-effort block number: ``3``, ``3.0``, ``"3"``, ``"block 3"`` -> ``3``; unusable -> ``0``.

    Python's JSON parser accepts ``Infinity``/``NaN`` and turns ``1e400`` into
    ``inf``; ``int(inf)`` raises ``OverflowError``, which must not escape and
    fail the whole file, so non-finite numbers count as unusable.
    """
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except OverflowError:
        return 0
    except (TypeError, ValueError):
        match = _DIGITS_RE.search(_text(value))
        return int(match.group(0)) if match else 0


def coerce_order_list(raw: Any) -> list[int]:
    """Positive, de-duplicated block numbers from a list, a scalar, or dicts holding an order key."""
    if isinstance(raw, dict) or isinstance(raw, (str, int, float)):
        raw = [raw]
    orders: list[int] = []
    for value in raw or []:
        if isinstance(value, dict):
            value = next((value.get(k) for k in _RELATED_KEYS if value.get(k)), None)
        order = coerce_order(value)
        if order > 0 and order not in orders:
            orders.append(order)
    return orders


def normalize_image_role(value: Any) -> str:
    """Map a free-form role to one of :data:`IMAGE_ROLES`; ``""`` when it cannot be mapped.

    An unknown role is left empty rather than guessed: a wrong role would
    change how a viewer shows the picture, an empty one only loses a hint.
    """
    role = _NON_ALNUM_RE.sub("_", _text(value).lower()).strip("_")
    role = ROLE_ALIASES.get(role, role)
    return role if role in IMAGE_ROLES else ""


def _first(item: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = item.get(key)
        if value:
            return value
    return None


def parse_visual(raw: Any) -> Optional[SlideVisual]:
    """Validate a parsed answer (``dict``) or raw model text.

    Returns ``None`` when no JSON object can be recovered (the caller falls
    back to the text-only chunk); otherwise a :class:`SlideVisual`, which may
    be empty.
    """
    data = raw if isinstance(raw, dict) else extract_json_object(raw if isinstance(raw, str) else None)
    if not isinstance(data, dict):
        return None

    visual = SlideVisual(
        visual_summary=_text(data.get("visual_summary")),
        notes=_text(data.get("notes")),
    )

    raw_notes = data.get("flow_notes", [])
    if isinstance(raw_notes, str):
        raw_notes = [raw_notes]
    seen_notes: set[str] = set()
    for item in raw_notes if isinstance(raw_notes, list) else []:
        note = _text(item)
        key = _WS_RE.sub(" ", note).lower()
        if note and key not in seen_notes:
            seen_notes.add(key)
            visual.flow_notes.append(note)

    raw_items = data.get("image_descriptions", [])
    if isinstance(raw_items, dict):
        raw_items = [raw_items]
    if not isinstance(raw_items, list):
        raw_items = []
    dropped = 0
    for item in raw_items:
        if not isinstance(item, dict):
            dropped += 1
            continue
        order = coerce_order(_first(item, _ORDER_KEYS))
        description = _text(item.get("description"))
        if order <= 0 or not description:
            dropped += 1
            continue
        confidence = _text(item.get("confidence")).lower()
        if confidence not in CONFIDENCE_LEVELS:
            visual.issues.append(f"invalid_confidence:{order}")
            confidence = "low"
        entry = {
            "block_order": order,
            "related_block_orders": coerce_order_list(item.get("related_block_orders", [])),
            "description": description,
            "description_en": _text(item.get("description_en")),
            "description_ko": _text(item.get("description_ko")),
            "confidence": confidence,
        }
        role = normalize_image_role(_first(item, _ROLE_KEYS))
        if role:
            entry["image_role"] = role
        else:
            visual.issues.append(f"invalid_image_role:{order}")
        visual.image_descriptions.append(entry)
    if dropped:
        visual.issues.append(f"dropped_descriptions:{dropped}")
    return visual
