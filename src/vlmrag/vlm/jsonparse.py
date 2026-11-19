"""Recover a JSON value from model output that is "almost" JSON.

Even with structured outputs enabled, real model output arrives wrapped in
Markdown code fences, preceded by a sentence of prose, followed by a trailing
remark, or (with OpenAI-compatible servers that ignore ``response_format``)
mixed with reasoning text. A naive ``text[text.find("{"):text.rfind("}") + 1]``
breaks as soon as a string value contains a brace or the prose after the object
contains one, so this module uses a single string-aware bracket scanner:

1. strip code fences and try ``json.loads`` on the whole text;
2. otherwise scan left to right for ``{`` / ``[``, track nesting while skipping
   everything inside string literals (honouring backslash escapes), and try to
   parse each balanced top-level candidate in order until one succeeds.

Unparseable input returns ``None`` (never raises) so callers can apply their own
fallback rules.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterator, Optional

__all__ = ["extract_json_value", "extract_json_object", "strip_code_fences", "iter_json_candidates"]

_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n?(.*?)```", re.DOTALL)
_OPENERS = {"{": "}", "[": "]"}


def strip_code_fences(text: str) -> str:
    """Return the body of the first fenced block, or the text unchanged.

    A fence that is opened but never closed (truncated output) is removed as
    well, so the scanner below still gets a chance at the partial content.
    """
    if not text:
        return ""
    match = _FENCE_RE.search(text)
    if match:
        return match.group(2).strip()
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        return stripped[first_newline + 1:].strip() if first_newline >= 0 else ""
    return stripped


def iter_json_candidates(text: str, openers: str = "{[") -> Iterator[str]:
    """Yield balanced top-level ``{...}`` / ``[...]`` substrings in order.

    Brackets inside string literals are ignored. A closing bracket that does not
    match the innermost opener abandons the current candidate and scanning
    resumes after its start, so one malformed fragment does not hide a valid
    object later in the text.
    """
    n = len(text)
    start = 0
    while start < n:
        # Find the next opener outside of any string literal.
        pos = -1
        in_string = False
        escape = False
        for i in range(start, n):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in openers:
                pos = i
                break
        if pos < 0:
            return
        stack: list[str] = []
        in_string = False
        escape = False
        end = -1
        for i in range(pos, n):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in _OPENERS:
                stack.append(_OPENERS[ch])
            elif ch in "}]":
                if not stack or stack[-1] != ch:
                    break  # mismatched closer: give up on this candidate
                stack.pop()
                if not stack:
                    end = i
                    break
        if end >= 0:
            yield text[pos:end + 1]
            start = end + 1
        else:
            start = pos + 1


def _loads(candidate: str) -> tuple[bool, Any]:
    try:
        return True, json.loads(candidate)
    except (ValueError, TypeError):
        return False, None


def extract_json_value(text: Optional[str]) -> Optional[Any]:
    """Return the first JSON object or array found in ``text``, else ``None``.

    A bare scalar (``"42"``, ``"true"``) is only accepted when it is the whole
    payload; techniques always ask for objects or arrays, so scalars found inside
    prose are noise.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    body = strip_code_fences(text)
    for attempt in (body, text.strip()):
        ok, value = _loads(attempt)
        if ok and isinstance(value, (dict, list)):
            return value
    sources = (body,) if body == text.strip() else (body, text)
    for source in sources:
        for candidate in iter_json_candidates(source):
            ok, value = _loads(candidate)
            if ok:
                return value
    ok, value = _loads(body)
    return value if ok and value is not None else None


def extract_json_object(text: Optional[str]) -> Optional[dict]:
    """Like :func:`extract_json_value` but only ever returns a ``dict``.

    If the first JSON value is an array (for example a model that wrapped its
    answer in ``[...]``), the scan continues for an object, and finally a
    one-element array holding an object is unwrapped.
    """
    value = extract_json_value(text)
    if isinstance(value, dict):
        return value
    if not isinstance(text, str):
        return None
    body = strip_code_fences(text)
    for source in (body, text):
        for candidate in iter_json_candidates(source, openers="{"):
            ok, parsed = _loads(candidate)
            if ok and isinstance(parsed, dict):
                return parsed
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        return value[0]
    return None
