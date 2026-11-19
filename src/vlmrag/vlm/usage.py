"""Thread-safe token and latency accounting per task.

VLM extraction cost is dominated by a handful of call sites (page tiles, table
crops, normalization passes). Recording usage per ``task`` makes it obvious
which stage to tune, and recording cache hits separately shows how much a
re-run actually cost.
"""

from __future__ import annotations

import threading
from typing import Optional

from .types import Usage

__all__ = ["UsageLedger"]

_TOKEN_FIELDS = ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens")


def _empty_row() -> dict:
    row = {"calls": 0, "cached": 0, "latency_s": 0.0}
    row.update({name: 0 for name in _TOKEN_FIELDS})
    return row


class UsageLedger:
    """Accumulates usage per task. Every public method is safe to call from threads.

    Cached responses count as calls (``cached`` is incremented) but contribute no
    tokens or latency, because no tokens were spent on them in this run.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: dict[str, dict] = {}

    def add(self, task: str, usage: Optional[Usage], latency: float = 0.0, cached: bool = False) -> None:
        with self._lock:
            row = self._rows.setdefault(task or "unknown", _empty_row())
            row["calls"] += 1
            if cached:
                row["cached"] += 1
                return
            row["latency_s"] += float(latency or 0.0)
            if usage is not None:
                for name in _TOKEN_FIELDS:
                    row[name] += int(getattr(usage, name, 0) or 0)

    def summary(self) -> dict:
        """Return ``{"tasks": {task: row}, "total": row}`` (a deep copy)."""
        with self._lock:
            tasks = {task: dict(row) for task, row in sorted(self._rows.items())}
        total = _empty_row()
        for row in tasks.values():
            for key in total:
                total[key] += row[key]
        for row in list(tasks.values()) + [total]:
            row["latency_s"] = round(row["latency_s"], 3)
        return {"tasks": tasks, "total": total}

    def merge(self, other: "UsageLedger") -> None:
        snapshot = other.summary()["tasks"]
        with self._lock:
            for task, row in snapshot.items():
                mine = self._rows.setdefault(task, _empty_row())
                for key, value in row.items():
                    mine[key] += value

    def format_table(self) -> str:
        """Human-readable summary used by the CLI."""
        data = self.summary()
        header = f"{'task':<28} {'calls':>6} {'cached':>6} {'in_tok':>10} {'out_tok':>10} {'reason':>8} {'latency_s':>10}"
        lines = [header, "-" * len(header)]
        rows = list(data["tasks"].items()) + [("TOTAL", data["total"])]
        for task, row in rows:
            lines.append(
                f"{task:<28} {row['calls']:>6} {row['cached']:>6} {row['input_tokens']:>10} "
                f"{row['output_tokens']:>10} {row['reasoning_tokens']:>8} {row['latency_s']:>10.1f}"
            )
        return "\n".join(lines)
