"""Per-op cycle breakdown.

The harness emits one profile CSV row per dispatch (dispatch_id, name,
op, shape, cycles). Summing that gives `cycles_firesim` /
`cycles_spike` -- the column the dashboard already shows. For
"I changed this conv kernel, did it help?" you need finer-grained
data: which op kind dominates this cell, what fraction of total
cycles, distribution per kind. The runner-side helper writes
`cycles_per_op.json` per cell with that breakdown; the extractors
below surface a few of its fields as dashboard columns and the rest
land in the per-cell detail section of `summary.md`.

Schema of cycles_per_op.json (emitted by runners/<runner>.py):

  {
    "schema_version": 1,
    "total_cycles": int,
    "n_ops": int,
    "by_op_kind": {
      "conv2d_s8": {"count": int, "total": int, "mean": float,
                    "min": int, "max": int, "share": float},
      ...
    },
    "by_dispatch": [
      {"dispatch_id": int, "op": str, "name": str, "shape": str,
       "cycles": int},
      ...
    ]
  }

Where `share` is each op kind's fraction of total_cycles.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def _load(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def total_cycles(path: Path) -> Optional[int]:
    return _load(path).get("total_cycles")


def n_ops(path: Path) -> Optional[int]:
    return _load(path).get("n_ops")


def dominant_share(path: Path) -> Optional[float]:
    """Fraction of total_cycles spent on the single op kind that
    consumes the most cycles. Highlights "which op is the bottleneck."
    A value of 0.78 means 78% of all cycles in this cell are inside
    one op kind."""
    by_kind = _load(path).get("by_op_kind") or {}
    if not by_kind:
        return None
    shares = [v.get("share", 0.0) for v in by_kind.values()]
    return max(shares) if shares else None


def top_op_breakdown(path: Path, k: int = 5
                     ) -> list[tuple[str, float, int]]:
    """Per-op-kind (name, share, total_cycles), sorted by share desc.
    Used by the dashboard renderer for the per-cell detail block, not
    by a metrics.yaml column."""
    by_kind = _load(path).get("by_op_kind") or {}
    items = [
        (kind, v.get("share", 0.0), v.get("total", 0))
        for kind, v in by_kind.items()
    ]
    items.sort(key=lambda t: t[1], reverse=True)
    return items[:k]


def synthesize(profile_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the cycles_per_op.json payload from the raw profile rows
    the runners parse out of the harness stdout. Lives here (in the
    ingest package) rather than in runners/* so the writer and the
    reader share one schema. Callers: `runners/spike.write_cycles_per_op_json`
    and the firesim counterpart."""
    by_dispatch: list[dict[str, Any]] = []
    by_kind: dict[str, dict[str, Any]] = {}
    total = 0
    for r in profile_rows:
        cyc = int(r.get("cycles", 0) or 0)
        total += cyc
        op = str(r.get("op", "")).strip() or "unknown"
        rec = {
            "dispatch_id": int(r.get("dispatch_id", -1) or -1),
            "op": op,
            "name": str(r.get("name", "")),
            "shape": str(r.get("shape", "")),
            "cycles": cyc,
        }
        by_dispatch.append(rec)
        slot = by_kind.setdefault(op, {
            "count": 0, "total": 0, "min": cyc, "max": cyc,
        })
        slot["count"] += 1
        slot["total"] += cyc
        slot["min"] = min(slot["min"], cyc)
        slot["max"] = max(slot["max"], cyc)

    for kind, slot in by_kind.items():
        slot["mean"] = float(slot["total"]) / slot["count"] if slot["count"] else 0.0
        slot["share"] = (float(slot["total"]) / float(total)) if total > 0 else 0.0

    return {
        "schema_version": 1,
        "total_cycles": total,
        "n_ops": len(profile_rows),
        "by_op_kind": by_kind,
        "by_dispatch": by_dispatch,
    }
