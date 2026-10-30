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
    "schema_version": 2,
    "total_cycles": int,
    "n_ops": int,
    "mean_cycles_per_dispatch": float,
    "stddev_cycles_per_dispatch": float,
    "by_op_kind": {
      "conv2d_s8": {"count": int, "total": int, "mean": float,
                    "min": int, "max": int, "share": float,
                    "median": float, "p50": float, "p90": float,
                    "p95": float, "stddev": float},
      ...
    },
    "by_dispatch": [
      {"dispatch_id": int, "op": str, "name": str, "shape": str,
       "cycles": int},
      ...
    ]
  }

Where `share` is each op kind's fraction of total_cycles. p50/p90/p95
percentiles are linear interpolations across the op-kind's dispatch
cycle counts (so small-cardinality ops are still defined). The
top-level `mean_cycles_per_dispatch` and `stddev_cycles_per_dispatch`
are over all dispatches regardless of op kind.

`schema_version: 2` adds the percentile and stddev fields. Older v1
files (from before the per-op-stats addition) still parse -- the
extractors return None when a field is missing.
"""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Any, Optional


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]. Defined for any
    non-empty list (single element returns that element)."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    rank = (q / 100.0) * (len(s) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(s[lo])
    frac = rank - lo
    return float(s[lo]) * (1 - frac) + float(s[hi]) * frac


def _stddev(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)


def parse_xpurt_trace_csv(csv_body: str) -> list[dict[str, Any]]:
    """Parse a trace CSV body (without surrounding markers) into a
    list of dict rows. Shared with the runners so both the writer
    and the cycles_per_op.synthesize call can consume the same
    parsed shape."""
    if not csv_body:
        return []
    return list(csv.DictReader(io.StringIO(csv_body)))


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


def mean_cycles_per_dispatch(path: Path) -> Optional[float]:
    """total_cycles / n_ops. The "typical cost" of one dispatch."""
    data = _load(path)
    v = data.get("mean_cycles_per_dispatch")
    if v is not None:
        return float(v)
    # Back-compat for v1 files: compute from total + n.
    tot = data.get("total_cycles")
    n = data.get("n_ops")
    if tot is None or not n:
        return None
    return float(tot) / float(n)


def stddev_cycles_per_dispatch(path: Path) -> Optional[float]:
    """Spread of per-dispatch cycles across the whole cell. A high
    value signals the cell mixes tiny + huge ops (potential
    fusion/granularity headroom)."""
    v = _load(path).get("stddev_cycles_per_dispatch")
    return float(v) if v is not None else None


def op_kind_p95_max(path: Path) -> Optional[float]:
    """Largest p95 across op kinds. Surfaces the worst-tail kernel --
    one slow conv2d_s8 with high p95 means a specific shape is hurting.
    Pair with `dominant_op_share`: high dominant share + high p95
    points at one specific bottleneck dispatch."""
    by_kind = _load(path).get("by_op_kind") or {}
    p95s = [v.get("p95") for v in by_kind.values()
            if v.get("p95") is not None]
    return max(p95s) if p95s else None


def op_kind_median_max(path: Path) -> Optional[float]:
    """Largest median cycles across op kinds. A "typical" cost ceiling
    for the cell -- robust to a single outlier dispatch."""
    by_kind = _load(path).get("by_op_kind") or {}
    meds = [v.get("median") for v in by_kind.values()
            if v.get("median") is not None]
    return max(meds) if meds else None


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


def synthesize(profile_rows: list[dict[str, Any]],
               trace_rows: Optional[list[dict[str, Any]]] = None,
               ) -> dict[str, Any]:
    """Build the cycles_per_op.json payload from the raw profile rows
    the runners parse out of the harness stdout. Lives here (in the
    ingest package) rather than in runners/* so the writer and the
    reader share one schema.

    When `trace_rows` are supplied (from the XPURT trace block on the
    hetero path), each by_dispatch entry is enriched with `core_kind`
    and `hart`, and a `by_kind_tile` rollup groups cycles per
    (op_kind, core_kind) so "conv2d_s8 on gemmini vs conv2d_s8 on
    rvv_opu" reads off in one place. The unmerged fields stay
    available for single-tile cells where trace rows are absent.
    """
    tile_by_id: dict[int, tuple[str, str]] = {}
    if trace_rows:
        for r in trace_rows:
            try:
                d_id = int(r.get("dispatch_id", -1))
            except (TypeError, ValueError):
                continue
            tile_by_id[d_id] = (
                str(r.get("core_kind", "")).strip(),
                str(r.get("hart", "")).strip(),
            )

    by_dispatch: list[dict[str, Any]] = []
    by_kind: dict[str, dict[str, Any]] = {}
    by_kind_values: dict[str, list[int]] = {}
    by_kind_tile: dict[tuple[str, str], dict[str, Any]] = {}
    by_kind_tile_values: dict[tuple[str, str], list[int]] = {}
    all_cycles: list[int] = []
    total = 0
    for r in profile_rows:
        cyc = int(r.get("cycles", 0) or 0)
        total += cyc
        all_cycles.append(cyc)
        op = str(r.get("op", "")).strip() or "unknown"
        raw_d_id = r.get("dispatch_id")
        d_id = -1 if raw_d_id is None else int(raw_d_id)
        core_kind, hart = tile_by_id.get(d_id, ("", ""))

        rec = {
            "dispatch_id": d_id,
            "op": op,
            "name": str(r.get("name", "")),
            "shape": str(r.get("shape", "")),
            "cycles": cyc,
        }
        if core_kind:
            rec["core_kind"] = core_kind
        if hart:
            rec["hart"] = hart
        by_dispatch.append(rec)

        slot = by_kind.setdefault(op, {
            "count": 0, "total": 0, "min": cyc, "max": cyc,
        })
        slot["count"] += 1
        slot["total"] += cyc
        slot["min"] = min(slot["min"], cyc)
        slot["max"] = max(slot["max"], cyc)
        by_kind_values.setdefault(op, []).append(cyc)

        if core_kind:
            key = (op, core_kind)
            tslot = by_kind_tile.setdefault(key, {
                "count": 0, "total": 0, "min": cyc, "max": cyc,
            })
            tslot["count"] += 1
            tslot["total"] += cyc
            tslot["min"] = min(tslot["min"], cyc)
            tslot["max"] = max(tslot["max"], cyc)
            by_kind_tile_values.setdefault(key, []).append(cyc)

    for kind, slot in by_kind.items():
        slot["mean"] = float(slot["total"]) / slot["count"] if slot["count"] else 0.0
        slot["share"] = (float(slot["total"]) / float(total)) if total > 0 else 0.0
        vals = by_kind_values.get(kind, [])
        if vals:
            slot["median"] = _percentile(list(map(float, vals)), 50.0)
            slot["p50"] = slot["median"]
            slot["p90"] = _percentile(list(map(float, vals)), 90.0)
            slot["p95"] = _percentile(list(map(float, vals)), 95.0)
            slot["stddev"] = _stddev(list(map(float, vals)))

    by_kind_tile_serialized: dict[str, dict[str, Any]] = {}
    for (op, core_kind), slot in by_kind_tile.items():
        slot["mean"] = float(slot["total"]) / slot["count"] if slot["count"] else 0.0
        slot["share"] = (float(slot["total"]) / float(total)) if total > 0 else 0.0
        vals = by_kind_tile_values.get((op, core_kind), [])
        if vals:
            slot["median"] = _percentile(list(map(float, vals)), 50.0)
            slot["p95"] = _percentile(list(map(float, vals)), 95.0)
            slot["stddev"] = _stddev(list(map(float, vals)))
        by_kind_tile_serialized[f"{op}@{core_kind}"] = slot

    n = len(all_cycles)
    mean_disp = (float(total) / n) if n else 0.0
    stddev_disp = _stddev(list(map(float, all_cycles))) if n else 0.0

    out: dict[str, Any] = {
        "schema_version": 2,
        "total_cycles": total,
        "n_ops": n,
        "mean_cycles_per_dispatch": mean_disp,
        "stddev_cycles_per_dispatch": stddev_disp,
        "by_op_kind": by_kind,
        "by_dispatch": by_dispatch,
    }
    if by_kind_tile_serialized:
        out["by_op_kind_x_tile"] = by_kind_tile_serialized
    return out
