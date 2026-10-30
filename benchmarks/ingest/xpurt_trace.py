"""Extractors for XPU-RT schedule trace artifacts.

Source file: ``xpurt_trace.csv`` -- the per-entry execution trace the
schedule-driven harness prints between ``MODELBLASTER_XPURT_TRACE_BEGIN``
and ``MODELBLASTER_XPURT_TRACE_END`` when the binary is built with
``-DMODELBLASTER_XPURT_TRACE=ON``. Each row carries
(entry_id, network, instance, dispatch_id, op, name, core_kind,
hart, predicted_start_ms, predicted_duration_ms, worker_kind_idx,
worker_hart, actual_start_cycles, actual_end_cycles). ``hart`` is where
the schedule placed the dispatch; ``worker_hart`` is where it actually
ran -- equal for every executed row since the walker spawns one worker
per (core_kind, hart), and -1 for a row no worker executed.

Metrics that fire only on heterogeneous targets. The aggregator's
nullable_if rules in ``config/metrics.yaml`` suppress them on
single-tile targets so the dashboard does not flag them as
"missing"; the extractors below additionally return ``None`` when
the trace CSV is absent (e.g. the binary was built without the
trace flag).

* ``makespan`` -- the wall-clock cycles end of the latest-finishing
  entry. Tracks the actual run length; pair with the schedule's
  predicted makespan for "did we beat / blow the predicted budget?"
* ``utilization_<kind>`` -- per-tile fraction of makespan spent
  doing useful work. Sums (end - start) across rows whose
  ``core_kind`` matches, divides by ``makespan * n_harts_of_kind``.
  Low utilization on a tile signals it's the bottleneck's stragglers
  or under-allocated.
* ``cross_tile_bytes`` -- static upper bound on bytes that have to
  move between physical tiles. Not implemented yet; computing it
  needs the per-dispatch output-tensor size from graph.json paired
  with the schedule's (producer_kind, consumer_kind) edges.
* ``deadline_met_rate`` -- periodic-workload deadline check. Reads
  ``wall_cycles.txt`` (per-instance wall cycles) against the
  workload row's ``period_ms``; not wired yet because none of
  today's workloads are periodic.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional


def _read(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _safe_max_end(rows: list[dict[str, str]]) -> Optional[int]:
    ends = [int(r["actual_end_cycles"]) for r in rows
            if r.get("actual_end_cycles", "").strip()]
    return max(ends) if ends else None


def _busy_cycles_by_kind(rows: list[dict[str, str]]) -> dict[str, int]:
    """Per-core_kind sum of (actual_end_cycles - actual_start_cycles)."""
    out: dict[str, int] = defaultdict(int)
    for r in rows:
        kind = r.get("core_kind", "").strip()
        start = r.get("actual_start_cycles", "").strip()
        end = r.get("actual_end_cycles", "").strip()
        if not kind or not start or not end:
            continue
        dur = max(0, int(end) - int(start))
        out[kind] += dur
    return out


def _hart_count_by_kind(rows: list[dict[str, str]]) -> dict[str, int]:
    """Distinct hart ids observed per core_kind. A trace where only
    one hart of a kind ran tells us the topology had one tile of
    that kind; we divide utilization by n_harts so 2 active tiles
    sharing 100% of the makespan reads as ~50% each instead of
    summing past 100%."""
    by_kind: dict[str, set[int]] = defaultdict(set)
    for r in rows:
        kind = r.get("core_kind", "").strip()
        hart = r.get("hart", "").strip()
        if not kind or not hart:
            continue
        by_kind[kind].add(int(hart))
    return {k: max(1, len(v)) for k, v in by_kind.items()}


def makespan(path: Path) -> Optional[int]:
    """Max actual_end_cycles across all entries. Cycles, not ms."""
    rows = _read(path)
    return _safe_max_end(rows)


def _utilization(rows: list[dict[str, str]], kind: str) -> Optional[float]:
    total_makespan = _safe_max_end(rows)
    if not total_makespan:
        return None
    busy = _busy_cycles_by_kind(rows).get(kind, 0)
    n_harts = _hart_count_by_kind(rows).get(kind, 1)
    denom = total_makespan * n_harts
    if denom <= 0:
        return None
    return float(busy) / float(denom)


def utilization_gemmini(path: Path) -> Optional[float]:
    return _utilization(_read(path), "gemmini")


def utilization_opu(path: Path) -> Optional[float]:
    return _utilization(_read(path), "rvv_opu")


def utilization_rvv(path: Path) -> Optional[float]:
    """Generic RVV-tile utilization for non-OPU heterogeneous configs
    (e.g. quad-rocket-saturn with two RVV harts)."""
    return _utilization(_read(path), "rvv")


def utilization_scalar(path: Path) -> Optional[float]:
    return _utilization(_read(path), "scalar")


def cross_tile_bytes(path: Path) -> Optional[int]:
    """Static upper bound on bytes that must cross tile boundaries,
    computed by the runner's hetero-artifacts helper from graph.json
    + schedule.json edges (see runners._hetero_artifacts). The path
    points at `cross_tile_estimate.json` -- this extractor returns
    the total. Per-edge detail lives in the JSON for offline
    inspection."""
    import json as _json
    if not path.exists():
        return None
    try:
        data = _json.loads(path.read_text())
    except Exception:
        return None
    v = data.get("total_bytes")
    return int(v) if v is not None else None


def _dispatches_of_kind(path: Path, kind: str) -> Optional[int]:
    """Count trace rows whose core_kind matches. None when the trace is
    empty (e.g. binary built without -DMODELBLASTER_XPURT_TRACE)."""
    rows = _read(path)
    if not rows:
        return None
    return sum(1 for r in rows
               if r.get("core_kind", "").strip() == kind)


def dispatches_per_tile_gemmini(path: Path) -> Optional[int]:
    return _dispatches_of_kind(path, "gemmini")


def dispatches_per_tile_opu(path: Path) -> Optional[int]:
    return _dispatches_of_kind(path, "rvv_opu")


def dispatches_per_tile_rvv(path: Path) -> Optional[int]:
    return _dispatches_of_kind(path, "rvv")


def dispatches_per_tile_scalar(path: Path) -> Optional[int]:
    return _dispatches_of_kind(path, "scalar")


def tile_load_imbalance(path: Path) -> Optional[float]:
    """Ratio of busiest tile's busy cycles to least-busy tile's busy
    cycles. Reads `_busy_cycles_by_kind` and divides max/min over the
    distinct core_kinds. Returns None when only one tile-kind ran
    (single-target metrics already cover that case). High ratios on
    hetero workloads point at placement decisions that left one tile
    idle while another saturated."""
    busy = _busy_cycles_by_kind(_read(path))
    nonzero = [v for v in busy.values() if v > 0]
    if len(nonzero) < 2:
        return None
    return float(max(nonzero)) / float(min(nonzero))


def schedule_parallelism_max(path: Path) -> Optional[int]:
    """Maximum number of dispatches running concurrently at any point
    in time, computed via a sweep over (start, end) intervals. A value
    of 1 means the schedule is fully serial; 2 means at some instant
    two tiles ran ops at the same time. The theoretical max equals
    the number of tiles in the registry; lower values mean
    inter-dispatch dependencies kept tiles waiting."""
    rows = _read(path)
    if not rows:
        return None
    events: list[tuple[int, int]] = []
    for r in rows:
        start = r.get("actual_start_cycles", "").strip()
        end = r.get("actual_end_cycles", "").strip()
        if not start or not end:
            continue
        events.append((int(start), +1))
        events.append((int(end), -1))
    if not events:
        return None
    events.sort()
    cur = peak = 0
    for _, delta in events:
        cur += delta
        if cur > peak:
            peak = cur
    return peak


def deadline_met_rate(path: Path) -> Optional[float]:
    """Not yet implemented. Requires the workload row to declare a
    `period_ms` (or `deadline_ms`), and the harness's per-instance
    wall_cycles entries to be paired against it. Lands when periodic
    workloads enter the matrix."""
    return None


def n_entries(path: Path) -> int:
    return len(_read(path))
