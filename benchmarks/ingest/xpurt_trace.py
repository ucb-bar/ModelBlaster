"""Extractors for XPU-RT schedule trace artifacts.

Two source files:

* `wall_cycles.txt` — one line per periodic instance:
      MODELBLASTER_WALL_CYCLES [<job_name>_inst<N>] <cycles>
  Aggregated to `makespan_cycles` (max across job instances).

* `xpurt_trace.csv` — emitted by the harness when XPURT_TRACE=1.
  Columns: entry_id, network, job_name, core_kind, hart,
           predicted_start_ms, predicted_dur_ms,
           actual_start_cycles, actual_end_cycles, bytes_moved
  Aggregated to per-tile utilization, cross-tile bytes, and
  deadline-met rate.

These metrics only fire on the `hetero_gemmini_opu` target (and any
future heterogeneous SoC). On single-tile targets the metrics.yaml
`nullable_if` rules suppress them; the extractors return None on a
single-tile-shaped trace as a safety net.

Implementation is deferred until the hetero_gemmini_opu backend lands
(P2.1). Returning None keeps the aggregator silent until then.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def makespan(path: Path) -> Optional[int]:
    return None


def utilization_gemmini(path: Path) -> Optional[float]:
    return None


def utilization_opu(path: Path) -> Optional[float]:
    return None


def cross_tile_bytes(path: Path) -> Optional[int]:
    return None


def deadline_met_rate(path: Path) -> Optional[float]:
    return None
