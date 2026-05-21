"""Extractors that read each run's `run.json`.

`run.json` is emitted by every arm driver at the end of a run and
captures the minimum reproducibility envelope: git SHA, env snapshot
path, wall-clock duration, peak RSS, exit status, and the workload
fields the cell was identified by. Arms may add extra fields (e.g.
cache_hit_rate for Arms B and C); the extractors below tolerate their
absence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def _load(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def wall_clock_s(path: Path) -> Optional[float]:
    data = _load(path)
    v = data.get("wall_clock_s")
    return float(v) if v is not None else None


def peak_rss_mb(path: Path) -> Optional[float]:
    data = _load(path)
    v = data.get("peak_rss_mb")
    return float(v) if v is not None else None


def cache_hit_rate(path: Path) -> Optional[float]:
    """Fraction of kernel cache hits over total kernel selections during
    this run. None when the arm does not track caching (Arm A) or did
    not record it."""
    data = _load(path)
    v = data.get("cache_hit_rate")
    return float(v) if v is not None else None


def exit_ok(path: Path) -> bool:
    data = _load(path)
    return data.get("exit_status") == "ok"
