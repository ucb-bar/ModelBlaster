"""Extractors for per-region cycle CSVs.

Single-model harnesses emit one CSV with columns:
    dispatch_id,name,op,shape,cycles
XPU-RT (multi-backend) runs emit a `backend` column prefix:
    backend,dispatch_id,name,op,shape,cycles
Both shapes are tolerated.

The runner wrapper copies the CSV that
`validation.runner_common.parse_profile` parsed out of the harness
stdout into `results/<arm>/<workload>/<run-id>/profile_<runner>.csv`
under the chosen runner suffix (spike or firesim).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional


def _read(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def sum_cycles(path: Path) -> Optional[int]:
    rows = _read(path)
    if not rows:
        return None
    total = 0
    for row in rows:
        v = row.get("cycles")
        if v is None or v == "":
            continue
        total += int(v)
    return total


def cycles_by_backend(path: Path) -> dict[str, int]:
    """Per-backend cycle totals for XPU-RT runs. Empty dict when the
    CSV has no `backend` column."""
    rows = _read(path)
    out: dict[str, int] = {}
    for row in rows:
        bk = row.get("backend")
        if bk is None or bk == "":
            continue
        v = row.get("cycles")
        if v is None or v == "":
            continue
        out[bk] = out.get(bk, 0) + int(v)
    return out


def op_count(path: Path) -> int:
    return len(_read(path))
