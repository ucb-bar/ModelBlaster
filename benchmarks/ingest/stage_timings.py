"""Extractors for the per-stage wall-clock breakdown.

``stage_timings.json`` is synthesized by the arm driver after parsing
the MODELBLASTER_STAGE_BEGIN/END markers ``examples/_run_lib.sh``
emits between each pipeline stage. Lets the dashboard answer "where
did the wall-clock budget go" without re-running with -x.

Schema:

  {
    "schema_version": 1,
    "extract_s":            float | null,
    "generate_skeleton_s":  float | null,
    "generate_kernels_s":   float | null,
    "build_s":              float | null,
    "run_s":                float | null,
    "total_stage_s":        float | null,  # sum of the above
  }

Stages that didn't fire (e.g. extract skipped via FORCE_EXTRACT=0
when the IR is already on disk) record their measured ~0 timing
rather than null -- the marker fires either way. Truly missing
markers (e.g. shell crashed before stage 5) read as null.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional


_END_RE = re.compile(r"^MODELBLASTER_STAGE_END:([A-Za-z_]+):([0-9.]+)$")


def _load(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _get(path: Path, key: str) -> Optional[float]:
    v = _load(path).get(key)
    return float(v) if v is not None else None


def extract_s(path: Path) -> Optional[float]:
    return _get(path, "extract_s")


def generate_skeleton_s(path: Path) -> Optional[float]:
    return _get(path, "generate_skeleton_s")


def generate_kernels_s(path: Path) -> Optional[float]:
    return _get(path, "generate_kernels_s")


def build_s(path: Path) -> Optional[float]:
    return _get(path, "build_s")


def run_s(path: Path) -> Optional[float]:
    return _get(path, "run_s")


def total_stage_s(path: Path) -> Optional[float]:
    return _get(path, "total_stage_s")


def parse_stdout(stdout: str) -> dict[str, Optional[float]]:
    """Walk stdout lines for MODELBLASTER_STAGE_END markers and return
    a {stage_name + _s: seconds} dict. Stages that didn't fire are
    absent from the result (the caller can decide null vs zero).
    """
    out: dict[str, Optional[float]] = {}
    for line in stdout.splitlines():
        m = _END_RE.match(line)
        if m:
            name = m.group(1)
            secs = float(m.group(2))
            out[f"{name}_s"] = secs
    return out


def synthesize(stdout: str) -> dict[str, Any]:
    """Build the stage_timings.json payload from a run.sh stdout dump.
    Standard stages: extract, generate_skeleton, generate_kernels,
    build, run. Computes total_stage_s as a sum of the present stages
    (skipped or missing stages do not double-count)."""
    parsed = parse_stdout(stdout)
    standard = ["extract_s", "generate_skeleton_s", "generate_kernels_s",
                "build_s", "run_s"]
    payload: dict[str, Any] = {"schema_version": 1}
    total = 0.0
    any_seen = False
    for k in standard:
        v = parsed.get(k)
        payload[k] = v
        if v is not None:
            total += v
            any_seen = True
    payload["total_stage_s"] = total if any_seen else None
    return payload
