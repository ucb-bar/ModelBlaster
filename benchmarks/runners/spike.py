"""Thin wrapper around the captured harness stdout for spike runs.

The Arm drivers shell out to `examples/<model>/run.sh` which runs the
codegen+build+spike chain end-to-end. This module's job is to:

* parse the harness markers out of the captured stdout via the
  existing `modelblaster.validation.runner_common` helpers, and
* write the parsed result into the per-cell artifact layout
  expected by the aggregator (`accuracy.json`, `profile_spike.csv`,
  `wall_cycles.txt`).

Keeping this layer thin avoids drift from `_run_lib.sh`'s env-var
handling — we don't re-run anything, we just translate the stdout
the existing pipeline already produces.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

from modelblaster.benchmarks.ingest import cycles_per_op
from modelblaster.validation import runner_common


RUNNER_NAME = "spike"

_XPURT_TRACE_BEGIN = "=== MODELBLASTER_XPURT_TRACE_BEGIN ==="
_XPURT_TRACE_END = "=== MODELBLASTER_XPURT_TRACE_END ==="


def parse_stdout(stdout: str, *, tag: Optional[str] = None
                 ) -> dict[str, Any]:
    """Pull verify, profile, wall-cycles, and (if present) the XPU-RT
    execution trace out of the harness's stdout. `tag` is the
    single-model harness's None (untagged) or the multi-model
    harness's per-model tag string."""
    verify = runner_common.parse_verify(stdout, tag=tag)
    profile = runner_common.parse_profile(stdout, tag=tag) or []
    wall = runner_common.parse_wall_cycles(stdout, tag=tag)
    trace = _extract_xpurt_trace(stdout)
    return {
        "verify": verify,
        "profile": profile,
        "wall_cycles": wall,
        "xpurt_trace": trace,
    }


def _extract_xpurt_trace(stdout: str) -> Optional[str]:
    """Return the CSV body between MODELBLASTER_XPURT_TRACE_BEGIN and
    MODELBLASTER_XPURT_TRACE_END, or None if the markers are absent
    (the binary wasn't built with -DMODELBLASTER_XPURT_TRACE=ON)."""
    if _XPURT_TRACE_BEGIN not in stdout or _XPURT_TRACE_END not in stdout:
        return None
    start = stdout.index(_XPURT_TRACE_BEGIN) + len(_XPURT_TRACE_BEGIN)
    end = stdout.index(_XPURT_TRACE_END, start)
    return stdout[start:end].strip()


def write_accuracy(out_dir: Path, verify: Optional[dict[str, Any]],
                   atol: Optional[float] = None,
                   rtol: Optional[float] = None) -> None:
    """Translate the harness's verify summary into the aggregator's
    `accuracy.json` shape. `linf` here is the harness's
    `max_abs_err` (the spike harness's in-binary compare against the
    baked-in PyTorch golden). RMSE and cosine are not directly emitted
    by the harness; runners that compute them off-line (e.g. via
    compare_<model>.py) populate them downstream."""
    if verify is None:
        return
    data: dict[str, Any] = {
        "schema_version": 1,
        "linf": float(verify["max_abs_err"]),
        "rmse": None,
        "cosine": None,
        "n_samples": int(verify["n"]),
        "bit_exact": float(verify["max_abs_err"]) == 0.0,
        "verify_pass": True,
    }
    if atol is not None:
        data["atol_used"] = float(atol)
        # The harness already passed the compare against atol — record it.
        data["verify_pass"] = float(verify["max_abs_err"]) <= float(atol)
    if rtol is not None:
        data["rtol_used"] = float(rtol)
    with open(out_dir / "accuracy.json", "w") as f:
        json.dump(data, f, indent=2)


def write_profile_csv(out_dir: Path, profile: list[dict[str, Any]]
                      ) -> None:
    """Write the profile rows in IREE-shape (dispatch_id,name,op,
    shape,cycles), suffixed by the runner name. Also emits the
    per-op-kind breakdown into `cycles_per_op.json` so the aggregator
    can surface dominant-op metrics and the top-5 table in summary.md
    without re-parsing the CSV."""
    if not profile:
        return
    path = out_dir / f"profile_{RUNNER_NAME}.csv"
    header = list(profile[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in profile:
            w.writerow(row)
    breakdown = cycles_per_op.synthesize(profile)
    with open(out_dir / "cycles_per_op.json", "w") as f:
        json.dump(breakdown, f, indent=2)


def write_wall_cycles(out_dir: Path, wall_cycles: Optional[int]) -> None:
    if wall_cycles is None:
        return
    (out_dir / "wall_cycles.txt").write_text(f"{wall_cycles}\n")


def write_xpurt_trace(out_dir: Path, trace_csv: Optional[str]) -> None:
    """Write the parsed XPURT_TRACE block as ``xpurt_trace.csv``. The
    first stdout line of the block is the column header emitted by the
    harness; keep it intact."""
    if not trace_csv:
        return
    (out_dir / "xpurt_trace.csv").write_text(trace_csv + "\n")
