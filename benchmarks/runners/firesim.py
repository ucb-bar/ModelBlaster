"""FireSim-side counterpart to runners/spike.py.

Same translation shape — parse the harness stdout via
`modelblaster.validation.runner_common`, write the aggregator's
artifact files — but the profile CSV is named `profile_firesim.csv`.

The reason these are separate runner modules (rather than one with a
runner name parameter) is that the per-runner artifacts differ in
provenance: spike profile cycles are not authoritative on accelerator
targets (extensions execute atomically), and the aggregator uses the
filename suffix to enforce the cycle-source-honesty policy. Keeping
the writers separate keeps that policy explicit at the producer side
too.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

from modelblaster.benchmarks.ingest import cycles_per_op
from modelblaster.validation import runner_common


RUNNER_NAME = "firesim"

_XPURT_TRACE_BEGIN = "=== MODELBLASTER_XPURT_TRACE_BEGIN ==="
_XPURT_TRACE_END = "=== MODELBLASTER_XPURT_TRACE_END ==="


def parse_stdout(stdout: str, *, tag: Optional[str] = None
                 ) -> dict[str, Any]:
    verify = runner_common.parse_verify(stdout, tag=tag)
    profile = runner_common.parse_profile(stdout, tag=tag) or []
    wall = runner_common.parse_wall_cycles(stdout, tag=tag)
    trace_csv = _extract_xpurt_trace(stdout)
    trace_rows = cycles_per_op.parse_xpurt_trace_csv(trace_csv) if trace_csv else []
    return {
        "verify": verify,
        "profile": profile,
        "wall_cycles": wall,
        "xpurt_trace": trace_csv,
        "xpurt_trace_rows": trace_rows,
    }


def _extract_xpurt_trace(stdout: str) -> Optional[str]:
    if _XPURT_TRACE_BEGIN not in stdout or _XPURT_TRACE_END not in stdout:
        return None
    start = stdout.index(_XPURT_TRACE_BEGIN) + len(_XPURT_TRACE_BEGIN)
    end = stdout.index(_XPURT_TRACE_END, start)
    return stdout[start:end].strip()


def write_accuracy(out_dir: Path, verify: Optional[dict[str, Any]],
                   atol: Optional[float] = None,
                   rtol: Optional[float] = None) -> None:
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
        data["verify_pass"] = float(verify["max_abs_err"]) <= float(atol)
    if rtol is not None:
        data["rtol_used"] = float(rtol)
    with open(out_dir / "accuracy.json", "w") as f:
        json.dump(data, f, indent=2)


def write_profile_csv(out_dir: Path, profile: list[dict[str, Any]],
                      trace_rows: Optional[list[dict[str, Any]]] = None,
                      ) -> None:
    if not profile:
        return
    path = out_dir / f"profile_{RUNNER_NAME}.csv"
    header = list(profile[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in profile:
            w.writerow(row)
    breakdown = cycles_per_op.synthesize(profile, trace_rows=trace_rows)
    with open(out_dir / "cycles_per_op.json", "w") as f:
        json.dump(breakdown, f, indent=2)


def write_wall_cycles(out_dir: Path, wall_cycles: Optional[int]) -> None:
    if wall_cycles is None:
        return
    (out_dir / "wall_cycles.txt").write_text(f"{wall_cycles}\n")


def write_xpurt_trace(out_dir: Path, trace_csv: Optional[str]) -> None:
    if not trace_csv:
        return
    (out_dir / "xpurt_trace.csv").write_text(trace_csv + "\n")
    _maybe_emit_scheduler_postmortem(out_dir)


def _maybe_emit_scheduler_postmortem(out_dir: Path) -> None:
    """If xpurt's postmortem comparator is importable AND a scheduler_report.json
    sibling exists, write scheduler_postmortem.json. Silent no-op otherwise —
    postmortem is best-effort observability, not a required artifact."""
    trace_path = out_dir / "xpurt_trace.csv"
    if not trace_path.exists():
        return
    report_path = out_dir / "scheduler_report.json"
    try:
        from postmortem import compare_trace  # xpurt installable
    except ImportError:
        try:
            import sys
            sys.path.insert(0, "/scratch2/agustin/XPU-RT/xpu-rt")
            from postmortem import compare_trace
        except ImportError:
            return
    try:
        compare_trace(
            str(trace_path),
            str(report_path) if report_path.exists() else None,
            write_to=str(out_dir / "scheduler_postmortem.json"),
        )
    except Exception:
        # Best-effort: do not let postmortem errors break a successful run.
        return
