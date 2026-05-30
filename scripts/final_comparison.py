"""Final QRB5165 head-to-head summary.

Pulls every captured fixture + FireSim measurement into one table and emits
a markdown summary that can be pasted into notes/baseline_2026-05-28.md.

Outputs:
  - stdout: human-readable table
  - notes/qrb5165_head_to_head.md: markdown table for the doc
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "benchmarks" / "results" / "A"
FIXTURES = REPO_ROOT / "schedule_fixtures"
QRB_PREDICTED_MS = 75.71

CONFIGS = [
    # (label, fixture_path, result_dir_pattern, notes)
    ("MOSEK no-yolo (2d+4m)",     "3way_mosek_dronet2_mlp4.json",   "3way_mosek_dronet2_mlp4"),
    ("HEFT  no-yolo (2d+4m)",     "3way_heft_dronet2_mlp4.json",    "3way_heft_dronet2_mlp4"),
    ("HEFT  qrb (1y160+2d+4m)",   "3way_heft_qrb.json",             "3way_heft_qrb"),
    ("MOSEK qrb (1y160+2d+4m)",   "3way_mosek_qrb.json",            "3way_mosek_qrb"),
    ("Naive baseline (1y160+1d+4m)",       None, "3way_baseline"),
    ("Naive conservative (1y160+2d+9m)",   None, "3way_conservative"),
    ("Naive camera-30hz (1y160+14d+45m)",  None, "3way_camera-30hz"),
    ("Naive camera-60hz (1y160+28d+90m)",  None, "3way_camera-60hz"),
    ("Naive imu-only-hi (1y160+1d+90m)",   None, "3way_imu-only-hi"),
]


def _read_fixture(path: pathlib.Path):
    if not path or not path.exists():
        return None
    fx = json.loads(path.read_text())
    p = fx.get("_provenance", {})
    return {
        "predicted_ms": p.get("makespan_ms"),
        "solver": p.get("solver", "?"),
        "n_dispatches": len(fx.get("dispatches", {})),
        "status": p.get("scheduler_report", {}).get("solver_status", "?"),
    }


def _read_actual(result_dir_pattern: str):
    cell = RESULTS / result_dir_pattern
    if not cell.exists():
        return None
    runs = sorted(d for d in cell.iterdir() if d.is_dir() and d.name != "latest")
    if not runs:
        return None
    last = runs[-1]
    trace = last / "xpurt_trace.csv"
    if not trace.exists():
        return None
    rows = [r for r in csv.DictReader(trace.open()) if r and r.get("actual_end_cycles","").strip()]
    if not rows:
        return None
    actual_ms = max(int(r["actual_end_cycles"]) for r in rows) / 1000.0  # mtime → ms
    # pass/fail
    uart = (last / "uartlog")
    overall = "?"
    if uart.exists():
        text = uart.read_text(errors="replace")
        if "OVERALL: PASS" in text:
            overall = "PASS"
        elif "OVERALL: FAIL" in text:
            overall = "FAIL"
    return {"actual_ms": actual_ms, "overall": overall, "run_id": last.name}


def main() -> int:
    print(f"{'Config':<40}{'Solver':>8}{'#ops':>6}{'Pred(ms)':>10}{'Actual(ms)':>12}{'vs qrb 75.71':>14}{'verify':>8}")
    print("-"*100)
    rows = []
    for label, fx_name, dir_pat in CONFIGS:
        fx_data = _read_fixture(FIXTURES / fx_name) if fx_name else None
        actual_data = _read_actual(dir_pat)
        rows.append((label, fx_data, actual_data))
        pred = fx_data["predicted_ms"] if fx_data else None
        actual = actual_data["actual_ms"] if actual_data else None
        solver = fx_data["solver"] if fx_data else "naive"
        n = fx_data["n_dispatches"] if fx_data else "-"
        verify = actual_data["overall"] if actual_data else "-"
        pred_s = f"{pred:.1f}" if pred is not None else "—"
        actual_s = f"{actual:.1f}" if actual is not None else "—"
        # vs qrb
        if pred is not None:
            ratio = pred / QRB_PREDICTED_MS
            vs = f"{ratio:.2f}× ({'BEAT' if ratio < 1 else 'lose'})"
        else:
            vs = "—"
        print(f"{label:<40}{solver:>8}{n:>6}{pred_s:>10}{actual_s:>12}{vs:>14}{verify:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
