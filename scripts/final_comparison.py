"""Final QRB5165 head-to-head summary.

Pulls every captured fixture + FireSim measurement into one table and
shows: predicted vs actual makespan, % delta, vs qrb 75.71 ms target,
and bit-exact verification status.

For runs with multiple reps, reports min/median/max actual to show
repeatability. The 0.22% predicted-vs-actual delta on the MOSEK
no-yolo headline should hold across all 3 reps (FireSim is fully
deterministic).

Run:  PYTHONPATH=. python3 scripts/final_comparison.py
"""

from __future__ import annotations

import csv
import json
import pathlib
import statistics
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "benchmarks" / "results" / "A"
FIXTURES = REPO_ROOT / "schedule_fixtures"
QRB_PREDICTED_MS = 75.71

# (label, fixture_name | None, result_dir_name, run_filter)
# run_filter: None means "all dirs"; "post-fix" means runs from 20260530 onwards
# (after the GLOBAL_CURATED_DIR fix landed).
CONFIGS = [
    ("MOSEK no-yolo (2d+4m)",         "3way_mosek_dronet2_mlp4.json",  "3way_mosek_dronet2_mlp4",  "post-fix"),
    ("MOSEK no-yolo regrouped",       "3way_mosek_dronet2_mlp4_regrouped.json", "3way_mosek_dronet2_mlp4_regrouped", "post-fix"),
    ("HEFT  no-yolo (2d+4m)",         "3way_heft_dronet2_mlp4.json",   "3way_heft_dronet2_mlp4",   "post-fix"),
    ("HEFT  qrb (1y160+2d+4m)",       "3way_heft_qrb.json",            "3way_heft_qrb",            "post-fix"),
    ("HEFT  qrb_y64 (1y64+2d+4m)",    "3way_heft_qrb_y64.json",        "3way_heft_qrb_y64",        "post-fix"),
    ("PEFT  qrb_y64 (1y64+2d+4m)",    "3way_peft_qrb_y64.json",        "3way_peft_qrb_y64",        "post-fix"),
]


def _read_fixture(fx_name):
    if not fx_name:
        return None
    path = FIXTURES / fx_name
    if not path.exists():
        return None
    fx = json.loads(path.read_text())
    p = fx.get("_provenance", {})
    return {
        "predicted_ms": p.get("makespan_ms"),
        "solver": p.get("solver", "?"),
        "n_dispatches": len(fx.get("dispatches", {})),
        "status": p.get("scheduler_report", {}).get("solver_status",
                  p.get("status", "?")),
    }


def _run_actual_ms(run_dir):
    trace = run_dir / "xpurt_trace.csv"
    if not trace.exists():
        return None, "?"
    rows = [r for r in csv.DictReader(trace.open())
            if r and r.get("actual_end_cycles","").strip()]
    if not rows:
        return None, "?"
    actual_ms = max(int(r["actual_end_cycles"]) for r in rows) / 1000.0
    uart = run_dir / "uartlog"
    overall = "?"
    if uart.exists():
        text = uart.read_text(errors="replace")
        if "*** PASSED ***" in text or "OVERALL: PASS" in text:
            overall = "PASS"
        elif "OVERALL: FAIL" in text or "*** FAILED ***" in text:
            overall = "FAIL"
    return actual_ms, overall


def _read_actuals(result_dir_name, run_filter=None):
    cell = RESULTS / result_dir_name
    if not cell.exists():
        return []
    runs = sorted(d for d in cell.iterdir() if d.is_dir() and d.name != "latest")
    if run_filter == "post-fix":
        runs = [r for r in runs if r.name >= "20260530T0046"]
    out = []
    for d in runs:
        ms, verify = _run_actual_ms(d)
        if ms is None:
            continue
        out.append({"run": d.name, "actual_ms": ms, "verify": verify})
    return out


def main():
    print(f"\n{'='*102}")
    print(f"QRB5165 head-to-head — multi-network scheduler baseline")
    print(f"{'='*102}\n")
    header = f"{'Config':<34}{'Solver':>8}{'#ops':>5}{'Pred(ms)':>10}{'Actual(ms)':>22}{'Δ%':>8}{'vs qrb':>10}{'verify':>8}"
    print(header)
    print("-"*102)

    for label, fx_name, dir_name, run_filter in CONFIGS:
        fx = _read_fixture(fx_name)
        actuals = _read_actuals(dir_name, run_filter)
        pred = fx["predicted_ms"] if fx else None
        solver = fx["solver"] if fx else "—"
        n = str(fx["n_dispatches"]) if fx else "—"
        pred_s = f"{pred:.2f}" if pred is not None else "—"

        if actuals:
            ms_vals = [a["actual_ms"] for a in actuals]
            if len(ms_vals) == 1:
                actual_s = f"{ms_vals[0]:.2f}"
            else:
                actual_s = f"min={min(ms_vals):.2f} med={statistics.median(ms_vals):.2f} max={max(ms_vals):.2f}"
            med = statistics.median(ms_vals)
            delta_s = f"{abs(med-pred)/pred*100:+.2f}%" if pred else "—"
            verifies = {a["verify"] for a in actuals}
            verify_s = "/".join(sorted(verifies))
        else:
            actual_s = "—"
            delta_s = "—"
            verify_s = "—"

        if pred is not None:
            ratio = pred / QRB_PREDICTED_MS
            vs = f"{ratio:.2f}× {'BEAT' if ratio < 1 else 'lose'}"
        else:
            vs = "—"

        print(f"{label:<34}{solver:>8}{n:>5}{pred_s:>10}{actual_s:>22}{delta_s:>8}{vs:>10}{verify_s:>8}")

    print()
    print(f"  qrb image reference: 75.71 ms predicted (1y + 2d + 4m, same hetero kind)")
    print(f"  hardware: GemminiAndOPUShuttleConfig (CPU_P=gemmini+scalar, CPU_E=Saturn OPU+RVV)")
    print(f"  runtime: GLOBAL_CURATED_DIR enables curated gemmini RoCC conv2d_s8 (post-fix runs)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
