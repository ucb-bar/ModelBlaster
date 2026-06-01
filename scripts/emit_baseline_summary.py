"""Emit a machine-readable baseline summary (JSON + CSV).

Reads schedule fixtures + the latest valid FireSim captures and produces:
  - notes/baseline_summary.json — schema-versioned dump for downstream tools
  - notes/baseline_summary.csv — flat row-per-(workload, solver, schedule_kind)

Run: python3 scripts/emit_baseline_summary.py
"""
from __future__ import annotations

import csv
import json
import pathlib
import statistics
from collections import defaultdict

REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "benchmarks" / "results" / "A"
FIXTURES = REPO / "schedule_fixtures"
NOTES = REPO / "notes"

QRB_PREDICTED_MS = 75.71

# (label, mix, kind, fixture, results_dir, run_filter)
ENTRIES = [
    ("MOSEK PACKED no-yolo",       "2d+4m",        "packed",   "3way_mosek_dronet2_mlp4.json",            "3way_mosek_dronet2_mlp4",            "post-fix"),
    ("HEFT  PACKED no-yolo",       "2d+4m",        "packed",   "3way_heft_dronet2_mlp4.json",             "3way_heft_dronet2_mlp4",             "post-fix"),
    ("HEFT  PACKED qrb_y64",       "1y64+2d+4m",   "packed",   "3way_heft_qrb_y64.json",                  "3way_heft_qrb_y64",                  "post-fix"),
    ("PEFT  PACKED qrb_y64",       "1y64+2d+4m",   "packed",   "3way_peft_qrb_y64.json",                  "3way_peft_qrb_y64",                  "post-fix"),
    ("MOSEK PERIODIC no-yolo",     "2d+4m@27+53Hz","periodic", "3way_mosek_dronet2_mlp4_periodic.json",   "3way_mosek_dronet2_mlp4_periodic",   "post-fix"),
    ("Partition PERIODIC no-yolo", "2d+4m@27+53Hz","periodic", "3way_partitioned_dronet2_mlp4.json",      "3way_partitioned_dronet2_mlp4",      "post-fix"),
    ("Partition PERIODIC qrb_y64", "1y+2d+4m@13+27+53Hz","periodic","3way_partitioned_qrb_y64.json",     "3way_partitioned_qrb_y64",           "post-fix"),
]


def _read_fixture(fx_name: str) -> dict | None:
    path = FIXTURES / fx_name
    if not path.exists():
        return None
    fx = json.loads(path.read_text())
    p = fx.get("_provenance", {})
    rep = p.get("scheduler_report") or {}
    return {
        "predicted_ms": p.get("makespan_ms"),
        "solver": p.get("solver", "?"),
        "n_dispatches": len(fx.get("dispatches", {})),
        "solver_status": rep.get("solver_status", p.get("status", "?")),
        "solve_wall_s": p.get("solve_wall_s"),
        "deadline_misses": p.get("deadline_misses", 0),
        "utilization": rep.get("utilization", {}),
        "critical_path_ms": rep.get("critical_path"),
        "cross_device_transitions": rep.get("cross_device_transitions"),
    }


def _read_reps(result_dir: str, expected_n: int | None, run_filter: str | None) -> list[dict]:
    cell = RESULTS / result_dir
    if not cell.exists():
        return []
    runs = sorted(d for d in cell.iterdir() if d.is_dir() and d.name != "latest")
    if run_filter == "post-fix":
        runs = [r for r in runs if r.name >= "20260530T0046"]
    out = []
    for d in runs:
        trace = d / "xpurt_trace.csv"
        if not trace.exists():
            continue
        rows = [r for r in csv.DictReader(trace.open()) if r and r.get("actual_end_cycles", "").strip()]
        if not rows:
            continue
        if expected_n is not None and len(rows) != expected_n:
            continue
        actual_ms = max(int(r["actual_end_cycles"]) for r in rows) / 1000.0
        verify = "?"
        stdout = d / "run_stdout.log"
        if stdout.exists():
            text = stdout.read_text(errors="replace")
            if "OVERALL: FAIL" in text:
                verify = "FAIL"
            elif "OVERALL: PASS" in text:
                verify = "PASS"
        out.append({"run": d.name, "actual_ms": actual_ms, "verify": verify, "n_dispatches": len(rows)})
    return out


def main() -> int:
    rows: list[dict] = []
    for label, mix, kind, fx_name, dir_name, run_filter in ENTRIES:
        fx = _read_fixture(fx_name)
        if fx is None:
            continue
        reps = _read_reps(dir_name, fx["n_dispatches"], run_filter)
        actual_summary = None
        if reps:
            ms_vals = [r["actual_ms"] for r in reps]
            verifies = sorted({r["verify"] for r in reps})
            actual_summary = {
                "n_reps": len(reps),
                "actual_ms_min": min(ms_vals),
                "actual_ms_med": statistics.median(ms_vals),
                "actual_ms_max": max(ms_vals),
                "delta_pct_med": 100.0 * (statistics.median(ms_vals) - fx["predicted_ms"]) / fx["predicted_ms"],
                "verify": "/".join(verifies),
                "per_rep": reps,
            }
        rows.append({
            "label": label,
            "mix": mix,
            "schedule_kind": kind,
            "solver": fx["solver"],
            "n_dispatches": fx["n_dispatches"],
            "solver_status": fx["solver_status"],
            "solve_wall_s": fx["solve_wall_s"],
            "deadline_misses": fx["deadline_misses"],
            "predicted_ms": fx["predicted_ms"],
            "predicted_vs_qrb": fx["predicted_ms"] / QRB_PREDICTED_MS if fx["predicted_ms"] else None,
            "actual": actual_summary,
            "fixture_path": str(FIXTURES / fx_name),
            "results_dir": dir_name,
        })

    summary = {
        "schema_version": 1,
        "generated_at": "2026-05-31",  # deterministic
        "bitstream": "GemminiAndOPUShuttleConfig",
        "qrb_image_predicted_ms": QRB_PREDICTED_MS,
        "rows": rows,
        "notes": {
            "yolov8_critical_path_ms": 55.77,
            "yolov8_critical_path_note": "yolov8_nano_64 sequential critical path — absolute floor for any schedule containing 1 yolov8",
            "ideal_parallel_floor_ms_1y_2d_4m": 53.20,
            "curated_kernel_verify_gate_status": "blocked: west subprocess loses PATH; curated rvv_opu kernels fall back to scalar reference at runtime",
            "operator_fusion_status": "Conv+SiLU detection wired in extract_int8 + CONV2D_SILU_S8 KernelSpec registered, but doesn't fire on yolov8 which has Conv→BN→SiLU (3 separate ops). Needs Conv+BN+SiLU 3-fold.",
        },
    }
    (NOTES / "baseline_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Flat CSV
    cols = [
        "label", "mix", "schedule_kind", "solver", "n_dispatches", "solver_status",
        "solve_wall_s", "deadline_misses", "predicted_ms", "predicted_vs_qrb",
        "n_reps", "actual_ms_min", "actual_ms_med", "actual_ms_max",
        "delta_pct_med", "verify",
    ]
    with open(NOTES / "baseline_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            actual = r.get("actual") or {}
            w.writerow({
                "label": r["label"],
                "mix": r["mix"],
                "schedule_kind": r["schedule_kind"],
                "solver": r["solver"],
                "n_dispatches": r["n_dispatches"],
                "solver_status": r["solver_status"],
                "solve_wall_s": f"{r['solve_wall_s']:.3f}" if r["solve_wall_s"] is not None else "",
                "deadline_misses": r["deadline_misses"],
                "predicted_ms": f"{r['predicted_ms']:.3f}",
                "predicted_vs_qrb": f"{r['predicted_vs_qrb']:.3f}" if r["predicted_vs_qrb"] is not None else "",
                "n_reps": actual.get("n_reps", 0),
                "actual_ms_min": f"{actual['actual_ms_min']:.3f}" if actual else "",
                "actual_ms_med": f"{actual['actual_ms_med']:.3f}" if actual else "",
                "actual_ms_max": f"{actual['actual_ms_max']:.3f}" if actual else "",
                "delta_pct_med": f"{actual['delta_pct_med']:+.3f}" if actual else "",
                "verify": actual.get("verify", "") if actual else "",
            })

    print(f"wrote {NOTES / 'baseline_summary.json'}")
    print(f"wrote {NOTES / 'baseline_summary.csv'}")
    print()
    print("Summary:")
    for r in rows:
        actual = r.get("actual") or {}
        ms_str = f"{actual['actual_ms_med']:.2f} ({actual['n_reps']} reps, {actual['verify']})" if actual else "—"
        print(f"  {r['label']:36s} pred={r['predicted_ms']:>6.2f} ms  actual={ms_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
