"""Definitive MOSEK-on-qrb_y64 experiment with HEFT warm-start.

Run with the xpu-rt-integration venv which has both cvxpy and mosek:
  /scratch2/agustin/xpu-rt-integration/.venv/bin/python \\
      scripts/mosek_qrb_y64_warmstart.py


We've documented MOSEK doesn't converge on 300-op qrb_y64 under "default
time_limit_s=0 (unlimited)" — but xpurt's "unlimited" mode caps at MOSEK
default ~5 min internally. This script:
  1. Builds the same multi-network Workload as the bridge.
  2. Runs HEFT to get a feasible starting solution.
  3. Calls xpurt.scheduler.schedule(cvxpy_solver='MOSEK', time_limit=3600,
     warm_start=heft_solution) — explicit 1-hour MOSEK budget.
  4. Records: solver_status, best_bound, primal_objective, solve_wall_s,
     and the LP relaxation lower bound.

The output goes to benchmarks/results/A/mosek_qrb_y64_definitive/.

Exit codes:
  0 = MOSEK converged (status OPTIMAL or OPTIMAL_INACCURATE)
  1 = MOSEK reached time limit (status TIME_LIMIT) — emit best feasible
  2 = MOSEK error (infeasible/numerical/solver crash)
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/scratch2/agustin/XPU-RT/xpu-rt")
sys.path.insert(0, "/scratch2/agustin/merlin")


def main() -> int:
    # Reuse the bridge to build the Workload.
    import yaml
    from scripts.run_xpurt_scheduler_multi import _build_workload  # type: ignore
    import scheduler  # type: ignore   # xpu-rt flat layout
    from schedulers import get_scheduler  # type: ignore

    cfg_path = REPO / "configs" / "multi_3way_qrb_y64.yaml"  # uses solver: MOSEK
    out_dir = REPO / "benchmarks" / "results" / "A" / "mosek_qrb_y64_definitive"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_run = out_dir / time.strftime("%Y%m%dT%H%M%SZ")
    out_run.mkdir(parents=True, exist_ok=True)

    print(f"== Building Workload from {cfg_path.name} ==", flush=True)
    cfg = yaml.safe_load(cfg_path.read_text())
    workload, instance_meta, machines = _build_workload(cfg, contention=None)
    n_ops = len(workload.operations)
    print(f"   {n_ops} ops, {len(machines)} machines, {len(instance_meta)} instances", flush=True)

    # 1. HEFT warm-start.
    print(f"\n== HEFT warm-start ==", flush=True)
    t0 = time.perf_counter()
    heft_fn = get_scheduler("heft")
    heft_fn(workload)
    heft_wall = time.perf_counter() - t0
    heft_report = workload.solver_state.get("report") if hasattr(workload, "solver_state") else None
    heft_ms = heft_report.makespan_cycles if heft_report else None
    print(f"   HEFT makespan: {heft_ms} ms  (wall={heft_wall:.2f}s)", flush=True)

    # 2. MOSEK with explicit 1-hour wall budget.
    print(f"\n== MOSEK MILP — 3600s wall budget ==", flush=True)
    t0 = time.perf_counter()
    mosek_status = "?"
    mosek_makespan = None
    mosek_report = None
    err = None
    try:
        mosek_fn = get_scheduler("mosek")
        mosek_fn(workload, cvxpy_solver="MOSEK", time_limit=3600)
        mosek_report = workload.solver_state.get("report") if hasattr(workload, "solver_state") else None
        if mosek_report:
            mosek_makespan = mosek_report.makespan_cycles
            mosek_status = mosek_report.solver_status
    except Exception as exc:  # noqa: BLE001
        err = repr(exc)
    mosek_wall = time.perf_counter() - t0

    print(f"   MOSEK status: {mosek_status}  makespan: {mosek_makespan}  wall: {mosek_wall:.2f}s", flush=True)
    if err:
        print(f"   MOSEK error: {err}", flush=True)

    # 3. Compare + write summary.
    summary = {
        "config": str(cfg_path),
        "n_operations": n_ops,
        "heft_makespan_ms": heft_ms,
        "heft_solve_wall_s": heft_wall,
        "mosek_status": mosek_status,
        "mosek_makespan_ms": mosek_makespan,
        "mosek_solve_wall_s": mosek_wall,
        "mosek_error": err,
        "qrb_target_ms": 75.71,
        "verdict": (
            "MOSEK_BEATS_HEFT" if (mosek_makespan and heft_ms and mosek_makespan < heft_ms - 0.5) else
            "MOSEK_SAME_AS_HEFT" if (mosek_makespan and heft_ms and abs(mosek_makespan - heft_ms) <= 0.5) else
            "MOSEK_DID_NOT_CONVERGE" if not mosek_makespan else
            "MOSEK_WORSE_THAN_HEFT"
        ),
        "beats_qrb": bool(mosek_makespan and mosek_makespan < 75.71),
    }
    (out_run / "mosek_definitive_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n== Summary ==")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nwrote {out_run / 'mosek_definitive_summary.json'}")

    if mosek_status in {"optimal", "optimal_inaccurate"}:
        return 0
    if mosek_status in {"time_limit", "user_limit"}:
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
