"""Bridge: ModelBlaster per-op cycle profiles → XPU-RT scheduler (MOSEK MIP) → ModelBlaster schedule fixture.

Pipeline:
    benchmarks/results/A/<single-backend>/<run>/profile_<runner>.csv   (per-op cycles)
      ─ convert ─►
    in-memory processing_times: dict[dispatch_name -> [time_per_machine, ...]]
      ─ xpu-rt scheduler.schedule(..., cvxpy_solver="MOSEK") ─►
    (t, alpha) — start times + machine assignments
      ─ emit ─►
    schedule_fixtures/<workload>_xpurt_mosek.json   (ModelBlaster-compatible fixture)

MOSEK is XPU-RT's "gold standard" — it solves the MIP from Anglani et al.
(https://doi.org/10.1016/j.cor.2023.106366 Section 2.1) to optimality (within
the time limit). HEFT and other list schedulers are useful baselines but
non-optimal. We benchmark against MOSEK because it's the reference for
"what's the best schedule possible for this workload."

Usage:
    PYTHONPATH=. uv run python -m scripts.run_xpurt_scheduler \\
        --workload dronet_hetero_int8 \\
        --target-backends gemmini,rvv_opu \\
        --runner firesim \\
        --output schedule_fixtures/dronet_xpurt_mosek.json \\
        [--time-limit 60]

Then update workloads.yaml so dronet_hetero_int8.xpurt_schedule_path points
at the new fixture, or add a new workload row `dronet_hetero_xpurt_mosek_int8`
that references it. Build + run that cell to compare cycles against the
hand-authored hetero baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
from typing import Optional

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results"
XPURT_ROOT = pathlib.Path("/scratch2/agustin/XPU-RT")
XPURT_PKG = XPURT_ROOT / "xpu-rt"
DRONET_DEPS_JSON = XPURT_PKG / "pytorch_workload" / "samples" / "dronet_dispatch_deps.json"

# Clock frequency for cycles → milliseconds. Targets all run at the
# emulated 1 GHz target clock; the absolute scale is irrelevant for
# scheduling (only relative weights matter), but keeping the units
# sensible matters for time_limit interpretation by MOSEK.
CYCLES_PER_MS = 1_000_000


def _find_latest_profile(workload: str, runner: str) -> pathlib.Path:
    """Most recent per-cell profile CSV for a (workload, runner)."""
    cell = RESULTS_ROOT / "A" / workload
    if not cell.exists():
        raise FileNotFoundError(f"no results dir: {cell}")
    profile_name = f"profile_{runner}.csv"
    candidates = sorted(
        [d for d in cell.iterdir() if d.is_dir() and d.name != "latest"
         and (d / profile_name).exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"no {profile_name} under any run dir of {workload}")
    return candidates[0] / profile_name


def _load_cycles_per_dispatch(csv_path: pathlib.Path) -> dict[int, int]:
    """Read ModelBlaster's profile CSV; return {dispatch_id -> cycles}."""
    out: dict[int, int] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            did = int(row["dispatch_id"])
            out[did] = int(row["cycles"])
    return out


def _emit_modelblaster_fixture(
    workload_name: str,
    dispatch_data: dict,
    operations,
    machines: list[str],
    t,                  # np.ndarray of start times (length n_ops)
    alpha,              # np.ndarray of machine assignments (length n_ops)
    proc_times: dict,   # {dispatch_name: [time_per_machine, ...]}
    out_path: pathlib.Path,
    solver: str,
    cycles_per_ms: float,
) -> None:
    """Convert XPU-RT scheduler output to the ModelBlaster fixture format
    that ingest_xpurt_schedule.py expects. Mirrors the schema of
    `schedule_fixtures/dronet_hetero_gemmini_opu.json`."""
    # XPU-RT emits 'CPU_P' / 'CPU_E' bare; ModelBlaster wants tile-suffixed
    # 'CPU_P#0' / 'CPU_E#0' (the tile index distinguishes multi-instance
    # machines of the same kind — here we have one of each).
    machine_to_target = {"CPU_P": "CPU_P#0", "CPU_E": "CPU_E#0"}

    dispatches_in = dispatch_data["dispatches"]

    # Operations are indexed in the order they were added to the Workload;
    # the workload_factory preserves the dispatch_deps.json insertion
    # order. Map back via the operation's name.
    out_dispatches: dict = {}
    for i, op in enumerate(operations):
        dname = op.operation_name
        # ModelBlaster's fixture keys are "<job>_dispatch_<id>". The
        # dispatch_deps.json uses bare "dispatch_<id>"; we prefix with the
        # job name (dronet).
        mb_key = f"{workload_name}_{dname}"
        did = dispatches_in[dname].get("id", i)
        deps_in = dispatches_in[dname].get("dependencies", [])
        deps_out = [f"{workload_name}_{d}" for d in deps_in]
        mach_idx = int(np.argmax(alpha[i]))
        kind = machines[mach_idx]
        hw_target = machine_to_target.get(kind, kind)
        duration = float(proc_times[dname][mach_idx])
        # Module name follows the existing fixture style:
        # "<job>$dispatch_<id>_<BACKEND>_<op>". Op name isn't strictly
        # required for ModelBlaster's ingest but it matches the
        # convention used by hand-authored fixtures and helps debugging.
        backend_tag = {"CPU_P": "GEMMINI", "CPU_E": "RVV_OPU"}.get(kind, kind)
        # We don't have the op kind here without re-reading the IR; use
        # a generic "op" tag — ingest_xpurt_schedule pulls the actual op
        # from graph.json by dispatch_id anyway.
        module_name = f"{workload_name}$dispatch_{did}_{backend_tag}_op"
        out_dispatches[mb_key] = {
            "id": did,
            "ordinal": 1,
            "total": 1,
            "dependencies": deps_out,
            "hardware_target": hw_target,
            "start_time": float(t[i]) * cycles_per_ms,  # ms → cycles
            "duration": duration * cycles_per_ms,
            "job_name": workload_name,
            "module_name": module_name,
        }

    fixture = {
        "dot_file": f"{workload_name}_xpurt_{solver.lower()}.json",
        "_provenance": {
            "generated_by": "scripts/run_xpurt_scheduler.py",
            "solver": solver,
            "cycles_per_ms_scale": cycles_per_ms,
            "notes": [
                f"Schedule produced by XPU-RT's {solver} solver from "
                f"measured per-op cycle profiles. Each dispatch's machine "
                f"assignment minimizes the workload's makespan subject to "
                f"the dependency graph in dronet_dispatch_deps.json and "
                f"the per-(dispatch, machine) processing times derived "
                f"from standalone-backend FireSim runs.",
            ],
        },
        "dispatches": out_dispatches,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(fixture, f, indent=2)
    print(f"wrote {out_path}  ({len(out_dispatches)} dispatches)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True,
        help="ModelBlaster hetero workload id (e.g. dronet_hetero_int8). "
             "We derive single-backend workload names by replacing "
             "_hetero_ with the backend's suffix.")
    ap.add_argument("--target-backends", required=True,
        help="comma list of backends, in CPU_P,CPU_E order. "
             "Example: gemmini,rvv_opu (gemmini→CPU_P#0, rvv_opu→CPU_E#0)")
    ap.add_argument("--runner", default="firesim",
        choices=["firesim", "spike"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--solver", default="MOSEK",
        help="cvxpy backend solver (default MOSEK)")
    ap.add_argument("--time-limit", type=float, default=120.0,
        help="MOSEK solve time limit in seconds (default 120)")
    ap.add_argument("--dispatch-deps",
        default=str(DRONET_DEPS_JSON),
        help="path to <model>_dispatch_deps.json (XPU-RT-side)")
    ap.add_argument("--model-name", default="dronet",
        help="model name to use in fixture keys (default dronet)")
    ap.add_argument("--cycles-source", default="db", choices=["db", "csv"],
        help="db: read median across reps from benchmarks/profile_db (default). "
             "csv: read the single most-recent profile_firesim.csv per backend.")
    ap.add_argument("--cycles-agg", default="median",
        choices=["median", "mean", "min", "max"],
        help="aggregation across reps when --cycles-source=db (default median)")
    args = ap.parse_args()

    backends = [b.strip() for b in args.target_backends.split(",")]
    if len(backends) != 2:
        ap.error("need exactly 2 backends today (CPU_P, CPU_E)")

    # 1. Load the dispatch dep graph (XPU-RT's pre-built shape).
    deps_path = pathlib.Path(args.dispatch_deps)
    if not deps_path.exists():
        print(f"error: dispatch_deps not found at {deps_path}",
              file=sys.stderr)
        return 1
    with deps_path.open() as f:
        dispatch_data = json.load(f)

    n_dispatches = len(dispatch_data["dispatches"])
    print(f"loaded {n_dispatches} dispatches from {deps_path}")

    # 2. Pull per-op cycles from each backend's standalone profile.
    base = args.workload.replace("_hetero_int8", "")
    processing_times: dict[str, list[float]] = {}
    for dname, dinfo in dispatch_data["dispatches"].items():
        did = dinfo["id"]
        processing_times[dname] = [0.0] * len(backends)
    # profile_db lookup expects target without _int8 suffix
    sys.path.insert(0, str(REPO_ROOT))
    from benchmarks.profile_db import query as profile_db_query  # type: ignore

    for bs_idx, bs in enumerate(backends):
        if args.cycles_source == "db":
            cycles = profile_db_query(
                network=args.model_name,
                target=bs,
                quant="int8",
                agg=args.cycles_agg,
            )
            if not cycles:
                print(f"error: profile_db has no records for "
                      f"({args.model_name}, {bs}, int8). "
                      f"Run `python -m benchmarks.profile_db ingest` first, "
                      f"or use --cycles-source csv.", file=sys.stderr)
                return 1
            print(f"  {bs}: {len(cycles)} cycles from profile_db "
                  f"(agg={args.cycles_agg})")
        else:
            single_workload = f"{base}_{bs}_int8"
            try:
                csv_path = _find_latest_profile(single_workload, args.runner)
            except FileNotFoundError as e:
                print(f"error: missing profile for {bs}: {e}", file=sys.stderr)
                return 1
            cycles = _load_cycles_per_dispatch(csv_path)
            print(f"  {bs}: read {len(cycles)} cycles from {csv_path}")
        for dname, dinfo in dispatch_data["dispatches"].items():
            did = dinfo["id"]
            if did in cycles:
                processing_times[dname][bs_idx] = cycles[did] / CYCLES_PER_MS
            else:
                # Dispatch absent from this backend's profile —
                # typically a zero-cost alias (view). Set a tiny
                # nonzero cost so MOSEK can place it.
                processing_times[dname][bs_idx] = 0.001

    # 3. Set up machines + transfer times. We have one of each
    # (CPU_P=gemmini tile, CPU_E=rvv_opu tile).
    machines = ["CPU_P", "CPU_E"]
    # Cross-tile transfer is non-zero but small for ms-scale ops.
    # Treat as constant ~0.01ms (a few hundred cycles at 1GHz) — the
    # actual cost is dominated by compute for accelerator workloads.
    import numpy as np
    transfer_times = np.array([[0.0,  0.01],
                               [0.01, 0.0]], dtype=float)

    # 4. Invoke the XPU-RT scheduler.
    sys.path.insert(0, str(XPURT_PKG))
    from workload_factory import create_workload_from_dependencies  # type: ignore
    from scheduler import schedule  # type: ignore

    workload = create_workload_from_dependencies(
        dispatch_data, processing_times, machines, transfer_times)
    print(f"built Workload: {len(workload.operations)} ops, "
          f"machines={machines}")

    # Constraint: requantization-producing ops (conv2d_s8, linear_s8) MUST
    # run on the gemmini tile. Mixing requantize across backends triggers
    # the cross-backend numerical drift bug documented in
    # notes/known_issues.md — gemmini and rvv_opu use different
    # multiply-and-shift rounding policies, so per-op intermediate outputs
    # disagree slightly and the chain accumulates linf~52 error. Each
    # backend is bit-exact END-TO-END against PyTorch golden as long as
    # the entire requantize chain stays on one backend.
    #
    # We expose this to MOSEK by marking CPU_E (rvv_opu) as infeasible
    # for those ops. Structural ops (maxpool, bn, relu, add, sigmoid)
    # are layout-passthrough — they don't requantize, so they can mix.
    #
    # combination index: 0 = CPU_P, 1 = CPU_E (matches our `machines`
    # list order; each machine_combination is a singleton in our setup).
    REQUANTIZE_OPS = {"conv2d_s8", "linear_s8"}
    # Pull op-kind from the IR (graph.json) since dispatch_deps.json
    # doesn't carry it.
    ir_path = (REPO_ROOT / "examples" / args.model_name / "int8"
               / "generated" / "graph.json")
    if ir_path.exists():
        ir = json.loads(ir_path.read_text())
        op_kind_by_did = {op["dispatch_id"]: op["op"]
                          for op in ir["ops"]
                          if op.get("dispatch_id") is not None}
        cpu_e_index = machines.index("CPU_E")
        constrained = 0
        for op in workload.operations:
            dname = op.operation_name  # "dispatch_<id>"
            try:
                did = int(dname.split("_")[1])
            except Exception:
                continue
            kind = op_kind_by_did.get(did)
            if kind in REQUANTIZE_OPS:
                op.infeasible_combinations = (
                    set(op.infeasible_combinations) | {cpu_e_index})
                constrained += 1
        print(f"infeasibility: constrained {constrained} requantize ops "
              f"({REQUANTIZE_OPS}) to CPU_P only")
    else:
        print(f"warning: no IR at {ir_path}; not constraining backends")

    print(f"invoking xpu-rt scheduler.schedule (solver={args.solver}, "
          f"time_limit={args.time_limit}s)...")
    t, alpha, _fused, _fmap = schedule(
        workload,
        cvxpy_solver=args.solver,
        time_limit=args.time_limit,
        verbose=True,
    )

    end_times = np.array(
        [t[i] + processing_times[op.operation_name][int(np.argmax(alpha[i]))]
         for i, op in enumerate(workload.operations)])
    makespan_ms = float(np.max(end_times))
    print(f"schedule produced: makespan_ms={makespan_ms:.3f}")

    # 5. Emit ModelBlaster fixture.
    _emit_modelblaster_fixture(
        workload_name=args.model_name,
        dispatch_data=dispatch_data,
        operations=workload.operations,
        machines=machines,
        t=t,
        alpha=alpha,
        proc_times=processing_times,
        out_path=pathlib.Path(args.output),
        solver=args.solver,
        cycles_per_ms=CYCLES_PER_MS,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
