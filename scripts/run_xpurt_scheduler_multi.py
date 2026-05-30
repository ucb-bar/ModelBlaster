"""Multi-network MOSEK MIP scheduler bridge.

Takes a YAML spec like configs/multi_3way_qrb.yaml that lists networks +
instance counts. Builds one combined Workload (every dispatch of every
instance becomes an Operation), applies cross-backend drift constraints
via infeasible_combinations, solves via xpurt.schedule() with MOSEK, and
emits a ModelBlaster schedule fixture JSON.

The bitstream is locked to the GemminiAndOPUShuttle hetero pair:
  CPU_P#0 = Gemmini RoCC + scalar fallback (Shuttle)
  CPU_E#0 = Saturn OPU + RVV (Shuttle)
No other bitstream variants are supported.

Usage:
    PYTHONPATH=. uv run python -m scripts.run_xpurt_scheduler_multi \\
        --config configs/multi_3way_qrb.yaml \\
        --output schedule_fixtures/3way_mosek_qrb.json

    # iterative tuning loop with contention multipliers
    PYTHONPATH=. uv run python -m scripts.run_xpurt_scheduler_multi \\
        --config configs/multi_3way_qrb.yaml \\
        --contention benchmarks/profile_db/contention_multipliers.json \\
        --output schedule_fixtures/3way_mosek_qrb_v2.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results"
XPURT_PKG = pathlib.Path("/scratch2/agustin/XPU-RT/xpu-rt")


def _build_dep_graph(ir: dict) -> Tuple[Dict[int, list[int]], Dict[int, str], Dict[int, str]]:
    """From a ModelBlaster graph.json, derive {dispatch_id -> dep ids},
    {dispatch_id -> op_type}, {dispatch_id -> op_name}.
    """
    ops = ir["ops"]
    tensor_producer: Dict[str, int] = {}
    for i, op in enumerate(ops):
        for t in op.get("outputs", []):
            tensor_producer[t] = i

    def resolve(tensor_name: str) -> Optional[int]:
        seen: set = set()
        while True:
            if tensor_name in seen:
                return None
            seen.add(tensor_name)
            if tensor_name not in tensor_producer:
                return None
            op = ops[tensor_producer[tensor_name]]
            did = op.get("dispatch_id")
            if did is not None:
                return did
            ins = op.get("inputs", [])
            if not ins:
                return None
            tensor_name = ins[0]

    deps: Dict[int, list[int]] = {}
    kind: Dict[int, str] = {}
    name: Dict[int, str] = {}
    for op in ops:
        did = op.get("dispatch_id")
        if did is None:
            continue
        kind[did] = op["op"]
        name[did] = op["name"]
        deps[did] = sorted({
            d for d in (resolve(t) for t in op.get("inputs", []))
            if d is not None and d != did
        })
    return deps, kind, name


def _build_workload(cfg: dict, contention: Optional[dict]):
    """Build an xpurt Workload from the YAML config + profile DB cycles."""
    sys.path.insert(0, str(XPURT_PKG))
    sys.path.insert(0, str(REPO_ROOT))
    from workload import Operation, Workload  # type: ignore
    from benchmarks.profile_db import query as profile_db_query  # type: ignore

    # Bitstream is locked: two singleton-combo machines (one op at a time
    # per tile). transfer_times is small but nonzero so the MIP prefers
    # keeping a chain on one tile when feasible.
    machines = ["CPU_P#0", "CPU_E#0"]
    transfer_times = np.array([[0.0, 0.01], [0.01, 0.0]], dtype=float)

    requantize_ops = set(cfg.get("requantize_ops", []))
    # Per-network whole-chain affinity: force all ops of a given network
    # to one tile. Heavier than requantize_ops (which is per-op-type) —
    # pin_to_cpu_p kills any rvv_opu placement for those networks, so the
    # whole inference chain is bit-exact-preserving on one backend. Use
    # when a network's int8 LUT kernels diverge across backends and the
    # schedule cost of forcing it onto one side is acceptable.
    pin_to_cpu_p = set(cfg.get("pin_to_cpu_p", []))
    pin_to_cpu_e = set(cfg.get("pin_to_cpu_e", []))
    cycles_per_ms = float(cfg.get("cycles_per_ms", 1_000_000))
    mults: Dict[str, float] = (contention or {}).get("multipliers", {})

    # Index ops by network name → reused across instances of that network.
    cache: Dict[str, dict] = {}

    def _load_network(network: str, quant: str) -> dict:
        key = f"{network}|{quant}"
        if key in cache:
            return cache[key]
        ir_path = REPO_ROOT / "examples" / network / quant / "generated" / "graph.json"
        ir = json.loads(ir_path.read_text())
        deps, kind, op_name = _build_dep_graph(ir)
        # Cycles per backend for this (network, quant). profile_db stores
        # solo cycles in rdcycle; divide by cycles_per_ms (1e6) for ms.
        cyc_gemmini = profile_db_query(network, "gemmini", quant, agg=cfg.get("cycles_agg", "median"))
        cyc_opu = profile_db_query(network, "rvv_opu", quant, agg=cfg.get("cycles_agg", "median"))
        cache[key] = {
            "deps": deps,
            "kind": kind,
            "op_name": op_name,
            "cyc_gemmini": cyc_gemmini,
            "cyc_opu": cyc_opu,
        }
        return cache[key]

    all_ops: List = []
    by_key: Dict[str, "Operation"] = {}  # key="<network>#<inst>_dispatch_<did>"

    next_job_id = 0
    instance_meta: List[dict] = []  # for fixture emission

    for net_cfg in cfg["networks"]:
        network = net_cfg["name"]
        quant = net_cfg["quant"]
        n_inst = int(net_cfg.get("instances", 1))
        data = _load_network(network, quant)

        for inst in range(n_inst):
            inst_prefix = f"{network}#{inst}"
            inst_job_id = next_job_id
            next_job_id += 1

            # First pass: create Operations (no predecessors yet — we
            # need all op objects before we can wire deps).
            op_by_did: Dict[int, "Operation"] = {}
            for did in sorted(data["deps"]):
                op_type = data["kind"][did]
                # Per-dispatch processing time (ms) on each machine.
                g_cyc = data["cyc_gemmini"].get(did, 0)
                o_cyc = data["cyc_opu"].get(did, 0)
                # Apply contention multipliers if provided.
                g_mult = mults.get(f"{network}|{op_type}|gemmini", 1.0)
                o_mult = mults.get(f"{network}|{op_type}|rvv_opu", 1.0)
                g_ms = max(0.001, g_cyc * g_mult / cycles_per_ms)
                o_ms = max(0.001, o_cyc * o_mult / cycles_per_ms)

                op = Operation(
                    processing_times=[g_ms, o_ms],
                    operation_name=f"{inst_prefix}_dispatch_{did}",
                )
                # Annotate metadata for fixture emission.
                op._mb_network = network        # noqa: SLF001
                op._mb_instance = inst
                op._mb_dispatch_id = did
                op._mb_op_type = op_type
                op._mb_op_name = data["op_name"][did]
                op._mb_job_id = inst_job_id

                # Cross-backend drift constraint: keep requantize ops on
                # CPU_P (gemmini). CPU_E (rvv_opu) is forbidden for them.
                if op_type in requantize_ops:
                    cpu_e_index = machines.index("CPU_E#0")
                    op.infeasible_combinations = set(op.infeasible_combinations) | {cpu_e_index}
                # Per-network whole-chain affinity.
                if network in pin_to_cpu_p:
                    cpu_e_index = machines.index("CPU_E#0")
                    op.infeasible_combinations = set(op.infeasible_combinations) | {cpu_e_index}
                if network in pin_to_cpu_e:
                    cpu_p_index = machines.index("CPU_P#0")
                    op.infeasible_combinations = set(op.infeasible_combinations) | {cpu_p_index}

                op_by_did[did] = op
                all_ops.append(op)
                by_key[op.operation_name] = op

            # Second pass: wire predecessors within this instance.
            for did, pred_dids in data["deps"].items():
                op = op_by_did[did]
                for pdid in pred_dids:
                    op.predecessors.append(op_by_did[pdid])

            instance_meta.append({
                "network": network,
                "instance": inst,
                "job_id": inst_job_id,
                "n_dispatches": len(op_by_did),
            })

    workload = Workload(all_ops, machines, transfer_times)
    return workload, instance_meta, machines


def _emit_fixture(workload, t, alpha, instance_meta, machines, cfg, out_path: pathlib.Path,
                  solve_wall_s: float, report_dict: Optional[dict]) -> dict:
    """Write the ModelBlaster schedule fixture JSON consumed by
    pipeline/ingest_xpurt_schedule.py.
    """
    out_dispatches: Dict[str, dict] = {}
    # Build (dispatch_name -> entry) keyed by the Operation's operation_name.
    machine_idx_to_target = {i: machines[i] for i in range(len(machines))}
    op_to_idx = {id(op): i for i, op in enumerate(workload.operations)}

    n_dispatches = len(workload.operations)
    for i, op in enumerate(workload.operations):
        combo_idx = int(np.argmax(alpha[i]))
        # In our config each machine_combination is a singleton, so
        # combo_idx == machine_idx.
        target = machine_idx_to_target[combo_idx]
        start_ms = float(t[i])
        duration_ms = float(op.processing_times[combo_idx])

        deps = []
        for pred in op.predecessors:
            pi = op_to_idx.get(id(pred))
            if pi is not None:
                deps.append(workload.operations[pi].operation_name)

        net = op._mb_network
        inst = op._mb_instance
        did = op._mb_dispatch_id
        op_type = op._mb_op_type
        backend_suffix = "GEMMINI_op" if target.startswith("CPU_P") else "RVV_OPU_op"
        out_dispatches[op.operation_name] = {
            "id": did,
            "ordinal": 1,
            "total": 1,
            "dependencies": deps,
            "hardware_target": target,
            "start_time": start_ms,
            "duration": duration_ms,
            "job_name": f"{net}{inst}" if inst > 0 or net != "yolov8_nano" else net,
            "module_name": f"{net}$dispatch_{did}_{backend_suffix}",
            "op_kind": op_type,
            "instance": inst,
        }

    git_sha = ""
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        pass

    makespan_ms = max(t + np.array([op.processing_times[int(np.argmax(alpha[i]))]
                                    for i, op in enumerate(workload.operations)]))
    fixture = {
        "dot_file": out_path.name,
        "_provenance": {
            "generated_by": "scripts/run_xpurt_scheduler_multi.py",
            "config": cfg.get("_config_path", "?"),
            "solver": cfg.get("solver", "MOSEK"),
            "solve_wall_s": solve_wall_s,
            "makespan_ms": float(makespan_ms),
            "horizon_target_ms": cfg.get("horizon_ms"),
            "cycles_source": cfg.get("cycles_source", "db"),
            "cycles_agg": cfg.get("cycles_agg", "median"),
            "contention_applied": bool(cfg.get("_contention_path")),
            "contention_path": cfg.get("_contention_path"),
            "git_sha": git_sha,
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "instances": instance_meta,
            "n_dispatches": n_dispatches,
        },
        "dispatches": out_dispatches,
    }
    if report_dict is not None:
        fixture["_provenance"]["scheduler_report"] = report_dict

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixture, indent=2) + "\n")
    return fixture


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, type=pathlib.Path,
                    help="YAML spec (configs/multi_3way_qrb.yaml)")
    ap.add_argument("--output", required=True, type=pathlib.Path,
                    help="output schedule fixture JSON path")
    ap.add_argument("--contention", type=pathlib.Path, default=None,
                    help="optional contention multipliers JSON to scale processing_times")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="override solver time limit (s)")
    ap.add_argument("--dry-run", action="store_true",
                    help="solve but don't write the fixture; print provenance + util")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    cfg["_config_path"] = str(args.config)
    if args.contention:
        contention = json.loads(args.contention.read_text())
        cfg["_contention_path"] = str(args.contention)
    else:
        contention = None

    sys.path.insert(0, str(XPURT_PKG))
    from schedulers import get_scheduler  # type: ignore

    workload, instance_meta, machines = _build_workload(cfg, contention)
    print(f"built Workload: {len(workload.operations)} ops, "
          f"{len(machines)} machines, {len(instance_meta)} network instances")

    time_limit = args.time_limit or float(cfg.get("time_limit_s", 120))
    # `solver` selects the registry entry. CVXPY-backed MILP entries
    # ("mosek", "milp_highs", ...) accept `cvxpy_solver` + `time_limit`
    # in kwargs; list/heuristic entries ("heft", "cpsat", ...) ignore
    # cvxpy_solver. Default is MOSEK for compatibility; HEFT / CPSAT
    # are the fast fallbacks for 100+-op workloads where MOSEK times out.
    solver = cfg.get("solver", "MOSEK")
    # Map the human-friendly "MOSEK" / "HIGHS" / ... names to registry keys.
    SOLVER_TO_REGISTRY = {
        "MOSEK": ("mosek", {"cvxpy_solver": "MOSEK", "time_limit": time_limit}),
        "HIGHS": ("milp_highs", {"cvxpy_solver": "HIGHS", "time_limit": time_limit}),
        "GUROBI": ("milp_gurobi", {"cvxpy_solver": "GUROBI", "time_limit": time_limit}),
        "HEFT":  ("heft",  {}),
        "CPSAT": ("cpsat", {"time_limit_s": time_limit}),
    }
    registry_name, solver_kwargs = SOLVER_TO_REGISTRY.get(
        solver.upper(),
        (solver.lower(), {"time_limit": time_limit}),
    )
    scheduler_fn = get_scheduler(registry_name)
    print(f"invoking scheduler.{registry_name}({solver_kwargs})")

    # CPSAT INTEGER-ROUNDING WARNING: scheduler_cpsat._to_int_us only rounds
    # processing_times to integers (min 1) before feeding CP-SAT. With sub-ms
    # durations (0.5 ms silu, 0.04 ms mlp.linear) this collapses many ops to
    # "1 ms" inside the solver — schedule looks valid to CP-SAT but the
    # back-projected float schedule has overlap / precedence violations of
    # up to ~0.5 ms. Use solver=heft for sub-ms workloads; CPSAT is fine
    # when most ops are ≥ a few ms.
    import time
    t0 = time.time()
    t, alpha, _fused, _fmap = scheduler_fn(workload, **solver_kwargs)
    solve_wall_s = time.time() - t0

    if t is None or alpha is None:
        print("ERROR: solver returned no schedule (infeasible or solver-error)")
        return 2

    # Pull the auto-attached SchedulerReport (xpu-rt populates it on the workload).
    report_obj = getattr(workload, "solver_state", {}).get("report")
    report_dict = None
    if report_obj is not None:
        from dataclasses import asdict
        report_dict = asdict(report_obj)

    makespan = float(max(t + np.array([op.processing_times[int(np.argmax(alpha[i]))]
                                        for i, op in enumerate(workload.operations)])))
    status = getattr(workload, "solver_state", {}).get("problem_status", "solved")
    print(f"solved: makespan={makespan:.3f} ms  solve_wall={solve_wall_s:.2f}s  status={status}")
    target = cfg.get("horizon_ms")
    if target is not None:
        print(f"target_horizon={target} ms -> {'WITHIN' if makespan <= target else 'OVER'} target")
    if report_obj is not None:
        for m, u in report_obj.utilization.items():
            print(f"  utilization {m}: {u['frac_busy']*100:.1f}% busy")
        g = report_obj.granularity
        print(f"  dispatch_durations: p50={g['p50']:.3f}ms p90={g['p90']:.3f}ms p99={g['p99']:.3f}ms")
        print(f"  buckets: {g['buckets']}")

    if args.dry_run:
        print("dry-run: not writing fixture")
        return 0

    _emit_fixture(workload, t, alpha, instance_meta, machines, cfg, args.output,
                  solve_wall_s, report_dict)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
