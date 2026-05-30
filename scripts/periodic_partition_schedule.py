"""Periodic-partition scheduler — solves each instance separately
within its time slice and merges.

Why this exists:
  - The MILP (MOSEK) formulation respects both min_start_t and
    max_end_t but doesn't converge on 300-op qrb_y64.
  - HEFT respects min_start_t but ignores max_end_t, so instances
    bleed past their periodic deadlines.

This partition scheme:
  1. For network N with K instances and horizon H, give each instance
     a slot of length L = H/K starting at s_i = i*L.
  2. Within each slot, solve the single-instance subgraph with HEFT
     (it can't violate max_end since it's bounded by the slot's
     compute capacity).
  3. If the single-instance compute exceeds L, the instance overruns —
     emit a deadline-missed marker but include it in the fixture so
     the Gantt shows the violation clearly.
  4. Merge per-instance schedules into a single fixture by translating
     each into its slot offset.

The result is a fixture where the i-th instance lives entirely in
[i*L, (i+1)*L] when feasible, and the Gantt visually shows true
periodic frequency.

Usage:
    python scripts/periodic_partition_schedule.py \\
        --config configs/multi_3way_qrb_y64_periodic.yaml \\
        --output schedule_fixtures/3way_partitioned_qrb_y64.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import sys
import time
from collections import defaultdict
from typing import Any

REPO = pathlib.Path(__file__).resolve().parent.parent
XPURT = pathlib.Path("/scratch2/agustin/XPU-RT/xpu-rt")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(XPURT))


def _build_dep_graph(ir: dict):
    """Same dep-graph derivation as scripts/run_xpurt_scheduler_multi.py.
    graph.json has 'ops' (every op with dispatch_id pointing to the
    parent dispatch) and 'dispatches' (just an id list)."""
    ops = ir["ops"]
    tensor_producer: dict = {}
    for i, op in enumerate(ops):
        for t in op.get("outputs", []):
            tensor_producer[t] = i

    def resolve(tname):
        seen = set()
        while True:
            if tname in seen:
                return None
            seen.add(tname)
            if tname not in tensor_producer:
                return None
            op = ops[tensor_producer[tname]]
            did = op.get("dispatch_id")
            if did is not None:
                return did
            ins = op.get("inputs", [])
            if not ins:
                return None
            tname = ins[0]

    deps, kind, op_name = {}, {}, {}
    for op in ops:
        did = op.get("dispatch_id")
        if did is None:
            continue
        kind[did] = op["op"]
        op_name[did] = op["name"]
        deps[did] = sorted({d for d in (resolve(t) for t in op.get("inputs", []))
                            if d is not None and d != did})
    return deps, kind, op_name


def _solve_single_instance(network: str, quant: str, machines: list[str],
                           requantize_ops: set[str], pin_p: set[str], pin_e: set[str],
                           transfer_times, cycles_per_ms: float) -> dict:
    """HEFT-solve one instance of `network` and return a dict of
    dispatch_id -> {machine, start_offset, duration}."""
    import numpy as np
    from workload import Operation, Workload
    from benchmarks.profile_db import query as profile_db_query
    from schedulers import get_scheduler

    ir_path = REPO / "examples" / network / quant / "generated" / "graph.json"
    ir = json.loads(ir_path.read_text())
    deps, op_kind, op_name = _build_dep_graph(ir)
    cyc_g = profile_db_query(network, "gemmini", quant, agg="median")
    cyc_o = profile_db_query(network, "rvv_opu", quant, agg="median")

    ops = []
    by_did: dict[int, Operation] = {}
    for did in sorted(deps):
        op_type = op_kind[did]
        g_ms = max(0.001, cyc_g.get(did, 0) / cycles_per_ms)
        o_ms = max(0.001, cyc_o.get(did, 0) / cycles_per_ms)
        op = Operation(processing_times=[g_ms, o_ms],
                       operation_name=f"{network}_dispatch_{did}")
        if op_type in requantize_ops or network in pin_p:
            op.infeasible_combinations = set(op.infeasible_combinations) | {machines.index("CPU_E#0")}
        if network in pin_e:
            op.infeasible_combinations = set(op.infeasible_combinations) | {machines.index("CPU_P#0")}
        by_did[did] = op
        ops.append(op)
        op._did = did
        op._op_type = op_type
        op._op_name = op_name[did]
    # Wire predecessors.
    for did, pred_dids in deps.items():
        for p in pred_dids:
            by_did[did].predecessors.append(by_did[p])

    workload = Workload(ops, machines, transfer_times)
    heft_fn = get_scheduler("heft")
    t, alpha, _, _ = heft_fn(workload)

    out: dict[int, dict] = {}
    if t is None:
        return out
    for i, op in enumerate(workload.operations):
        m_idx = int(np.argmax(alpha[i]))
        duration = op.processing_times[m_idx]
        out[op._did] = {
            "dispatch_id": op._did,
            "op_type": op._op_type,
            "op_name": op._op_name,
            "machine_idx": m_idx,
            "machine": machines[m_idx],
            "start_offset": float(t[i]),
            "duration": float(duration),
            "deps": list(deps[op._did]),
        }
    return out


def main() -> int:
    import numpy as np
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=pathlib.Path)
    ap.add_argument("--output", required=True, type=pathlib.Path)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    machines = ["CPU_P#0", "CPU_E#0"]
    transfer_times = np.array([[0.0, 0.01], [0.01, 0.0]], dtype=float)
    requantize_ops = set(cfg.get("requantize_ops", []))
    pin_p = set(cfg.get("pin_to_cpu_p", []))
    pin_e = set(cfg.get("pin_to_cpu_e", []))
    horizon = float(cfg["horizon_ms"])
    cycles_per_ms = float(cfg.get("cycles_per_ms", 1_000_000))

    # Schedule one instance per network — same shape across instances.
    template: dict[str, dict] = {}
    for net_cfg in cfg["networks"]:
        net = net_cfg["name"]
        template[net] = _solve_single_instance(
            net, net_cfg["quant"], machines,
            requantize_ops, pin_p, pin_e, transfer_times, cycles_per_ms,
        )
        # Per-instance makespan = max(start + duration).
        per_inst_makespan = max((d["start_offset"] + d["duration"]
                                 for d in template[net].values()), default=0.0)
        print(f"  {net}: per-instance makespan {per_inst_makespan:.2f} ms")

    # Build a candidate per-instance schedule, then SERIALIZE on each tile
    # via ASAP packing while honoring per-instance min_start_t (slot start)
    # and intra-instance precedence. This eliminates tile-overlap because
    # each op only starts after the latest of:
    #   (1) all its dependencies finish,
    #   (2) the tile it picks finishes its previous op,
    #   (3) its instance's slot start.
    candidates: list[dict] = []
    instance_meta = []
    deadline_misses = 0
    job_id_counter = 0
    for net_cfg in cfg["networks"]:
        net = net_cfg["name"]
        n_inst = int(net_cfg.get("instances", 1))
        slot_len = horizon / max(n_inst, 1)
        for inst in range(n_inst):
            slot_start = inst * slot_len
            slot_end = (inst + 1) * slot_len
            for did, t in template[net].items():
                key = f"{net}#{inst}_dispatch_{did}"
                candidates.append({
                    "key": key,
                    "network": net,
                    "instance": inst,
                    "did": did,
                    "op_name": t["op_name"],
                    "op_type": t["op_type"],
                    "machine": t["machine"],
                    "duration": t["duration"],
                    "deps": [f"{net}#{inst}_dispatch_{d}" for d in t["deps"]],
                    "tentative_start_offset": t["start_offset"],
                    "slot_start": slot_start,
                    "slot_end": slot_end,
                })
            instance_meta.append({
                "network": net,
                "instance": inst,
                "job_id": job_id_counter,
                "n_dispatches": len(template[net]),
            })
            job_id_counter += 1

    # Topological + ASAP pass. Order candidates by (slot_start,
    # tentative_start_offset) so that earlier slots fire first; within a
    # slot, intra-instance order matches the HEFT-template offsets.
    candidates.sort(key=lambda c: (c["slot_start"], c["tentative_start_offset"]))

    end_times: dict[str, float] = {}
    tile_free_at: dict[str, float] = defaultdict(float)
    dispatches: dict[str, dict] = {}

    # Walk in topo + slot order. For each op, its start time =
    #   max(slot_start, max(end_times[dep] for dep in deps), tile_free_at[tile])
    # then update end_times and tile_free_at.
    # Robust handling of cross-dep ordering: if a dep isn't placed yet,
    # we put the candidate back at the end of the queue and re-process.
    queue = list(candidates)
    iter_count = 0
    while queue:
        iter_count += 1
        if iter_count > len(candidates) * 5:
            print(f"  ERROR: ASAP pass not converging after {iter_count} iters")
            break
        c = queue.pop(0)
        # All deps must already be placed.
        if any(d not in end_times for d in c["deps"]):
            queue.append(c)
            continue
        slot_start = c["slot_start"]
        dep_end = max((end_times[d] for d in c["deps"]), default=0.0)
        tile = c["machine"]
        tile_ready = tile_free_at[tile]
        start = max(slot_start, dep_end, tile_ready)
        end = start + c["duration"]
        end_times[c["key"]] = end
        tile_free_at[tile] = end
        if end > c["slot_end"] + 1e-6:
            # deadline miss for this dispatch — instance may bleed past slot
            pass
        dispatches[c["key"]] = {
            "id": c["did"],
            "job_name": f"{c['network']}{c['instance']}",
            "module_name": c["op_name"],
            "hardware_target": tile,
            "start_time": start,
            "duration": c["duration"],
            "op": c["op_type"],
            "dependencies": c["deps"],
        }

    # Recompute deadline-miss count per instance.
    for ins in instance_meta:
        keys = [k for k in dispatches if k.startswith(f"{ins['network']}#{ins['instance']}_")]
        if not keys:
            continue
        inst_end = max(dispatches[k]["start_time"] + dispatches[k]["duration"] for k in keys)
        n_inst_net = sum(1 for j in instance_meta if j["network"] == ins["network"])
        slot_len = horizon / max(n_inst_net, 1)
        slot_end = (ins["instance"] + 1) * slot_len
        if inst_end > slot_end + 1e-6:
            deadline_misses += 1
            print(f"  DEADLINE MISS: {ins['network']} instance {ins['instance']} "
                  f"ended at {inst_end:.2f} ms, slot ends at {slot_end:.2f} ms")

    # The job_name format above ("dronet0") collides with our existing
    # _split_job_name. Keep it because the ingest path handles it via
    # provenance.instances.
    makespan = max(d["start_time"] + d["duration"] for d in dispatches.values())

    out = {
        "_provenance": {
            "generated_by": "scripts/periodic_partition_schedule.py",
            "config": str(args.config),
            "solver": "periodic_partition_heft",
            "scheduler_strategy": "per-instance HEFT within fixed time slot",
            "makespan_ms": makespan,
            "horizon_target_ms": horizon,
            "n_dispatches": len(dispatches),
            "deadline_misses": deadline_misses,
            "instances": instance_meta,
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        },
        "dispatches": dispatches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {args.output}")
    print(f"  total dispatches: {len(dispatches)}")
    print(f"  makespan: {makespan:.2f} ms (target horizon {horizon} ms)")
    print(f"  deadline misses: {deadline_misses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
