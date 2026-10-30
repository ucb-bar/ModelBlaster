#!/usr/bin/env python3
"""Hand-author a 2-tile XPU-RT schedule from a graph.json IR.

The real XPU-RT schedule comes out of FreshScheduler against profiled
per-op cycles. That requires running each model on each backend
end-to-end first (chicken-and-egg with the harness we're trying to
exercise). This generator produces a *valid* schedule (DAG matches the
IR, hardware_target labels resolve, start_times are monotonic in
topological order) so the hetero dispatch path can be smoked. Replace
each fixture with a real FreshScheduler-emitted one once the
profiling loop is live.

The policy assigns ops to one of two tiles:

  ``gemmini_main_opu_skip``
      "Main path" conv2d_s8 / linear_s8 ops -> CPU_P  (gemmini tile)
      Residual skip-path conv2d_s8 ops      -> CPU_E  (opu tile)
      Elementwise / pool / norm / activation -> CPU_E  (opu tile;
      gemmini cannot handle those, so the rvv_baseline on the OPU
      tile picks them up).

CPU_P / CPU_E are the abstract slot labels XPU-RT uses; the cores
registry resolves CPU_P -> gemmini, CPU_E -> rvv_opu via the
--cpu-p-kind / --cpu-e-kind env vars the harness sets.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any


# Ops whose curated kernels can use the Gemmini RoCC tiled_matmul; the
# scheduler is free to route any of these to the gemmini tile. Anything
# else falls back to the OPU tile's RVV/scalar baseline.
_GEMMINI_CAPABLE = frozenset({"conv2d_s8", "linear_s8", "matmul_s8"})

# Per-op estimated cycles on each tile. Hand-tuned placeholders --
# they exist so makespan ordering is plausible but they are NOT
# measured; real numbers come from FreshScheduler's profiling step.
_PLACEHOLDER_CYCLES = {
    ("gemmini", "conv2d_s8"):     20_000,
    ("gemmini", "linear_s8"):      8_000,
    ("gemmini", "matmul_s8"):      8_000,
    ("rvv_opu", "conv2d_s8"):     35_000,
    ("rvv_opu", "linear_s8"):     15_000,
    ("rvv_opu", "matmul_s8"):     15_000,
    ("rvv_opu", "batchnorm2d_s8"): 4_000,
    ("rvv_opu", "relu_s8"):        2_000,
    ("rvv_opu", "maxpool2d_s8"):   3_000,
    ("rvv_opu", "add_s8"):         2_500,
    ("rvv_opu", "sigmoid_s8"):     5_000,
}
_DEFAULT_CYCLES = 5_000
_CLOCK_HZ = 1_000_000_000.0  # 1 GHz nominal -- duration in ms


def _placeholder_duration_ms(kind: str, op: str) -> float:
    cycles = _PLACEHOLDER_CYCLES.get((kind, op), _DEFAULT_CYCLES)
    return cycles / _CLOCK_HZ * 1000.0


def _gemmini_main_opu_skip(ops: list[dict[str, Any]]
                           ) -> dict[int, str]:
    """Place ops on tiles per the gemmini_main_opu_skip policy.

    Returns a dict from dispatch_id to abstract slot label
    (CPU_P / CPU_E). Ops without a dispatch_id (zero-cost views,
    reshapes) are not placed -- the ingest layer marks them as
    -1 and the runtime emits the completion semaphore without
    invoking a kernel.
    """
    placement: dict[int, str] = {}

    # First, find "skip path" conv ops -- those whose dispatch_id is the
    # second input of a downstream add_s8. dronet's residual blocks
    # follow this shape. Generalizes well to other ResNet-style nets;
    # for purely sequential graphs, the skip set is empty and every
    # gemmini-capable op goes to CPU_P.
    by_id = {op["dispatch_id"]: op for op in ops
             if op.get("dispatch_id") is not None}
    skip_ids: set[int] = set()
    for op in ops:
        if op["op"] != "add_s8":
            continue
        deps = op.get("depends_on", [])
        if len(deps) < 2:
            continue
        # Second dependency = skip-path producer.
        skip_ids.add(int(deps[1]))

    for op in ops:
        d = op.get("dispatch_id")
        if d is None:
            continue
        op_kind = op["op"]
        if op_kind in _GEMMINI_CAPABLE:
            placement[d] = "CPU_E#0" if d in skip_ids else "CPU_P#0"
        else:
            # Elementwise / pool / norm / activation -- OPU's RVV
            # baseline picks these up. Gemmini RoCC cannot.
            placement[d] = "CPU_E#0"
    return placement


def _topo_order(ops: list[dict[str, Any]]) -> list[int]:
    """Kahn's algorithm over the dispatch_id graph. Zero-cost ops
    are skipped; their dependents reference the underlying compute
    dispatch_id."""
    by_id = {op["dispatch_id"]: op for op in ops
             if op.get("dispatch_id") is not None}
    indeg: dict[int, int] = {d: 0 for d in by_id}
    for d, op in by_id.items():
        for dep in op.get("depends_on", []) or []:
            if dep in by_id:
                indeg[d] += 1

    q = deque(sorted(d for d, n in indeg.items() if n == 0))
    out: list[int] = []
    while q:
        d = q.popleft()
        out.append(d)
        for d2, op in by_id.items():
            for dep in op.get("depends_on", []) or []:
                if dep == d:
                    indeg[d2] -= 1
                    if indeg[d2] == 0:
                        q.append(d2)
    if len(out) != len(by_id):
        raise SystemExit(
            f"topological sort short-circuited: {len(out)}/{len(by_id)} "
            f"ops. The IR has a cycle or my walker has a bug."
        )
    return out


def gen_schedule(graph_path: Path, *, job_name: str,
                 module_name: str, policy: str) -> dict[str, Any]:
    g = json.loads(graph_path.read_text())
    ops = g["ops"]
    by_id = {op["dispatch_id"]: op for op in ops
             if op.get("dispatch_id") is not None}

    if policy == "gemmini_main_opu_skip":
        placement = _gemmini_main_opu_skip(ops)
    else:
        raise SystemExit(f"unknown policy: {policy}")

    order = _topo_order(ops)

    dispatches: dict[str, Any] = {}
    cursor_ms = 0.0
    for d in order:
        op = by_id[d]
        slot = placement[d]
        # Slot -> kind (for placeholder duration lookup only). The
        # real (kind, hart) resolution happens in ingest via the
        # cores registry + cpu_*_kind env vars.
        kind = {"CPU_P#0": "gemmini", "CPU_E#0": "rvv_opu"}.get(slot, "rvv_opu")
        dur_ms = _placeholder_duration_ms(kind, op["op"])
        entry = {
            "id": int(d),
            "ordinal": 1,
            "total": 1,
            "dependencies": [
                f"{job_name}_dispatch_{int(dep)}"
                for dep in (op.get("depends_on") or [])
                if dep in by_id
            ],
            "hardware_target": slot,
            "start_time": round(cursor_ms, 6),
            "duration": round(dur_ms, 6),
            "job_name": job_name,
            "module_name": f"{module_name}$dispatch_{int(d)}_{kind.upper()}_{op['op']}",
        }
        # Optional time_dependency for cross-job ordering -- not used
        # in single-network fixtures.
        dispatches[f"{job_name}_dispatch_{int(d)}"] = entry
        cursor_ms += dur_ms

    return {
        "dot_file": f"{job_name}_hetero_gemmini_opu_handauthored.json",
        "_provenance": {
            "generated_by": "scripts/gen_hetero_schedule.py",
            "policy": policy,
            "source_ir": str(graph_path),
            "notes": [
                "Hand-authored placeholder schedule. Timing values are "
                "synthetic; only the DAG, hardware_target labels, and "
                "topological start_time ordering are load-bearing.",
                "Replace with a FreshScheduler-emitted schedule against "
                "profiled per-op cycles once the harness produces "
                "results.csv on the target backends.",
            ],
        },
        "dispatches": dispatches,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ir", required=True,
                    help="path to graph.json produced by extract_graph")
    ap.add_argument("--out", required=True,
                    help="path to write the schedule.json fixture")
    ap.add_argument("--job-name", required=True,
                    help="canonical job name (matches MODELS env var, e.g. 'dronet')")
    ap.add_argument("--module-name", default=None,
                    help="schedule module_name prefix (default: <job-name>)")
    ap.add_argument("--policy", default="gemmini_main_opu_skip",
                    choices=["gemmini_main_opu_skip"])
    args = ap.parse_args()

    sched = gen_schedule(
        Path(args.ir),
        job_name=args.job_name,
        module_name=args.module_name or args.job_name,
        policy=args.policy,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sched, indent=2) + "\n")

    n = len(sched["dispatches"])
    by_slot: dict[str, int] = {}
    for d in sched["dispatches"].values():
        by_slot[d["hardware_target"]] = by_slot.get(d["hardware_target"], 0) + 1
    print(f"wrote {out}  ({n} dispatches: " +
          ", ".join(f"{k}={v}" for k, v in sorted(by_slot.items())) + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
