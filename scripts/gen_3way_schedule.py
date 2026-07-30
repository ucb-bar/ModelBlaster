"""Generate a combined 3-network heterogeneous schedule fixture.

Mirrors the prior qrb5165 robotics-style setup:
  - **yolov8n** (1 instance): main perception, full op chain across both tiles
  - **dronet** (1 instance, can be replicated): steering/collision policy
  - **mlp_control** (N instances at 5ms intervals): high-rate control loop

Routing policy (same constraints as our single-network MOSEK fixture —
see notes/known_issues.md on cross-backend numerical drift):
  - Requantization-producing ops (conv2d_s8, linear_s8) → gemmini (CPU_P#0)
  - Structural / passthrough ops → opu (CPU_E#0)
  - mlp_control is fp32; no quantization-drift constraint, but its tight
    linear+elu chain runs much faster on opu's RVV than on gemmini's
    scalar fallback, so we keep it on CPU_E#0 throughout.

Scheduling is a simple topological-order layout (not MOSEK) for speed and
predictability. Each op's start_time is the max of its dependency end-times,
respecting tile capacity (one op per tile at a time). Periodic instances
of mlp_control are pinned to their period boundaries via start_time floor.

Usage:
    PYTHONPATH=. uv run python -m scripts.gen_3way_schedule \\
        --output schedule_fixtures/3way_yolov8_dronet_mlp.json \\
        --mlp-instances 4 --mlp-period-ms 5
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, Optional, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_ir(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def _build_dep_graph(ir: dict) -> Tuple[Dict[int, list[int]], Dict[int, str]]:
    """Returns (deps_by_did, op_kind_by_did). Skip-through view ops."""
    ops = ir["ops"]
    tensor_producer = {}
    for i, op in enumerate(ops):
        for t in op.get("outputs", []):
            tensor_producer[t] = i

    def resolve(tensor_name):
        seen = set()
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

    deps_by_did: Dict[int, list[int]] = {}
    kind_by_did: Dict[int, str] = {}
    for op in ops:
        did = op.get("dispatch_id")
        if did is None:
            continue
        kind_by_did[did] = op["op"]
        deps = []
        for in_t in op.get("inputs", []):
            d = resolve(in_t)
            if d is not None and d != did:
                deps.append(d)
        deps_by_did[did] = sorted(set(deps))
    return deps_by_did, kind_by_did


def _topo_layout(
    deps_by_did: Dict[int, list[int]],
    kind_by_did: Dict[int, str],
    cycles_per_op: Dict[int, float],
    start_offset_ms: float,
    job_prefix: str,
    requantize_ops: set[str],
) -> Dict[str, dict]:
    """Produce a tile-aware topological layout. Each op gets:
      - hardware_target (CPU_P#0 for requantize, CPU_E#0 else)
      - start_time = max(dep end times) + cross-tile transfer (0 for now)
      - duration   = cycles_per_op[did]
    Honors single-op-per-tile-at-a-time (sequential within a tile).
    """
    # Topological order via Kahn
    in_deg = {did: 0 for did in deps_by_did}
    rev = {did: [] for did in deps_by_did}
    for did, deps in deps_by_did.items():
        for d in deps:
            in_deg[did] += 1
            rev[d].append(did)
    ready = [d for d, n in in_deg.items() if n == 0]
    order: list[int] = []
    while ready:
        ready.sort()
        d = ready.pop(0)
        order.append(d)
        for s in rev[d]:
            in_deg[s] -= 1
            if in_deg[s] == 0:
                ready.append(s)

    if len(order) != len(deps_by_did):
        raise RuntimeError(
            f"topological order incomplete: {len(order)}/{len(deps_by_did)} "
            f"(cycle in deps?)")

    # Layout: per-tile current-end time; each op runs after its deps AND
    # after the tile's previous op finishes.
    tile_end = {"CPU_P#0": start_offset_ms, "CPU_E#0": start_offset_ms}
    end_time: Dict[int, float] = {}
    out: Dict[str, dict] = {}
    for did in order:
        op_kind = kind_by_did[did]
        target = "CPU_P#0" if op_kind in requantize_ops else "CPU_E#0"
        dep_end = max((end_time[d] for d in deps_by_did[did]),
                      default=start_offset_ms)
        start = max(dep_end, tile_end[target])
        dur = cycles_per_op.get(did, 1000.0) / 1_000_000.0  # cycles → ms @ 1GHz
        end = start + dur
        tile_end[target] = end
        end_time[did] = end
        key = f"{job_prefix}_dispatch_{did}"
        out[key] = {
            "id": did,
            "ordinal": 1,
            "total": 1,
            "dependencies": [f"{job_prefix}_dispatch_{d}"
                              for d in deps_by_did[did]],
            "hardware_target": target,
            "start_time": start,
            "duration": dur,
            "job_name": job_prefix,
            "module_name": f"{job_prefix}$dispatch_{did}_{op_kind}",
        }
    return out


def _load_cycles_from_profile(csv_path: pathlib.Path) -> Dict[int, float]:
    """Read profile CSV (cycles per dispatch_id)."""
    import csv
    out: Dict[int, float] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            out[int(row["dispatch_id"])] = float(row["cycles"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True,
        help="output schedule fixture path")
    ap.add_argument("--mlp-instances", type=int, default=4,
        help="number of periodic mlp_control instances (default 4)")
    ap.add_argument("--mlp-period-ms", type=float, default=5.0,
        help="mlp_control inter-arrival period in ms (default 5)")
    ap.add_argument("--dronet-instances", type=int, default=1,
        help="number of dronet instances (default 1)")
    ap.add_argument("--dronet-period-ms", type=float, default=0.0,
        help="dronet inter-arrival period in ms (default 0 = all at t=0)")
    ap.add_argument("--yolov8n-instances", type=int, default=1,
        help="number of yolov8n instances (default 1; rarely > 1)")
    ap.add_argument("--yolov8n-period-ms", type=float, default=0.0,
        help="yolov8n inter-arrival period in ms (default 0)")
    args = ap.parse_args()

    requantize = {"conv2d_s8", "linear_s8", "linear", "linear_fp32"}

    # Load IRs.
    yolov8_ir = _load_ir(REPO_ROOT / "examples/yolov8_nano/int8/generated/graph.json")
    dronet_ir = _load_ir(REPO_ROOT / "examples/dronet/int8/generated/graph.json")
    # mlp_control: int8 path is now unblocked (elu_s8 reference kernel).
    # Fall back to fp32 only if int8 hasn't been extracted yet.
    _mlp_int8 = REPO_ROOT / "examples/mlp_control/int8/generated/graph.json"
    _mlp_fp32 = REPO_ROOT / "examples/mlp_control/fp32/generated/graph.json"
    mlp_ir    = _load_ir(_mlp_int8 if _mlp_int8.exists() else _mlp_fp32)

    # Per-op cycles from standalone profile runs (firesim per_dispatch CSV).
    # Use gemmini variant — it's the fastest path for cycle estimation;
    # the actual runtime cycles will be whatever the binary measures.
    yolov8_cyc = _load_cycles_from_profile(
        REPO_ROOT / "benchmarks/results/A/yolov8n_gemmini_int8/latest/profile_firesim.csv")
    dronet_cyc = _load_cycles_from_profile(
        REPO_ROOT / "benchmarks/results/A/dronet_gemmini_int8/latest/profile_firesim.csv")
    # mlp_control hasn't been benchmarked standalone yet; use placeholder
    # cycles (50k per linear, 5k per elu) — purely for start_time
    # ordering; the actual run uses real cycles.
    mlp_cyc = {0: 36000, 1: 25000, 2: 256000, 3: 16000,
               4: 64000, 5: 7000, 6: 2000}

    print(f"yolov8n: 212 ops x {args.yolov8n_instances} instances "
          f"(period={args.yolov8n_period_ms}ms)")
    print(f"dronet:   30 ops x {args.dronet_instances} instances "
          f"(period={args.dronet_period_ms}ms)")
    print(f"mlp_control: 7 ops x {args.mlp_instances} instances "
          f"(period={args.mlp_period_ms}ms)")

    # Build per-network dep graphs.
    yolo_deps, yolo_kind = _build_dep_graph(yolov8_ir)
    dronet_deps, dronet_kind = _build_dep_graph(dronet_ir)
    mlp_deps, mlp_kind = _build_dep_graph(mlp_ir)

    # Layout each network.
    all_dispatches: Dict[str, dict] = {}

    # yolov8n: 1+ instances. Default is 1 at t=0 (single perception cycle).
    for i in range(args.yolov8n_instances):
        offset = i * args.yolov8n_period_ms
        suffix = "" if args.yolov8n_instances == 1 else str(i)
        all_dispatches.update(_topo_layout(
            yolo_deps, yolo_kind, yolov8_cyc,
            start_offset_ms=offset, job_prefix=f"yolov8_nano{suffix}",
            requantize_ops=requantize))

    # dronet: 1+ instances at dronet-period-ms intervals.
    for i in range(args.dronet_instances):
        offset = i * args.dronet_period_ms
        suffix = "" if args.dronet_instances == 1 else str(i)
        all_dispatches.update(_topo_layout(
            dronet_deps, dronet_kind, dronet_cyc,
            start_offset_ms=offset, job_prefix=f"dronet{suffix}",
            requantize_ops=requantize))

    # mlp_control: N instances, each pinned at i * period_ms.
    # We pass start_offset to push the WHOLE chain to that time;
    # tile_end seeds at start_offset so each instance fits in its
    # period without colliding with other mlp instances.
    for i in range(args.mlp_instances):
        offset = i * args.mlp_period_ms
        all_dispatches.update(_topo_layout(
            mlp_deps, mlp_kind, mlp_cyc,
            start_offset_ms=offset, job_prefix=f"mlp_control{i}",
            requantize_ops=requantize))

    # Summary.
    from collections import Counter
    tgt_dist = Counter(e["hardware_target"] for e in all_dispatches.values())
    job_dist = Counter(e["job_name"] for e in all_dispatches.values())
    horizon = max(e["start_time"] + e["duration"]
                  for e in all_dispatches.values())
    print(f"total dispatches: {len(all_dispatches)}")
    print(f"tile distribution: {dict(tgt_dist)}")
    print(f"jobs:              {dict(job_dist)}")
    print(f"time horizon: {horizon:.2f}ms")

    out = {
        "dot_file": "3way_yolov8_dronet_mlp.json",
        "_provenance": {
            "generated_by": "scripts/gen_3way_schedule.py",
            "policy": "topo-by-tile + all-convs/linears-on-gemmini",
            "notes": [
                "Three-network schedule: yolov8_nano (1, main perception) + "
                "dronet (1, steering policy) + mlp_control (N periodic at "
                "5ms intervals, lightweight control loop). Mirrors the "
                "qrb5165 robotics-style setup from "
                "/scratch2/dima/misc_sw/FreshScheduler/schedules/"
                "scheduled_networks_3way_dronet5ms_mlp2ms_yolov8_qrb5165_"
                "greedy_profiled.json. Mapped from qrb5165's 3-tile (CPU_P, "
                "CPU_E, CPU_X) to our 2-tile GemminiAndOPUShuttleConfig "
                "(gemmini=CPU_P, opu=CPU_E). Hand-laid topological order "
                "rather than MOSEK because of multi-job dep graphs needing "
                "periodicity constraints — MOSEK with proper periodicity "
                "for this scale is the next iteration.",
            ],
        },
        "dispatches": all_dispatches,
    }

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
