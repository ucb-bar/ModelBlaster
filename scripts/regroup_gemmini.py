"""Re-sort the gemmini dispatches in a schedule fixture so each network
instance's gemmini ops execute consecutively (no cross-network switches).

The contention model (benchmarks/contention_model.py) shows that gemmini
conv2d_s8 ops blow up 80-100× when consecutive dispatches come from
different networks — the scratchpad/weight cache gets thrashed. This
post-processor reorders the schedule so each network instance's gemmini
ops execute as a contiguous burst.

Semantics:
  - Group dispatches by hardware_target.
  - For CPU_P (gemmini) ops: order entries by (job_name, original_start_time).
    All of network A's gemmini ops finish before B's start.
  - For CPU_E (rvv_opu) ops: keep relative order, but their start_times
    are recomputed to respect deps on the new gemmini schedule.
  - Re-derive start_times: each op's new start_time = max(dep_finish_times,
    same_tile_previous_finish).
  - Preserve all other fixture fields (durations, dependencies, etc.).

Usage:
    python scripts/regroup_gemmini.py \\
        --in  schedule_fixtures/3way_heft_dronet2_mlp4.json \\
        --out schedule_fixtures/3way_heft_dronet2_mlp4_regrouped.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from typing import Dict


def regroup(fixture_path: pathlib.Path, out_path: pathlib.Path) -> dict:
    fx = json.loads(fixture_path.read_text())
    dispatches: Dict[str, dict] = fx["dispatches"]

    # Strategy: only constrain gemmini side. Pick an instance order on
    # CPU_P (gemmini); for each consecutive instance pair, add a virtual
    # dep: instance B's first gemmini op must wait for instance A's last
    # gemmini op. RVV/OPU tile stays unconstrained — its ordering is
    # determined by deps + earliest-start-on-tile naturally.

    # 1. Identify gemmini ops grouped by job_name.
    gem_by_job: Dict[str, list] = defaultdict(list)
    for name, d in dispatches.items():
        if d["hardware_target"].startswith("CPU_P"):
            gem_by_job[d["job_name"]].append(name)
    # Sort within each job by original start_time (preserves intra-network dep order).
    for j in gem_by_job:
        gem_by_job[j].sort(key=lambda n: (dispatches[n]["start_time"], dispatches[n].get("id", 0)))

    # 2. Pick instance order — alphabetic is deterministic; could be by
    #    total gemmini work descending for SRPT-like priority but
    #    alphabetic is simpler to reason about.
    job_order = sorted(gem_by_job.keys())

    # 3. Build augmented dep set with cross-instance virtual deps.
    aug_deps: Dict[str, list] = {k: list(d["dependencies"]) for k, d in dispatches.items()}
    last_gemmini_per_job: Dict[str, str] = {}
    for j in job_order:
        ordered = gem_by_job[j]
        if not ordered:
            continue
        first_in_j = ordered[0]
        last_in_j = ordered[-1]
        # virtual dep: first gemmini op of j depends on last gemmini op of previous job
        for prev_j in job_order:
            if prev_j == j:
                break
            if prev_j in last_gemmini_per_job:
                prev_last = last_gemmini_per_job[prev_j]
                if prev_last not in aug_deps[first_in_j]:
                    aug_deps[first_in_j].append(prev_last)
        last_gemmini_per_job[j] = last_in_j
        # Internal: each gemmini op (after the first) depends on the previous in the same job.
        # The original fixture already encodes intra-network deps via the data flow,
        # but to be safe, also chain them.
        for i in range(1, len(ordered)):
            prev = ordered[i-1]
            if prev not in aug_deps[ordered[i]]:
                aug_deps[ordered[i]].append(prev)

    # 4. Topological layout using augmented deps.
    # Each op's new start = max(aug_dep_finish, same_tile_prev_finish).
    in_deg = {k: len(aug_deps[k]) for k in aug_deps}
    rev = defaultdict(list)
    for k, deps in aug_deps.items():
        for d in deps:
            rev[d].append(k)
    ready = [k for k, n in in_deg.items() if n == 0]
    order: list = []
    while ready:
        # Pick ready op with earliest original_start_time to mimic input ordering.
        ready.sort(key=lambda n: dispatches[n]["start_time"])
        n = ready.pop(0)
        order.append(n)
        for succ in rev[n]:
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                ready.append(succ)
    if len(order) != len(dispatches):
        raise RuntimeError(
            f"topological order incomplete: {len(order)}/{len(dispatches)} "
            f"(cycle in augmented deps?)"
        )

    end_times: Dict[str, float] = {}
    new_start: Dict[str, float] = {}
    tile_last_end: Dict[str, float] = defaultdict(float)
    for name in order:
        deps_end = max((end_times[d] for d in aug_deps[name]), default=0.0)
        tile = dispatches[name]["hardware_target"]
        start = max(deps_end, tile_last_end[tile])
        duration = dispatches[name]["duration"]
        new_start[name] = start
        end_times[name] = start + duration
        tile_last_end[tile] = end_times[name]

    # Build the new fixture.
    new_dispatches: Dict[str, dict] = {}
    for name, d in dispatches.items():
        new_d = dict(d)
        new_d["start_time"] = new_start[name]
        new_dispatches[name] = new_d

    makespan = max(end_times.values()) if end_times else 0.0
    new_prov = dict(fx.get("_provenance", {}))
    new_prov["regrouped_by"] = "scripts/regroup_gemmini.py"
    new_prov["regrouped_source"] = str(fixture_path)
    new_prov["original_makespan_ms"] = new_prov.get("makespan_ms", None)
    new_prov["makespan_ms"] = makespan

    out_fx = dict(fx)
    out_fx["_provenance"] = new_prov
    out_fx["dispatches"] = new_dispatches
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_fx, indent=2) + "\n")

    # Validate
    violations = 0
    for k, v in new_dispatches.items():
        for dep in v["dependencies"]:
            if v["start_time"] + 1e-6 < new_dispatches[dep]["start_time"] + new_dispatches[dep]["duration"]:
                violations += 1
    per_tile = defaultdict(list)
    for k, v in new_dispatches.items():
        per_tile[v["hardware_target"]].append((v["start_time"], v["start_time"] + v["duration"]))
    overlaps = 0
    for tile, items in per_tile.items():
        items.sort()
        for i in range(1, len(items)):
            if items[i][0] + 1e-6 < items[i-1][1]:
                overlaps += 1
    return {
        "n_dispatches": len(new_dispatches),
        "original_makespan_ms": new_prov.get("original_makespan_ms"),
        "regrouped_makespan_ms": makespan,
        "dep_violations": violations,
        "tile_overlaps": overlaps,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="inp", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()
    info = regroup(args.inp, args.out)
    print(f"wrote {args.out}")
    for k, v in info.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
