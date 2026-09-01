"""Wall-clock makespan acceptance criterion for split candidates.

Splits help at the scheduler level (tiles run in parallel on different
cores), NOT at the summed per-op cycle level. This script reads a
Contract-2 split hint, applies it to the workload's dispatch graph
(via apply_split_hint.py on the IR + ad-hoc JSON surgery on the
zephyr-chipyard-sw dispatch_graph.json that Dima's scheduler reads),
runs the decomposed scheduler before AND after, and reports the
measured Δmakespan.

The acceptance criterion the decision loop should use for splits:

    accept iff Δmakespan_us > epsilon_us AND no deadline miss introduced

Note: the per-tile cycle counts come from the existing profile-DB
linear-scaling assumption (tile_cycles ≈ orig_cycles × tile_N / orig_N).
For a fully measurement-grounded version, the per-tile cycles would
come from a FireSim/spike measurement of the rewritten harness; that's
documented as a follow-up and not blocking the wall-clock comparison.

Usage:
  scripts/wallclock_split_eval.py \
      --hint <hint.json> \
      --networks-json <toplevel/networks_*.json> \
      --solver decomposed \
      --out-dir <artifacts/.../wallclock>
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path("/scratch2/agustin/ModelBlaster")
XPURT_ROOT = Path("/scratch2/agustin/XPU-RT")
PY = "/scratch2/agustin/miniforge3/envs/merlin-dev/bin/python"


def _dispatch_graph_path_for_network(networks_json: Path,
                                     network_name: str) -> Path | None:
    """Read the workload JSON, locate the dispatch_deps_path for a given
    network. Path is XPU-RT-relative."""
    j = json.loads(networks_json.read_text())
    net = j.get("networks", {}).get(network_name)
    if not net:
        return None
    rel = net.get("dispatch_deps_path")
    if not rel:
        return None
    return XPURT_ROOT / rel


def apply_split_to_dispatch_graph(disp_graph: dict, split_ops: list) -> dict:
    """Apply a split hint to a Dima-format dispatch graph dict.

    The dispatch_graph.json has a flat `dispatches` dict mapping
    dispatch_<id> → {id, dependencies}. For each (op_id, n_splits) in
    `split_ops`, replace dispatch_<id> with N tile dispatches whose
    dependencies match the original. Downstream consumers that
    depended on dispatch_<id> get rewired to depend on ALL N tiles
    (correct semantics: the original output is the concat of tiles)."""
    out = copy.deepcopy(disp_graph)
    disps = out["dispatches"]
    # Build name→dispatch_id index
    next_id = max((int(d["id"]) for d in disps.values()), default=-1) + 1
    rename: dict[str, list[str]] = {}  # orig name → [tile_0_name, ...]
    new_disps: dict = {}
    for name, d in disps.items():
        did = int(d["id"])
        spec = next((s for s in split_ops if s.get("op") == did), None)
        if spec is None:
            new_disps[name] = d
            continue
        n = int(spec.get("n_splits", 2))
        tile_names = []
        for t in range(n):
            tn = f"{name}.tile_{t}"
            tile = copy.deepcopy(d)
            tile["id"] = next_id; next_id += 1
            tile["split_from"] = {"orig": name, "tile": t, "n_splits": n}
            new_disps[tn] = tile
            tile_names.append(tn)
        rename[name] = tile_names

    # Rewire dependencies
    for name, d in new_disps.items():
        deps = d.get("dependencies", [])
        rewired: list[str] = []
        for dep in deps:
            if dep in rename:
                rewired.extend(rename[dep])
            else:
                rewired.append(dep)
        d["dependencies"] = rewired

    out["dispatches"] = new_disps
    out["_split_applied"] = {"orig_n": len(disps), "new_n": len(new_disps)}
    return out


def run_scheduler(networks_json: Path, solver: str, out_dir: Path) -> dict:
    """Invoke Dima's scheduler, return parsed metrics dict."""
    cmd = [
        PY, str(XPURT_ROOT / "scripts" / "run_xpurt_schedule.py"),
        "--networks-json", str(networks_json),
        "--solver", solver,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{XPURT_ROOT/'xpu-rt'}"
    log = out_dir / "scheduler.log"
    p = subprocess.run(cmd, cwd=str(XPURT_ROOT), env=env,
                       capture_output=True, text=True)
    log.write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    if p.returncode != 0:
        raise SystemExit(f"scheduler failed; see {log}")
    # Parse from stdout: "  makespan_us=75.57  deadline_miss=0  cross_dev=180  solver_s=1.769"
    metrics = {}
    for line in p.stdout.splitlines():
        if "makespan_us=" in line:
            for tok in line.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    try:
                        metrics[k] = float(v) if "." in v else int(v)
                    except ValueError:
                        metrics[k] = v
            break
    return metrics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hint", required=True, type=Path)
    ap.add_argument("--networks-json", required=True, type=Path)
    ap.add_argument("--solver", default="decomposed")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hint = json.loads(args.hint.read_text())
    if hint.get("contract") != "modelblaster.split_hints/v1":
        raise SystemExit("hint must be modelblaster.split_hints/v1")

    print(f"[wallclock_split_eval] hint={args.hint}")

    # 1. BEFORE: scheduler on original networks_json
    before = run_scheduler(args.networks_json, args.solver, args.out_dir)
    (args.out_dir / "before.json").write_text(json.dumps(before, indent=2))
    print(f"[wallclock_split_eval] BEFORE: {before}")

    # 2. Apply split to dispatch_graph.json for each network in the hint
    backups: list[tuple[Path, Path]] = []
    try:
        for entry in hint.get("networks", []):
            net = entry["network"]
            split_ops = entry.get("split_ops", [])
            graph_path = _dispatch_graph_path_for_network(
                args.networks_json, net)
            if graph_path is None:
                raise SystemExit(f"no dispatch_deps_path for network {net}")
            if not graph_path.exists():
                raise SystemExit(f"missing dispatch graph at {graph_path}")
            backup = args.out_dir / f"{net}_orig_dispatch_graph.json"
            shutil.copy(graph_path, backup)
            backups.append((graph_path, backup))
            orig = json.loads(graph_path.read_text())
            rewritten = apply_split_to_dispatch_graph(orig, split_ops)
            rewritten_path = args.out_dir / f"{net}_split_dispatch_graph.json"
            rewritten_path.write_text(json.dumps(rewritten, indent=2))
            graph_path.write_text(json.dumps(rewritten, indent=2))
            print(f"[wallclock_split_eval] swapped {graph_path.name}: "
                  f"{len(orig['dispatches'])} -> {len(rewritten['dispatches'])} dispatches")

        # 3. AFTER: scheduler with the swapped graphs
        after = run_scheduler(args.networks_json, args.solver, args.out_dir)
        (args.out_dir / "after.json").write_text(json.dumps(after, indent=2))
        print(f"[wallclock_split_eval] AFTER:  {after}")

    finally:
        # Always restore originals
        for target, backup in backups:
            shutil.copy(backup, target)
        print("[wallclock_split_eval] restored originals")

    # 4. Report
    delta_us = before.get("makespan_us", 0) - after.get("makespan_us", 0)
    delta_pct = (100.0 * delta_us / before["makespan_us"]
                 if before.get("makespan_us") else 0.0)
    decision = ("ACCEPT" if delta_us > 0.5 and
                after.get("deadline_miss", 0) <= before.get("deadline_miss", 0)
                else "REJECT")
    report = {
        "hint": str(args.hint),
        "solver": args.solver,
        "before": before, "after": after,
        "delta_makespan_us": delta_us,
        "delta_pct": delta_pct,
        "decision": decision,
    }
    (args.out_dir / "wallclock_summary.json").write_text(
        json.dumps(report, indent=2))
    print(f"\n[wallclock_split_eval] decision: {decision} "
          f"(Δ={delta_us:+.2f} µs, {delta_pct:+.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
