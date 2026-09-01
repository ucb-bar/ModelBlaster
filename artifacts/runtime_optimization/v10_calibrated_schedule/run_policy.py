"""Drive hybrid_periodic_mosek_yolo with the CURRENT (bit-exact) PDB
and report the new makespan.

Compares against the v9 schedule (which was generated against the
prebitexact_backup CSVs and predicted 70 ms but FireSim measured
786 ms).
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

XPURT = Path("/scratch2/agustin/XPU-RT")
PY = "/scratch2/agustin/miniforge3/envs/merlin-dev/bin/python"
OUT = Path("/scratch2/agustin/ModelBlaster/artifacts/runtime_optimization/v10_calibrated_schedule")
WORKLOAD = XPURT / "data/toplevel/networks_1yolo_4mlp_2dronet_firesim.json"

OUT.mkdir(parents=True, exist_ok=True)


def ensure_mlp_gemmini_csv() -> None:
    """profile_loader is strict-mode. The mlp_control gemmini_q31 results.csv
    isn't under `gen/profile/...` but exists under
    `zephyr-chipyard-sw/gen/profile/sweep_v8/...`. Symlink it into the
    expected location so the loader finds it.
    """
    src = XPURT / ("zephyr-chipyard-sw/gen/profile/sweep_v8/gemmini_q31/"
                   "firesim_rocket_saturn/mlp_control/mlp_control.fp32/"
                   "mlp_control_firesim_rocket_saturn_gemmini_q31_mlp_control.fp32/"
                   "topo_0/results.csv")
    dst_dir = XPURT / ("gen/profile/gemmini_q31/firesim_rocket_saturn/"
                       "mlp_control/mlp_control.fp32/"
                       "mlp_control_firesim_rocket_saturn_gemmini_q31_mlp_control.fp32/"
                       "topo_0")
    dst = dst_dir / "results.csv"
    if not src.exists():
        print(f"WARN: no sweep_v8 mlp gemmini CSV at {src}")
        return
    if dst.exists():
        print(f"  {dst} already exists; skipping symlink")
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst)
    print(f"  symlinked {dst} -> {src}")


def main() -> int:
    print("=" * 70)
    print("v10 — recalibrate hybrid_periodic_mosek_yolo against current PDB")
    print("=" * 70)
    ensure_mlp_gemmini_csv()

    sys.path.insert(0, str(XPURT))
    sys.path.insert(0, str(XPURT / "xpu-rt"))
    sys.path.insert(0, str(XPURT / "scripts"))
    from policies.hybrid_periodic_mosek_yolo import hybrid_periodic_mosek_yolo

    t0 = time.perf_counter()
    result = hybrid_periodic_mosek_yolo(str(WORKLOAD), time_limit=300.0)
    wall = time.perf_counter() - t0
    print()
    print(f"Policy wall: {wall:.1f}s")
    print(f"Status: {result.get('status')}")
    print(f"Makespan: {result.get('makespan')}")
    print(f"Deadline misses: {result.get('n_deadline_miss')}")
    print(f"Dispatches: {result.get('n_dispatches')}")

    (OUT / "policy_result.json").write_text(json.dumps(result, indent=2))

    fp = result.get("fixture_path")
    if fp and Path(fp).exists():
        shutil.copy2(fp, OUT / "scheduled_hybrid_v10.json")
        print(f"Copied fixture -> {OUT / 'scheduled_hybrid_v10.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
