"""Take the v9 schedule's (dispatch -> hardware_target) placement and
recompute the per-hart total using the CURRENT (bit-exact) FireSim PDB.

This is the "what-if v9's placement was scored against measured cycles"
analysis — gives the lower-bound makespan if the solver hadn't been
fooled by the stale prebitexact_backup numbers.
"""
from __future__ import annotations
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

V9_SCHEDULE = Path("/scratch2/agustin/XPU-RT/schedules/scheduled_networks_1yolo_4mlp_2dronet_firesim_hybrid.json")
PDB_BASE = Path("/scratch2/agustin/XPU-RT/gen/profile")
OUT = Path("/scratch2/agustin/ModelBlaster/artifacts/runtime_optimization/v10_calibrated_schedule")


def load_pdb(p: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    if not p.exists():
        return out
    with p.open() as f:
        for row in csv.DictReader(f):
            try:
                out[int(row["dispatch_id"])] = float(row["mean_time"])
            except (ValueError, TypeError):
                continue
    return out


PDBS = {
    ("yolov8_nano", "V256D128_rvv"): load_pdb(PDB_BASE / "V256D128_rvv/firesim_rocket_saturn/yolov8_nano/yolov8_nano.int8/yolov8_nano_firesim_rocket_saturn_V256D128_rvv_yolov8_nano.int8/topo_0/results.csv"),
    ("yolov8_nano", "gemmini_q31"): load_pdb(PDB_BASE / "gemmini_q31/firesim_rocket_saturn/yolov8_nano/yolov8_nano.int8/yolov8_nano_firesim_rocket_saturn_gemmini_q31_yolov8_nano.int8/topo_0/results.csv"),
    ("dronet", "V256D128_rvv"): load_pdb(PDB_BASE / "V256D128_rvv/firesim_rocket_saturn/dronet/dronet.int8/dronet_firesim_rocket_saturn_V256D128_rvv_dronet.int8/topo_0/results.csv"),
    ("dronet", "gemmini_q31"): load_pdb(PDB_BASE / "gemmini_q31/firesim_rocket_saturn/dronet/dronet.int8/dronet_firesim_rocket_saturn_gemmini_q31_dronet.int8/topo_0/results.csv"),
    ("mlp_control", "V256D128_rvv"): load_pdb(PDB_BASE / "V256D128_rvv/firesim_rocket_saturn/mlp_control/mlp_control.fp32/mlp_control_firesim_rocket_saturn_RVV_mlp_control.fp32/topo_0/results.csv"),
    ("mlp_control", "gemmini_q31"): load_pdb(PDB_BASE / "gemmini_q31/firesim_rocket_saturn/mlp_control/mlp_control.fp32/mlp_control_firesim_rocket_saturn_gemmini_q31_mlp_control.fp32/topo_0/results.csv"),
}

# CPU_P maps to gemmini_q31; CPU_E maps to V256D128_rvv (per workload).
HW_MAP = {"CPU_P": "gemmini_q31", "CPU_E": "V256D128_rvv"}


def main() -> None:
    sched = json.loads(V9_SCHEDULE.read_text())
    per_hw_old = defaultdict(float)
    per_hw_new = defaultdict(float)
    missing = 0
    matched = 0

    for name, entry in sched["dispatches"].items():
        old_dur = float(entry["duration"])
        hw_target = entry["hardware_target"].split("#")[0]
        hw = HW_MAP[hw_target]
        per_hw_old[hw_target] += old_dur

        # Find which network and dispatch_id
        # name pattern: '<net_id_with_instance>_dispatch_<N>' e.g.
        # 'mlp_control0_dispatch_5', 'yolov8_nano_dispatch_169'
        # base network name without instance digit:
        net_base = None
        for nb in ("yolov8_nano", "dronet", "mlp_control"):
            if name.startswith(nb):
                net_base = nb
                break
        if net_base is None:
            missing += 1
            continue
        m = re.search(r"_dispatch_(\d+)$", name)
        if not m:
            missing += 1
            continue
        did = int(m.group(1))
        new_t = PDBS.get((net_base, hw), {}).get(did)
        if new_t is None:
            missing += 1
            continue
        matched += 1
        per_hw_new[hw_target] += new_t

    summary = {
        "n_dispatches": len(sched["dispatches"]),
        "n_matched": matched,
        "n_missing": missing,
        "per_hw_old_ms": dict(per_hw_old),
        "per_hw_new_ms": dict(per_hw_new),
        "old_max_lane_ms": max(per_hw_old.values()) if per_hw_old else 0.0,
        "new_max_lane_ms": max(per_hw_new.values()) if per_hw_new else 0.0,
    }
    print(json.dumps(summary, indent=2))
    (OUT / "v9_placement_recost.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
