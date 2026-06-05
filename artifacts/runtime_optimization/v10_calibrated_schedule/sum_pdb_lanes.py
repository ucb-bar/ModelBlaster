"""Sum the per-op cost on each lane (rvv_opu vs gemmini) using the
CURRENT FireSim PDB. Reproduces the lower-bound calculations in the
v10 REPORT.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

BASE = Path("/scratch2/agustin/XPU-RT/gen/profile")


def load(p: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    if not p.exists():
        return out
    with p.open() as f:
        for row in csv.DictReader(f):
            try:
                did = int(row["dispatch_id"])
                out[did] = {
                    "time_ms": float(row["mean_time"]),
                    "op": row.get("op", ""),
                    "shape": row.get("shape", ""),
                }
            except (ValueError, TypeError):
                continue
    return out


def main() -> None:
    yolo_rvv = load(BASE / "V256D128_rvv/firesim_rocket_saturn/yolov8_nano/yolov8_nano.int8/yolov8_nano_firesim_rocket_saturn_V256D128_rvv_yolov8_nano.int8/topo_0/results.csv")
    yolo_gem = load(BASE / "gemmini_q31/firesim_rocket_saturn/yolov8_nano/yolov8_nano.int8/yolov8_nano_firesim_rocket_saturn_gemmini_q31_yolov8_nano.int8/topo_0/results.csv")
    dronet_rvv = load(BASE / "V256D128_rvv/firesim_rocket_saturn/dronet/dronet.int8/dronet_firesim_rocket_saturn_V256D128_rvv_dronet.int8/topo_0/results.csv")
    dronet_gem = load(BASE / "gemmini_q31/firesim_rocket_saturn/dronet/dronet.int8/dronet_firesim_rocket_saturn_gemmini_q31_dronet.int8/topo_0/results.csv")
    mlp_rvv = load(BASE / "V256D128_rvv/firesim_rocket_saturn/mlp_control/mlp_control.fp32/mlp_control_firesim_rocket_saturn_RVV_mlp_control.fp32/topo_0/results.csv")
    mlp_gem = load(BASE / "gemmini_q31/firesim_rocket_saturn/mlp_control/mlp_control.fp32/mlp_control_firesim_rocket_saturn_gemmini_q31_mlp_control.fp32/topo_0/results.csv")

    out = {}
    for name, rvv, gem in [
        ("yolov8_nano", yolo_rvv, yolo_gem),
        ("dronet", dronet_rvv, dronet_gem),
        ("mlp_control", mlp_rvv, mlp_gem),
    ]:
        rvv_total = sum(v["time_ms"] for v in rvv.values())
        gem_total = sum(v["time_ms"] for v in gem.values())
        all_ids = set(rvv) | set(gem)
        lb = sum(
            min(rvv.get(k, {"time_ms": 1e9})["time_ms"], gem.get(k, {"time_ms": 1e9})["time_ms"])
            for k in all_ids
        )
        out[name] = {
            "n_ops": len(all_ids),
            "rvv_total_ms": rvv_total,
            "gem_total_ms": gem_total,
            "min_lower_bound_ms": lb,
        }

    # The workload: 4 mlp + 2 dronet + 1 yolo
    wl_lb = {
        "mlp_control_x4_lb_ms": 4 * out["mlp_control"]["min_lower_bound_ms"],
        "dronet_x2_lb_ms": 2 * out["dronet"]["min_lower_bound_ms"],
        "yolov8_x1_lb_ms": out["yolov8_nano"]["min_lower_bound_ms"],
        "total_lb_ms": (
            4 * out["mlp_control"]["min_lower_bound_ms"]
            + 2 * out["dronet"]["min_lower_bound_ms"]
            + out["yolov8_nano"]["min_lower_bound_ms"]
        ),
    }
    print(json.dumps({"per_network": out, "workload_lb": wl_lb}, indent=2))
    Path("/scratch2/agustin/ModelBlaster/artifacts/runtime_optimization/v10_calibrated_schedule/pdb_lane_summary.json").write_text(
        json.dumps({"per_network": out, "workload_lb": wl_lb}, indent=2)
    )


if __name__ == "__main__":
    main()
