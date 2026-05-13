"""Trajectory-space comparison: PyTorch fp32 vs our int8 (quant+dequant).

Element-wise int8 max_abs_err doesn't translate cleanly to "is this
model still usable for navigation?". This script dequantizes the int8
golden in io.npz back to fp32 using the output scales recorded in
graph.json, then compares the waypoint sequences against the fp32
PyTorch reference in the units the steering loop actually consumes
(meters, degrees, rad/s).

The output answers: "how much does our int8 forward bend the
trajectory away from the fp32 baseline, in robot-frame coordinates?"

Run via:
    conda activate xpurt
    cd zephyr-chipyard-sw
    PYTHONPATH=. python agents/examples/vint/compare_waypoints.py

Outputs (per calibration sample):
  * fp32 (x, y, θ) waypoints from PyTorch ViNT
  * int8-quantized waypoints from io.npz
  * per-waypoint position delta (meters)
  * per-waypoint heading delta (degrees)
  * pilot steering signal: ω from waypoint idx=2 atan2 + gain=2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VINT_TRAIN = _REPO_ROOT / "sims/external/visualnav-transformer/train"
if str(_VINT_TRAIN) not in sys.path:
    sys.path.insert(0, str(_VINT_TRAIN))


def _dequantize(int8_arr: np.ndarray, scale: float) -> np.ndarray:
    return int8_arr.astype(np.float32) * scale


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ir-dir", default="agents/examples/vint/int8/generated",
                   help="dir containing graph.json + io.npz (output dir of "
                        "extract_graph_export).")
    p.add_argument("--n-samples", type=int, default=4,
                   help="number of calibration samples to evaluate.")
    p.add_argument("--waypoint-idx", type=int, default=2,
                   help="which of ViNT's 5 waypoints to map to a steering "
                        "command (matches pilot_forest_with_vint.py default).")
    p.add_argument("--omega-gain", type=float, default=2.0,
                   help="atan2(y,x) → ω gain, matching the pilot's default.")
    args = p.parse_args()

    ir = json.load(open(Path(args.ir_dir) / "graph.json"))
    io = np.load(Path(args.ir_dir) / "io.npz")
    out_names = ir["output"]["tensors"]
    out_scales = [ir["tensors"][n]["quant"]["scale"] for n in out_names]
    # ViNT output layout: dist (1 elem, scale 0) + action (20 elem, scale 1).
    if len(out_scales) != 2:
        raise SystemExit(f"expected 2 output tensors, got {len(out_scales)}")
    print(f"output scales: dist={out_scales[0]:.5f}  action={out_scales[1]:.5f}")
    print(f"action int8 dynamic range: ±{127*out_scales[1]:.3f} (m)")
    print()

    from agents.models import vint as vint_mod
    samples = vint_mod.get_calibration_samples(max(args.n_samples, 1))
    model = vint_mod.get_model()

    # The int8 golden in io.npz was computed from samples[0] only —
    # so the int8 row is meaningful only for sample 0; for the rest
    # the script just prints fp32 baseline waypoints (useful to show
    # how the fp32 trajectory varies across inputs).
    int8_out = io["output"]   # int8 buffer for sample[0]
    dist_int8_s0 = float(int8_out[0]) * out_scales[0]
    # The io.npz "action" portion is the PRE-cumsum, PRE-normalize
    # delta tensor (linear_24's output in ViNT) — the IR resolved
    # `reshape_2` through the tail-op alias chain back to that
    # captured intermediate. Apply the tail post-process in scalar
    # here so the comparison matches what the pilot's steering
    # consumes (cumsum'd waypoints + L2-normalized angle).
    action_int8_deltas = (
        int8_out[1:21].astype(np.float32) * out_scales[1]
    ).reshape(5, 4)
    action_int8_s0 = action_int8_deltas.copy()
    # cumsum the (dx, dy) cols.
    action_int8_s0[:, :2] = np.cumsum(action_int8_s0[:, :2], axis=0)
    # L2-normalize the (sin, cos) cols.
    n = np.linalg.norm(action_int8_s0[:, 2:], axis=1, keepdims=True)
    n = np.clip(n, 1e-12, None)
    action_int8_s0[:, 2:] = action_int8_s0[:, 2:] / n

    for i in range(args.n_samples):
        obs, goal = samples[i]
        with torch.no_grad():
            dist_fp32, action_fp32 = model(obs, goal)
        a_fp = action_fp32[0].cpu().numpy()
        print(f"=== sample {i} ===   "
              f"fp32 dist_pred = {float(dist_fp32[0,0]):+.3f}")
        if i == 0:
            print(f"  (int8 dist_pred = {dist_int8_s0:+.3f})")
        header = (f"  {'wp':>2s} | {'fp32 (x,y)':>18s} | "
                  f"{'int8 (x,y)':>18s} | Δpos(m) | "
                  f"fp32_θ°  int8_θ°  Δθ°")
        print(header)
        for wp in range(5):
            fx, fy, fs, fc = a_fp[wp]
            fang = np.degrees(np.arctan2(fs, fc))
            if i == 0:
                ix, iy, isin, ico = action_int8_s0[wp]
                iang = np.degrees(np.arctan2(isin, ico))
                err = float(np.hypot(fx - ix, fy - iy))
                print(f"  {wp:>2d} | ({fx:+6.3f},{fy:+6.3f})  | "
                      f"({ix:+6.3f},{iy:+6.3f})  | {err:>6.3f}  | "
                      f"{fang:+7.1f} {iang:+7.1f}  {abs(fang-iang):>5.1f}")
            else:
                print(f"  {wp:>2d} | ({fx:+6.3f},{fy:+6.3f})  | "
                      f"{'-':>18s} | {'-':>6s}  | "
                      f"{fang:+7.1f} {'-':>7s}  {'-':>5s}")
        f_pick = a_fp[args.waypoint_idx]
        f_omega = np.arctan2(f_pick[1], f_pick[0]) * args.omega_gain
        if i == 0:
            i_pick = action_int8_s0[args.waypoint_idx]
            i_omega = np.arctan2(i_pick[1], i_pick[0]) * args.omega_gain
            print(f"  pilot ω@idx{args.waypoint_idx}: "
                  f"fp32={f_omega:+.3f}rad/s  int8={i_omega:+.3f}rad/s  "
                  f"Δ={abs(f_omega - i_omega):.3f}rad/s")
        else:
            print(f"  pilot ω@idx{args.waypoint_idx}: "
                  f"fp32={f_omega:+.3f}rad/s")
        print()


if __name__ == "__main__":
    main()
