"""Four-way trajectory comparison: PyTorch fp32 / int8 ceiling / ONNX
int8 (QDQ) / spike actual.

ViNT outputs a (1, 5, 4) waypoint tensor (cumsum'd (x, y) + L2-normalized
(sin θ, cos θ)) plus a dist_pred scalar. The four versions:

 1. PyTorch fp32 — the ground truth the trained model emits.
 2. "int8 representation ceiling" — quantize PyTorch fp32 outputs to
    int8 with our extracted scales, then dequantize + apply tail.
    Upper bound on accuracy if our int8 forward were lossless.
 3. ONNX int8 (QDQ) — vint_int8.onnx from the existing
    sims/scripts/utils/quantize_vint.py. The QDQ format stores int8
    values between ops but the underlying compute is fp32, so this is
    a strong reference for "what stock industry-tooling int8 PTQ
    produces". A big gap between this and (4) means our true-int8
    forward (not the quant scheme) is the dominant drift source.
 4. Spike actual — our binary's output after running 354 real int8
    ops through the EfficientNet body + transformer + composite tail.
    Read from the MODELBLASTER_OUTPUT_BEGIN/END block in spike stdout.

The script runs all four on the same input (the first calibration
sample, which is also what io.npz pinned the golden against) and prints
a side-by-side table in waypoint-space units.

Run via:
    conda activate zephyr   # zephyr env has spike on PATH
    cd zephyr-chipyard-sw
    # Build first if you haven't (bash modelblaster/examples/vint/run.sh).
    PYTHONPATH=. python modelblaster/examples/vint/compare_waypoints.py
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VINT_TRAIN = _REPO_ROOT / "sims/external/visualnav-transformer/train"
if str(_VINT_TRAIN) not in sys.path:
    sys.path.insert(0, str(_VINT_TRAIN))


def _run_spike(elf: str, spike_bin: str) -> str:
    """Run spike on the elf, return stdout text."""
    res = subprocess.run(
        [spike_bin, "--isa=rv64gcv_zicntr", elf],
        capture_output=True, text=True, timeout=600,
    )
    return res.stdout + res.stderr


def _parse_output_block(text: str) -> np.ndarray:
    """Extract the fp32 values inside the MODELBLASTER_OUTPUT_BEGIN/END markers
    the harness emits (one float per line)."""
    m = re.search(
        r"=== MODELBLASTER_OUTPUT_BEGIN ===\n(.*?)\n=== MODELBLASTER_OUTPUT_END ===",
        text, re.DOTALL,
    )
    if not m:
        raise RuntimeError("no MODELBLASTER_OUTPUT block in spike stdout — did "
                           "the harness build with the dump enabled?")
    vals = [float(line) for line in m.group(1).strip().split("\n") if line.strip()]
    return np.asarray(vals, dtype=np.float32)


def _apply_tail_to_fp32(deltas_fp32: np.ndarray) -> np.ndarray:
    """Numpy mirror of the C kernel's tail (matches our reference impl)."""
    deltas = deltas_fp32.reshape(5, 4)
    out = np.zeros_like(deltas)
    out[:, 0] = np.cumsum(deltas[:, 0])
    out[:, 1] = np.cumsum(deltas[:, 1])
    norms = np.linalg.norm(deltas[:, 2:], axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    out[:, 2:] = deltas[:, 2:] / norms
    return out


def _print_wp_row(label: str, dist: float, wp_arr: np.ndarray, *,
                  ref: np.ndarray | None = None):
    """Print a (5, 4) waypoint table with optional Δ-vs-reference column."""
    print(f"--- {label} ---")
    print(f"  dist_pred = {dist:+.3f}")
    hdr = f"  {'wp':>2s} | {'x':>8s} {'y':>8s} {'sin':>8s} {'cos':>8s}"
    if ref is not None:
        hdr += f" | {'Δpos':>8s} {'Δθ°':>7s}"
    print(hdr)
    for wp in range(5):
        x, y, s, c = wp_arr[wp]
        line = f"  {wp:>2d} | {x:+8.3f} {y:+8.3f} {s:+8.3f} {c:+8.3f}"
        if ref is not None:
            rx, ry, rs, rc = ref[wp]
            dpos = float(np.hypot(x - rx, y - ry))
            ang_a = np.degrees(np.arctan2(s, c))
            ang_r = np.degrees(np.arctan2(rs, rc))
            dtheta = abs(ang_a - ang_r)
            if dtheta > 180: dtheta = 360 - dtheta
            line += f" | {dpos:>7.3f}m {dtheta:>6.1f}"
        print(line)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-dir",
                   default="modelblaster/examples/vint/int8/build/scalar")
    p.add_argument("--ir-dir",
                   default="modelblaster/examples/vint/int8/generated")
    p.add_argument("--spike",
                   default="/scratch2/dima/miniforge3/envs/zephyr/bin/spike")
    p.add_argument("--waypoint-idx", type=int, default=2)
    p.add_argument("--omega-gain", type=float, default=2.0)
    p.add_argument("--onnx-int8", default=str(
        _REPO_ROOT.parent / "sims/external/visualnav-transformer/"
                            "deployment/model_weights/vint_int8.onnx"),
                   help="QDQ int8 ONNX path; set empty string to skip.")
    args = p.parse_args()

    elf = Path(args.build_dir) / "zephyr" / "zephyr.elf"
    if not elf.exists():
        sys.exit(f"build the binary first: bash modelblaster/examples/vint/run.sh "
                 f"(missing {elf})")

    # ---- (1) PyTorch fp32 on the same input the binary saw ---------------
    from modelblaster.models import vint as vint_mod
    samples = vint_mod.get_calibration_samples(1)
    obs0, goal0 = samples[0]
    model = vint_mod.get_model()
    with torch.no_grad():
        dist_t, action_t = model(obs0, goal0)
    fp32_dist = float(dist_t[0, 0])
    fp32_wp = action_t[0].cpu().numpy()  # (5, 4), already cumsum'd by ViNT

    # ---- (2) int8 representation ceiling ---------------------------------
    # io.npz["output"] is the captured tensor (PyTorch fp32 at linear_24,
    # which is the pre-tail intermediate), passed through our composite
    # kernel's numpy oracle to apply cumsum + normalize. So this is the
    # "perfect int8 forward" upper bound on accuracy.
    io = np.load(Path(args.ir_dir) / "io.npz")
    # io.npz["output"] is the composite's expected output (21 fp32):
    ceil_dist = float(io["output"][0])
    ceil_wp = io["output"][1:21].reshape(5, 4)

    # ---- (3) QDQ ONNX int8 ------------------------------------------------
    onnx_dist, onnx_wp = None, None
    if args.onnx_int8 and Path(args.onnx_int8).is_file():
        try:
            import onnxruntime as ort  # noqa: PLC0415
            sess = ort.InferenceSession(
                args.onnx_int8, providers=["CPUExecutionProvider"])
            in_names = [i.name for i in sess.get_inputs()]
            feed = {in_names[0]: obs0.numpy().astype(np.float32),
                    in_names[1]: goal0.numpy().astype(np.float32)}
            onnx_outs = sess.run(None, feed)
            onnx_dist = float(onnx_outs[0].ravel()[0])
            onnx_wp = onnx_outs[1][0]  # (5, 4)
        except Exception as e:
            print(f"[warn] ONNX int8 inference failed ({e}); skipping row 3.")
    else:
        print(f"[info] {args.onnx_int8} not present; skipping ONNX int8 row.")

    # ---- (4) Spike actual output -----------------------------------------
    print(f"running spike on {elf} ...", flush=True)
    spike_text = _run_spike(str(elf), args.spike)
    spike_arr = _parse_output_block(spike_text)
    spike_dist = float(spike_arr[0])
    spike_wp = spike_arr[1:21].reshape(5, 4)

    # ---- Side-by-side ---------------------------------------------------
    print()
    _print_wp_row("PyTorch fp32 (ground truth)", fp32_dist, fp32_wp)
    print()
    _print_wp_row("int8 representation ceiling", ceil_dist, ceil_wp, ref=fp32_wp)
    if onnx_wp is not None:
        print()
        _print_wp_row("ONNX int8 (QDQ — int8 storage, fp32 compute)",
                       onnx_dist, onnx_wp, ref=fp32_wp)
    print()
    _print_wp_row("spike actual (real int8 compute)", spike_dist, spike_wp,
                  ref=fp32_wp)

    # Pilot steering signal at the chosen waypoint
    pick = args.waypoint_idx
    def _omega(wp):
        return np.arctan2(wp[pick, 1], wp[pick, 0]) * args.omega_gain
    print(f"\nPilot ω@wp{pick}:")
    print(f"  fp32:    {_omega(fp32_wp):+.3f} rad/s")
    print(f"  ceiling: {_omega(ceil_wp):+.3f} rad/s  Δ vs fp32 = "
          f"{abs(_omega(ceil_wp) - _omega(fp32_wp)):.3f}")
    if onnx_wp is not None:
        print(f"  onnx:    {_omega(onnx_wp):+.3f} rad/s  Δ vs fp32 = "
              f"{abs(_omega(onnx_wp) - _omega(fp32_wp)):.3f}")
    print(f"  spike:   {_omega(spike_wp):+.3f} rad/s  Δ vs fp32 = "
          f"{abs(_omega(spike_wp) - _omega(fp32_wp)):.3f}")


if __name__ == "__main__":
    main()
