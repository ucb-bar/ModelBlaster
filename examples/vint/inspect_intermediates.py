"""Compare spike vs PyTorch fp32 at chosen intermediate tensors.

Usage:
    # 1. Tell the extractor which tensors to inspect, rebuild:
    bash modelblaster/examples/vint/run.sh    # but pass VINT_INSPECT=t1,t2,...
    # OR run extract_graph_export directly:
    PYTHONPATH=. python -m modelblaster.pipeline.extract_graph_export \\
        --model vint --quant int8 --num-calibration 16 --per-channel \\
        --inspect linear,linear_1,cat_1,linear_24 \\
        --out-dir modelblaster/examples/vint/int8/generated
    # then rebuild + run spike via run.sh (FORCE_EXTRACT=0).

    # 2. Run this script — it executes spike, parses the inspect blocks,
    #    and prints a per-tensor summary of spike vs PyTorch.
    PYTHONPATH=. python modelblaster/examples/vint/inspect_intermediates.py

Output for each inspected tensor:
    name    n_elems    fp32 max_abs    spike max_abs    Δ max_abs    cosine_sim

Cosine similarity gives a magnitude-invariant view (signal direction);
max-abs delta is the worst single-element fp32 error after dequant.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]

_INSPECT_RE = re.compile(
    r"=== MODELBLASTER_INSPECT_BEGIN \[(?P<name>[^\]]+)\] === "
    r"scale=(?P<scale>[^ ]+) dtype=(?P<dtype>\w+) n=(?P<n>\d+)\n"
    r"(?P<body>.*?)"
    r"=== MODELBLASTER_INSPECT_END \[\1\] ===",
    re.DOTALL,
)


def _parse_inspects(text: str) -> dict:
    out = {}
    for m in _INSPECT_RE.finditer(text):
        name = m.group("name")
        scale = float(m.group("scale"))
        dtype = m.group("dtype")
        n = int(m.group("n"))
        body = m.group("body").strip().split("\n")
        if len(body) != n:
            print(f"[warn] {name}: expected {n} elements, got {len(body)}",
                  file=sys.stderr)
        vals = np.asarray([float(x) for x in body], dtype=np.float32)
        if dtype == "i8":
            vals = vals * scale  # dequantize to fp32 domain
        out[name] = {"values": vals, "scale": scale, "dtype": dtype}
    return out


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12: return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-dir",
                   default="modelblaster/examples/vint/int8/build/scalar")
    p.add_argument("--ir-dir",
                   default="modelblaster/examples/vint/int8/generated")
    p.add_argument("--spike",
                   default="/scratch2/dima/miniforge3/envs/zephyr/bin/spike")
    p.add_argument("--spike-isa", default="rv64gcv_zicntr",
                   help="ISA passed via --isa= to spike. Use "
                        "rv64gcv_zicntr_zfh for fp16 binaries — without "
                        "zfh, the fp16 ops soft-emulate at ~10x cost and "
                        "blow past the 900s timeout for ViNT-scale models.")
    args = p.parse_args()

    elf = Path(args.build_dir) / "zephyr" / "zephyr.elf"
    ref_npz = Path(args.ir_dir) / "inspect_ref.npz"
    if not elf.exists():
        sys.exit(f"missing {elf}; build the binary first")
    if not ref_npz.exists():
        sys.exit(f"missing {ref_npz}; rerun extract_graph_export with "
                 f"--inspect <tensor names>")

    print(f"loading PyTorch fp32 references from {ref_npz} ...", flush=True)
    ref = np.load(ref_npz)
    inspect_names = [k for k in ref.files if not k.endswith("__scale")]
    print(f"  {len(inspect_names)} tensors with fp32 ref: {inspect_names}",
          flush=True)

    print(f"running spike on {elf} ...", flush=True)
    res = subprocess.run(
        [args.spike, f"--isa={args.spike_isa}", str(elf)],
        capture_output=True, text=True, timeout=1800,
    )
    spike_blocks = _parse_inspects(res.stdout + res.stderr)
    if not spike_blocks:
        sys.exit("no MODELBLASTER_INSPECT_BEGIN/END blocks in spike stdout — "
                 "did the build pick up the inspect tensors?")
    print(f"  spike dumped {len(spike_blocks)} inspect blocks",
          flush=True)

    # Per-tensor side-by-side.
    print()
    print(f"{'tensor':<22s} {'n':>7s} {'|fp32|_max':>12s} "
          f"{'|spike|_max':>12s} {'Δ_max':>10s} {'cos_sim':>8s} {'comment':<30s}")
    print("-" * 110)
    for nm in inspect_names:
        ref_arr = ref[nm].ravel()
        if nm not in spike_blocks:
            print(f"{nm:<22s}  spike block missing")
            continue
        spike_arr = spike_blocks[nm]["values"]
        if spike_arr.size != ref_arr.size:
            print(f"{nm:<22s}  size mismatch: ref={ref_arr.size} "
                  f"spike={spike_arr.size}")
            continue
        # Reference: PyTorch fp32 values. But the spike returns values
        # at the int8 representation's scale — so we compare apples to
        # apples by also quantize+dequant the fp32 reference at the
        # same scale (the "int8 ceiling" at this tensor).
        scale = spike_blocks[nm]["scale"]
        ref_quant = np.round(ref_arr / scale).clip(-127, 127).astype(np.int8)
        ref_ceil = ref_quant.astype(np.float32) * scale
        delta = spike_arr - ref_ceil
        ref_max_abs = float(np.abs(ref_arr).max())
        spike_max_abs = float(np.abs(spike_arr).max())
        delta_max_abs = float(np.abs(delta).max())
        cos = _cos(ref_arr, spike_arr)
        # Heuristic flag: spike magnitude < 25% of ref suggests strong
        # compression somewhere upstream.
        comp = "compressed!" if spike_max_abs < 0.25 * ref_max_abs else ""
        print(f"{nm:<22s} {ref_arr.size:>7d} {ref_max_abs:>12.4f} "
              f"{spike_max_abs:>12.4f} {delta_max_abs:>10.4f} "
              f"{cos:>8.4f} {comp:<30s}")


if __name__ == "__main__":
    main()
