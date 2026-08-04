"""Generate random calibration data (.raw + input_list.txt) for each
trampoline-phase ONNX.

For DSP quantization we need per-phase calibration samples — the quantizer
uses these to choose activation-tensor int8 ranges. For TIMING profiling
the data distribution doesn't matter (DSP execution time is data-distribution
invariant), so random fp32 samples are sufficient.

Each phase gets:
  vision_slices_v3/trampolines/calibration/<phase>/input_<NUM>.raw  (N samples)
  vision_slices_v3/trampolines/calibration/<phase>_cal_list.txt
      one line per sample, listing the input .raw paths

The cal_list format matches qairt-quantizer's expectation:
  When a phase has multiple inputs, the line is "name1:=path1 name2:=path2".

Usage:
  python gen_trampoline_calibration.py \\
      --tramp-dir vision_slices_v3/trampolines \\
      --num-samples 10
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import onnx


def gen_one_phase(onnx_path: Path, cal_dir: Path, num_samples: int) -> Path:
    """Returns the calibration list path."""
    model = onnx.load(str(onnx_path))
    phase_name = onnx_path.stem
    phase_cal_dir = cal_dir / phase_name
    phase_cal_dir.mkdir(parents=True, exist_ok=True)

    # Each input gets one .raw blob per sample, named by tensor.
    input_specs = []
    for inp in model.graph.input:
        dims = [d.dim_value if d.dim_value > 0 else 1 for d in inp.type.tensor_type.shape.dim]
        elem_type = inp.type.tensor_type.elem_type
        if elem_type == onnx.TensorProto.FLOAT:
            dtype = np.float32
        elif elem_type == onnx.TensorProto.INT32:
            dtype = np.int32
        elif elem_type == onnx.TensorProto.INT64:
            dtype = np.int64
        else:
            dtype = np.float32   # default
        input_specs.append((inp.name, dims, dtype))

    list_path = cal_dir / f"{phase_name}_cal_list.txt"
    rng = np.random.default_rng(seed=hash(phase_name) % (2**32))
    with open(list_path, "w") as f:
        for sample_i in range(num_samples):
            tokens = []
            for inp_name, dims, dtype in input_specs:
                # Sanitize input name for filename (ONNX names contain '/', '.', etc.)
                safe = inp_name.replace("/", "_").replace(":", "_").replace(".", "_")
                raw_path = phase_cal_dir / f"input_{sample_i:02d}_{safe}.raw"
                # Random values: gaussian for fp32, small-magnitude integer for int
                if dtype == np.float32:
                    data = rng.standard_normal(size=dims).astype(np.float32) * 0.5
                else:
                    data = rng.integers(0, 100, size=dims).astype(dtype)
                data.tofile(str(raw_path))
                tokens.append(f"{inp_name}:={raw_path.absolute()}")
            f.write(" ".join(tokens) + "\n")
    return list_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tramp-dir", required=True,
                    help="vision_slices_v3/trampolines (output of extract_trampoline_phases.py)")
    ap.add_argument("--num-samples", type=int, default=10)
    args = ap.parse_args()

    tramp_dir = Path(args.tramp_dir)
    cal_dir = tramp_dir / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for onnx_file in sorted(tramp_dir.glob("dsp_seg_*_tramp_p*.onnx")):
        list_path = gen_one_phase(onnx_file, cal_dir, args.num_samples)
        n += 1
    print(f"Generated calibration data for {n} trampoline phases")
    print(f"  ({args.num_samples} samples each)")
    print(f"  Output: {cal_dir}")


if __name__ == "__main__":
    main()
