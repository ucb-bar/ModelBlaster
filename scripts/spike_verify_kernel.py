#!/usr/bin/env python3
"""Cross-compile a candidate kernel to rv64gcv, run on spike (functional
RVV simulator), compare bit-exact against the reference impl. The
production verify path.

The test harness:
  1. Generate random inputs for each registered shape.
  2. Cross-compile a test binary that calls the kernel and writes the
     output to a known address.
  3. Run on spike with RVV 1.0 enabled.
  4. Same for the reference impl.
  5. Diff outputs element-wise; require max_abs_err == 0.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pipeline.reference_kernels as rk


RISCV_GCC = "/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc"
SPIKE = "/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike"
PK = "/scratch2/agustin/chipyard/.conda-env/riscv-tools/riscv64-unknown-elf/bin/pk"
# rv64gc + V (vector). Use lp64d ABI for fp.
ISA = "rv64gcv_zicsr_zifencei"
ABI = "lp64d"


def _check_tools():
    for tool in [RISCV_GCC, SPIKE, PK]:
        if not Path(tool).exists():
            print(f"WARN: {tool} not found")
    print(f"GCC : {RISCV_GCC}")
    print(f"spike: {SPIKE}")
    print(f"pk  : {PK}")


def _harness_main_bn_silu(input_bin, scale_bin, bias_bin,
                            n, c, h, w, out_path):
    """Generate the test main() for BN+SiLU verify on spike. The
    input/scale/bias are pre-baked as static arrays."""
    n_el = n * c * h * w
    return f"""
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

extern void kernel_batchnorm2d_silu_s8(
    const int8_t *input, const float *scale, const float *bias,
    int8_t *output,
    int N, int C, int H, int W,
    float bn_scale_in, float bn_scale_out,
    int bn_activation_min, int bn_activation_max,
    float silu_scale_in, float silu_scale_out,
    int silu_activation_min, int silu_activation_max);

static const int8_t INPUT[{n_el}] = {{ {input_bin} }};
static const float  SCALE[{c}]   = {{ {scale_bin} }};
static const float  BIAS [{c}]   = {{ {bias_bin}  }};
static int8_t OUT[{n_el}];

int main(void) {{
    kernel_batchnorm2d_silu_s8(
        INPUT, SCALE, BIAS, OUT,
        {n}, {c}, {h}, {w},
        0.01f, 0.01f, -128, 127,
        0.01f, 0.01f, -128, 127);
    /* Print output as hex bytes — spike's tohost path. */
    for (int i = 0; i < {n_el}; i++) {{
        printf("%02x\\n", (unsigned char)OUT[i]);
    }}
    return 0;
}}
"""


def _format_array(arr, dtype="i8"):
    """Format numpy array as a C initializer list."""
    if dtype == "i8":
        return ",".join(str(int(v)) for v in arr)
    elif dtype == "f32":
        return ",".join(f"{float(v):.9g}f" for v in arr)
    raise ValueError(dtype)


def _xc_and_run(kernel_c, harness_c, workdir, label):
    """Cross-compile {kernel + harness}, run on spike, return stdout."""
    src = Path(workdir) / f"{label}.c"
    elf = Path(workdir) / f"{label}.elf"
    src.write_text(
        "#include <stdint.h>\n#include <stddef.h>\n#include <math.h>\n"
        "#include <riscv_vector.h>\n"
        + kernel_c + "\n\n" + harness_c
    )
    cmd = [
        RISCV_GCC,
        "-march=" + ISA, "-mabi=" + ABI,
        "-O2", "-Wall",
        str(src), "-o", str(elf), "-lm",
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if cp.returncode != 0:
        return None, f"compile fail: {cp.stderr[-500:]}"
    cmd = [SPIKE, "--isa=" + ISA, PK, str(elf)]
    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if cp.returncode != 0:
        return None, f"spike fail rc={cp.returncode}: {cp.stderr[-300:]}"
    return cp.stdout, "ok"


def _parse_hex_output(output):
    """Parse `printf("%02x\\n", ...)` lines into a list of uint8."""
    vals = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            vals.append(int(line, 16))
        except ValueError:
            continue
    return np.array(vals, dtype=np.uint8).view(np.int8)


def verify_bn_silu(candidate_c, spec, workdir, shape):
    """Verify a single candidate against the reference on one shape."""
    N, C, H, W = shape["N"], shape["C"], shape["H"], shape["W"]
    n_el = N * C * H * W
    rng = np.random.default_rng(42)
    x = rng.integers(-128, 128, size=n_el, dtype=np.int8)
    s = rng.uniform(0.5, 1.5, size=C).astype(np.float32)
    b = rng.uniform(-1.0, 1.0, size=C).astype(np.float32)
    harness = _harness_main_bn_silu(
        _format_array(x, "i8"),
        _format_array(s, "f32"),
        _format_array(b, "f32"),
        N, C, H, W, "")
    # Candidate.
    out_cand, st_cand = _xc_and_run(candidate_c, harness, workdir, "cand")
    if out_cand is None:
        return {"shape": shape, "status": st_cand, "max_abs_err": None}
    y_cand = _parse_hex_output(out_cand)
    # Reference.
    out_ref, st_ref = _xc_and_run(spec.reference_impl, harness, workdir, "ref")
    if out_ref is None:
        return {"shape": shape, "status": st_ref, "max_abs_err": None}
    y_ref = _parse_hex_output(out_ref)
    if len(y_cand) != n_el or len(y_ref) != n_el:
        return {"shape": shape, "status": "output_size_mismatch",
                "max_abs_err": None,
                "n_cand": len(y_cand), "n_ref": len(y_ref)}
    diff = np.abs(y_cand.astype(np.int32) - y_ref.astype(np.int32))
    return {"shape": shape, "status": "ok",
            "max_abs_err": int(diff.max()),
            "n_diff": int((diff > 0).sum()), "n_el": n_el}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-c", required=True,
                    help="Candidate kernel .c file")
    ap.add_argument("--spec", required=True,
                    choices=["batchnorm2d_silu_s8"])
    ap.add_argument("--shapes", default=None,
                    help="comma-separated shape indices to test (default: first 2)")
    args = ap.parse_args()

    _check_tools()

    spec_map = {"batchnorm2d_silu_s8": rk.BATCHNORM2D_SILU_S8}
    spec = spec_map[args.spec]
    code = Path(args.kernel_c).read_text()
    print(f"\nCandidate: {args.kernel_c} ({len(code)} chars)")

    if args.shapes:
        shape_indices = [int(s) for s in args.shapes.split(",")]
    else:
        # Default: test the smallest 2 shapes for speed (each spike run
        # takes 30-60s).
        shape_indices = [4, 5]  # the small shapes at the tail
    shapes_to_test = [spec.extra_shapes[i] for i in shape_indices
                       if i < len(spec.extra_shapes)]

    with tempfile.TemporaryDirectory(prefix="spike_verify_") as workdir:
        all_pass = True
        for shape in shapes_to_test:
            print(f"\n--- shape {shape} ---")
            r = verify_bn_silu(code, spec, workdir, shape)
            print(f"  status={r['status']}  max_abs_err={r.get('max_abs_err')}")
            if r['status'] != "ok" or r.get('max_abs_err', 1) != 0:
                all_pass = False

    print("\n" + "=" * 50)
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
