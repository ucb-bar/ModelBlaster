#!/usr/bin/env python3
"""E4 verify: bit-exact correctness of the new fused KernelSpec
reference impls vs the unfused chain on the host (ctypes loader).

For each new fused KernelSpec:
  1. Compile the fused reference_impl as a shared library.
  2. Compile the unfused single-op reference_impls as shared libraries.
  3. Generate random int8 inputs + plausible quantization params.
  4. Call the fused kernel; call the chain (op_a → tmp → op_b).
  5. Compare element-wise; require max_abs_err == 0.

This is the bit-exact verification oracle that any future
Bedrock-generated kernel must also pass — exactly the gate documented
in `artifacts/kernels/<pair>/measurement_report.md`.
"""

from __future__ import annotations

import ctypes
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.reference_kernels import (
    BATCHNORM2D_S8, BATCHNORM2D_SILU_S8, CONV2D_BATCHNORM2D_S8,
    CONV2D_S8, SILU_S8,
)
from pipeline.verify_kernel import host_compile


RNG = np.random.default_rng(42)


def _load(so_path, sym, argtypes, restype=None):
    lib = ctypes.CDLL(so_path)
    fn = getattr(lib, sym)
    fn.argtypes = argtypes
    fn.restype = restype
    return fn


def verify_batchnorm2d_silu_s8(workdir):
    """Run the fused BN+SiLU and compare to unfused BN→SiLU chain."""
    print("\n=== batchnorm2d_silu_s8 ===")

    # Compile the three kernels (fused, bn, silu).
    fused_so = host_compile(BATCHNORM2D_SILU_S8.reference_impl,
                              "fused_bn_silu", workdir)
    bn_so   = host_compile(BATCHNORM2D_S8.reference_impl,
                              "ref_bn", workdir)
    silu_so = host_compile(SILU_S8.reference_impl,
                              "ref_silu", workdir)

    i8p = ctypes.POINTER(ctypes.c_int8)
    fp  = ctypes.POINTER(ctypes.c_float)
    cint = ctypes.c_int
    cfloat = ctypes.c_float

    fused = _load(fused_so, "kernel_batchnorm2d_silu_s8", [
        i8p, fp, fp, i8p, cint, cint, cint, cint,
        cfloat, cfloat, cint, cint,
        cfloat, cfloat, cint, cint,
    ])
    bn = _load(bn_so, "kernel_batchnorm2d_s8", [
        i8p, fp, fp, i8p, cint, cint, cint, cint,
        cfloat, cfloat, cint, cint,
    ])
    silu = _load(silu_so, "kernel_silu_s8", [
        i8p, i8p, cint, cfloat, cfloat, cint, cint,
    ])

    all_pass = True
    for shape in BATCHNORM2D_SILU_S8.extra_shapes:
        N, C, H, W = shape["N"], shape["C"], shape["H"], shape["W"]
        n_el = N * C * H * W
        x = RNG.integers(-128, 128, size=n_el, dtype=np.int8)
        bn_scale = RNG.uniform(0.5, 1.5, size=C).astype(np.float32)
        bn_bias = RNG.uniform(-1.0, 1.0, size=C).astype(np.float32)
        # Plausible quantization params.
        bn_scale_in = 0.01
        bn_scale_out = 0.01
        silu_scale_in = 0.01
        silu_scale_out = 0.01
        bn_act_min, bn_act_max = -128, 127
        silu_act_min, silu_act_max = -128, 127

        y_fused = np.zeros_like(x)
        fused(
            x.ctypes.data_as(i8p),
            bn_scale.ctypes.data_as(fp),
            bn_bias.ctypes.data_as(fp),
            y_fused.ctypes.data_as(i8p),
            N, C, H, W,
            bn_scale_in, bn_scale_out, bn_act_min, bn_act_max,
            silu_scale_in, silu_scale_out, silu_act_min, silu_act_max,
        )

        # Unfused chain.
        y_bn = np.zeros_like(x)
        bn(
            x.ctypes.data_as(i8p),
            bn_scale.ctypes.data_as(fp),
            bn_bias.ctypes.data_as(fp),
            y_bn.ctypes.data_as(i8p),
            N, C, H, W,
            bn_scale_in, bn_scale_out, bn_act_min, bn_act_max,
        )
        y_chain = np.zeros_like(x)
        silu(
            y_bn.ctypes.data_as(i8p),
            y_chain.ctypes.data_as(i8p),
            n_el, silu_scale_in, silu_scale_out, silu_act_min, silu_act_max,
        )

        diff = np.abs(y_fused.astype(np.int32) - y_chain.astype(np.int32))
        max_err = int(diff.max())
        n_diff = int((diff > 0).sum())
        status = "PASS" if max_err == 0 else "FAIL"
        print(f"  shape={shape}  max_abs_err={max_err}  n_diff={n_diff}/{n_el}  {status}")
        if max_err != 0:
            all_pass = False
    return all_pass


def verify_conv2d_batchnorm2d_s8(workdir):
    """Run the fused conv+BN and compare to unfused conv→BN chain."""
    print("\n=== conv2d_batchnorm2d_s8 ===")

    fused_so = host_compile(CONV2D_BATCHNORM2D_S8.reference_impl,
                              "fused_conv_bn", workdir)
    conv_so = host_compile(CONV2D_S8.reference_impl,
                              "ref_conv", workdir)
    bn_so   = host_compile(BATCHNORM2D_S8.reference_impl,
                              "ref_bn2", workdir)

    i8p = ctypes.POINTER(ctypes.c_int8)
    i32p = ctypes.POINTER(ctypes.c_int32)
    fp  = ctypes.POINTER(ctypes.c_float)
    cint = ctypes.c_int
    cfloat = ctypes.c_float

    fused = _load(fused_so, "kernel_conv2d_batchnorm2d_s8", [
        i8p, i8p, i32p, fp, fp, i8p,
        cint, cint, cint, cint, cint,
        cint, cint, cint, cint, cint, cint,
        cint, cint, cint, cint, cint,
        cint, cint,
        cfloat, cfloat, cint, cint,
    ])
    conv = _load(conv_so, "kernel_conv2d_s8", [
        i8p, i8p, i32p, i8p,
        cint, cint, cint, cint, cint,
        cint, cint, cint, cint, cint, cint,
        cint, cint, cint,
        cint, cint, cint, cint,
    ])
    bn = _load(bn_so, "kernel_batchnorm2d_s8", [
        i8p, fp, fp, i8p, cint, cint, cint, cint,
        cfloat, cfloat, cint, cint,
    ])

    # Use a small subset of shapes to keep verify fast.
    SMALL_SHAPES = [
        s for s in CONV2D_BATCHNORM2D_S8.extra_shapes
        if (s["IH"] * s["IW"] * s["IC"]) <= 256
    ]
    if not SMALL_SHAPES:
        SMALL_SHAPES = CONV2D_BATCHNORM2D_S8.extra_shapes[-1:]

    all_pass = True
    for shape in SMALL_SHAPES:
        N, IC, IH, IW = shape["N"], shape["IC"], shape["IH"], shape["IW"]
        OC, KH, KW = shape["OC"], shape["KH"], shape["KW"]
        SH, SW, PH, PW = shape["SH"], shape["SW"], shape["PH"], shape["PW"]
        OH = (IH + 2*PH - KH) // SH + 1
        OW = (IW + 2*PW - KW) // SW + 1

        x = RNG.integers(-128, 128, size=N*IC*IH*IW, dtype=np.int8)
        w = RNG.integers(-128, 128, size=OC*IC*KH*KW, dtype=np.int8)
        b = RNG.integers(-1000, 1000, size=OC, dtype=np.int32)
        bn_scale = RNG.uniform(0.5, 1.5, size=OC).astype(np.float32)
        bn_bias = RNG.uniform(-1.0, 1.0, size=OC).astype(np.float32)

        # Quantization params.
        input_offset = 0
        filter_offset = 0
        conv_output_offset = 0
        conv_output_multiplier = 1073741824  # 2^30 (Q0.31, scale=0.5)
        conv_output_shift = 1
        conv_act_min, conv_act_max = -128, 127
        bn_scale_in = 0.01
        bn_scale_out = 0.01
        bn_act_min, bn_act_max = -128, 127

        y_fused = np.zeros(N*OC*OH*OW, dtype=np.int8)
        fused(
            x.ctypes.data_as(i8p), w.ctypes.data_as(i8p),
            b.ctypes.data_as(i32p),
            bn_scale.ctypes.data_as(fp), bn_bias.ctypes.data_as(fp),
            y_fused.ctypes.data_as(i8p),
            N, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW,
            input_offset, filter_offset, conv_output_offset,
            conv_output_multiplier, conv_output_shift,
            conv_act_min, conv_act_max,
            bn_scale_in, bn_scale_out, bn_act_min, bn_act_max,
        )

        # Unfused chain.
        y_conv = np.zeros(N*OC*OH*OW, dtype=np.int8)
        conv(
            x.ctypes.data_as(i8p), w.ctypes.data_as(i8p),
            b.ctypes.data_as(i32p), y_conv.ctypes.data_as(i8p),
            N, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW,
            input_offset, filter_offset, conv_output_offset,
            conv_output_multiplier, conv_output_shift,
            conv_act_min, conv_act_max,
        )
        y_chain = np.zeros_like(y_conv)
        bn(
            y_conv.ctypes.data_as(i8p),
            bn_scale.ctypes.data_as(fp), bn_bias.ctypes.data_as(fp),
            y_chain.ctypes.data_as(i8p),
            N, OC, OH, OW,
            bn_scale_in, bn_scale_out, bn_act_min, bn_act_max,
        )

        diff = np.abs(y_fused.astype(np.int32) - y_chain.astype(np.int32))
        max_err = int(diff.max())
        n_diff = int((diff > 0).sum())
        status = "PASS" if max_err == 0 else "FAIL"
        print(f"  shape={shape}  max_abs_err={max_err}  n_diff={n_diff}/{len(y_fused)}  {status}")
        if max_err != 0:
            all_pass = False
    return all_pass


def main():
    with tempfile.TemporaryDirectory(prefix="verify_kernels_") as workdir:
        ok_bn = verify_batchnorm2d_silu_s8(workdir)
        ok_conv = verify_conv2d_batchnorm2d_s8(workdir)

    print("\n" + "=" * 60)
    print(f"batchnorm2d_silu_s8:  {'PASS' if ok_bn else 'FAIL'}")
    print(f"conv2d_batchnorm2d_s8: {'PASS' if ok_conv else 'FAIL'}")
    print("=" * 60)
    return 0 if (ok_bn and ok_conv) else 1


if __name__ == "__main__":
    sys.exit(main())
