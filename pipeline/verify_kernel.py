"""Host-compile a candidate C kernel and numerically compare it to the
reference at the given shapes. Used by the LLM kernel-gen retry loop.

The candidate and the reference are each compiled into their own .so
(so symbols don't collide), loaded via ctypes, and called against the
same random inputs at every test shape. Returns (ok, diagnostic).
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np

from agents.pipeline.reference_kernels import KernelSpec


HOST_CC = os.environ.get("HOST_CC", "cc")


class CompileError(RuntimeError):
    pass


# Mirror the base includes that emit_kernels_c prepends on the cross-compile
# path, so candidates verified on the host can reference FLT_MAX, size_t, etc.
_HOST_PROLOGUE = (
    "#include <stddef.h>\n"
    "#include <float.h>\n"
    "#include <stdint.h>\n"
    "#include <math.h>\n"
)


def host_compile(c_source: str, label: str, workdir: str) -> str:
    """Compile `c_source` into a shared library; return the .so path."""
    src_path = os.path.join(workdir, f"{label}.c")
    so_path = os.path.join(workdir, f"{label}.so")
    with open(src_path, "w") as f:
        if "#include" not in c_source:
            f.write(_HOST_PROLOGUE)
        f.write(c_source)
    cmd = [HOST_CC, "-O2", "-fPIC", "-shared", "-Wno-unused-result",
           src_path, "-o", so_path, "-lm"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise CompileError(
            f"host compile failed for {label}:\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr:\n{proc.stderr}"
        )
    return so_path


def _load(so_path: str, op: str, spec: KernelSpec):
    lib = ctypes.CDLL(so_path)
    fn = getattr(lib, f"kernel_{op}")
    fn.argtypes = spec.argtypes_factory()
    fn.restype = None
    return fn


def _gen_inputs_linear(shape: dict, rng: np.random.Generator):
    M, K, N = shape["M"], shape["K"], shape["N"]
    inp = rng.standard_normal((M, K), dtype=np.float32)
    w = rng.standard_normal((N, K), dtype=np.float32)
    b = rng.standard_normal((N,), dtype=np.float32)
    return inp, w, b, np.zeros((M, N), dtype=np.float32)


def _gen_inputs_matmul(shape: dict, rng: np.random.Generator):
    M, K, N = shape["M"], shape["K"], shape["N"]
    A = rng.standard_normal((M, K), dtype=np.float32)
    B = rng.standard_normal((K, N), dtype=np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    return A, B, C


def _gen_inputs_matmul_ta(shape: dict, rng: np.random.Generator):
    M, K, N = shape["M"], shape["K"], shape["N"]
    A = rng.standard_normal((K, M), dtype=np.float32)  # stored (K, M)
    B = rng.standard_normal((K, N), dtype=np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    return A, B, C


def _gen_inputs_matmul_tb(shape: dict, rng: np.random.Generator):
    M, K, N = shape["M"], shape["K"], shape["N"]
    A = rng.standard_normal((M, K), dtype=np.float32)
    B = rng.standard_normal((N, K), dtype=np.float32)  # stored (N, K)
    C = np.zeros((M, N), dtype=np.float32)
    return A, B, C


def _gen_inputs_matmul_tatb(shape: dict, rng: np.random.Generator):
    M, K, N = shape["M"], shape["K"], shape["N"]
    A = rng.standard_normal((K, M), dtype=np.float32)  # stored (K, M)
    B = rng.standard_normal((N, K), dtype=np.float32)  # stored (N, K)
    C = np.zeros((M, N), dtype=np.float32)
    return A, B, C


def _gen_inputs_bmm(shape: dict, rng: np.random.Generator):
    batch, M, K, N = shape["batch"], shape["M"], shape["K"], shape["N"]
    A = rng.standard_normal((batch, M, K), dtype=np.float32)
    B = rng.standard_normal((batch, K, N), dtype=np.float32)
    C = np.zeros((batch, M, N), dtype=np.float32)
    return A, B, C


def _gen_inputs_matmul_f16(shape: dict, rng: np.random.Generator):
    M, K, N = shape["M"], shape["K"], shape["N"]
    A = rng.standard_normal((M, K), dtype=np.float32).astype(np.float16)
    B = rng.standard_normal((K, N), dtype=np.float32).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)
    return A, B, C


def _gen_inputs_matmul_ta_f16(shape: dict, rng: np.random.Generator):
    M, K, N = shape["M"], shape["K"], shape["N"]
    A = rng.standard_normal((K, M), dtype=np.float32).astype(np.float16)
    B = rng.standard_normal((K, N), dtype=np.float32).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)
    return A, B, C


def _gen_inputs_matmul_tb_f16(shape: dict, rng: np.random.Generator):
    M, K, N = shape["M"], shape["K"], shape["N"]
    A = rng.standard_normal((M, K), dtype=np.float32).astype(np.float16)
    B = rng.standard_normal((N, K), dtype=np.float32).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)
    return A, B, C


def _gen_inputs_matmul_tatb_f16(shape: dict, rng: np.random.Generator):
    M, K, N = shape["M"], shape["K"], shape["N"]
    A = rng.standard_normal((K, M), dtype=np.float32).astype(np.float16)
    B = rng.standard_normal((N, K), dtype=np.float32).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)
    return A, B, C


def _gen_inputs_bmm_f16(shape: dict, rng: np.random.Generator):
    batch, M, K, N = shape["batch"], shape["M"], shape["K"], shape["N"]
    A = rng.standard_normal((batch, M, K), dtype=np.float32).astype(np.float16)
    B = rng.standard_normal((batch, K, N), dtype=np.float32).astype(np.float16)
    C = np.zeros((batch, M, N), dtype=np.float16)
    return A, B, C


def _gen_inputs_relu(shape: dict, rng: np.random.Generator):
    n = shape["n"]
    inp = rng.standard_normal((n,), dtype=np.float32)
    return inp, np.zeros((n,), dtype=np.float32)


def _gen_inputs_elu(shape: dict, rng: np.random.Generator):
    n = shape["n"]
    # Modest range so expf doesn't get too far into the saturation tail.
    inp = rng.standard_normal((n,), dtype=np.float32) * 4.0
    return inp, np.zeros((n,), dtype=np.float32)


def _gen_inputs_pointwise(shape: dict, rng: np.random.Generator):
    """Default input gen for pointwise activations whose only knob is
    the element count `n`. Modest scaling keeps expf-style activations
    away from saturation tails (matches _gen_inputs_elu)."""
    n = shape["n"]
    inp = rng.standard_normal((n,), dtype=np.float32) * 4.0
    return inp, np.zeros((n,), dtype=np.float32)


def _gen_inputs_reduce(shape: dict, rng: np.random.Generator):
    """Reduction kernels — input is logically [outer, reduce, inner],
    flattened. Output is [outer, inner]. Float-output reductions
    (sum/mean/max/min/prod) use the standard-normal-distributed
    input; arg-reductions output int64 indices and need a separate
    output buffer dtype, but they share the float input."""
    outer, reduce, inner = shape["outer"], shape["reduce"], shape["inner"]
    inp = rng.standard_normal((outer * reduce * inner,), dtype=np.float32)
    out = np.zeros((outer * inner,), dtype=np.float32)
    return inp, out


def _gen_inputs_argreduce(shape: dict, rng: np.random.Generator):
    outer, reduce, inner = shape["outer"], shape["reduce"], shape["inner"]
    inp = rng.standard_normal((outer * reduce * inner,), dtype=np.float32)
    out = np.zeros((outer * inner,), dtype=np.int64)
    return inp, out


def _conv2d_oh_ow(shape: dict) -> tuple[int, int]:
    OH = (shape["IH"] + 2 * shape["PH"] - shape["KH"]) // shape["SH"] + 1
    OW = (shape["IW"] + 2 * shape["PW"] - shape["KW"]) // shape["SW"] + 1
    return OH, OW


def _gen_inputs_conv2d(shape: dict, rng: np.random.Generator):
    N, IC, IH, IW = shape["N"], shape["IC"], shape["IH"], shape["IW"]
    OC = shape["OC"]
    KH, KW = shape["KH"], shape["KW"]
    OH, OW = _conv2d_oh_ow(shape)
    inp = rng.standard_normal((N, IC, IH, IW), dtype=np.float32)
    w = rng.standard_normal((OC, IC, KH, KW), dtype=np.float32)
    b = rng.standard_normal((OC,), dtype=np.float32)
    out = np.zeros((N, OC, OH, OW), dtype=np.float32)
    return inp, w, b, out


def _gen_inputs_maxpool2d(shape: dict, rng: np.random.Generator):
    N, C, IH, IW = shape["N"], shape["C"], shape["IH"], shape["IW"]
    KH, KW, SH, SW = shape["KH"], shape["KW"], shape["SH"], shape["SW"]
    PH, PW = shape.get("PH", 0), shape.get("PW", 0)
    DH, DW = shape.get("DH", 1), shape.get("DW", 1)
    OH = (IH + 2*PH - DH*(KH-1) - 1) // SH + 1
    OW = (IW + 2*PW - DW*(KW-1) - 1) // SW + 1
    inp = rng.standard_normal((N, C, IH, IW), dtype=np.float32)
    out = np.zeros((N, C, OH, OW), dtype=np.float32)
    return inp, out


def _gen_inputs_add(shape: dict, rng: np.random.Generator):
    n = shape["n"]
    a = rng.standard_normal((n,), dtype=np.float32)
    b = rng.standard_normal((n,), dtype=np.float32)
    out = np.zeros((n,), dtype=np.float32)
    return a, b, out


def _gen_inputs_batchnorm2d(shape: dict, rng: np.random.Generator):
    N, C, H, W = shape["N"], shape["C"], shape["H"], shape["W"]
    inp = rng.standard_normal((N, C, H, W), dtype=np.float32)
    scale = rng.standard_normal((C,), dtype=np.float32)
    bias = rng.standard_normal((C,), dtype=np.float32)
    out = np.zeros((N, C, H, W), dtype=np.float32)
    return inp, scale, bias, out


def _gen_inputs_sigmoid(shape: dict, rng: np.random.Generator):
    n = shape["n"]
    # Keep magnitudes modest so expf doesn't overflow / saturate the compare.
    inp = rng.standard_normal((n,), dtype=np.float32) * 4.0
    out = np.zeros((n,), dtype=np.float32)
    return inp, out


def _gen_inputs_linear_s8(shape: dict, rng: np.random.Generator):
    M, K, N = shape["M"], shape["K"], shape["N"]
    inp = rng.integers(-128, 128, size=(M, K), dtype=np.int8)
    w = rng.integers(-128, 128, size=(N, K), dtype=np.int8)
    # Bias uses a moderate range so requantize doesn't saturate every output.
    b = rng.integers(-1024, 1024, size=(N,), dtype=np.int32)
    out = np.zeros((M, N), dtype=np.int8)
    return inp, w, b, out


def _gen_inputs_relu_s8(shape: dict, rng: np.random.Generator):
    n = shape["n"]
    inp = rng.integers(-128, 128, size=(n,), dtype=np.int8)
    out = np.zeros((n,), dtype=np.int8)
    return inp, out


def _gen_inputs_conv2d_s8(shape: dict, rng: np.random.Generator):
    N, IC, IH, IW = shape["N"], shape["IC"], shape["IH"], shape["IW"]
    OC = shape["OC"]
    KH, KW = shape["KH"], shape["KW"]
    OH, OW = _conv2d_oh_ow(shape)
    inp = rng.integers(-128, 128, size=(N, IC, IH, IW), dtype=np.int8)
    w = rng.integers(-128, 128, size=(OC, IC, KH, KW), dtype=np.int8)
    b = rng.integers(-1024, 1024, size=(OC,), dtype=np.int32)
    out = np.zeros((N, OC, OH, OW), dtype=np.int8)
    return inp, w, b, out


def _gen_inputs_maxpool2d_s8(shape: dict, rng: np.random.Generator):
    N, C, IH, IW = shape["N"], shape["C"], shape["IH"], shape["IW"]
    KH, KW, SH, SW = shape["KH"], shape["KW"], shape["SH"], shape["SW"]
    PH, PW = shape.get("PH", 0), shape.get("PW", 0)
    DH, DW = shape.get("DH", 1), shape.get("DW", 1)
    OH = (IH + 2*PH - DH*(KH-1) - 1) // SH + 1
    OW = (IW + 2*PW - DW*(KW-1) - 1) // SW + 1
    inp = rng.integers(-128, 128, size=(N, C, IH, IW), dtype=np.int8)
    out = np.zeros((N, C, OH, OW), dtype=np.int8)
    return inp, out


def _gen_inputs_add_s8(shape: dict, rng: np.random.Generator):
    n = shape["n"]
    a = rng.integers(-128, 128, size=(n,), dtype=np.int8)
    b = rng.integers(-128, 128, size=(n,), dtype=np.int8)
    out = np.zeros((n,), dtype=np.int8)
    return a, b, out


def _gen_inputs_batchnorm2d_s8(shape: dict, rng: np.random.Generator):
    N, C, H, W = shape["N"], shape["C"], shape["H"], shape["W"]
    inp = rng.integers(-128, 128, size=(N, C, H, W), dtype=np.int8)
    scale = rng.standard_normal((C,), dtype=np.float32) * 0.1
    bias = rng.standard_normal((C,), dtype=np.float32) * 0.5
    out = np.zeros((N, C, H, W), dtype=np.int8)
    return inp, scale, bias, out


def _gen_inputs_sigmoid_s8(shape: dict, rng: np.random.Generator):
    n = shape["n"]
    inp = rng.integers(-128, 128, size=(n,), dtype=np.int8)
    out = np.zeros((n,), dtype=np.int8)
    return inp, out


def _fp(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _i8p(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))


def _h16p(arr: np.ndarray):
    return arr.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))


def _i32p(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def _run_kernel(fn, op: str, shape: dict, inputs):
    if op == "linear":
        inp, w, b, out = inputs
        fn(_fp(inp), _fp(w), _fp(b), _fp(out),
           shape["M"], shape["K"], shape["N"])
        return out
    if op == "relu":
        inp, out = inputs
        fn(_fp(inp), _fp(out), shape["n"])
        return out
    if op == "elu":
        inp, out = inputs
        fn(_fp(inp), _fp(out), shape["n"], ctypes.c_float(1.0))
        return out
    if op == "conv2d":
        inp, w, b, out = inputs
        fn(_fp(inp), _fp(w), _fp(b), _fp(out),
           shape["N"], shape["IC"], shape["IH"], shape["IW"], shape["OC"],
           shape["KH"], shape["KW"], shape["SH"], shape["SW"],
           shape["PH"], shape["PW"])
        return out
    if op == "maxpool2d":
        inp, out = inputs
        fn(_fp(inp), _fp(out),
           shape["N"], shape["C"], shape["IH"], shape["IW"],
           shape["KH"], shape["KW"], shape["SH"], shape["SW"],
           shape.get("PH", 0), shape.get("PW", 0),
           shape.get("DH", 1), shape.get("DW", 1))
        return out
    if op == "add":
        a, b, out = inputs
        fn(_fp(a), _fp(b), _fp(out), shape["n"])
        return out
    if op == "batchnorm2d":
        inp, scale, bias, out = inputs
        fn(_fp(inp), _fp(scale), _fp(bias), _fp(out),
           shape["N"], shape["C"], shape["H"], shape["W"])
        return out
    if op == "sigmoid":
        inp, out = inputs
        fn(_fp(inp), _fp(out), shape["n"])
        return out
    # KernelBench Phase 2 activations.
    if op in ("tanh", "gelu", "gelu_exact", "selu",
              "hardsigmoid", "softplus", "softsign", "swish"):
        inp, out = inputs
        fn(_fp(inp), _fp(out), shape["n"])
        return out
    if op == "leaky_relu":
        inp, out = inputs
        fn(_fp(inp), _fp(out), shape["n"], ctypes.c_float(0.01))
        return out
    if op == "hardtanh":
        inp, out = inputs
        fn(_fp(inp), _fp(out), shape["n"],
           ctypes.c_float(-1.0), ctypes.c_float(1.0))
        return out
    # KernelBench Phase 2 reductions.
    if op in ("sum_dim", "mean_dim", "max_dim", "min_dim", "prod_dim"):
        inp, out = inputs
        fn(_fp(inp), _fp(out), shape["outer"], shape["reduce"], shape["inner"])
        return out
    if op in ("argmax_dim", "argmin_dim"):
        inp, out = inputs
        i64p = ctypes.POINTER(ctypes.c_int64)
        fn(_fp(inp), out.ctypes.data_as(i64p),
           shape["outer"], shape["reduce"], shape["inner"])
        return out
    if op in ("l1_norm", "l2_norm"):
        inp, out = inputs
        fn(_fp(inp), _fp(out), shape["outer"], shape["reduce"], shape["inner"])
        return out
    if op == "frobenius_norm":
        inp, out = inputs
        fn(_fp(inp), _fp(out), shape["n"])
        return out
    if op == "linear_s8":
        inp, w, b, out = inputs
        # Use a representative quant-param set: zero offsets (symmetric) and a
        # well-conditioned multiplier+shift in the muRISCV-NN range. Different
        # values are exercised at the model level when a real linear_s8 IR
        # invocation runs end-to-end.
        fn(_i8p(inp), _i8p(w), _i32p(b), _i8p(out),
           shape["M"], shape["K"], shape["N"],
           0, 0, 0,                        # input/filter/output offsets
           1 << 30, 8,                     # multiplier, shift (Q0.31 / 256)
           -128, 127)                      # full int8 range
        return out
    if op == "relu_s8":
        inp, out = inputs
        fn(_i8p(inp), _i8p(out), shape["n"])
        return out
    if op == "conv2d_s8":
        inp, w, b, out = inputs
        fn(_i8p(inp), _i8p(w), _i32p(b), _i8p(out),
           shape["N"], shape["IC"], shape["IH"], shape["IW"], shape["OC"],
           shape["KH"], shape["KW"], shape["SH"], shape["SW"],
           shape["PH"], shape["PW"],
           0, 0, 0,         # offsets
           1 << 30, 8,      # multiplier, shift
           -128, 127)
        return out
    if op == "maxpool2d_s8":
        inp, out = inputs
        fn(_i8p(inp), _i8p(out),
           shape["N"], shape["C"], shape["IH"], shape["IW"],
           shape["KH"], shape["KW"], shape["SH"], shape["SW"],
           shape.get("PH", 0), shape.get("PW", 0),
           shape.get("DH", 1), shape.get("DW", 1))
        return out
    if op == "add_s8":
        a, b, out = inputs
        fn(_i8p(a), _i8p(b), _i8p(out), shape["n"],
           ctypes.c_float(0.05), ctypes.c_float(0.07), ctypes.c_float(0.1),
           -128, 127)
        return out
    if op == "batchnorm2d_s8":
        inp, scale, bias, out = inputs
        fn(_i8p(inp), _fp(scale), _fp(bias), _i8p(out),
           shape["N"], shape["C"], shape["H"], shape["W"],
           ctypes.c_float(0.05), ctypes.c_float(0.1),
           -128, 127)
        return out
    if op == "sigmoid_s8":
        inp, out = inputs
        fn(_i8p(inp), _i8p(out), shape["n"],
           ctypes.c_float(0.05), ctypes.c_float(0.01), 0, 127)
        return out
    if op in ("matmul", "matmul_ta", "matmul_tb", "matmul_tatb"):
        A, B, C = inputs
        fn(_fp(A), _fp(B), _fp(C), shape["M"], shape["K"], shape["N"])
        return C
    if op == "bmm":
        A, B, C = inputs
        fn(_fp(A), _fp(B), _fp(C), shape["batch"], shape["M"], shape["K"], shape["N"])
        return C
    if op in ("matmul_f16", "matmul_ta_f16", "matmul_tb_f16", "matmul_tatb_f16"):
        A, B, C = inputs
        fn(_h16p(A), _h16p(B), _h16p(C), shape["M"], shape["K"], shape["N"])
        return C
    if op == "bmm_f16":
        A, B, C = inputs
        fn(_h16p(A), _h16p(B), _h16p(C), shape["batch"], shape["M"], shape["K"], shape["N"])
        return C
    raise NotImplementedError(op)


# Op kinds that produce integer outputs — verify must compare bit-exactly
# (no atol/rtol tolerance) since integer math is deterministic.
_INTEGER_OPS = {"linear_s8", "relu_s8", "conv2d_s8", "maxpool2d_s8",
                "add_s8", "batchnorm2d_s8", "sigmoid_s8"}


@dataclass
class VerifyResult:
    ok: bool
    message: str
    failing_shape: Optional[dict] = None
    max_abs_err: float = 0.0
    max_rel_err: float = 0.0
    # Per-op cycle counts as reported by the spike harness (only populated
    # for spike-harness verify; host-ctypes verify leaves it None). Used
    # by callers to compare cached algorithm options by speed.
    cycles_by_op: Optional[dict] = None


def verify(
    spec: KernelSpec,
    candidate_c: str,
    shapes: list[dict],
    *,
    n_trials: int = 3,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    seed: int = 0,
) -> VerifyResult:
    """Compile the candidate, run it alongside the reference at each shape, and
    return whether numerics match within tolerance.

    The reference is compiled fresh per call (cheap, ~50ms) so this stays a
    self-contained check.
    """
    if not shapes:
        return VerifyResult(False, "no shapes to test against")

    op = spec.op
    workdir = tempfile.mkdtemp(prefix=f"verify_{op}_")
    try:
        try:
            ref_so = host_compile(spec.reference_impl, f"ref_{op}", workdir)
        except CompileError as e:
            return VerifyResult(False, f"reference failed to compile: {e}")
        try:
            cand_so = host_compile(candidate_c, f"cand_{op}", workdir)
        except CompileError as e:
            return VerifyResult(False, f"candidate failed to compile:\n{e}")

        try:
            ref_fn = _load(ref_so, op, spec)
            cand_fn = _load(cand_so, op, spec)
        except (OSError, AttributeError) as e:
            return VerifyResult(False, f"failed to load symbol kernel_{op}: {e}")

        rng = np.random.default_rng(seed)
        worst_abs = 0.0
        worst_rel = 0.0
        for shape in shapes:
            for _ in range(n_trials):
                if op == "linear":
                    inputs_ref = _gen_inputs_linear(shape, rng)
                elif op == "relu":
                    inputs_ref = _gen_inputs_relu(shape, rng)
                elif op == "elu":
                    inputs_ref = _gen_inputs_elu(shape, rng)
                elif op == "conv2d":
                    inputs_ref = _gen_inputs_conv2d(shape, rng)
                elif op == "maxpool2d":
                    inputs_ref = _gen_inputs_maxpool2d(shape, rng)
                elif op == "add":
                    inputs_ref = _gen_inputs_add(shape, rng)
                elif op == "batchnorm2d":
                    inputs_ref = _gen_inputs_batchnorm2d(shape, rng)
                elif op == "sigmoid":
                    inputs_ref = _gen_inputs_sigmoid(shape, rng)
                elif op in ("leaky_relu", "tanh", "swish",
                            "gelu", "gelu_exact", "selu",
                            "hardsigmoid", "softplus", "softsign",
                            "hardtanh"):
                    inputs_ref = _gen_inputs_pointwise(shape, rng)
                elif op in ("sum_dim", "mean_dim", "max_dim",
                            "min_dim", "prod_dim"):
                    inputs_ref = _gen_inputs_reduce(shape, rng)
                elif op in ("argmax_dim", "argmin_dim"):
                    inputs_ref = _gen_inputs_argreduce(shape, rng)
                elif op in ("l1_norm", "l2_norm"):
                    # Same shape as a reduction but the output is the
                    # full input shape (broadcast division), so size
                    # the output buffer accordingly.
                    outer = shape["outer"]; reduce = shape["reduce"]; inner = shape["inner"]
                    inp = rng.standard_normal((outer * reduce * inner,),
                                              dtype=np.float32)
                    out = np.zeros((outer * reduce * inner,), dtype=np.float32)
                    inputs_ref = (inp, out)
                elif op == "frobenius_norm":
                    inputs_ref = _gen_inputs_pointwise(shape, rng)
                elif op == "linear_s8":
                    inputs_ref = _gen_inputs_linear_s8(shape, rng)
                elif op == "relu_s8":
                    inputs_ref = _gen_inputs_relu_s8(shape, rng)
                elif op == "conv2d_s8":
                    inputs_ref = _gen_inputs_conv2d_s8(shape, rng)
                elif op == "maxpool2d_s8":
                    inputs_ref = _gen_inputs_maxpool2d_s8(shape, rng)
                elif op == "add_s8":
                    inputs_ref = _gen_inputs_add_s8(shape, rng)
                elif op == "batchnorm2d_s8":
                    inputs_ref = _gen_inputs_batchnorm2d_s8(shape, rng)
                elif op == "sigmoid_s8":
                    inputs_ref = _gen_inputs_sigmoid_s8(shape, rng)
                elif op == "matmul":
                    inputs_ref = _gen_inputs_matmul(shape, rng)
                elif op == "matmul_ta":
                    inputs_ref = _gen_inputs_matmul_ta(shape, rng)
                elif op == "matmul_tb":
                    inputs_ref = _gen_inputs_matmul_tb(shape, rng)
                elif op == "matmul_tatb":
                    inputs_ref = _gen_inputs_matmul_tatb(shape, rng)
                elif op == "bmm":
                    inputs_ref = _gen_inputs_bmm(shape, rng)
                elif op == "matmul_f16":
                    inputs_ref = _gen_inputs_matmul_f16(shape, rng)
                elif op == "matmul_ta_f16":
                    inputs_ref = _gen_inputs_matmul_ta_f16(shape, rng)
                elif op == "matmul_tb_f16":
                    inputs_ref = _gen_inputs_matmul_tb_f16(shape, rng)
                elif op == "matmul_tatb_f16":
                    inputs_ref = _gen_inputs_matmul_tatb_f16(shape, rng)
                elif op == "bmm_f16":
                    inputs_ref = _gen_inputs_bmm_f16(shape, rng)
                else:
                    return VerifyResult(False, f"unsupported op {op}")
                inputs_cand = tuple(a.copy() for a in inputs_ref)

                try:
                    out_ref = _run_kernel(ref_fn, op, shape, inputs_ref)
                    out_cand = _run_kernel(cand_fn, op, shape, inputs_cand)
                except Exception as e:  # ctypes / segfault wrapper
                    return VerifyResult(
                        False,
                        f"candidate crashed on shape={shape}: {e}",
                        failing_shape=shape,
                    )

                # Cast to int64 to avoid wraparound when subtracting int8/int32.
                if op in _INTEGER_OPS:
                    abs_err = np.abs(out_ref.astype(np.int64) - out_cand.astype(np.int64))
                    denom = np.maximum(np.abs(out_ref.astype(np.int64)), 1)
                    # Bit-exact: tolerance is zero for integer ops.
                    op_atol = 0
                    op_rtol = 0.0
                else:
                    abs_err = np.abs(out_ref - out_cand)
                    denom = np.maximum(np.abs(out_ref), 1e-12)
                    op_atol = atol
                    op_rtol = rtol
                rel_err = abs_err / denom
                worst_abs = max(worst_abs, float(abs_err.max()))
                worst_rel = max(worst_rel, float(rel_err.max()))

                if not np.all(abs_err <= op_atol + op_rtol * np.abs(out_ref)):
                    # Find the index of the FIRST mismatched flat element so
                    # the retry prompt can point at exactly what went wrong.
                    bad_mask = abs_err > atol + rtol * np.abs(out_ref)
                    first_flat = int(np.argmax(bad_mask.flatten()))
                    first_idx = np.unravel_index(first_flat, out_ref.shape)
                    # Show a small window of ref/cand around the first mismatch
                    # so e.g. "channel oc=4 onward is wrong" is obvious.
                    flat_ref = out_ref.flatten()
                    flat_cand = out_cand.flatten()
                    win_start = max(0, first_flat - 2)
                    win_end = min(len(flat_ref), first_flat + 6)
                    ref_win = [round(float(v), 6) for v in flat_ref[win_start:win_end]]
                    cand_win = [round(float(v), 6) for v in flat_cand[win_start:win_end]]
                    diag = (
                        f"numerical mismatch on shape={shape}: "
                        f"max_abs_err={float(abs_err.max()):.3g} "
                        f"max_rel_err={float(rel_err.max()):.3g} "
                        f"(atol={atol:g} rtol={rtol:g})\n"
                        f"  first divergence at flat_idx={first_flat} "
                        f"(decoded shape index = {tuple(int(i) for i in first_idx)})\n"
                        f"  output[{win_start}:{win_end}] ref:  {ref_win}\n"
                        f"  output[{win_start}:{win_end}] cand: {cand_win}"
                    )
                    return VerifyResult(
                        False, diag, failing_shape=shape,
                        max_abs_err=worst_abs, max_rel_err=worst_rel,
                    )

        return VerifyResult(
            True,
            f"ok ({len(shapes)} shapes × {n_trials} trials, "
            f"max_abs_err={worst_abs:.3g} max_rel_err={worst_rel:.3g})",
            max_abs_err=worst_abs, max_rel_err=worst_rel,
        )
    finally:
        with contextlib.suppress(OSError):
            shutil.rmtree(workdir)
