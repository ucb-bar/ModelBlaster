"""int8 extraction with fp32 islands (``extract_graph --quant int8 --int8-mixed POLICY``).

The FX int8 extractor (``extract_int8``) lowers a model only when EVERY node has
an int8 lowering, and it runs the whole graph on per-tensor int8 grids. A
transformer like SmolVLA has neither property: its graphs are full of
constants (additive -inf masks, RoPE tables, position rows), broadcasts,
slices and residual streams whose outliers no per-tensor int8 scale holds,
and the numerically fragile parts (softmax inputs, norms, the flow-matching
time embedding, the action head) lose accuracy in int8 for no speed: the
time is in the linears (~75% of a SmolVLA chunk on the Deck).

This pass therefore starts from the fp32 FX lowering (every op the generic
fp32 kernels run, ``get_attr`` constants materialised as constant tensors,
``expand`` a zero-stride ``strided_copy``) and moves the policy's ops -- the
linears -- to int8:

* ``linear`` -> ``quantize_rows_s8`` + ``linear_dq_s8``. The activation is
  quantized per ROW (per token) at run time: ``quantize_rows_s8`` writes the
  int8 rows and each row's scale (max |x'| / 127 of that row). Before that,
  each input column is multiplied by a calibrated SmoothQuant factor
  (x' = x * col_inv[k], col_inv = 1 / sm) which the weight absorbs
  (W' = W * sm, so x' W'^T = x W^T), so an outlier channel no longer sets
  every row's grid: sm_k = max|x_k|^a / max|W_:,k|^(1-a) over the
  calibration set and over every int8 reader of x (q / k / v share one
  quantized tensor and one sm), a = ``alpha`` (default 0.5). The weight is
  int8 per output channel. ``linear_dq_s8`` accumulates in int32 and writes
  fp32: ``acc * row_scale[m] * w_scale[n]`` rounded once, + fp32 bias. So
  the residual stream, RoPE, attention, softmax and the norms stay fp32 and
  never see an int8 grid.
* ops named in ``keep_fp32`` (fnmatch patterns over IR op names) stay fp32.
* ``outliers`` ({pattern: n}): for the linears it names, the n input columns with the
  largest calibrated |x| leave the int8 product and run as an fp32 linear on an
  ``index_select`` of them, added to the int8 part (LLM.int8()-style decomposition).
* ``alpha_by`` ({pattern: a}): the smoothing strength of the tensors those linears read.
* fp32 fusions that cut launches (on by default; ``fuse_norm`` / ``fuse_rope`` / ``fuse_swiglu`` /
  ``fold_layout`` false to skip): SwiGLU's silu + mul -> ``silu_mul``, the RMSNorm chain -> ``rms_norm`` (its weight folded
  into the quantizer's column factor), RoPE -> ``rope_half``, and attention's permute
  copies folded into ``bmm_strided`` operands (``layout_fold``). ``fuse_attention`` (off:
  slower on the Deck) makes unmasked attention one flash-style ``attention`` op.
* constants stay the fp32 constant tensors of the fp32 lowering (no linear
  reads one in these graphs; one that did would be quantized like any
  activation, its row scales computed by quantize_rows_s8 on the device).
  Additive masks stay in the add-before-softmax fp32 domain: softmax is fp32,
  so exp(-inf) = 0 exactly as in the model (softmax_s8 would have to clip
  -inf onto an int8 grid, which is not softmax).

Calibration (``extract_int8``'s rule: representative inputs, max over the
set, ``MB_INT8_CALIB_POLICY=pNN`` for a pooled percentile) sets the
per-column maxima the smoothing factors come from; the row scales are
measured at run time, so an input unlike the calibration set cannot saturate
a row.

The resulting IR has ``quant: int8_mixed``: f32 and i8 tensors side by side.
Its golden output is the IR walked on the CPU: fp32 ops as ``walk_torch``
computes them, the int8 kinds bit-exactly as their reference C
(``TORCH_RULES`` reproduce it with exact float64 / IEEE fp32 steps; the tests
compare them with the compiled reference_impl). Walkers hold the int8
kernels to bit-exactness against the reference C and the fp32 ops to the
fp32 per-slot bound (walk_vulkan.evaluate_int8_mixed).
"""
from __future__ import annotations

import ctypes
import fnmatch
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

MIXED = "int8_mixed"
QMAX = 127.0
POLICIES = {
    # every linear -> quantize_rows_s8 + linear_dq_s8; everything else fp32
    "linear": {"int8_kinds": ["linear"], "alpha": 0.5},
}
KINDS = ("quantize_rows_s8", "linear_dq_s8", "linear3_dq_s8", "linear_dq_add_s8")

# ------------------------------------------------------------------ kernel specs
_SPECS = [
    ("quantize_rows_s8",
     "void kernel_quantize_rows_s8(const float *input, const float *col_inv, int8_t *output, "
     "float *row_scale, int M, int K)",
     "Dynamic per-row int8 quantization of an fp32 [M, K] tensor, after a per-column factor\n"
     "(SmoothQuant smoothing; all ones for none):\n"
     "  x'[m, k]     = input[m, k] * col_inv[k]                    (fp32)\n"
     "  amax         = max_k |x'[m, k]|                            (NaN ignored)\n"
     "  row_scale[m] = amax > 0 ? amax / 127.0f : 1.0f             (fp32 division)\n"
     "  inv          = 1.0f / row_scale[m]\n"
     "  output[m, k] = clamp(roundf(x'[m, k] * inv), -127, 127)    (roundf: half away from zero;\n"
     "                                                             NaN -> -127)\n"
     "Inserted by the int8_mixed pass (pipeline/int8_mixed.py) where a linear_dq_s8 reads an\n"
     "fp32 tensor; the row scales are linear_dq_s8's.",
     "void kernel_quantize_rows_s8(const float *input, const float *col_inv, int8_t *output,\n"
     "                             float *row_scale, int M, int K) {\n"
     "    for (int m = 0; m < M; m++) {\n"
     "        const float *x = input + (size_t)m * K;\n"
     "        float amax = 0.0f;\n"
     "        for (int k = 0; k < K; k++) {\n"
     "            float v = fabsf(x[k] * col_inv[k]);\n"
     "            if (v > amax) amax = v;\n"
     "        }\n"
     "        const float s = amax > 0.0f ? amax / 127.0f : 1.0f;\n"
     "        const float inv = 1.0f / s;\n"
     "        row_scale[m] = s;\n"
     "        for (int k = 0; k < K; k++) {\n"
     "            const float v = x[k] * col_inv[k];\n"
     "            float q = roundf(v * inv);\n"
     "            if (!(q >= -127.0f)) q = -127.0f;\n"
     "            if (q > 127.0f) q = 127.0f;\n"
     "            output[(size_t)m * K + k] = (int8_t)q;\n"
     "        }\n"
     "    }\n}\n",
     [{"M": 1, "K": 16}, {"M": 5, "K": 257}]),
    ("linear_dq_s8",
     "void kernel_linear_dq_s8(const int8_t *input, const float *row_scale, const int8_t *weight, "
     "const float *bias, const float *w_scale, float *output, int M, int K, int N)",
     "int8 x int8 fully-connected layer with an fp32 (dequantized) output:\n"
     "  input int8 [M, K] with per-row scales row_scale[M] (quantize_rows_s8), weight int8 [N, K]\n"
     "  with per-output-channel scales w_scale[N], bias fp32 [N] or NULL, output fp32 [M, N]\n"
     "  acc = sum_k input[m, k] * weight[n, k]                       (int32, exact)\n"
     "  y   = (float)((double)acc * ((double)row_scale[m] * (double)w_scale[n]))\n"
     "  output[m, n] = bias ? y + bias[n] : y                         (fp32 add)\n"
     "The scale product is exact in double; the dequantized value is rounded once to fp32.",
     "void kernel_linear_dq_s8(const int8_t *input, const float *row_scale, const int8_t *weight,\n"
     "                         const float *bias, const float *w_scale, float *output, int M, int K, int N) {\n"
     "    for (int m = 0; m < M; m++) {\n"
     "        for (int n = 0; n < N; n++) {\n"
     "            int32_t acc = 0;\n"
     "            for (int k = 0; k < K; k++)\n"
     "                acc += (int32_t)input[(size_t)m * K + k] * (int32_t)weight[(size_t)n * K + k];\n"
     "            const double s = (double)row_scale[m] * (double)w_scale[n];\n"
     "            float y = (float)((double)acc * s);\n"
     "            if (bias) y = y + bias[n];\n"
     "            output[(size_t)m * N + n] = y;\n"
     "        }\n"
     "    }\n}\n",
     [{"M": 1, "K": 16, "N": 8}, {"M": 5, "K": 36, "N": 9}]),
    ("linear3_dq_s8",
     "void kernel_linear3_dq_s8(const int8_t *input, const float *row_scale, const int8_t *weight, "
     "const float *bias, const float *w_scale, float *out0, float *out1, float *out2, int M, int K, "
     "int N0, int N1, int N2)",
     "Sibling linear_dq_s8 ops that read the same quantized input, as one product (the int8_mixed\n"
     "pass's merge_siblings: q / k / v, gate / up): weight int8 [N0+N1+N2, K] (the siblings' weights\n"
     "stacked), w_scale / bias [N0+N1+N2] (bias NULL or stacked), and the output columns split back:\n"
     "  y[m, n] = linear_dq_s8(input, row_scale, weight, bias, w_scale)[m, n]   (exactly as there)\n"
     "  out0[m, n] = y[m, n] (n < N0);  out1[m, n - N0] = y[m, n] (N0 <= n < N0+N1);\n"
     "  out2[m, n - N0 - N1] = y[m, n] (the rest; out2 NULL when N2 == 0)",
     "void kernel_linear3_dq_s8(const int8_t *input, const float *row_scale, const int8_t *weight,\n"
     "                          const float *bias, const float *w_scale, float *out0, float *out1, float *out2,\n"
     "                          int M, int K, int N0, int N1, int N2) {\n"
     "    const int N = N0 + N1 + N2;\n"
     "    for (int m = 0; m < M; m++) {\n"
     "        for (int n = 0; n < N; n++) {\n"
     "            int32_t acc = 0;\n"
     "            for (int k = 0; k < K; k++)\n"
     "                acc += (int32_t)input[(size_t)m * K + k] * (int32_t)weight[(size_t)n * K + k];\n"
     "            const double s = (double)row_scale[m] * (double)w_scale[n];\n"
     "            float y = (float)((double)acc * s);\n"
     "            if (bias) y = y + bias[n];\n"
     "            if (n < N0) out0[(size_t)m * N0 + n] = y;\n"
     "            else if (n < N0 + N1) out1[(size_t)m * N1 + (n - N0)] = y;\n"
     "            else out2[(size_t)m * N2 + (n - N0 - N1)] = y;\n"
     "        }\n"
     "    }\n}\n",
     [{"M": 1, "K": 16, "N0": 8, "N1": 4, "N2": 0}, {"M": 5, "K": 36, "N0": 9, "N1": 3, "N2": 6}]),
    ("linear_dq_add_s8",
     "void kernel_linear_dq_add_s8(const int8_t *input, const float *row_scale, const int8_t *weight, "
     "const float *bias, const float *w_scale, const float *addend, float *output, int M, int K, int N)",
     "linear_dq_s8 followed by the residual add that reads it (the int8_mixed pass's fuse_residual):\n"
     "  output[m, n] = addend[m, n] + linear_dq_s8(input, row_scale, weight, bias, w_scale)[m, n]\n"
     "(the linear's fp32 result exactly as linear_dq_s8 rounds it, then one fp32 add).",
     "void kernel_linear_dq_add_s8(const int8_t *input, const float *row_scale, const int8_t *weight,\n"
     "                             const float *bias, const float *w_scale, const float *addend, float *output,\n"
     "                             int M, int K, int N) {\n"
     "    for (int m = 0; m < M; m++) {\n"
     "        for (int n = 0; n < N; n++) {\n"
     "            int32_t acc = 0;\n"
     "            for (int k = 0; k < K; k++)\n"
     "                acc += (int32_t)input[(size_t)m * K + k] * (int32_t)weight[(size_t)n * K + k];\n"
     "            const double s = (double)row_scale[m] * (double)w_scale[n];\n"
     "            float y = (float)((double)acc * s);\n"
     "            if (bias) y = y + bias[n];\n"
     "            output[(size_t)m * N + n] = addend[(size_t)m * N + n] + y;\n"
     "        }\n"
     "    }\n}\n",
     [{"M": 1, "K": 16, "N": 8}, {"M": 5, "K": 36, "N": 9}]),
]


_ROPE = ("rope_half",
         "void kernel_rope_half(const float *input, const float *cos_t, const float *sin_t, float *output, "
         "int R, int H, int half)",
         "Rotary position embedding, non-interleaved halves (lerobot / Llama apply_rope), fp32:\n"
         "  input / output [R, H, 2*half] (R positions, H heads), cos_t / sin_t [R, half]\n"
         "  x1 = input[r, h, :half], x2 = input[r, h, half:]\n"
         "  output[r, h, :half] = x1 * cos - x2 * sin;  output[r, h, half:] = x2 * cos + x1 * sin\n"
         "(each product and sum rounded to fp32). The int8_mixed pass fuses the generic lowering's\n"
         "two slices, four multiplies, sub, add and cat into it.",
         "void kernel_rope_half(const float *input, const float *cos_t, const float *sin_t, float *output,\n"
         "                      int R, int H, int half) {\n"
         "    for (int r = 0; r < R; r++)\n"
         "        for (int h = 0; h < H; h++)\n"
         "            for (int j = 0; j < half; j++) {\n"
         "                const float *x = input + ((size_t)r * H + h) * 2 * half;\n"
         "                float *y = output + ((size_t)r * H + h) * 2 * half;\n"
         "                const float c = cos_t[(size_t)r * half + j], s = sin_t[(size_t)r * half + j];\n"
         "                const float a = x[j] * c, b = x[half + j] * s, d = x[half + j] * c, e = x[j] * s;\n"
         "                y[j] = a - b;\n"
         "                y[half + j] = d + e;\n"
         "            }\n}\n",
         [{"R": 3, "H": 2, "half": 4}])


_SILU_MUL = ("silu_mul",
             "void kernel_silu_mul(const float *a, const float *b, float *output, int n)",
             "SwiGLU's gate: output[i] = silu(a[i]) * b[i], silu(x) = x / (1 + exp(-x)), fp32 (each step\n"
             "rounded to fp32). The int8_mixed pass fuses the generic lowering's silu (ewise1) and the mul\n"
             "that reads it into it (fuse_swiglu).",
             "void kernel_silu_mul(const float *a, const float *b, float *output, int n) {\n"
             "    for (int i = 0; i < n; i++) {\n"
             "        const float x = a[i];\n"
             "        const float s = x / (1.0f + expf(-x));\n"
             "        output[i] = s * b[i];\n"
             "    }\n}\n",
             [{"n": 37}])


def kernel_specs(kernel_spec_cls) -> dict:
    """KernelSpecs of the int8_mixed kinds (reference_kernels registers them)."""
    i8p, fp = ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_float)
    argt = {"quantize_rows_s8": [fp, fp, i8p, fp, ctypes.c_int, ctypes.c_int],
            "linear_dq_s8": [i8p, fp, i8p, fp, fp, fp, ctypes.c_int, ctypes.c_int, ctypes.c_int],
            "linear3_dq_s8": [i8p, fp, i8p, fp, fp, fp, fp, fp] + [ctypes.c_int] * 5,
            "linear_dq_add_s8": [i8p, fp, i8p, fp, fp, fp, fp, ctypes.c_int, ctypes.c_int, ctypes.c_int]}
    argt["rope_half"] = [fp, fp, fp, fp, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    argt["silu_mul"] = [fp, fp, fp, ctypes.c_int]
    return {op: kernel_spec_cls(op=op, signature=sig, semantics=sem, reference_impl=impl, extra_shapes=shapes,
                                argtypes_factory=(lambda a=argt[op]: list(a)))
            for op, sig, sem, impl, shapes in _SPECS + [_ROPE, _SILU_MUL]}


# ------------------------------------------------------------------ the int8 kinds on the CPU (bit-exact)
def _round_half_away(v: torch.Tensor) -> torch.Tensor:
    """C roundf / round: half away from zero (v - trunc(v) is exact)."""
    t = torch.trunc(v)
    return torch.where((v - t).abs() >= 0.5, t + torch.sign(v), t)


def quantize_rows(x: torch.Tensor, col_inv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """kernel_quantize_rows_s8 on [M, K] (IEEE fp32 steps in the reference's order)."""
    v = x.float() * col_inv.float()
    a = v.abs()
    a = torch.where(torch.isnan(a), torch.zeros_like(a), a).amax(1)      # (NaN never wins `v > amax`)
    s = torch.where(a > 0, a / torch.tensor(127.0), torch.ones_like(a))
    inv = torch.tensor(1.0) / s
    q = _round_half_away(v * inv[:, None])
    q = torch.where(q >= -QMAX, q, torch.full_like(q, -QMAX)).clamp(max=QMAX)
    return q.to(torch.int8), s


def _r_quantize_rows(ir, op, ins):
    sh = op["shape"]
    q, s = quantize_rows(ins[0].reshape(sh["M"], sh["K"]), ins[1].reshape(-1))
    return [q, s]


def _r_linear_dq(ir, op, ins):
    sh = op["shape"]
    x = ins[0].reshape(sh["M"], sh["K"]).double()
    w = torch.from_numpy(np.asarray(ir.weights[op["weight"]])).double()
    acc = x @ w.T                                   # integer-valued, |partial sums| < 2^53: exact
    ws = torch.from_numpy(np.asarray(ir.weights[op["w_scale"]], np.float32)).double()
    s = ins[1].reshape(-1, 1).double() * ws[None, :]                 # exact in double (24 x 24 bits)
    y = (acc * s).float()
    if op.get("bias"):
        y = y + torch.from_numpy(np.asarray(ir.weights[op["bias"]], np.float32))
    return [y]


def _r_linear3_dq(ir, op, ins):
    sh = op["shape"]
    (y,) = _r_linear_dq(ir, {**op, "shape": {"M": sh["M"], "K": sh["K"], "N": sh["N0"] + sh["N1"] + sh["N2"]}}, ins)
    outs = [y[:, :sh["N0"]], y[:, sh["N0"]:sh["N0"] + sh["N1"]], y[:, sh["N0"] + sh["N1"]:]]
    return [o.contiguous() for o in outs[:len(op["outputs"])]]


def _r_linear_dq_add(ir, op, ins):
    (y,) = _r_linear_dq(ir, op, ins[:2])
    return [ins[2].reshape(y.shape).float() + y]


def _r_silu_mul(ir, op, ins):
    return [torch.nn.functional.silu(ins[0].float()) * ins[1].reshape(ins[0].shape).float()]


def _r_rope_half(ir, op, ins):
    sh = op["shape"]
    R, H, d = sh["R"], sh["H"], sh["half"]
    x = ins[0].reshape(R, H, 2 * d).float()
    c, s_ = ins[1].reshape(R, 1, d).float(), ins[2].reshape(R, 1, d).float()
    x1, x2 = x[..., :d], x[..., d:]
    return [torch.cat([x1 * c - x2 * s_, x2 * c + x1 * s_], dim=-1)]


TORCH_RULES = {"quantize_rows_s8": _r_quantize_rows, "linear_dq_s8": _r_linear_dq, "linear3_dq_s8": _r_linear3_dq,
               "linear_dq_add_s8": _r_linear_dq_add, "rope_half": _r_rope_half, "silu_mul": _r_silu_mul}


# ------------------------------------------------------------------ IR walk on the CPU
def _walk_torch():
    h = str(Path(__file__).resolve().parents[1] / "gpu_eval" / "harness")
    if h not in sys.path:
        sys.path.insert(0, h)
    import walk_torch  # noqa: PLC0415
    return walk_torch


def as_ir(graph: dict, weights: dict, golden_in=None, golden_out=None):
    _walk_torch()
    from mbir import IR  # noqa: PLC0415
    e = np.zeros(0, np.float32)
    return IR(graph["name"], graph, weights, e if golden_in is None else golden_in,
              e if golden_out is None else golden_out)


def walk(ir, inputs: dict, record=None, keep=None) -> dict:
    """The IR on the CPU: fp32 ops by walk_torch's rules, the int8 kinds bit-exactly
    (TORCH_RULES; other `_s8` kinds are not supported here). inputs: {name: tensor}.
    record(name, value) sees every produced tensor; keep: names kept in the returned
    env (default: the graph outputs), everything else leaves at its last use."""
    wt = _walk_torch()
    env = {k: (v if torch.is_tensor(v) else torch.from_numpy(np.asarray(v))).reshape(ir.tshape(k))
           for k, v in inputs.items()}
    env.update(ir.const_tensors("cpu"))
    keep = set(keep or ()) | set(ir.output_names())
    last = {}
    for k, op in enumerate(ir.ops):
        for t in op["inputs"]:
            last[t] = k
    with torch.no_grad():
        for k, op in enumerate(ir.ops):
            ins = [env[t] for t in op["inputs"]]
            rule = TORCH_RULES.get(op["op"])
            if rule is not None:
                outs = rule(ir, op, ins)
            else:
                fn = wt._rule(op["op"])
                if fn is None or op["op"].endswith("_s8"):
                    raise NotImplementedError(f"int8_mixed walk: no CPU rule for {op['op']}")
                outs = fn(ir, op, ins, "cpu")
            for name, val in zip(op["outputs"], outs):
                env[name] = val.reshape(ir.tshape(name))
                if record is not None:
                    record(name, env[name])
            for t in op["inputs"]:
                if t not in keep and last.get(t, -1) <= k and not ir.graph["tensors"].get(t, {}).get("const"):
                    env.pop(t, None)
    return {k: v for k, v in env.items() if k in keep}


# ------------------------------------------------------------------ the pass
def _calib_col_ranges(ir, samples: list[dict], names: dict) -> dict:
    """Per-column max |x| (finite values) of each tensor in `names` ({tensor: K}) over all the
    calibration samples, or a pooled per-column percentile under MB_INT8_CALIB_POLICY=pNN."""
    pol = os.environ.get("MB_INT8_CALIB_POLICY", "max")
    if pol != "max" and not (pol.startswith("p") and pol[1:].replace(".", "", 1).isdigit()):
        raise SystemExit(f"MB_INT8_CALIB_POLICY={pol!r}: expected 'max' or 'pNN[.N]'")
    q = None if pol == "max" else float(pol[1:]) / 100.0
    rng, pool = {}, {}

    def rec(name, v):
        if name not in names or not v.is_floating_point():
            return
        a = v.detach().abs().reshape(-1, names[name]).double()
        a = torch.where(torch.isfinite(a), a, torch.zeros_like(a))
        m = a.amax(0)
        rng[name] = m if name not in rng else torch.maximum(rng[name], m)
        if q is not None:
            pool.setdefault(name, []).append(a)

    for s in samples:
        for k in set(names) & set(s):              # graph inputs
            rec(k, torch.as_tensor(np.asarray(s[k])))
        walk(ir, s, record=rec)
    if q is not None:
        for name, chunks in pool.items():
            rng[name] = torch.minimum(rng[name], torch.quantile(torch.cat(chunks), q, dim=0))
    return rng


def resolve_policy(policy) -> dict:
    """A policy name (POLICIES), a JSON string or a dict ({"base": name, "alpha": a,
    "keep_fp32": [patterns]}) -> the full policy dict."""
    if isinstance(policy, str):
        policy = policy.strip()
        if policy.startswith("{"):
            policy = json.loads(policy)
        elif policy in POLICIES:
            policy = {"base": policy}
        else:
            raise ValueError(f"int8_mixed policy {policy!r}: expected one of {sorted(POLICIES)} or a JSON dict")
    base = policy.get("base", "linear")
    if base not in POLICIES:
        raise ValueError(f"int8_mixed policy base {base!r}: expected one of {sorted(POLICIES)}")
    p = dict(POLICIES[base])
    p.update(policy)
    p["base"] = base
    p.setdefault("keep_fp32", [])
    return p


def quantize_graph(ir, samples: list[dict], policy="linear") -> tuple[dict, dict, dict]:
    """fp32 IR (mbir.IR) + calibration samples ({input name: tensor}) -> (graph, weights, report)
    of the int8_mixed IR. The fp32 IR is not modified."""
    pol = resolve_policy(policy)
    alpha = float(pol["alpha"])
    keep = [str(p) for p in pol["keep_fp32"]]
    sel = [op for op in ir.ops if op["op"] in pol["int8_kinds"]
           and not any(fnmatch.fnmatchcase(op["name"], p) for p in keep)]
    for op in sel:
        if op["op"] != "linear":
            raise NotImplementedError(f"int8_mixed: no int8 lowering for {op['op']}")
    tensors = {k: dict(v) for k, v in ir.graph["tensors"].items()}
    weights = dict(ir.weights)
    readers: dict[str, list] = {}
    for op in sel:
        readers.setdefault(op["inputs"][0], []).append(op)
    cols = _calib_col_ranges(ir, samples, {x: rd[0]["shape"]["K"] for x, rd in readers.items()})

    # SmoothQuant factors, one per quantized tensor (shared by all of its int8 readers)
    col_inv, outl = {}, {}
    for x, rd in readers.items():
        K = rd[0]["shape"]["K"]
        wmax = np.zeros(K)
        for op in rd:
            wmax = np.maximum(wmax, np.abs(np.asarray(ir.weights[op["weight"]], np.float64)).max(axis=0))
        xmax = cols[x].numpy() if x in cols else np.ones(K)
        a = alpha
        for pat, v in pol.get("alpha_by", {}).items():
            if any(fnmatch.fnmatchcase(op["name"], pat) for op in rd):
                a = float(v)
        sm = np.ones(K)
        if a > 0:
            sm = np.maximum(xmax, 1e-5) ** a / np.maximum(wmax, 1e-5) ** (1.0 - a)
            sm = np.where(np.isfinite(sm) & (sm > 0), sm, 1.0)
        col_inv[x] = (1.0 / sm).astype(np.float32)
        # outlier columns (LLM.int8()-style decomposition): the n columns with the largest calibrated
        # |x| leave the int8 product (col_inv 0, their weight columns 0) and run as a small fp32 linear
        n_out = max([int(v) for op in rd for pat, v in pol.get("outliers", {}).items()
                     if fnmatch.fnmatchcase(op["name"], pat)] or [0])
        if n_out > 0:
            cols_x = np.sort(np.argsort(-xmax, kind="stable")[:min(n_out, K)])
            outl[x] = cols_x
            col_inv[x][cols_x] = 0.0

    sel_ids = {id(op) for op in sel}
    qname: dict[str, tuple[str, str]] = {}
    ops = []
    for op in ir.ops:
        if id(op) not in sel_ids:
            ops.append(op)
            continue
        x = op["inputs"][0]
        sh = op["shape"]
        M, K = int(sh["M"]), int(sh["K"])
        if x not in qname:
            q, s, ci = f"{x}__q8", f"{x}__q8_scale", f"{x}__col_inv"
            qname[x] = (q, s)
            weights[f"const.{ci}"] = col_inv[x]
            tensors[ci] = {"shape": [K], "dtype": "f32", "quant": None, "const": f"const.{ci}"}
            tensors[q] = {"shape": [M, K], "dtype": "i8", "quant": {"scale": "per_row", "zero_point": 0}}
            tensors[s] = {"shape": [M], "dtype": "f32", "quant": None}
            ops.append({"name": q, "op": "quantize_rows_s8", "inputs": [x, ci], "outputs": [q, s],
                        "shape": {"M": M, "K": K}})
        q, s = qname[x]
        if x in outl and f"{x}__outl" not in tensors:
            cols_x, xo, xi = outl[x], f"{x}__outl", f"{x}__outl_idx"
            weights[f"const.{xi}"] = cols_x.astype(np.float32)
            tensors[xi] = {"shape": [len(cols_x)], "dtype": "f32", "quant": None, "const": f"const.{xi}"}
            tensors[xo] = {"shape": [M, len(cols_x)], "dtype": "f32", "quant": None}
            ops.append({"name": xo, "op": "index_select", "inputs": [x, xi], "outputs": [xo],
                        "shape": {"outer": M, "A": K, "J": len(cols_x), "inner": 1}})
        # W' = W * sm, with sm = 1 / col_inv as stored (so x' W'^T = x W^T up to the fp32 x');
        # outlier columns (col_inv 0) get zero weight columns here and their own fp32 linear below
        ci64 = col_inv[x].astype(np.float64)
        w = np.asarray(ir.weights[op["weight"]], np.float64) / np.where(ci64 > 0, ci64, np.inf)[None, :]
        ws = (np.maximum(np.abs(w).max(axis=1), 1e-30) / QMAX).astype(np.float32)
        wq = _round_half_away(torch.from_numpy(w / ws.astype(np.float64)[:, None]))
        wkey, skey = f"{op['weight']}.q8", f"{op['weight']}.wscale"
        weights[wkey] = wq.clamp(-QMAX, QMAX).to(torch.int8).numpy()
        weights[skey] = ws
        new = {k: v for k, v in op.items() if k not in ("dispatch_id", "depends_on", "hardware_target")}
        new.update(op="linear_dq_s8", inputs=[q, s], weight=wkey, w_scale=skey)
        if x in outl:
            y, N = op["outputs"][0], int(sh["N"])
            yq, yo, okey = f"{y}__i8part", f"{y}__outlpart", f"{op['weight']}.outl"
            weights[okey] = np.ascontiguousarray(np.asarray(ir.weights[op["weight"]], np.float32)[:, outl[x]])
            for t in (yq, yo):
                tensors[t] = {"shape": list(tensors[y]["shape"]), "dtype": "f32", "quant": None}
            new["outputs"] = [yq]
            ops.append(new)
            ops.append({"name": yo, "op": "linear", "inputs": [f"{x}__outl"], "outputs": [yo], "weight": okey,
                        "bias": None, "shape": {"M": M, "K": len(outl[x]), "N": N}})
            ops.append({"name": f"{y}__sum", "op": "add", "inputs": [yq, yo], "outputs": [y],
                        "shape": {"n": M * N}})
        else:
            ops.append(new)
    if pol.get("merge_siblings", True):
        ops = _merge_siblings(ops, tensors, weights, int(pol.get("merge_max_rows", 1 << 30)))
    if pol.get("fuse_residual", True):
        ops = _fuse_residual(ops, tensors, set(ir.output_names()))
    if pol.get("fuse_swiglu", True):
        ops = _fuse_silu_mul(ops, set(ir.output_names()))
    if pol.get("fuse_norm", True):
        ops = _fuse_rms_norm(ops, tensors, weights, set(ir.output_names()))
    if pol.get("fuse_rope", True):
        ops = _fuse_rope(ops, tensors, weights, set(ir.output_names()))
    # the fp32 weights of the linears that went to int8 leave the IR (unless something else reads them)
    used = {op.get(r) for op in ops for r in ("weight", "bias", "w_scale", "gamma_key", "beta_key")}
    gone = {op["weight"] for op in sel} - used
    weights = {k: v for k, v in weights.items() if k not in gone}
    from modelblaster.pipeline.extract_graph import _annotate_dispatches  # noqa: PLC0415
    g = dict(ir.graph)
    g.update(quant=MIXED, ops=ops, tensors=tensors)
    if pol.get("fuse_attention", False):
        g = _fuse_attention_fp32(g)
        ops, tensors = g["ops"], g["tensors"]
    if pol.get("fold_layout", True):
        # attention's permute copies folded into strided bmm operands (pipeline/layout_fold.py)
        from modelblaster.pipeline.layout_fold import fold  # noqa: PLC0415
        g, frep = fold(g)
        ops, tensors = g["ops"], g["tensors"]
    g["dispatches"] = _annotate_dispatches(ops)
    kinds = {}
    for op in ops:
        if op["op"] != "view":
            kinds[op["op"]] = kinds.get(op["op"], 0) + 1
    report = {"policy": pol, "n_calibration": len(samples),
              "op_kinds": dict(sorted(kinds.items(), key=lambda kv: -kv[1])),
              "n_int8_linears": len(sel),
              "n_fp32_linears": sum(1 for op in ops if op["op"] == "linear" and not op["name"].endswith("__outlpart")),
              "fp32_linears": [op["name"] for op in ops if op["op"] == "linear" and not op["name"].endswith("__outlpart")],
              "outlier_columns": {x: len(c) for x, c in outl.items()}}
    g["int8_mixed"] = dict(report)
    return g, weights, report


def _merge_siblings(ops: list, tensors: dict, weights: dict, max_rows: int) -> list:
    """linear_dq_s8 ops reading the same quantized input (q / k / v; gate / up) -> one linear3_dq_s8
    per group of up to 3: the weights / scales / biases stacked, one launch instead of 2-3 (and a
    wider product: no split-K on the Deck's skinny path). Bit-exact: every output column is computed
    exactly as before. Placed where the first sibling was (its input exists there already)."""
    groups: dict[tuple, list] = {}
    for op in ops:
        if op["op"] == "linear_dq_s8" and int(op["shape"]["M"]) <= max_rows:
            groups.setdefault((op["inputs"][0], op["inputs"][1]), []).append(op)
    replace, drop = {}, set()
    for (q, s), sib in groups.items():
        for g0 in range(0, len(sib) - 1, 3):
            grp = sib[g0:g0 + 3]
            if len(grp) < 2:
                continue
            Ns = [int(op["shape"]["N"]) for op in grp]
            name = "+".join(op["name"] for op in grp)
            wkey, skey, bkey = f"{name}.q8", f"{name}.wscale", None
            weights[wkey] = np.ascontiguousarray(np.concatenate([np.asarray(weights[op["weight"]]) for op in grp], 0))
            weights[skey] = np.ascontiguousarray(np.concatenate([np.asarray(weights[op["w_scale"]], np.float32) for op in grp]))
            if any(op.get("bias") for op in grp):
                bkey = f"{name}.bias"
                weights[bkey] = np.concatenate([np.asarray(weights[op["bias"]], np.float32) if op.get("bias")
                                                else np.zeros(n, np.float32) for op, n in zip(grp, Ns)])
            sh = {"M": grp[0]["shape"]["M"], "K": grp[0]["shape"]["K"], "N0": Ns[0], "N1": Ns[1],
                  "N2": Ns[2] if len(Ns) > 2 else 0}
            replace[id(grp[0])] = {"name": name, "op": "linear3_dq_s8", "inputs": [q, s],
                                   "outputs": [op["outputs"][0] for op in grp], "weight": wkey, "w_scale": skey,
                                   "bias": bkey, "shape": sh, "merged": [op["name"] for op in grp]}
            drop |= {id(op) for op in grp[1:]}
    out = []
    for op in ops:
        if id(op) in drop:
            continue
        out.append(replace.get(id(op), op))
    used = {op.get(r) for op in out for r in ("weight", "bias", "w_scale")}
    for k in [k for k in weights if (k.endswith(".q8") or k.endswith(".wscale")) and k not in used]:
        del weights[k]
    return out


def _fuse_residual(ops: list, tensors: dict, outputs: set) -> list:
    """add(r, linear_dq_s8(..)) whose linear output nothing else reads -> linear_dq_add_s8 (the residual
    add in the linear's epilogue): one launch and one [M, N] round trip less per residual. Bit-exact
    (the same fp32 add). The fused op sits where the add was (the residual exists there)."""
    readers: dict[str, int] = {}
    for op in ops:
        for t in op["inputs"]:
            readers[t] = readers.get(t, 0) + 1
    prod = {op["outputs"][0]: op for op in ops if op["op"] == "linear_dq_s8"}
    fused, drop = {}, set()
    for op in ops:
        if op["op"] != "add" or len(op["inputs"]) != 2:
            continue
        for j in (0, 1):
            t, r = op["inputs"][j], op["inputs"][1 - j]
            lin = prod.get(t)
            if lin is None or id(lin) in drop or readers.get(t) != 1 or t in outputs or t == r:
                continue
            n = int(lin["shape"]["M"]) * int(lin["shape"]["N"])
            if int(op["shape"].get("n", -1)) != n or tensors.get(r, {}).get("shape") is None \
                    or int(np.prod(tensors[r]["shape"])) != n:
                continue
            new = dict(lin)
            new.update(op="linear_dq_add_s8", inputs=list(lin["inputs"]) + [r], outputs=list(op["outputs"]),
                       name=lin["name"])
            fused[id(op)] = new
            drop.add(id(lin))
            break
    return [fused.get(id(op), op) for op in ops if id(op) not in drop]


def _fuse_silu_mul(ops: list, outputs: set) -> list:
    """mul(silu(g), u) (SwiGLU: ewise1 silu, then the mul that alone reads it) -> one silu_mul:
    one launch and one round trip less per MLP. The same fp32 expressions (fp32 tolerance)."""
    readers: dict[str, int] = {}
    for op in ops:
        for t in op["inputs"]:
            readers[t] = readers.get(t, 0) + 1
    silu = {op["outputs"][0]: op for op in ops if op["op"] == "ewise1" and op.get("opc") == 14}
    fused, drop = {}, set()
    for op in ops:
        if op["op"] != "mul" or len(op["inputs"]) != 2:
            continue
        for j in (0, 1):
            t, u = op["inputs"][j], op["inputs"][1 - j]
            sop = silu.get(t)
            if sop is None or readers.get(t) != 1 or t in outputs or t == u \
                    or int(sop["shape"]["n"]) != int(op["shape"]["n"]):
                continue
            fused[id(op)] = {"name": f"{op['name']}__swiglu", "op": "silu_mul", "inputs": [sop["inputs"][0], u],
                             "outputs": list(op["outputs"]), "shape": {"n": int(op["shape"]["n"])}}
            drop.add(id(sop))
            break
    return [fused.get(id(op), op) for op in ops if id(op) not in drop]


def _fuse_rms_norm(ops: list, tensors: dict, weights: dict, outputs: set) -> list:
    """The generic lowering's RMSNorm chain feeding quantize_rows_s8 --
        square -> mean -> + eps -> rsqrt -> x * r -> w * (.)  -> quantize_rows_s8(col_inv)
    -- becomes rms_norm (one fp32 row kernel) -> quantize_rows_s8 with col_inv * w: the norm's weight
    folds into the per-column factor (the weighted tensor is read by nothing else). 6 launches -> 1;
    with an fp32 reader the weight multiply stays (6 -> 2).
    The fp32 values move by fp32 rounding only (rms_norm divides by sqrt instead of multiplying
    by rsqrt; one multiply fewer), within the fp32 bound; the golden is this IR's own walk."""
    prod = {t: i for i, op in enumerate(ops) for t in op["outputs"]}
    users: dict = {}
    for i, op in enumerate(ops):
        for t in op["inputs"]:
            users.setdefault(t, []).append(i)

    def only(t, kind, fn=None):
        u = users.get(t, [])
        if len(u) != 1 or t in outputs:
            return None
        op = ops[u[0]]
        return u[0] if op["op"] == kind and (fn is None or op.get("fn") == fn) else None

    drop, new_at = set(), {}
    for i, sq in enumerate(ops):
        if sq["op"] != "ewise1" or sq.get("fn") != "square":
            continue
        x = sq["inputs"][0]
        im = only(sq["outputs"][0], "reduce", "mean")
        if im is None or ops[im]["shape"].get("inner") != 1:
            continue
        ie = only(ops[im]["outputs"][0], "ewise_scalar", "add")
        ir_ = only(ops[ie]["outputs"][0], "ewise1", "rsqrt") if ie is not None else None
        imul = only(ops[ir_]["outputs"][0], "ewise2", "mul") if ir_ is not None else None
        if imul is None or ops[imul]["inputs"] != [x, ops[ir_]["outputs"][0]]:
            continue
        nrm = ops[imul]["outputs"][0]
        iw = only(nrm, "ewise2", "mul")
        if iw is None:
            continue
        wname = ops[iw]["inputs"][0]
        y = ops[iw]["outputs"][0]
        if not tensors.get(wname, {}).get("const") or ops[iw]["inputs"][1] != nrm or y in outputs:
            continue
        qs = users.get(y, [])
        M, K = int(ops[im]["shape"]["outer"]), int(ops[im]["shape"]["R"])
        wv = np.asarray(weights[tensors[wname]["const"]], np.float32).reshape(-1)
        if wv.size != K:
            continue
        new_at[i] = {"name": f"{nrm}__rms", "op": "rms_norm", "inputs": [x], "outputs": [nrm],
                     "eps": float(ops[ie]["c"]), "shape": {"outer": M, "reduce": K, "inner": 1}}
        drop |= {im, ie, ir_, imul}
        if not qs or any(ops[j]["op"] != "quantize_rows_s8" for j in qs):
            continue                     # (an fp32 reader: the weight multiply stays)
        drop.add(iw)
        for j in qs:
            q = ops[j]
            ci = q["inputs"][1]
            key = f"const.{ci}__w"
            weights[key] = (np.asarray(weights[tensors[ci]["const"]], np.float32) * wv).astype(np.float32)
            tensors[f"{ci}__w"] = dict(tensors[ci], const=key)
            ops[j] = dict(q, inputs=[nrm, f"{ci}__w"])
    out = []
    for i, op in enumerate(ops):
        if i in new_at:
            out.append(new_at[i])
        elif i not in drop:
            out.append(op)
    used = {t for op in out for t in op["inputs"] + op["outputs"]} | outputs
    for t in [t for t in tensors if t not in used and t in prod]:
        del tensors[t]
    for t in [t for t, v in tensors.items() if v.get("const") and t not in used]:
        weights.pop(tensors[t]["const"], None)
        del tensors[t]
    return out


def _fuse_rope(ops: list, tensors: dict, weights: dict, outputs: set) -> list:
    """The generic lowering's RoPE (two strided slices of x [R*H, 2d], x1*cos, x2*sin, sub, x2*cos,
    x1*sin, add, cat) -> one rope_half: 9 launches -> 1. Same fp32 products and sums."""
    prod = {t: i for i, op in enumerate(ops) for t in op["outputs"]}
    users: dict = {}
    for i, op in enumerate(ops):
        for t in op["inputs"]:
            users.setdefault(t, []).append(i)

    def const(t):
        m = tensors.get(t, {})
        return np.asarray(weights[m["const"]], np.float32) if m.get("const") else None

    def mul_of(t):                       # t = ewise2 mul(slice, const) -> (slice, const array, broadcast dims)
        i = prod.get(t)
        if i is None or ops[i]["op"] != "ewise2" or ops[i].get("fn") != "mul" or len(users.get(t, [])) != 1:
            return None
        a, b = ops[i]["inputs"]
        cb = const(b)
        return (i, a, cb, ops[i]["shape"]) if cb is not None else None

    drop, new_at = set(), {}
    for ic, cat in enumerate(ops):
        sh = cat["shape"]
        if cat["op"] != "catn" or sh.get("n_in") != 2 or sh.get("inner") != 1 or sh.get("c0") != sh.get("c1"):
            continue
        d, rows = int(sh["c0"]), int(sh["outer"])
        A, B = cat["inputs"]
        ia, ib = prod.get(A), prod.get(B)
        if ia is None or ib is None or len(users.get(A, [])) != 1 or len(users.get(B, [])) != 1:
            continue
        if not (ops[ia]["op"] == "ewise2" and ops[ia].get("fn") == "sub" and ops[ib]["op"] == "add"):
            continue
        P, Q = (mul_of(t) for t in ops[ia]["inputs"])
        Rm, Sm = (mul_of(t) for t in ops[ib]["inputs"])
        if None in (P, Q, Rm, Sm):
            continue
        x1, x2 = P[1], Q[1]
        if Rm[1] != x2 or Sm[1] != x1 or not (np.array_equal(P[2], Rm[2]) and np.array_equal(Q[2], Sm[2])):
            continue
        i1, i2 = prod.get(x1), prod.get(x2)
        if i1 is None or i2 is None or ops[i1]["op"] != "strided_copy" or ops[i2]["op"] != "strided_copy":
            continue
        s1, s2 = ops[i1]["shape"], ops[i2]["shape"]
        src = ops[i1]["inputs"][0]
        if (ops[i2]["inputs"][0] != src or s1["off"] != 0 or s2["off"] != d
                or any(s[k] != v for s in (s1, s2) for k, v in (("d4", rows), ("d5", d), ("s4", 2 * d), ("s5", 1)))
                or any(s[f"d{k}"] != 1 for s in (s1, s2) for k in range(4))
                or len(users.get(x1, [])) != 2 or len(users.get(x2, [])) != 2):
            continue
        m = P[3]                         # broadcast of the table: [R, H, d] against [R, 1, d]
        R, H = int(m["d3"]), int(m["d4"])
        if R * H != rows or m["d5"] != d or m["sb4"] != 0 or P[2].size != R * d or Q[2].size != R * d:
            continue
        cn, sn = f"{cat['outputs'][0]}__cos", f"{cat['outputs'][0]}__sin"
        for nm, arr in ((cn, P[2]), (sn, Q[2])):
            weights[f"const.{nm}"] = np.ascontiguousarray(arr.reshape(R, d))
            tensors[nm] = {"shape": [R, d], "dtype": "f32", "quant": None, "const": f"const.{nm}"}
        new_at[ic] = {"name": f"{cat['outputs'][0]}__rope", "op": "rope_half", "inputs": [src, cn, sn],
                      "outputs": cat["outputs"], "shape": {"R": R, "H": H, "half": d}}
        drop |= {i1, i2, P[0], Q[0], Rm[0], Sm[0], ia, ib}
    out = [new_at.get(i, op) for i, op in enumerate(ops) if i not in drop]
    used = {t for op in out for t in op["inputs"] + op["outputs"]} | outputs
    for t in [t for t in tensors if t not in used and t in prod]:
        del tensors[t]
    for t in [t for t, v in tensors.items() if v.get("const") and t not in used]:
        weights.pop(tensors[t]["const"], None)
        del tensors[t]
    return out


_ATT_KINDS = ("bmm", "mul_scalar", "ewise_scalar", "ewise2", "softmax", "softmax_dim", "relu", "ewise1")


def _fuse_attention_fp32(g: dict) -> dict:
    """attention_fuse (written for the fp16 kinds) on this fp32 graph: bmm -> scale -> softmax -> bmm
    chains (the permute copies around them folded into strides) -> one fp32 ``attention`` op, the
    flash-style kernel (no score matrix). Chains with an additive mask are left alone."""
    import copy  # noqa: PLC0415
    from modelblaster.pipeline import attention_fuse  # noqa: PLC0415
    h = copy.deepcopy(g)
    for op in h["ops"]:
        if op["op"] in _ATT_KINDS:
            op["op"] += "_f16"
    h, rep = attention_fuse.fuse(h)
    if not rep["fused"]:
        return g
    for op in h["ops"]:
        if op["op"] == attention_fuse.KIND:
            op["op"] = "attention"
            op.pop("approx", None)
        elif op["op"].endswith("_f16") and op["op"][:-4] in _ATT_KINDS:
            op["op"] = op["op"][:-4]
    return h


def inputs_of(ir, sample) -> dict:
    """A model sample (tensor / list, placeholder order) -> {graph input name: float32 tensor}."""
    names = [p["name"] for p in ir.graph["input"].get("packed_inputs") or [{"name": ir.graph["input"]["tensor"]}]]
    xs = list(sample) if isinstance(sample, (list, tuple)) else [sample]
    if len(xs) != len(names):
        raise ValueError(f"sample has {len(xs)} tensors, the IR {len(names)} inputs")
    return {n: torch.as_tensor(x).detach().float().reshape(ir.tshape(n)) for n, x in zip(names, xs)}


def extract_mixed(fp32_dir: str, out_dir: str, calibration_samples: list | None, policy="linear") -> dict:
    """fp32 IR dir (extract_graph --quant fp32) -> int8_mixed IR in out_dir. The golden input
    is the fp32 IR's; calibration_samples (model samples) default to it."""
    _walk_torch()
    from mbir import load_ir  # noqa: PLC0415
    ir = load_ir(fp32_dir)
    golden = dict(ir.input_tensors("cpu"))
    samples = [inputs_of(ir, s) for s in calibration_samples] if calibration_samples else [golden]
    g, w, report = quantize_graph(ir, samples, policy)
    mixed = as_ir(g, w)
    out = walk(mixed, golden)
    o = torch.cat([out[t].reshape(-1).float() for t in mixed.output_names()])
    r = torch.from_numpy(ir.golden_out.reshape(-1).astype(np.float32))
    d = o.double() - r.double()
    report["vs_fp32_golden"] = {"cosine": float(torch.nn.functional.cosine_similarity(o.double(), r.double(), dim=0)),
                                "max_abs_err": float(d.abs().max()), "ref_abs_max": float(r.abs().max())}
    g["int8_mixed"]["vs_fp32_golden"] = report["vs_fp32_golden"]
    os.makedirs(out_dir, exist_ok=True)
    Path(out_dir, "graph.json").write_text(json.dumps(g, indent=1))
    np.savez(Path(out_dir, "weights.npz"), **w)
    io = np.load(Path(fp32_dir, "io.npz"))
    np.savez(Path(out_dir, "io.npz"), input=io["input"], output=o.numpy().astype(np.float32))
    v = report["vs_fp32_golden"]
    print(f"[int8_mixed] {g['name']}: {report['n_int8_linears']} int8 linears ({report['n_fp32_linears']} "
          f"kept fp32), calibrated on {len(samples)} samples; vs the fp32 golden: cosine {v['cosine']:.6f}, "
          f"max |err| {v['max_abs_err']:.3g} (|ref| <= {v['ref_abs_max']:.3g})", flush=True)
    return report
