"""Harness 1: walk a ModelBlaster IR on the GPU with PyTorch ops.

Answers "is graph.json + weights.npz enough to run the model on a GPU?"
independently of any CUDA kernel work, and doubles as the per-op oracle
walk_cuda.py diffs against.

An `_f16` op runs its fp32 rule and rounds every output to fp16 (and the
recurrent state of lstm_seq/gru_seq every step): the fp16 contract is fp16
storage with fp32 math, so this is the value an exact fp16 kernel stores.

    python walk_torch.py IR_DIR [IR_DIR ...] [--json out.json] [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import sys

import torch
import torch.nn.functional as F

from mbir import IR, compare, load_ir, strict_fp32


def _round_f16(t):
    """The binary16 value an fp16 kernel stores for an fp32 result (RNE; inf past 65504)."""
    return t.half().to(t.dtype) if t.is_floating_point() else t


def _state_rounding(op):
    return _round_f16 if op["op"].endswith("_f16") else (lambda t: t)


def _unary(fn):
    return lambda ir, op, ins, dev: [fn(ins[0], op)]


def _pool_is_ceil(sh) -> bool:
    floor_oh = (sh["IH"] + 2 * sh.get("PH", 0) - sh.get("DH", 1) * (sh["KH"] - 1) - 1) // sh["SH"] + 1
    return sh.get("OH", floor_oh) > floor_oh


OPS = {
    "relu": _unary(lambda x, op: F.relu(x)),
    "relu6": _unary(lambda x, op: x.clamp(0.0, 6.0)),
    "silu": _unary(lambda x, op: F.silu(x)),
    "sigmoid": _unary(lambda x, op: torch.sigmoid(x)),
    "elu": _unary(lambda x, op: F.elu(x, alpha=op.get("alpha", 1.0))),
    "add": lambda ir, op, ins, dev: [ins[0] + ins[1]],
    # a view with `offset` is a contiguous window into its input
    "view": lambda ir, op, ins, dev: [ins[0].reshape(-1)[
        op.get("offset", 0):op.get("offset", 0) + op["shape"]["n"]].reshape(ir.tshape(op["outputs"][0]))],
    # conv1d is extracted as conv2d with IH=1 on a 3-D tensor: shape from the op
    "conv2d": lambda ir, op, ins, dev: [F.conv2d(
        _nchw(ins[0], op["shape"]), ir.weight(op["weight"], dev), ir.weight(op.get("bias"), dev),
        stride=(op["shape"]["SH"], op["shape"]["SW"]),
        padding=(op["shape"]["PH"], op["shape"]["PW"]),
        dilation=(op["shape"].get("DH", 1), op["shape"].get("DW", 1)))],
    "conv2d_dw": lambda ir, op, ins, dev: [F.conv2d(
        ins[0], ir.weight(op["weight"], dev), ir.weight(op.get("bias"), dev),
        stride=(op["shape"]["SH"], op["shape"]["SW"]),
        padding=(op["shape"]["PH"], op["shape"]["PW"]), groups=op["shape"]["C"])],
    # extract_graph drops ceil_mode but records the ceil-mode OH/OW; infer it.
    "maxpool2d": lambda ir, op, ins, dev: [F.max_pool2d(
        ins[0], (op["shape"]["KH"], op["shape"]["KW"]),
        (op["shape"]["SH"], op["shape"]["SW"]),
        (op["shape"].get("PH", 0), op["shape"].get("PW", 0)),
        (op["shape"].get("DH", 1), op["shape"].get("DW", 1)),
        ceil_mode=_pool_is_ceil(op["shape"]))],
    # ModelBlaster folds BN at extract time: y = x * scale[c] + bias_fused[c].
    "batchnorm2d": lambda ir, op, ins, dev: [
        ins[0] * ir.weight(op["weight"], dev).view(1, -1, 1, 1)
        + ir.weight(op["bias"], dev).view(1, -1, 1, 1)],
    "linear": lambda ir, op, ins, dev: [F.linear(
        ins[0].reshape(op["shape"]["M"], op["shape"]["K"]),
        ir.weight(op["weight"], dev), ir.weight(op.get("bias"), dev))],
    "matmul": lambda ir, op, ins, dev: [
        ins[0].reshape(op["shape"]["M"], op["shape"]["K"]) @ ins[1].reshape(op["shape"]["K"], op["shape"]["N"])],
    "softmax": lambda ir, op, ins, dev: [torch.softmax(
        ins[0].reshape(op["shape"]["M"], op["shape"]["K"]), dim=-1)],
    "layer_norm": lambda ir, op, ins, dev: [F.layer_norm(
        ins[0].reshape(op["shape"]["M"], op["shape"]["K"]), (op["shape"]["K"],),
        ir.weight(op["gamma_key"], dev), ir.weight(op["beta_key"], dev), op.get("eps", 1e-5))],
    "sdpa": lambda ir, op, ins, dev: [F.scaled_dot_product_attention(
        *[t.reshape(op["shape"]["BH"], op["shape"]["S"], op["shape"]["D"]) for t in ins])],
    # MB reference divides by KH*KW unconditionally == count_include_pad=True
    "avgpool2d": lambda ir, op, ins, dev: [F.avg_pool2d(
        ins[0], (op["shape"]["KH"], op["shape"]["KW"]), (op["shape"]["SH"], op["shape"]["SW"]),
        (op["shape"].get("PH", 0), op["shape"].get("PW", 0)), count_include_pad=True)],
    "adaptive_avg_pool2d": lambda ir, op, ins, dev: [ins[0].mean(dim=(2, 3), keepdim=True)],
    "chunk2_c1": lambda ir, op, ins, dev: list(ins[0].split(op["shape"]["c_each"], dim=1)),
    "cat2_c1": lambda ir, op, ins, dev: [torch.cat(ins, dim=1)],
    "cat3_c1": lambda ir, op, ins, dev: [torch.cat(ins, dim=1)],
    "cat4_c1": lambda ir, op, ins, dev: [torch.cat(ins, dim=1)],
    "upsample_nearest": lambda ir, op, ins, dev: [F.interpolate(
        ins[0], scale_factor=op["shape"]["scale"], mode="nearest")],
}


# ---- generic ops (modelblaster/pipeline/reference_kernels_ext.py) ---------
_R6 = range(6)


def _d6(sh, p="d"):
    return [sh[f"{p}{i}"] for i in _R6]


def _strided_index(sh, p="s", dev="cpu"):
    """off + sum_k i_k * s_k over the [d0..d5] box (strides may be negative,
    which torch.as_strided rejects: a flip is an affine copy too)."""
    d = _d6(sh)
    idx = torch.full(d, sh.get("off", 0), dtype=torch.long, device=dev)
    for k, (n, st) in enumerate(zip(d, _d6(sh, p))):
        idx = idx + (torch.arange(n, device=dev) * st).reshape([n if j == k else 1 for j in range(6)])
    return idx


def _strided_copy(ir, op, ins, dev):
    sh = op["shape"]
    x = ins[0].reshape(-1)
    return [x[_strided_index(sh, dev=x.device)]]


def _gather(ir, op, ins, dev):
    x, idx = ins[0].reshape(-1), ins[1].reshape(-1).long()
    return [torch.where(idx >= 0, x[idx.clamp(min=0)], torch.zeros((), device=x.device))]


def _strided_scatter(ir, op, ins, dev):
    sh = op["shape"]
    y = ins[0].reshape(-1).clone()
    y[_strided_index(sh, dev=y.device).reshape(-1)] = ins[1].reshape(-1)
    return [y]


def _wrap(idx, A):
    idx = idx.long()
    return torch.where(idx < 0, idx + A, idx)


def _index_select(ir, op, ins, dev):
    sh = op["shape"]
    x = ins[0].reshape(sh["outer"], sh["A"], sh["inner"])
    return [torch.index_select(x, 1, _wrap(ins[1].reshape(-1), sh["A"]))]


def _gather_dim(ir, op, ins, dev):
    sh = op["shape"]
    x = ins[0].reshape(sh["outer"], sh["A"], sh["inner"])
    idx = _wrap(ins[1].reshape(sh["outer"], sh["J"], sh["inner"]), sh["A"])
    return [torch.gather(x, 1, idx)]


def _scatter_dim(ir, op, ins, dev):
    sh = op["shape"]
    x = ins[0].reshape(sh["outer"], sh["A"], sh["inner"])
    idx = _wrap(ins[1].reshape(sh["outer"], sh["J"], sh["inner"]), sh["A"])
    src = ins[2].reshape(sh["outer"], sh["J"], sh["inner"])
    return [x.scatter_add(1, idx, src) if sh["mode"] else x.scatter(1, idx, src)]


def _topk(ir, op, ins, dev):
    sh = op["shape"]
    x = ins[0].reshape(sh["outer"], sh["A"], sh["inner"])
    # stable sort: ties keep the lower index first, like the kernels
    v, i = torch.sort(x, dim=1, descending=bool(sh["largest"]), stable=True)
    return [v[:, :sh["K"]], i[:, :sh["K"]].float()]


def _convnd(ir, op, ins, dev):
    sh = op["shape"]
    x = ins[0].reshape(sh["N"], sh["IC"], sh["ID"], sh["IH"], sh["IW"])
    w = ins[1].reshape(sh["OC"], sh["IC"] // sh["G"], sh["KD"], sh["KH"], sh["KW"])
    b = ins[2].reshape(-1) if len(ins) > 2 else None
    return [F.conv3d(x, w, b, (sh["SD"], sh["SH"], sh["SW"]), (sh["PD"], sh["PH"], sh["PW"]),
                     (sh["DD"], sh["DH"], sh["DW"]), sh["G"])]


def _catn(ir, op, ins, dev):
    sh = op["shape"]
    parts = [t.reshape(sh["outer"], sh[f"c{k}"], sh["inner"]) for k, t in enumerate(ins)]
    return [torch.cat(parts, dim=1)]


_E1 = {0: torch.neg, 1: torch.exp, 2: torch.log, 3: torch.sqrt, 4: torch.rsqrt, 5: torch.abs,
       6: torch.tanh, 7: torch.sigmoid, 8: F.relu, 9: torch.sin, 10: torch.cos, 11: torch.erf,
       12: F.gelu, 13: lambda x: F.gelu(x, approximate="tanh"), 14: F.silu,
       16: torch.reciprocal, 17: torch.square, 18: F.softplus, 21: F.hardswish,
       22: F.hardsigmoid, 23: torch.log1p, 24: torch.expm1, 25: torch.sign, 26: torch.floor,
       27: torch.ceil, 28: torch.round, 29: lambda x: (x == 0).float(), 30: lambda x: x,
       31: F.mish}


def _ewise1(ir, op, ins, dev):
    c, x = op["opc"], ins[0]
    if c == 15:
        return [x.clamp(op["p0"], op["p1"])]
    if c == 19:
        return [F.leaky_relu(x, op["p0"])]
    if c == 20:
        return [F.elu(x, op["p0"])]
    return [_E1[c](x)]


_E2 = {0: lambda a, b, al: a + al * b, 1: lambda a, b, al: a - al * b,
       2: lambda a, b, al: a * b, 3: lambda a, b, al: a / b,
       4: lambda a, b, al: torch.maximum(a, b), 5: lambda a, b, al: torch.minimum(a, b),
       6: lambda a, b, al: torch.pow(a, b),
       7: lambda a, b, al: torch.where(b != 0, torch.full_like(a, al), a),
       8: lambda a, b, al: (a == b).float(), 9: lambda a, b, al: (a != b).float(),
       10: lambda a, b, al: (a < b).float(), 11: lambda a, b, al: (a <= b).float(),
       12: lambda a, b, al: (a > b).float(), 13: lambda a, b, al: (a >= b).float()}


def _ewise2(ir, op, ins, dev):
    sh = op["shape"]
    d = _d6(sh)
    a = torch.as_strided(ins[0].reshape(-1), d, _d6(sh, "sa"))
    b = torch.as_strided(ins[1].reshape(-1), d, _d6(sh, "sb"))
    return [_E2[op["opc"]](a, b, op.get("alpha", 1.0)).contiguous()]


_ES = {0: lambda x, c: x + c, 1: lambda x, c: x - c, 2: lambda x, c: c - x,
       3: lambda x, c: x * c, 4: lambda x, c: x / c, 5: lambda x, c: c / x,
       6: lambda x, c: torch.pow(x, c), 7: lambda x, c: torch.pow(torch.full_like(x, c), x),
       8: lambda x, c: x.clamp(min=c), 9: lambda x, c: x.clamp(max=c),
       10: lambda x, c: (x == c).float(), 11: lambda x, c: (x != c).float(),
       12: lambda x, c: (x < c).float(), 13: lambda x, c: (x <= c).float(),
       14: lambda x, c: (x > c).float(), 15: lambda x, c: (x >= c).float()}


def _reduce(ir, op, ins, dev):
    sh, c = op["shape"], op["opc"]
    x = ins[0].reshape(sh["outer"], sh["R"], sh["inner"])
    r = {0: lambda: x.sum(1), 1: lambda: x.mean(1), 2: lambda: x.amax(1), 3: lambda: x.amin(1),
         4: lambda: x.prod(1), 5: lambda: (x * x).sum(1), 6: lambda: x.norm(dim=1),
         7: lambda: x.argmax(1).float(), 8: lambda: x.argmin(1).float(),
         9: lambda: x.var(1, unbiased=False), 10: lambda: x.var(1, unbiased=True)}[c]()
    return [r]


def _softmax_dim(ir, op, ins, dev):
    sh = op["shape"]
    x = ins[0].reshape(sh["outer"], sh["R"], sh["inner"])
    return [(torch.log_softmax if op.get("log_out") else torch.softmax)(x, dim=1)]


def _conv2d_grouped(ir, op, ins, dev):
    sh = op["shape"]
    return [F.conv2d(ins[0].reshape(sh["N"], sh["IC"], sh["IH"], sh["IW"]), ir.weight(op["weight"], dev), ir.weight(op.get("bias"), dev),
                     (sh["SH"], sh["SW"]), (sh["PH"], sh["PW"]), (sh["DH"], sh["DW"]), sh["G"])]


def _maxpool2d_ex(ir, op, ins, dev):
    sh = op["shape"]
    y = F.max_pool2d(ins[0], (sh["KH"], sh["KW"]), (sh["SH"], sh["SW"]), (sh["PH"], sh["PW"]),
                     (sh["DH"], sh["DW"]), ceil_mode=True)
    return [y[:, :, :sh["OH"], :sh["OW"]]]


def _nchw(x, sh):
    return x.reshape(sh["N"], sh["IC"], sh["IH"], sh["IW"])


def _conv_transpose2d(ir, op, ins, dev):
    sh = op["shape"]
    opad = tuple(max(0, _out_pad(sh["I" + a], sh["O" + a], sh["K" + a], sh["S" + a], sh["P" + a],
                                 sh["D" + a])) for a in "HW")
    y = F.conv_transpose2d(_nchw(ins[0], sh), ir.weight(op["weight"], dev), ir.weight(op.get("bias"), dev),
                           (sh["SH"], sh["SW"]), (sh["PH"], sh["PW"]), opad, groups=sh["G"],
                           dilation=(sh["DH"], sh["DW"]))
    return [y[:, :, :sh["OH"], :sh["OW"]]]


def _out_pad(i, o, k, s, p, d):
    return o - ((i - 1) * s - 2 * p + d * (k - 1) + 1)


def _conv3d(ir, op, ins, dev):
    sh = op["shape"]
    return [F.conv3d(ins[0], ir.weight(op["weight"], dev), ir.weight(op.get("bias"), dev),
                     (sh["SD"], sh["SH"], sh["SW"]), (sh["PD"], sh["PH"], sh["PW"]),
                     (sh["DD"], sh["DH"], sh["DW"]), sh["G"])]


def _conv_transpose3d(ir, op, ins, dev):
    sh = op["shape"]
    opad = tuple(_out_pad(sh["I" + a], sh["O" + a], sh["K" + a], sh["S" + a], sh["P" + a], sh["D" + a])
                 for a in "DHW")
    return [F.conv_transpose3d(ins[0], ir.weight(op["weight"], dev), ir.weight(op.get("bias"), dev),
                               (sh["SD"], sh["SH"], sh["SW"]), (sh["PD"], sh["PH"], sh["PW"]),
                               opad, sh["G"], (sh["DD"], sh["DH"], sh["DW"]))]


def _pool3d_ceil(sh) -> bool:
    floor_od = (sh["ID"] + 2 * sh["PD"] - sh.get("DD", 1) * (sh["KD"] - 1) - 1) // sh["SD"] + 1
    return sh["OD"] > floor_od or _pool_is_ceil(sh)


def _maxpool3d(ir, op, ins, dev):
    sh = op["shape"]
    return [F.max_pool3d(ins[0], (sh["KD"], sh["KH"], sh["KW"]), (sh["SD"], sh["SH"], sh["SW"]),
                         (sh["PD"], sh["PH"], sh["PW"]), (sh["DD"], sh["DH"], sh["DW"]),
                         ceil_mode=_pool3d_ceil(sh))]


def _avgpool3d(ir, op, ins, dev):  # MB divides by KD*KH*KW (count_include_pad)
    sh = op["shape"]
    return [F.avg_pool3d(ins[0], (sh["KD"], sh["KH"], sh["KW"]), (sh["SD"], sh["SH"], sh["SW"]),
                         (sh["PD"], sh["PH"], sh["PW"]), ceil_mode=_pool3d_ceil(sh),
                         count_include_pad=True, divisor_override=sh["KD"] * sh["KH"] * sh["KW"])]


def _group_norm(ir, op, ins, dev):
    sh = op["shape"]
    x = ins[0].reshape(sh["N"], sh["C"], sh["HW"])
    y = F.group_norm(x, sh["G"], ir.weight(op["gamma_key"], dev).reshape(-1),
                     ir.weight(op["beta_key"], dev).reshape(-1), op.get("eps", 1e-5))
    return [y.reshape(ins[0].shape)]


def _lstm_seq(ir, op, ins, dev):
    x, h, c, w_ih, w_hh, bg = ins
    sh = op["shape"]
    B, T, H = sh["B"], sh["T"], sh["H"]
    x = x.reshape(B, T, sh["I"])
    h, c = h.reshape(B, H).clone(), c.reshape(B, H).clone()
    y = torch.empty(B, T, H, device=x.device)
    q = _state_rounding(op)                 # fp16: h and c are stored every step
    for s_ in range(T):
        t = T - 1 - s_ if sh["reverse"] else s_
        g = x[:, t] @ w_ih.T + h @ w_hh.T + bg
        i, f, gg, o = g.chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(gg)
        h = q(torch.sigmoid(o) * torch.tanh(c))
        c = q(c)
        y[:, t] = h
    return [y, h, c]


def _gru_seq(ir, op, ins, dev):
    x, h, w_ih, w_hh, b_ih, b_hh = ins
    sh = op["shape"]
    B, T, H = sh["B"], sh["T"], sh["H"]
    x = x.reshape(B, T, sh["I"])
    h = h.reshape(B, H).clone()
    y = torch.empty(B, T, H, device=x.device)
    q = _state_rounding(op)
    for s_ in range(T):
        t = T - 1 - s_ if sh["reverse"] else s_
        gi = x[:, t] @ w_ih.T + b_ih
        gh = h @ w_hh.T + b_hh
        r = torch.sigmoid(gi[:, :H] + gh[:, :H])
        z = torch.sigmoid(gi[:, H:2 * H] + gh[:, H:2 * H])
        n = torch.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
        h = q((1 - z) * n + z * h)
        y[:, t] = h
    return [y, h]


def _reduce3(fn):
    def rule(ir, op, ins, dev):
        sh = op["shape"]
        R = sh.get("reduce", sh.get("R"))
        return [fn(ins[0].reshape(sh["outer"], R, sh["inner"]))]
    return rule


OPS.update({
    "lstm_seq": _lstm_seq, "gru_seq": _gru_seq,
    "gelu": _unary(lambda x, op: F.gelu(x)),
    "sum_dim": _reduce3(lambda x: x.sum(1)), "mean_dim": _reduce3(lambda x: x.mean(1)),
    "max_dim": _reduce3(lambda x: x.amax(1)), "min_dim": _reduce3(lambda x: x.amin(1)),
    "strided_copy": _strided_copy, "gather": _gather, "catn": _catn,
    "ewise1": _ewise1, "ewise2": _ewise2,
    "ewise_scalar": lambda ir, op, ins, dev: [_ES[op["opc"]](ins[0], op["c"])],
    "reduce": _reduce, "softmax_dim": _softmax_dim, "conv2d_grouped": _conv2d_grouped,
    "maxpool2d_ex": _maxpool2d_ex, "conv_transpose2d": _conv_transpose2d,
    "bmm": lambda ir, op, ins, dev: [torch.bmm(
        ins[0].reshape(op["shape"]["batch"], op["shape"]["M"], op["shape"]["K"]),
        ins[1].reshape(op["shape"]["batch"], op["shape"]["K"], op["shape"]["N"]))],
    "mul_scalar": lambda ir, op, ins, dev: [ins[0] * op["scalar"]],
    "mul": lambda ir, op, ins, dev: [ins[0] * ins[1]],
    "tanh": _unary(lambda x, op: torch.tanh(x)),
    "leaky_relu": _unary(lambda x, op: F.leaky_relu(x, op.get("negative_slope", 0.01))),
    "hardtanh": _unary(lambda x, op: x.clamp(op.get("min_val", -1.0), op.get("max_val", 1.0))),
    "softplus": _unary(lambda x, op: F.softplus(x)),
    "conv3d": _conv3d, "conv_transpose3d": _conv_transpose3d,
    "maxpool3d": _maxpool3d, "avgpool3d": _avgpool3d, "group_norm": _group_norm,
    # C = A.T @ B with A stored (K, M); C = A @ B.T with B stored (N, K)
    "matmul_ta": lambda ir, op, ins, dev: [
        ins[0].reshape(op["shape"]["K"], op["shape"]["M"]).t() @ ins[1].reshape(op["shape"]["K"], op["shape"]["N"])],
    "matmul_tb": lambda ir, op, ins, dev: [
        ins[0].reshape(op["shape"]["M"], op["shape"]["K"]) @ ins[1].reshape(op["shape"]["N"], op["shape"]["K"]).t()],
    "index_select": _index_select, "gather_dim": _gather_dim, "scatter_dim": _scatter_dim,
    "topk": _topk, "convnd": _convnd, "strided_scatter": _strided_scatter,
    # MB's `gelu_exact` is the tanh (MinGPT) form; `gelu` is the erf form
    "gelu_exact": _unary(lambda x, op: F.gelu(x, approximate="tanh")),
    "cumsum": lambda ir, op, ins, dev: [torch.cumsum(
        ins[0].reshape(op["shape"]["outer"], op["shape"]["axis"], op["shape"]["inner"]), dim=1)],
    "cumprod": lambda ir, op, ins, dev: [torch.cumprod(
        ins[0].reshape(op["shape"]["outer"], op["shape"]["axis"], op["shape"]["inner"]), dim=1)],
})


def _rule(kind: str):
    """An `_f16` kind runs its fp32 rule (walk rounds the outputs)."""
    return OPS.get(kind) or (OPS.get(kind[:-4]) if kind.endswith("_f16") else None)


# ---- Level1-only ops (losses, norms, index ops); semantics as their KernelSpecs
def _orit(sh, x):
    return x.reshape(sh["outer"], sh.get("reduce", sh.get("axis", sh.get("R"))), sh["inner"])


def _excl_cumsum(ir, op, ins, dev):
    sh = op["shape"]
    x = ins[0].reshape(-1, sh["N"])[:sh["Bout"]]
    return [torch.cat([torch.zeros(sh["Bout"], 1, dtype=x.dtype, device=x.device), x.cumsum(1)], dim=1)]


OPS.update({
    "hinge_loss": lambda ir, op, ins, dev: [torch.clamp(
        1 - ins[0].reshape(-1, op["shape"]["targ_len"]) * ins[1].reshape(1, -1), min=0).mean()],
    "diag_matmul": lambda ir, op, ins, dev: [ins[0].reshape(-1, 1) * ins[1].reshape(op["shape"]["N"], op["shape"]["M"])],
    "triu": lambda ir, op, ins, dev: [torch.triu(ins[0].reshape(op["shape"]["M"], op["shape"]["N"]),
                                                op["shape"].get("diagonal", 0))],
    "tril": lambda ir, op, ins, dev: [torch.tril(ins[0].reshape(op["shape"]["M"], op["shape"]["N"]),
                                                op["shape"].get("diagonal", 0))],
    "matmul_tatb": lambda ir, op, ins, dev: [
        ins[0].reshape(op["shape"]["K"], op["shape"]["M"]).T @ ins[1].reshape(op["shape"]["N"], op["shape"]["K"]).T],
    "log_softmax": lambda ir, op, ins, dev: [torch.log_softmax(
        ins[0].reshape(op["shape"]["M"], op["shape"]["K"]), dim=-1)],
    "swish": _unary(lambda x, op: x * torch.sigmoid(x)),
    "selu": _unary(lambda x, op: F.selu(x)),
    "hardsigmoid": _unary(lambda x, op: F.hardsigmoid(x)),
    "softsign": _unary(lambda x, op: x / (1 + x.abs())),
    "log": _unary(lambda x, op: torch.log(x)),
    "rms_norm": lambda ir, op, ins, dev: [(lambda x: x / torch.sqrt((x * x).mean(1, keepdim=True) + op["eps"]))(
        _orit(op["shape"], ins[0]))],
    "frobenius_norm": lambda ir, op, ins, dev: [ins[0] / torch.sqrt((ins[0] * ins[0]).sum())],
    "mean_abs_norm": lambda ir, op, ins, dev: [(lambda x: x / x.abs().mean(1, keepdim=True))(_orit(op["shape"], ins[0]))],
    "l2_norm": lambda ir, op, ins, dev: [(lambda x: x / torch.sqrt((x * x).sum(1, keepdim=True)))(_orit(op["shape"], ins[0]))],
    "argmax_dim": lambda ir, op, ins, dev: [_orit(op["shape"], ins[0]).argmax(1)],
    "argmin_dim": lambda ir, op, ins, dev: [_orit(op["shape"], ins[0]).argmin(1)],
    "flip": lambda ir, op, ins, dev: [_orit(op["shape"], ins[0]).flip(1)],
    "exclusive_cumsum": _excl_cumsum,
    "mse_loss": lambda ir, op, ins, dev: [((ins[0].reshape(-1) - ins[1].reshape(-1)) ** 2).mean()],
    "cross_entropy_loss": lambda ir, op, ins, dev: [F.cross_entropy(
        ins[0].reshape(op["shape"]["N"], op["shape"]["C"]), ins[1].reshape(-1).long())],
    "huber_loss": lambda ir, op, ins, dev: [F.smooth_l1_loss(
        ins[0].reshape(-1), ins[1].reshape(-1), beta=op.get("beta", 1.0))],
    "kldiv_loss": lambda ir, op, ins, dev: [F.kl_div(
        ins[0].reshape(op["shape"]["N"], op["shape"]["C"]), ins[1].reshape(op["shape"]["N"], op["shape"]["C"]),
        reduction="batchmean")],
    "triplet_loss": lambda ir, op, ins, dev: [F.triplet_margin_loss(
        *(t.reshape(op["shape"]["B"], op["shape"]["F"]) for t in ins), margin=op.get("margin", 1.0), p=2)],
})


def walk(ir: IR, dev="cuda", keep_all=False):
    """Returns (output_flat, env) — env maps every IR tensor to its value."""
    env = ir.input_tensors(dev)
    env.update(ir.const_tensors(dev))
    def run_op(op):
        if "sub_ops" in op:                    # fused op (kb_proposals): its parts in order
            for sub in op["sub_ops"]:
                run_op(sub)
            return
        fn = _rule(op["op"])
        if fn is None:
            raise NotImplementedError(op["op"])
        outs = fn(ir, op, [env[t] for t in op["inputs"]], dev)
        if op["op"].endswith("_f16"):
            outs = [_round_f16(v) for v in outs]
        for name, val in zip(op["outputs"], outs):
            env[name] = val.reshape(ir.tshape(name)) if name in ir.graph["tensors"] else val

    for op in ir.ops:
        run_op(op)
    out = torch.cat([env[t].reshape(-1) for t in ir.output_names()])
    return out, env


def run_one(d: str, dev="cuda") -> dict:
    ir = load_ir(d)
    ops = sorted({o["op"] for o in ir.ops})
    missing = [o for o in ops if _rule(o) is None]
    rec = {"ir": ir.name, "n_ops": len(ir.ops), "op_types": ops, "unsupported": missing}
    if missing:
        rec["ok"] = False
        return rec
    with torch.no_grad():
        out, _ = walk(ir, dev)
        rec.update(compare(out, ir.golden_out, ir.dtype))
        if dev == "cuda":
            # warm timing of the whole walk (eager, includes launch overhead)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):
                walk(ir, dev)
            end.record()
            torch.cuda.synchronize()
            rec["eager_ms"] = start.elapsed_time(end) / 10
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("irs", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    strict_fp32()
    results = []
    for d in a.irs:
        try:
            r = run_one(d, dev=a.device)
        except Exception as e:  # report and keep going
            r = {"ir": d, "ok": False, "error": f"{type(e).__name__}: {e}"}
        results.append(r)
        print(json.dumps(r))
    if a.json:
        json.dump(results, open(a.json, "w"), indent=1)
    sys.exit(0 if all(r.get("ok") for r in results) else 1)


if __name__ == "__main__":
    main()
