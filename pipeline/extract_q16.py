"""Integer lowering of a frozen quantisation PLAN: int16 activations, split dispatches and
per-channel scales on top of extract_int8 (patches/0103, opt-in).

A plan (iiswc-tutorial fpga/pynq-z2/modelblaster/moonshine/q16_plan.py) says, per tensor,
per-tensor int8 / int16, per-channel int8, or multi-range int8, and per weight, per-tensor,
per-row or row groups, plus weight ROWS (rows divided by a_c; the output carries scale*a_c).
Nothing here chooses a range: every scale is the plan's.

How: extract_int8 runs UNCHANGED on the plan's ranges (range_overrides) over a copy of the
model with the rows applied, so every op the plan leaves at per-tensor int8 comes out exactly
as the stock extractor emits it.  Then the ops the plan changes are rewritten:

  conv2d_s8, output multi-range   split16_s8 (hi, lo) + G*R*2 stock conv2d_s8 dispatches
                                  + mrcombine_s16
  conv2d_s8, int16 in/out, per-row conv2d_s16_pc
  tanh_s8 / gelu_s8 on int16      lut16_s16 (int16 out) or lut16_pc_s8 (per-channel int8 out)
  groupnorm_s8 on int16           groupnorm_s16
  permute4_s8 on int16            permute4_s16
  add_s8, per-channel output      add_pc_s8 / add_s16_pc_s8
  layernorm_s8 on per-channel or  layernorm_pc_s8 / layernorm_s16_s8
  int16 input
  linear_s8, per-row weights      linear_s8_pc (the stock kind)

and the golden is simulated here, op by op, in integers (the stock kinds exactly as
extract_int8's MB_INT8_GOLDEN_C_FLOAT=1 simulator computes them, so a reference-kernel build
reproduces it at every dispatch).

THE SPLIT DISPATCH (a multi-range output of a conv whose input is int16), on an engine that
only multiplies int8 by int8 and requantises per tensor:
  hi = min((x + 128) >> 8, 127),  lo = clamp8(x - 256*hi)      so x = 256*hi + lo exactly
  (for x < 32640; the top 128 codes saturate).  The row-offset split x = 256*(x>>8) +
  ((x&0xFF)-128) + 128 was measured and rejected: a small negative x gives hi = -1, so H and
  L each carry ~256*sum(w) and cancel, and both saturate in the fine range (Moonshine R:
  28.6 % WER in float simulation against 8.45 % for this split).
  Per row group g (int8 weights at scale sw_g, rows in ascending order) and per range k
  (step t_k = range/127/ratio_k):
    H_{g,k} = conv2d_s8(hi, w_g, bias 0)           multiplier 256*s_in*sw_g/t_k
    L_{g,k} = conv2d_s8(lo, w_g, round(b/(s_in*sw_g)))   multiplier s_in*sw_g/t_k
  (H and L get distinct weight arrays: same bytes, different bias).  Per output element the
  finest range whose H and L codes are both off the rails (-128, 127) is taken, coarsest if
  none: out = clamp16((z_H + z_L) * ratio_max/ratio_k), int16 at t_max = range/127/ratio_max.

THE NORMALISATIONS (groupnorm_s16, layernorm_pc_s8, layernorm_s16_s8) share one exact
integer core, their own golden (there is no float reference to be drift against):
  u      the input codes, times m_c = round(s_c/s_ref * 2^F) for a per-channel input
         (F = 24; F = 0 and m = 1 for a per-tensor one)
  S = sum u, Q = sum u^2, A = K*Q - S^2, V = A + E,   E = max(1, round(eps*K^2/(s_ref/2^F)^2))
  R = isqrt(floor(2^120 / V))
  t_i = floor((K*u_i - S) * R / 2^44)            (the normalised value in Q16)
  out = clamp(floor((t_i*g_c + b_c*2^16 + 2^31) / 2^32))
        g_c = round(gamma_c/s_out * 2^16), b_c = round(beta_c/s_out * 2^16)
All floors are arithmetic shifts / floor divisions; round is half-to-even at extraction.

    python -m modelblaster.pipeline.extract_q16 --model moonshine_enc --plan plan.json --out-dir DIR
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import tempfile

import numpy as np
import torch

from . import extract_graph as eg

_QMAX = {8: 127, 16: 32767}
_NORM_P2 = 120      # R = isqrt(2^120 / V)
_NORM_SH = 44       # t = (K*u - S) * R >> 44   -> Q16
_LN_F = 24          # per-channel input multiplier precision


# ---------------------------------------------------------------------------------------
# golden primitives (numpy / Python integers)
# ---------------------------------------------------------------------------------------
def _rint(x):
    return np.rint(np.asarray(x, dtype=np.float64))


def split16(x16: np.ndarray, part: str) -> np.ndarray:
    x = x16.astype(np.int32)
    hi = np.minimum((x + 128) >> 8, 127)
    if part == "hi":
        return hi.astype(np.int8)
    return np.clip(x - 256 * hi, -128, 127).astype(np.int8)


def mrcombine(zs: list, sh: dict, gidx, lidx, ratios) -> np.ndarray:
    """zs[(g*R + k)*2 + (0 H | 1 L)] int8 [N, OC_g, OH*OW]."""
    N, OC, HW = sh["N"], sh["OC"], sh["OH"] * sh["OW"]
    G, R = sh["G"], sh["R"]
    rmax = int(ratios[-1])
    out = np.zeros((N, OC, HW), dtype=np.int64)
    for c in range(OC):
        g, j = int(gidx[c]), int(lidx[c])
        val = None
        for k in range(R):
            zh = zs[(g * R + k) * 2][:, j, :].astype(np.int64)
            zl = zs[(g * R + k) * 2 + 1][:, j, :].astype(np.int64)
            v = (zh + zl) * (rmax // int(ratios[k]))
            if val is None:
                val = v
            else:
                ok = (zh > -128) & (zh < 127) & (zl > -128) & (zl < 127)
                val = np.where(ok, v, val)
        out[:, c, :] = val
    return np.clip(out, -32768, 32767).astype(np.int16)


def norm_core(u: np.ndarray, E: int) -> np.ndarray:
    """u int64 [rows, K] -> t (Python-int object array) [rows, K], exactly as the C core."""
    rows, K = u.shape
    out = np.empty((rows, K), dtype=object)
    for r in range(rows):
        ur = [int(v) for v in u[r]]
        S = sum(ur)
        Q = sum(v * v for v in ur)
        V = K * Q - S * S + int(E)
        R = math.isqrt((1 << _NORM_P2) // V)
        out[r] = [((K * v - S) * R) >> _NORM_SH for v in ur]
    return out


def norm_affine(t: np.ndarray, g: np.ndarray, b: np.ndarray, cidx: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """t object [.., K] Q16; g, b int64 per channel; cidx channel of each position."""
    gg = [int(v) for v in g]
    bb = [int(v) << 16 for v in b]
    flat = t.reshape(-1)
    ci = cidx.reshape(-1)
    res = np.empty(flat.shape[0], dtype=np.int64)
    for i in range(flat.shape[0]):
        c = int(ci[i])
        v = (int(flat[i]) * gg[c] + bb[c] + (1 << 31)) >> 32
        res[i] = min(max(v, lo), hi)
    return res.reshape(t.shape)


def sim_groupnorm_s16(x16, sh, W, op):
    N, C, HW = sh["N"], sh["C"], sh["H"] * sh["W"]
    x = x16.reshape(N, C * HW).astype(np.int64)
    t = norm_core(x, op["quant"]["eps_q"])
    cidx = np.repeat(np.arange(C), HW)[None, :].repeat(N, axis=0)
    y = norm_affine(t, W[op["gmul"]], W[op["badd"]], cidx, -32768, 32767)
    return y.astype(np.int16).reshape(-1)


def sim_layernorm(xin, sh, W, op):
    M, K = sh["M"], sh["K"]
    x = xin.reshape(M, K).astype(np.int64)
    if op["op"] == "layernorm_pc_s8":
        x = x * W[op["umul"]].astype(np.int64)[None, :]
    t = norm_core(x, op["quant"]["eps_q"])
    cidx = np.tile(np.arange(K), (M, 1))
    y = norm_affine(t, W[op["gmul"]], W[op["badd"]], cidx, -128, 127)
    return y.astype(np.int8).reshape(-1)


def sim_add_pc(a, b, sh, W, op):
    n, C = sh["n"], sh["C"]
    aa = a.reshape(-1).astype(np.int64)
    bb = b.reshape(-1).astype(np.int64)
    ma = np.tile(W[op["amul"]].astype(np.int64), n // C)
    mb = np.tile(W[op["bmul"]].astype(np.int64), n // C)
    v = (aa * ma + bb * mb + (1 << 23)) >> 24
    return np.clip(v, -128, 127).astype(np.int8)


def sim_lut16_s16(x16, W, op):
    return W[op["weight"]][x16.reshape(-1).astype(np.int64) + 32768].astype(np.int16)


def sim_lut16_pc_s8(x16, sh, W, op):
    N, C, HW = sh["N"], sh["C"], sh["HW"]
    q = W[op["weight"]][x16.reshape(N, C, HW).astype(np.int64) + 32768].astype(np.int64)
    m = W[op["mult"]].astype(np.int64).reshape(1, C, 1)
    v = (q * m + (1 << 31)) >> 32
    return np.clip(v, -128, 127).astype(np.int8).reshape(-1)


def sim_conv2d_s16_pc(x16, sh, W, op):
    x = x16.reshape(sh["N"], sh["IC"], sh["IH"], sh["IW"]).astype(np.int64)
    acc = _conv_acc64(x, W[op["weight"]], sh) + W[op["bias"]].astype(np.int64).reshape(1, -1, 1, 1)
    mult = W[op["mult"]].astype(np.int64)
    shift = W[op["shift"]].astype(np.int64)
    out = np.empty(acc.shape, dtype=np.int64)
    for c in range(sh["OC"]):
        tot = 31 + int(shift[c])
        v = (acc[:, c].astype(object) * int(mult[c]) + (1 << (tot - 1))) >> tot
        out[:, c] = v.astype(np.int64)
    return np.clip(out, -32768, 32767).astype(np.int16).reshape(-1)


def _conv_acc64(x, w_q, sh):
    N, IC, IH, IW = sh["N"], sh["IC"], sh["IH"], sh["IW"]
    OC, OH, OW, KH, KW = sh["OC"], sh["OH"], sh["OW"], sh["KH"], sh["KW"]
    SH, SW, PH, PW = sh["SH"], sh["SW"], sh["PH"], sh["PW"]
    ph_hi = max(0, (OH - 1) * SH + KH - PH - IH)
    pw_hi = max(0, (OW - 1) * SW + KW - PW - IW)
    xp = np.pad(x.reshape(N, IC, IH, IW).astype(np.int64), ((0, 0), (0, 0), (PH, ph_hi), (PW, pw_hi)))
    w = np.asarray(w_q).reshape(OC, -1).astype(np.int64)
    out = np.zeros((N, OC, OH, OW), dtype=np.int64)
    for n in range(N):
        win = np.lib.stride_tricks.sliding_window_view(xp[n], (KH, KW), axis=(1, 2))
        win = win[:, : (OH - 1) * SH + 1: SH, : (OW - 1) * SW + 1: SW]
        cols = np.ascontiguousarray(win.transpose(1, 2, 0, 3, 4)).reshape(OH * OW, -1)
        out[n] = (cols @ w.T).T.reshape(OC, OH, OW)
    return out


def sim_permute4(x, sh, dtype):
    d4 = [sh["d0"], sh["d1"], sh["d2"], sh["d3"]]
    p4 = [sh["p0"], sh["p1"], sh["p2"], sh["p3"]]
    return np.ascontiguousarray(x.reshape(d4).transpose(p4)).reshape(-1).astype(dtype)


def _sim_stock(op, acts, W):
    """The stock kinds this lowering keeps, as extract_int8's C-exact simulator computes
    them (MB_INT8_GOLDEN_C_FLOAT=1)."""
    k = op["op"]
    x = acts[op["inputs"][0]]
    sh = op.get("shape", {})
    q = op.get("quant", {})
    if k == "view":
        return x
    if k == "linear_s8":
        in_2d = x.reshape(sh["M"], sh["K"]).astype(np.int32)
        w_2d = W[op["weight"]].reshape(sh["N"], sh["K"]).astype(np.int32)
        acc = (in_2d + q["input_offset"]) @ (w_2d + q["filter_offset"]).T + W[op["bias"]].astype(np.int32)
        v = eg._requantize_int(acc, q["output_multiplier"], q["output_shift"]) + q["output_offset"]
        return np.clip(v, q["activation_min"], q["activation_max"]).astype(np.int8)
    if k == "linear_s8_pc":
        in_2d = x.reshape(sh["M"], sh["K"]).astype(np.int32)
        w_2d = W[op["weight"]].reshape(sh["N"], sh["K"]).astype(np.int32)
        acc = in_2d @ w_2d.T + W[op["bias"]].astype(np.int32)
        v = eg._requantize_int_per_oc(acc, W[q["output_multiplier_per_oc_key"]],
                                      W[q["output_shift_per_oc_key"]], oc_axis=1)
        return np.clip(v, q["activation_min"], q["activation_max"]).astype(np.int8)
    if k == "conv2d_s8":
        return eg._sim_conv2d_s8(x, sh, q, W[op["weight"]], W[op["bias"]]).reshape(-1)
    if k == "permute4_s8":
        if np.float32(q["scale_in"]) != np.float32(q["scale_out"]):
            raise NotImplementedError(f"{op['name']}: requantising permute4_s8")
        return sim_permute4(x, sh, np.int8)
    if k == "gelu_s8":
        kInvSqrt2 = np.float32(0.70710678118)
        f = x.astype(np.float32) * np.float32(q["scale_in"])
        erf = np.vectorize(math.erf, otypes=[np.float32])
        y = np.float32(0.5) * f * (np.float32(1.0) + erf(f * kInvSqrt2))
        v = eg._rha64(y / np.float32(q["scale_out"]))
        return np.clip(v, q["activation_min"], q["activation_max"]).astype(np.int8)
    if k == "softmax_s8":
        xi = x.reshape(sh["M"], sh["K"]).astype(np.float32)
        z = (xi - xi.max(axis=1, keepdims=True)) * np.float32(q["scale_in"])
        e = eg._exp32(z)
        ssum = np.add.accumulate(e, axis=1, dtype=np.float32)[:, -1:]
        v = eg._rha64((e / ssum) / np.float32(q["scale_out"]))
        return np.clip(v, -128, 127).astype(np.int8)
    if k == "matmul_b_s8":
        B_, M_, K_, N_ = sh["B"], sh["M"], sh["K"], sh["N"]
        tb = int(q.get("transpose_b", sh.get("transpose_b", 0)))
        a = acts[op["inputs"][0]].reshape(B_, M_, K_).astype(np.int64)
        bm = acts[op["inputs"][1]]
        bm = (bm.reshape(B_, N_, K_).transpose(0, 2, 1) if tb else bm.reshape(B_, K_, N_)).astype(np.int64)
        acc = a @ bm
        total = (np.float32(q["scale_a"]) * np.float32(q["scale_b"])) / (
            np.float32(q["scale_out"]) * np.float32(q.get("scale_div_sqrt_dk", 1.0)))
        v = eg._rha64(acc.astype(np.float32) * np.float32(total))
        return np.clip(v, q["activation_min"], q["activation_max"]).astype(np.int8).reshape(-1)
    if k == "rope_s8":
        T_, H_, D_, R_ = sh["T"], sh["H"], sh["D"], sh["R"]
        f = x.reshape(T_, H_, D_).astype(np.float32) * np.float32(q["scale_in"])
        c = W[op["weight"]].astype(np.float32).reshape(T_, 1, R_ // 2)
        s = W[op["bias"]].astype(np.float32).reshape(T_, 1, R_ // 2)
        y = f.copy()
        x0, x1 = f[:, :, 0:R_:2], f[:, :, 1:R_:2]
        y[:, :, 0:R_:2] = x0 * c + (-x1) * s
        y[:, :, 1:R_:2] = x1 * c + x0 * s
        v = eg._rha64(y / np.float32(q["scale_out"]))
        return np.clip(v, q["activation_min"], q["activation_max"]).astype(np.int8).reshape(-1)
    raise NotImplementedError(f"extract_q16 simulator: op kind {k}")


def simulate(ir: dict, W: dict, x16: np.ndarray, dump: dict | None = None) -> np.ndarray:
    acts = {ir["input"]["tensor"]: x16.reshape(-1)}
    for op in ir["ops"]:
        k = op["op"]
        sh = op.get("shape", {})
        x = acts[op["inputs"][0]]
        if k == "split16_s8":
            y = split16(x, op["quant"]["part"])
        elif k == "mrcombine_s16":
            zs = [acts[t].reshape(sh["N"], -1, sh["OH"] * sh["OW"]) for t in op["inputs"]]
            y = mrcombine(zs, sh, W[op["gidx"]], W[op["lidx"]], W[op["ratios"]]).reshape(-1)
        elif k == "lut16_s16":
            y = sim_lut16_s16(x, W, op)
        elif k == "lut16_pc_s8":
            y = sim_lut16_pc_s8(x, sh, W, op)
        elif k == "groupnorm_s16":
            y = sim_groupnorm_s16(x, sh, W, op)
        elif k in ("layernorm_pc_s8", "layernorm_s16_s8"):
            y = sim_layernorm(x, sh, W, op)
        elif k in ("add_pc_s8", "add_s16_pc_s8"):
            y = sim_add_pc(x, acts[op["inputs"][1]], sh, W, op)
        elif k == "conv2d_s16_pc":
            y = sim_conv2d_s16_pc(x, sh, W, op)
        elif k == "permute4_s16":
            y = sim_permute4(x, sh, np.int16)
        else:
            y = _sim_stock(op, acts, W)
        acts[op["outputs"][0]] = np.asarray(y).reshape(-1)
    if dump is not None:
        dump.update(acts)
    return acts[ir["output"]["tensors"][0]]


# ---------------------------------------------------------------------------------------
# lowering
# ---------------------------------------------------------------------------------------
def _gelu64(x):
    from scipy.special import erf  # noqa: PLC0415
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def _mult_shift(real: float):
    m, s = eg._requantize_multiplier_shift(float(real))
    return int(m), int(s)


class _Lowering:
    def __init__(self, model, plan, ir, W):
        self.model, self.plan, self.ir, self.W = model, plan, ir, W
        self.P = plan["tensors"]
        self.T = ir["tensors"]
        self.ops_out = []

    # ---- tensor metadata ------------------------------------------------------------
    def _set(self, name, shape, dtype, scale, scale_pc=None, axis=None):
        meta = {"shape": list(shape), "dtype": dtype, "quant": {"scale": float(scale), "zero_point": 0}}
        if scale_pc is not None:
            meta["quant"]["scale_pc"] = [float(v) for v in scale_pc]
            meta["quant"]["axis"] = int(axis)
        self.T[name] = meta

    def q(self, name):
        """(dtype, scalar scale, per-channel vector or None, axis)"""
        m = self.T[name]
        qq = m["quant"]
        return (m["dtype"], qq["scale"], (np.asarray(qq["scale_pc"], dtype=np.float64)
                                          if "scale_pc" in qq else None), qq.get("axis"))

    def plan_meta(self, name, shape):
        """metadata of a plan tensor from its plan entry (rows vectors applied later)"""
        p = self.P[name]
        if p["kind"] == "pt":
            self._set(name, shape, "i16" if p["bits"] == 16 else "i8", p["range"] / _QMAX[p["bits"]])
        elif p["kind"] == "pc":
            s = np.maximum(np.asarray(p["ranges"], dtype=np.float64), 1e-8) / 127.0
            self._set(name, shape, "i8", float(s.max()), s, p["dim"])
        else:
            raise ValueError(name)

    # ---- helpers ------------------------------------------------------------------
    def module(self, op):
        return self.model.get_submodule(op["name"])

    def emit(self, op):
        self.ops_out.append(op)

    def _chan_vec(self, name):
        """(dtype, per-channel scale over the LAST axis) -- a per-tensor scale repeated"""
        dt, s, v, ax = self.q(name)
        shape = self.T[name]["shape"]
        C = shape[-1]
        if v is None:
            return dt, np.full(C, s, dtype=np.float64)
        if ax != len(shape) - 1:
            raise NotImplementedError(f"{name}: per-channel axis {ax} is not the last of {shape}")
        return dt, v

    # ---- passes -------------------------------------------------------------------
    def run(self):
        rows = self.plan.get("rows", {})
        for op in self.ir["ops"]:
            k = op["op"]
            out = op["outputs"][0]
            inp = op["inputs"][0]
            shape_out = self.T[out]["shape"]
            if k == "conv2d_s8":
                p = self.P.get(out)
                wgrid = self.plan["weights"].get(op["name"], {"grid": "pt"})
                if p and p["kind"] == "multi":
                    self.split_conv(op, p, wgrid)
                elif p and p["kind"] == "pt" and p["bits"] == 16:
                    self.conv_s16_pc(op, p, wgrid)
                else:
                    if wgrid["grid"] != "pt" or self.q(inp)[0] != "i8":
                        raise NotImplementedError(f"{op['name']}: conv lowering for {p} / {wgrid}")
                    self.emit(op)
            elif k in ("tanh_s8", "gelu_s8") and self.q(inp)[0] == "i16":
                self.lut16(op, k)
            elif k == "groupnorm_s8":
                self.groupnorm16(op)
            elif k in ("permute4_s8", "view"):
                self.passthrough(op)
            elif k == "add_s8":
                self.add_pc(op)
            elif k == "layernorm_s8":
                dt, s, v, ax = self.q(inp)
                if dt == "i8" and v is None:
                    self.emit(op)
                else:
                    self.layernorm(op)
            elif k == "linear_s8":
                wgrid = self.plan["weights"].get(op["name"], {"grid": "pt"})
                if self.q(inp)[0] != "i8" or self.q(inp)[2] is not None:
                    raise NotImplementedError(f"{op['name']}: linear on {self.q(inp)[:1]}")
                if wgrid["grid"] == "pc":
                    self.linear_pc(op)
                elif wgrid["grid"] == "pt":
                    self.emit(op)
                else:
                    raise NotImplementedError(f"{op['name']}: linear weight grid {wgrid['grid']}")
                if op["name"] in rows:
                    dt, s, _, _ = self.q(out)
                    a = np.asarray(rows[op["name"]], dtype=np.float64)
                    self._set(out, shape_out, "i8", s, s * a, len(shape_out) - 1)
            # silu_s8 BELONGS HERE AND NOWHERE ELSE, and finding that out was one read.
            # The lut16 branch above is for an int16 input -- a 65,536-entry table -- and the
            # decoder's silu input is int8.  At int8 with per-tensor scales the rule for a
            # pointwise op is `emit(op)`: pass it through unchanged, exactly as gelu_s8 and
            # tanh_s8 already are.  So the missing rule is one identifier, it shares no
            # arithmetic with any table builder, and int_silu_s8_table is a separate job that
            # neither needs this nor is needed by it.  The guard below still refuses an int16
            # or per-channel silu, which is the case that WOULD need a lut16 rule.
            # mul_s8 and cat2_c1_s8 BELONG HERE FOR THE SAME REASON silu_s8 does, and the
            # guard three lines below is the test: every input int8, none per-channel.  Checked
            # against the decoder's own IR rather than assumed -- mul_s8 288 int8 inputs over
            # 144 ops, cat2_c1_s8 552 over 276, zero per-channel in either.  A binary op needs
            # no q16 lowering when its operands are already per-tensor int8: the kernel it keeps
            # does its own requantise, exactly as the unary pointwise ops do.  The guard still
            # refuses an int16 or per-channel mul/cat2, which is the safe failure direction.
            elif k in ("gelu_s8", "softmax_s8", "matmul_b_s8", "rope_s8", "tanh_s8",
                       "silu_s8", "mul_s8", "cat2_c1_s8"):
                for t in op["inputs"]:
                    dt, s, v, ax = self.q(t)
                    if dt != "i8" or v is not None:
                        raise NotImplementedError(f"{op['name']}: {k} on {dt} per-channel={v is not None}")
                self.emit(op)
            else:
                raise NotImplementedError(f"extract_q16: no lowering rule for {k} ({op['name']})")
        self.ir["ops"] = self.ops_out

    def _prepare_input(self, name):
        if name == self.ir["input"]["tensor"] and name in self.P:
            self.plan_meta(name, self.T[name]["shape"])

    def split_conv(self, op, p, wgrid):
        inp = op["inputs"][0]
        out = op["outputs"][0]
        self._prepare_input(inp)
        dt, s_in, v, _ = self.q(inp)
        if dt != "i16" or v is not None:
            raise NotImplementedError(f"{op['name']}: split dispatch needs a per-tensor int16 input")
        sh = dict(op["shape"])
        mod = self.module(op)
        Wf = mod.weight.detach().float()
        bf = (mod.bias.detach().double().cpu().numpy() if mod.bias is not None
              else np.zeros(Wf.shape[0], dtype=np.float64))
        OC = int(Wf.shape[0])
        groups = ([list(range(OC))] if wgrid["grid"] == "pt" else
                  [sorted(g) for g in wgrid["groups"]] if wgrid["grid"] == "rg" else None)
        if groups is None:
            raise NotImplementedError(f"{op['name']}: split dispatch with weight grid {wgrid['grid']}")
        ratios = sorted(int(r) for r in p["ratios"])
        t_c = p["range"] / 127.0
        base = op["name"]
        hi, lo = f"{out}__hi", f"{out}__lo"
        in_shape = self.T[inp]["shape"]
        for nm, part in ((hi, "hi"), (lo, "lo")):
            self._set(nm, in_shape, "i8", s_in * (256.0 if part == "hi" else 1.0))
            self.emit({"name": f"{base}.split_{part}", "op": "split16_s8", "inputs": [inp], "outputs": [nm],
                       "shape": {"n": int(np.prod(in_shape))}, "quant": {"part": part}})
        gidx = np.zeros(OC, dtype=np.int32)
        lidx = np.zeros(OC, dtype=np.int32)
        ins = []
        for g, rows_g in enumerate(groups):
            for j, c in enumerate(rows_g):
                gidx[c], lidx[c] = g, j
            Wg = Wf[rows_g]
            sw = eg._scale_from_max_abs(Wg)
            wq = eg._quantize_per_tensor_sym(Wg, sw)
            bL = _rint(bf[rows_g] / (s_in * sw))
            if np.abs(bL).max(initial=0) >= 2 ** 31:
                raise OverflowError(f"{base} group {g}: L bias exceeds int32")
            keys = {}
            for part, bias in (("H", np.zeros(len(rows_g), dtype=np.int32)), ("L", bL.astype(np.int32))):
                wk, bk = f"{base}.g{g}.{part}.weight_q", f"{base}.g{g}.{part}.bias_q"
                self.W[wk] = wq.copy()
                self.W[bk] = bias
                keys[part] = (wk, bk)
            for k_, r in enumerate(ratios):
                t_k = t_c / r
                for part, src, mul in (("H", hi, 256.0), ("L", lo, 1.0)):
                    real = mul * s_in * sw / t_k
                    m_, s_ = _mult_shift(real)
                    nm = f"{out}__g{g}_r{k_}_{part}"
                    o_shape = [sh["N"], len(rows_g), sh["OH"], sh["OW"]]
                    self._set(nm, o_shape, "i8", t_k)
                    csh = dict(sh)
                    csh["OC"] = len(rows_g)
                    self.emit({"name": f"{base}.g{g}.r{k_}.{part}", "op": "conv2d_s8", "inputs": [src],
                               "outputs": [nm], "weight": keys[part][0], "bias": keys[part][1],
                               "shape": csh,
                               "quant": {"input_offset": 0, "filter_offset": 0, "output_offset": 0,
                                         "output_multiplier": m_, "output_shift": s_,
                                         "activation_min": -128, "activation_max": 127,
                                         "real_multiplier": real}})
                    ins.append(nm)
        for key, arr in (("gidx", gidx), ("lidx", lidx),
                         ("goc", np.asarray([len(g) for g in groups], dtype=np.int32)),
                         ("ratios", np.asarray(ratios, dtype=np.int32))):
            self.W[f"{base}.mrc.{key}"] = arr
        t_out = t_c / ratios[-1]
        self._set(out, self.T[out]["shape"], "i16", t_out)
        self.emit({"name": f"{base}.combine", "op": "mrcombine_s16", "inputs": ins, "outputs": [out],
                   "gidx": f"{base}.mrc.gidx", "lidx": f"{base}.mrc.lidx", "goc": f"{base}.mrc.goc",
                   "ratios": f"{base}.mrc.ratios",
                   "shape": {"N": sh["N"], "OC": OC, "OH": sh["OH"], "OW": sh["OW"],
                             "G": len(groups), "R": len(ratios)},
                   "quant": {"range": p["range"], "ratios": ratios}})
        for key in (op["weight"], op["bias"]):
            self.W.pop(key, None)

    def conv_s16_pc(self, op, p, wgrid):
        inp, out = op["inputs"][0], op["outputs"][0]
        self._prepare_input(inp)
        dt, s_in, v, _ = self.q(inp)
        if dt != "i16" or v is not None or wgrid["grid"] != "pc":
            raise NotImplementedError(f"{op['name']}: conv2d_s16_pc needs int16 per-tensor input and per-row weights")
        mod = self.module(op)
        Wf = mod.weight.detach().float().cpu().numpy()
        bf = (mod.bias.detach().double().cpu().numpy() if mod.bias is not None
              else np.zeros(Wf.shape[0], dtype=np.float64))
        OC = Wf.shape[0]
        sw = np.maximum(np.abs(Wf).reshape(OC, -1).max(axis=1), 1e-8) / 127.0
        wq = np.clip(np.rint(Wf.reshape(OC, -1) / sw[:, None].astype(np.float32)), -127, 127).astype(np.int8)
        s_out = p["range"] / 32767.0
        mult = np.zeros(OC, dtype=np.int32)
        shift = np.zeros(OC, dtype=np.int32)
        for c in range(OC):
            mult[c], shift[c] = _mult_shift(s_in * float(sw[c]) / s_out)
            if 31 + int(shift[c]) < 1:
                raise OverflowError(f"{op['name']} row {c}: multiplier >= 2^31")
        base = op["name"]
        self.W.pop(op["weight"], None)
        self.W.pop(op["bias"], None)
        keys = {k: f"{base}.{k}" for k in ("weight_q16pc", "bias_q64", "mult", "shift")}
        self.W[keys["weight_q16pc"]] = wq.reshape(Wf.shape)
        self.W[keys["bias_q64"]] = _rint(bf / (s_in * sw)).astype(np.int64)
        self.W[keys["mult"]] = mult
        self.W[keys["shift"]] = shift
        self._set(out, self.T[out]["shape"], "i16", s_out)
        self.emit({"name": base, "op": "conv2d_s16_pc", "inputs": [inp], "outputs": [out],
                   "weight": keys["weight_q16pc"], "bias": keys["bias_q64"],
                   "mult": keys["mult"], "shift": keys["shift"], "shape": dict(op["shape"])})

    def lut16(self, op, kind):
        inp, out = op["inputs"][0], op["outputs"][0]
        dt, s_in, _, _ = self.q(inp)
        f = np.tanh if kind == "tanh_s8" else _gelu64
        x = np.arange(-32768, 32768, dtype=np.float64) * float(s_in)
        y = f(x)
        p = self.P[out]
        shape = self.T[out]["shape"]
        base = op["name"]
        if p["kind"] == "pt" and p["bits"] == 16:
            s_out = p["range"] / 32767.0
            self._set(out, shape, "i16", s_out)
            self.W[f"{base}.lut16"] = np.clip(_rint(y / s_out), -32768, 32767).astype(np.int16)
            self.emit({"name": base, "op": "lut16_s16", "inputs": [inp], "outputs": [out],
                       "weight": f"{base}.lut16", "shape": {"n": int(np.prod(shape)), "fn": kind[:4]}})
        elif p["kind"] == "pc" and p["bits"] == 8:
            self.plan_meta(out, shape)
            _, _, s_c, ax = self.q(out)
            if ax != 1 or len(shape) != 4:
                raise NotImplementedError(f"{base}: per-channel LUT expects NCHW channel axis 1")
            s_ref = float(s_c.max())
            s_c = np.maximum(s_c, s_ref * 2.0 ** -15)
            Q = _rint(y / s_ref * 65536.0)
            if np.abs(Q).max() >= 2 ** 31:
                raise OverflowError(f"{base}: LUT exceeds int32")
            self.W[f"{base}.lut16q"] = Q.astype(np.int32)
            self.W[f"{base}.mult_pc"] = _rint(s_ref / s_c * 65536.0).astype(np.int64)
            N, C = shape[0], shape[1]
            self.emit({"name": base, "op": "lut16_pc_s8", "inputs": [inp], "outputs": [out],
                       "weight": f"{base}.lut16q", "mult": f"{base}.mult_pc",
                       "shape": {"N": N, "C": C, "HW": int(np.prod(shape[2:])), "fn": kind[:4]}})
        else:
            raise NotImplementedError(f"{base}: LUT to {p}")

    def groupnorm16(self, op):
        inp, out = op["inputs"][0], op["outputs"][0]
        dt, s_in, v, _ = self.q(inp)
        p = self.P[out]
        if dt != "i16" or v is not None or not (p["kind"] == "pt" and p["bits"] == 16):
            raise NotImplementedError(f"{op['name']}: groupnorm lowering needs int16 in and out")
        mod = self.module(op)
        sh = op["shape"]
        n = sh["C"] * sh["H"] * sh["W"]
        s_out = p["range"] / 32767.0
        base = op["name"]
        g = mod.weight.detach().double().cpu().numpy()
        b = mod.bias.detach().double().cpu().numpy()
        self.W.pop(op["weight"], None)
        self.W.pop(op["bias"], None)
        self.W[f"{base}.gmul"] = _rint(g / s_out * 65536.0).astype(np.int64)
        self.W[f"{base}.badd"] = _rint(b / s_out * 65536.0).astype(np.int64)
        E = max(1, int(_rint(float(mod.eps) * float(n) ** 2 / float(s_in) ** 2)))
        self._check_norm(base, n, E, self.W[f"{base}.gmul"], self.W[f"{base}.badd"], 15)
        self._set(out, self.T[out]["shape"], "i16", s_out)
        self.emit({"name": base, "op": "groupnorm_s16", "inputs": [inp], "outputs": [out],
                   "gmul": f"{base}.gmul", "badd": f"{base}.badd",
                   "shape": dict(sh), "quant": {"eps_q": E, "eps": float(mod.eps)}})

    def _check_norm(self, base, K, E, g, b, in_bits):
        if E >= 2 ** 63:
            raise OverflowError(f"{base}: eps_q {E} exceeds int64")
        tmax = math.sqrt(K) * 65536.0 * 1.001
        if tmax * float(np.abs(g).max()) + float(np.abs(b).max()) * 65536.0 >= 2.0 ** 62:
            raise OverflowError(f"{base}: affine product may exceed int64")

    def layernorm(self, op):
        inp, out = op["inputs"][0], op["outputs"][0]
        dt, s, v, ax = self.q(inp)
        sh = op["shape"]
        M, K = sh["M"], sh["K"]
        mod = self.module(op)
        base = op["name"]
        p = self.P[out]
        if not (p["kind"] == "pt" and p["bits"] == 8):
            raise NotImplementedError(f"{base}: layernorm output {p}")
        s_out = p["range"] / 127.0
        g = (mod.weight.detach().double().cpu().numpy() if mod.weight is not None else np.ones(K))
        b = (mod.bias.detach().double().cpu().numpy() if getattr(mod, "bias", None) is not None else np.zeros(K))
        self.W.pop(op["weight"], None)
        self.W.pop(op.get("bias") or "", None)
        self.W[f"{base}.gmul"] = _rint(g / s_out * 65536.0).astype(np.int64)
        self.W[f"{base}.badd"] = _rint(b / s_out * 65536.0).astype(np.int64)
        eps = float(op["quant"]["eps"])
        new = {"name": base, "inputs": [inp], "outputs": [out], "gmul": f"{base}.gmul", "badd": f"{base}.badd",
               "shape": {"M": M, "K": K}}
        if dt == "i16" and v is None:
            E = max(1, int(_rint(eps * K * K / (s * s))))
            new["op"] = "layernorm_s16_s8"
        elif dt == "i8":
            _, vec = self._chan_vec(inp)
            s_ref = float(vec.max())
            umul = _rint(vec / s_ref * 2.0 ** _LN_F)
            if umul.min() < 1:
                raise OverflowError(f"{base}: a channel scale is below 2^-24 of the largest")
            self.W[f"{base}.umul"] = umul.astype(np.int32)
            new["umul"] = f"{base}.umul"
            E = max(1, int(_rint(eps * K * K * (2.0 ** _LN_F / s_ref) ** 2)))
            new["op"] = "layernorm_pc_s8"
        else:
            raise NotImplementedError(f"{base}: layernorm on {dt}")
        self._check_norm(base, K, E, self.W[f"{base}.gmul"], self.W[f"{base}.badd"], 8)
        new["quant"] = {"eps_q": E, "eps": eps}
        self._set(out, self.T[out]["shape"], "i8", s_out)
        self.emit(new)

    def add_pc(self, op):
        a_n, b_n = op["inputs"]
        out = op["outputs"][0]
        p = self.P[out]
        shape = self.T[out]["shape"]
        if p["kind"] != "pc" or p["dim"] != len(shape) - 1:
            if p["kind"] == "pt" and p["bits"] == 8 and all(self.q(t)[0] == "i8" and self.q(t)[2] is None
                                                            for t in (a_n, b_n)):
                self.emit(op)
                return
            raise NotImplementedError(f"{op['name']}: add output {p['kind']} on dim {p.get('dim')}")
        self.plan_meta(out, shape)
        _, so = self._chan_vec(out)
        da, sa = self._chan_vec(a_n)
        db, sb = self._chan_vec(b_n)
        if da == "i8" and db == "i16":
            a_n, b_n, da, db, sa, sb = b_n, a_n, db, da, sb, sa
        if db != "i8" or da not in ("i8", "i16"):
            raise NotImplementedError(f"{op['name']}: add of {da} and {db}")
        ma, mb = _rint(sa / so * 2.0 ** 24), _rint(sb / so * 2.0 ** 24)
        lim = 2.0 ** 62 / 2.0 ** 15
        if max(ma.max(), mb.max()) >= lim:
            raise OverflowError(f"{op['name']}: add multiplier too large")
        base = op["name"]
        self.W[f"{base}.amul"] = ma.astype(np.int64)
        self.W[f"{base}.bmul"] = mb.astype(np.int64)
        self.emit({"name": base, "op": "add_s16_pc_s8" if da == "i16" else "add_pc_s8",
                   "inputs": [a_n, b_n], "outputs": [out], "amul": f"{base}.amul", "bmul": f"{base}.bmul",
                   "shape": {"n": int(np.prod(shape)), "C": int(shape[-1])}})

    def linear_pc(self, op):
        inp, out = op["inputs"][0], op["outputs"][0]
        mod = self.module(op)
        Wf = mod.weight.detach().float().cpu().numpy()
        bf = (mod.bias.detach().double().cpu().numpy() if mod.bias is not None
              else np.zeros(Wf.shape[0], dtype=np.float64))
        N = Wf.shape[0]
        sw = np.maximum(np.abs(Wf).reshape(N, -1).max(axis=1), 1e-8) / 127.0
        wq = np.clip(np.rint(Wf / sw[:, None].astype(np.float32)), -127, 127).astype(np.int8)
        _, s_in, _, _ = self.q(inp)
        s_out = self.q(out)[1]
        mult = np.zeros(N, dtype=np.int32)
        shift = np.zeros(N, dtype=np.int32)
        for c in range(N):
            mult[c], shift[c] = _mult_shift(s_in * float(sw[c]) / s_out)
        bq = _rint(bf / (s_in * sw))
        if np.abs(bq).max(initial=0) >= 2 ** 31:
            raise OverflowError(f"{op['name']}: bias exceeds int32")
        base = op["name"]
        self.W[op["weight"]] = wq
        self.W[op["bias"]] = bq.astype(np.int32)
        mk, sk = f"{base}.output_multiplier_per_oc", f"{base}.output_shift_per_oc"
        self.W[mk], self.W[sk] = mult, shift
        new = dict(op)
        new["op"] = "linear_s8_pc"
        qq = dict(op["quant"])
        qq.pop("output_multiplier", None)
        qq.pop("output_shift", None)
        qq["output_multiplier_per_oc_key"] = mk
        qq["output_shift_per_oc_key"] = sk
        new["quant"] = qq
        self.emit(new)

    def passthrough(self, op):
        inp, out = op["inputs"][0], op["outputs"][0]
        dt, s, v, ax = self.q(inp)
        shape_in = self.T[inp]["shape"]
        shape_out = self.T[out]["shape"]
        new_ax = None
        if v is not None:
            if op["op"] == "permute4_s8":
                sh = op["shape"]
                pad = 4 - len(shape_in)
                p4 = [sh["p0"], sh["p1"], sh["p2"], sh["p3"]]
                new_ax = p4.index(ax + pad) - (4 - len(shape_out))
            else:
                nz_in = [i for i, d in enumerate(shape_in) if d != 1]
                nz_out = [i for i, d in enumerate(shape_out) if d != 1]
                if ([shape_in[i] for i in nz_in] != [shape_out[i] for i in nz_out]) or ax not in nz_in:
                    raise NotImplementedError(f"{op['name']}: view moves a per-channel axis")
                new_ax = nz_out[nz_in.index(ax)]
        self._set(out, shape_out, dt, s, v, new_ax)
        if op["op"] == "permute4_s8" and dt == "i16":
            new = dict(op)
            new["op"] = "permute4_s16"
            new["quant"] = {}
            self.emit(new)
        else:
            if op["op"] == "permute4_s8":
                op = dict(op)
                op["quant"] = dict(op["quant"], scale_in=s, scale_out=s)
            self.emit(op)


def _range_overrides(plan: dict) -> dict:
    ro = {}
    for k, p in plan["tensors"].items():
        if p["kind"] in ("pt", "multi"):
            ro[k] = float(p["range"])
        elif p["kind"] == "pc":
            ro[k] = float(max(p["ranges"]))
    return ro


def _apply_rows(model, plan):
    m = copy.deepcopy(model)
    for mod, a in plan.get("rows", {}).items():
        sub = m.get_submodule(mod)
        at = torch.tensor(a, dtype=torch.float32)
        with torch.no_grad():
            sub.weight.copy_(sub.weight / at.view([-1] + [1] * (sub.weight.dim() - 1)))
            if sub.bias is not None:
                sub.bias.copy_(sub.bias / at)
    return m


def extract_q16(model, sample_input, name: str, out_dir: str, plan: dict, fusion_target=None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    m = _apply_rows(model.eval(), plan)
    dump_env = os.environ.pop("MB_INT8_DUMP_ACTIVATIONS", None)
    try:
        with tempfile.TemporaryDirectory(prefix="extract_q16_") as tmp:
            eg.extract_int8(m, sample_input, name, tmp, calibration_samples=None,
                            fusion_target=fusion_target, range_overrides=_range_overrides(plan))
            ir = json.load(open(os.path.join(tmp, "graph.json")))
            W = dict(np.load(os.path.join(tmp, "weights.npz")))
    finally:
        if dump_env is not None:
            os.environ["MB_INT8_DUMP_ACTIVATIONS"] = dump_env
    for op in ir["ops"]:
        for f in ("dispatch_id", "hardware_target", "depends_on"):
            op.pop(f, None)
    low = _Lowering(m, plan, ir, W)
    low.run()
    ir["dispatches"] = eg._annotate_dispatches(ir["ops"])
    ir["quant"] = "int8"
    ir["q16_plan"] = {k: plan.get(k) for k in ("candidate", "quant_fix_key", "record")}
    # keep only referenced weights
    used = set()
    for op in ir["ops"]:
        for k, v in op.items():
            if isinstance(v, str) and v in W:
                used.add(v)
        for k, v in op.get("quant", {}).items():
            if isinstance(v, str) and v in W:
                used.add(v)
    W = {k: v for k, v in W.items() if k in used}
    # the input, on its plan grid
    xin = ir["input"]["tensor"]
    s_in = ir["tensors"][xin]["quant"]["scale"]
    if ir["tensors"][xin]["dtype"] != "i16":
        raise NotImplementedError("extract_q16: the plan's input is not int16")
    x = sample_input.detach().cpu().double().numpy()
    x16 = np.clip(_rint(x / s_in), -32768, 32767).astype(np.int16).reshape(-1)
    acts = {}
    out = simulate(ir, W, x16, acts)
    with open(os.path.join(out_dir, "graph.json"), "w") as f:
        json.dump(ir, f, indent=2)
    np.savez(os.path.join(out_dir, "weights.npz"), **W)
    np.savez(os.path.join(out_dir, "io.npz"), input=x16, output=out.astype(np.int8), input0=x16)
    if dump_env:
        np.savez(dump_env, **{k: np.asarray(v).reshape(-1) for k, v in acts.items()})
    print(f"[extract_q16] {name}: {len(ir['ops'])} ops, {len(ir['dispatches'])} dispatches, "
          f"{len(W)} weight arrays -> {out_dir}")
    return ir


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fusion-target", default=None)
    a = ap.parse_args()
    import importlib  # noqa: PLC0415
    mm = importlib.import_module(f"modelblaster.models.{a.model}")
    plan = json.load(open(a.plan))
    extract_q16(mm.get_model(), mm.get_sample_input(), a.model, a.out_dir, plan, a.fusion_target)


if __name__ == "__main__":
    main()
