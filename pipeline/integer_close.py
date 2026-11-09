#!/usr/bin/env python3
"""Rewrite float-requant ops into integer-closed kinds (integer_kernels.py), at build time.

  integer_close.py GRAPH.json WEIGHTS.npz --out-dir DIR [--widths WIDTHS.json] [--a16-headroom 8]
                   [--passes ...]

Every scale becomes an integer multiplier/shift or a table, so no op in the rewritten graph does float arithmetic at
run time (the one exception is a model's CPU tail op, e.g. vint_action_post_acc). The constants are derived here from
the IR's own scales with integer_kernels' contract formulas; a qparams file can then replace any of them verbatim
(apply_qparams.py).

Scales. --scales SCALES.json fixes tensor scales (e.g. a network input's) before anything is derived.

Widths. WIDTHS.json maps IR tensor names to bit widths: 8 (default), 15 or 16 (held in int16), or 64 (raw
accumulators, linear outputs only). A widened tensor's scale becomes headroom * 127 * s8 / (2^(b-1) - 1): the int8
calibration's range with --a16-headroom (default 8) to spare. Ops touching a non-int8 tensor take width-generic kinds.

Passes (all on by default):
  swish    sigmoid(x) -> mul(sigma, x), sigma read only by that mul -> one SiLU op on x
  lut      the remaining sigmoid / gelu -> an activation op
           (an activation op is lut_s8 for an int8 input and gate_q16 for an int16 input)
  mul_c1   mul_c1_s8 -> mul_c1_q
  gap      adaptive_avg_pool2d_s8 to 1x1 -> gap_q
  add      add_s8 -> add_q; an add reading a placement concat of differently scaled rows gets a per-row multiplier
  concat   cat2_c1_s8 -> cat2_c1_q (placement) when its inputs share the output scale, or when its only consumer is
           an add (which then absorbs the row scales)
  wide     conv / depthwise / linear touching a non-int8 tensor -> conv2d_q / linear_q (a preceding pad is folded
           into the conv's pads); a ReLU after such a linear is fused into it
  attn     matmul_b_s8 -> matmul_b_q_s8, softmax_s8 -> softmax_q_s8
  moves    slice_c_s8 / permute4_s8 -> slice_c_q / permute4_q, pure moves (the output keeps the input's grid)
  norm     layer_norm_s8 with int16 in and out -> layer_norm_q16
  raw      vint_action_post on raw (64-bit) linear outputs -> vint_action_post_acc
  dce      ops whose outputs nothing reads (and that are not graph outputs) are removed

Build-time rounding: tables and multipliers round half away from zero; requant multipliers are M in [2^30, 2^31) with
s = -exponent (s < 0 is a pre-shift); ADD uses the largest S <= 46 with every multiplier <= 2^23 - 1.

The pass is opt-in. A graph that does not run it is unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

PASSES = ("swish", "lut", "mul_c1", "gap", "add", "concat", "wide", "attn", "moves", "norm", "raw", "dce")
DTYPE = {8: "i8", 15: "i16", 16: "i16", 32: "i32", 64: "i64"}
NPDT = {"i8": np.int8, "i16": np.int16, "i32": np.int32, "i64": np.int64}


def rha(v):
    v = np.asarray(v, dtype=np.float64)
    return (np.sign(v) * np.floor(np.abs(v) + 0.5)).astype(np.int64)


def quant_mult(m: float) -> tuple[int, int]:
    """Real multiplier m > 0 -> (M, s) with m = M * 2^-(31 + s), M in [2^30, 2^31); s < 0 is a pre-shift."""
    if m <= 0:
        return 0, 0
    f, e = math.frexp(float(m))
    M = int(rha(f * 2 ** 31))
    if M == 2 ** 31:
        M, e = 2 ** 30, e + 1
    s = -e
    if s > 62:
        return 0, 0
    return M, s


def add_mults(ms: list[float]) -> tuple[list[int], int]:
    mx = max(ms)
    S = 46
    while S > 1 and int(rha(mx * 2 ** S)) > 2 ** 23 - 1:
        S -= 1
    return [int(rha(v * 2 ** S)) for v in ms], S


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def _silu(v):
    return v * _sigmoid(v)


def _gelu(v):
    return 0.5 * v * (1.0 + np.vectorize(math.erf)(v / math.sqrt(2.0)))


def _phi(v):
    return 0.5 * (1.0 + np.vectorize(math.erf)(v / math.sqrt(2.0)))


def lut8_table(f, s_in: float, s_out: float) -> np.ndarray:
    v = (np.arange(256) - 128) * float(s_in)
    return np.clip(rha(f(v) / float(s_out)), -128, 127).astype(np.int8)


def gate_table(kind: str) -> np.ndarray:
    """The shared 65-entry Q0.15 table: sigmoid over [-8, 8] (silu, sigmoid) or Phi over [-4, 4] (gelu)."""
    k = np.arange(65)
    if kind == "gelu":
        return rha(2 ** 15 * _phi(-4 + 8 * k / 64)).astype(np.int32)
    return rha(2 ** 15 * _sigmoid(-8 + 16 * k / 64)).astype(np.int32)


def rsqrt_table() -> np.ndarray:
    return rha(2 ** 15 / np.sqrt(1 + np.arange(49) / 16)).astype(np.int32)


def exp_table(s_in: float) -> np.ndarray:
    return np.minimum(65535, rha(65536 * np.exp(-np.arange(256) * float(s_in)))).astype(np.int32)


class Closer:
    def __init__(self, ir: dict, weights: dict, widths: dict | None = None, headroom: float = 8.0,
                 scales: dict | None = None):
        self.ir = ir
        self.scale_override = dict(scales or {})
        self.w = dict(weights)
        self.widths = dict(widths or {})
        self.headroom = headroom
        self.counts: dict[str, int] = {}

    # ---- tensors
    def meta(self, t):
        return self.ir["tensors"][t]

    def scale(self, t: str) -> float:
        return float(self.meta(t)["quant"]["scale"])

    def set_scale(self, t, s):
        self.meta(t).setdefault("quant", {})["scale"] = float(s)

    def bits(self, t) -> int:
        return int(self.widths.get(t, 8))

    def consumers(self, t: str) -> list[dict]:
        return [op for op in self.ir["ops"] if t in op.get("inputs", [])]

    def _count(self, k):
        self.counts[k] = self.counts.get(k, 0) + 1

    def _apply_widths(self):
        for t, b in self.widths.items():
            m = self.ir["tensors"].get(t)
            if m is None:
                raise SystemExit(f"integer_close: widths names {t!r}, which the graph does not have")
            b = int(b)
            if b not in DTYPE:
                raise SystemExit(f"integer_close: {t}: width {b} is not one of {sorted(DTYPE)}")
            if b == 8 or m.get("dtype") == DTYPE[b]:
                continue
            if (m.get("quant") or {}).get("kind") == "constant_buffer":
                # a constant: requantize its stored codes onto the wider grid
                s8 = self.scale(t)
                v = self.w[t].astype(np.float64) * s8
                s = max(np.abs(v).max(), 1e-30) / ((1 << (b - 1)) - 1)
                self.w[t] = np.clip(rha(v / s), -(1 << (b - 1)) + 1, (1 << (b - 1)) - 1).astype(NPDT[DTYPE[b]])
                self.set_scale(t, s)
            elif b < 64:
                self.set_scale(t, self.headroom * 127 * self.scale(t) / ((1 << (b - 1)) - 1 if b != 15 else 16383))
            m["dtype"] = DTYPE[b]

    # ---- builders
    def _act(self, op_like, name, x, y, f, kind):
        """SiLU / sigmoid / GELU: lut_s8 on an int8 input, gate_q16 on an int16 one."""
        n = int(op_like["shape"]["n"])
        if self.bits(x) == 8:
            if self.bits(y) != 8:
                raise SystemExit(f"integer_close: {name}: int8 input, {self.bits(y)}-bit output")
            key = f"{name}.table"
            self.w[key] = lut8_table(f, self.scale(x), self.scale(y))
            return {"name": name, "op": "lut_s8", "precision": "int8", "inputs": [x], "outputs": [y],
                    "shape": {"n": n}, "quant": {"table_key": key}}
        if self.bits(x) != 16 or self.bits(y) != 16:
            raise SystemExit(f"integer_close: {name}: gate_q16 needs int16 in and out")
        key = f"{name}.table"
        self.w[key] = gate_table(kind)
        Mx, sx = quant_mult(self.scale(x) * 2 ** 11)
        q = {"kind": kind, "Mx": Mx, "sx": sx, "table_key": key}
        if kind == "sigmoid":
            self.set_scale(y, 2.0 ** -15)
        else:
            q["M2"], q["s2"] = quant_mult(self.scale(x) * 2.0 ** -15 / self.scale(y))
        return {"name": name, "op": "gate_q16", "precision": "int16", "inputs": [x], "outputs": [y],
                "shape": {"n": n}, "quant": q}

    def _per_oc(self, op, rm_new):
        """Re-derive per-oc (M, s) arrays in place from real multipliers."""
        q = op["quant"]
        M = np.zeros(len(rm_new), np.int32)
        s = np.zeros(len(rm_new), np.int32)
        for i, m in enumerate(rm_new):
            M[i], s[i] = quant_mult(float(m))
        self.w[q["output_multiplier_per_oc_key"]] = M
        self.w[q["output_shift_per_oc_key"]] = s

    def _w_scales(self, op, x, y):
        """Per-oc weight scales implied by the op's current multipliers and its tensors' original scales."""
        q = op["quant"]
        M = self.w[q["output_multiplier_per_oc_key"]].astype(np.float64)
        s = self.w[q["output_shift_per_oc_key"]].astype(np.float64)
        return M * 2.0 ** -(31 + s) * self._s0[y] / self._s0[x]

    # ---- the pass
    def run(self, passes=PASSES) -> dict:
        self._s0 = {t: float((m.get("quant") or {}).get("scale", 1.0)) for t, m in self.ir["tensors"].items()}
        for t, sc in self.scale_override.items():
            if t not in self.ir["tensors"]:
                raise SystemExit(f"integer_close: scales names {t!r}, which the graph does not have")
            self.set_scale(t, sc)
        self._apply_widths()
        ops = self.ir["ops"]
        out = []
        drop = set()
        outputs = set(self.ir["output"].get("tensors") or [])
        producer = {o: op for op in ops for o in op["outputs"]}
        rename = {}
        for op in ops:
            if op["name"] in drop:
                continue
            op["inputs"] = [rename.get(t, t) for t in op["inputs"]]
            k = op["op"]
            if k == "sigmoid_s8" and "swish" in passes:
                sig = op["outputs"][0]
                cons = self.consumers(sig)
                x = op["inputs"][0]
                if len(cons) == 1 and cons[0]["op"] == "mul_s8" and sorted(cons[0]["inputs"]) == sorted([sig, x]) \
                        and sig not in outputs:
                    mul = cons[0]
                    out.append(self._act(mul, mul["name"], x, mul["outputs"][0], _silu, "silu"))
                    drop.add(mul["name"])
                    self.ir["tensors"].pop(sig, None)
                    self._count("swish")
                    continue
            if k in ("sigmoid_s8", "gelu_s8") and "lut" in passes:
                f, kind = (_sigmoid, "sigmoid") if k == "sigmoid_s8" else (_gelu, "gelu")
                out.append(self._act(op, op["name"], op["inputs"][0], op["outputs"][0], f, kind))
                self._count("lut")
                continue
            if k == "mul_c1_s8" and "mul_c1" in passes:
                g, x = op["inputs"]
                y = op["outputs"][0]
                M, s = quant_mult(self.scale(x) * self.scale(g) / self.scale(y))
                out.append({"name": op["name"], "op": "mul_c1_q", "inputs": [g, x], "outputs": [y],
                            "shape": dict(op["shape"]), "quant": {"M": M, "s": s}})
                self._count("mul_c1")
                continue
            if k == "adaptive_avg_pool2d_s8" and "gap" in passes and op["shape"].get("OH") == 1 \
                    and op["shape"].get("OW") == 1:
                sh = op["shape"]
                hw = int(sh["IH"]) * int(sh["IW"])
                x, y = op["inputs"][0], op["outputs"][0]
                M, s = quant_mult(self.scale(x) / (hw * self.scale(y)))
                out.append({"name": op["name"], "op": "gap_q", "inputs": [x], "outputs": [y],
                            "shape": {"N": int(sh["N"]), "C": int(sh["C"]), "HW": hw}, "quant": {"M": M, "s": s}})
                self._count("gap")
                continue
            if k == "cat3_c1_s8" and "concat" in passes and len(set(self.scale(t) for t in op["inputs"])) == 1 \
                    and len(set(self.bits(t) for t in op["inputs"] + op["outputs"])) == 1:
                # three inputs on one grid (e.g. one tensor repeated for a batch): two placement concats
                a, b, c = op["inputs"]
                y = op["outputs"][0]
                sh = op["shape"]
                ca, cb, cc = sh["C_inputs"]
                mid = f"{op['name']}__ab"
                self.ir["tensors"][mid] = {"shape": [sh["N"], ca + cb, sh["H"], sh["W"]],
                                           "dtype": self.meta(y)["dtype"], "quant": {"scale": self.scale(a)}}
                self.set_scale(y, self.scale(a))
                geo = {"N": sh["N"], "H": sh["H"], "W": sh["W"]}
                out.append({"name": f"{op['name']}__0", "op": "cat2_c1_q", "inputs": [a, b], "outputs": [mid],
                            "shape": {**geo, "C_inputs": [ca, cb]}, "quant": {}})
                out.append({"name": op["name"], "op": "cat2_c1_q", "inputs": [mid, c], "outputs": [y],
                            "shape": {**geo, "C_inputs": [ca + cb, cc]}, "quant": {}})
                self._count("concat")
                continue
            if k == "cat2_c1_s8" and "concat" in passes:
                a, b = op["inputs"]
                y = op["outputs"][0]
                # Inputs on one grid: the concat is placement, and its output takes that grid (its own captured
                # scale is the same tensor's max-abs, and is replaced so nothing downstream reads a stale one).
                same = self.scale(a) == self.scale(b) and self.bits(a) == self.bits(b) == self.bits(y)
                if same:
                    self.set_scale(y, self.scale(a))
                cons = self.consumers(y)
                feeds_add = len(cons) == 1 and "add" in passes and y not in outputs and (
                    (cons[0]["op"] == "add_s8" and cons[0]["inputs"][0] == y)
                    or (cons[0]["op"] == "add_tile_s8" and cons[0]["inputs"][1] == y))
                if same or feeds_add:
                    sh = op["shape"]
                    rec = {"name": op["name"], "op": "cat2_c1_q", "inputs": [a, b], "outputs": [y],
                           "shape": {"N": sh["N"], "H": sh["H"], "W": sh["W"], "C_inputs": list(sh["C_inputs"])},
                           "quant": {}}
                    if feeds_add:
                        # one multiplier per row of the concat (repeating per batch item N); a row is H*W codes
                        rec["row_scales"] = ([self.scale(a)] * sh["C_inputs"][0] + [self.scale(b)] * sh["C_inputs"][1])
                        rec["row_cols"] = int(sh["H"]) * int(sh["W"])
                    out.append(rec)
                    self._count("concat")
                    continue
            if k in ("add_s8", "add_tile_s8") and "add" in passes:
                a, b = op["inputs"]
                if k == "add_tile_s8":
                    b, a = a, b          # add_tile_s8 is (tile, x): the tile is the broadcast operand
                y = op["outputs"][0]
                cat = next((o for o in out if a in o["outputs"] and o.get("row_scales")), None)
                if cat is None and k == "add_tile_s8":
                    raise SystemExit(f"integer_close: {op['name']}: add_tile_s8 only after a placement concat")
                if cat is not None:
                    rs = cat["row_scales"]
                    cols = cat["row_cols"]
                    n = int(np.prod(self.meta(y)["shape"]))
                    rows = n // cols
                    ms, S = add_mults([r / self.scale(y) for r in rs] + [self.scale(b) / self.scale(y)])
                    ma, mb = ms[:len(rs)], ms[len(rs)]
                    b_rows = int(np.prod(self.meta(b)["shape"])) // cols
                    shape = {"rows": rows, "cols": cols, "n_ma": len(rs), "b_rows": b_rows}
                else:
                    (m0, mb), S = add_mults([self.scale(a) / self.scale(y), self.scale(b) / self.scale(y)])
                    ma = [m0]
                    shape = {"rows": 1, "cols": int(op["shape"]["n"]), "n_ma": 1, "b_rows": 1}
                key = f"{op['name']}.ma"
                self.w[key] = np.array(ma, np.int32)
                out.append({"name": op["name"], "op": "add_q", "inputs": [a, b], "outputs": [y], "shape": shape,
                            "quant": {"ma_key": key, "mb": mb, "S": S}})
                self._count("add")
                continue
            if k in ("conv2d_s8_pc", "depthwise_conv2d_s8_pc") and "wide" in passes:
                x, y = op["inputs"][0], op["outputs"][0]
                if self.bits(x) != 8 or self.bits(y) != 8:
                    sh = dict(op["shape"])
                    pl = pr = sh["PW"]
                    pt = pb = sh["PH"]
                    pad = producer.get(x)
                    if pad is not None and pad["op"] == "pad_s8" and len(self.consumers(x)) == 1:
                        ps = pad["shape"]
                        pl, pr, pt, pb = pl + ps["pad_left"], pr + ps["pad_right"], pt + ps["pad_top"], \
                            pb + ps["pad_bottom"]
                        x = pad["inputs"][0]
                        out.remove(pad)
                        self.ir["tensors"].pop(pad["outputs"][0], None)
                        sh["IH"], sh["IW"] = ps["IH"], ps["IW"]
                        self._count("pad_fold")
                    sw = self._w_scales(op, op["inputs"][0], y)
                    self._per_oc(op, self.scale(x) * sw / self.scale(y))
                    if op.get("bias"):
                        self.w[op["bias"]] = rha(self.w[op["bias"]].astype(np.float64) * self._s0[op["inputs"][0]]
                                                 / self.scale(x)).astype(np.int32)
                    groups = sh["groups"]
                    ic = sh["OC"] if k.startswith("depthwise") else sh["IC"]
                    out.append({"name": op["name"], "op": "conv2d_q", "inputs": [x], "outputs": [y],
                                "weight": op["weight"], "bias": op.get("bias"),
                                "shape": {"N": sh["N"], "IC": ic, "IH": sh["IH"], "IW": sh["IW"], "OC": sh["OC"],
                                          "KH": sh["KH"], "KW": sh["KW"], "SH": sh["SH"], "SW": sh["SW"],
                                          "pad_l": pl, "pad_r": pr, "pad_t": pt, "pad_b": pb, "groups": groups},
                                "quant": {k2: op["quant"][k2] for k2 in ("output_multiplier_per_oc_key",
                                                                         "output_shift_per_oc_key")}})
                    self._count("conv_wide")
                    continue
            if k == "linear_s8_pc" and "wide" in passes:
                x, y = op["inputs"][0], op["outputs"][0]
                if self.bits(x) != 8 or self.bits(y) != 8:
                    sw = self._w_scales(op, x, y)
                    if self.bits(y) == 64:
                        self.w[f"{op['name']}.out_scale"] = (self.scale(x) * sw).astype(np.float64)
                    else:
                        self._per_oc(op, self.scale(x) * sw / self.scale(y))
                    if op.get("bias"):
                        self.w[op["bias"]] = rha(self.w[op["bias"]].astype(np.float64) * self._s0[x]
                                                 / self.scale(x)).astype(np.int32)
                    rec = {"name": op["name"], "op": "linear_q", "inputs": [x], "outputs": [y],
                           "weight": op["weight"], "bias": op.get("bias"), "shape": dict(op["shape"]),
                           "quant": {k2: op["quant"][k2] for k2 in ("output_multiplier_per_oc_key",
                                                                    "output_shift_per_oc_key")}}
                    rec["quant"]["relu"] = 0
                    cons = self.consumers(y)
                    if self.bits(y) < 64 and len(cons) == 1 and cons[0]["op"] == "relu_s8" and y not in outputs:
                        relu = cons[0]
                        rec["quant"]["relu"] = 1
                        rec["outputs"] = [relu["outputs"][0]]
                        self.meta(relu["outputs"][0]).update({"dtype": self.meta(y)["dtype"]})
                        self.set_scale(relu["outputs"][0], self.scale(y))
                        self.widths[relu["outputs"][0]] = self.bits(y)
                        drop.add(relu["name"])
                        self.ir["tensors"].pop(y, None)
                        self._count("relu_fuse")
                    out.append(rec)
                    self._count("linear_wide")
                    continue
            if k == "repeat_q":
                # a move: the copies keep the source's grid and width
                self.set_scale(op["outputs"][0], self.scale(op["inputs"][0]))
                b_in = self.bits(op["inputs"][0])
                if b_in != self.bits(op["outputs"][0]):
                    self.widths[op["outputs"][0]] = b_in
                    self.meta(op["outputs"][0])["dtype"] = DTYPE[b_in]
                out.append(op)
                continue
            if k in ("slice_c_s8", "permute4_s8") and "moves" in passes:
                # a pure move of codes: the output keeps the input's grid and width, and no float scale reaches
                # the kernel
                self.set_scale(op["outputs"][0], self.scale(op["inputs"][0]))
                b_in = self.bits(op["inputs"][0])
                if b_in != self.bits(op["outputs"][0]):
                    self.widths[op["outputs"][0]] = b_in
                    self.meta(op["outputs"][0])["dtype"] = DTYPE[b_in]
                out.append({"name": op["name"], "op": "slice_c_q" if k == "slice_c_s8" else "permute4_q",
                            "inputs": list(op["inputs"]), "outputs": list(op["outputs"]),
                            "shape": dict(op["shape"]), "quant": {}})
                self._count("moves")
                continue
            if k == "matmul_b_s8" and "attn" in passes:
                a, b = op["inputs"]
                y = op["outputs"][0]
                q = op["quant"]
                M, s = quant_mult(self.scale(a) * self.scale(b) / (self.scale(y) * float(q.get("scale_div_sqrt_dk", 1.0))))
                out.append({"name": op["name"], "op": "matmul_b_q_s8", "inputs": [a, b], "outputs": [y],
                            "shape": dict(op["shape"]), "quant": {"transpose_b": int(q.get("transpose_b", 0)),
                                                                   "M_q": M, "s_q": s}})
                self._count("matmul_b")
                continue
            if k == "softmax_s8" and "attn" in passes:
                x, y = op["inputs"][0], op["outputs"][0]
                key = f"{op['name']}.texp"
                self.w[key] = exp_table(self.scale(x))
                self.set_scale(y, 1.0 / 127)
                out.append({"name": op["name"], "op": "softmax_q_s8", "inputs": [x], "outputs": [y],
                            "shape": dict(op["shape"]), "quant": {"table_key": key}})
                self._count("softmax")
                continue
            if k == "layer_norm_s8" and "norm" in passes and self.bits(op["inputs"][0]) == 16 \
                    and self.bits(op["outputs"][0]) == 16:
                x, y = op["inputs"][0], op["outputs"][0]
                q = op["quant"]
                D = int(op["shape"]["K"])
                gam = self.w[q["gamma_key"]].astype(np.float64).ravel() * float(q["scale_gamma"])
                bet = self.w[q["beta_key"]].astype(np.float64).ravel() * float(q["scale_beta"])
                Gb = 0
                while Gb < 40 and np.abs(rha(gam * 2 ** (Gb + 1))).max() <= 2 ** 23 - 1:
                    Gb += 1
                M, s = quant_mult(2.0 ** -(13 + Gb) / self.scale(y))
                base = op["name"]
                self.w[f"{base}.gam"] = rha(gam * 2 ** Gb).astype(np.int32)
                self.w[f"{base}.bet"] = rha(bet * 2 ** (13 + Gb)).astype(np.int64)
                self.w[f"{base}.rsqrt"] = rsqrt_table()
                E = max(1, int(rha(D * D * float(q.get("eps", 1e-5)) / self.scale(x) ** 2)))
                out.append({"name": op["name"], "op": "layer_norm_q16", "inputs": [x], "outputs": [y],
                            "shape": dict(op["shape"]),
                            "quant": {"E": E, "M": M, "s": s, "Gbits": Gb, "gamma_key": f"{base}.gam",
                                      "beta_key": f"{base}.bet",
                                      "rsqrt_key": f"{base}.rsqrt"}})
                self._count("norm")
                continue
            if k == "vint_action_post" and "raw" in passes and all(self.bits(t) == 64 for t in op["inputs"]):
                d, a = op["inputs"]
                out.append({"name": op["name"], "op": "vint_action_post_acc", "inputs": [d, a],
                            "outputs": list(op["outputs"]), "shape": dict(op["shape"]),
                            "quant": {"dist_scale_key": f"{producer[d]['name']}.out_scale",
                                      "act_scale_key": f"{producer[a]['name']}.out_scale"}})
                self._count("raw_tail")
                continue
            if k == "pad_s8" and self.bits(op["inputs"][0]) != 8:
                out.append(op)    # folded into its conv when that conv is reached; refused below otherwise
                continue
            out.append(op)
        for op in out:
            op.pop("row_scales", None)
            op.pop("row_cols", None)
            if op["op"] == "pad_s8" and self.bits(op["inputs"][0]) != 8:
                raise SystemExit(f"integer_close: {op['name']}: a pad of a {self.bits(op['inputs'][0])}-bit tensor "
                                 f"that no conv absorbs")
        if "dce" in passes:
            while True:
                read = {t for op in out for t in op["inputs"]}
                dead = [op for op in out if not any(t in read or t in outputs for t in op["outputs"])]
                if not dead:
                    break
                for op in dead:
                    out.remove(op)
                    for t in op["outputs"]:
                        self.ir["tensors"].pop(t, None)
                    self._count("dce")
        for op in out:
            b15 = {t: 15 for t in op["inputs"] + op["outputs"] if self.bits(t) == 15}
            if b15:
                op.setdefault("quant", {})["bits"] = b15
        self.ir["ops"] = out
        from modelblaster.pipeline.extract_graph import _annotate_dispatches
        self.ir["dispatches"] = _annotate_dispatches(out)
        return self.counts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("graph")
    ap.add_argument("weights")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--widths", default=None, help="JSON: IR tensor name -> bits (8, 15, 16, 64)")
    ap.add_argument("--a16-headroom", type=float, default=8.0)
    ap.add_argument("--scales", default=None,
                    help="JSON: IR tensor name -> scale, set before anything is derived (e.g. a fixed input scale)")
    ap.add_argument("--passes", default=",".join(PASSES))
    a = ap.parse_args()
    ir = json.load(open(a.graph))
    w = dict(np.load(a.weights))
    widths = json.load(open(a.widths)) if a.widths else None
    c = Closer(ir, w, widths, a.a16_headroom, json.load(open(a.scales)) if a.scales else None)
    counts = c.run(tuple(p for p in a.passes.split(",") if p))
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    json.dump(c.ir, open(od / "graph.json", "w"), indent=2)
    np.savez(od / "weights.npz", **c.w)
    print(f"[integer_close] {counts} -> {od}")


if __name__ == "__main__":
    main()
