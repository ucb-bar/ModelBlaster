#!/usr/bin/env python3
"""Rewrite float-requant ops into integer-closed kinds (integer_kernels.py), at build time.

  integer_close.py GRAPH.json WEIGHTS.npz --out-dir DIR [--passes swish,lut,mul_c1,gap,add]

Every scale becomes an integer multiplier/shift or a table, so no op in the rewritten graph does float arithmetic at
run time. The constants are derived here from the IR's own scales with integer_kernels' contract formulas; a qparams
file can then replace any of them verbatim (apply_qparams.py).

Passes (all on by default):
  swish   sigmoid_s8(x) -> mul_s8(sigma, x), with sigma read only by that mul -> one lut_s8 of SiLU on x
  lut     the remaining sigmoid_s8 and gelu_s8 (int8 in, int8 out) -> lut_s8
  mul_c1  mul_c1_s8 -> mul_c1_q
  gap     adaptive_avg_pool2d_s8 to 1x1 -> gap_q
  add     add_s8 -> add_q

Build-time rounding: tables and multipliers use round half away from zero; requant multipliers are M in [2^30, 2^31)
with s = -exponent (s < 0 is a pre-shift); ADD uses the largest S <= 46 with every multiplier <= 2^23 - 1.

The pass is opt-in. A graph that does not run it is unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

PASSES = ("swish", "lut", "mul_c1", "gap", "add")


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


def lut8_table(f, s_in: float, s_out: float) -> np.ndarray:
    v = (np.arange(256) - 128) * float(s_in)
    return np.clip(rha(f(v) / float(s_out)), -128, 127).astype(np.int8)


class Closer:
    def __init__(self, ir: dict, weights: dict):
        self.ir = ir
        self.w = dict(weights)
        self.counts: dict[str, int] = {}

    def scale(self, t: str) -> float:
        return float(self.ir["tensors"][t]["quant"]["scale"])

    def consumers(self, t: str) -> list[dict]:
        return [op for op in self.ir["ops"] if t in op.get("inputs", [])]

    def _count(self, k):
        self.counts[k] = self.counts.get(k, 0) + 1

    def _lut(self, op, name, x, y, f):
        key = f"{name}.table"
        self.w[key] = lut8_table(f, self.scale(x), self.scale(y))
        n = op["shape"]["n"]
        return {"name": name, "op": "lut_s8", "precision": "int8", "inputs": [x], "outputs": [y],
                "shape": {"n": int(n)}, "quant": {"table_key": key, "scale_in": self.scale(x),
                                                  "scale_out": self.scale(y)}}

    def run(self, passes=PASSES) -> dict:
        ops = self.ir["ops"]
        out = []
        drop = set()
        outputs = set(self.ir["output"].get("tensors") or [])
        for op in ops:
            if op["name"] in drop:
                continue
            k = op["op"]
            if k == "sigmoid_s8" and "swish" in passes:
                sig = op["outputs"][0]
                cons = self.consumers(sig)
                x = op["inputs"][0]
                if len(cons) == 1 and cons[0]["op"] == "mul_s8" and sorted(cons[0]["inputs"]) == sorted([sig, x]) \
                        and sig not in outputs:
                    mul = cons[0]
                    out.append(self._lut(mul, mul["name"], x, mul["outputs"][0], _silu))
                    drop.add(mul["name"])
                    self.ir["tensors"].pop(sig, None)
                    self._count("swish")
                    continue
            if k in ("sigmoid_s8", "gelu_s8") and "lut" in passes:
                f = _sigmoid if k == "sigmoid_s8" else _gelu
                out.append(self._lut(op, op["name"], op["inputs"][0], op["outputs"][0], f))
                self._count("lut")
                continue
            if k == "mul_c1_s8" and "mul_c1" in passes:
                g, x = op["inputs"]
                y = op["outputs"][0]
                M, s = quant_mult(self.scale(x) * self.scale(g) / self.scale(y))
                out.append({"name": op["name"], "op": "mul_c1_q", "precision": "int8", "inputs": [g, x],
                            "outputs": [y], "shape": dict(op["shape"]), "quant": {"M": M, "s": s}})
                self._count("mul_c1")
                continue
            if k == "adaptive_avg_pool2d_s8" and "gap" in passes and op["shape"].get("OH") == 1 \
                    and op["shape"].get("OW") == 1:
                sh = op["shape"]
                hw = int(sh["IH"]) * int(sh["IW"])
                x, y = op["inputs"][0], op["outputs"][0]
                M, s = quant_mult(self.scale(x) / (hw * self.scale(y)))
                out.append({"name": op["name"], "op": "gap_q", "precision": "int8", "inputs": [x], "outputs": [y],
                            "shape": {"N": int(sh["N"]), "C": int(sh["C"]), "HW": hw}, "quant": {"M": M, "s": s}})
                self._count("gap")
                continue
            if k == "add_s8" and "add" in passes:
                a, b = op["inputs"]
                y = op["outputs"][0]
                (ma, mb), S = add_mults([self.scale(a) / self.scale(y), self.scale(b) / self.scale(y)])
                key = f"{op['name']}.ma"
                self.w[key] = np.array([ma], np.int32)
                n = int(op["shape"]["n"])
                out.append({"name": op["name"], "op": "add_q", "precision": "int8", "inputs": [a, b],
                            "outputs": [y], "shape": {"rows": 1, "cols": n, "n_ma": 1, "b_rows": 1},
                            "quant": {"ma_key": key, "mb": mb, "S": S}})
                self._count("add")
                continue
            out.append(op)
        self.ir["ops"] = out
        from modelblaster.pipeline.extract_graph import _annotate_dispatches
        self.ir["dispatches"] = _annotate_dispatches(out)
        return self.counts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("graph")
    ap.add_argument("weights")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--passes", default=",".join(PASSES))
    a = ap.parse_args()
    ir = json.load(open(a.graph))
    w = dict(np.load(a.weights))
    c = Closer(ir, w)
    counts = c.run(tuple(p for p in a.passes.split(",") if p))
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    json.dump(c.ir, open(od / "graph.json", "w"), indent=2)
    np.savez(od / "weights.npz", **c.w)
    print(f"[integer_close] {counts} -> {od}")


if __name__ == "__main__":
    main()
