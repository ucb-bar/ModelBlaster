"""Synthetic branchy IRs for the merge_sibling_convs tests (real IR dirs: graph.json / weights.npz / io.npz, golden from the
torch walker).

    x[1,4,H,W] -> stem 1x1 conv(4->S) -> relu -> s
    s -> branch_i: conv(k_i, oc_i, stride, dilation, groups, padding) [-> act] -> b_i        (i = 0..n-1, or a pool branch)
    cat_c1(b_0, ..., b_n-1) -> cat_out -> adaptive_avg_pool2d -> pooled -> linear -> out

A branch spec is a dict: k (odd kernel or (kh, kw)), oc, stride=1, dil=1, groups=1, act="relu"|"silu"|None, bias=True,
pad=None (None = centred "same": dil*(k-1)//2), or {"pool": True, "oc": S} for an average-pool branch (not a conv).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline.tests import rewrite_fixtures as F  # noqa: E402

S = 4
SUF = F._suffix


def _k(b):
    k = b.get("k", 1)
    return (k, k) if isinstance(k, int) else tuple(k)


def graph_and_weights(branches, quant="fp32", hw=(6, 6), seed=0, cin=4, modules=1):
    """``modules`` > 1 chains identical modules: module m > 0 reads the previous module's concat (IC = its channel total) and its ops /
    tensors / weights are prefixed ``m{m}_``."""
    H, W = hw
    t = F._tdt(quant)
    suf = SUF(quant)
    r = np.random.default_rng(seed)
    wt = lambda *s: (r.standard_normal(s) * 0.4).astype(np.float32)  # noqa: E731
    tens = {"x": [1, cin, H, W], "sq_o": [1, S, H, W], "s": [1, S, H, W]}
    ops = [F.conv_op("sq", "x", "sq_o", "sq.weight", "sq.bias", cin, S, 1, quant, hw), F.relu_op("sq_relu", "sq_o", "s", S * H * W, quant)]
    w = {"sq.weight": wt(S, cin, 1, 1), "sq.bias": wt(S)}
    src, ic = "s", S
    for m in range(modules):
        p = "" if m == 0 else f"m{m}_"
        ins, ch = [], []
        for i, b in enumerate(branches):
            oc = b["oc"]
            if b.get("pool"):
                ops.append({"name": f"{p}b{i}", "op": "avgpool2d" + suf, "inputs": [src], "outputs": [f"{p}b{i}"],
                            "shape": {"N": 1, "C": ic, "IH": H, "IW": W, "OH": H, "OW": W, "KH": 3, "KW": 3, "SH": 1, "SW": 1, "PH": 1, "PW": 1}})
                tens[f"{p}b{i}"] = [1, ic, H, W]
                ins.append(f"{p}b{i}")
                ch.append(ic)
                continue
            kh, kw = _k(b)
            st, dl, g = b.get("stride", 1), b.get("dil", 1), b.get("groups", 1)
            ph = b["pad"] if b.get("pad") is not None else dl * (kh - 1) // 2
            pw = b["pad"] if b.get("pad") is not None else dl * (kw - 1) // 2
            oh = (H + 2 * ph - dl * (kh - 1) - 1) // st + 1
            ow = (W + 2 * pw - dl * (kw - 1) - 1) // st + 1
            kind = "conv2d_grouped" if g != 1 else "conv2d"
            op = {"name": f"{p}c{i}", "op": kind + suf, "inputs": [src], "outputs": [f"{p}c{i}_o"], "weight": f"{p}c{i}.weight",
                  "shape": {"N": 1, "IC": ic, "IH": H, "IW": W, "OC": oc, "OH": oh, "OW": ow, "KH": kh, "KW": kw, "SH": st, "SW": st,
                            "PH": ph, "PW": pw, "DH": dl, "DW": dl, **({"G": g} if g != 1 else {})}}
            w[f"{p}c{i}.weight"] = wt(oc, ic // g, kh, kw)
            if b.get("bias", True):
                op["bias"] = f"{p}c{i}.bias"
                w[f"{p}c{i}.bias"] = wt(oc)
            ops.append(op)
            tens[f"{p}c{i}_o"] = [1, oc, oh, ow]
            last = f"{p}c{i}_o"
            if b.get("act", "relu"):
                ops.append({"name": f"{p}a{i}", "op": b.get("act", "relu") + suf, "inputs": [last], "outputs": [f"{p}a{i}_o"], "shape": {"n": oc * oh * ow}})
                tens[f"{p}a{i}_o"] = [1, oc, oh, ow]
                last = f"{p}a{i}_o"
            ins.append(last)
            ch.append(oc)
        OH, OW = tens[ins[0]][2:]
        tot = sum(ch)
        out = "cat_out" if m == modules - 1 else f"{p}cat_out"
        if m == 0 and modules > 1:
            out = "cat_out_0"
        ops.append({"name": f"{p}cat", "op": f"cat{len(ins)}_c1" + suf, "inputs": ins, "outputs": [out],
                    "shape": {"N": 1, "H": OH, "W": OW, "C_inputs": ch, "C_total": tot}})
        tens[out] = [1, tot, OH, OW]
        src, ic, H, W = out, tot, OH, OW
    ops.append({"name": "pool", "op": "adaptive_avg_pool2d" + suf, "inputs": [src], "outputs": ["pooled"],
                "shape": {"N": 1, "C": ic, "IH": H, "IW": W}})
    ops.append({"name": "fc", "op": "linear" + suf, "inputs": ["pooled"], "outputs": ["out"], "weight": "fc.weight", "bias": "fc.bias",
                "shape": {"M": 1, "K": ic, "N": 4}})
    tens.update(pooled=[1, ic, 1, 1], out=[1, 4])
    w.update({"fc.weight": wt(4, ic), "fc.bias": wt(4)})
    for i, o in enumerate(ops):
        o["dispatch_id"], o["depends_on"] = i, [i - 1] if i else []
    g = {"name": "branchy", "version": 1, "quant": quant, "input": {"tensor": "x"}, "output": {"tensors": ["out"], "tensor": "out"},
         "tensors": {k: {"shape": v, "dtype": t, "quant": None} for k, v in tens.items()}, "ops": ops, "dispatches": list(range(len(ops)))}
    dt = F._wdt(quant)
    w = {k: v.astype(dt) for k, v in w.items()}
    if quant == "native":
        g["weight_dtypes"] = {k: "f16" for k in w}
    return g, w


def branchy_ir(d, branches, quant="fp32", hw=(6, 6), seed=0, cin=4, modules=1):
    g, w = graph_and_weights(branches, quant, hw, seed, cin, modules)
    return F.write_ir(Path(d), g, w)


FIRE = [{"k": 1, "oc": 2}, {"k": 3, "oc": 3}]
