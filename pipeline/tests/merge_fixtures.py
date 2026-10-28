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


def graph_and_weights(branches, quant="fp32", hw=(6, 6), seed=0, cin=4):
    H, W = hw
    t = F._tdt(quant)
    suf = SUF(quant)
    r = np.random.default_rng(seed)
    wt = lambda *s: (r.standard_normal(s) * 0.4).astype(np.float32)  # noqa: E731
    tens = {"x": [1, cin, H, W], "sq_o": [1, S, H, W], "s": [1, S, H, W]}
    ops = [F.conv_op("sq", "x", "sq_o", "sq.weight", "sq.bias", cin, S, 1, quant, hw), F.relu_op("sq_relu", "sq_o", "s", S * H * W, quant)]
    w = {"sq.weight": wt(S, cin, 1, 1), "sq.bias": wt(S)}
    ins, ch = [], []
    OH = OW = None
    for i, b in enumerate(branches):
        oc = b["oc"]
        if b.get("pool"):
            ops.append({"name": f"b{i}", "op": "avgpool2d" + suf, "inputs": ["s"], "outputs": [f"b{i}"],
                        "shape": {"N": 1, "C": S, "IH": H, "IW": W, "OH": H, "OW": W, "KH": 3, "KW": 3, "SH": 1, "SW": 1, "PH": 1, "PW": 1}})
            tens[f"b{i}"] = [1, S, H, W]
            ins.append(f"b{i}")
            ch.append(S)
            continue
        kh, kw = _k(b)
        st, dl, g = b.get("stride", 1), b.get("dil", 1), b.get("groups", 1)
        ph = b["pad"] if b.get("pad") is not None else dl * (kh - 1) // 2
        pw = b["pad"] if b.get("pad") is not None else dl * (kw - 1) // 2
        oh = (H + 2 * ph - dl * (kh - 1) - 1) // st + 1
        ow = (W + 2 * pw - dl * (kw - 1) - 1) // st + 1
        OH, OW = oh, ow
        kind = "conv2d_grouped" if g != 1 else "conv2d"
        op = {"name": f"c{i}", "op": kind + suf, "inputs": ["s"], "outputs": [f"c{i}_o"], "weight": f"c{i}.weight",
              "shape": {"N": 1, "IC": S, "IH": H, "IW": W, "OC": oc, "OH": oh, "OW": ow, "KH": kh, "KW": kw, "SH": st, "SW": st,
                        "PH": ph, "PW": pw, "DH": dl, "DW": dl, **({"G": g} if g != 1 else {})}}
        w[f"c{i}.weight"] = wt(oc, S // g, kh, kw)
        if b.get("bias", True):
            op["bias"] = f"c{i}.bias"
            w[f"c{i}.bias"] = wt(oc)
        ops.append(op)
        tens[f"c{i}_o"] = [1, oc, oh, ow]
        last = f"c{i}_o"
        if b.get("act", "relu"):
            ops.append({"name": f"a{i}", "op": b.get("act", "relu") + suf, "inputs": [last], "outputs": [f"a{i}_o"], "shape": {"n": oc * oh * ow}})
            tens[f"a{i}_o"] = [1, oc, oh, ow]
            last = f"a{i}_o"
        ins.append(last)
        ch.append(oc)
    OH, OW = tens[ins[0]][2:]
    tot = sum(ch)
    ops.append({"name": "cat", "op": f"cat{len(ins)}_c1" + suf, "inputs": ins, "outputs": ["cat_out"],
                "shape": {"N": 1, "H": OH, "W": OW, "C_inputs": ch, "C_total": tot}})
    ops.append({"name": "pool", "op": "adaptive_avg_pool2d" + suf, "inputs": ["cat_out"], "outputs": ["pooled"],
                "shape": {"N": 1, "C": tot, "IH": OH, "IW": OW}})
    ops.append({"name": "fc", "op": "linear" + suf, "inputs": ["pooled"], "outputs": ["out"], "weight": "fc.weight", "bias": "fc.bias",
                "shape": {"M": 1, "K": tot, "N": 4}})
    tens.update(cat_out=[1, tot, OH, OW], pooled=[1, tot, 1, 1], out=[1, 4])
    w.update({"fc.weight": wt(4, tot), "fc.bias": wt(4)})
    for i, o in enumerate(ops):
        o["dispatch_id"], o["depends_on"] = i, [i - 1] if i else []
    g = {"name": "branchy", "version": 1, "quant": quant, "input": {"tensor": "x"}, "output": {"tensors": ["out"], "tensor": "out"},
         "tensors": {k: {"shape": v, "dtype": t, "quant": None} for k, v in tens.items()}, "ops": ops, "dispatches": list(range(len(ops)))}
    dt = F._wdt(quant)
    w = {k: v.astype(dt) for k, v in w.items()}
    if quant == "native":
        g["weight_dtypes"] = {k: "f16" for k in w}
    return g, w


def branchy_ir(d, branches, quant="fp32", hw=(6, 6), seed=0, cin=4):
    g, w = graph_and_weights(branches, quant, hw, seed, cin)
    return F.write_ir(Path(d), g, w)


FIRE = [{"k": 1, "oc": 2}, {"k": 3, "oc": 3}]
