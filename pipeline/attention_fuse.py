"""Attention fusion: bmm -> (scale) -> (causal masked_fill) -> softmax|relu -> bmm  ==>  one `attention_f16` op.

The flow runs attention as separate slots and materialises the [B*nh, T, T] score tensor in HBM three or four
times (bmm write, scale, mask, softmax, bmm read). l3_farm/analysis/attention_fusion.md measures what that costs
(31_VisionAttention: 37.9 ms of 38.0 ms; a flash-style kernel does the same in 1.4-1.7 ms). This pass recognises
the chain and replaces it with one op of the fixed-signature kind ``attention_f16``
(reference_kernels_ext.ATTENTION):

    O[b0,b1,m,:] = act(scale * q[b0,b1,m,:] . k[b0,b1,n,:]^T  (n > m masked when causal))  @ v[b0,b1,n,:]
    act = softmax over n (act=0) or relu with no normalisation (act=1, 50_ReLUSelfAttention)

q/k/v and O are addressed through (off, s0, s1, row-stride) with unit inner stride, so the split / head
transpose / k^T copies in front of the bmms and the output transpose behind them are folded in with
layout_fold.source_map / sink_map (the same algebra and point-wise safety net as bmm_strided).

Matched spellings (each optional stage may be absent; ``view`` ops - reshape, dropout in eval - are skipped):
    bmm_f16 | fused__bmm_f16__<stages...>       (stages of a fused op are matched like separate ops)
    mul_scalar_f16 | ewise_scalar_f16 mul/div (positive scale)
    ewise2_f16 masked_fill(alpha=-inf) with a CONSTANT mask equal to triu(1) (checked on the values) -> causal=1
    softmax_f16 (K == N) | softmax_dim_f16 (inner == 1, R == N, log_out == 0)   -> act=0
    relu_f16 | ewise1_f16 relu                                                   -> act=1
    bmm_f16 reading the probabilities as its A operand (plain op: a fused PV bmm is left alone)
Scale and mask may come in either order (scale > 0 keeps -inf). Anything else breaks the match and the graph is
left as it was. The rewrite is NOT bit-identical (the reference is exact fp32 math rounded once, the original
chain rounds every intermediate to fp16), so verification is a tolerance check (``approx`` on the op).

    python -m modelblaster.pipeline.attention_fuse IR_DIR [--profile profile.json] [--out graph.json]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from . import layout_fold as lf

KIND = "attention_f16"
HINT_ID = "attention:fuse_sdpa"
BMM = {"bmm_f16"}
SOFTMAX = {"softmax_f16", "softmax_dim_f16"}
MAXD = 256
VERIFIED_TRIU = "verified:triu1"     # const_lookup answer for a mask the proposer already checked on its values


def _stages(op):
    return op.get("sub_ops") or [op]


def _scale_of(st):
    if st["op"] == "mul_scalar_f16":
        return float(st["scalar"])
    if st["op"] == "ewise_scalar_f16" and st.get("opc") in (3, 4):      # EWISE_SCALAR_OPS mul / div
        c = float(st["c"])
        return c if st["opc"] == 3 else (1.0 / c if c else None)
    return None


def _is_causal_mask(graph, st, batch, M, N, const_lookup) -> bool:
    """masked_fill(-inf) whose mask operand is a constant equal to triu(ones(M, N), 1) for every batch."""
    if st["op"] != "ewise2_f16" or not (st.get("fn") == "masked_fill" or st.get("opc") == 7):
        return False
    if not (isinstance(st.get("alpha"), float) and math.isinf(st["alpha"]) and st["alpha"] < 0):
        return False
    if M != N or const_lookup is None:
        return False
    mt = st["inputs"][1]
    key = (graph["tensors"].get(mt) or {}).get("const")
    arr = const_lookup(key) if key else None
    if arr is None:
        return False
    if isinstance(arr, str):
        return arr == VERIFIED_TRIU
    import numpy as np
    sh = st["shape"]
    flat = np.ascontiguousarray(arr).reshape(-1)
    dims = [int(sh[f"d{i}"]) for i in range(6)]
    strides = [int(sh[f"sb{i}"]) for i in range(6)]
    if lf._prod(dims) != batch * M * N:
        return False
    view = np.lib.stride_tricks.as_strided(flat, dims, [s * flat.itemsize for s in strides]).reshape(batch, M, N)
    want = np.triu(np.ones((M, N), dtype=bool), 1)
    return bool(((view != 0) == want[None]).all())


def _follow(readers, outs, t):
    """The single reader of tensor t (skipping view ops) -> (op, tensor it reads, [views]) or None."""
    views = []
    while True:
        r = readers.get(t, [])
        if len(r) != 1 or t in outs:
            return None
        o = r[0]
        if o["op"] == "view" and not o.get("offset"):
            views.append(o)
            t = o["outputs"][0]
            continue
        return o, t, views


def match(graph, op, prod, readers, outs, const_lookup=None):
    """op: a bmm (plain or head of a fused group) computing scores. -> match dict or None."""
    st0 = _stages(op)[0]
    if st0["op"] not in BMM:
        return None
    sh = st0["shape"]
    batch, M, D, N = int(sh["batch"]), int(sh["M"]), int(sh["K"]), int(sh["N"])
    scale, causal, act = 1.0, 0, None
    masks = []
    absorbed = [op]
    # stages fused into the score bmm's own op, then following ops
    pending = list(_stages(op)[1:])
    t = op["outputs"][0]
    pv = None
    while True:
        if not pending:
            nxt = _follow(readers, outs, t)
            if nxt is None:
                return None
            o, _, views = nxt
            absorbed += views
            if o["op"] in BMM and "sub_ops" not in o:
                pv = o
                break
            if act is not None:
                return None
            absorbed.append(o)
            pending = list(_stages(o))
            t = o["outputs"][0]
            if any(s.get("sub_ops") for s in pending):
                return None
        st = pending.pop(0)
        if act is not None:
            return None                       # something between softmax/relu and the PV bmm
        sc = _scale_of(st)
        if sc is not None:
            if sc <= 0:
                return None
            scale *= sc
        elif st["op"] == "ewise2_f16":
            if causal or not _is_causal_mask(graph, st, batch, M, N, const_lookup):
                return None
            causal = 1
            masks.append((graph["tensors"].get(st["inputs"][1]) or {}).get("const"))
        elif st["op"] in SOFTMAX:
            s2 = st["shape"]
            if st["op"] == "softmax_f16" and int(s2["K"]) != N:
                return None
            if st["op"] == "softmax_dim_f16" and (int(s2.get("inner", 1)) != 1 or int(s2["R"]) != N
                                                  or st.get("log_out")):
                return None
            act = 0
        elif st["op"] == "relu_f16" or (st["op"] == "ewise1_f16" and (st.get("fn") == "relu" or st.get("opc") == 8)):
            act = 1
        else:
            return None
    if act is None:
        return None
    ps = pv["shape"]
    if int(ps["batch"]) != batch or int(ps["M"]) != M or int(ps["K"]) != N:
        return None
    DV = int(ps["N"])
    if D > MAXD or DV > MAXD:
        return None
    return {"bmm1": op, "bmm2": pv, "absorbed": absorbed, "batch": batch, "M": M, "N": N, "D": D, "DV": DV,
            "scale": scale, "causal": causal, "act": act, "masks": masks}


def _fit4(m, sizes):
    f = lf.fit(m[1], sizes)
    return f


def _build(graph, mt, prod, readers, outs):
    """Fold the q/k/v/o copies and pick (B0, B1). -> (new op, dead ops) or None."""
    b1op, pv = mt["bmm1"], mt["bmm2"]
    st0 = _stages(b1op)[0]
    bt, M, N, D, DV = mt["batch"], mt["M"], mt["N"], mt["D"], mt["DV"]
    q_t, kT_t = st0["inputs"][:2]
    v_t = pv["inputs"][1]
    o_t = pv["outputs"][0]
    maps = {}
    for nm, t, natural in (("q", q_t, lf._natural(bt, M, D)), ("k", kT_t, lf._natural(bt, D, N)),
                           ("v", v_t, lf._natural(bt, N, DV))):
        root, m, ops = lf.source_map(graph, t, prod, readers, outs)
        if not any(o["op"] in lf.COPY_OPS for o in ops):
            root, m, ops = t, natural, []
        maps[nm] = (root, m, ops)
    sk = lf.sink_map(graph, o_t, prod, readers, outs)
    maps["o"] = sk if sk else (o_t, lf._natural(bt, M, DV), [])
    cands = {1}
    for nm, (r, c) in (("q", (M, D)), ("k", (D, N)), ("v", (N, DV)), ("o", (M, DV))):
        dims = [x for x in maps[nm][1][1] if x[0] != 1]
        acc = 1
        for d, _ in reversed(dims):
            acc *= d
            if acc % (r * c) == 0 and bt % (acc // (r * c)) == 0:
                cands.add(acc // (r * c))
        for d, _ in dims:
            if bt % d == 0:
                cands.add(d)
    for B1 in sorted(cands, reverse=True):
        B0 = bt // B1
        fq = lf.fit(maps["q"][1][1], [B0, B1, M, D])
        fk = lf.fit(maps["k"][1][1], [B0, B1, D, N])     # k^T: (d, n) -> k_sd must be 1
        fv = lf.fit(maps["v"][1][1], [B0, B1, N, DV])
        fo = lf.fit(maps["o"][1][1], [B0, B1, M, DV])
        if None in (fq, fk, fv, fo):
            continue
        if (fq[3], fk[2], fv[3], fo[3]) != (1, 1, 1, 1) and not (D == 1 or DV == 1):
            continue
        shape = {"B0": B0, "B1": B1, "M": M, "N": N, "D": D, "DV": DV,
                 "q_off": maps["q"][1][0], "q_s0": fq[0], "q_s1": fq[1], "q_sm": fq[2],
                 "k_off": maps["k"][1][0], "k_s0": fk[0], "k_s1": fk[1], "k_sn": fk[3],
                 "v_off": maps["v"][1][0], "v_s0": fv[0], "v_s1": fv[1], "v_sn": fv[2],
                 "o_off": maps["o"][1][0], "o_s0": fo[0], "o_s1": fo[1], "o_sm": fo[2],
                 "causal": mt["causal"], "act": mt["act"]}
        if max(abs(int(v)) for v in shape.values()) >= 2 ** 31:
            return None                                # int ABI
        copies = [o for nm in "qkvo" for o in maps[nm][2]]
        name = "+".join([b1op["name"], *[o["name"] for o in mt["absorbed"][1:] if o["op"] != "view"], pv["name"]])
        new = {"name": name, "op": KIND,
               "inputs": [maps["q"][0], maps["k"][0], maps["v"][0]], "outputs": [maps["o"][0]],
               "shape": shape, "scale": mt["scale"], "approx": True,
               "absorbed": [o["name"] for o in mt["absorbed"] + [pv]],
               "folded": [o["name"] for o in copies if o["op"] in lf.COPY_OPS],
               "dispatch_id": b1op.get("dispatch_id"), "hardware_target": b1op.get("hardware_target", "any"),
               "depends_on": []}
        return new, mt["absorbed"] + [pv] + copies
    return None


def fuse(graph: dict, *, op_us: dict | None = None, const_lookup=None, only: set | None = None):
    """Rewrite every matching attention chain. -> (new graph, report)."""
    g = copy.deepcopy(graph)
    rep = {"fused": [], "removed_us": 0.0}
    changed = True
    while changed:
        changed = False
        prod, readers, outs = lf._index(g)
        for op in list(g["ops"]):
            if only and op["name"] not in only:
                continue
            mt = match(g, op, prod, readers, outs, const_lookup)
            if mt is None:
                continue
            r = _build(g, mt, prod, readers, outs)
            if r is None:
                continue
            new, dead = r
            dead_names = {o["name"] for o in dead}
            keep_t = set(new["inputs"]) | set(new["outputs"])
            pos = min(i for i, o in enumerate(g["ops"]) if o["name"] in dead_names)
            ops = [o for o in g["ops"] if o["name"] not in dead_names]
            # insert where the first absorbed op was, but after the producers of its inputs
            prods = [i for i, o in enumerate(ops) if set(o["outputs"]) & set(new["inputs"])]
            ins_at = max([pos] + [i + 1 for i in prods]) if prods else pos
            ins_at = min(ins_at, len(ops))
            ops.insert(ins_at, new)
            g["ops"] = ops
            for o in dead:
                for s in _stages(o) + [o]:
                    for t in s["outputs"]:
                        if t not in keep_t:
                            g["tensors"].pop(t, None)
            us = sum((op_us or {}).get(n, 0.0) for n in dead_names)
            rep["fused"].append({"op": new["name"], "head": mt["bmm1"]["name"],
                                 "removed": sorted(dead_names), "removed_us": round(us, 1),
                                 "causal": new["shape"]["causal"], "act": new["shape"]["act"],
                                 "scale": new["scale"], "shape": new["shape"], "masks": mt["masks"]})
            rep["removed_us"] += us
            changed = True
            break
    rep["removed_us"] = round(rep["removed_us"], 1)
    return g, rep


def attention_hint(source: dict, ops=None, status: str = "proposed", masks=None) -> dict:
    h = {"id": HINT_ID, "kind": "fuse_attention", "target": "cuda", "status": status,
         "source": [source], "evidence": []}
    if masks:
        h["verified_masks"] = masks
    if ops:
        h["ops"] = ops
    return h


def propose(graph: dict, op_us: dict, *, const_lookup=None, total_us: float | None = None,
            min_gain_pct: float = 2.0, ir: str | None = None, seed_frac: float = 0.25) -> list[dict]:
    """Worth trying when the removed ops are >= min_gain_pct of the model. The fused kernel's own time is
    unknown before KernelBlaster runs; ``seed_frac`` discounts the estimate (measured: a flash-class kernel takes
    4-12% of the removed time on 31/43/44/50)."""
    from .graph_hints import unique_names
    graph = unique_names(copy.deepcopy(graph))         # hint op names must be the names apply_hints sees
    _, rep = fuse(graph, op_us=op_us, const_lookup=const_lookup)
    masks = sorted({k for f in rep["fused"] for k in f.get("masks", [])})
    total = total_us or sum(op_us.values()) or 1.0
    gain = rep["removed_us"] * (1.0 - seed_frac)
    pct = 100.0 * gain / total
    if not rep["fused"] or pct < min_gain_pct:
        return []
    return [attention_hint({"by": "mb-cost", "ir": ir, "est_gain_us": round(gain, 1), "est_gain_pct": round(pct, 2),
                            "chains": [f["removed"] for f in rep["fused"]]},
                           [f["head"] for f in rep["fused"]], masks=masks)]


def const_lookup_for(ir_dir: Path):
    """weights.npz-backed constant lookup (mask values) for an IR dir."""
    import numpy as np
    p = Path(ir_dir) / "weights.npz"
    if not p.exists():
        return None
    w = np.load(p)
    return lambda key: w[key] if key in w.files else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ir")
    ap.add_argument("--profile")
    ap.add_argument("--out")
    a = ap.parse_args()
    p = Path(a.ir)
    g = json.loads((p / "graph.json" if p.is_dir() else p).read_text())
    op_us = {}
    if a.profile:
        op_us = {o["name"]: o["us"] or 0.0 for o in json.loads(Path(a.profile).read_text())["ops"]}
    g2, rep = fuse(g, op_us=op_us, const_lookup=const_lookup_for(p if p.is_dir() else p.parent))
    print(json.dumps(rep, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(g2, indent=1))


if __name__ == "__main__":
    main()
