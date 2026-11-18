#!/usr/bin/env python3
"""A3 item 3, zero LLM: how good a start is the NEAREST other shape's winner for a conv shape that has no winner?
Uses the shape-family pilot's cross-evaluation (family_data/pilot/matrix.json: each of the sweep's 61 DenseNet121 conv
winners run on all 61 conv instances in one problem; a kernel that fails on some shape has no times) and the sweep's
round-0 profile (hand-seed time per op). For every shape: candidates = the other shapes' winners ordered by
search_fixes.shape_distance (nearest first); seed = the first of the nearest k that runs (KernelBlaster keeps a
candidate only if it passes and is faster). -> nearest_eval.json + summary (vs the hand seed, vs the shape's own
LLM winner, vs the best other kernel)."""
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
L3 = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(L3, "..", "ModelBlaster", ".."))
from ModelBlaster.pipeline.search_fixes import shape_distance  # noqa: E402

m = json.load(open(os.path.join(L3, "analysis", "family_data", "pilot", "matrix.json")))["runs"]
prof = json.load(open(os.path.join(L3, "results", "15_DenseNet121", "work", "r00", "profile", "15_DenseNet121",
                                   "profile.json")))
shape = {o["name"]: o["shape"] for o in prof["ops"] if o["op"] == "conv2d_f16"}
seed = {o["name"]: o["us"] for o in prof["ops"] if o["op"] == "conv2d_f16"}
own = {}
for kid, v in m.items():
    for ow in v["owner"]:
        own.setdefault(ow["op"], kid)
rows = []
for K in (1, 2, 3):
    for op, sh in shape.items():
        mine = own.get(op)
        cands = sorted(((shape_distance(sh, shape[ow["op"]]), kid) for kid, v in m.items() for ow in v["owner"]
                        if ow["op"] != op and ow["op"] in shape), key=lambda x: x[0])
        seen, order = set(), []
        for d, kid in cands:
            if kid not in seen and kid != mine:
                seen.add(kid)
                order.append((d, kid))
        t = {kid: (m[kid]["per_op_us"] or {}).get(op) for _, kid in order}
        valid = [(d, kid) for d, kid in order[:K] if t.get(kid)]
        best_k = min(valid, key=lambda x: t[x[1]]) if valid else None
        own_us = (m[mine]["per_op_us"] or {}).get(op) if mine else None
        best_other = min((x for x in t.values() if x), default=None)
        rows.append({"k": K, "op": op, "seed_us": seed[op], "own_us": own_us, "best_other_us": best_other,
                     "nearest_us": t[best_k[1]] if best_k else None, "nearest_dist": best_k[0] if best_k else None,
                     "n_tried": min(K, len(order)), "taken": bool(best_k and t[best_k[1]] < 0.98 * seed[op])})
out = {}
for K in (1, 2, 3):
    rs = [r for r in rows if r["k"] == K]
    tk = [r for r in rs if r["taken"]]
    vs_own = [r["nearest_us"] / r["own_us"] for r in tk if r["own_us"]]
    out[K] = {"shapes": len(rs), "taken": len(tk), "speedup_vs_seed_median": round(st.median(r["seed_us"] / r["nearest_us"] for r in tk), 1) if tk else None,
              "vs_own_winner_median": round(st.median(vs_own), 3) if vs_own else None,
              "vs_own_winner_p25_p75": [round(sorted(vs_own)[len(vs_own) // 4], 3), round(sorted(vs_own)[3 * len(vs_own) // 4], 3)] if vs_own else None,
              "within_10pct_of_own": sum(1 for x in vs_own if x <= 1.1), "faster_than_own": sum(1 for x in vs_own if x < 1.0)}
json.dump({"summary": out, "rows": rows}, open(os.path.join(HERE, "nearest_eval.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
