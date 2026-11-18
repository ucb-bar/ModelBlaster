#!/usr/bin/env python3
"""Zero-GPU predicted upper bound from EXACT-shape donors (record speedups vs seed on the same shape): per problem the share of the
summed op time that exact foreign kernels could remove: seed op: share*(1-1/sp_donor); weak op: share*(1-cur/sp_donor);
winner op: share*(1-sp_own/sp_donor) where the donor is >3% better by record speedup. Also with the 'all' pool (any verified record)."""
import json
from pathlib import Path
D = Path(__file__).resolve().parent / "data"
C = json.load(open(D / "coverage.json"))
res = {}
for p, d in C.items():
    tot = {t: 0.0 for t in ("store", "win", "all")}
    det = []
    for o in d["ops"]:
        if o["ctype"] == "region":
            continue
        for t in tot:
            s = o[t]
            sp = s["exact_best_speedup"]
            if not s["n_exact"] or not sp:
                continue
            if o["ctype"] == "seed":
                g = o["share"] * (1 - 1 / sp) if sp > 1.03 else 0
            else:
                own = o.get("own_best_sp_exact") or o.get("speedup_vs_seed") or 1.0
                g = o["share"] * (1 - own / sp) if sp > own * 1.03 else 0
            tot[t] += g
            if t == "win" and g > 0.003:
                det.append((o["name"], o["kind"], round(o["share"], 3), o["ctype"], round(sp, 2), o.get("own_best_sp_exact"), round(g, 3)))
    res[p] = {"pred": {t: round(v, 4) for t, v in tot.items()}, "det": det[:6]}
json.dump(res, open(D / "potential.json", "w"), indent=1)
for p, r in res.items():
    if r["pred"]["all"] > 0.005:
        print(p.ljust(34), r["pred"], r["det"][:2])
