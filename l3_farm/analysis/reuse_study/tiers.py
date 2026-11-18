#!/usr/bin/env python3
"""Sensitivity of the coverage numbers to the donor pool: store (gk_impl v0258) / win (store + any problem's cache winners) / all verified records /
prev (first sweep's records, other problems). Mean over problems of the share of summed op time (all 42 finished; 33 non-region)."""
import json, collections
from pathlib import Path
D = Path(__file__).resolve().parent / "data"
C = json.load(open(D / "coverage.json")); PV = json.load(open(D / "prev_cov.json"))
def lvl(s): return "b" if s["n_exact"] else ("c" if s["dmin"] is not None and s["dmin"] <= 2.5 else "d")
res = {}
for tier in ("store", "win", "all"):
    a = collections.Counter(); nr = 0
    for p, d in C.items():
        for o in d["ops"]:
            if o["ctype"] == "region":
                a["region"] += o["share"]; continue
            S = "S" if (o["ctype"] == "seed" or o["weak"]) else "A"
            a[S + lvl(o[tier])] += o["share"]
    res[tier] = a
res["prev"] = collections.Counter()
for p, c in PV.items():
    for k, v in c.items(): res["prev"][k] += v
n = len(C); nreg = sum(1 for p, d in C.items() if all(o["ctype"] != "region" for o in d["ops"]))
print("n problems", n, "non-region", nreg)
print("tier | mean share exact (A,S) | near<=2.5 (A,S) | none (A,S) | S total | ops weighted any donor")
for t, a in res.items():
    f = lambda k: 100 * a[k] / n
    print(f"{t:6s} A:b {f('Ab'):5.1f} c {f('Ac'):5.1f} d {f('Ad'):5.1f} | S:b {f('Sb'):4.1f} c {f('Sc'):4.1f} d {f('Sd'):4.1f} | region {f('region'):4.1f} | any donor (non-region basis) {100*(a['Ab']+a['Ac']+a['Sb']+a['Sc'])/(n-a['region']):5.1f}%")
json.dump({t: dict(a) for t, a in res.items()}, open(D / "tiers.json", "w"))
