#!/usr/bin/env python3
"""Histfree table (problem x arm N/A/B/C) from results_histfree/<alias>/work/model_runtime.json. Usage: python3 analysis/hist_free/ablation_table.py"""
import json, os, tarfile, math, glob
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
m = json.load(open(ROOT + "/analysis/headline/histfree_map.json"))["names"]
def final(a):
    d = f"{ROOT}/results_histfree/{a}/work/model_runtime.json"
    try:
        if os.path.exists(d): return list(json.load(open(d)).values())[0]["model_e2e_us"]
        t = tarfile.open(f"{ROOT}/results_histfree/{a}/final.tar.gz")
        for n in t.getnames():
            if n.endswith("model_runtime.json"): return list(json.load(t.extractfile(n)).values())[0]["model_e2e_us"]
    except Exception: return None
def wall(a):
    try:
        e = json.load(open(f"{ROOT}/results_histfree/{a}/run_meta.json"))["events"]; return (e[-1]["t"] - e[0]["t"]) / 60
    except Exception: return None
head = {}
for f in glob.glob(ROOT + "/results_headline*/*/work/model_runtime.json"):
    head[os.path.basename(os.path.dirname(os.path.dirname(f)))] = list(json.load(open(f)).values())[0]["model_e2e_us"]
R = {(o, a): (final(al), wall(al)) for al, (a, o) in m.items()}
probs = sorted({o for o, a in R}, key=lambda p: int(p.split("_")[0]))
arms = ["N", "A", "B", "C"]
print("| problem | headline | " + " | ".join(f"{a} us (wall min)" for a in arms) + " |"); print("|---|---|" + "---|" * 4)
for p in probs:
    c = []
    for a in arms:
        v = R.get((p, a)); c.append("-" if not v or not v[0] else f"{v[0]:,.0f} ({v[1]:.0f})" if v[1] else f"{v[0]:,.0f}")
    print(f"| {p} | {head.get(p,0):,.0f} | " + " | ".join(c) + " |")
def gm(x): return math.exp(sum(map(math.log, x)) / len(x)) if x else float("nan")
print()
for a in arms[1:]:
    r = [R[(p, a)][0] / R[(p, "N")][0] for p in probs if R.get((p, a), (None,))[0] and R.get((p, "N"), (None,))[0]]
    print(f"geomean {a}/N = {gm(r):.3f} (n={len(r)}; <1 = faster than N)")
