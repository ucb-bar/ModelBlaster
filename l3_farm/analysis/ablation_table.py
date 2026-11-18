#!/usr/bin/env python3
"""Build the ablation table (per problem x arm) from results_ablation/<alias>/final.tar.gz; prints markdown. Usage: python3 analysis/ablation_table.py [> out.md]"""
import json, os, tarfile, math, glob, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = json.load(open(ROOT + "/analysis/headline/ablation_map.json"))["names"]
def final(alias):
    f = f"{ROOT}/results_ablation/{alias}/final.tar.gz"
    d = f"{ROOT}/results_ablation/{alias}/work/model_runtime.json"
    try:
        if os.path.exists(d):
            return list(json.load(open(d)).values())[0]["model_e2e_us"]
        t = tarfile.open(f)
        for n in t.getnames():
            if n.endswith("model_runtime.json"):
                return list(json.load(t.extractfile(n)).values())[0]["model_e2e_us"]
    except Exception:
        return None
def wall(alias):
    try:
        e = json.load(open(f"{ROOT}/results_ablation/{alias}/run_meta.json"))["events"]
        return (e[-1]["t"] - e[0]["t"]) / 60
    except Exception:
        return None
head = {}
for f in glob.glob(ROOT + "/results_headline*/*/work/model_runtime.json"):
    head[os.path.basename(os.path.dirname(os.path.dirname(f)))] = list(json.load(open(f)).values())[0]["model_e2e_us"]
arms = ["A", "B", "C", "D", "E", "F", "G", "Z0", "Z1", "Z2"]
R = {}
for alias, (arm, orig) in m.items():
    R[(orig, arm)] = (final(alias), wall(alias))
probs = sorted({o for o, a in R}, key=lambda p: int(p.split("_")[0]))
print("| problem | headline | " + " | ".join(arms) + " |")
print("|---|---|" + "---|" * len(arms))
for p in probs:
    cells = []
    for a in arms:
        v = R.get((p, a))
        cells.append("-" if v is None else (f"{v[0]:,.0f}" if v[0] else "incomplete"))
    print(f"| {p} | {head.get(p, 0):,.0f} | " + " | ".join(cells) + " |")
def gm(xs): return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
print("\nGeomean of (Z0 / arm) over problems done in both (>1: arm faster than the same-session Z0 baseline), and of (headline / arm):")
for a in arms:
    z = [R[(p, "Z0")][0] / R[(p, a)][0] for p in probs if R.get((p, a), (None,))[0] and R.get((p, "Z0"), (None,))[0] and a != "Z0"]
    h = [head[p] / R[(p, a)][0] for p in probs if R.get((p, a), (None,))[0] and p in head]
    if h: print(f"- {a}: vs Z0 {gm(z):.3f} (n={len(z)}); vs headline {gm(h):.3f} (n={len(h)})")
