#!/usr/bin/env python3
"""Beam test table (problem x arm I4r/I2/B4/B2 + controls) from results_beam/<alias>/work. Usage: python3 analysis/beam_v2/beam_table.py"""
import json, os, glob, math, re
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
m = json.load(open(ROOT + "/analysis/headline/beam_map.json"))["names"]
CTRL = {"1_MLP": 1893, "11_VGG16": 2523, "19_MobileNetV1": 343, "33_VanillaRNN": 2429, "41_GRUBidirectional": 3132,
        "7_GoogleNetInceptionV1": 611, "27_RegNet": 811, "43_MinGPTCausalAttention": 2571}
def final(a):
    try: return list(json.load(open(f"{ROOT}/results_beam/{a}/work/model_runtime.json")).values())[0]["model_e2e_us"]
    except Exception: return None
def stats(a):
    w = f"{ROOT}/results_beam/{a}/work"
    rounds = len(glob.glob(w + "/r0*"))
    calls = ok = err = 0
    for f in glob.glob(w + "/r0*/run/telemetry.jsonl"):
        for l in open(f, errors="replace"):
            if '"t":"llm"' in l and '"calls"' in l:
                calls += 1
                if '"error"' in l: err += 1
    return rounds, calls, err
head = {}
for f in glob.glob(ROOT + "/results_headline*/*/work/model_runtime.json"):
    head[os.path.basename(os.path.dirname(os.path.dirname(f)))] = list(json.load(open(f)).values())[0]["model_e2e_us"]
R = {}
for al, (arm, orig) in m.items():
    R[(orig, arm)] = (final(al),) + stats(al)
probs = sorted({o for o, a in R}, key=lambda p: int(p.split("_")[0]))
arms = ["I4r", "I2", "B4", "B2"]
print("| problem | control N | headline | " + " | ".join(f"{a} us" for a in arms) + " |"); print("|---|---|---|" + "---|" * 4)
for p in probs:
    c = []
    for a in arms:
        v = R.get((p, a)); c.append("-" if not v or not v[0] else f"{v[0]:,.0f}")
    print(f"| {p} | {CTRL.get(p,0):,} | {head.get(p,0):,.0f} | " + " | ".join(c) + " |")
def gm(x): return math.exp(sum(map(math.log, x)) / len(x)) if x else float("nan")
print()
for a in arms:
    r = [R[(p, a)][0] / CTRL[p] for p in probs if R.get((p, a), (None,))[0] and p in CTRL]
    q = [R[(p, a)][0] / R[(p, "I4r")][0] for p in probs if R.get((p, a), (None,))[0] and R.get((p, "I4r"), (None,))[0]]
    cc = [R[(p, a)][2] for p in probs if R.get((p, a), (None,) * 3)[0]]
    print(f"- {a}: geomean us/control N = {gm(r):.3f} (n={len(r)}); geomean us/I4r = {gm(q):.3f}")
