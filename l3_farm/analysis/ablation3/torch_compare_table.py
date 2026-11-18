#!/usr/bin/env python3
"""Build the data table of torch_compare.md from torch_compare_data/out + e1_data/e1_results.json + t2_all/REPORT.md. Prints markdown."""
import json, math, re, statistics as st
from pathlib import Path
H = Path(__file__).resolve().parent; A = H.parent
O = H / "torch_compare_data/out"
PROBS = ["34_VanillaRNNHidden", "18_SqueezeNet", "14_DenseNet121DenseBlock", "46_NetVladWithGhostClusters", "47_NetVladNoGhostClusters", "8_ResNetBasicBlock",
         "15_DenseNet121", "7_GoogleNetInceptionV1", "1_MLP", "4_LeNet5", "9_ResNet18", "20_MobileNetV2", "33_VanillaRNN", "39_GRU", "43_MinGPTCausalAttention", "45_UNetSoftmax"]
e1 = {x["problem"]: x for x in json.load(open(A / "e1_data/e1_results.json"))}
t2 = {}
for ln in (A / "t2_all/REPORT.md").read_text().splitlines():
    c = [x.strip().replace(",", "") for x in ln.strip("|").split("|")]
    if len(c) >= 8 and re.match(r"\d+_", c[0]):
        try: t2[c[0]] = float(c[2]) * float(c[4])         # CUDA-only us x "CUDA x torch" = the T2 harness's torch time
        except ValueError: pass
def rb(n, f="rb"):
    try:
        l = open(O / f"{n}.{f}.txt").read().splitlines()[-1].split(",")
        v = float(l[2]), float(l[3])                       # p50, p90
        return None if math.isnan(v[0]) else v
    except Exception: return None
def tb(n):
    ev, gr = [], []
    for i in (1, 2, 3):
        try:
            d = json.loads(open(O / f"{n}.tb{i}.txt").read().strip()[len("@@POSTHOC@@"):]); ev.append(d["eager_us"]); gr.append(d.get("graph_us"))
        except Exception: pass
    return ev, [g for g in gr if g]
def sp(v): return (max(v) - min(v)) / st.median(v) * 100 if len(v) > 1 else float("nan")
def f(x, nd=1): return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:,.{nd}f}"
def gm(xs): xs = [x for x in xs if x]; return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
rows, R = [], {k: [] for k in ("rb_tb", "tb_gr", "rb_e1e", "tb_e1e", "tbg_e1g", "rbc_tb")}
print("| problem | run_baselines p50 | p90 | torch_bench eager (med of 3) | spread % | torch_bench graph | spread % | E1 old eager | E1 old graph | T2-harness torch | rb/tb eager | tb eager/graph | rb/E1 eager | tb eager/E1 eager | tb graph/E1 graph | rb_compile p50 (falls back to eager) |")
print("|" + "---|" * 16)
for n in PROBS:
    r = rb(n); ev, gr = tb(n); c = rb(n, "rbc"); e = e1.get(n, {}).get("torch") or {}
    me, mg = (st.median(ev) if ev else None), (st.median(gr) if gr else None)
    ratio = lambda a, b: (a / b) if a and b else None
    x = dict(rb_tb=ratio(r and r[0], me), tb_gr=ratio(me, mg), rb_e1e=ratio(r and r[0], e.get("eager")), tb_e1e=ratio(me, e.get("eager")), tbg_e1g=ratio(mg, e.get("graph")),
             rbc_tb=ratio(c and c[0], me))
    for k, v in x.items():
        if v: R[k].append((n, v))
    print(f"| {n} | {f(r and r[0])} | {f(r and r[1])} | {f(me)} | {f(sp(ev), 2)} | {f(mg)} | {f(sp(gr), 2)} | {f(e.get('eager'))} | {f(e.get('graph'))} | {f(t2.get(n))} | "
          + " | ".join(f(x[k], 3) for k in ("rb_tb", "tb_gr", "rb_e1e", "tb_e1e", "tbg_e1g")) + f" | {f(c and c[0])} |")
print("\n| ratio | n | geomean | median | worst (max/min) |\n|---|---|---|---|---|")
for k, v in R.items():
    vs = [b for _, b in v]
    if vs: print(f"| {k} | {len(vs)} | {gm(vs):.3f} | {st.median(vs):.3f} | max {max(v, key=lambda t: t[1])[0]} {max(vs):.3f}; min {min(v, key=lambda t: t[1])[0]} {min(vs):.3f} |")
ab = PROBS[:8]
g8 = [dict(R["tb_gr"]).get(n) for n in ab]
print(f"\ntorch_bench eager/graph geomean: 8 ablation problems {gm(g8):.3f}; all 16 {gm([b for _, b in R['tb_gr']]):.3f}")
print("\nPairs differing by more than 5% (|ratio-1|>0.05):")
for k, v in R.items():
    print(f"- {k}: " + ", ".join(f"{n} {b:.3f}" for n, b in v if abs(b - 1) > 0.05))
