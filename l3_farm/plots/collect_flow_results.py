#!/usr/bin/env python3
"""Collect the per-problem results of the flow runs into data/flow_results.csv (one row per problem x dataset), for plot_flow.py. Local files only.
columns: dataset, problem, status (ok | failed | unsupported), model_us (flow's final whole-model time, CUDA-graph replay basis), seed_us (the flow's own unoptimised seed),
         torch_eager_us, torch_graph_us (same-device torch measurement of that run; L40S L3 = E1 torch_bench), note.
datasets: sol_pilot, sol_a6_pilot36, sol_small_base, sol_small_a6, sol_large_base, sol_large_a6, sol_overcap_a6, sol_adapt (variant workloads), l3_l40s_sweep, l3_l40s_A0..A6 (ablation), qrb_l2 (OpenCL, vs seed only)."""
import csv, glob, json, os, re
from pathlib import Path
H = Path(__file__).resolve().parent; F = H.parent
rows = []
def add(ds, prob, status, model=None, seed=None, te=None, tg=None, note=""):
    rows.append(dict(dataset=ds, problem=prob, status=status, model_us=model, seed_us=seed, torch_eager_us=te, torch_graph_us=tg, note=note))
def ir(path):
    try: return list(json.load(open(path))["irs"].values())[0]
    except Exception: return None
def sol(ds, inst, results="results"):
    n = 0
    for p in sorted(glob.glob(f"{F}/{inst}/{results}/9*")):
        r = ir(p + "/work/summary.json"); name = os.path.basename(p)
        if r is None:
            st = (open(p + "/status").read().strip() if os.path.exists(p + "/status") else "")
            if st.startswith("failed:extract"): add(ds, name, "unsupported", note=st)
            continue
        ok = bool(r.get("model_passed") and r.get("model_e2e_us"))
        add(ds, name, "ok" if ok else "failed", r.get("model_e2e_us") if ok else None, r.get("seed_e2e_us"), r.get("torch_eager_us"), r.get("torch_graph_us"))
        n += 1
    print(ds, n)
sol("sol_pilot", "pilot_sol/instance"); sol("sol_a6_pilot36", "pilot_sol/a6sol3_instance")
sol("sol_small_base", "pilot_sol/basesmall_instance"); sol("sol_small_a6", "pilot_sol/a6small_instance")
sol("sol_large_base", "pilot_sol/baselarge_instance"); sol("sol_large_a6", "pilot_sol/a6large_instance")
sol("sol_overcap_a6", "pilot_sol/a6big_instance"); sol("sol_adapt", "sol_adapt/instance")
# Level 3 on the L40S: torch from E1's torch_bench (the flow's own torch step was skipped there: no torch python on those nodes)
e1 = {x["problem"]: x["torch"] for x in json.load(open(F / "analysis/e1_data/e1_results.json")) if x.get("torch")}
def l3(ds, p, torch_name):
    r = ir(p + "/work/summary.json")
    if r is None: return False
    t = e1.get(torch_name, {}); ok = bool(r.get("model_passed") and r.get("model_e2e_us"))
    add(ds, torch_name, "ok" if ok else "failed", r.get("model_e2e_us") if ok else None, r.get("seed_e2e_us"), t.get("eager"), t.get("graph")); return True
n = sum(l3("l3_l40s_sweep", p, os.path.basename(p)) for p in sorted(glob.glob(f"{F}/results_headline/*")))
print("l3_l40s_sweep", n)
for p in sorted(glob.glob(f"{F}/results_abl3/*")):
    m = re.match(r"(\d+)_(.*)", os.path.basename(p))
    if not m or not os.path.isdir(p): continue
    a = int(m.group(1)); arm = a // 1000 - 11; num = a % 1000
    l3(f"l3_l40s_A{arm}", p, f"{num}_{m.group(2)}")
# Qualcomm OpenCL L2 (speedup over the flow's own seed; no torch on the board)
q = json.load(open(F / "qrb_a6/RESULTS_L2_all.json"))
for k, v in q.items(): add("qrb_l2", f"{k}_{v['name']}", "ok", None, None, None, None, note=f"speedup_vs_seed={v['speedup']:.6f}")
with open(H / "data/flow_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(len(rows), "rows ->", H / "data/flow_results.csv")
