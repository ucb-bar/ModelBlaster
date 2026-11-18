#!/usr/bin/env python3
"""Build analysis/t2_all/REPORT.md tables from `l3_all_sched.py status` + the flow headline finals. Run from l3_farm."""
import re, json, glob, os, math, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out = subprocess.run(["python3", "l3_all_sched.py", "status"], cwd=ROOT + "/baseline_cc", capture_output=True, text=True).stdout
R = {}
pat = re.compile(r"^(\S+)\|(cuda|e13)\s+(done|failed)\s+node \d+ att=(\d) t=(\S+?)m turns=(\S+) ~\$(\S+)(?: correct=(\w+) (\S+?)us x(\S+))?")
for line in out.splitlines():
    m = pat.match(line)
    if not m: continue
    p, mode, st, att, t, turns, cost, ok, us, x = m.groups()
    f = lambda v: None if v in (None, "None") else float(v)
    R[(p, mode)] = dict(st=st, t=f(t), turns=f(turns), cost=f(cost) or 0.0, ok=ok == "True", us=f(us), x=f(x))
head = {}
for f in glob.glob(ROOT + "/results_headline*/*/work/model_runtime.json"):
    head[os.path.basename(os.path.dirname(os.path.dirname(f)))] = list(json.load(open(f)).values())[0]["model_e2e_us"]
probs = sorted({p for p, _ in R}, key=lambda p: int(p.split("_")[0]))
def cell(r):
    if not r: return "-"
    if r["st"] == "failed": return "FAILED (interface)"
    if not r["ok"] or not r["us"]: return "incorrect"
    return f"{r['us']:,.0f}"
print("| problem | flow headline us | CUDA-only us | E13 us | CUDA x torch | E13 x torch | CUDA min/$ | E13 min/$ |")
print("|---|---|---|---|---|---|---|---|")
for p in probs:
    c, e = R.get((p, "cuda")), R.get((p, "e13"))
    def mc(r): return "-" if not r or r["t"] is None else f"{r['t']:.0f}/${r['cost']:.1f}"
    def xx(r): return "-" if not r or not r["x"] or not r["ok"] else f"{r['x']:.2f}"
    print(f"| {p} | {head.get(p,0):,.0f} | {cell(c)} | {cell(e)} | {xx(c)} | {xx(e)} | {mc(c)} | {mc(e)} |")
def gm(v): return math.exp(sum(map(math.log, v)) / len(v)) if v else float("nan")
print()
for mode in ("cuda", "e13"):
    ok = [p for p in probs if R.get((p, mode), {}).get("ok") and R[(p, mode)]["us"]]
    both = [p for p in ok if head.get(p)]
    print(f"- {mode}: correct+timed {len(ok)}/{len(probs)}; failed {sum(1 for p in probs if R.get((p,mode),{}).get('st')=='failed')}; incorrect {sum(1 for p in probs if R.get((p,mode)) and R[(p,mode)]['st']=='done' and not R[(p,mode)]['ok'])}; geomean speedup vs torch {gm([R[(p,mode)]['x'] for p in ok if R[(p,mode)]['x']]):.2f}; geomean (this mode us / flow headline us) = {gm([R[(p,mode)]['us']/head[p] for p in both]):.2f} over {len(both)} problems (>1 = slower than the flow); total cost ${sum(r['cost'] for (p,m),r in R.items() if m==mode):.0f}")
both = [p for p in probs if all(R.get((p, m), {}).get("ok") and R[(p, m)]["us"] for m in ("cuda", "e13"))]
print(f"- both modes valid on {len(both)} problems: geomean cuda/e13 us = {gm([R[(p,'cuda')]['us']/R[(p,'e13')]['us'] for p in both]):.2f} (<1 = CUDA-only faster)")
print(f"- flow headline geomean x torch is 2.57 (46 problems)")
