import json, collections
from pathlib import Path
D = Path(__file__).resolve().parent / "data"
S = json.load(open(D / "summary.json")); C = json.load(open(D / "coverage.json")); P = json.load(open(D / "problems.json"))
POT = json.load(open(D / "potential.json"))
f = lambda x: f"{100*x:.0f}"
L = ["| problem | e2e us | own | shared-in | seed/weak | region | S: exact/near/none | any donor | at seed / refresh / pre-start undeliv. / during / after | start (h) | store ver | zero-GPU exact-pred |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
t0 = min(d["t_start"] for d in P.values())
for p, d in sorted(S.items(), key=lambda kv: int(kv[0].split("_")[0])):
    c, t = d["c"], d["t"]
    L.append(f"| {p} | {C[p]['e2e_us']:.0f} | {f(c.get('Aown',0))} | {f(c.get('Ain',0))} | {f(c.get('S',0))} | {f(c.get('region',0))} | {f(c.get('Sb',0))}/{f(c.get('Sc',0))}/{f(c.get('Sd',0))} | {f(t.get('cov',0))} | {f(t.get('seed',0))}/{f(t.get('refresh',0))}/{f(t.get('pre_start_undelivered',0))}/{f(t.get('during_run',0))}/{f(t.get('after_finish',0))} | {(P[p]['t_start']-t0)/3600:.1f} | {P[p]['seed_snapshot'] or '-'} | {f(POT[p]['pred']['win'])} |")
(D / "table_compact.md").write_text("\n".join(L))
print(len(L))
