#!/usr/bin/env python3
"""Coverage of the headline problems' seed/weak ops (and all ops) by kernels of the FIRST sweep (results/*), other problems only:
what an up-front seeded store would have offered at t=0 (before any headline problem started)."""
import json, glob, collections, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from coverage import shape_distance, skey
L3 = Path(__file__).resolve().parents[2]
D = Path(__file__).resolve().parent / "data"
P = json.load(open(D / "problems.json"))
pool = {}
for f in glob.glob(str(L3 / "results" / "*" / "work" / "knowledge" / "impl" / "index.jsonl")):
    home = f.split("/")[-5]
    for ln in open(f):
        try: r = json.loads(ln)
        except ValueError: continue
        if (r.get("speedup_vs_seed") or 0) < 1.02 or r.get("target", "cuda") != "cuda": continue
        o = (r.get("problem_id") or "").split("::")[0] if "::" in (r.get("problem_id") or "") else home
        k = (o, r["op"], r["code_sha"])
        row = pool.setdefault(k, {"origin": o, "op": r["op"], "shapes": [], "sp": 0.0})
        for s in r.get("shapes") or []:
            if isinstance(s, dict) and s not in row["shapes"]: row["shapes"].append(s)
        row["sp"] = max(row["sp"], r.get("speedup_vs_seed") or 0)
by = collections.defaultdict(list)
for r in pool.values():
    r["keys"] = {skey(s) for s in r["shapes"]}; by[r["op"]].append(r)
print("prev pool", len(pool), "kinds", len(by))
res = {}
tot = collections.Counter()
for p, d in P.items():
    C = collections.Counter()
    for o in d["ops"]:
        if o["ctype"] == "region": C["region"] += o["share"]; continue
        isS = o["ctype"] == "seed" or o.get("weak")
        sk = skey(o["shape"]); ex = False; dm = None
        for r in by.get(o["kind"], []):
            if r["origin"] == p: continue
            if sk in r["keys"]: ex = True
            if r["shapes"]:
                dd = min(shape_distance(o["shape"], s) for s in r["shapes"]); dm = dd if dm is None else min(dm, dd)
        lv = "b" if ex else ("c" if dm is not None and dm <= 2.5 else "d")
        C[("S" if isS else "A") + lv] += o["share"]
    res[p] = dict(C)
    for k, v in C.items(): tot[k] += v
n = len(P)
print({k: round(v / n, 4) for k, v in sorted(tot.items())})
json.dump(res, open(D / "prev_cov.json", "w"))
for p, c in res.items():
    if c.get("Sb", 0) + c.get("Sc", 0) + c.get("Sd", 0) > 0.01: print(p, {k: round(v, 3) for k, v in c.items() if k[0] == "S"})
