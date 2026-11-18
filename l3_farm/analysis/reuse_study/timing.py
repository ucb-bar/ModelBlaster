#!/usr/bin/env python3
"""Availability timing summary: coverage (win pool) classified by when the first compatible donor reached the consumer, ordered by consumer start."""
import json, collections, datetime as dt
from pathlib import Path
D = Path(__file__).resolve().parent / "data"
P = json.load(open(D / "problems.json")); C = json.load(open(D / "coverage.json"))
t0 = min(d["t_start"] for d in P.values())
rows = []
for p, d in P.items():
    cnt = collections.Counter()
    tot_nr = 0
    for o in C[p]["ops"]:
        if o["ctype"] == "region":
            continue
        tot_nr += o["share"]
        s = o["win"]
        L = "b" if s["n_exact"] else ("c" if s["dmin"] is not None and s["dmin"] <= 2.5 else "d")
        if L == "d":
            cnt["none"] += o["share"]; continue
        av = (s["avail_exact"] if L == "b" else s["avail_near"])
        a = av or {}
        ideal = a.get("created_before_start")
        cnt["covered"] += o["share"]
        cnt["ideal_pre_start"] += o["share"] if ideal else 0
        cnt["seed"] += o["share"] if a.get("seed") else 0
        cnt["seed_or_refresh"] += o["share"] if (a.get("seed") or a.get("refresh")) else 0
        cnt["pre_start_not_delivered"] += o["share"] if (ideal and not a.get("seed")) else 0
    reg = 1 - tot_nr
    rows.append((d["t_start"], p, int(d["seed_snapshot"][1:]) if d["seed_snapshot"] else 0, tot_nr, cnt))
rows.sort()
print("order | problem | store version at start | non-region share | covered | ideal pre-start | delivered at seed | seed or refresh | pre-start but not delivered | start (h from sweep start)")
for t, p, v, nr, c in rows:
    print(f"{p[:30]:30s} v{v:04d} nr={nr:.2f} cov={c['covered']:.2f} ideal={c['ideal_pre_start']:.2f} seed={c['seed']:.2f} s+r={c['seed_or_refresh']:.2f} undelivered={c['pre_start_not_delivered']:.2f} t={(t-t0)/3600:.1f}h")
# terciles by start order
n = len(rows)
for name, sl in (("first third", rows[: n // 3]), ("middle", rows[n // 3: 2 * n // 3]), ("last third", rows[2 * n // 3:])):
    k = collections.Counter(); w = 0
    for _, p, v, nr, c in sl:
        w += nr
        for kk, vv in c.items():
            k[kk] += vv
    print(name, len(sl), {kk: round(vv / w, 3) for kk, vv in k.items()})
json.dump([(t, p, v, nr, dict(c)) for t, p, v, nr, c in rows], open(D / "timing.json", "w"))
