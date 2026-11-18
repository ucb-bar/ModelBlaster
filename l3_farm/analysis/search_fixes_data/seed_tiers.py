#!/usr/bin/env python3
"""A3 item 3: did ImplStore.retrieve's seed / adapt tiers fire for the sweep's per-item problems, and how often would a
nearest-shape seed have been available? Replays every run's implementation store (results/*/work/knowledge/impl) in
time order: for each slot problem (its first record = its start) the records of OTHER problems stored before it
(same op + signature + dtype + arch, speedup_vs_seed >= 1.02 for adapt / nearest) decide:
  exact   -- the seed tier (a record covering the problem's shape buckets) would fire,
  adapt   -- adapt tier (shares a bucket, does not cover all) -- per-item problems have one bucket, so ~never,
  nearest -- no exact/adapt record but some same-op record for other shapes exists (search_fixes --seed-nearest),
  none    -- nothing of that op yet.
-> seed_tiers.json + printed table (by round)."""
import collections
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results")
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "ModelBlaster", "third_party", "KernelBlaster", "src"))
from kernelblaster.knowledge.impl_store import shape_bucket  # noqa: E402


def dist(a, b):
    keys = set(a) | set(b)
    t = 0.0
    for k in keys:
        x, y = a.get(k), b.get(k)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            t += abs(math.log2((abs(x) + 1) / (abs(y) + 1)))
        elif x != y:
            t += 2
    return t / max(len(keys), 1)


rows = []
for f in sorted(glob.glob(os.path.join(RES, "*", "work", "knowledge", "impl", "index.jsonl"))):
    prob = f.split(os.sep)[-5]
    recs = [json.loads(l) for l in open(f) if l.strip()]
    recs.sort(key=lambda r: r["ts"])
    start = {}
    for r in recs:
        start.setdefault(r["problem_id"], r)
    for pid, r0 in start.items():
        t0 = r0["ts"]
        rnd = pid.rsplit("__r", 1)[-1] if "__r" in pid else "?"
        want = set(r0["buckets"])
        prior = [r for r in recs if r["ts"] < t0 and r["problem_id"] != pid and r["op"] == r0["op"]
                 and r["signature_sha"] == r0["signature_sha"] and r["dtype"] == r0["dtype"] and r["arch"] == r0["arch"]]
        exact = [r for r in prior if want <= set(r["buckets"])]
        good = [r for r in prior if (r.get("speedup_vs_seed") or 0) > 1.05 or (r.get("speedup_vs_seed") or 0) >= 1.02]
        near = [r for r in good if want & set(r["buckets"])]
        if exact:
            tier = "exact"
        elif near:
            tier = "adapt"
        elif good:
            tier = "nearest"
        else:
            tier = "none"
        d = min((min(dist(s, t) for s in r["shapes"] for t in r0["shapes"]) for r in good), default=None)
        rows.append({"problem": prob, "pid": pid, "round": rnd, "op": r0["op"], "tier": tier,
                     "nearest_dist": round(d, 3) if d is not None else None, "n_prior_good": len(good)})
by = collections.defaultdict(collections.Counter)
for r in rows:
    by[r["round"]][r["tier"]] += 1
    by["all"][r["tier"]] += 1
out = {"problems": len(rows), "by_round": {k: dict(v) for k, v in sorted(by.items())}, "rows": rows}
json.dump(out, open(os.path.join(HERE, "seed_tiers.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
ds = sorted(r["nearest_dist"] for r in rows if r["tier"] == "nearest")
if ds:
    print("nearest-tier distance quartiles", ds[len(ds) // 4], ds[len(ds) // 2], ds[3 * len(ds) // 4])
