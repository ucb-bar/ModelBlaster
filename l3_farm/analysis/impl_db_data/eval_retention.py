"""Retention coverage (A5): for every (op, signature, exact shape) the sweep timed, does the store keep a kernel within
10% of the best record ever measured on that shape?  winners-only (gk_impl, the current merge rule) vs the pool
(gk_pool, usefulness + diversity retention, cap 16 per op).  Zero LLM, reads a store COPY.

    python3 eval_retention.py STORE_COPY VID [--out impl_db_data/retention_eval.json]
"""
import argparse, collections, glob, json, sqlite3, sys
from pathlib import Path
FARM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path.home() / "Documents/ModelBlaster-clone/ModelBlaster/third_party/KernelBlaster/src"))
from kernelblaster.knowledge.impl_pool import shape_key  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("store"); ap.add_argument("vid")
ap.add_argument("--out", default=str(Path(__file__).with_name("retention_eval.json"))); a = ap.parse_args()
timed = collections.defaultdict(dict)
for f in glob.glob(str(FARM / "results/*/work/knowledge/impl/index.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if len(r["shapes"]) != 1 or (r.get("speedup_vs_seed") or 0) < 1.0:
            continue
        k = (r["op"], r["signature_sha"], shape_key(r["shapes"][0]))
        timed[k][r["code_sha"]] = min(r["slot_us"], timed[k].get(r["code_sha"], 1e30))
c = sqlite3.connect(f"file:{a.store}/snapshots/{a.vid}.db?mode=ro&immutable=1", uri=True)
win = collections.defaultdict(set)
for (d,) in c.execute("SELECT data FROM gk_impl"):
    d = json.loads(d); win[(d["op"], d["signature_sha"])].add(d["code_sha"])
pool = collections.defaultdict(set)
for (d,) in c.execute("SELECT data FROM gk_pool"):
    d = json.loads(d); pool[(d["op"], d["signature_sha"])].add(d["code_sha"])
res = {}
for name, S in (("winners", win), ("pool", pool)):
    hit = best = n = 0
    per_op = collections.Counter(); per_op_n = collections.Counter()
    for (op, sig, sk), t in timed.items():
        b = min(t.values()); good = {cs for cs, us in t.items() if us <= 1.10 * b}
        n += 1; per_op_n[op] += 1
        if good & S[(op, sig)]:
            hit += 1; per_op[op] += 1
        if min(t, key=t.get) in S[(op, sig)]:
            best += 1
    res[name] = {"shapes": n, "kept_within_10pct": round(hit / n, 3), "kept_best": round(best / n, 3),
                 "kernels": sum(len(v) for v in S.values()),
                 "worst_ops": sorted(((op, round(per_op[op] / per_op_n[op], 2), per_op_n[op]) for op in per_op_n
                                      if per_op_n[op] >= 5), key=lambda x: x[1])[:8]}
Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
