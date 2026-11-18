"""Retention-cap sensitivity (in memory, no store): per op pool of the sweep's verified records (speedup >= 1.0),
Pool.retain(cap) -> share of timed shapes that keep a kernel within 10% of the best record on that shape, and the
share of total best time covered (shapes weighted by their best time)."""
import collections, glob, json, sys
from pathlib import Path
FARM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path.home() / "Documents/ModelBlaster-clone/ModelBlaster/third_party/KernelBlaster/src"))
from kernelblaster.knowledge.impl_pool import Pool, shape_key  # noqa: E402
recs = collections.defaultdict(list)
for f in glob.glob(str(FARM / "results/*/work/knowledge/impl/index.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if (r.get("speedup_vs_seed") or 0) >= 1.02:
            recs[(r["op"], r["signature_sha"], r["arch"], r["dtype"])].append(r)
out = {}
for cap in (8, 16, 32, 64):
    hit = n = tw = hw = 0.0
    for (op, sig, arch, dt), rs in recs.items():
        p = Pool(op, sig, arch, dt)
        for r in rs:
            p.add_kernel(r["code_sha"], record=r)
        timed = collections.defaultdict(dict)
        for r in rs:
            if len(r["shapes"]) == 1:
                k = shape_key(r["shapes"][0]); timed[k][r["code_sha"][:16]] = min(r["slot_us"], timed[k].get(r["code_sha"][:16], 1e30))
        p.retain(cap)
        for sk, t in timed.items():
            b = min(t.values()); n += 1; tw += b
            if any(k in p.kernels and us <= 1.1 * b for k, us in t.items()):
                hit += 1; hw += b
    out[cap] = {"shapes_within_10pct": round(hit / n, 3), "time_weighted": round(hw / tw, 3)}
    print(cap, out[cap], flush=True)
Path(__file__).with_name("retention_caps.json").write_text(json.dumps(out, indent=1))
