"""Summarize validate.py metrics JSONL files: per-tag tables over the held-out set, by image source.
   python summarize.py metrics.jsonl [...]"""
import json, sys, collections
import numpy as np
recs = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
by = collections.defaultdict(list)
for r in recs:
    by[r["tag"]].append(r)
def src(r):
    o, im = r["obs"], str(r.get("image") or "")
    if not o.startswith("h"):
        return "orig13"
    return im.split(":")[0] + ("_aug" if "#" in im else "")
for tag, rs in by.items():
    cos = np.array([r["cos"] for r in rs]); me = np.array([r["maxerr"] for r in rs]); rm = np.array([r["rmse"] for r in rs])
    jm = np.array([r["joint_maxerr"] for r in rs]); jmean = np.array([r["joint_meanerr"] for r in rs])
    print(f"== {tag}: n={len(rs)}  cos mean {cos.mean():.6f} median {np.median(cos):.6f} p5 {np.percentile(cos, 5):.6f} "
          f"min {cos.min():.6f}  #<0.999 {int((cos < 0.999).sum())}  #<0.998 {int((cos < 0.998).sum())}")
    print(f"   max|err| mean {me.mean():.4f} p95 {np.percentile(me, 95):.4f} max {me.max():.4f}; rmse mean {rm.mean():.5f}")
    print("   per-joint (dims 0-5) max|err|: mean " + " ".join(f"{x:.4f}" for x in jm.mean(0)) +
          " | worst " + " ".join(f"{x:.4f}" for x in jm.max(0)))
    print("   per-joint mean|err|: " + " ".join(f"{x:.5f}" for x in jmean.mean(0)))
    g = collections.defaultdict(list)
    for r in rs:
        g[src(r)].append(r["cos"])
    print("   by source: " + "; ".join(f"{k} n={len(v)} mean {np.mean(v):.5f} min {np.min(v):.5f}" for k, v in sorted(g.items())))
    worst = sorted(rs, key=lambda r: r["cos"])[:6]
    print("   worst: " + "; ".join(f"{r['obs']}({r.get('image')}) {r['cos']:.5f}" for r in worst))
