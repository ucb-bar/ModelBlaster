"""A: family problems (seed one-slot, A composite, per-instance split) and their times on this node."""
import json, sys
sys.path.insert(0, "/tmp/krisd/family/pyn")
import famlib as F
from modelblaster.pipeline.kb_ingest import CudaCache
P = F.W / "pilot"; P.mkdir(exist_ok=True)
out = {}
famA = F.export_family(P / "famA", CudaCache(F.W / "cacheA"))
famSeed = F.export_family(P / "famSeed")
famSplit = F.export_family(P / "famSplit", split_each=True)
for name, prob, reps in (("A", famA, 5), ("seed", famSeed, 3)):
    r = F.time_problem(prob, reps=reps, name=name)
    out[name] = F.family_summary(r, prob)
    print(name, {k: v for k, v in out[name].items() if k != "per_op_us"}, flush=True)
m = F.manifest(famSeed)
out["instances"] = [{"op": o["name"], "shape": next(i["shape"] for s in m["slots"] for i in s["instances"] if i["op_name"] == o["name"])}
                    for o in m["ops"]]
out["signature"] = m["slots"][0]["signature"]
json.dump(out, open(P / "A_family.json", "w"), indent=1)
