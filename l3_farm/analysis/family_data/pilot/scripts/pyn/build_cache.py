"""A: rebuild the sweep's DenseNet121 kernel cache from its rounds' winners.json (ingested in round order, as
kb_graph_opt did), then replay the IR rejections the sweep made (profile fallbacks) until cache_plan picks the
same impl per op as the sweep's final deployed problem (final_manifest.json)."""
import json, re, sys
sys.path.insert(0, "/tmp/krisd/family/pyn")
import famlib as F
from modelblaster.pipeline.kb_ingest import CudaCache, ingest, cache_plan
from modelblaster.pipeline.kb_export import slot_id_of
from modelblaster.pipeline.kb_proposals import with_fused_specs
from modelblaster.pipeline import graph_hints as gh

out = F.W / "cacheA"
if out.exists():
    import shutil; shutil.rmtree(out)
cache = CudaCache(out)
specs, g, _ = F.specs_and_graph()
g = gh.unique_names(g)
specs = with_fused_specs(g, specs)
for r in ("r00", "r01", "r02"):
    rep = ingest(F.SWEEP / r / "winners" / "winners.json", cache, specs)
    print(r, {s: sum(1 for x in rep if x["status"] == s) for s in {x["status"] for x in rep}})

fm = json.loads((F.SWEEP / "final_manifest.json").read_text())
want = {}
seed = {s["id"]: s.get("seed_source", "") for s in fm["slots"]}
for o in fm["ops"]:
    t = seed.get(o["slot"], "")
    m = re.fullmatch(r"cache:([0-9a-f]{12})", t)
    want[o["name"]] = m.group(1) if m else ("hand" if t == "hand" else None)
ir_name = g["name"]
for it in range(12):
    cp = cache_plan(g, cache, f"sm_{F.ARCH}", specs)
    have = {}
    for o in cp["graph"]["ops"]:
        if o["op"] not in specs:
            continue
        sid = slot_id_of(o, specs)
        t = cp["tags"].get(sid, "hand")
        m = re.fullmatch(r"cache:([0-9a-f]{12})", t)
        have[o["name"]] = m.group(1) if m else ("hand" if t == "hand" else t)
    bad = {n: (w, have.get(n)) for n, w in want.items() if w is not None and have.get(n) != w}
    print("iteration", it, "mismatches", len(bad), list(bad.items())[:6])
    if not bad:
        break
    changed = False
    for n, (w, h) in bad.items():
        e = next((e for e in cache.entries if e["impl_id"] == h), None)
        if e is not None and ir_name not in e["rejected"] and (w == "hand" or any(x["impl_id"] == w for x in cache.entries)):
            e["rejected"].append(ir_name)
            changed = True
    if not changed:
        break
cache.save()
conv = [o for o in cp["graph"]["ops"] if o["op"] == F.OP]
print("conv instances", len(conv), "cache entries", len(cache.entries), "rejected",
      sum(1 for e in cache.entries if e["rejected"]))
json.dump({"mismatches": {k: list(v) for k, v in bad.items()}}, open(F.W / "logs" / "cacheA_check.json", "w"), indent=1)
