"""A-repaired (the one shape the sweep left on the seed gets the best A kernel for it) and A-pool4 (per-shape
routing over 4 of A's family-valid kernels; zero LLM)."""
import hashlib, json, sys
sys.path.insert(0, "/tmp/krisd/family/pyn")
import famlib as F, model as M
from modelblaster.pipeline.kb_family_tune import select_pool
P = F.W / "pilot"
mx = json.loads((P / "matrix.json").read_text())["runs"]
src = M.final_sources()
codes = {hashlib.sha1(c.encode()).hexdigest()[:12]: c for sid, c in src.items() if sid.startswith("conv2d_f16")}
m = F.manifest(M.MP)
SEED_OP = "dense_blocks.3.layers.14.2"     # the shape the sweep left on the hand seed (its winner was rejected)
hand = [s for s in m["slots"] if s["op"] == F.OP and s["instances"][0]["op_name"] == SEED_OP]
print("hand slots", [(s["id"], s["instances"][0]["op_name"]) for s in hand])
rep = {}
for s in hand:
    op = s["instances"][0]["op_name"]
    h = min((r["per_op_us"][op], h) for h, r in mx.items() if r["passed"])[1]
    rep[s["id"]] = codes[h]
    print("repair", s["id"], "with", h, mx[h]["per_op_us"][op])
(P / "A_rep.json").write_text(json.dumps(rep))
if "--rep-only" in sys.argv:
    sys.exit(0)
valid = sorted((r["sum"], h) for h, r in mx.items() if r["passed"])
r = select_pool(P / "famSeed", [codes[h] for _, h in valid], P / "A_pool4", arch=F.ARCH, kbeval=F.kbeval(), reps=3,
                max_kernels=4)
print(json.dumps({k: v for k, v in r.items() if k not in ("tuned_source", "composite_us")}, indent=1))
if r.get("tuned_source"):
    (P / "A_pool4.cu").write_text(r["tuned_source"])
