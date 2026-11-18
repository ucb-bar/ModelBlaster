"""Cross-evaluation of A's per-shape conv winners: every winner run on EVERY shape of the family (one-slot
family problem, CUPTI per-op times). Answers: how shape-specific are the per-shape kernels? What does the best
single A kernel (zero LLM: pick one) give for the whole family, and a per-shape pick among A's kernels?"""
import hashlib, json, sys
sys.path.insert(0, "/tmp/krisd/family/pyn")
import famlib as F, model as M

P = F.W / "pilot"
fam = P / "famSeed"
out_f = P / "matrix.json"
res = json.loads(out_f.read_text()) if out_f.exists() else {"runs": {}}
src = M.final_sources()
m = F.manifest(M.MP)
owner = {}
for s in m["slots"]:
    if s["op"] == F.OP and s["id"] in src:
        owner.setdefault(hashlib.sha1(src[s["id"]].encode()).hexdigest()[:12], []).append(
            {"slot": s["id"], "op": s["instances"][0]["op_name"], "seed": s.get("seed_source")})
codes = {}
for s in m["slots"]:
    if s["op"] == F.OP and s["id"] in src:
        codes[hashlib.sha1(src[s["id"]].encode()).hexdigest()[:12]] = src[s["id"]]
print("distinct conv kernels in A:", len(codes), flush=True)
for i, (h, code) in enumerate(sorted(codes.items())):
    if h in res["runs"]:
        continue
    r = F.time_problem(fam, sources={"conv2d_f16": code}, reps=3, name=f"w{h}", work=P / "matrix_time")
    fsum = F.family_summary(r, fam)
    res["runs"][h] = {"owner": owner.get(h), "passed": fsum["passed"], "per_op_us": fsum["per_op_us"],
                      "sum": fsum["family_kernel_us"], "e2e": fsum["e2e_us"],
                      "err": None if fsum["passed"] else (r.get("log") or "")[-300:]}
    print(i, h, owner.get(h, [{}])[0].get("op"), fsum["passed"], fsum["family_kernel_us"], flush=True)
    out_f.write_text(json.dumps(res, indent=1))
