"""Family kernels deployed as ONE conv2d_f16 slot (61 instances) in the whole model; interleaved rounds."""
import json, statistics as st, sys
from pathlib import Path
sys.path.insert(0, "/tmp/krisd/family/pyn")
import famlib as F, model as M
P = F.W / "pilot"
var = {}
for a in sys.argv[2:]:
    n, f = a.split("=", 1)
    var[n] = Path(f).read_text()
if not (M.MP1 / "manifest.json").exists():
    M.build_single(next(iter(var.values())))
base = {s["id"]: None for s in F.manifest(M.MP1)["slots"]}
fs = M.final_sources()
srcs0 = {k: v for k, v in fs.items() if k in base}
out = {k: {"model_e2e": [], "family_in_model": [], "passed": []} for k in var}
for r in range(int(sys.argv[1])):
    for k, s in var.items():
        res = F.time_problem(M.MP1, sources={**srcs0, F.OP: s}, reps=5, name=f"s{k}{r}", work=P / "cmp1_time")
        att = res.get("attr_slot_us") or {}
        out[k]["passed"].append(bool(res.get("passed")))
        out[k]["model_e2e"].append(res.get("e2e_us"))
        out[k]["family_in_model"].append(att.get(F.OP))
        print(r, k, res.get("passed"), res.get("e2e_us"), att.get(F.OP), (res.get("errors") or [])[:2], flush=True)
med = lambda v: round(st.median([x for x in v if x is not None]), 2) if any(x is not None for x in v) else None
summ = {k: {m: (all(v) if m == "passed" else med(v)) for m, v in d.items()} for k, d in out.items()}
summ["_raw"] = out
(P / "compare1.json").write_text(json.dumps(summ, indent=1))
print(json.dumps({k: v for k, v in summ.items() if k != "_raw"}, indent=1))
