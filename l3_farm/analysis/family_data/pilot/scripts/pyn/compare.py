"""Final quiet comparison on this node, interleaved rounds (every variant once per round, so drift hits all alike).

Variants: "A" (the sweep's deployed per-shape winners) plus NAME=FILE arguments:
  NAME=x.cu    one family kernel for all 61 conv2d_f16 slots (B, C, pools)
  NAME=x.json  {slot id: source} overrides on top of A (e.g. A-repaired)
Metrics per variant (median over rounds):
  model_e2e        whole deployed DenseNet121 (90 slots), CUDA-graph replay
  family_in_model  CUPTI-attributed time of the 61 conv instances inside that run
  family_alone     the conv family partition alone (.cu variants and A only)
Usage: python compare.py [ROUNDS] NAME=FILE ..."""
import json, statistics as st, sys
from pathlib import Path
sys.path.insert(0, "/tmp/krisd/family/pyn")
import famlib as F, model as M

P = F.W / "pilot"
rounds = int(next((a for a in sys.argv[1:] if a.isdigit()), "3"))
var = {"A": ("A", None)}
for a in sys.argv[1:]:
    if "=" in a:
        n, f = a.split("=", 1)
        var[n] = ("json", json.loads(Path(f).read_text())) if f.endswith(".json") else ("cu", Path(f).read_text())
conv = M.conv_slots()
fam_ops = [o["name"] for o in F.manifest(P / "famSeed")["ops"]]


def sources(kind, x):
    if kind == "cu":
        return M.variant_sources(x)
    s = M.variant_sources()
    if kind == "json":
        s.update(x)
    return s


out = {k: {"model_e2e": [], "family_in_model": [], "family_alone": [], "passed": []} for k in var}
per_op = {k: {} for k in var}
for r in range(rounds):
    for k, (kind, x) in var.items():
        res = F.time_problem(M.MP, sources=sources(kind, x), reps=5, name=f"m{k}{r}", work=P / "cmp_time")
        att = res.get("attr_slot_us") or {}
        out[k]["passed"].append(bool(res.get("passed")))
        out[k]["model_e2e"].append(res.get("e2e_us"))
        out[k]["family_in_model"].append(round(sum(att.get(c, 0.0) for c in conv), 2))
        ops = res.get("attr_op_us") or {}
        for o in fam_ops:
            per_op[k].setdefault(o, []).append(ops.get(o))
        fa = None
        if kind == "A":
            fr = F.time_problem(P / "famA", reps=5, name=f"fA{r}", work=P / "cmp_time")
            fa = F.family_summary(fr, P / "famA")
        elif kind == "cu":
            fr = F.time_problem(P / "famSeed", sources={"conv2d_f16": x}, reps=5, name=f"f{k}{r}", work=P / "cmp_time")
            fa = F.family_summary(fr, P / "famSeed")
        if fa is not None:
            out[k]["family_alone"].append(fa["family_kernel_us"])
            out[k]["passed"].append(bool(fa["passed"]))
        print(r, k, out[k]["model_e2e"][-1], out[k]["family_in_model"][-1], fa and fa["family_kernel_us"], flush=True)
med = lambda v: round(st.median([x for x in v if x is not None]), 2) if any(x is not None for x in v) else None
summary = {k: {m: (all(v) if m == "passed" else med(v)) for m, v in d.items()} for k, d in out.items()}
summary["_per_op_median"] = {k: {o: med(v) for o, v in d.items()} for k, d in per_op.items()}
summary["_raw"] = out
(P / "compare.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if not k.startswith("_")}, indent=1))
