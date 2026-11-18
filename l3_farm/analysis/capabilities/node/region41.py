#!/usr/bin/env python3
"""A2 item 4 on the node (zero LLM): 41's final IR with the rnn_stack region (composite seed from the kept winners)
vs the per-op program, both with the shipped kernel mapping and with the forward winner v0 for every GRU (A1's reuse
result); checks + flow timing, interleaved. Usage: region41.py BUNDLE WORK(with ir_final from prune41) REPS"""
import json, sys
from pathlib import Path
from modelblaster.pipeline.kb_export import export_problem
from modelblaster.pipeline.kb_proposals import _all_specs, with_fused_specs, write_variant_ir
from modelblaster.pipeline.kb_model_run import _time_builtin
from modelblaster.pipeline import region_slots as rs

bundle, work, reps = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
fin = work / "ir_final"
g = json.loads((fin / "graph.json").read_text())
specs = with_fused_specs(g, dict(_all_specs()))
kman = json.loads((bundle / "problem" / "manifest.json").read_text())
src = {s["id"]: (bundle / "problem" / s["file"]).read_text() for s in kman["slots"]}
v0 = {**src, **{k: src["gru_seq_f16__v0"] for k in src if k.startswith("gru_seq_f16__v")}}
names = rs.find_regions(g, "rnn_stack", specs)[0]
gr, spec = rs.fuse_region(g, names, specs, "rnn_stack")
rir = write_variant_ir(fin, gr, work / "ir_region")
import os
probs = {}
# (key, ir, sources, walker concurrency, region seed parallel)
arms = [("v0", fin, v0, "off", "0"), ("v0_conc", fin, v0, "on", "0"),
        ("v0_region_seq", rir, v0, "off", "0"), ("v0_region_par", rir, v0, "off", "1"),
        ("v0_region_par_conc", rir, v0, "on", "1")]
if len(sys.argv) > 4:
    arms = [a for a in arms if a[0] in sys.argv[4].split(",")]
for key, ir, s, conc, par in arms:
    os.environ["MB_REGION_SEED_PARALLEL"] = par
    p = export_problem(ir, work / f"r_{key}", slot_sources=s, specs={**specs, spec.op: spec},
                       strategy="whole", concurrency=conc)
    m = json.loads((p / "manifest.json").read_text())
    print(key, "slots", [(x["id"], x["seed_source"]) for x in m["slots"]], flush=True)
    probs[key] = p
res = {k: [] for k in probs}
for r in range(reps):
    for k in (list(probs) if r % 2 == 0 else list(probs)[::-1]):
        t = _time_builtin(probs[k], arch="89", work=work / f"bt_{k}", reps=1)
        res[k].append({x: t.get(x) for x in ("passed", "e2e_us", "e2e_basis", "graph_nodes")})
        if not t.get("passed"):
            res[k][-1]["log"] = (t.get("log") or "")[-1500:]
        print(k, res[k][-1], flush=True)
med = {k: sorted(x["e2e_us"] for x in v if x.get("e2e_us"))[len(v) // 2] if any(x.get("e2e_us") for x in v) else None
       for k, v in res.items()}
out = {"members": len(names), "kind": spec.op, "runs": res, "median_e2e_us": med}
(work / "region41.json").write_text(json.dumps(out, indent=1))
print("@@REGION41@@" + json.dumps({"median": med, "kind": spec.op}), flush=True)
