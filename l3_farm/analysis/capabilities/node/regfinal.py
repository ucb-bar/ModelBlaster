#!/usr/bin/env python3
"""Paired timing (flow method, interleaved): 41 as one region slot with the LLM's best region kernel vs the best per-op
schedule (forward winner everywhere + concurrent branches). Usage: regfinal.py REPS"""
import json, shutil, sys
from pathlib import Path
from modelblaster.pipeline.kb_export import export_problem
from modelblaster.pipeline.kb_proposals import _all_specs, with_fused_specs
from modelblaster.pipeline.kb_model_run import _time_builtin

A = Path("/tmp/krisd/a2"); reps = int(sys.argv[1])
reg = A / "regfinal" / "region"
shutil.rmtree(A / "regfinal", ignore_errors=True)
shutil.copytree(A / "reg" / "prob", reg)
for f in (A / "reg" / "run" / "a2_41_region" / "best" / "slots").glob("*.cu"):
    shutil.copyfile(f, reg / "slots" / f.name)
fin = A / "runs" / "prune41" / "ir_final"
g = json.loads((fin / "graph.json").read_text())
specs = with_fused_specs(g, dict(_all_specs()))
bundle = A / "b41"
kman = json.loads((bundle / "problem" / "manifest.json").read_text())
src = {s["id"]: (bundle / "problem" / s["file"]).read_text() for s in kman["slots"]}
v0 = {**src, **{k: src["gru_seq_f16__v0"] for k in src if k.startswith("gru_seq_f16__v")}}
conc = export_problem(fin, A / "regfinal" / "v0_conc", slot_sources=v0, specs=specs, strategy="whole", concurrency="on")
probs = {"region_llm": reg, "v0_conc": conc}
res = {k: [] for k in probs}
for r in range(reps):
    for k in (list(probs) if r % 2 == 0 else list(probs)[::-1]):
        t = _time_builtin(probs[k], arch="89", work=A / "regfinal" / f"bt_{k}", reps=1)
        res[k].append({x: t.get(x) for x in ("passed", "e2e_us", "e2e_basis", "graph_nodes")})
        print(k, res[k][-1], flush=True)
med = {k: sorted(x["e2e_us"] for x in v if x.get("e2e_us"))[len(v) // 2] for k, v in res.items()}
(A / "regfinal" / "regfinal.json").write_text(json.dumps({"runs": res, "median": med}, indent=1))
print("@@REGFINAL@@" + json.dumps(med))
