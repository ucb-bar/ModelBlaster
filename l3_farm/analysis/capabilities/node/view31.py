#!/usr/bin/env python3
"""A2 item 6a on the node: 31_VisionAttention's attention scale behind a view (attn#10 bmm -> attn#11 view ->
attn#12 ewise_scalar, 12.6 ms of 38 ms). Exports (zero LLM) the original sub-graph with the shipped winners and the
fuse_view variant (fused bmm+scale slot, composite seed = the shipped bmm winner + the shipped scale kernel), checks
and times both. Usage: view31.py BUNDLE WORK  -> WORK/view31.json; WORK/fused = the problem for a KB run."""
import json, os, shutil, subprocess, sys
from pathlib import Path
from modelblaster.pipeline import graph_hints as gh, view_fusion as vf
from modelblaster.pipeline.kb_export import export_problem, build_and_run
from modelblaster.pipeline.kb_proposals import _all_specs, with_fused_specs, write_variant_ir

bundle, work = Path(sys.argv[1]), Path(sys.argv[2])
work.mkdir(parents=True, exist_ok=True)
src = work / "ir_src"
if not (src / ".done").exists():
    subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file", str(bundle / "problem.py"),
                    "--bench-target-mb", "0", "--quant", "fp16", "--out-dir", str(src)], check=True, capture_output=True)
    (src / ".done").touch()
fin = work / "ir_final"; fin.mkdir(exist_ok=True)
shutil.copyfile(bundle / "ir" / "graph.json", fin / "graph.json")
for f in ("weights.npz", "io.npz"):
    if not (fin / f).exists():
        os.symlink(src / f, fin / f)
g = gh.unique_names(json.loads((fin / "graph.json").read_text()))
specs = with_fused_specs(g, dict(_all_specs()))
kman = json.loads((bundle / "problem" / "manifest.json").read_text())
srcs = {s["id"]: (bundle / "problem" / s["file"]).read_text() for s in kman["slots"]}
op_us = {"attn#12": 12652.4, "attn#10": 6401.4}
hs = vf.propose(g, op_us, specs, min_gain_pct=0.5, ir="kb_problem", total_us=37993.5)
assert hs and hs[0]["chains"] == [["attn#10", "attn#12"]], hs
h = {**hs[0], "status": "accepted"}
g2, added, prov = gh.apply_hints(g, [h], specs)
specs2 = {**specs, **added}
vir = write_variant_ir(fin, g2, work / "ir_variant")
names = ["attn#10", "attn#11", "attn#12"]
oidx = [i for i, o in enumerate(g["ops"]) if o["name"] in names]
orig = export_problem(fin, work / "orig", op_indices=oidx, slot_sources=srcs, specs=specs, strategy="hint-original",
                      name="a2_31_scale_orig", concurrency="off")
new = prov["applied"][0]["op"]
vidx = [i for i, o in enumerate(g2["ops"]) if o["name"] == new]
seed = {"bmm_f16": srcs["bmm_f16__v1"], "ewise_scalar_f16": srcs["ewise_scalar_f16"]}
fused = export_problem(vir, work / "fused", op_indices=vidx, slot_sources=seed, specs=specs2, strategy="hint-variant",
                       name="a2_31_scale_fused", concurrency="off")
res = {"hint": h, "applied": prov["applied"]}
for tag, p in (("orig", orig), ("fused", fused)):
    m = json.loads((p / "manifest.json").read_text())
    r = build_and_run(p, arch="89", work=work / f"build_{tag}")
    res[tag] = {"slots": [(s["id"], s["seed_source"]) for s in m["slots"]], "passed": r.get("passed"),
                "partition_us": r.get("partition_us"), "e2e_us": r.get("e2e_us"), "slot_us": r.get("slot_us"),
                "log": None if r.get("passed") else (r.get("log") or "")[-1500:]}
    print(tag, json.dumps(res[tag])[:800], flush=True)
(work / "view31.json").write_text(json.dumps(res, indent=1, default=str))
print("@@VIEW31@@" + json.dumps({k: {kk: res[k].get(kk) for kk in ("passed", "partition_us", "slot_us")}
                                  for k in ("orig", "fused")}), flush=True)
