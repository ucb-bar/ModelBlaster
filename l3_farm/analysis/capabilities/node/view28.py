#!/usr/bin/env python3
"""A2 item 6a on the node (zero LLM): 28_VisionTransformer's kept final IR with every fuse_view proposal applied
(linear+add through dropout views, bmm+scale through the attention views); whole model exported with the kept slot
sources (fused stages: kept sources by kind where present), driver checks + timing, original vs variant."""
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
hs = vf.propose(g, {o["name"]: 1.0 for o in g["ops"]}, specs, min_gain_pct=0.0, ir="kb_problem")
g2, added, prov = gh.apply_hints(g, [{**h, "status": "accepted"} for h in hs], specs)
vir = write_variant_ir(fin, g2, work / "ir_variant")
res = {"hints": [(h["fused_kind"], len(h["chains"])) for h in hs], "applied": len(prov["applied"]),
       "skipped": prov["skipped"]}
for tag, ir, sp in (("orig", fin, specs), ("variant", vir, {**specs, **added})):
    p = export_problem(ir, work / tag, slot_sources=srcs, specs=sp, strategy="whole", concurrency="off")
    r = build_and_run(p, arch="89", work=work / f"build_{tag}")
    res[tag] = {"passed": r.get("passed"), "e2e_us": r.get("e2e_us"), "partition_us": r.get("partition_us"),
                "log": None if r.get("passed") else (r.get("log") or "")[-1500:]}
    print(tag, json.dumps(res[tag])[:600], flush=True)
(work / "view28.json").write_text(json.dumps(res, indent=1, default=str))
print("@@VIEW28@@" + json.dumps({k: res[k] for k in ("hints", "applied")} | {t: {"passed": res[t]["passed"], "e2e_us": res[t]["e2e_us"]} for t in ("orig", "variant")}))
