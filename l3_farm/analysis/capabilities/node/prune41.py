#!/usr/bin/env python3
"""A2 item 1 on the node: 41's shipped final program exported with dead-op pruning on and off (kept slot sources,
sequential walker), both timed the flow's way (kb_model_run._time_builtin), interleaved. Usage: prune41.py BUNDLE WORK REPS"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
from modelblaster.pipeline.kb_export import export_problem
from modelblaster.pipeline.kb_proposals import _all_specs
from modelblaster.pipeline.kb_model_run import _time_builtin

bundle, work, reps = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
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
kman = json.loads((bundle / "problem" / "manifest.json").read_text())
sources = {s["id"]: (bundle / "problem" / s["file"]).read_text() for s in kman["slots"]}
probs = {}
for mode in ("on", "off"):
    p = export_problem(fin, work / f"prob_{mode}", slot_sources=sources, specs=dict(_all_specs()), strategy="whole",
                       name=kman.get("name"), prune_dead=(mode == "on"), concurrency="off")
    m = json.loads((p / "manifest.json").read_text())
    probs[mode] = p
    print(mode, "ops", len(m["ops"]), "pruned", m["partition"].get("pruned_dead"), flush=True)
res = {"on": [], "off": []}
for r in range(reps):
    for mode in (("on", "off") if r % 2 == 0 else ("off", "on")):
        t = _time_builtin(probs[mode], arch="89", work=work / f"bt_{mode}", reps=1)
        res[mode].append({k: t.get(k) for k in ("passed", "e2e_us", "e2e_basis", "graph_nodes")})
        print(mode, res[mode][-1], flush=True)
med = {m: sorted(x["e2e_us"] for x in v if x.get("e2e_us"))[len(v) // 2] for m, v in res.items()}
out = {"runs": res, "median_e2e_us": med, "saved_us": med["off"] - med["on"]}
(work / "prune41.json").write_text(json.dumps(out, indent=1))
print("@@PRUNE41@@" + json.dumps(out), flush=True)
