#!/usr/bin/env python3
"""Real fold shapes: layout_fold.fold on the six final IRs (l3_farm/results/<p>/work/final/<p>/ir/graph.json) ->
cases_real.json (every bmm_strided_f16 op or fused-head stage 0, with its int params). Pure python (no torch)."""
import json, sys
from pathlib import Path
L3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(L3.parent / "ModelBlaster" / ".."))   # makes `ModelBlaster` importable as a package dir
import importlib.util
spec = importlib.util.spec_from_file_location("layout_fold", L3.parent / "ModelBlaster" / "pipeline" / "layout_fold.py")
lf = importlib.util.module_from_spec(spec); spec.loader.exec_module(lf)
NAMES = "43_MinGPTCausalAttention 44_MiniGPTBlock 49_Mamba2ReturnFinalState 50_ReLUSelfAttention 46_NetVladWithGhostClusters 47_NetVladNoGhostClusters".split()
out = []
for p in NAMES:
    g = json.loads((L3 / f"results/{p}/work/final/{p}/ir/graph.json").read_text())
    g2, info = lf.fold(g)
    for o in g2["ops"]:
        for s in ([o] if o["op"] == "bmm_strided_f16" else o.get("sub_ops", [])[:1] if o["op"].startswith("fused__bmm_strided") else []):
            if s["op"] == "bmm_strided_f16":
                out.append({"problem": p, "op": o["name"], "head_of": o["op"] if o is not s else None, **s["shape"]})
Path(__file__).with_name("cases_real.json").write_text(json.dumps(out, indent=1))
for c in out: print(c)
