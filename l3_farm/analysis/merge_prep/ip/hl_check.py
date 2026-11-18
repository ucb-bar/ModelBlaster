"""Node side: construct Config + every hook object with the headline flags; print round key + resolved settings + conflicts."""
import json, sys, tempfile
from pathlib import Path
from modelblaster.pipeline import kb_graph_opt as K, kb_resume, kb_efficiency
H = json.load(open(sys.argv[1]))
kw = {}
fields = K.Config.__dataclass_fields__
for e in H["flags"]:
    n = e["flag"].lstrip("-").replace("-", "_"); assert n in fields, n
    t = fields[n].type; v = e["value"]
    kw[n] = int(v) if "int" in str(t) and "None" not in str(t) else float(v) if "float" in str(t) else v
M = "azure/anthropic/claude-opus-5-5"
cfg = K.Config(kb=Path("/x"), model=M, reflect_model=M, upkeep_model=M, arch="89", rounds=3, rollouts=4, steps=4, variant_steps=2, max_variants=4,
               min_gain_pct=0.5, plan_mode="all", plan_top_k=32, plan_min_share=0.002, **kw)
w = Path(tempfile.mkdtemp())
eff = kb_efficiency.resolve(cfg, w, {"sent": [], "rounds": []}); cfg = kb_efficiency.apply(cfg, eff)
from modelblaster.pipeline import cuda_learnings as cl, search_fixes as sf, capabilities as cp, graph_search as gs
ext, fx, caps, g = cl.Ext(cfg, w), sf.Fixes(cfg, w), cp.Caps(cfg, w), gs.GraphSearch(cfg, w, [])
key = kb_resume.round_key(cfg, [Path("/x/ir")])
for o in (ext, fx, caps, g): key.update(o.round_key())
key.update(kb_efficiency.key_fields(cfg))
import os
for o in (ext, fx, caps): o.apply_env()
print(json.dumps({"key": key, "decide_kinds": [ext.decide_kinds(), caps.decide_kinds(), g.decide_kinds()],
                  "env": {k: v for k, v in os.environ.items() if k.startswith(("MB_", "KERNELBLASTER_"))},
                  "graph_search": g.resolved(), "exec_mode": g.exec_mode()}, indent=1, default=str))
