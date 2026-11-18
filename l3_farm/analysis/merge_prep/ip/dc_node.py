"""Default-behaviour check, runs ON A NODE under a tree (PYTHONPATH=<tree root containing modelblaster/>).
usage: dc_node.py INPUTS OUT problem [problem...]   env MB_PRUNE_DEAD as set by the caller.
Per problem: plan from the SWEEP's saved round-0 profile (plan_mode all, top_k 32, min_share 0.002 = the farm's flags), export_plan with an empty cache,
whole-IR export, round key with all-default Config. No GPU, no LLM."""
import json, sys, os, shutil, hashlib
from pathlib import Path
inp, out = Path(sys.argv[1]), Path(sys.argv[2]); probs = sys.argv[3:]
out.mkdir(parents=True, exist_ok=True)
from modelblaster.pipeline import kb_graph_opt as KGO, kb_resume
from modelblaster.pipeline.kb_plan import make_plan, export_plan
from modelblaster.pipeline.kb_ingest import CudaCache
from modelblaster.pipeline.kb_proposals import _all_specs
from modelblaster.pipeline.kb_export import export_problem

def build_key(irs, work):
    M = "azure/anthropic/claude-opus-5-5"
    kw = dict(kb=Path("/x"), model=M, reflect_model=M, arch="89", rounds=3, rollouts=4, steps=4, variant_steps=2, max_variants=4,
              min_gain_pct=0.5, plan_mode="all", plan_top_k=32, plan_min_share=0.002, upkeep_model=M)
    cfg = KGO.Config(**kw)
    notes = []
    state = {"sent": [], "rounds": []}
    try:
        from modelblaster.pipeline import kb_efficiency
        eff = kb_efficiency.resolve(cfg, work, state); cfg = kb_efficiency.apply(cfg, eff)
    except ImportError: kb_efficiency = None
    key = kb_resume.round_key(cfg, irs)
    try:
        from modelblaster.pipeline import kb_global; key.update(kb_global.key_fields(cfg, None))
    except Exception as e: notes.append(f"gk:{e}")
    for name, mod, cls in (("cuda_learnings", "cuda_learnings", "Ext"), ("search_fixes", "search_fixes", "Fixes"), ("capabilities", "capabilities", "Caps")):
        try:
            m = __import__(f"modelblaster.pipeline.{mod}", fromlist=["x"]); o = getattr(m, cls)(cfg, work); key.update(o.round_key())
        except ImportError: pass
    try:
        from modelblaster.pipeline import graph_search as g; key.update(g.GraphSearch(cfg, work, irs).round_key())
    except ImportError: pass
    if kb_efficiency: key.update(kb_efficiency.key_fields(cfg))
    return key, cfg, notes

res = {}
for p in probs:
    d = out / p; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    ir = inp / p / "ir"
    prof = json.loads((inp / p / "profile.json").read_text())
    prof["ir_dir"] = prof["source_ir"] = str(ir)
    r = {}
    try:
        plan = make_plan([prof], coverage=2.0, top_k=32, per_item=True, min_share=0.002, exclude=set())
        (d / "plan.json").write_text(json.dumps(plan, indent=1, sort_keys=True, default=str))
        names = {q.get("id", q["ir"]): f"{q.get('id', q['ir'])}__r00" for q in plan["problems"]}
        dirs = export_plan(plan, d / "problems", cache=CudaCache(d / "cache"), arch="89", specs=dict(_all_specs()), names=names)
        r["plan_problems"] = len(plan["problems"]); r["exported"] = len(dirs)
    except Exception as e:
        r["plan_error"] = f"{type(e).__name__}: {e}"
    try:
        m = export_problem(ir, d / "whole", specs=dict(_all_specs()), name=p + "__whole", strategy="whole")
        r["whole"] = str(m)
    except Exception as e:
        r["whole_error"] = f"{type(e).__name__}: {e}"
    key, cfg, notes = build_key([ir], d / "w")
    (d / "round_key.json").write_text(json.dumps({**key, "irs": [Path(x).name for x in key["irs"]]}, indent=1, sort_keys=True, default=str))
    res[p] = r
(out / "summary.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
