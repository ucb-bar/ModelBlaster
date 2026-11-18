#!/usr/bin/env python3
"""--graph-concurrency on the sweep's shipped models, zero LLM (node side).

  conc_check.py NAME DATA_DIR OUT_DIR [--reps 5]

The shipped final model (its slot sources, guarded on their instances' shapes, accepted graph hints applied) is
exported twice -- sequential walker and concurrent branches (split budget) -- and both are re-timed interleaved.
Reports the dependency analysis (max width, branches, lanes) and the times. -> OUT_DIR/conc_check.json
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("data")
ap.add_argument("out")
ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args()

from modelblaster.pipeline import graph_hints as gh                                 # noqa: E402
from modelblaster.pipeline.kb_export import build_and_run, export_problem           # noqa: E402
from modelblaster.pipeline.kb_ingest import CudaCache, cache_plan                    # noqa: E402
from modelblaster.pipeline.kb_proposals import (_all_specs, _median, remeasure, spec_from_kind,  # noqa: E402
                                                with_fused_specs, write_variant_ir)
from modelblaster.pipeline.kernel_abi import parse_signature                         # noqa: E402

data, out = Path(a.data).resolve(), Path(a.out).resolve()
out.mkdir(parents=True, exist_ok=True)
ir = out / "ir"
if not (ir / ".done").exists():
    cp = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file",
                         str(data / "problem.py"), "--bench-target-mb", "0", "--quant", "fp16", "--out-dir", str(ir)],
                        capture_output=True, text=True)
    if cp.returncode:
        sys.exit("extract failed: " + (cp.stdout + cp.stderr)[-2000:])
    (ir / ".done").write_text("1")
specs = dict(_all_specs())
doc = gh.load(data / "work" / "graph_hints.json")
g, added, _ = gh.apply_hints(json.loads((ir / "graph.json").read_text()), doc["hints"], specs)
specs.update(added)
specs = with_fused_specs(g, specs)
cache = CudaCache(out / "cache")
cache.entries.clear()
fin = data / "work" / "final" / a.name / "problem"
man = json.loads((fin / "manifest.json").read_text())
for s in man["slots"]:
    spec = specs.get(s["op"]) or spec_from_kind(s["op"], specs)
    cache.add(op=s["op"], entry=parse_signature(spec.signature)[0], signature=spec.signature, arch="sm_89",
              shapes=[i["shape"] for i in s["instances"]], code=(fin / s["file"]).read_text(), slot_us=0.0,
              speedup=None, impl_id=f"final:{s['id']}", source={})
plan = cache_plan(g, cache, "sm_89", specs)
d = write_variant_ir(ir, plan["graph"], out / "plan_ir")
probs, rec = {}, {"name": a.name}
for conc in ("off", "on"):
    probs[conc] = export_problem(d, out / f"p_{conc}", slot_sources=plan["sources"], slot_source_tags=plan["tags"],
                                 sub_sources=plan["sub_sources"], specs=specs, strategy="whole", concurrency=conc)
mc = json.loads((probs["on"] / "manifest.json").read_text())
rec["concurrency"] = mc.get("concurrency")
rec["slots_from_cache"] = sum(1 for s in mc["slots"] if str(s.get("seed_source", "")).startswith("cache:"))
rec["slots"] = len(mc["slots"])
rec["graph"] = {}
for conc, p in probs.items():
    r = build_and_run(p, arch="89", runs=1, graph=True, work=p / "build_info")
    rec["graph"][conc] = {k: r.get(k) for k in ("passed", "e2e_us", "e2e_basis", "graph_nodes", "graph_skip",
                                                 "concurrency_fallback")}
    if not r.get("passed"):
        rec["graph"][conc]["log"] = (r.get("log") or "")[-1500:]
if all(v.get("passed") for v in rec["graph"].values()):
    info = {}
    t = remeasure([(probs["off"], {}), (probs["on"], {})], arch="89", reps=a.reps, info=info)
    rec["times"] = {"seq": t[0], "conc": t[1]}
    rec["median_us"] = {"seq": _median(t[0]), "conc": _median(t[1])}
    rec["failures"] = info.get("failures")
(out / "conc_check.json").write_text(json.dumps(rec, indent=1, default=str))
c = rec.get("concurrency") or {}
print(json.dumps({"name": a.name, "median_us": rec.get("median_us"), "max_width": c.get("max_width"),
                  "n_branch_ops": len(c.get("branches") or {}), "lanes": c.get("lanes"),
                  "graph": {k: (v.get("passed"), v.get("e2e_basis"), v.get("graph_nodes"), v.get("graph_skip"))
                            for k, v in rec["graph"].items()},
                  "cache_slots": f"{rec['slots_from_cache']}/{rec['slots']}"}, default=str))
