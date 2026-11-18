#!/usr/bin/env python3
"""Zero-LLM validation of --reuse-winners and --graph-concurrency on a finished sweep problem (node side).

  recompose.py NAME DATA_DIR OUT_DIR [--mode siblings|global] [--gimpl DIR] [--reps 5]

DATA_DIR holds the sweep's results/<NAME> pieces: problem.py, work/rNN/winners/ (winners.json + code/),
work/graph_hints.json, work/final/<NAME>/problem/ (the shipped final model, for the as-shipped check).
Steps: extract the IR (as the farm worker does), rebuild the kernel cache by ingesting every round's winners in
order (what the sweep's cache was), profile the model as deployed, run kb_reuse.cross_apply (the pass the flow runs
after every round with --reuse-winners), then build and re-time, interleaved, the whole model
  A as shipped (sequential), B after reuse (sequential), C after reuse + concurrent branches (split budget),
  D as shipped + concurrent branches
(each run = the full driver: output check + per-op checks + CUDA-graph replay timing). -> OUT_DIR/recompose.json
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("data")
ap.add_argument("out")
ap.add_argument("--mode", default="siblings")
ap.add_argument("--gimpl", default="")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--arch", default="89")
ap.add_argument("--skip-reuse", action="store_true")
a = ap.parse_args()

from modelblaster.pipeline import graph_hints as gh                       # noqa: E402
from modelblaster.pipeline import kb_reuse                                # noqa: E402
from modelblaster.pipeline.kb_export import export_problem                # noqa: E402
from modelblaster.pipeline.kb_ingest import CudaCache, cache_plan, ingest  # noqa: E402
from modelblaster.pipeline.kb_plan import profile_ir                      # noqa: E402
from modelblaster.pipeline.kb_proposals import _all_specs, _median, remeasure, with_fused_specs, write_variant_ir  # noqa: E402

data, out = Path(a.data).resolve(), Path(a.out).resolve()
out.mkdir(parents=True, exist_ok=True)
rec = {"name": a.name, "mode": a.mode, "t0": time.time()}
ir = out / "ir"
if not (ir / ".done").exists():
    t = time.time()
    cp = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file",
                         str(data / "problem.py"), "--bench-target-mb", "0", "--quant", "fp16", "--out-dir", str(ir)],
                        capture_output=True, text=True)
    if cp.returncode:
        sys.exit("extract failed: " + (cp.stdout + cp.stderr)[-2000:])
    (ir / ".done").write_text("1")
    rec["extract_s"] = round(time.time() - t, 1)
specs = dict(_all_specs())
cache_dir = out / "cache"
shutil.rmtree(cache_dir, ignore_errors=True)
cache = CudaCache(cache_dir)
rec["ingest"] = {}
for w in sorted((data / "work").glob("r[0-9][0-9]/winners/winners.json")):
    rec["ingest"][w.parent.parent.name] = [r["status"] for r in ingest(w, cache, specs)]
hints = data / "work" / "graph_hints.json"
doc = gh.load(hints) if hints.exists() else gh.new_doc()
g0 = json.loads((ir / "graph.json").read_text())
g, added, _ = gh.apply_hints(g0, doc["hints"], specs)
specs.update(added)
specs = with_fused_specs(g, specs)


def export(tag, conc, policy="split"):
    plan = cache_plan(g, cache, f"sm_{a.arch}", specs)
    d = write_variant_ir(ir, plan["graph"], out / f"{tag}_ir")
    p = export_problem(d, out / tag, slot_sources=plan["sources"], slot_source_tags=plan["tags"],
                       sub_sources=plan["sub_sources"], specs=specs, strategy="whole", concurrency=conc,
                       sm_budget_policy=policy)
    m = json.loads((p / "manifest.json").read_text())
    return p, {s["id"]: hashlib.sha1((p / s["file"]).read_text().encode()).hexdigest()[:12] for s in m["slots"]}, m


# as shipped: same sources as the sweep's final model?
pa, sa, ma = export("A_shipped", "off")
fin = data / "work" / "final" / a.name / "problem"
if fin.exists():
    fm = json.loads((fin / "manifest.json").read_text())
    shipped = {s["id"]: hashlib.sha1((fin / s["file"]).read_text().encode()).hexdigest()[:12] for s in fm["slots"]}
    rec["as_shipped_identical"] = shipped == sa
    rec["shipped_slots"], rec["rebuilt_slots"] = shipped, sa
pd, _, md = export("D_shipped_conc", "on")
rec["concurrency_D"] = md.get("concurrency")
prof = profile_ir(ir, out / "profile", hints_doc=doc, cache=cache, arch=a.arch, specs=dict(_all_specs()), runs=1)
rec["profile"] = {"passed": prof["passed"], "e2e_us": prof["e2e_us"],
                  "ops": [(o["name"], o["op"], round(o["us"] or 0, 1)) for o in prof["ops"]]}
backup = json.loads(json.dumps(cache.entries))
if not a.skip_reuse:
    t = time.time()
    rep = kb_reuse.cross_apply(ir, doc, cache, arch=a.arch, specs=dict(_all_specs()), work=out / "reuse",
                               profile=prof, mode=a.mode, impl_dirs=[Path(a.gimpl)] if a.gimpl else [])
    rec["reuse"] = rep
    rec["reuse_s"] = round(time.time() - t, 1)
pb, sb, mb = export("B_reuse", "off")
pc, sc, mc = export("C_reuse_conc", "on")
rec["concurrency_C"] = mc.get("concurrency")
rec["slots_B"] = sb
t = time.time()
info = {}
times = remeasure([(pa, {}), (pb, {}), (pc, {}), (pd, {})], arch=a.arch, reps=a.reps, info=info)
rec["remeasure_s"] = round(time.time() - t, 1)
rec["times"] = {k: v for k, v in zip(("A_shipped", "B_reuse", "C_reuse_conc", "D_shipped_conc"), times)}
rec["median_us"] = {k: _median(v) for k, v in rec["times"].items()}
rec["failures"] = info.get("failures")
# one full driver run each: is the e2e a graph replay (capture worked), how many nodes, did it fall back?
from modelblaster.pipeline.kb_export import build_and_run                    # noqa: E402
rec["graph"] = {}
for tag, pdir in (("A_shipped", pa), ("C_reuse_conc", pc), ("D_shipped_conc", pd)):
    r = build_and_run(pdir, arch=a.arch, runs=1, graph=True, work=pdir / "build_info")
    rec["graph"][tag] = {k: r.get(k) for k in ("passed", "e2e_us", "e2e_basis", "graph_nodes", "graph_skip",
                                                "concurrency_fallback", "partition_us")}
    rec["graph"][tag]["outputs"] = r.get("outputs")
(out / "recompose.json").write_text(json.dumps(rec, indent=1, default=str))
print(json.dumps({"name": a.name, "median_us": rec["median_us"], "as_shipped_identical": rec.get("as_shipped_identical"),
                  "reuse": {k: (rec.get("reuse") or {}).get(k) for k in ("changed", "reverted", "evals")},
                  "guard": (rec.get("reuse") or {}).get("guard"), "failures": rec["failures"],
                  "graph": rec["graph"]}, indent=1, default=str))
