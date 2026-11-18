#!/usr/bin/env python3
"""Warm-start a flow work dir from a finished sweep problem (node side), so a flow run spends LLM calls only on
graph decisions (the new fusion kinds), not on re-optimising every kernel.

  warm.py NAME DATA_DIR IR_DIR WORK_DIR [--reset-hint ID ...]

* cache/: every round's winners ingested in order (the sweep's cache), then the shipped final model's slot sources,
  each guarded on its instances' shapes (so the warm model deploys exactly what the sweep shipped);
* graph_hints.json: the sweep's hints; --reset-hint moves a decided hint's evidence to ``evidence_sweep`` and makes it
  ``proposed`` again (e.g. 11_VGG16's conv+ReLU, rejected in the sweep because the fused slot lost its per-shape
  conv kernels: re-decided with per-shape inheritance);
* state.json: every op instance's plan key (exact and canonical) marked sent: the plan is empty, the rounds only
  propose, cross-apply (--reuse-winners) and decide.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("data")
ap.add_argument("ir")
ap.add_argument("work")
ap.add_argument("--reset-hint", action="append", default=[])
ap.add_argument("--arch", default="89")
a = ap.parse_args()

from modelblaster.pipeline import graph_hints as gh                                    # noqa: E402
from modelblaster.pipeline.kb_ingest import CudaCache, ingest, signature_sha            # noqa: E402
from modelblaster.pipeline.kb_plan import CANON_FLAGS, canon_fields, canon_shape, item_key  # noqa: E402
from modelblaster.pipeline.kb_proposals import _all_specs, spec_from_kind               # noqa: E402
from modelblaster.pipeline.kernel_abi import parse_signature                            # noqa: E402

data, ir, work = Path(a.data), Path(a.ir), Path(a.work)
if not (ir / ".done").exists():
    cp = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file",
                         str(data / "problem.py"), "--bench-target-mb", "0", "--quant", "fp16", "--out-dir", str(ir)],
                        capture_output=True, text=True)
    if cp.returncode:
        sys.exit("extract failed: " + (cp.stdout + cp.stderr)[-2000:])
    (ir / ".done").write_text("1")
if work.exists():
    shutil.rmtree(work)
work.mkdir(parents=True)
specs = dict(_all_specs())
cache = CudaCache(work / "cache")
rep = {}
for w in sorted((data / "work").glob("r[0-9][0-9]/winners/winners.json")):
    rep[w.parent.parent.name] = [r["status"] for r in ingest(w, cache, specs)]
fin = data / "work" / "final" / a.name / "problem"
man = json.loads((fin / "manifest.json").read_text())
n_final = 0
for s in man["slots"]:
    spec = specs.get(s["op"]) or spec_from_kind(s["op"], specs)
    entry, _ = parse_signature(spec.signature)
    e, st = cache.add(op=s["op"], entry=entry, signature=spec.signature, arch=f"sm_{a.arch}",
                      shapes=[i["shape"] for i in s["instances"]], code=(fin / s["file"]).read_text(), slot_us=0.0,
                      speedup=None, impl_id=f"final:{s['id']}", source={"by": "sweep-final", "slot": s["id"]})
    n_final += 1
cache.save()
doc = gh.load(data / "work" / "graph_hints.json")
for h in doc["hints"]:
    if h["id"] in a.reset_hint:
        h["evidence_sweep"] = h.pop("evidence", [])
        h["evidence"] = []
        h["status"] = "proposed"
gh.save(doc, work / "graph_hints.json")
sent = set()
for s in man["slots"]:
    spec = specs.get(s["op"]) or spec_from_kind(s["op"], specs)
    entry, _ = parse_signature(spec.signature)
    sha = signature_sha(spec.signature)
    for i in s["instances"]:
        sent.add(item_key(entry, sha, i["shape"]))
        sent.add(item_key(entry, sha, canon_shape(i["shape"], canon_fields(s["op"], specs, CANON_FLAGS))))
(work / "state.json").write_text(json.dumps({"sent": sorted(sent), "rounds": []}, indent=1))
print(json.dumps({"ingest": {k: len(v) for k, v in rep.items()}, "final_slots": n_final, "entries": len(cache.entries),
                  "sent": len(sent), "hints": [(h["id"], h["status"]) for h in doc["hints"]]}))
