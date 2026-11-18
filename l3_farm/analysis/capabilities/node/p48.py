#!/usr/bin/env python3
"""A2 item 3 on the node: extract P with the precision policy, export the whole model (hand seeds), build + run the
driver (per-op isolated checks, partition outputs vs the reference walk, CUDA-graph timing). Also the plain fp16
extract's export for comparison (the sweep's failure). Usage: p48.py PROBLEM.py WORK"""
import json, sys, time, traceback
from pathlib import Path
from modelblaster.pipeline import precision_policy as pp
from modelblaster.pipeline.kb_export import export_problem, build_and_run

bench, work = Path(sys.argv[1]), Path(sys.argv[2])
work.mkdir(parents=True, exist_ok=True)
res = {}
for mode in ("auto", "off"):
    t0 = time.time()
    r = {}
    try:
        rep = pp.extract(bench, work / f"ir_{mode}", extra=["--bench-target-mb", "0"], policy=mode)
        r["policy"] = {k: rep.get(k) for k in ("applied", "reason", "detector", "check")}
        r["policy"]["n_fp32_ops"] = len(rep.get("fp32_ops") or [])
        prob = export_problem(work / f"ir_{mode}", work / f"prob_{mode}")
        man = json.loads((prob / "manifest.json").read_text())
        r["dtype"], r["slots"] = man["dtype"], [s["id"] for s in man["slots"]]
        b = build_and_run(prob, arch="89", work=work / f"build_{mode}")
        r["passed"] = b.get("passed")
        r["e2e_us"] = b.get("e2e_us") or b.get("partition_us")
        r["slots_ok"] = {k: v.get("ok") for k, v in (b.get("slots") or {}).items()}
        r["keys"] = sorted(b)
        r["log_tail"] = (b.get("log") or "")[-1500:]
    except Exception:
        r["error"] = traceback.format_exc()[-3000:]
    r["wall_s"] = round(time.time() - t0, 1)
    res[mode] = r
    print("@@P48@@" + json.dumps({mode: r}, default=str), flush=True)
(work / "p48.json").write_text(json.dumps(res, indent=1, default=str))
