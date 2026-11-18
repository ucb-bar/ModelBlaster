#!/usr/bin/env python3
"""CPU-only export pre-flight of the A6 mechanisms on one SOL (--quant native) problem. usage: sweep.py NAME OUTJSON  (env: PYTHONPATH dir with `modelblaster` -> worktree; PF_IR = dir of extracted IRs)
Per problem: graph g0 (extract_graph --quant native of the problem file, real weights) and g_f (g0 + every `fuse` hint of the pilot run's graph_hints.json applied = what round 0's graph move leaves).
On each graph: hints region:all, region windows (default sizes 256, <= 3 candidates), merge_sibling_convs. For every hint: prepare_hint's two exports (orig / variant over the hint's ops, strategy hint-*) and the
plan path's export of the whole variant graph (export_problem like kb_plan.export_plan, no cache). Result per (graph, kind): ok | error text (first line)."""
import json, os, sys, time, traceback, shutil, tempfile
from pathlib import Path
name, outp = sys.argv[1], sys.argv[2]
IRS = Path(os.environ["PF_IR"]); L3 = Path("/Users/krisd/Documents/ModelBlaster-clone/l3_farm")
from modelblaster.pipeline import graph_hints as gh, region_slots as rs, region_windows as rw, merge_convs as mc
from modelblaster.pipeline.kb_export import export_problem
from modelblaster.pipeline.kb_proposals import _all_specs, with_fused_specs, write_variant_ir
tmp = Path(tempfile.mkdtemp(prefix="pf_", dir=os.environ.get("PF_TMP")))
res = {"problem": name, "cells": {}}
def short(e): return (type(e).__name__ + ": " + str(e).splitlines()[0])[:300] if str(e) else type(e).__name__
ir = IRS / name
g_raw = json.loads((ir / "graph.json").read_text())
g0 = json.loads(json.dumps(g_raw))                      # RAW names, as every graph.json the flow carries (apply_hints / profiles uniquify internally)
res["ops"] = len(g0["ops"]); res["kinds_cast"] = sum(o["op"].startswith("cast_") for o in g0["ops"])
base_specs = dict(_all_specs())
sp0 = with_fused_specs(g0, base_specs)
u = lambda g: gh.unique_names(json.loads(json.dumps(g)))
# pilot hints: the fuse hints
fuse_h = []
hp = L3 / "pilot_sol/instance/results" / name / "work/graph_hints.json"
if hp.exists():
    for h in json.loads(hp.read_text())["hints"]:
        if h["kind"] == "fuse":
            fuse_h.append({**h, "status": "accepted"})
graphs = {"g0": (g0, sp0)}
try:
    gf, _, prov = gh.apply_hints(json.loads(json.dumps(g0)), fuse_h, sp0, statuses=("accepted",)) if fuse_h else (g0, None, None)
    if fuse_h and gf is not g0:
        graphs["g_f"] = (gf, with_fused_specs(gf, sp0))
        res["fused_ops"] = sum(o["op"].startswith("fused__") for o in gf["ops"])
except Exception as e:
    res["cells"]["g_f:build"] = "ERROR " + short(e)
cfg = rw.WindowConfig(mode="auto", sizes=(256,), max_candidates=3) if hasattr(rw.WindowConfig, "sizes") else None
os.environ["MB_REGION_MAX_MEMBERS"] = str(max(256, cfg.member_cap()))
def hints_for(g, sp):
    out = {}
    try:
        names = rs.find_regions(u(g), "all", sp)
        out["region:all"] = [{"id": "region:all", "kind": "fuse_region", "pattern": "all", "status": "proposed"}] if names else []
    except Exception as e:
        out["region:all"] = "ERROR propose " + short(e)
    try:
        cands, rep = rw.propose_windows(u(g), sp, cfg)
        out["windows"] = [rw.hint_for(c, "m") for c in cands]
    except Exception as e:
        out["windows"] = "ERROR propose " + short(e)
    try:
        hs, rep = mc.propose(u(g), None, sp, None, ir="m")
        out["merge"] = hs
    except Exception as e:
        out["merge"] = "ERROR propose " + short(e)
    return out
def try_export(g0_, hint, sp, tag):
    """-> 'ok' | error. prepare_hint's orig/variant exports + the plan-style export of the whole variant graph."""
    msgs = []
    try:
        h = {**hint, "status": "proposed"}
        vg, _, prov = gh.apply_hints(json.loads(json.dumps(g0_)), [h], sp, statuses=None)
        if not prov["applied"]:
            return "skip: " + "; ".join(s["why"] for s in prov["skipped"])[:200]
        sp2 = with_fused_specs(vg, sp)
        d = tmp / tag; d.mkdir(parents=True, exist_ok=True)
        oir = write_variant_ir(ir, g0_, d / "orig_ir"); vir = write_variant_ir(ir, vg, d / "variant_ir")
        names0 = {o["name"] for o in g0_["ops"]}
        vnames = {prov["renamed"].get(n, n) for n in names0 if n in prov.get("renamed", {})} | set(prov.get("added", []))
        vidx = [i for i, o in enumerate(vg["ops"]) if o["name"] in vnames] or list(range(len(vg["ops"])))
        oidx = [i for i, o in enumerate(g0_["ops"]) if o["name"] in set(prov.get("renamed", {}))] or list(range(len(g0_["ops"])))
        for what, f in (("orig", lambda: export_problem(oir, d / "orig", op_indices=oidx, specs=sp2, name="o", strategy="hint-original")),
                        ("variant", lambda: export_problem(vir, d / "variant", op_indices=vidx, specs=sp2, name="v", strategy="hint-variant")),
                        ("plan", lambda: export_problem(vir, d / "plan", op_indices=list(range(len(vg["ops"]))), specs=sp2, name="p", strategy="plan"))):
            try:
                f()
            except Exception as e:
                msgs.append(f"{what}: {short(e)}")
        shutil.rmtree(d, ignore_errors=True)
    except Exception as e:
        msgs.append("apply: " + short(e))
    return "ok" if not msgs else "ERROR " + " | ".join(msgs)
for gk, (g, sp) in graphs.items():
    hs = hints_for(g, sp)
    for kind, lst in hs.items():
        key = f"{gk}:{kind}"
        if isinstance(lst, str):
            res["cells"][key] = lst; continue
        if not lst:
            res["cells"][key] = "none proposed"; continue
        rows = []
        for h in lst:
            t = time.time()
            rows.append(f"{h['id']}: " + try_export(g, h, sp, f"{gk}_{kind}_{len(rows)}") + f" ({time.time()-t:.0f}s)")
        res["cells"][key] = rows
Path(outp).write_text(json.dumps(res, indent=1))
shutil.rmtree(tmp, ignore_errors=True)
