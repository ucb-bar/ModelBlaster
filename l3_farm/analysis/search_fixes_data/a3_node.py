#!/usr/bin/env python3
"""A3 search fixes, node side (zero LLM): rebuild a sweep problem's final model, audit it with the final-model guard,
repair it by routing, and exercise the profile-fallback router on a GPU.

  a3_node.py audit NAME DATA OUT [--route] [--reps 5] [--arch 89]
  a3_node.py inject NAME DATA OUT --op OPNAME [--reps 3]

DATA = the sweep's pieces of results/NAME (packed by pack_data.sh): problem.py, work/rNN/winners/, work/rNN/profile/
<ir>/profile.json, work/graph_hints.json, work/final/NAME/problem/{manifest.json,slots/}.

audit:  extract the IR (as the farm worker), rebuild the kernel cache by ingesting every round's winners in order and
        replaying the sweep's profile-fallback rejections (-> the shipped final model; checked slot by slot against the
        kept final problem), profile it as deployed (per-op CUPTI times), run the final-model guard
        (search_fixes.seed_left with the sweep's round-0 seed times). --route: route every flagged op kind (zero LLM,
        search_fixes.route_ops, incumbents = the shipped kernels), then re-time shipped (A) vs repaired (B) interleaved
        (kb_proposals.remeasure, CUDA-graph replay of the whole model) and profile + guard B.
inject: the same rebuild, then a deliberately failing kernel (launch with 4096 threads per block -> invalid argument)
        is cached as the best winner for op OPNAME's shape and the model is profiled with the router on: the profile
        must reject it for this IR, route a verified kernel of another shape to the instance and pass.
-> OUT/audit.json | OUT/inject.json
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from modelblaster.pipeline import graph_hints as gh
from modelblaster.pipeline import search_fixes as sf
from modelblaster.pipeline.kb_export import export_problem
from modelblaster.pipeline.kb_ingest import CudaCache, cache_plan, ingest
from modelblaster.pipeline.kb_plan import profile_ir
from modelblaster.pipeline.kb_proposals import _all_specs, _median, remeasure, with_fused_specs, write_variant_ir

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["audit", "inject", "exact", "repair"])
ap.add_argument("name")
ap.add_argument("data")
ap.add_argument("out")
ap.add_argument("--route", action="store_true")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--arch", default="89")
ap.add_argument("--op", default="")
ap.add_argument("--share", type=float, default=0.10)
ap.add_argument("--min-speedup", type=float, default=1.25)
a = ap.parse_args()

data, out = Path(a.data).resolve(), Path(a.out).resolve()
out.mkdir(parents=True, exist_ok=True)
rec = {"name": a.name, "cmd": a.cmd, "t0": time.time()}
ir = out / "ir"
if not (ir / ".done").exists():
    cp = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file",
                         str(data / "problem.py"), "--bench-target-mb", "0", "--quant", "fp16", "--out-dir", str(ir)],
                        capture_output=True, text=True)
    if cp.returncode:
        sys.exit("extract failed: " + (cp.stdout + cp.stderr)[-2000:])
    (ir / ".done").write_text("1")
specs0 = dict(_all_specs())
cache_dir = out / "cache"
import shutil  # noqa: E402
shutil.rmtree(cache_dir, ignore_errors=True)
cache = CudaCache(cache_dir)
rec["ingest"] = {}
# the accepted fusions' specs too: their winners are in winners.json (ingested as "fusion-not-accepted" without them)
from modelblaster.pipeline.kb_proposals import accepted_fused_specs  # noqa: E402
hints0 = data / "work" / "graph_hints.json"
specs_i = {**specs0, **(accepted_fused_specs(hints0, specs0) if hints0.exists() else {})}
for w in sorted((data / "work").glob("r[0-9][0-9]/winners/winners.json")):
    rec["ingest"][w.parent.parent.name] = [r["status"] for r in ingest(w, cache, specs_i)]
# the sweep's profile-fallback rejections, per IR name
profs = sorted((data / "work").glob("r[0-9][0-9]/profile/*/profile.json"))
rej = []
for pf in profs:
    p = json.loads(pf.read_text())
    for step in p.get("cache_fallback") or []:
        for _, iid in (step.get("rejected") or {}).items():
            rej.append((p["ir"], iid))
for irn, iid in rej:
    for e in cache.entries:
        if e["impl_id"] == iid and irn not in e["rejected"]:
            e["rejected"].append(irn)
cache.save()
rec["replayed_rejections"] = rej
first = json.loads(profs[0].read_text()) if profs else None
seeds = sf.seed_times(first)
hints = data / "work" / "graph_hints.json"
doc = gh.load(hints) if hints.exists() else gh.new_doc()
g0 = json.loads((ir / "graph.json").read_text())
g, added, _ = gh.apply_hints(g0, doc["hints"], dict(specs0))
specs = dict(specs0)
specs.update(added)
specs = with_fused_specs(g, specs)


def export(tag):
    plan = cache_plan(g, cache, f"sm_{a.arch}", specs)
    d = write_variant_ir(ir, plan["graph"], out / f"{tag}_ir")
    p = export_problem(d, out / tag, slot_sources=plan["sources"], slot_source_tags=plan["tags"],
                       sub_sources=plan["sub_sources"], specs=specs, strategy="whole")
    m = json.loads((p / "manifest.json").read_text())
    return p, {s["id"]: hashlib.sha1((p / s["file"]).read_text().encode()).hexdigest()[:12] for s in m["slots"]}, m


def guard_of(prof):
    ops = [{"name": r["name"], "op": r["op"], "us": r["us"], "seed_source": r["seed"], "slot": r["slot"]}
           for r in prof["ops"]]
    return sf.seed_left(ops, seeds, share=a.share, min_speedup=a.min_speedup), ops


pa, sa, ma = export("A_shipped")
fin = data / "work" / "final" / a.name / "problem"
if fin.exists():
    fm = json.loads((fin / "manifest.json").read_text())
    shipped = {s["id"]: hashlib.sha1((fin / s["file"]).read_text().encode()).hexdigest()[:12] for s in fm["slots"]}
    rec["as_shipped_identical"] = shipped == sa
    rec["slot_diff"] = {k: (shipped.get(k), sa.get(k)) for k in set(shipped) | set(sa) if shipped.get(k) != sa.get(k)}

if a.cmd == "exact":
    # the shipped final model exactly: the kept final IR graph (hints and cache splits applied) + the kept slot sources
    import copy
    from modelblaster.pipeline.kb_export import build_and_run
    fg = json.loads((data / "work" / "final" / a.name / "ir" / "graph.json").read_text())
    fman = json.loads((fin / "manifest.json").read_text())
    fspecs = with_fused_specs(fg, dict(specs_i))
    fdir = write_variant_ir(ir, fg, out / "fin_ir")
    srcs = {s["id"]: (fin / s["file"]).read_text() for s in fman["slots"]}
    tags = {s["id"]: s.get("seed_source") for s in fman["slots"]}

    def exp(tag, sources):
        return export_problem(fdir, out / tag, slot_sources=sources, slot_source_tags=tags, specs=fspecs,
                              strategy="whole")

    def prof_ops(p):
        r = build_and_run(p, arch=a.arch, runs=3, graph=True, work=p / "build_prof")
        m = json.loads((p / "manifest.json").read_text())
        attr, ev = r.get("attr_op_us") or {}, r.get("op_us") or {}
        ops = [{"name": o["name"], "op": o["op"], "slot": o["slot"], "seed_source": tags.get(o["slot"]),
                "us": attr.get(o["name"], ev.get(o["name"]))} for o in m["ops"]]
        return r, ops, m

    pA = exp("A_exact", srcs)
    rA, opsA, mA = prof_ops(pA)
    rec["exact_match"] = {"slots": sorted(s["id"] for s in mA["slots"]) == sorted(srcs),
                          "ops": [o["name"] for o in mA["ops"]] == [o["name"] for o in fman["ops"]],
                          "data_sha1_kept": fman.get("data_sha1") == mA.get("data_sha1")}
    flagged = sf.seed_left(opsA, seeds, share=a.share, min_speedup=a.min_speedup)
    rec["A"] = {"passed": rA.get("passed"), "e2e_us": rA.get("e2e_us"), "flagged": flagged, "ops": opsA}
    if a.route and flagged:
        fgu = gh.unique_names(copy.deepcopy(fg))
        shape_of = {o["name"]: o.get("shape", {}) for o in fgu["ops"]}
        inc = {o["name"]: srcs[o["slot"]] for o in mA["ops"]}
        by_kind = {}
        for f in flagged:
            if f["op"] in fspecs and not f["op"].startswith("fused__"):
                by_kind.setdefault(f["op"], []).append(f["name"])
        rec["routes"], override, notes = [], {}, []
        t = time.time()
        for kind, names in by_kind.items():
            r = sf.route_ops(kind=kind, ops=names, graph=fgu, src_ir=fdir, cache=cache, specs=fspecs, arch=a.arch,
                             ir_name=fman["ir_name"], work=out / f"route_{kind}", incumbent=inc, tag=sf.REPAIR_TAG,
                             why={"final_guard": names})
            rec["routes"].append(r)
            for w in r.get("written") or []:
                e = next(x for x in cache.entries if x["impl_id"] == w["impl_id"])
                guard = set(e["shape_guard"])
                for n in names:
                    if json.dumps(shape_of[n], sort_keys=True) not in guard:
                        continue
                    sid = next(o["slot"] for o in mA["ops"] if o["name"] == n)
                    mates = [o["name"] for o in mA["ops"] if o["slot"] == sid]
                    if all(json.dumps(shape_of[m], sort_keys=True) in guard for m in mates):
                        override[sid] = cache.code(e)
                    else:
                        notes.append(f"{n}: slot {sid} also serves {mates}; not overridden")
        rec["route_s"] = round(time.time() - t, 1)
        rec["override_slots"], rec["override_notes"] = sorted(override), notes
        pB = exp("B_routed", {**srcs, **override})
        info = {}
        times = remeasure([(pA, {}), (pB, {})], arch=a.arch, reps=a.reps, info=info)
        rec["times"] = {"A_shipped": times[0], "B_routed": times[1]}
        rec["median_us"] = {k: _median(v) for k, v in rec["times"].items()}
        rec["failures"] = info.get("failures")
        rB, opsB, _ = prof_ops(pB)
        tags.update({k: "route" for k in override})
        rec["B"] = {"passed": rB.get("passed"), "e2e_us": rB.get("e2e_us"),
                    "flagged": sf.seed_left([{**o, "seed_source": "route" if o["slot"] in override else o["seed_source"]}
                                             for o in opsB], seeds, share=a.share, min_speedup=a.min_speedup),
                    "ops": opsB}
    (out / "exact.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({"name": a.name, "exact_match": rec["exact_match"], "A_e2e": rec["A"]["e2e_us"],
                      "A_passed": rec["A"]["passed"],
                      "flagged": [(f["name"], f["op"], f["share"], f["speedup_vs_seed"], f["on_seed_kernel"])
                                  for f in flagged],
                      "routes": [(x.get("kind"), x.get("result"), x.get("written")) for x in rec.get("routes", [])],
                      "median_us": rec.get("median_us"),
                      "B_flagged": [f["name"] for f in (rec.get("B") or {}).get("flagged", [])]}, indent=1, default=str))
elif a.cmd == "repair":
    # the flow's own end-of-run path: verify_ir (final) -> Fixes.final_guard(mode repair) -> route -> verify again
    from modelblaster.pipeline.kb_ingest import verify_ir
    work = out / "work"
    pdir = work / "r00" / "profile" / ir.name
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "profile.json").write_text(profs[0].read_text())          # the sweep's round-0 profile (seed times)
    hp = out / "graph_hints.json"
    gh.save(doc, hp)

    class Cfg:
        arch, final_guard, fallback_repair = a.arch, "repair", "on"
    fx = sf.Fixes(Cfg(), work)

    def reverify(d):
        return verify_ir(Path(d), cache, arch=a.arch, specs=dict(specs_i), hints=hp, work=work / "final" / Path(d).name)
    t = time.time()
    final = {ir.name: reverify(ir)}
    rec["verify_before"] = {k: final[ir.name].get(k) for k in ("passed", "partition_us", "e2e_us")}
    guard = fx.final_guard([ir], final, {"sent": []}, cache, dict(specs_i), reverify=reverify, hints_doc=doc)
    rec["seconds"] = round(time.time() - t, 1)
    rec["guard"] = guard
    rec["verify_after"] = {k: final[ir.name].get(k) for k in ("passed", "partition_us", "e2e_us")}
    (out / "repair.json").write_text(json.dumps(rec, indent=1, default=str))
    g1 = guard["irs"][next(iter(guard["irs"]))]
    print(json.dumps({"before": rec["verify_before"], "after": rec["verify_after"],
                      "flagged": [(f["name"], f["share"], f["on_seed_kernel"], f["reason"]) for f in g1["flagged"]],
                      "repair": {k: v for k, v in (g1.get("repair") or {}).items() if k != "routes"},
                      "routes": [(x.get("kind"), x.get("result"), x.get("written")) for x in
                                 (g1.get("repair") or {}).get("routes", [])], "seconds": rec["seconds"]},
                     indent=1, default=str))
elif a.cmd == "audit":
    prof = profile_ir(ir, out / "profA", hints_doc=doc, cache=cache, arch=a.arch, specs=dict(specs0), runs=3)
    flagged, ops = guard_of(prof)
    rec["A"] = {"passed": prof["passed"], "e2e_us": prof["e2e_us"], "cache_fallback": prof.get("cache_fallback"),
                "flagged": flagged, "ops": ops}
    if a.route and flagged:
        bman = json.loads((pa / "manifest.json").read_text())
        src_of_slot = {s["id"]: (pa / s["file"]).read_text() for s in bman["slots"]}
        inc = {r["name"]: src_of_slot[r["slot"]] for r in bman["ops"]}
        plan = cache_plan(g, cache, f"sm_{a.arch}", specs)
        by_kind = {}
        for f in flagged:
            if f["op"] in specs and not f["op"].startswith("fused__"):
                by_kind.setdefault(f["op"], []).append(f["name"])
        rec["routes"] = []
        t = time.time()
        for kind, names in by_kind.items():
            r = sf.route_ops(kind=kind, ops=names, graph=plan["graph"], src_ir=ir, cache=cache, specs=specs,
                             arch=a.arch, ir_name=bman["ir_name"], work=out / f"route_{kind}", incumbent=inc,
                             tag=sf.REPAIR_TAG, why={"final_guard": names})
            rec["routes"].append(r)
        rec["route_s"] = round(time.time() - t, 1)
        pb, sb, mb = export("B_routed")
        rec["slots_changed"] = {k: (sa.get(k), v) for k, v in sb.items() if sa.get(k) != v}
        info = {}
        times = remeasure([(pa, {}), (pb, {})], arch=a.arch, reps=a.reps, info=info)
        rec["times"] = {"A_shipped": times[0], "B_routed": times[1]}
        rec["median_us"] = {k: _median(v) for k, v in rec["times"].items()}
        rec["failures"] = info.get("failures")
        profb = profile_ir(ir, out / "profB", hints_doc=doc, cache=cache, arch=a.arch, specs=dict(specs0), runs=3)
        fb, opsb = guard_of(profb)
        rec["B"] = {"passed": profb["passed"], "e2e_us": profb["e2e_us"], "flagged": fb, "ops": opsb,
                    "cache_fallback": profb.get("cache_fallback")}
    (out / "audit.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({"name": a.name, "identical": rec.get("as_shipped_identical"), "A_e2e": rec["A"]["e2e_us"],
                      "flagged": [(f["name"], f["op"], f["share"], f["speedup_vs_seed"], f["on_seed_kernel"])
                                  for f in flagged],
                      "median_us": rec.get("median_us"), "B_flagged": [f["name"] for f in rec.get("B", {}).get(
                          "flagged", [])]}, indent=1, default=str))
else:
    from modelblaster.pipeline.cuda_dtypes import cuda_signature
    from modelblaster.pipeline.kernel_abi import parse_signature
    op = next(o for o in gh.unique_names(json.loads(json.dumps(g)))["ops"] if o["name"] == a.op)
    kind = op["op"]
    sig = cuda_signature(specs[kind].signature)
    entry, params = parse_signature(specs[kind].signature)
    bad = ("#include <cuda_fp16.h>\n#include <cuda_runtime.h>\n__global__ void mb_bad_k(int x) {}\n"
           f"{sig} {{\n  // deliberately invalid launch: 4096 threads per block\n  mb_bad_k<<<1, 4096>>>(0);\n}}\n")
    e, st = cache.add(op=kind, entry=entry, signature=specs[kind].signature, arch=f"sm_{a.arch}",
                      shapes=[op.get("shape", {})], code=bad, slot_us=0.001, speedup=1000.0, impl_id="inject:bad",
                      source={"by": "a3-inject"})
    cache.save()
    rec["injected"] = {"op": a.op, "kind": kind, "status": st}
    log = []
    t = time.time()
    prof = profile_ir(ir, out / "profI", hints_doc=doc, cache=cache, arch=a.arch, specs=dict(specs0), runs=1,
                      on_fallback=sf.fallback_router(enabled=True, log=log))
    rec["seconds"] = round(time.time() - t, 1)
    rec["passed"] = prof["passed"]
    rec["e2e_us"] = prof["e2e_us"]
    rec["cache_fallback"] = prof.get("cache_fallback")
    rec["routes"] = log
    rec["op_after"] = next((r for r in prof["ops"] if r["name"] == a.op), None)
    rec["injected_rejected_for"] = next(x["rejected"] for x in cache.entries if x["impl_id"] == "inject:bad")
    (out / "inject.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({k: rec.get(k) for k in ("injected", "passed", "e2e_us", "op_after", "injected_rejected_for",
                                              "seconds")}, indent=1, default=str))
    print(json.dumps([{k: x.get(k) for k in ("attempt", "kind", "ops", "result", "written", "pool")} for x in log],
                     indent=1, default=str))
