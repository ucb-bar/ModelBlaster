#!/usr/bin/env python3
"""E1 (X-120) node side: zero-LLM recomposition of one finished sweep problem (no LLM, no network).

  e1_node.py NAME DATA_DIR OUT_DIR [--reps 6] [--audit]

DATA_DIR = e1_prep.py's data/<NAME> (problem.py, work/rNN/winners, work/graph_hints.json, work/final/<NAME>/{problem,ir},
gimpl/ = this problem's 'global' pool from the other problems' implementation-store records).

Steps (MB_PRUNE_DEAD=0 everywhere so the program has the sweep's ops): extract IR as the farm worker does; rebuild the
sweep's kernel cache by ingesting every round's winners (+ overlay of the shipped final slots that the ingest does not
reproduce); profile the deployed model; kb_reuse.cross_apply with mode=siblings and (from the same start) mode=global;
build six whole models (shipped / siblings / global, each sequential and with --graph-concurrency on) and re-time them
interleaved (kb_proposals.remeasure: build once, run round-robin, fresh process per run, CUDA-graph replay); run every
exe once more for the per-output / per-slot correctness record.
-> OUT_DIR/e1.json
"""
import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ["MB_PRUNE_DEAD"] = "0"
ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("data")
ap.add_argument("out")
ap.add_argument("--reps", type=int, default=6)
ap.add_argument("--arch", default="89")
ap.add_argument("--audit", action="store_true", help="also a wide global pass (8 kinds, share >= 0.3%%)")
ap.add_argument("--only-fixed", action="store_true", help="skip the as-shipped reuse passes; only the write-back-fixed ones")
ap.add_argument("--nearest", action="store_true", help="only the global-nearest-K passes (nk2, nk4, nk4b) + orig")
ap.add_argument("--npool", default="npool", help="dir with index.jsonl + code/ of all problems' verified kernels")
ap.add_argument("--nk", default="nk2,nk4,nk4b", help="which nearest passes (nk<K>[b]; b = impl_pool metric)")
a = ap.parse_args()

from modelblaster.pipeline import graph_hints as gh                       # noqa: E402
from modelblaster.pipeline import kb_reuse                                # noqa: E402
from modelblaster.pipeline.kb_export import export_problem, parse_driver_output  # noqa: E402
from modelblaster.pipeline.kb_ingest import CudaCache, cache_plan, ingest  # noqa: E402
from modelblaster.pipeline.kb_plan import profile_ir                      # noqa: E402
from modelblaster.pipeline.kb_proposals import (_all_specs, _median, remeasure, spec_from_kind,  # noqa: E402
                                                with_fused_specs, write_variant_ir)
from modelblaster.pipeline.kernel_abi import parse_signature              # noqa: E402

data, out = Path(a.data).resolve(), Path(a.out).resolve()
out.mkdir(parents=True, exist_ok=True)
NAME, ARCH = a.name, a.arch
rec = {"name": NAME, "t0": time.time(), "reps": a.reps, "errors": []}


def err(stage):
    rec["errors"].append({"stage": stage, "tb": traceback.format_exc()[-2500:]})
    print("ERROR", stage, traceback.format_exc()[-1500:], flush=True)


def save():
    (out / "e1.json").write_text(json.dumps(rec, indent=1, default=str))


def sha12(s):
    return hashlib.sha1(s.encode()).hexdigest()[:12]


# ---- global pool: siblings' best (as in the flag) PLUS the similar-shape records of the other problems.
# kb_reuse.pool_candidates(mode=global) sorts everything by us-per-instance and cuts at max_cands, so records of small
# shapes from other problems crowd the problem's own winners out (41: the best sibling kernel dropped, global was
# WORSE than siblings). E1 measures global as a superset of siblings: this wrapper lives in this script only.
_pc_orig = kb_reuse.pool_candidates
_GEXTRA = 4


def _pool_superset(kind, cache, *, mode="siblings", impl_dirs=None, max_cands=6, **kw):
    if mode != "global":
        return _pc_orig(kind, cache, mode=mode, impl_dirs=impl_dirs, max_cands=max_cands, **kw)
    sib = _pc_orig(kind, cache, mode="siblings", impl_dirs=None, max_cands=max_cands, **kw)
    allc = _pc_orig(kind, cache, mode="global", impl_dirs=impl_dirs, max_cands=10 ** 6, **kw)
    have = {c["sha"] for c in sib}
    sim = {}
    for d in impl_dirs or []:
        f = Path(d) / "index.jsonl"
        if f.exists():
            for ln in f.read_text().splitlines():
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                sim[r.get("code_sha")] = r.get("_sim", 99)
    extra = [c for c in allc if c["sha"] not in have and str(c["from"]).startswith("impl:")]
    extra.sort(key=lambda c: sim.get(c["sha"], 99))
    return sib + extra[:_GEXTRA]



# ---- global-nearest-K (E1 addendum): own winners (siblings, as shipped) + per TARGET SHAPE the K nearest verified kernels
# of the same kind+signature from OTHER problems, ranked by shape distance (not cut globally by us-per-instance).
# Saved as analysis/e1_kb_reuse_nearest.patch (the same code as a kb_reuse.pool_candidates mode="nearest").
import importlib.util, math as _m                                         # noqa: E402
from modelblaster.pipeline import kb_ingest                               # noqa: E402
from modelblaster.pipeline.search_fixes import shape_distance as _dist_a  # noqa: E402

_LAST = {}
_ci_orig = kb_ingest.cache_plan


def _cp_capture(*aa, **kk):
    r = _ci_orig(*aa, **kk)
    _LAST["gb"] = r["graph"]
    return r


kb_ingest.cache_plan = _cp_capture
try:
    _sp = importlib.util.spec_from_file_location("_impl_pool", str(Path(kb_reuse.__file__).resolve().parents[1] /
                                                 "third_party/KernelBlaster/src/kernelblaster/knowledge/impl_pool.py"))
    _ip = importlib.util.module_from_spec(_sp)
    _sp.loader.exec_module(_ip)
    _dist_b = _ip.shape_distance
except Exception:                                                         # noqa: BLE001
    _dist_b = None
_NEAR = {"K": 4, "metric": "a", "cap": 12, "pool": None}
_RECCACHE = {}


def _pool_nearest(kind, cache, *, arch, specs, ir_name, mode="siblings", impl_dirs=None, max_cands=6, **kw):
    if mode != "nearest":
        return _pool_superset(kind, cache, arch=arch, specs=specs, ir_name=ir_name, mode=mode, impl_dirs=impl_dirs,
                              max_cands=max_cands, **kw)
    from modelblaster.pipeline.cuda_dtypes import cuda_signature, op_dtype
    from modelblaster.pipeline.kb_ingest import signature_sha
    from modelblaster.pipeline.kernel_abi import parse_signature
    own = _pc_orig(kind, cache, arch=arch, specs=specs, ir_name=ir_name, mode="siblings", max_cands=max_cands)
    sig = signature_sha(cuda_signature(specs[kind].signature))
    dtype = op_dtype(kind)
    gb = _LAST.get("gb") or {"ops": []}
    seen_t, targets = set(), []
    for o in gb["ops"]:
        if o["op"] == kind:
            k = json.dumps(o.get("shape", {}), sort_keys=True)
            if k not in seen_t:
                seen_t.add(k)
                targets.append(o.get("shape", {}))
    key = str(_NEAR["pool"])
    if key not in _RECCACHE:
        _RECCACHE[key] = kb_reuse._impl_records([_NEAR["pool"]])
    dist = _dist_b if (_NEAR["metric"] == "b" and _dist_b) else _dist_a
    have = {c["sha"] for c in own}
    by = {}
    for r in _RECCACHE[key]:
        if r.get("op") != kind or r.get("signature_sha") != sig or r.get("dtype", "fp32") != dtype \
                or r.get("arch") != arch or r.get("target", "cuda") != "cuda" or r.get("_from_problem") == NAME:
            continue
        sh = [x for x in (r.get("shapes") or []) if isinstance(x, dict)]
        if not sh:
            continue
        code = r["_code"].read_text()
        sha = kb_reuse._sha(code)
        if sha in have:
            continue
        d = [min(dist(t, x) for x in sh) for t in targets]
        if sha not in by or max(d) < max(by[sha]["_d"]):
            by[sha] = {"code": code, "sha": sha, "from": f"impl:{r.get('id')}:{r.get('_from_problem')}",
                       "us_per_inst": 0.0, "verified": False, "_d": d, "_set": max(d) if d else 0.0}
    cl = list(by.values())
    pick = {}
    for ti in range(len(targets)):                       # per target shape: its K nearest (distinct codes)
        for rank, c in enumerate(sorted(cl, key=lambda c: c["_d"][ti])[:_NEAR["K"]]):
            e = pick.setdefault(c["sha"], [c, rank, ti])
            if rank < e[1]:
                e[1] = rank
    extra = [e[0] for e in sorted(pick.values(), key=lambda e: (e[1], e[0]["_set"]))][:_NEAR["cap"]]
    for c in extra:
        c["from"] += f":set{c['_set']:.2f}"
    kb_reuse._NEAR_LOG.setdefault(kind, []).append({"targets": len(targets), "foreign": len(extra), "own": len(own),
                                                    "pool_size": len(cl)})
    return own + extra


kb_reuse._NEAR_LOG = {}
kb_reuse.MODES = tuple(kb_reuse.MODES) + ("nearest",)
kb_reuse.pool_candidates = _pool_nearest

# ---------------------------------------------------------------- IR, cache
ir = out / "ir"
if not (ir / ".done").exists():
    t = time.time()
    cp = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file",
                         str(data / "problem.py"), "--bench-target-mb", "0", "--quant", "fp16", "--out-dir", str(ir)],
                        capture_output=True, text=True)
    if cp.returncode:
        rec["errors"].append({"stage": "extract", "tb": (cp.stdout + cp.stderr)[-2000:]})
        save()
        sys.exit("extract failed")
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
fin = data / "work" / "final" / NAME / "problem"
fman = json.loads((fin / "manifest.json").read_text())
shipped = {s["id"]: sha12((fin / s["file"]).read_text()) for s in fman["slots"]}
rec["shipped_slots"] = len(shipped)


def export(tag, conc, policy="split"):
    plan = cache_plan(g, cache, f"sm_{ARCH}", specs)
    d = write_variant_ir(ir, plan["graph"], out / f"{tag}_ir")
    p = export_problem(d, out / tag, slot_sources=plan["sources"], slot_source_tags=plan["tags"],
                       sub_sources=plan["sub_sources"], specs=specs, strategy="whole", concurrency=conc,
                       sm_budget_policy=policy)
    m = json.loads((p / "manifest.json").read_text())
    return p, {s["id"]: sha12((p / s["file"]).read_text()) for s in m["slots"]}, m


try:
    p0, h0, m0 = export("probe_A", "off")
    bad = sorted(k for k in shipped if h0.get(k) != shipped[k])
    rec["identity"] = {"before_overlay": {"slots": len(shipped), "differ": bad, "extra": sorted(set(h0) - set(shipped))}}
    if bad:
        for s in fman["slots"]:
            if s["id"] in bad:
                spec = specs.get(s["op"]) or spec_from_kind(s["op"], specs)
                cache.add(op=s["op"], entry=parse_signature(spec.signature)[0], signature=spec.signature,
                          arch=f"sm_{ARCH}", shapes=[i["shape"] for i in s["instances"]],
                          code=(fin / s["file"]).read_text(), slot_us=0.001, speedup=None,
                          impl_id=f"final:{s['id']}", source={"by": "e1-overlay"})
        cache.save()
        p0, h0, m0 = export("probe_A2", "off")
        bad = sorted(k for k in shipped if h0.get(k) != shipped[k])
    rec["identity"]["after_overlay"] = {"differ": bad, "extra": sorted(set(h0) - set(shipped))}
    rec["as_shipped_identical"] = not bad and not (set(h0) - set(shipped))
except Exception:
    err("identity")
save()

# ---------------------------------------------------------------- profile + reuse passes
entries0 = copy.deepcopy(cache.entries)
prof = None
try:
    prof = profile_ir(ir, out / "profile", hints_doc=doc, cache=cache, arch=ARCH, specs=dict(_all_specs()), runs=1)
    rec["profile"] = {"passed": prof["passed"], "e2e_us": prof["e2e_us"],
                      "ops": [(o["name"], o["op"], round(o["us"] or 0, 1)) for o in prof["ops"]]}
except Exception:
    err("profile")
save()
gdir = [data / "gimpl"] if (data / "gimpl" / "index.jsonl").exists() else []
states = {"orig": entries0}
rec["reuse"] = {}
# kb_reuse._write_back drops a shape's incumbent when its code is not among the pool's max_cands sources while the
# kind's old entries are superseded: such shapes silently fall back to the hand seed (15_DenseNet121 65 ms vs 2.4 ms,
# 10_ResNet101 52 vs 3.2 ms, 20_MobileNetV2 2.7 vs 0.41 ms) and the model guard reverts the whole pass. The "f" passes
# add the incumbent's code (known from the deployed model) to the candidates handed to _write_back. Same patch as
# e1_kb_reuse_writeback.patch (for the owner of kb_reuse.py); applied here by wrapping the function in this process.
_wb_orig = kb_reuse._write_back


def _wb_fixed(cache_, kind, specs_, arch_, ir_name, table, pick, cands, inc_sha, inc_of_op, krep):
    have = {c["sha"] for c in cands}
    extra = []
    for k, row in table.items():
        sha = inc_sha.get(k)
        if sha and sha not in have and row["ops"] and row["ops"][0] in inc_of_op:
            extra.append({"code": inc_of_op[row["ops"][0]], "sha": sha, "from": "incumbent", "us_per_inst": 0.0,
                          "verified": False})
            have.add(sha)
    krep["incumbents_added"] = len(extra)
    return _wb_orig(cache_, kind, specs_, arch_, ir_name, table, pick, list(cands) + extra, inc_sha, inc_of_op, krep)


passes = []
if not a.only_fixed and not a.nearest:
    passes += [("sib", dict(mode="siblings"), False), ("glob", dict(mode="global", impl_dirs=gdir), False)]


def do_pass(tag, kw, fixed):
    cache.entries[:] = copy.deepcopy(entries0)
    cache.save()
    kb_reuse._write_back = _wb_fixed if fixed else _wb_orig
    t = time.time()
    try:
        r = kb_reuse.cross_apply(ir, doc, cache, arch=ARCH, specs=dict(_all_specs()), work=out / f"reuse_{tag}",
                                 profile=prof, **kw)
        r["wall_s"] = round(time.time() - t, 1)
        rec["reuse"][tag] = r
        states[tag] = copy.deepcopy(cache.entries)
    except Exception:
        err(f"reuse_{tag}")
    finally:
        kb_reuse._write_back = _wb_orig
    save()


if prof is not None and a.nearest:
    for tag in a.nk.split(","):
        _NEAR.update(K=int(tag[2:].rstrip("b")), metric="b" if tag.endswith("b") else "a",
                     cap=3 * int(tag[2:].rstrip("b")) + 2, pool=Path(a.npool).resolve())
        kb_reuse._NEAR_LOG = {}
        do_pass(tag, dict(mode="nearest"), True)
        if tag in rec["reuse"]:
            rec["reuse"][tag]["near_log"] = kb_reuse._NEAR_LOG
elif prof is not None:
    for tag, kw, fixed in passes:
        do_pass(tag, kw, fixed)
    # fixed passes: always when only-fixed, else only for a mode whose as-shipped pass was reverted by its guard
    for tag, kw in (("sib", dict(mode="siblings")), ("glob", dict(mode="global", impl_dirs=gdir))):
        r = rec["reuse"].get(tag) or {}
        if a.only_fixed or r.get("reverted") or tag not in rec["reuse"]:
            do_pass(tag + "f", kw, True)
    if a.audit:
        do_pass("widef", dict(mode="global", impl_dirs=gdir, max_kinds=8, min_share=0.003), True)

# ---------------------------------------------------------------- arms
arms = {}          # name -> (problem dir, slot hashes, manifest)
alias = {}         # name -> name it equals (same slot sources / same schedule)
for st, ents in states.items():
    cache.entries[:] = copy.deepcopy(ents)
    cache.save()
    try:
        arms[f"{st}_seq"] = export(f"{st}_seq", "off")
    except Exception:
        err(f"export {st}_seq")
        continue
    try:
        pc, hc, mc = export(f"{st}_conc", "on")
        c = mc.get("concurrency") or {}
        rec.setdefault("concurrency", {})[st] = {"max_width": c.get("max_width"), "lanes": c.get("lanes"),
                                                  "branch_ops": len(c.get("branches") or {})}
        if (c.get("max_width") or 1) > 1:
            arms[f"{st}_conc"] = (pc, hc, mc)
        else:
            alias[f"{st}_conc"] = f"{st}_seq"
    except Exception:
        err(f"export {st}_conc")
seen = {}
for k in list(arms):
    key = (k.split("_")[1], json.dumps(arms[k][1], sort_keys=True))
    if key in seen:
        alias[k] = seen[key]
    else:
        seen[key] = k
for k in list(alias):
    arms.pop(k, None)
rec["alias"] = alias
rec["arm_slots"] = {}
for k, (p, h, m) in arms.items():
    rec["arm_slots"][k] = {s["id"]: {"op": s["op"], "seed_source": str(s.get("seed_source", ""))[:60],
                                     "n": len(s.get("instances", [])), "sha": h[s["id"]]} for s in m["slots"]}
    rec.setdefault("arm_ops", {})[k] = {r["name"]: r["slot"] for r in m["ops"]}
save()

order = list(arms)
rec["order"] = order
times, info = {}, {}
t = time.time()
try:
    inf = {}
    ts = remeasure([(arms[k][0], {}) for k in order], arch=ARCH, reps=a.reps, info=inf)
    times = dict(zip(order, ts))
    info["failures"] = inf.get("failures")
except Exception:
    err("remeasure_all")
    for k in order:                                      # isolate the arm that does not build / run
        try:
            inf = {}
            times[k] = remeasure([(arms[k][0], {})], arch=ARCH, reps=a.reps, info=inf)[0]
            info.setdefault("failures", []).extend(inf.get("failures") or [])
        except Exception:
            err(f"remeasure {k}")
            times[k] = []
rec["remeasure_s"] = round(time.time() - t, 1)
rec["times"] = times
rec["median_us"] = {k: _median(v) if v else None for k, v in times.items()}
rec["failures"] = info.get("failures")
save()

# one more run per arm: the checks (outputs vs the fp32 reference, per-slot) and graph facts
rec["check"] = {}
env = {**os.environ, "MB_GRAPH": "1"}
for k in order:
    exe = Path(arms[k][0]) / ("build_remeasure%d" % order.index(k)) / "problem"
    try:
        rp = subprocess.run([str(exe)], capture_output=True, text=True, timeout=900, env=env)
        r = parse_driver_output(rp.stdout)
        rec["check"][k] = {"passed": r.get("passed"), "e2e_us": r.get("e2e_us"), "e2e_basis": r.get("e2e_basis"),
                           "graph_nodes": r.get("graph_nodes"), "graph_skip": r.get("graph_skip"),
                           "outputs": r.get("outputs"), "bad_slots": [s for s, v in (r.get("slots") or {}).items() if not v.get("ok")],
                           "n_slots_checked": len(r.get("slots") or {}), "errors": r.get("errors")[:3],
                           "deadlock": "MB_CONC_DEADLOCK" in (rp.stdout + rp.stderr)}
    except Exception:
        err(f"check {k}")
rec["t1"] = time.time()
save()
print(json.dumps({"name": NAME, "median_us": rec["median_us"], "alias": alias, "failures": rec["failures"]}, default=str))
