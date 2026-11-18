#!/usr/bin/env python3
"""Reuse-uplift study, node side (no LLM, no network). Adapted from analysis/e1_tools/e1_node.py.

  reuse_node.py NAME DATA_DIR OUT_DIR [--reps 6] [--arms ex,nk4,gbest,cold] [--npool npool]

Rebuilds the headline problem's shipped kernel cache (ingest of every round's winners + overlay of the shipped final slots),
then, from the same start, applies zero-LLM cross-problem reuse passes (kb_reuse.cross_apply: every candidate is run on every
shape of its op kind, the per-shape best is kept if it beats the incumbent by 3%, whole-model guard) with different donor pools:
  orig  : shipped (own winners; includes the sweep's own sibling reuse)
  ex    : own + EXACT-shape foreign kernels (donor verified on the op's exact shape)
  nk4   : own + the 4 NEAREST (shape distance) foreign kernels per target shape (exact ones first)
  gbest : own + the 4 globally best (speedup vs seed) foreign kernels of the kind, any shape
  cold  : NO own/shipped winners at all (empty cache -> hand seeds); only the foreign nearest-4/exact/global-best pool
Foreign = origin problem != NAME. Every arm is built as the whole model with the shipped schedule (concurrency as shipped), timed
as a CUDA graph, 6 fresh-process interleaved runs, per-slot + output correctness check from a last run.
-> OUT_DIR/ru.json
"""
import argparse, copy, hashlib, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

os.environ["MB_PRUNE_DEAD"] = "0"
ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("data")
ap.add_argument("out")
ap.add_argument("--reps", type=int, default=6)
ap.add_argument("--arch", default="89")
ap.add_argument("--npool", default="npool")
ap.add_argument("--arms", default="sibw,ex,nk4,gbest,cold")
ap.add_argument("--max-kinds", type=int, default=6)
ap.add_argument("--min-share", type=float, default=0.01)
ap.add_argument("--t-start", type=float, default=None, help="consumer start epoch: arms early/late filter donors by creation time")
a = ap.parse_args()

from modelblaster.pipeline import graph_hints as gh                       # noqa: E402
from modelblaster.pipeline import kb_reuse, kb_ingest                     # noqa: E402
from modelblaster.pipeline.kb_export import export_problem, parse_driver_output  # noqa: E402
from modelblaster.pipeline.kb_ingest import CudaCache, cache_plan, ingest  # noqa: E402
from modelblaster.pipeline.kb_plan import profile_ir                      # noqa: E402
from modelblaster.pipeline.kb_proposals import (_all_specs, _median, remeasure, spec_from_kind,  # noqa: E402
                                                with_fused_specs, write_variant_ir)
from modelblaster.pipeline.kernel_abi import parse_signature              # noqa: E402
from modelblaster.pipeline.search_fixes import shape_distance as dist     # noqa: E402

data, out = Path(a.data).resolve(), Path(a.out).resolve()
out.mkdir(parents=True, exist_ok=True)
NAME, ARCH = a.name, a.arch
rec = {"name": NAME, "t0": time.time(), "reps": a.reps, "errors": [], "arms_req": a.arms}


def err(stage):
    rec["errors"].append({"stage": stage, "tb": traceback.format_exc()[-2500:]})
    print("ERROR", stage, traceback.format_exc()[-1500:], flush=True)


def save():
    (out / "ru.json").write_text(json.dumps(rec, indent=1, default=str))


def sha12(s):
    return hashlib.sha1(s.encode()).hexdigest()[:12]


# ---- donor pools: per target shape / kind, from the other problems' verified kernels (npool = index.jsonl + code/)
_LAST = {}
_ci_orig = kb_ingest.cache_plan


def _cp_capture(*aa, **kk):
    r = _ci_orig(*aa, **kk)
    _LAST["gb"] = r["graph"]
    return r


kb_ingest.cache_plan = _cp_capture
_pc_orig = kb_reuse.pool_candidates
ARM = {"mode": None, "K": 4, "cap": 14, "pool": Path(a.npool).resolve(), "own": True}
_RECS = {}


def _pool_arm(kind, cache, *, arch, specs, ir_name, mode="siblings", impl_dirs=None, max_cands=6, **kw):
    if mode != "arm":
        return _pc_orig(kind, cache, arch=arch, specs=specs, ir_name=ir_name, mode=mode, impl_dirs=impl_dirs,
                        max_cands=max_cands, **kw)
    from modelblaster.pipeline.cuda_dtypes import cuda_signature, op_dtype
    from modelblaster.pipeline.kb_ingest import signature_sha
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
    if "recs" not in _RECS:
        _RECS["recs"] = kb_reuse._impl_records([ARM["pool"]])
    have = {c["sha"] for c in own}
    by = {}
    for r in _RECS["recs"]:
        if r.get("op") != kind or r.get("signature_sha") != sig or r.get("dtype", "fp32") != dtype \
                or r.get("arch") != arch or r.get("_from_problem") == NAME:
            continue
        sh = [x for x in (r.get("shapes") or []) if isinstance(x, dict)]
        if not sh:
            continue
        if ARM["mode"] in ("early", "late") and a.t_start and r.get("_create_ts"):
            if (r["_create_ts"] < a.t_start) != (ARM["mode"] == "early"):
                continue
        code = r["_code"].read_text()
        sha = kb_reuse._sha(code)
        if sha in have:
            continue
        d = [min(dist(t, x) for x in sh) for t in targets]
        c = by.get(sha)
        if c is None:
            by[sha] = {"code": code, "sha": sha, "from": f"impl:{r.get('id')}:{r.get('_from_problem')}", "us_per_inst": 0.0,
                       "verified": False, "_d": d, "_set": max(d) if d else 0.0, "_sp": r.get("speedup_vs_seed") or 0.0}
    cl = list(by.values())
    K, mode_ = ARM["K"], ARM["mode"]
    pick = {}
    if mode_ in ("nk", "cold", "early", "late"):                     # per target shape: its K nearest distinct kernels (exact = distance 0)
        for ti in range(len(targets)):
            for rank, c in enumerate(sorted(cl, key=lambda c: (c["_d"][ti], -c["_sp"]))[:K]):
                e = pick.setdefault(c["sha"], [c, rank, ti])
                if rank < e[1]:
                    e[1] = rank
    elif mode_ == "ex":                             # only kernels verified on exactly the target shape (distance 0)
        for ti in range(len(targets)):
            for rank, c in enumerate(sorted([c for c in cl if c["_d"][ti] == 0.0], key=lambda c: -c["_sp"])[:K]):
                e = pick.setdefault(c["sha"], [c, rank, ti])
                if rank < e[1]:
                    e[1] = rank
    elif mode_ == "gbest":                          # the K best by speedup-vs-seed of the kind, whatever the shape
        for rank, c in enumerate(sorted(cl, key=lambda c: -c["_sp"])[:K]):
            pick[c["sha"]] = [c, rank, 0]
    extra = [e[0] for e in sorted(pick.values(), key=lambda e: (e[1], e[0]["_set"]))][:ARM["cap"]]
    for c in extra:
        c["from"] += f":set{c['_set']:.2f}"
    kb_reuse._NEAR_LOG.setdefault(kind, []).append({"targets": len(targets), "foreign": len(extra), "own": len(own),
                                                    "pool_size": len(cl),
                                                    "dists": [round(min(c["_d"]) if c["_d"] else 0, 2) for c in extra][:8]})
    return own + extra


kb_reuse._NEAR_LOG = {}
kb_reuse.MODES = tuple(kb_reuse.MODES) + ("arm",)
kb_reuse.pool_candidates = _pool_arm

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
CONC = "on" if (fman.get("concurrency") or {}).get("mode") == "on" else "off"
rec["shipped_slots"], rec["shipped_conc"] = len(shipped), CONC


def export(tag, conc, policy="split"):
    plan = cache_plan(g, cache, f"sm_{ARCH}", specs)
    d = write_variant_ir(ir, plan["graph"], out / f"{tag}_ir")
    p = export_problem(d, out / tag, slot_sources=plan["sources"], slot_source_tags=plan["tags"],
                       sub_sources=plan["sub_sources"], specs=specs, strategy="whole", concurrency=conc,
                       sm_budget_policy=policy)
    m = json.loads((p / "manifest.json").read_text())
    return p, {s["id"]: sha12((p / s["file"]).read_text()) for s in m["slots"]}, m


try:
    p0, h0, m0 = export("probe_A", CONC)
    bad = sorted(k for k in shipped if h0.get(k) != shipped[k])
    rec["identity"] = {"before_overlay": {"slots": len(shipped), "differ": bad, "extra": sorted(set(h0) - set(shipped))}}
    if bad:
        for s in fman["slots"]:
            if s["id"] in bad:
                spec = specs.get(s["op"]) or spec_from_kind(s["op"], specs)
                cache.add(op=s["op"], entry=parse_signature(spec.signature)[0], signature=spec.signature,
                          arch=f"sm_{ARCH}", shapes=[i["shape"] for i in s["instances"]],
                          code=(fin / s["file"]).read_text(), slot_us=0.001, speedup=None,
                          impl_id=f"final:{s['id']}", source={"by": "ru-overlay"})
        cache.save()
        p0, h0, m0 = export("probe_A2", CONC)
        bad = sorted(k for k in shipped if h0.get(k) != shipped[k])
    rec["identity"]["after_overlay"] = {"differ": bad, "extra": sorted(set(h0) - set(shipped))}
    rec["as_shipped_identical"] = not bad and not (set(h0) - set(shipped))
except Exception:
    err("identity")
save()

# ---------------------------------------------------------------- profile + passes
entries0 = copy.deepcopy(cache.entries)
prof = None
try:
    prof = profile_ir(ir, out / "profile", hints_doc=doc, cache=cache, arch=ARCH, specs=dict(_all_specs()), runs=1)
    rec["profile"] = {"passed": prof["passed"], "e2e_us": prof["e2e_us"],
                      "ops": [(o["name"], o["op"], round(o["us"] or 0, 1)) for o in prof["ops"]]}
except Exception:
    err("profile")
save()
states = {"orig": entries0}
rec["reuse"] = {}
_wb_orig = kb_reuse._write_back


def _wb_fixed(cache_, kind, specs_, arch_, ir_name, table, pick, cands, inc_sha, inc_of_op, krep):
    have = {c["sha"] for c in cands}
    extra = []
    for k, row in table.items():
        sha = inc_sha.get(k)
        if sha and sha not in have and row["ops"] and row["ops"][0] in inc_of_op:
            extra.append({"code": inc_of_op[row["ops"][0]], "sha": sha, "from": "incumbent", "us_per_inst": 0.0, "verified": False})
            have.add(sha)
    krep["incumbents_added"] = len(extra)
    return _wb_orig(cache_, kind, specs_, arch_, ir_name, table, pick, list(cands) + extra, inc_sha, inc_of_op, krep)


def do_pass(tag, mode, base_entries, profile):
    cache.entries[:] = copy.deepcopy(base_entries)
    cache.save()
    ARM["mode"] = mode
    kb_reuse._write_back = _wb_fixed
    kb_reuse._NEAR_LOG = {}
    t = time.time()
    try:
        r = kb_reuse.cross_apply(ir, doc, cache, arch=ARCH, specs=dict(_all_specs()), work=out / f"reuse_{tag}",
                                 profile=profile, mode="arm", max_kinds=a.max_kinds, min_share=a.min_share)
        r["wall_s"] = round(time.time() - t, 1)
        r["near_log"] = kb_reuse._NEAR_LOG
        rec["reuse"][tag] = r
        states[tag] = copy.deepcopy(cache.entries)
    except Exception:
        err(f"reuse_{tag}")
    finally:
        kb_reuse._write_back = _wb_orig
    save()


arms_req = [x for x in a.arms.split(",") if x]
if prof is not None:
    for tag in arms_req:
        if tag == "ex":
            do_pass("ex", "ex", entries0, prof)
        elif tag == "nk4":
            do_pass("nk4", "nk", entries0, prof)
        elif tag == "sibw":                          # control: own siblings only, same wide kind limits (no foreign kernels)
            do_pass("sibw", "sibw", entries0, prof)
        elif tag in ("early", "late"):
            do_pass(tag, tag, entries0, prof)
        elif tag == "gbest":
            do_pass("gbest", "gbest", entries0, prof)
        elif tag == "cold":
            try:
                cache.entries[:] = []
                cache.save()
                profc = profile_ir(ir, out / "profile_cold", hints_doc=doc, cache=cache, arch=ARCH, specs=dict(_all_specs()), runs=1)
                rec["profile_cold"] = {"passed": profc["passed"], "e2e_us": profc["e2e_us"],
                                       "ops": [(o["name"], o["op"], round(o["us"] or 0, 1)) for o in profc["ops"]]}
                save()
                do_pass("cold", "cold", [], profc)
                states.setdefault("cold", [])          # no change = pure hand seeds
            except Exception:
                err("cold")
                states.setdefault("cold", [])

# ---------------------------------------------------------------- arms
arms, alias = {}, {}
for st, ents in states.items():
    cache.entries[:] = copy.deepcopy(ents)
    cache.save()
    try:
        arms[st] = export(st, CONC)
    except Exception:
        err(f"export {st}")
seen = {}
for k in list(arms):
    key = json.dumps(arms[k][1], sort_keys=True)
    if key in seen:
        alias[k] = seen[key]
    else:
        seen[key] = k
for k in list(alias):
    arms.pop(k, None)
rec["alias"] = alias
rec["arm_slots"] = {}
for k, (p, h, m) in arms.items():
    rec["arm_slots"][k] = {s["id"]: {"op": s["op"], "seed_source": str(s.get("seed_source", ""))[:80], "n": len(s.get("instances", [])),
                                     "sha": h[s["id"]]} for s in m["slots"]}
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
    for k in order:
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

rec["check"] = {}
env = {**os.environ, "MB_GRAPH": "1"}
for k in order:
    exe = Path(arms[k][0]) / ("build_remeasure%d" % order.index(k)) / "problem"
    try:
        rp = subprocess.run([str(exe)], capture_output=True, text=True, timeout=900, env=env)
        r = parse_driver_output(rp.stdout)
        rec["check"][k] = {"passed": r.get("passed"), "e2e_us": r.get("e2e_us"), "e2e_basis": r.get("e2e_basis"),
                           "graph_nodes": r.get("graph_nodes"), "outputs": r.get("outputs"),
                           "bad_slots": [s for s, v in (r.get("slots") or {}).items() if not v.get("ok")],
                           "n_slots_checked": len(r.get("slots") or {}), "errors": (r.get("errors") or [])[:3]}
    except Exception:
        err(f"check {k}")
rec["t1"] = time.time()
save()
print(json.dumps({"name": NAME, "median_us": rec["median_us"], "alias": alias, "failures": rec["failures"]}, default=str))
