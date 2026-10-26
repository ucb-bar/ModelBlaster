"""Winner reuse across split variants, shapes and problems (``--reuse-winners off|siblings|global``).

The Level 3 sweep optimised every op x shape as its own KernelBlaster problem and never tried one problem's
winner on another's instances: 41_GRUBidirectional ran its reverse and layer-0 GRU ops on kernels 1.8x slower than
the forward winner, which is valid for them (``reverse`` is a runtime flag), 9,380 -> 6,703 us at zero LLM cost
(l3_farm/analysis/cuda_vs_flow.md). The shape-family pilot found the same for convs: routing a small pool of
verified kernels per shape beat the per-shape set (l3_farm/analysis/shape_family.md). This module is that routing
step for the per-shape flow, with no LLM call:

**cross_apply** -- for each op kind with a measurable share of the model and more than one verified source:

1. *pool*: the kind's verified sources -- this work dir's cache entries (``siblings``) and, with ``global``, the
   implementation store's records (``knowledge/impl``: this run's verified candidates and the global store's
   winners materialised into it by ``--knowledge-global``), one per code, best first, at most ``max_cands``;
2. *incumbent*: what the model deploys today for every instance (the exported whole-model slot of that op);
3. *matrix*: one problem holding every instance of the kind, one slot per distinct shape (``slot_variant`` rwK),
   built once per candidate with that candidate in every shape slot (round 0: the incumbents). The driver checks
   every slot on its own instances against the reference (``MB_SLOT_CHECK``: the guard "correct on that shape")
   and attributes the GPU time per slot (CUPTI). A round where some shape failed is re-timed with the incumbent
   in the failing slots (the driver only times a passing partition);
4. *choice*: per shape the fastest passing source, if it beats the incumbent by ``min_gain``; ties keep the
   incumbent, so no shape is slowed by construction (the "do not slow dominant shapes" guard);
5. *write-back*: one cache entry (``reuse``) per chosen source, guarded on exactly the shapes it was chosen and
   validated for, provenance in ``source``; the kind's earlier entries are ``superseded`` for this IR, so
   ``cache_plan`` deploys the routing table as is (other IRs are unaffected);
6. *model guard*: the whole model before and after, built and re-timed back to back (``remeasure``); a failing
   or slower (> ``guard_pct``) model reverts the write-back. Everything is recorded in the report.

**pool_candidates** is also what the seeding path uses (``seed_candidates``): a new slot problem gets the pool's
best sources as validated trial seeds (KernelBlaster tries them at start-up and keeps one only if it passes and
is faster).
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

MODES = ("off", "siblings", "global")
RW = "rw"                     # slot_variant prefix of the matrix problem's per-shape slots


def _sha(code: str) -> str:
    return hashlib.sha1(code.encode()).hexdigest()


def _skey(shape: dict) -> str:
    return json.dumps(shape, sort_keys=True)


# ---------------------------------------------------------------- the pool

def _impl_records(dirs: list[Path]) -> list[dict]:
    out = []
    for d in dirs:
        f = Path(d) / "index.jsonl"
        if not f.exists():
            continue
        for ln in f.read_text().splitlines():
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            c = Path(d) / "code" / f"{r.get('code_sha')}.cu"
            if c.exists():
                out.append({**r, "_code": c})
    return out


def pool_candidates(kind: str, cache, *, arch: str, specs: dict, ir_name: str | None, mode: str = "siblings",
                    impl_dirs: list[Path] | None = None, max_cands: int = 6) -> list[dict]:
    """The verified sources of ``kind`` (one per code): [{"code", "sha", "from", "us_per_inst", "verified"}],
    best first (model-verified on this IR, then the time per instance it was measured at)."""
    from .cuda_dtypes import cuda_signature, op_dtype
    from .kb_ingest import signature_sha, superseded
    from .kernel_abi import parse_signature

    if mode == "off" or kind not in specs:
        return []
    entry, _ = parse_signature(specs[kind].signature)
    sig = signature_sha(cuda_signature(specs[kind].signature))
    dtype = op_dtype(kind)
    by: dict[str, dict] = {}
    for e in cache.entries:
        if (e["entry"], e["signature_sha"], e["arch"], e.get("dtype", "fp32")) != (entry, sig, arch, dtype):
            continue
        if ir_name is not None and (ir_name in e["rejected"] or superseded(e, ir_name)):
            continue
        code = cache.code(e)
        c = {"code": code, "sha": _sha(code), "from": f"cache:{e['impl_id']}",
             "us_per_inst": (e.get("slot_us") or 0.0) / max(len(e["shape_guard"]), 1),
             "verified": ir_name in e.get("model_verified", [])}
        if c["sha"] not in by or (c["verified"], -c["us_per_inst"]) > (by[c["sha"]]["verified"], -by[c["sha"]]["us_per_inst"]):
            by[c["sha"]] = c
    if mode == "global":
        for r in _impl_records(impl_dirs or []):
            if r.get("op") != kind or r.get("signature_sha") != sig or r.get("dtype", "fp32") != dtype \
                    or r.get("arch") != arch or r.get("target", "cuda") != "cuda":
                continue
            if (r.get("speedup_vs_seed") or 0.0) < 1.02:
                continue
            code = r["_code"].read_text()
            c = {"code": code, "sha": _sha(code), "from": f"impl:{r.get('id')}:{r.get('problem_id')}",
                 "us_per_inst": (r.get("slot_us") or 0.0) / max(len(r.get("shapes") or [1]), 1), "verified": False}
            if c["sha"] not in by or c["us_per_inst"] < by[c["sha"]]["us_per_inst"]:
                by.setdefault(c["sha"], c)
    cands = sorted(by.values(), key=lambda c: (not c["verified"], c["us_per_inst"] or float("inf")))
    return cands[:max(0, int(max_cands))]


# ---------------------------------------------------------------- running problems

class Runner:
    """Build + run exported problems with per-run source overrides: ModelBlaster's builtin driver build here,
    or KernelBlaster's evaluator (``kbeval``) -> parse_driver_output-like dicts."""

    def __init__(self, *, arch: str, kbeval=None, gpu_model: str | None = None, work: Path | None = None):
        self.arch, self.kbeval, self.gpu_model, self.work = str(arch).removeprefix("sm_"), kbeval, gpu_model, work
        self.runs = 0

    def run(self, jobs: list[tuple[str, Path, dict]], *, name: str = "reuse", env: dict | None = None) -> dict:
        """jobs: (id, problem dir, {slot id: source}) -> {id: result}."""
        self.runs += len(jobs)
        if self.kbeval is not None:
            from .kb_eval import driver_result
            js = [{"id": i, "problem": str(p), "sources": src, "reps": 1} for i, p, src in jobs]
            kr = self.kbeval.run(js, arch=self.arch, work=Path(self.work or jobs[0][1].parent) / "kb_time",
                                 gpu_model=self.gpu_model, name=name, driver_env=env)
            out = {}
            for i, _, _ in jobs:
                rows = (kr["jobs"].get(i) or {}).get("runs") or []
                out[i] = driver_result(rows[0]) if rows else {"passed": False, "slots": {}, "log": "no run"}
            return out
        from .kb_export import build_and_run
        return {i: build_and_run(p, arch=self.arch, sources=src, work=Path(p) / f"build_{i}", runs=1, graph=True,
                                 env=env)
                for i, p, src in jobs}


def _slot_time(res: dict, sid: str) -> float | None:
    v = (res.get("attr_slot_us") or {}).get(sid)
    if v is None:
        v = (res.get("slot_us") or {}).get(sid)
    return v


# ---------------------------------------------------------------- one op kind

def matrix_graph(graph: dict, ops: list[str]) -> tuple[dict, dict[str, int]]:
    """``graph`` with ``ops`` given one slot per distinct shape (slot_variant rw<k>) -> (graph, {op: k})."""
    g = copy.deepcopy(graph)
    keys: dict[str, int] = {}
    where = {}
    for o in g["ops"]:
        if o["name"] in ops:
            k = keys.setdefault(_skey(o.get("shape", {})), len(keys))
            o["slot_variant"] = f"{RW}{k}"
            o.pop("hint", None)
            where[o["name"]] = k
    return g, where


def measure_kind(kind: str, graph: dict, src_ir: Path, ops: list[str], incumbent: dict[str, str], cands: list[dict],
                 *, specs: dict, runner: Runner, work: Path) -> dict:
    """Every candidate on every distinct shape of ``ops`` (instances of ``kind``) -> per shape
    {"incumbent": {"us", "ok"}, "cands": {sha: {"us", "ok"}}, "n": instances}; ``incumbent`` maps op -> source."""
    from .kb_export import export_problem, slot_id_of
    from .kb_proposals import write_variant_ir

    g, where = matrix_graph(graph, ops)
    nshape = 1 + max(where.values())
    work.mkdir(parents=True, exist_ok=True)
    vir = write_variant_ir(src_ir, g, work / "ir")
    idx = [i for i, o in enumerate(g["ops"]) if o["name"] in where]
    prob = export_problem(vir, work / "problem", op_indices=idx, specs=specs, strategy="reuse-matrix",
                          name=f"reuse__{kind}", concurrency="off")
    sid_of = {}
    for o in g["ops"]:
        if o["name"] in where:
            sid_of[where[o["name"]]] = slot_id_of(o, specs)
    n_of = {k: sum(1 for v in where.values() if v == k) for k in range(nshape)}
    inc_src = {}
    for name, k in where.items():
        inc_src.setdefault(k, incumbent.get(name))
    rounds = [("inc", {sid_of[k]: inc_src[k] for k in range(nshape) if inc_src.get(k)})]
    for c in cands:
        rounds.append((c["sha"][:12], {sid_of[k]: c["code"] for k in range(nshape)}))
    res = runner.run([(f"m_{t}", prob, src) for t, src in rounds], name=f"reuse_{kind}")
    table = {k: {"n": n_of[k], "slot": sid_of[k], "incumbent": None, "cands": {}} for k in range(nshape)}
    retime = []
    for t, src in rounds:
        r = res[f"m_{t}"]
        for k in range(nshape):
            sc = (r.get("slots") or {}).get(sid_of[k]) or {}
            ok = bool(sc.get("ok"))
            us = _slot_time(r, sid_of[k]) if r.get("passed") else None
            rec = {"ok": ok, "us": (us / n_of[k]) if us is not None else None, "rel_err": sc.get("rel_err")}
            if t == "inc":
                table[k]["incumbent"] = rec
            else:
                table[k]["cands"][t] = rec
        if not r.get("passed") and t != "inc":
            good = [k for k in range(nshape) if table[k]["cands"][t]["ok"]]
            if good:
                # the driver times only a passing partition: re-time with the incumbent where this one failed
                mix = {sid_of[k]: (src[sid_of[k]] if k in good else inc_src.get(k)) for k in range(nshape)}
                retime.append((t, good, {s: c for s, c in mix.items() if c}))
    if retime:
        res2 = runner.run([(f"t_{t}", prob, src) for t, _, src in retime], name=f"reuse_{kind}_retime")
        for t, good, _ in retime:
            r = res2[f"t_{t}"]
            for k in good:
                sc = (r.get("slots") or {}).get(sid_of[k]) or {}
                us = _slot_time(r, sid_of[k]) if r.get("passed") else None
                table[k]["cands"][t].update(ok=bool(sc.get("ok")) and r.get("passed"),
                                            us=(us / n_of[k]) if us is not None else None, retimed=True)
    shapes = {}
    for o in g["ops"]:
        if o["name"] in where:
            shapes.setdefault(where[o["name"]], o.get("shape", {}))
    for k in table:
        table[k]["shape"] = shapes[k]
        table[k]["ops"] = sorted(n for n, v in where.items() if v == k)
    return {"problem": str(prob), "table": table, "rounds": [t for t, _ in rounds]}


def choose(table: dict, cands: list[dict], incumbent_sha: dict[int, str | None], min_gain: float) -> dict:
    """Per shape: the fastest passing candidate if it beats the incumbent by ``min_gain`` -> {k: sha12 | None}."""
    out = {}
    for k, row in table.items():
        inc = row.get("incumbent") or {}
        best, best_us = None, None
        for t, rec in row["cands"].items():
            if rec.get("ok") and rec.get("us") is not None and (best_us is None or rec["us"] < best_us):
                best, best_us = t, rec["us"]
        inc_us = inc.get("us") if inc.get("ok") else None
        if best is not None and best != (incumbent_sha.get(k) or "")[:12] and \
                (inc_us is None or best_us < inc_us * (1.0 - min_gain)):
            out[k] = best
        else:
            out[k] = None
    return out


# ---------------------------------------------------------------- the pass

def cross_apply(source_ir: Path, hints_doc: dict, cache, *, arch: str, specs: dict, work: Path, profile: dict,
                mode: str = "siblings", impl_dirs: list[Path] | None = None, kbeval=None,
                gpu_model: str | None = None, min_share: float = 0.01, max_kinds: int = 4, max_cands: int = 6,
                min_gain: float = 0.03, guard_pct: float = 0.01, guard_reps: int = 3, runner: Runner | None = None,
                remeasure_fn=None) -> dict:
    """The zero-LLM cross-application pass for one IR (see the module doc) -> report (also ``work/reuse.json``)."""
    from . import graph_hints as gh
    from .kb_export import ALIAS_OPS, export_problem
    from .kb_ingest import cache_plan
    from .kb_plan import hinted_ir
    from .kb_proposals import with_fused_specs, write_variant_ir

    work = Path(work)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    rep = {"mode": mode, "ir": profile.get("ir"), "kinds": [], "changed": False}
    if mode not in MODES or mode == "off":
        return rep
    arch_l = f"sm_{str(arch).removeprefix('sm_')}"
    runner = runner or Runner(arch=arch, kbeval=kbeval, gpu_model=gpu_model, work=work)
    hdir, specs, g = hinted_ir(Path(source_ir), hints_doc or gh.new_doc(), work / "hinted", dict(specs))
    specs = with_fused_specs(g, specs)
    base = cache_plan(g, cache, arch_l, specs)
    gb = base["graph"]
    ir_name = gb.get("name") or profile.get("ir")
    rep["ir"] = ir_name
    total = profile.get("e2e_us") or sum((o.get("us") or 0.0) for o in profile.get("ops", [])) or 1.0
    by_kind: dict[str, float] = {}
    for o in profile.get("ops", []):
        by_kind[o["op"]] = by_kind.get(o["op"], 0.0) + (o.get("us") or 0.0)
    kinds = [k for k, us in sorted(by_kind.items(), key=lambda kv: -kv[1])
             if us / total >= min_share and k not in ALIAS_OPS][:max_kinds]
    # the incumbent of every op: the deployed whole model's slot sources
    bdir = write_variant_ir(Path(source_ir), gb, work / "base_ir")
    bprob = export_problem(bdir, work / "base_problem", slot_sources=base["sources"], slot_source_tags=base["tags"],
                           sub_sources=base["sub_sources"], specs=specs, strategy="whole", concurrency="off")
    bman = json.loads((bprob / "manifest.json").read_text())
    src_of_slot = {s["id"]: (bprob / s["file"]).read_text() for s in bman["slots"]}
    inc_of_op = {r["name"]: src_of_slot[r["slot"]] for r in bman["ops"]}
    backup = json.loads(json.dumps(cache.entries))
    for kind in kinds:
        ops = [o["name"] for o in gb["ops"] if o["op"] == kind]
        cands = pool_candidates(kind, cache, arch=arch_l, specs=specs, ir_name=ir_name, mode=mode,
                                impl_dirs=impl_dirs, max_cands=max_cands)
        inc_shas = {_sha(inc_of_op[n]) for n in ops if n in inc_of_op}
        krep = {"kind": kind, "share": round(by_kind[kind] / total, 4), "ops": len(ops),
                "pool": [{"sha": c["sha"][:12], "from": c["from"]} for c in cands]}
        rep["kinds"].append(krep)
        if not ops or not cands or {c["sha"] for c in cands} <= inc_shas and len(inc_shas) <= 1:
            krep["skipped"] = "nothing to try (one source, already deployed everywhere)"
            continue
        try:
            m = measure_kind(kind, gb, Path(source_ir), ops, inc_of_op, cands, specs=specs, runner=runner,
                             work=work / f"m_{kind}")
        except Exception as e:  # noqa: BLE001 -- a failed measurement leaves the kind as it is
            krep["error"] = f"{type(e).__name__}: {e}"[:800]
            continue
        table = m["table"]
        inc_sha = {k: _sha(inc_of_op[row["ops"][0]]) if row["ops"][0] in inc_of_op else None
                   for k, row in table.items()}
        pick = choose(table, cands, inc_sha, min_gain)
        krep["table"] = {str(k): {kk: v for kk, v in row.items() if kk != "slot"} for k, row in table.items()}
        krep["choice"] = {str(k): v for k, v in pick.items()}
        if not any(pick.values()):
            krep["result"] = "incumbent kept on every shape"
            continue
        _write_back(cache, kind, specs, arch_l, ir_name, table, pick, cands, inc_sha, inc_of_op, krep)
        rep["changed"] = True
    if rep["changed"]:
        cache.save()
        rep["guard"] = _model_guard(Path(source_ir), g, cache, base, arch_l, specs, work, runner, guard_pct,
                                    guard_reps, remeasure_fn)
        if not rep["guard"].get("ok"):
            cache.entries[:] = backup
            cache.save()
            rep["reverted"] = rep["guard"].get("why")
            rep["changed"] = False
    rep["evals"] = runner.runs
    (work / "reuse.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep


def _write_back(cache, kind, specs, arch, ir_name, table, pick, cands, inc_sha, inc_of_op, krep) -> None:
    from .kernel_abi import parse_signature

    entry, _ = parse_signature(specs[kind].signature)
    by_sha = {c["sha"][:12]: c for c in cands}
    groups: dict[str, dict] = {}            # code sha -> {"code", "shapes", "us", "from"}
    for k, row in table.items():
        t = pick.get(k)
        if t is not None:
            c = by_sha[t]
            g = groups.setdefault(c["sha"], {"code": c["code"], "shapes": [], "us": 0.0, "from": c["from"],
                                              "per_shape": {}})
            us = row["cands"][t]["us"]
        else:
            # the incumbent stays: kept as a reuse entry only when it is a cached kernel (a hand / fused seed has none)
            sha = inc_sha.get(k)
            c = next((x for x in cands if x["sha"] == sha), None)
            if c is None:
                continue
            g = groups.setdefault(sha, {"code": c["code"], "shapes": [], "us": 0.0, "from": c["from"], "per_shape": {}})
            us = (row.get("incumbent") or {}).get("us")
        g["shapes"].append(row["shape"])
        g["us"] += (us or 0.0) * row["n"]
        g["per_shape"][_skey(row["shape"])] = us
    from .cuda_dtypes import cuda_signature
    from .kb_ingest import signature_sha
    sig = signature_sha(cuda_signature(specs[kind].signature))
    old = [e for e in cache.entries if e["entry"] == entry and e["signature_sha"] == sig and e["arch"] == arch
           and ir_name not in e["rejected"]]
    for e in old:
        sup = e.setdefault("superseded", [])
        if ir_name not in sup:
            sup.append(ir_name)
    added = []
    for sha, g in groups.items():
        e = cache.add_reuse(op=kind, entry=entry, signature=specs[kind].signature, arch=arch, shapes=g["shapes"],
                            code=g["code"], slot_us=round(g["us"], 3), impl_id=f"reuse:{sha[:12]}",
                            source={"by": "mb-reuse", "ir": ir_name, "from": g["from"],
                                    "shape_us": g["per_shape"]})
        added.append({"impl_id": e["impl_id"], "from": g["from"], "shapes": len(g["shapes"]), "us": round(g["us"], 2)})
    krep["written"] = added
    krep["superseded"] = [e["impl_id"] for e in old]
    krep["result"] = "routed"
    krep["before_us"] = round(sum(((row.get("incumbent") or {}).get("us") or 0.0) * row["n"]
                                  for row in table.values()), 2)
    krep["after_us"] = round(sum(g["us"] for g in groups.values())
                             + sum(((row.get("incumbent") or {}).get("us") or 0.0) * row["n"]
                                   for k, row in table.items() if pick.get(k) is None and not any(
                                       _skey(row["shape"]) in g["per_shape"] for g in groups.values())), 2)


def _model_guard(src_ir: Path, g: dict, cache, base_before: dict, arch: str, specs: dict, work: Path, runner: Runner,
                 guard_pct: float, reps: int, remeasure_fn) -> dict:
    """The whole model before vs after the write-back, back to back: must pass and not be slower."""
    from .kb_export import export_problem
    from .kb_ingest import cache_plan
    from .kb_proposals import _median, remeasure, write_variant_ir

    after = cache_plan(g, cache, arch, specs)
    probs = []
    for tag, plan in (("before", base_before), ("after", after)):
        d = write_variant_ir(src_ir, plan["graph"], work / f"guard_{tag}_ir")
        probs.append(export_problem(d, work / f"guard_{tag}", slot_sources=plan["sources"],
                                    slot_source_tags=plan["tags"], sub_sources=plan["sub_sources"], specs=specs,
                                    strategy="whole"))
    fn = remeasure_fn or (lambda ps: remeasure([(p, {}) for p in ps], arch=arch.removeprefix("sm_"), reps=reps,
                                               kbeval=runner.kbeval, gpu_model=runner.gpu_model,
                                               work=work / "guard_remeasure" if runner.kbeval is not None else None))
    try:
        t = fn(probs)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "why": f"guard run failed: {type(e).__name__}: {e}"[:600]}
    b, a = _median(t[0]), _median(t[1])
    out = {"before_us": b, "after_us": a, "before_runs": t[0], "after_runs": t[1]}
    if a is None:
        return {**out, "ok": False, "why": "the model with the routed kernels did not pass"}
    if b is not None and a > b * (1.0 + guard_pct):
        return {**out, "ok": False, "why": f"model slower: {a:.1f} vs {b:.1f} us"}
    return {**out, "ok": True, "gain": (b - a) / b if b else None}


# ---------------------------------------------------------------- seeding new slot problems

def add_seed_candidates(problem_dir: Path, cache, *, arch: str, specs: dict, mode: str = "siblings",
                        impl_dirs: list[Path] | None = None, n: int = 3) -> dict:
    """Give a freshly exported slot problem the pool's best sources as trial seeds (``manifest["seed_candidates"]``,
    files ``slots/<id>.cand<k>.cu``): KernelBlaster tries them at start-up and keeps one only if it passes this
    problem's checks and is faster (``OptMultislot._seed_candidates``). Slots already seeded from a cached winner of
    their own shapes are left alone. -> {slot id: [from, ...]}."""
    if mode in (None, "", "off"):
        return {}
    problem_dir = Path(problem_dir)
    mf = problem_dir / "manifest.json"
    man = json.loads(mf.read_text())
    arch_l = f"sm_{str(arch).removeprefix('sm_')}"
    out = {}
    for s in man["slots"]:
        if str(s.get("seed_source", "")).startswith("cache:"):
            continue
        cur = (problem_dir / s["file"]).read_text()
        cands = [c for c in pool_candidates(s["op"], cache, arch=arch_l, specs=specs, ir_name=man.get("ir_name"),
                                            mode=mode, impl_dirs=impl_dirs, max_cands=n + 1)
                 if c["sha"] != _sha(cur)][:n]
        if not cands:
            continue
        lst = []
        for k, c in enumerate(cands):
            f = f"slots/{s['id']}.cand{k}.cu"
            (problem_dir / f).write_text(c["code"])
            lst.append({"file": f, "from": c["from"]})
        man.setdefault("seed_candidates", {})[s["id"]] = lst
        out[s["id"]] = [c["from"] for c in cands]
    if out:
        mf.write_text(json.dumps(man, indent=1))
    return out
