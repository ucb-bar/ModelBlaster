"""Search fixes for the per-shape flow (A3; l3_farm/analysis/search_fixes.md).

The Level 3 sweep left dominant ops on their naive seed kernels without saying so: 15_DenseNet121's 960-channel 7x7
conv lost its winner to a whole-model rejection (``kb_plan.profile_ir`` profile fallback), stayed on the hand seed for
the rest of the run (its plan item was already in ``state["sent"]``) and cost 42% of the model; routing an existing
kernel of another shape to it takes the model from 2,436 to ~1,487 us at zero LLM cost (shape_family.md). Four fixes:

1. **fallback routing + re-queue** (``--fallback-repair on|route|requeue|off``, default ``on`` = both; kill switch
   ``off``). *Route* (zero LLM, inside the profile fallback loop): for the instances whose cached winner was just
   rejected, the verified kernels of the same op + signature for OTHER shapes (cache entries not rejected for this IR,
   this run's implementation store, nearest shapes first) are built on exactly those instances (one slot per shape,
   the driver's isolated per-slot check = correctness on that shape, including launch-limit failures such as
   ``invalid argument`` from hard-coded grids / shared memory); the fastest passing one that beats the fallback source
   becomes a ``reuse`` cache entry guarded on those shapes (provenance ``source.by = mb-fallback-route``), and the
   profile loop re-runs the whole model, so a kernel that fails there is rejected in turn and the next one is tried.
   *Re-queue*: an item whose instances are still on a seed kernel after a fallback leaves the ``sent`` set for this
   round's plan (at most ``--fallback-requeue-max`` times per item) and its problem's slot semantics carry the
   rejection (impl id + the whole-model error), so the LLM knows why the last winner was dropped.
2. **final-model guard** (``--final-guard warn|repair|off``, default ``warn``): after the final verification, every
   op above ``--final-guard-share`` (0.10) of the model's time that still runs its seed kernel or is within
   ``--final-guard-min-speedup`` (1.25x) of its seed time is reported loudly (stdout + ``summary.json``
   ``search_fixes.final_guard`` and ``irs[ir].seed_left``) with a reason (profile fallback / verify rejection / sent but
   no winner / never sent). ``repair`` additionally routes those ops (zero LLM, as in 1) and verifies again.
3. **nearest-shape seeds** (``--seed-nearest off|on``, default off): a freshly exported slot problem not seeded from
   its own cached winner gets the verified kernels of the NEAREST shapes (same op + signature; shape distance below) as
   trial seeds through the manifest's ``seed_candidates`` (A1's path: KernelBlaster tries them at start-up and keeps
   one only if it passes on this problem's shapes and is faster). In the sweep the ImplStore's ``adapt`` tier fired
   in 32 of 23,757 steps and its ``seed`` tier only for exact shape buckets in rounds >= 1.
4. **rollout lineage** (KernelBlaster ``KERNELBLASTER_MULTISLOT_ROLLOUT_ACCEPT=lineage`` /
   ``KERNELBLASTER_MULTISLOT_ROLLOUT_RESYNC``; ModelBlaster ``--rollout-accept lineage``, ``--rollout-resync X``):
   see ``opt_multislot.rollout_decision`` / ``should_resync``. Default off.

All decisions are recorded (``round.json`` ``search_fixes``, ``state.json`` ``requeued``, the profile's
``cache_fallback[].route``, ``summary.json``).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

FALLBACK_MODES = ("on", "route", "requeue", "off")
GUARD_MODES = ("warn", "repair", "off")
CFG_DEFAULTS = {"fallback_repair": "on", "fallback_requeue_max": 1, "final_guard": "warn",
                "final_guard_share": 0.10, "final_guard_min_speedup": 1.25, "seed_nearest": "off",
                "rollout_resync": 0.0}
ROUTE_TAG = "mb-fallback-route"
REPAIR_TAG = "mb-final-repair"


def _sha(code: str) -> str:
    return hashlib.sha1(code.encode()).hexdigest()


def _skey(shape: dict) -> str:
    return json.dumps(shape, sort_keys=True)


# ---------------------------------------------------------------- shape distance (item 3, and candidate order in 1/2)

def shape_distance(a: dict, b: dict) -> float:
    """How far apart two instance shapes of one op kind are: mean |log2| ratio over their numeric fields (+1 so
    zeros and flags count), plus 2 per field only one of them has. 0 = same shape. A fused op's shape
    ({"a": stage-0 shape, ...}) is compared stage by stage."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return 0.0 if a == b else 8.0
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    tot = 0.0
    for k in keys:
        x, y = a.get(k), b.get(k)
        if isinstance(x, dict) or isinstance(y, dict):
            tot += shape_distance(x or {}, y or {})
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            tot += abs(math.log2((abs(x) + 1.0) / (abs(y) + 1.0)))
        elif x != y:
            tot += 2.0
    return tot / len(keys)


def set_distance(targets: list[dict], shapes: list[dict]) -> float:
    """A candidate verified on ``shapes`` for ``targets``: the worst target's distance to its nearest shape."""
    if not targets or not shapes:
        return float("inf")
    return max(min(shape_distance(t, s) for s in shapes) for t in targets)


# ---------------------------------------------------------------- the candidate pool (with shapes)

def _impl_records(dirs) -> list[dict]:
    out = []
    for d in dirs or []:
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


def rejected_shas(cache, kind_entry: str, ir_name: str | None) -> set[str]:
    """Code shas of the cache entries of a C entry point rejected for this IR (never offered again there)."""
    if ir_name is None:
        return set()
    out = set()
    for e in cache.entries:
        if e["entry"] == kind_entry and ir_name in (e.get("rejected") or ()):
            try:
                out.add(_sha(cache.code(e)))
            except OSError:
                pass
    return out


def nearest_pool(kind: str, cache, *, arch: str, specs: dict, ir_name: str | None, targets: list[dict],
                 impl_dirs=None, exclude: set | None = None, max_cands: int = 4) -> list[dict]:
    """Verified sources of ``kind`` (same C entry, signature, dtype, arch) for any shapes, one per code, nearest to
    ``targets`` first (then faster per instance): cache entries not rejected / superseded for this IR and
    implementation-store records at least 1.02x over their seed. Codes in ``exclude`` (and every code rejected for
    this IR) are left out. -> [{"code", "sha", "from", "shapes", "dist", "us_per_inst"}]."""
    from .cuda_dtypes import cuda_signature, op_dtype
    from .kb_ingest import signature_sha, superseded
    from .kernel_abi import parse_signature

    if kind not in specs:
        return []
    arch = f"sm_{str(arch).removeprefix('sm_')}"
    entry, _ = parse_signature(specs[kind].signature)
    sig = signature_sha(cuda_signature(specs[kind].signature))
    dtype = op_dtype(kind)
    bad = set(exclude or ()) | rejected_shas(cache, entry, ir_name)
    by: dict[str, dict] = {}

    def offer(c):
        if c["sha"] in bad:
            return
        o = by.get(c["sha"])
        if o is None:
            by[c["sha"]] = c
        else:                                          # one code verified on several shape sets: all its shapes
            o["shapes"] = o["shapes"] + [s for s in c["shapes"] if s not in o["shapes"]]
            o["us_per_inst"] = min(o["us_per_inst"], c["us_per_inst"])
    for e in cache.entries:
        if (e["entry"], e["signature_sha"], e["arch"], e.get("dtype", "fp32")) != (entry, sig, arch, dtype):
            continue
        if ir_name is not None and (ir_name in (e.get("rejected") or ()) or superseded(e, ir_name)):
            continue
        try:
            code = cache.code(e)
        except OSError:
            continue
        shapes = [json.loads(s) for s in e["shape_guard"]]
        offer({"code": code, "sha": _sha(code), "from": f"cache:{e['impl_id']}", "shapes": shapes,
               "us_per_inst": (e.get("slot_us") or 0.0) / max(len(shapes), 1)})
    for r in _impl_records(impl_dirs):
        if r.get("op") != kind or r.get("signature_sha") != sig or r.get("dtype", "fp32") != dtype \
                or r.get("arch") != arch or r.get("target", "cuda") != "cuda":
            continue
        if (r.get("speedup_vs_seed") or 0.0) < 1.02:
            continue
        code = r["_code"].read_text()
        shapes = list(r.get("shapes") or [])
        offer({"code": code, "sha": _sha(code), "from": f"impl:{r.get('id')}:{r.get('problem_id')}",
               "shapes": shapes, "us_per_inst": (r.get("slot_us") or 0.0) / max(len(shapes), 1)})
    for c in by.values():
        c["dist"] = round(set_distance(targets, c["shapes"]), 4)
    cands = sorted(by.values(), key=lambda c: (c["dist"], c["us_per_inst"] or float("inf")))
    return cands[:max(0, int(max_cands))]


# ---------------------------------------------------------------- routing (items 1 and 2b), zero LLM

def route_ops(*, kind: str, ops: list[str], graph: dict, src_ir: Path, cache, specs: dict, arch: str, ir_name: str,
              work: Path, runner=None, kbeval=None, gpu_model: str | None = None, impl_dirs=None,
              incumbent: dict | None = None, exclude: set | None = None, max_cands: int = 4,
              min_gain: float = 0.03, tag: str = ROUTE_TAG, why: dict | None = None, measure_fn=None) -> dict:
    """Give the instances ``ops`` (of ``kind``, in ``graph`` = the deployed graph of ``src_ir``) the fastest verified
    kernel of another shape that passes on their own shapes. Every candidate runs in one problem holding exactly
    these instances, one slot per distinct shape (``kb_reuse.measure_kind``): the driver checks each slot against
    the reference on its own instances (wrong results and launch failures alike) and times it. Per shape the fastest
    passing candidate is taken if it beats ``incumbent`` (op -> source; missing = the export's seed) by
    ``min_gain``; it is written as a ``reuse`` cache entry guarded on exactly the shapes it won (provenance in
    ``source``). -> report {"kind", "ops", "pool", "table", "choice", "written", "result"}."""
    from . import kb_reuse
    from .kernel_abi import parse_signature

    work = Path(work)
    rep = {"kind": kind, "ops": list(ops), "result": "nothing to try"}
    shapes = []
    for o in graph["ops"]:
        if o["name"] in ops and o.get("shape", {}) not in shapes:
            shapes.append(o.get("shape", {}))
    cands = nearest_pool(kind, cache, arch=arch, specs=specs, ir_name=ir_name, targets=shapes,
                         impl_dirs=impl_dirs, exclude=exclude, max_cands=max_cands)
    rep["pool"] = [{"sha": c["sha"][:12], "from": c["from"], "dist": c["dist"]} for c in cands]
    if not ops or not cands:
        return rep
    incumbent = dict(incumbent or {})
    inc_codes = {_sha(v) for v in incumbent.values() if v}
    cands = [c for c in cands if c["sha"] not in inc_codes] or cands
    try:
        if measure_fn is not None:
            m = measure_fn(kind, graph, Path(src_ir), list(ops), incumbent, cands)
        else:
            runner = runner or kb_reuse.Runner(arch=arch, kbeval=kbeval, gpu_model=gpu_model, work=work)
            m = kb_reuse.measure_kind(kind, graph, Path(src_ir), list(ops), incumbent, cands, specs=specs,
                                      runner=runner, work=work)
    except Exception as e:  # noqa: BLE001 -- a failed measurement leaves the instances as they are
        rep.update(result="measure failed", error=f"{type(e).__name__}: {e}"[:800])
        return rep
    table = m["table"]
    inc_sha = {k: _sha(incumbent[row["ops"][0]]) if incumbent.get(row["ops"][0]) else None for k, row in table.items()}
    pick = kb_reuse.choose(table, cands, inc_sha, min_gain)
    rep["table"] = {str(k): {kk: v for kk, v in row.items() if kk != "slot"} for k, row in table.items()}
    rep["choice"] = {str(k): v for k, v in pick.items()}
    if not any(pick.values()):
        rep["result"] = "no candidate passes and beats the incumbent"
        return rep
    entry, _ = parse_signature(specs[kind].signature)
    by_sha = {c["sha"][:12]: c for c in cands}
    groups: dict[str, dict] = {}
    for k, row in table.items():
        t = pick.get(k)
        if t is None:
            continue
        g = groups.setdefault(t, {"shapes": [], "us": 0.0, "per_shape": {}})
        us = row["cands"][t]["us"]
        g["shapes"].append(row["shape"])
        g["us"] += (us or 0.0) * row["n"]
        g["per_shape"][_skey(row["shape"])] = {"us": us, "incumbent_us": (row.get("incumbent") or {}).get("us")}
    rep["written"] = []
    arch_l = f"sm_{str(arch).removeprefix('sm_')}"
    for t, g in groups.items():
        c = by_sha[t]
        e = cache.add_reuse(op=kind, entry=entry, signature=specs[kind].signature, arch=arch_l, shapes=g["shapes"],
                            code=c["code"], slot_us=round(g["us"], 3), impl_id=f"route:{t}",
                            source={"by": tag, "ir": ir_name, "from": c["from"], "dist": c["dist"],
                                    "shape_us": g["per_shape"], **({"why": why} if why else {})})
        rep["written"].append({"impl_id": e["impl_id"], "from": c["from"], "dist": c["dist"],
                               "shapes": len(g["shapes"]), "us": round(g["us"], 2)})
    cache.save()
    rep["result"] = "routed"
    return rep


def fallback_router(*, enabled: bool, kbeval=None, gpu_model: str | None = None, impl_dirs=None, max_cands: int = 4,
                    min_gain: float = 0.03, measure_fn=None, log: list | None = None):
    """The ``on_fallback`` callback for ``kb_plan.profile_ir`` (None when routing is off)."""
    if not enabled:
        return None

    def on_fallback(*, ir_dir, graph, bad, man, res, cache, specs, arch, work, attempt, **_):
        from .kb_ingest import cache_plan

        ir_name = man["ir_name"]
        slot_ops: dict[str, list] = {}
        for r in man["ops"]:
            slot_ops.setdefault(r["slot"], []).append(r["name"])
        errors = res.get("errors") or []
        out = []
        by_kind: dict[str, list] = {}
        for key, e in bad.items():
            sid = key.split("/")[0]
            if "/" in key:                     # a stage kernel of a fused slot: the fused slot re-seeds (no route)
                continue
            for n in slot_ops.get(sid, []):
                by_kind.setdefault(e["op"], [])
                if n not in by_kind[e["op"]]:
                    by_kind[e["op"]].append(n)
        if not by_kind:
            return None
        # what the next profile would deploy: the fallback sources after the rejection
        plan = cache_plan(graph, cache, f"sm_{str(arch).removeprefix('sm_')}", specs)
        gplan = plan["graph"]
        for kind, ops in by_kind.items():
            why = {"rejected": sorted({e["impl_id"] for e in bad.values() if e["op"] == kind}),
                   "errors": [x for x in errors if any(n in x for n in ops)][:6] or errors[:3]}
            r = route_ops(kind=kind, ops=ops, graph=gplan, src_ir=Path(ir_dir), cache=cache, specs=specs, arch=arch,
                          ir_name=ir_name, work=Path(work) / "fallback_route" / f"a{attempt}_{kind}",
                          kbeval=kbeval, gpu_model=gpu_model, impl_dirs=impl_dirs, max_cands=max_cands,
                          min_gain=min_gain, why=why, measure_fn=measure_fn)
            r.pop("table", None)
            out.append(r)
        if log is not None:
            log.extend({"ir": ir_name, "attempt": attempt, **r} for r in out)
        return out
    return on_fallback


# ---------------------------------------------------------------- re-queue (item 1)

def item_keys_of(rec: dict, canon: dict | None, specs: dict | None) -> list[str]:
    """The plan item keys an op record (profile ``ops[]`` entry) can have: exact and canonical."""
    from .kb_plan import canon_fields, canon_shape, item_key
    keys = [item_key(rec["entry"], rec["signature_sha"], rec.get("shape", {}))]
    if canon is not None:
        k2 = item_key(rec["entry"], rec["signature_sha"],
                      canon_shape(rec.get("shape", {}), canon_fields(rec["op"], specs, canon)))
        if k2 not in keys:
            keys.append(k2)
    return keys


def stuck_after_fallback(profile: dict) -> dict[str, dict]:
    """Ops a profile fallback took a cached winner from that still run a seed (no cache source) in the profile:
    {op name: {"rejected": [impl ids], "errors": [...]}}."""
    out: dict[str, dict] = {}
    fb = profile.get("cache_fallback") or []
    if not fb:
        return out
    seed = {r["name"]: str(r.get("seed") or "") for r in profile.get("ops", [])}
    for step in fb:
        for key, names in (step.get("ops") or {}).items():
            for n in names:
                if n in seed and not seed[n].startswith("cache:"):
                    d = out.setdefault(n, {"rejected": [], "errors": []})
                    imp = (step.get("rejected") or {}).get(key)
                    if imp and imp not in d["rejected"]:
                        d["rejected"].append(imp)
                    d["errors"] += [x for x in step.get("errors") or [] if n in x and x not in d["errors"]]
    return out


def requeue(profiles: list[dict], sent: set, state: dict, round_no: int, *, max_per_key: int = 1,
            canon: dict | None = None, specs: dict | None = None) -> tuple[set, dict]:
    """Items to plan again this round although they were sent before: their instances lost their winner to a profile
    fallback and still run a seed. -> (keys, {key: note}); ``state["requeued"]`` counts each key (at most
    ``max_per_key`` re-queues), so a kernel that keeps failing in the model does not loop."""
    done = state.setdefault("requeued", {})
    keys, notes = set(), {}
    for p in profiles:
        stuck = stuck_after_fallback(p)
        for r in p.get("ops", []):
            if r["name"] not in stuck:
                continue
            for k in item_keys_of(r, canon, specs):
                if k not in sent or k in keys:
                    continue
                rec = done.setdefault(k, {"n": 0, "rounds": []})
                if round_no in rec["rounds"]:              # resumed round: already counted
                    keys.add(k)
                elif rec["n"] < max_per_key:
                    rec["n"] += 1
                    rec["rounds"].append(round_no)
                    keys.add(k)
                else:
                    continue
                notes[k] = {"op": r["name"], "kind": r["op"], "ir": p["ir"], **stuck[r["name"]]}
    return keys, notes


# ---- sent-on-success (live_debug/search_inefficiency_cross_problem.md: ~70% of round >= 1 plan items were `sent-before`)

def sent_on_success() -> bool:
    """``MB_SENT_ON_SUCCESS=0`` restores the old rule: every dispatched plan item is ``sent``."""
    return os.environ.get("MB_SENT_ON_SUCCESS", "1").strip().lower() not in ("0", "off", "false", "no")


def sent_retry_max() -> int:
    """Times a lost item may be planned again before it is marked sent anyway (``MB_SENT_RETRY_MAX``, default 2)."""
    try:
        return max(0, int(os.environ.get("MB_SENT_RETRY_MAX", "2")))
    except ValueError:
        return 2


def result_measured(res: dict | None) -> bool:
    """Did a plan problem's KernelBlaster run end in a measured outcome? Yes: it passed (accepted or not faster), or
    it was measured and failed as a kernel (baseline timed, final not ok, no infrastructure signature), or an accepted
    fusion absorbed its ops (``external_stop``). No: no result, or a result lost to infrastructure (``final_infra``,
    scp / NoDevice / relay errors) or without a baseline time: those items stay eligible."""
    from . import kb_proposals
    if not res:
        return False
    if res.get("final_ok"):
        return True
    if (res.get("efficiency") or {}).get("external_stop"):
        return True
    # the broader infra test (scp / NoDevice / relay text without the tag) when fix 2 is in, else the tag + [infra] text
    infra = getattr(kb_proposals, "is_infra_result", None) or (
        lambda r: bool(r.get("final_infra") or kb_proposals.is_infra_text(r.get("final_error"))))
    if infra(res):
        return False
    return res.get("baseline_partition_us") is not None


def sent_keys(plan: dict, results: list[dict], state: dict, round_no: int, *, max_retries: int | None = None
              ) -> tuple[list[str], dict]:
    """The selected plan items to mark sent after a round -> (keys, {key: why held}). An item whose problem has no
    measured outcome stays eligible, at most ``max_retries`` times (``state["sent_retry"]`` = {key: {n, rounds}},
    resume-safe like ``requeued``); the last time it is sent anyway, so a persistently failing item ends."""
    mx = sent_retry_max() if max_retries is None else max_retries
    by_name = {str(r.get("problem")): r for r in results if r}
    probs = plan.get("problems") or []
    pos = results if len(results) == len(probs) else []
    key_res: dict[str, dict | None] = {}
    for i, p in enumerate(probs):
        pid = p.get("id", p["ir"])
        res = by_name.get(f"{pid}__r{round_no:02d}") or next(
            (r for n, r in by_name.items() if n.endswith(f"{pid}__r{round_no:02d}")), None) or \
            (pos[i] if pos else None)
        for k in p.get("keys") or []:
            key_res[k] = res
    retry = state.setdefault("sent_retry", {})
    out, held = [], {}
    for it in plan["items"]:
        if not it.get("selected"):
            continue
        k = it["key"]
        res = key_res.get(k)
        if k not in key_res or result_measured(res):
            out.append(k)                           # (an item with no problem of its own: as before)
            continue
        rec = retry.setdefault(k, {"n": 0, "rounds": []})
        if round_no in rec["rounds"]:               # resumed round: already counted
            held[k] = "requeued (resumed round)"
            continue
        if rec["n"] < mx:
            rec["n"] += 1
            rec["rounds"].append(round_no)
            held[k] = "no measured outcome: " + ("no result" if not res else
                                                  str(res.get("final_error") or "baseline not timed")[:160])
        else:
            out.append(k)                           # retries exhausted: sent
            rec["gave_up"] = round_no
    return out, held


def rejection_note(note: dict) -> str:
    errs = "; ".join(note.get("errors") or [])[:600]
    return ("\n[ModelBlaster] The previous winner for this op/shape (impl "
            f"{', '.join(note.get('rejected') or ['?'])}) passed its own problem but FAILED when the whole model was "
            f"built and run{': ' + errs if errs else ''}. The model links every slot into one program and runs them "
            "in one process: do not rely on the launch configuration of another shape (grid, block, dynamic "
            "shared memory must be derived from this call's arguments and stay within the device limits without "
            "per-process opt-ins that can fail), do not leave a CUDA error behind (e.g. a failing "
            "cudaFuncSetAttribute), and keep static state per instance shape.")


def annotate_requeued(dirs: list[Path], plan: dict, notes: dict) -> dict:
    """Add the rejection note to the slot semantics (-> the KernelBlaster prompt) of re-queued problems."""
    out = {}
    if not notes:
        return out
    for d, prob in zip(dirs, plan.get("problems", [])):
        hit = [notes[k] for k in prob.get("keys", []) if k in notes]
        if not hit:
            continue
        mf = Path(d) / "manifest.json"
        man = json.loads(mf.read_text())
        names = {n["op"] for n in hit}
        for s in man["slots"]:
            if any(i.get("op_name") in names for i in s.get("instances", [])):
                s["semantics"] = (s.get("semantics") or "") + rejection_note(hit[0])
                out.setdefault(Path(d).name, []).append(s["id"])
        man["requeued"] = hit
        mf.write_text(json.dumps(man, indent=1))
    return out


# ---------------------------------------------------------------- final-model guard (item 2)

def seed_times(profile: dict | None) -> dict[str, float]:
    """op name -> its time on the hand seed (the first profile's ops that ran one)."""
    if not profile:
        return {}
    return {r["name"]: r["us"] for r in profile.get("ops", []) if r.get("us") is not None and r.get("seed") == "hand"}


def _on_seed(tag: str) -> bool:
    tag = str(tag or "")
    return tag == "hand" or tag == "fused-seed"


def seed_left(ops: list[dict], seeds: dict[str, float], *, share: float = 0.10, min_speedup: float = 1.25,
              total_us: float | None = None) -> list[dict]:
    """Dominant ops (>= ``share`` of the summed op time) still on their seed kernel (tag ``hand`` / a fused seed
    without stage winners) or less than ``min_speedup`` faster than their seed time (a fused op: the sum of its
    stages' seed times). ``ops``: [{"name", "op", "us", "seed_source"}]. -> flagged ops, largest first."""
    tot = total_us or sum((o.get("us") or 0.0) for o in ops) or 1.0
    out = []
    for o in ops:
        us = o.get("us")
        if us is None or us / tot < share:
            continue
        parts = str(o["name"]).split("+")
        s_us = sum(seeds[n] for n in parts) if all(n in seeds for n in parts) else None
        sp = (s_us / us) if s_us and us else None
        on = _on_seed(o.get("seed_source"))
        if on or (sp is not None and sp < min_speedup):
            out.append({"name": o["name"], "op": o["op"], "us": round(us, 2), "share": round(us / tot, 4),
                        "seed_us": round(s_us, 2) if s_us else None,
                        "speedup_vs_seed": round(sp, 3) if sp else None, "on_seed_kernel": on,
                        "seed_source": o.get("seed_source"), "slot": o.get("slot")})
    return sorted(out, key=lambda x: -x["share"])


def classify(flag: dict, *, fallback_ops: set, verify_bad: set, sent: set, keys: list[str]) -> str:
    if flag["name"] in fallback_ops:
        return "profile-fallback: its winner failed in the whole model"
    if flag.get("slot") in verify_bad:
        return "final verification rejected its winner"
    if any(k in sent for k in keys):
        return "sent to KernelBlaster, no faster winner kept for it"
    return "never sent (below the plan's min share / top-k, or a new op of an accepted rewrite)"


# ---------------------------------------------------------------- the flow's switches

class Fixes:
    """This run's search fixes (kb_graph_opt hooks); every method is a no-op when its switch is off."""

    def __init__(self, cfg, work: Path):
        self.cfg, self.work = cfg, Path(work)
        g = lambda k: getattr(cfg, k, CFG_DEFAULTS[k])          # noqa: E731
        self.fallback = (g("fallback_repair") or "on").lower()
        if self.fallback not in FALLBACK_MODES:
            raise ValueError(f"--fallback-repair must be {'|'.join(FALLBACK_MODES)}, not {self.fallback!r}")
        self.requeue_max = int(g("fallback_requeue_max") if g("fallback_requeue_max") is not None else 1)
        self.guard = (g("final_guard") or "warn").lower()
        if self.guard not in GUARD_MODES:
            raise ValueError(f"--final-guard must be {'|'.join(GUARD_MODES)}, not {self.guard!r}")
        self.guard_share = float(g("final_guard_share") or 0.10)
        self.guard_speedup = float(g("final_guard_min_speedup") or 1.25)
        self.nearest = (g("seed_nearest") or "off").lower()
        if self.nearest not in ("off", "on"):
            raise ValueError(f"--seed-nearest must be off|on, not {self.nearest!r}")
        self.resync = float(g("rollout_resync") or 0.0)
        self.rollout_accept = (getattr(cfg, "rollout_accept", "auto") or "auto").lower()
        self.route_log: list = []
        self.reports: dict = {"requeued": {}, "annotated": {}, "nearest_seeds": {}, "sent_held": {},
                              "routes": self.route_log}

    @property
    def cuda(self) -> bool:
        """The cache-entry routing / re-ingest below read CUDA arch keys (``sm_<N>``) and ``.cu`` slot sources: on another
        slot target (``--target opencl``) they are off (the profile's own fallback -- reject a failing winner, redo the
        profile on the next best source -- and the final-guard warning are target-independent and stay)."""
        return getattr(self.cfg, "target", "cuda") == "cuda"

    @property
    def routing(self) -> bool:
        return self.fallback in ("on", "route") and self.cuda

    @property
    def requeueing(self) -> bool:
        return self.fallback in ("on", "requeue") and self.requeue_max > 0

    def apply_env(self) -> None:
        """KernelBlaster's rollout lineage switches (item 4) for its subprocesses; the cache's keep-beside-rejected rule
        (kb_ingest.CudaCache.add) follows the fallback repair's kill switch."""
        if self.fallback == "off":
            os.environ["MB_CACHE_BESIDE_REJECTED"] = "0"
        if self.rollout_accept == "lineage":
            os.environ["KERNELBLASTER_MULTISLOT_ROLLOUT_ACCEPT"] = "lineage"
        if self.resync > 0:
            os.environ["KERNELBLASTER_MULTISLOT_ROLLOUT_RESYNC"] = str(self.resync)

    def round_key(self) -> dict:
        """Only behaviour changes that are not the defaults (existing work dirs keep their key; the default-on
        fallback repair restores correctness and does not change the key)."""
        k = {}
        if self.fallback != "on":
            k["fallback_repair"] = self.fallback
        if self.nearest != "off":
            k["seed_nearest"] = self.nearest
        if self.resync > 0:
            k["rollout_resync"] = self.resync
        return k

    def impl_dirs(self) -> list[Path]:
        d = [self.work / "knowledge" / "impl"]
        seed = getattr(self.cfg, "knowledge_global_seed", "") or ""
        if seed:
            d.append(Path(seed) / "impl")
        return [x for x in d if x.exists()]

    # ---- item 1
    def on_fallback(self, kbeval=None, gpu_model: str | None = None):
        return fallback_router(enabled=self.routing, kbeval=kbeval, gpu_model=gpu_model, impl_dirs=self.impl_dirs(),
                               log=self.route_log)

    def exclude(self, sent: set, profiles: list[dict], state: dict, round_no: int, plan_kw: dict,
                specs: dict) -> set:
        if not self.requeueing:
            return sent
        keys, notes = requeue(profiles, sent, state, round_no, max_per_key=self.requeue_max,
                              canon=plan_kw.get("canon"), specs=specs)
        self._notes = notes
        if keys:
            self.reports["requeued"][str(round_no)] = notes
            print(json.dumps({"search_fixes": {"round": round_no, "requeued": sorted(notes)}})[:2000], flush=True)
        return sent - keys

    def mark_sent(self, plan: dict, results: list[dict], state: dict, round_no: int) -> list[str]:
        """After a round's plan run: ``state["sent"]`` += the selected items that produced a measured outcome
        (``MB_SENT_ON_SUCCESS``, default on; off = every selected item, as before). Items lost to infrastructure or
        never evaluated stay eligible for the next round's plan, ``MB_SENT_RETRY_MAX`` times each."""
        if not sent_on_success():
            keys = [it["key"] for it in plan["items"] if it.get("selected")]
            state["sent"] += keys
            return keys
        keys, held = sent_keys(plan, results, state, round_no)
        state["sent"] += [k for k in keys if k not in state["sent"]]
        if held:
            self.reports["sent_held"][str(round_no)] = held
            print(json.dumps({"search_fixes": {"round": round_no, "not_sent": sorted(held)}})[:2000], flush=True)
        return keys

    def seed_problems(self, dirs: list[Path], plan: dict, cache, specs: dict, round_no: int | None = None) -> list[Path]:
        notes = getattr(self, "_notes", {}) or {}
        if notes:
            try:
                a = annotate_requeued(dirs, plan, notes)
            except Exception as e:  # noqa: BLE001 -- a note never breaks a round
                a = {"error": f"{type(e).__name__}: {e}"[:300]}
            if a:
                self.reports["annotated"][str(round_no)] = a
        if self.nearest == "on":
            for d in dirs:
                try:
                    r = add_nearest_seed_candidates(d, cache, arch=self.cfg.arch, specs=specs,
                                                    impl_dirs=self.impl_dirs())
                except Exception as e:  # noqa: BLE001 -- seeding is best effort
                    r = {"error": f"{type(e).__name__}: {e}"[:300]}
                if r:
                    self.reports["nearest_seeds"][Path(d).name] = r
        return dirs

    def ingest_requeued(self, results: list[dict], cache, specs: dict, round_no: int | None = None) -> list | None:
        """Slot problems that worked on shapes whose cached winner was rejected in a whole model (re-queued items,
        or items planned again for any other reason) put their own best slot source into the cache directly.
        Both stores keep only the FASTEST kernel per shape set -- KernelBlaster's winners export and CudaCache.add --
        and the fastest is the rejected kernel, so a re-optimized, slower but working winner never reached the
        deployed model: 15_DenseNet121's 960x7x7 conv was re-optimized in round 1 (8.22 us in its own problem) and
        dropped next to the rejected 8.19 us entry. The cache keeps it beside the rejected one (CudaCache.add)."""
        if self.fallback == "off" or not self.cuda:
            return None
        from .kb_ingest import _shape_key
        from .kernel_abi import parse_signature
        arch = f"sm_{str(self.cfg.arch).removeprefix('sm_')}"
        out = []
        for x in results:
            rdir = Path(x.get("_dir") or "")
            try:
                res = json.loads((rdir / "result.json").read_text())
                pdir = Path(res.get("problem_dir") or "")
                man = json.loads((pdir / "manifest.json").read_text())
            except (OSError, ValueError):
                continue
            if not res.get("final_ok"):
                continue
            for slot in man["slots"]:
                op = slot["op"]
                guard = sorted({_shape_key(i.get("shape", {})) for i in slot.get("instances", [])})
                same = [e for e in cache.entries if e["op"] == op and e["arch"] == arch and e["shape_guard"] == guard
                        and not e.get("reuse")]
                if not same or any(not e.get("rejected") for e in same):
                    continue                                 # no rejected winner of these shapes: normal ingest
                b = (res.get("slots") or {}).get(slot["id"]) or {}
                f = rdir / "best" / "slots" / f"{slot['id']}.cu"
                if not b.get("best_us") or not f.exists() or op not in specs:
                    continue
                sp = (b.get("baseline_us") or 0.0) / b["best_us"]
                if sp < 1.05:
                    out.append({"problem": pdir.name, "slot": slot["id"], "status": "below-min-speedup", "speedup": sp})
                    continue
                e, st = cache.add(op=op, entry=parse_signature(specs[op].signature)[0], signature=specs[op].signature,
                                  arch=arch, shapes=[i.get("shape", {}) for i in slot.get("instances", [])],
                                  code=f.read_text(), slot_us=b["best_us"], speedup=sp,
                                  impl_id=b.get("impl_id") or f"own:{pdir.name}:{slot['id']}", dtype=man.get("dtype"),
                                  source={"by": "mb-beside-rejected", "problem_id": pdir.name, "round": round_no})
                out.append({"problem": pdir.name, "slot": slot["id"], "impl_id": e["impl_id"], "status": st,
                            "speedup": round(sp, 3)})
        if out:
            cache.save()
        return out or None

    def round_record(self) -> dict | None:
        """What the fixes did since the last call (one round): re-queued items, annotated problems, nearest seeds,
        fallback routes."""
        seen = getattr(self, "_seen", {"routes": 0, "requeued": set(), "annotated": set(), "nearest_seeds": set(),
                                       "sent_held": set()})
        r = {}
        for k in ("requeued", "annotated", "nearest_seeds", "sent_held"):
            new = {kk: v for kk, v in self.reports[k].items() if kk not in seen[k]}
            if new:
                r[k] = new
                seen[k] |= set(new)
        if len(self.route_log) > seen["routes"]:
            r["routes"] = [{k: x.get(k) for k in ("ir", "attempt", "kind", "ops", "result", "written", "pool")}
                           for x in self.route_log[seen["routes"]:]]
            seen["routes"] = len(self.route_log)
        self._seen = seen
        return r or None

    # ---- item 2
    def final_guard(self, irs: list[Path], final: dict, state: dict, cache, specs: dict, *, reverify=None,
                    kbeval=None, gpu_model: str | None = None, hints_doc: dict | None = None) -> dict | None:
        """After the final verification: flag dominant ops left on (or near) their seed; ``repair``: route them
        (zero LLM) and ``reverify(ir_dir)`` -> a new verify result. -> report (also printed loudly)."""
        if self.guard == "off":
            return None
        rep = {"mode": self.guard, "share": self.guard_share, "min_speedup": self.guard_speedup, "irs": {}}
        sent = set(state.get("sent") or [])
        for d in irs:
            name = Path(d).name
            v = final.get(name) or {}
            ops = v.get("ops") or []
            first = _first_profile(self.work, name)
            r = {"flagged": [], "ops_timed": bool(ops)}
            if not ops:
                r["note"] = "the final verification recorded no per-op times (failed run?)"
                rep["irs"][v.get("ir", name)] = r
                continue
            flagged = seed_left(ops, seed_times(first), share=self.guard_share, min_speedup=self.guard_speedup)
            fb_ops = _fallback_ops(self.work, name)
            vbad = {k.split("/")[0] for k, s in (v.get("slots") or {}).items() if not s.get("ok")}
            for f in flagged:
                rec = next((o for o in ops if o["name"] == f["name"]), {})
                keys = item_keys_of(rec, None, specs) if rec.get("entry") else []
                f["reason"] = classify(f, fallback_ops=fb_ops, verify_bad=vbad, sent=sent, keys=keys)
            r["flagged"] = flagged
            if flagged and self.guard == "repair" and reverify is not None:
                r["repair"] = self._repair(Path(d), v, flagged, cache, specs, kbeval, gpu_model, hints_doc,
                                           reverify)
                if r["repair"].get("verify") is not None:
                    final[name] = r["repair"].pop("verify")
            rep["irs"][v.get("ir", name)] = r
        n = sum(len(x["flagged"]) for x in rep["irs"].values())
        rep["n_flagged"] = n
        if n:
            for ir, x in rep["irs"].items():
                for f in x["flagged"]:
                    print(f"[kb_graph_opt] WARNING final-model guard: {ir}: {f['name']} ({f['op']}) is "
                          f"{100 * f['share']:.0f}% of the model and "
                          + ("still runs its seed kernel" if f["on_seed_kernel"] else
                             f"only {f['speedup_vs_seed']}x faster than its seed")
                          + f" -- {f['reason']}", flush=True)
        return rep

    def _repair(self, ir_dir: Path, v: dict, flagged: list[dict], cache, specs, kbeval, gpu_model, hints_doc,
                reverify) -> dict:
        from . import graph_hints as gh
        from .kb_export import export_problem
        from .kb_ingest import cache_plan
        from .kb_plan import hinted_ir
        from .kb_proposals import with_fused_specs, write_variant_ir

        work = self.work / "final_repair" / ir_dir.name
        out = {"routes": [], "before_us": v.get("partition_us")}
        try:
            _, sp, g = hinted_ir(ir_dir, hints_doc or gh.new_doc(), work / "hinted", dict(specs))
            sp = with_fused_specs(g, sp)
            arch = f"sm_{str(self.cfg.arch).removeprefix('sm_')}"
            base = cache_plan(g, cache, arch, sp)
            bdir = write_variant_ir(ir_dir, base["graph"], work / "base_ir")
            bprob = export_problem(bdir, work / "base_problem", slot_sources=base["sources"],
                                   slot_source_tags=base["tags"], sub_sources=base["sub_sources"], specs=sp,
                                   strategy="whole", concurrency="off")
            bman = json.loads((bprob / "manifest.json").read_text())
            src_of_slot = {s["id"]: (bprob / s["file"]).read_text() for s in bman["slots"]}
            inc = {r["name"]: src_of_slot[r["slot"]] for r in bman["ops"]}
            by_kind: dict[str, list] = {}
            for f in flagged:
                if f["op"] in sp and not f["op"].startswith("fused__"):
                    by_kind.setdefault(f["op"], []).append(f["name"])
            backup = json.loads(json.dumps(cache.entries))
            for kind, names in by_kind.items():
                r = route_ops(kind=kind, ops=names, graph=base["graph"], src_ir=ir_dir, cache=cache, specs=sp,
                              arch=arch, ir_name=bman["ir_name"], work=work / f"route_{kind}", kbeval=kbeval,
                              gpu_model=gpu_model, impl_dirs=self.impl_dirs(), incumbent=inc, tag=REPAIR_TAG,
                              why={"final_guard": names})
                r.pop("table", None)
                out["routes"].append(r)
            if any(r.get("result") == "routed" for r in out["routes"]):
                nv = reverify(ir_dir)
                out["after_us"] = nv.get("partition_us")
                if not nv.get("passed") or (out["before_us"] and nv.get("partition_us")
                                            and nv["partition_us"] > out["before_us"]):
                    cache.entries[:] = backup                 # keep the verified state
                    cache.save()
                    out["reverted"] = "the repaired model failed or was slower"
                    out["verify"] = reverify(ir_dir)
                else:
                    out["verify"] = nv
        except Exception as e:  # noqa: BLE001 -- the guard never breaks the run's result
            out["error"] = f"{type(e).__name__}: {e}"[:800]
        return out

    def summary(self) -> dict:
        return {"fallback_repair": self.fallback, "final_guard": self.guard, "seed_nearest": self.nearest,
                **({"rollout_resync": self.resync} if self.resync else {}),
                "requeued": self.reports["requeued"], "routes": [
                    {k: x.get(k) for k in ("ir", "attempt", "kind", "ops", "result", "written")}
                    for x in self.route_log],
                **({"nearest_seeds": self.reports["nearest_seeds"]} if self.reports["nearest_seeds"] else {})}


def _profiles(work: Path, name: str) -> list[Path]:
    return sorted(Path(work).glob(f"r[0-9][0-9]/profile/{name}/profile.json"))


def _first_profile(work: Path, name: str) -> dict | None:
    ps = _profiles(work, name)
    return json.loads(ps[0].read_text()) if ps else None


def _fallback_ops(work: Path, name: str) -> set:
    out = set()
    for p in _profiles(work, name):
        d = json.loads(p.read_text())
        for step in d.get("cache_fallback") or []:
            for names in (step.get("ops") or {}).values():
                out.update(names)
    return out


# ---------------------------------------------------------------- nearest-shape seeds (item 3)

def add_nearest_seed_candidates(problem_dir: Path, cache, *, arch: str, specs: dict, impl_dirs=None,
                                n: int = 2) -> dict:
    """Trial seeds from the verified kernels of the NEAREST shapes (same op + signature) for every slot of a freshly
    exported problem that is not seeded from its own cached winner: prepended to ``manifest["seed_candidates"]``
    (A1's ``kb_reuse.add_seed_candidates`` path; duplicates by code dropped). KernelBlaster tries each at start-up
    and keeps one only if it passes this problem's checks and is >= 2% faster, and the seeded composite is not
    slower end to end. -> {slot id: [(from, dist), ...]}."""
    problem_dir = Path(problem_dir)
    mf = problem_dir / "manifest.json"
    man = json.loads(mf.read_text())
    out = {}
    for s in man["slots"]:
        if str(s.get("seed_source", "")).startswith("cache:"):
            continue
        targets = [i.get("shape", {}) for i in s.get("instances", [])]
        cur = (problem_dir / s["file"]).read_text()
        have = man.get("seed_candidates", {}).get(s["id"], [])
        have_sha = {_sha((problem_dir / c["file"]).read_text()) for c in have if (problem_dir / c["file"]).exists()}
        cands = [c for c in nearest_pool(s["op"], cache, arch=arch, specs=specs, ir_name=man.get("ir_name"),
                                         targets=targets, impl_dirs=impl_dirs, exclude={_sha(cur)} | have_sha,
                                         max_cands=n)]
        if not cands:
            continue
        lst = []
        for k, c in enumerate(cands):
            f = f"slots/{s['id']}.near{k}.cu"
            (problem_dir / f).write_text(c["code"])
            lst.append({"file": f, "from": c["from"], "dist": c["dist"], "by": "nearest-shape"})
        man.setdefault("seed_candidates", {})[s["id"]] = lst + have
        out[s["id"]] = [(c["from"], c["dist"]) for c in cands]
    if out:
        mf.write_text(json.dumps(man, indent=1))
    return out
