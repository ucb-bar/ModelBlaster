"""Unapply audit of pre-applied fusions (``--graph-priors preapply|standard``; l3_farm/analysis/preapply_audit.md).

A hint pre-applied from a prior (``graph_search.GraphSearch.preapply``: accepted without a measurement) is never
checked afterwards: no accept / reject evidence, no way back. This module is that way back, at zero LLM cost:

1. **Record** (at pre-apply time, :func:`record_preapplied`): the hint, its stage ops (name, kind, shape) and the
   pre-fusion graph, in ``<work>/preapply_audit/preapplied.json`` + ``pre/<hint>__<ir>.graph.json``.
2. **Alternative graphs** (:func:`run_audit`): the INCUMBENT graph (the IR with every accepted hint applied, exactly what
   ``kb_model_run.model_runtime`` composes) and, per pre-applied hint, the incumbent with that hint reverted by an
   ``unfuse`` hint (``graph_hints._apply_unfuse_cuda``: the fused ops of the hint become their stages again). The
   reverted graph is checked to equal what the final model would deploy once the hint is rejected (doc revert).
3. **Stage kernels**: ``kb_ingest.cache_plan`` picks the best verified kernel per slot (cache winners by op / shape, fused
   slots seeded from cached stage kernels, hand seeds otherwise). Stage ops without a cached winner first go through
   A3's ``search_fixes.route_ops`` (A1's ``kb_reuse``: verified kernels of the nearest other shapes, built on exactly
   these instances, kept only if they pass and beat the seed; ``--graph-preapply-route off`` skips it).
4. **Verify + time**: every graph is exported as a whole-model problem and built; the driver's golden check must pass
   (a run that does not pass is not timed). The incumbent and all alternatives are timed back to back, round-robin
   (``kb_proposals.remeasure`` through ``kb_graph_opt._remeasure_once``: CUDA-graph replay, the leased device or the
   KernelBlaster pool, as the final model run), ``--remeasure`` reps each.
5. **Decide** (:func:`decide`): revert iff the alternative is faster by at least the margin (``--graph-preapply-margin``,
   default the flow's ``--accept-gain`` 3%) and strictly faster; a tie or a slower alternative keeps the fusion.
   Several pre-applied hints: each is audited independently against the incumbent, the reverts that pass are applied
   greedily together and timed once more against the incumbent; if the combination does not beat the incumbent by the
   margin only the best single revert is applied. With several IRs the decision is on the summed time and no IR may
   get worse by more than the margin.
6. **Learn**: every result is a ``graph_decisions.jsonl`` decision (phase ``preapply_audit``, measured), hint evidence
   (``by: preapply-audit``, original = unfused, variant = fused, like a measured fusion decision), and a decision row in
   ``state.json`` ``rounds[-1].decisions`` -- where the structural statistics (``knowledge.v2.priors.rows_from_state``)
   read their outcomes. A revert sets the hint ``rejected``; the final verification and model run then deploy the graph
   without it.

Never blocks the run: any failure is recorded (``audit_error``) and the incumbent kept.
"""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import time
from pathlib import Path

from . import graph_hints as gh

AUDIT_BY = "preapply-audit"
DIRNAME = "preapply_audit"
ROUTE_TAG = "mb-preapply-audit-route"


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def _median(xs: list) -> float | None:
    xs = sorted(x for x in xs if x)
    return xs[len(xs) // 2] if len(xs) % 2 else (0.5 * (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) if xs else None)


# ---------------------------------------------------------------- 1. record at pre-apply time

def _stage_ops(graph: dict, chains: list[list[str]]) -> list[dict]:
    """Name / kind / shape of the ops (stages) the chains cover in ``graph``."""
    by = {a: o for o in graph["ops"] for a in gh.atoms(o["name"])}
    out = []
    for ch in chains:
        for a in ch:
            o = by.get(a)
            if o is None:
                continue
            st = next((s for s in o.get("sub_ops", []) if s.get("name") == a), o)
            out.append({"name": a, "op": st.get("op"), "shape": st.get("shape")})
    return out


def records_path(work: Path) -> Path:
    return Path(work) / DIRNAME / "preapplied.json"


def load_records(work: Path) -> dict:
    try:
        return json.loads(records_path(work).read_text())
    except (OSError, ValueError):
        return {}


def record_preapplied(work: Path, r: int, h: dict, ir: str | None, prior: dict | None, graphs: dict) -> dict:
    """What the audit needs to rebuild the graph without hint ``h`` (see the module doc)."""
    from .kb_resume import atomic_write_json
    d = Path(work) / DIRNAME
    (d / "pre").mkdir(parents=True, exist_ok=True)
    chains = [list(c) for c in (h.get("chains") or h.get("tried_chains") or [])]
    snaps, stages = {}, {}
    for irn, g in graphs.items():
        present = {a for o in g["ops"] for a in gh.atoms(o["name"])}
        if chains and not any(a in present for ch in chains for a in ch):
            continue
        f = d / "pre" / f"{_safe(h['id'])}__{_safe(irn)}.graph.json"
        f.write_text(json.dumps(g))
        snaps[irn] = str(f)
        stages[irn] = _stage_ops(g, chains)
    rec = {"hint": h["id"], "round": r, "ir": ir, "fused_kind": h.get("fused_kind"), "chains": chains,
           "class": (prior or {}).get("class"), "prior": prior, "pre_graph": snaps, "stage_ops": stages,
           "ts": gh._now()}
    allr = load_records(work)
    allr[h["id"]] = rec
    atomic_write_json(records_path(work), allr)
    return rec


# ---------------------------------------------------------------- 2. which hints, which graphs

def pending_ids(doc: dict) -> list[str]:
    """Accepted hints whose acceptance came from a prior without a measurement (``by: graph-prior``, ``measured`` False)
    and that no audit has looked at yet."""
    out = []
    for h in doc["hints"]:
        if h.get("status") != "accepted":
            continue
        ev = h.get("evidence") or []
        if any(e.get("by") == "graph-prior" and e.get("measured") is False and e.get("decision") == "accepted"
               for e in ev) and not any(e.get("by") == AUDIT_BY for e in ev):
            out.append(h["id"])
    return out


def unfuse(inc_graph: dict, fused_ops: list[str], specs: dict, hid: str) -> tuple[dict, dict]:
    """The incumbent graph with the fused ops ``fused_ops`` restored to their stages (the ``unfuse`` hint kind)."""
    h = {"id": f"unfuse:{hid}", "kind": "unfuse", "target": "cuda", "ops": list(fused_ops)}
    g, _, prov = gh.apply_hints(inc_graph, [h], dict(specs), statuses=None)
    return g, prov


def graph_sig(g: dict) -> list:
    return sorted((o["name"], o["op"]) for o in g["ops"])


def deployed_match(g0: dict, doc: dict, hid: str, alt: dict, specs: dict):
    """True when the alternative equals the graph the final model deploys once ``hid`` is rejected (the doc with the hint
    reverted, applied to the original IR); else a message."""
    hints = copy.deepcopy(doc["hints"])
    for h in hints:
        if h["id"] == hid:
            h["status"] = "rejected"
    g2, _, _ = gh.apply_hints(g0, hints, dict(specs))
    a, b = graph_sig(g2), graph_sig(alt)
    if a == b:
        return True
    return f"alternative differs from the deployed revert: only in alt {[x for x in b if x not in a][:4]}, " \
           f"only in deployed {[x for x in a if x not in b][:4]}"


# ---------------------------------------------------------------- 3. decision rule (pure)

def decide(inc_us: float | None, alt_us: float | None, margin: float) -> dict:
    """Revert iff the alternative (unfused) is faster than the incumbent (fused) by at least ``margin`` (and strictly):
    ``gain_revert`` = (incumbent - alternative) / incumbent. Tie / slower alternative: keep."""
    if not inc_us or not alt_us or inc_us <= 0 or alt_us <= 0:
        return {"verdict": "error", "reason": "missing time", "gain_revert": None}
    g = (inc_us - alt_us) / inc_us
    if g > 0 and g >= margin:
        return {"verdict": "revert", "reason": "alt_faster", "gain_revert": g}
    return {"verdict": "keep", "reason": "fusion_faster" if g <= -margin else "tie", "gain_revert": g}


def decide_multi(pairs: list[tuple[float, float]], margin: float) -> dict:
    """Several IRs: the decision on the summed times; no IR may get worse by more than the margin."""
    pairs = [(i, a) for i, a in pairs if i and a]
    if not pairs:
        return {"verdict": "error", "reason": "missing time", "gain_revert": None}
    if len(pairs) == 1:
        return decide(pairs[0][0], pairs[0][1], margin)
    d = decide(sum(i for i, _ in pairs), sum(a for _, a in pairs), margin)
    if d["verdict"] == "revert" and min((i - a) / i for i, a in pairs) <= -margin:
        return {**d, "verdict": "keep", "reason": "slower_in_an_ir"}
    return d


# ---------------------------------------------------------------- 4. build, route, time (GPU side; injectable)

def compose_model(ir_dir: Path, graph: dict, specs: dict, cache, work: Path, arch: str) -> Path:
    """The whole-model problem of ``graph`` as ``kb_model_run.model_runtime`` composes it: cache winners per slot (fused
    slots seeded from cached stage kernels), hand seeds otherwise."""
    from .kb_export import export_problem
    from .kb_ingest import cache_plan
    from .kb_proposals import write_variant_ir
    plan = cache_plan(graph, cache, f"sm_{str(arch).removeprefix('sm_')}", specs)
    vir = write_variant_ir(Path(ir_dir), plan["graph"], Path(work) / "ir")
    return export_problem(vir, Path(work) / "model", slot_sources=plan["sources"], slot_source_tags=plan["tags"],
                          sub_sources=plan["sub_sources"], specs=specs, strategy="whole")


def route_stage_ops(alt_graph: dict, stage_names: set, *, ir_dir: Path, cache, specs: dict, arch: str, work: Path,
                    kbeval=None, gpu_model: str | None = None, min_gain: float = 0.03, max_cands: int = 4) -> list[dict]:
    """Zero-LLM nearest-shape routing (A3 ``search_fixes.route_ops``) for the stage ops of the alternative that no cached
    winner covers: per kind, the verified kernels of other shapes are tried on exactly these instances and the fastest
    passing one that beats the seed becomes a ``reuse`` cache entry. -> one report per kind."""
    from . import search_fixes as sfx
    from .kb_export import ALIAS_OPS, slot_id_of
    from .kb_ingest import cache_plan
    from .kb_proposals import with_fused_specs
    plan = cache_plan(alt_graph, cache, f"sm_{str(arch).removeprefix('sm_')}", specs)
    g = plan["graph"]
    sp = with_fused_specs(g, specs)
    by_kind: dict[str, list[str]] = {}
    for o in g["ops"]:
        if o["name"] in stage_names and o["op"] in sp and o["op"] not in ALIAS_OPS and slot_id_of(o, sp) not in plan["used"]:
            by_kind.setdefault(o["op"], []).append(o["name"])
    out = []
    for kind, ops in by_kind.items():
        r = sfx.route_ops(kind=kind, ops=ops, graph=g, src_ir=Path(ir_dir), cache=cache, specs=sp, arch=arch,
                          ir_name=g["name"], work=Path(work) / f"route_{_safe(kind)}", kbeval=kbeval,
                          gpu_model=gpu_model, max_cands=max_cands, min_gain=min_gain, tag=ROUTE_TAG)
        r.pop("table", None)
        out.append(r)
    return out


def default_timer(cfg, leases, recorded: list):
    """-> timer(problems, work) -> (times per problem, info): the flow's own re-measurement
    (``kb_graph_opt._remeasure_once``: interleaved round-robin, ``cfg.remeasure`` reps, CUDA-graph replay, leased device
    or KernelBlaster pool); an infrastructure failure is retried like a hint's re-measurement."""
    def timer(problems: list, work: Path):
        from . import kb_graph_opt as go
        from .kb_eval import required_hardware
        from .kb_proposals import is_infra_text
        ev = go.kb_eval_for(cfg) if go.use_kb_eval(cfg) else None
        hw = None if ev is None else required_hardware(list(recorded), gpu_model=cfg.gpu_model, arch=cfg.arch,
                                                       allow_mismatch=cfg.allow_gpu_mismatch)
        pair = [(Path(p), {}) for p in problems]
        times, info = None, {}
        for a in range(max(1, int(getattr(cfg, "remeasure_attempts", 1)))):
            if a:
                time.sleep(float(getattr(cfg, "remeasure_retry_s", 0.0)) * a)
            info = {}
            try:
                times = go._remeasure_once(pair, cfg, leases, ev, hw, info, Path(work) / (f"a{a}" if a else "."))
            except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as e:
                times, info["failures"] = None, [{"error": f"{type(e).__name__}: {e}"[:800], "infra": True}]
            if times and times[0]:
                break
            if not all(f.get("infra") or is_infra_text(str(f.get("error"))) for f in info.get("failures") or [{}]):
                break
        return times, info
    return timer


# ---------------------------------------------------------------- 5. the audit

def _paired(inc: list, alt: list) -> float | None:
    r = [a / i for i, a in zip(inc, alt) if i and a]
    return _median(r)


def _time(timer, problems: list, work: Path):
    out = timer(problems, work)
    return out if isinstance(out, tuple) else (out, {})


def run_audit(*, irs: list, doc: dict, ids: list[str], specs: dict, cache, cfg, work: Path, margin: float,
              timer, compose=compose_model, router=None, arch: str | None = None) -> dict:
    """Audit the pre-applied hints ``ids`` of ``doc`` (read only: ``doc`` is not changed). ``timer(problems, work)`` ->
    (times per problem [list of us per rep, empty = no passing run], info); ``compose(ir_dir, graph, specs, cache, work,
    arch)`` -> problem dir; ``router(alt_graph, stage_names, ir_dir, specs, work)`` -> route reports (None = no routing).
    -> {"margin", "hints": {id: {"verdict", "reason", ..., "irs": {...}}}, "reverted": [...], "kept": [...],
        "skipped": [...], "combined": {...}, "ir_errors": {...}}"""
    arch = str(arch or cfg.arch).removeprefix("sm_")
    work = Path(work)
    rep: dict = {"margin": margin, "hints": {hid: {"hint": hid, "irs": {}} for hid in ids}, "ir_errors": {},
                 "combined": None}
    ctx: dict[str, dict] = {}
    for ir_dir in irs:
        name = Path(ir_dir).name
        try:
            c = _audit_ir(Path(ir_dir), name, doc, ids, specs, cache, work / _safe(name), arch, compose, timer, router,
                          rep)
            if c:
                ctx[name] = c
        except Exception as e:  # noqa: BLE001 -- recorded, the incumbent stays
            rep["ir_errors"][name] = f"{type(e).__name__}: {e}"[:600]
    # per-hint verdict over the IRs
    reverts = []
    for hid, hr in rep["hints"].items():
        timed = [(n, x) for n, x in hr["irs"].items() if x.get("status") == "timed"]
        if timed:
            d = decide_multi([(x["inc_us"], x["alt_us"]) for _, x in timed], margin)
            inc, alt = sum(x["inc_us"] for _, x in timed), sum(x["alt_us"] for _, x in timed)
            hr.update(d, inc_us=inc, alt_us=alt, ir_count=len(timed))
            if d["verdict"] == "revert":
                reverts.append(hid)
            continue
        bad = [x for x in hr["irs"].values() if x.get("status") == "alt_failed"]
        errs = [x for x in hr["irs"].values() if x.get("status") == "audit_error"]
        if bad:
            hr.update(verdict="keep", reason="alt_failed", gain_revert=None,
                      why="; ".join(str(x.get("why")) for x in bad)[:600])
        elif errs or any(n in rep["ir_errors"] for n in [Path(i).name for i in irs]):
            hr.update(verdict="error", reason="audit_error", gain_revert=None,
                      why="; ".join([str(x.get("why")) for x in errs] + list(rep["ir_errors"].values()))[:600])
        else:
            hr.update(verdict="skip", reason="not_applicable", gain_revert=None)
    if len(reverts) > 1:
        reverts = _combine(rep, ctx, reverts, work, arch, compose, timer, margin, specs, cache)
    for hid in rep["hints"]:
        hr = rep["hints"][hid]
        if hr["verdict"] == "revert" and hid not in reverts:
            hr.update(verdict="keep", reason="combined_interaction")
    rep["reverted"] = [h for h in ids if rep["hints"][h]["verdict"] == "revert"]
    rep["kept"] = [h for h in ids if rep["hints"][h]["verdict"] == "keep"]
    rep["skipped"] = [h for h in ids if rep["hints"][h]["verdict"] in ("skip", "error")]
    return rep


def _audit_ir(ir_dir: Path, name: str, doc: dict, ids: list[str], specs: dict, cache, work: Path, arch: str, compose,
              timer, router, rep: dict) -> dict | None:
    from .kb_proposals import _all_specs
    g0 = json.loads((ir_dir / "graph.json").read_text())
    sp = dict(specs or _all_specs())
    inc_g, added, _ = gh.apply_hints(g0, doc["hints"], sp)
    sp.update(added)
    alts: dict[str, dict] = {}
    for hid in ids:
        r = rep["hints"][hid]["irs"].setdefault(name, {})
        fused = [o["name"] for o in inc_g["ops"] if o.get("hint") == hid and "sub_ops" in o]
        if not fused:
            r.update(status="not_applicable", why="no op of this hint in the incumbent graph (absorbed by a longer "
                                                  "chain's hint, or the pattern does not occur in this IR)")
            continue
        alt_g, prov = unfuse(inc_g, fused, sp, hid)
        if not prov["applied"]:
            r.update(status="audit_error", why="unfuse did not apply: " + json.dumps(prov["skipped"])[:300])
            continue
        chk = deployed_match(g0, doc, hid, alt_g, sp)
        if chk is not True:
            r.update(status="audit_error", why=chk)
            continue
        stages = [n for a in prov["applied"] for n in a["to"]]
        r.update(fused_ops=fused, stage_ops=stages)
        alts[hid] = {"graph": alt_g, "stages": set(stages)}
    if not alts:
        return None
    if router is not None:                      # route first (writes the cache), then compose all against one cache
        for hid, a in alts.items():
            try:
                rr = router(a["graph"], a["stages"], ir_dir, sp, work / f"route_{_safe(hid)}")
                rep["hints"][hid]["irs"][name]["route"] = rr
            except Exception as e:  # noqa: BLE001 -- routing is an improvement, never a requirement
                rep["hints"][hid]["irs"][name]["route"] = {"error": f"{type(e).__name__}: {e}"[:400]}
    inc_p = compose(ir_dir, inc_g, sp, cache, work / "inc", arch)
    order, probs = [], [inc_p]
    for k, (hid, a) in enumerate(alts.items()):
        try:
            probs.append(compose(ir_dir, a["graph"], sp, cache, work / f"alt{k}", arch))
            order.append(hid)
        except Exception as e:  # noqa: BLE001
            rep["hints"][hid]["irs"][name].update(status="audit_error",
                                                  why=f"alternative export failed: {type(e).__name__}: {e}"[:500])
    if not order:
        return None
    times, info = _time(timer, probs, work / "time")
    if not times or not times[0]:
        raise RuntimeError("incumbent: no passing timed run (" + json.dumps(info.get("failures"))[:400] + ")")
    inc_t = times[0]
    for k, hid in enumerate(order, 1):
        r = rep["hints"][hid]["irs"][name]
        alt_t = times[k] if k < len(times) else []
        if not alt_t:
            fl = [f for f in (info.get("failures") or []) if str(f.get("problem")) == str(probs[k])]
            r.update(status="alt_failed", inc_us=_median(inc_t),
                     why="the alternative has no passing timed run (verification failed or the build): " +
                         (json.dumps(fl)[:400] if fl else "driver check"))
            continue
        r.update(status="timed", inc_us=_median(inc_t), alt_us=_median(alt_t), inc_runs=inc_t, alt_runs=alt_t,
                 paired_ratio=_paired(inc_t, alt_t), device=info.get("device"))
    return {"ir_dir": ir_dir, "inc_g": inc_g, "sp": sp, "alts": alts, "probs": probs, "order": order, "inc_t": inc_t}


def _combine(rep: dict, ctx: dict, reverts: list[str], work: Path, arch: str, compose, timer, margin: float,
             specs: dict, cache) -> list[str]:
    """Greedy combination: all reverts that passed, applied together and timed once against the incumbent; accepted if
    the combination is faster by the margin, else only the best single revert stays."""
    best = max(reverts, key=lambda h: rep["hints"][h]["gain_revert"])
    comb = {"hints": list(reverts), "irs": {}, "best_single": best}
    pairs = []
    try:
        for name, c in ctx.items():
            hs = [h for h in reverts if h in c["order"] and rep["hints"][h]["irs"][name].get("status") == "timed"]
            if not hs:
                continue
            if len(hs) == 1:                      # nothing combined in this IR: its single measurement stands
                x = rep["hints"][hs[0]]["irs"][name]
                pairs.append((x["inc_us"], x["alt_us"]))
                comb["irs"][name] = {"hints": hs, "inc_us": x["inc_us"], "alt_us": x["alt_us"], "reused": True}
                continue
            ops = [o for h in hs for o in rep["hints"][h]["irs"][name]["fused_ops"]]
            g, prov = unfuse(c["inc_g"], ops, c["sp"], "+".join(hs))
            p = compose(c["ir_dir"], g, c["sp"], cache, work / _safe(name) / "comb", arch)
            times, info = _time(timer, [c["probs"][0], p], work / _safe(name) / "time_comb")
            if not times or not times[0] or len(times) < 2 or not times[1]:
                raise RuntimeError("combined alternative: no passing timed run")
            inc, alt = _median(times[0]), _median(times[1])
            pairs.append((inc, alt))
            comb["irs"][name] = {"hints": hs, "inc_us": inc, "alt_us": alt, "inc_runs": times[0], "alt_runs": times[1]}
        d = decide_multi(pairs, margin)
        comb.update(d)
    except Exception as e:  # noqa: BLE001
        comb.update(verdict="error", reason=f"{type(e).__name__}: {e}"[:400])
    rep["combined"] = comb
    return list(reverts) if comb.get("verdict") == "revert" else [best]


# ---------------------------------------------------------------- 6. commit: doc, events, state

def _fused_gain(inc_us, alt_us):
    """The flow's fusion gain: (original - variant) / original with original = unfused, variant = fused."""
    return (alt_us - inc_us) / alt_us if inc_us and alt_us else None


def commit(gs, doc: dict, rep: dict, *, phase: str) -> list[dict]:
    """Write the verdicts: hint evidence (a revert rejects the hint), graph decisions, ... -> decision rows for
    ``state.json`` (same keys as a measured hint decision)."""
    rows = []
    em = gs.exec_mode("whole")
    for hid, hr in rep["hints"].items():
        v = hr["verdict"]
        if v in ("skip", "error"):
            gs.event("audit_error" if v == "error" else "audit_skip", phase=phase, hint=hid, reason=hr.get("reason"),
                     why=hr.get("why"), irs=_brief(hr))
            continue
        reverted = v == "revert"
        inc, alt = hr.get("inc_us"), hr.get("alt_us")
        measured = reverted or hr.get("reason") in ("tie", "fusion_faster", "slower_in_an_ir", "combined_interaction")
        ev = {"decision": ("rejected" if reverted else "accepted") if measured else "undecided",
              "by": AUDIT_BY, "phase": phase, "measured": bool(measured), "round": gs.round,
              "ts": gh._now(), "problem": "whole-model", "original_us": alt, "variant_us": inc,
              "gain": _fused_gain(inc, alt), "audit_gain_revert": hr.get("gain_revert"), "margin": rep["margin"],
              "verdict": v, "reason": hr.get("reason"), "irs": _brief(hr),
              **({"why": hr["why"]} if hr.get("why") else {}), "exec_mode": em}
        if reverted and rep.get("combined"):
            ev["combined"] = {k: rep["combined"].get(k) for k in ("verdict", "reason", "gain_revert", "hints", "best_single")}
        h = gh.record_decision(doc, hid, ev)
        if reverted:
            h["status"] = "rejected"                  # a revert wins over earlier votes
        status = h["status"]
        ops = [o for x in hr["irs"].values() for o in x.get("fused_ops") or []]
        gs.event("decision", phase="preapply_audit", hint=hid, status=("rejected" if reverted else
                                                                          "accepted" if measured else "undecided"),
                 measured=bool(measured), evidence=ev, exec_mode=em, steps=0, llm_calls=0, usd=0.0,
                 verdict=v, gain=ev["gain"], original_us=alt, variant_us=inc)
        rows.append({"hint": hid, "status": "rejected" if reverted else "accepted" if measured else "undecided",
                     "hint_status": status, "gain": ev["gain"], "original_us": alt, "variant_us": inc, "ops": ops,
                     "phase": "preapply_audit", "measured": bool(measured), "verdict": v})
    return rows


def _brief(hr: dict) -> dict:
    return {n: {k: x.get(k) for k in ("status", "inc_us", "alt_us", "paired_ratio", "why", "device") if k in x}
            | ({"route": [{k: r.get(k) for k in ("kind", "result", "written", "error")} for r in x["route"]]
                if isinstance(x.get("route"), list) else x["route"]} if "route" in x else {})
            for n, x in hr["irs"].items()}


def audit_hook(gs, phase: str, *, irs: list, doc: dict, hints_path: Path, cache, specs: dict, cfg, leases,
               recorded: list, state: dict | None, state_path: Path | None, compose=compose_model, timer=None,
               router="default") -> dict | None:
    """The hook behind ``GraphSearch.audit`` (see there). ``timer`` / ``router`` / ``compose`` are injectable for tests."""
    from .kb_resume import atomic_write_json
    ids = pending_ids(doc)
    if not ids:
        return None
    t0 = time.time()
    gs.event("audit_start", phase=phase, hints=ids, margin=gs.audit_margin, route=gs.audit_route)
    if not int(getattr(cfg, "remeasure", 0) or 0) and timer is None:
        rep = {"margin": gs.audit_margin, "hints": {}, "reverted": [], "kept": [], "skipped": ids, "combined": None,
               "audit_error": "--remeasure 0: no re-measurement, the audit cannot time"}
        gs.event("audit_error", phase=phase, error=rep["audit_error"])
        rec = {"phase": phase, **rep}
        gs.audit_results.append(rec)
        return rec
    tag = phase if phase != "round" or gs.round is None else f"round{int(gs.round):02d}"
    work = gs.work / DIRNAME / tag
    if timer is None:
        timer = default_timer(cfg, leases, recorded)
    if router == "default":
        router = None
        if gs.audit_route:
            def router(alt_g, stages, ir_dir, sp, w):         # noqa: E306
                from . import kb_graph_opt as go
                ev = go.kb_eval_for(cfg) if go.use_kb_eval(cfg) else None
                return route_stage_ops(alt_g, stages, ir_dir=ir_dir, cache=cache, specs=sp, arch=str(cfg.arch), work=w,
                                       kbeval=ev, gpu_model=getattr(cfg, "gpu_model", None),
                                       min_gain=float(getattr(cfg, "accept_gain", 0.03)))
    try:
        rep = run_audit(irs=irs, doc=doc, ids=ids, specs=specs, cache=cache, cfg=cfg, work=work, margin=gs.audit_margin,
                        timer=timer, compose=compose, router=router)
    except Exception as e:  # noqa: BLE001 -- never blocks the run
        rep = {"margin": gs.audit_margin, "hints": {}, "reverted": [], "kept": [], "skipped": ids, "combined": None,
               "audit_error": f"{type(e).__name__}: {e}"[:600]}
        gs.event("audit_error", phase=phase, error=rep["audit_error"])
    rows = commit(gs, doc, rep, phase=phase) if rep["hints"] else []
    if rows or rep.get("reverted"):
        gh.save(doc, hints_path)
    if rows and state is not None and state.get("rounds"):
        last = state["rounds"][-1]
        last.setdefault("decisions", []).extend(rows)
        if state_path is not None:
            atomic_write_json(state_path, state)
            rj = gs.work / f"r{int(last.get('round', len(state['rounds']) - 1)):02d}" / "round.json"
            if rj.exists():
                try:
                    atomic_write_json(rj, last)
                except OSError:
                    pass
    out = {"phase": phase, "seconds": round(time.time() - t0, 1), **{k: v for k, v in rep.items() if k != "hints"},
           "hints": {h: {k: v for k, v in hr.items() if k != "irs"} | {"irs": _brief(hr)}
                     for h, hr in rep["hints"].items()}, "decisions": rows}
    gs.audit_results.append(out)
    gs.event("audit_end", phase=phase, reverted=rep.get("reverted"), kept=rep.get("kept"),
             skipped=rep.get("skipped"), seconds=out["seconds"], error=rep.get("audit_error"))
    try:
        (work).mkdir(parents=True, exist_ok=True)
        atomic_write_json(work / "audit.json", out)
    except OSError:
        pass
    return out
