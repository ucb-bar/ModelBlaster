"""ModelBlaster as the graph optimiser: KernelBlaster rounds to a fixpoint.

One round, for a set of IRs sharing a work dir (hints, kernel cache and
KernelBlaster knowledge carry over between rounds and runs):

1. **profile** every IR in its current state -- accepted graph hints applied,
   cached winners in place (``kb_plan.profile_ir``);
2. **MB proposals** from the whole-model profiles
   (``graph_hints.propose_from_profile``: elementwise consumers worth fusing,
   including consumers of already-fused ops, so chains grow);
3. **plan** -- rank op instances by (entry, signature, shape) across the IRs,
   select under the budget, skipping items sent in earlier rounds
   (``kb_plan``); export the problems seeded from the cache, run
   KernelBlaster on them, ingest its winners; its ``proposals.json`` (fuse,
   split) become hints;
4. **decide** -- for the undecided hints with the largest estimated gain
   (at most ``max_variants``): export the affected ops as an *original*
   problem and, with the hint applied, as a *variant* problem, both seeded
   with the same current-best kernels; KernelBlaster optimises both with the
   same budget; both composite bests are rebuilt and re-timed back to back
   (``kb_proposals.remeasure``); the hint is accepted if the variant is
   ``accept_gain`` faster. Accepted hints apply from the next round on, and
   the variant's new slots (the fused kernel, the split variants) go into the
   cache;
5. stop when a round accepts nothing and has nothing new to send (the
   fixpoint), or after ``rounds``.

Finally every IR is verified at model level with everything applied
(``kb_ingest.verify_ir``) and ``summary.json`` lists first vs final times.

    python -m modelblaster.pipeline.kb_graph_opt IR_DIR [...] --work W --kb KB_CHECKOUT \\
        [--model claude-code:sonnet] [--plugin stubs.py] [--rounds 4] \\
        [--coverage 0.8 --top-k 6] [--rollouts 2 --steps 4 --variant-steps 3] \\
        [--max-variants 3 --remeasure 5 --accept-gain 0.03] [--arch 75]

Work dir: ``graph_hints.json`` (modelblaster.graph_hints/v1), ``cache/``
(mb-cuda-cache/1), ``knowledge/`` (KernelBlaster), ``state.json``,
``rNN/{profile,plan.json,problems,run,winners,variants,round.json}``,
``summary.json``.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import graph_hints as gh
from .kb_export import base_slot_id, export_problem
from .kb_ingest import CudaCache, cache_plan, ingest, verify_ir
from .kb_plan import export_plan, make_plan, profile_ir
from .kb_proposals import (ACCEPT_GAIN, _all_specs, _median, best_sources, find_chains, remeasure,
                           with_fused_specs, write_variant_ir)
from .kernel_abi import parse_signature


@dataclass
class Config:
    kb: Path
    model: str = "claude-code:sonnet"
    reflect_model: str | None = None
    plugins: list = field(default_factory=list)
    python: str = sys.executable
    arch: str = "75"
    rounds: int = 4
    rollouts: int = 2
    steps: int = 4
    variant_steps: int | None = None
    max_fix: int = 2
    coverage: float = 0.8
    top_k: int | None = 6
    min_share: float = 0.01
    min_gain_pct: float = 2.0
    max_variants: int = 3
    remeasure: int = 5
    accept_gain: float = ACCEPT_GAIN
    profile_runs: int = 3
    seed: int = 0


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:80]


class KBRunner:
    """KernelBlaster's ``scripts/run_multislot.py`` as a subprocess."""

    def __init__(self, cfg: Config, knowledge: Path):
        self.cfg, self.knowledge = cfg, knowledge
        self.calls = 0

    def run(self, problems: list[Path], out: Path, *, steps: int | None = None,
            winners: Path | None = None) -> list[dict]:
        c = self.cfg
        cmd = [c.python, str(Path(c.kb) / "scripts" / "run_multislot.py"), *map(str, problems), "--out", str(out),
               "--knowledge-dir", str(self.knowledge), "--model", c.model, "--arch", c.arch,
               "--rollouts", str(c.rollouts), "--steps", str(steps if steps is not None else c.steps),
               "--max-fix", str(c.max_fix), "--seed", str(c.seed)]
        if c.reflect_model:
            cmd += ["--reflect-model", c.reflect_model]
        if winners is not None:
            cmd += ["--export-winners", str(winners)]
        for p in c.plugins:
            cmd += ["--plugin", str(p)]
        out.mkdir(parents=True, exist_ok=True)
        with open(out.with_suffix(".log"), "w") as log:
            cp = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(c.kb))
        if cp.returncode != 0:
            raise RuntimeError(f"KernelBlaster failed ({cp.returncode}); see {out.with_suffix('.log')}")
        res = []
        for d in problems:
            name = json.loads((Path(d) / "manifest.json").read_text())["name"]
            f = out / name / "result.json"
            r = json.loads(f.read_text()) if f.exists() else {"problem": name, "final_ok": False}
            r["_dir"] = str(out / name)
            self.calls += r.get("llm_calls", 0) or 0
            res.append(r)
        return res


def _usage(results: list[dict]) -> dict:
    tot = {"llm_calls": 0, "cost_usd": 0.0}
    for r in results:
        tot["llm_calls"] += r.get("llm_calls", 0) or 0
        tot["cost_usd"] += (r.get("llm_usage") or {}).get("cost_usd", 0.0) or 0.0
    return tot


def _est_gain(h: dict) -> float:
    return max((s.get("est_gain_pct") or 0.0) for s in h.get("source", [{}]) or [{}])


def _hint_ir(h: dict, profiles: list[dict]) -> dict | None:
    irs = [s.get("ir") for s in h.get("source", []) if s.get("ir")]
    return next((p for ir in irs for p in profiles if p["ir"] == ir), None)


def _scope_ops(h: dict, graph: dict, specs: dict) -> list[str]:
    """Current op names a hint touches in ``graph``."""
    if h["kind"] == "fuse":
        chains = h.get("chains") or find_chains(graph, h["fused_kind"], specs)
        names = []
        for ch in chains:
            try:
                names += gh._locate(graph, ch)
            except ValueError:
                continue
        return list(dict.fromkeys(names))
    if h["kind"] == "split_slot":
        return [o["name"] for o in graph["ops"] if o["op"] in specs and base_slot_id(o["op"], specs) == h["slot"]]
    return []


def _ingest_slots(cache: CudaCache, problem: Path, result: dict, slots: list[str], specs: dict, tag: str) -> list:
    """Put a run's best sources for ``slots`` into the cache (guarded on the
    problem's instance shapes), whatever their speedup over their seed: the
    hint's acceptance already measured them against the unchanged graph."""
    man = json.loads((Path(problem) / "manifest.json").read_text())
    best = best_sources(Path(result["_dir"]))
    out = []
    for s in man["slots"]:
        if s["id"] not in slots or s["id"] not in best:
            continue
        spec = specs.get(s["op"])
        if spec is None:
            from .kb_proposals import spec_from_kind
            spec = spec_from_kind(s["op"], specs)
        entry, _ = parse_signature(spec.signature)
        sr = result.get("slots", {}).get(s["id"], {})
        e, st = cache.add(op=s["op"], entry=entry, signature=spec.signature, arch=f"sm_{man_arch(result)}",
                          shapes=[i["shape"] for i in s["instances"]], code=best[s["id"]],
                          slot_us=sr.get("best_us") or sr.get("final_us") or 0.0,
                          speedup=(sr["baseline_us"] / sr["best_us"]) if sr.get("baseline_us") and sr.get("best_us") else None,
                          impl_id=f"{result.get('run_id', 'run')}:{s['id']}", source={"hint": tag})
        out.append({"slot": s["id"], "status": st})
    cache.save()
    return out


def man_arch(result: dict) -> str:
    return str(result.get("arch", "sm_75")).removeprefix("sm_")


def evaluate_hint(h: dict, prof: dict, out: Path, *, kb: KBRunner, cache: CudaCache, doc: dict,
                  cfg: Config, specs: dict) -> dict:
    """Build original + variant problems for one hint on one IR, optimise
    both with the same budget, re-measure, decide, record."""
    base_graph = json.loads((Path(prof["ir_dir"]) / "graph.json").read_text())
    specs = with_fused_specs(base_graph, specs)
    base = cache_plan(base_graph, cache, f"sm_{cfg.arch}", specs)
    g0 = base["graph"]
    names = _scope_ops(h, g0, specs)
    ev = {"problem": None, "ir": prof["ir"], "ts": gh._now()}
    if not names:
        ev.update(decision="rejected", why="hint does not occur in this IR")
        gh.record_decision(doc, h, ev)
        return {"hint": h["id"], "status": "rejected", "why": ev["why"]}
    hh = dict(h)
    if h["kind"] == "fuse":                             # scope the variant to this IR's instances
        hh["chains"] = [c for c in (h.get("chains") or [])
                        if all(a in {x for n in names for x in gh.atoms(n)} for a in c)] or None
    vg, _, prov = gh.apply_hints(g0, [hh], specs, statuses=None)
    if not prov["applied"]:
        ev.update(decision="rejected", why="; ".join(s["why"] for s in prov["skipped"])[:500])
        gh.record_decision(doc, h, ev)
        return {"hint": h["id"], "status": "rejected", "why": ev["why"]}
    vplan = cache_plan(vg, cache, f"sm_{cfg.arch}", specs, split=False)
    out.mkdir(parents=True, exist_ok=True)
    src_ir = Path(prof["source_ir"])
    oir = write_variant_ir(src_ir, g0, out / "orig_ir")
    vir = write_variant_ir(src_ir, vplan["graph"], out / "variant_ir")
    stem = _safe(f"{prof['ir']}__{h['id']}")
    oidx = [i for i, o in enumerate(g0["ops"]) if o["name"] in set(names)]
    vnames = {prov["renamed"].get(n, n) for n in names}
    vidx = [i for i, o in enumerate(vplan["graph"]["ops"]) if o["name"] in vnames]
    seeds = {**base["sources"], **vplan["sources"]}
    tags = {**base["tags"], **vplan["tags"]}
    op_ = export_problem(oir, out / "orig", op_indices=oidx, specs=specs, name=stem + "__orig",
                         strategy="hint-original", slot_sources=base["sources"], slot_source_tags=base["tags"],
                         sub_sources=base["sub_sources"])
    vp_ = export_problem(vir, out / "variant", op_indices=vidx, specs=specs, name=stem + "__variant",
                         strategy="hint-variant", slot_sources=seeds, slot_source_tags=tags,
                         sub_sources=vplan["sub_sources"])
    ro, rv = kb.run([op_, vp_], out / "run", steps=cfg.variant_steps)
    ev.update(problem=stem, original_us=ro.get("final_partition_us"), variant_us=rv.get("final_partition_us"),
              kb_original_us=ro.get("final_partition_us"), kb_variant_us=rv.get("final_partition_us"),
              llm_calls=(ro.get("llm_calls") or 0) + (rv.get("llm_calls") or 0))
    ok = bool(ro.get("final_ok") and rv.get("final_ok"))
    if ok and cfg.remeasure:
        t = remeasure([(op_, best_sources(Path(ro["_dir"]))), (vp_, best_sources(Path(rv["_dir"])))],
                      arch=cfg.arch, reps=cfg.remeasure)
        ok = bool(t[0] and t[1])
        if ok:
            ev.update(original_us=_median(t[0]), variant_us=_median(t[1]),
                      remeasured={"original_us": t[0], "variant_us": t[1]})
    gain = (ev["original_us"] - ev["variant_us"]) / ev["original_us"] if ok else None
    status = "accepted" if ok and gain >= cfg.accept_gain else "rejected"
    ev.update(gain=gain, decision=status, applied=prov["applied"])
    rec = gh.record_decision(doc, h, ev)
    res = {"hint": h["id"], "status": status, "hint_status": rec["status"], "gain": gain,
           "original_us": ev["original_us"], "variant_us": ev["variant_us"], "ops": names}
    if rec["status"] == "accepted":
        man = json.loads((Path(vp_) / "manifest.json").read_text())
        changed = [s["id"] for s in man["slots"]
                   if s["op"].startswith("fused__") or s.get("variant_of")]
        res["cached"] = _ingest_slots(cache, vp_, rv, changed, specs, h["id"])
    return res


def run(irs: list[Path], work: Path, cfg: Config) -> dict:
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    hints_path, state_path = work / "graph_hints.json", work / "state.json"
    doc, cache = gh.load(hints_path), CudaCache(work / "cache")
    state = json.loads(state_path.read_text()) if state_path.exists() else {"sent": [], "rounds": []}
    kb = KBRunner(cfg, work / "knowledge")
    specs = dict(_all_specs())
    first = None
    for r in range(len(state["rounds"]), len(state["rounds"]) + cfg.rounds):
        R = work / f"r{r:02d}"
        t0 = time.time()
        rec = {"round": r, "kb_runs": [], "decisions": []}
        profiles = [profile_ir(Path(d), R / "profile" / Path(d).name, hints_doc=doc, cache=cache, arch=cfg.arch,
                               specs=specs, runs=cfg.profile_runs) for d in irs]
        bad = [p["ir"] for p in profiles if not p["passed"]]
        if bad:
            raise RuntimeError(f"round {r}: profile failed for {bad}; see {R / 'profile'}")
        first = first or {p["ir"]: p["partition_us"] for p in profiles}
        rec["partition_us"] = {p["ir"]: p["partition_us"] for p in profiles}
        for p in profiles:                                  # MB's own proposals
            g = json.loads((Path(p["ir_dir"]) / "graph.json").read_text())
            gh.merge(doc, gh.propose_from_profile(g, {o["name"]: o["us"] or 0.0 for o in p["ops"]}, specs,
                                                   doc=doc, min_gain_pct=cfg.min_gain_pct, ir=p["ir"]))
        plan = make_plan(profiles, coverage=cfg.coverage, top_k=cfg.top_k, min_share=cfg.min_share,
                         exclude=set(state["sent"]))
        (R / "plan.json").write_text(json.dumps(plan, indent=1))
        rec["plan"] = {"selected_share": plan["selected_share"],
                       "problems": [(p["ir"], len(p["op_names"])) for p in plan["problems"]]}
        if plan["problems"]:
            dirs = export_plan(plan, R / "problems", cache=cache, arch=cfg.arch, specs=specs,
                               names={p["ir"]: f"{p['ir']}__r{r:02d}" for p in plan["problems"]})
            results = kb.run(dirs, R / "run", winners=R / "winners")
            rec["kb_runs"] = [{k: x.get(k) for k in ("problem", "baseline_partition_us", "final_partition_us",
                                                      "speedup", "final_ok", "llm_calls")} for x in results]
            if (R / "winners" / "winners.json").exists():
                rec["ingest"] = ingest(R / "winners" / "winners.json", cache, specs)
            state["sent"] += [it["key"] for it in plan["items"] if it.get("selected")]
            for x in results:
                pf = Path(x["_dir"]) / "proposals.json"
                if pf.exists():
                    props = json.loads(pf.read_text())
                    for h in gh.from_kb_proposals(props, specs=specs):
                        for s in h["source"]:
                            s["ir"] = json.loads((Path(props["problem_dir"]) / "manifest.json").read_text())["ir_name"]
                        gh.merge(doc, [h])
        gh.save(doc, hints_path)
        todo = [h for h in doc["hints"] if h.get("status") == "proposed" and h.get("target", "cuda") == "cuda"
                and h["kind"] in ("fuse", "split_slot") and _est_gain(h) >= cfg.min_gain_pct]
        todo.sort(key=lambda h: -_est_gain(h))
        for h in todo[:cfg.max_variants]:
            prof = _hint_ir(h, profiles)
            if prof is None:
                continue
            d = evaluate_hint(h, prof, R / "variants" / _safe(h["id"]), kb=kb, cache=cache, doc=doc,
                              cfg=cfg, specs=specs)
            rec["decisions"].append(d)
            gh.save(doc, hints_path)
        rec["seconds"] = round(time.time() - t0, 1)
        rec["llm_calls_total"] = kb.calls
        state["rounds"].append(rec)
        state_path.write_text(json.dumps(state, indent=1))
        (R / "round.json").write_text(json.dumps(rec, indent=1))
        accepted = [d for d in rec["decisions"] if d.get("hint_status") == "accepted"]
        print(json.dumps({"round": r, "partition_us": rec["partition_us"], "plan": rec["plan"]["problems"],
                          "decisions": [(d["hint"], d["status"], d.get("gain")) for d in rec["decisions"]]}),
              flush=True)
        if not accepted and not plan["problems"] and len(todo) <= cfg.max_variants:
            state["fixpoint"] = r
            break
    final = {}
    for d in irs:
        final[Path(d).name] = verify_ir(Path(d), cache, arch=cfg.arch, specs=specs, hints=hints_path,
                                        work=work / "final" / Path(d).name)
    summary = {"irs": {}, "rounds": len(state["rounds"]), "fixpoint": state.get("fixpoint"),
               "accepted": [h["id"] for h in doc["hints"] if h.get("status") == "accepted"],
               "rejected": [h["id"] for h in doc["hints"] if h.get("status") == "rejected"],
               "llm_calls": kb.calls}
    first0 = state["rounds"][0]["partition_us"] if state["rounds"] else (first or {})
    for name, v in final.items():
        ir = v.get("ir", name)
        f0 = first0.get(ir)
        summary["irs"][ir] = {"first_us": f0, "final_us": v.get("partition_us"), "passed": v.get("passed"),
                              "speedup": (f0 / v["partition_us"]) if f0 and v.get("partition_us") else None,
                              "hints": v.get("hints"), "fused": v.get("fused"),
                              "cache_split": v.get("cache_split")}
    (work / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("irs", nargs="+")
    ap.add_argument("--work", required=True)
    ap.add_argument("--kb", required=True, help="KernelBlaster checkout")
    ap.add_argument("--model", default="claude-code:sonnet")
    ap.add_argument("--reflect-model")
    ap.add_argument("--plugin", action="append", default=[], help="passed to run_multislot.py (stub: models)")
    ap.add_argument("--python", default=sys.executable, help="python for KernelBlaster")
    for k, v in asdict(Config(kb=Path("."))).items():
        if k in ("kb", "model", "reflect_model", "plugins", "python"):
            continue
        ap.add_argument("--" + k.replace("_", "-"), type=type(v) if v is not None else
                        (int if k in ("top_k", "variant_steps") else str), default=v)
    a = ap.parse_args()
    cfg = Config(kb=Path(a.kb).resolve(), model=a.model, reflect_model=a.reflect_model,
                 plugins=[str(Path(p).resolve()) for p in a.plugin], python=a.python,
                 **{k: getattr(a, k) for k in asdict(Config(kb=Path("."))) if k not in
                    ("kb", "model", "reflect_model", "plugins", "python")})
    print(json.dumps(run([Path(d) for d in a.irs], Path(a.work), cfg), indent=1))


if __name__ == "__main__":
    main()
