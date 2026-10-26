"""Graph search redesign (FEATURE_TRACKER F1/F8; l3_farm/analysis/graph_search_redesign.md) as opt-in hooks of
``kb_graph_opt``. With the defaults (``--graph-search rounds`` and every other ``--graph-*`` flag at its default)
nothing here runs, the flow is the round-based one, and existing work dirs keep their ``kb_resume`` round key.

    --graph-search rounds|interleaved    rounds (default): profile -> proposals -> plan -> KB run -> ingest -> decide.
                                         interleaved: the round's hint decisions run INSIDE the plan's KernelBlaster
                                         run (one process: the knowledge store stays single-writer), each hint on a
                                         successive-halving rung budget (``--graph-rung-steps``, ``--graph-eta``,
                                         ``--graph-cap-steps``); a hint is re-timed (paired) and decided as soon as its
                                         two problems finish (variant_policy.checkpoint: accepted / promising ->
                                         next rung next round / rejected); an accepted hint is applied to the live
                                         graph at once: the plan problems whose ops it absorbs get a stop file
                                         (KernelBlaster: no new steps) and its fused slots go into the cache.
    --graph-priors auto|off|rank|preapply
                                         dataset priors (graph_priors; partition-scoped): rank = order candidates by
                                         est. gain x P(accept); preapply = also accept hints whose prior lower bound
                                         is >= --graph-prior-conf (>= --graph-prior-min-n decisions) without a
                                         measurement, before the round's plan (--graph-explore: share of them still
                                         decided by measurement). auto = preapply with interleaved, else off.
    --graph-characterize N               round 0: a cheap pass (N steps x --graph-char-rollouts on the
                                         --graph-char-top largest op items, winners cached) + re-profile + roofline
                                         headroom per op, before any graph decision. -1 = auto (1 with interleaved).
    --graph-survival auto|on|off         plan slot problems get steps in proportion to P(slot survives into the final
                                         graph) (pending hints that would absorb its ops, their priors);
                                         auto = on with interleaved.
    --graph-diverse-beam N               N graph-variant members (baseline, high-prior fusions, + uncertain
                                         fusions, + layout fold, + attention); one plan over all members' profiles
                                         (items keyed by op signature: common ops optimised once, the kernel cache is
                                         shared), paired whole-model re-timing of the members at every round end,
                                         successive halving to one member whose hints are then accepted.
    --graph-layout-fold on|off           propose fold_layout hints (layout_fold matcher; composite seeds).

Every graph decision is a line in ``<work>/graph_decisions.jsonl`` (time, round, phase, budget used, evidence,
prior used) and is summarised in ``round.json`` / ``summary.json`` ``graph_search``.
"""
from __future__ import annotations

import copy
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import graph_hints as gh
from . import graph_priors as gp
from . import variant_policy as vp

CFG_DEFAULTS = {"graph_search": "rounds", "graph_priors": "auto", "graph_prior_conf": 0.55, "graph_prior_min_n": 3,
                "graph_explore": 0.1, "graph_prior_min_gain": 0.0, "graph_characterize": -1, "graph_char_top": 8,
                "graph_char_rollouts": 1, "graph_survival": "auto", "graph_survival_head": 0.25,
                "graph_rung_steps": 0, "graph_eta": 2.0, "graph_cap_steps": 8, "graph_defer": 1,
                "graph_diverse_beam": 0, "graph_layout_fold": "off"}
PRICES = {"in": 5.0, "out": 25.0}            # $/M tokens (assumed Opus list prices; no cache split in result.json)

CURRENT: dict = {"gs": None}


def current() -> "GraphSearch | None":
    return CURRENT.get("gs")


def usd(usage: dict | None) -> float:
    u = usage or {}
    c = float(u.get("cost_usd") or 0.0)
    if c:
        return c
    return (float(u.get("prompt_tokens") or 0) * PRICES["in"] + float(u.get("completion_tokens") or 0) * PRICES["out"]) / 1e6


def _manifest(d: Path) -> dict:
    return json.loads((Path(d) / "manifest.json").read_text())


def set_budget(problem: Path, budget: dict) -> bool:
    """Write ``mb_budget`` into a problem's manifest (KernelBlaster run_multislot: per-problem rollouts / steps /
    stop_patience). Idempotent: unchanged when equal (the problem hash must not change across a resume)."""
    f = Path(problem) / "manifest.json"
    man = json.loads(f.read_text())
    b = {k: v for k, v in budget.items() if v is not None}
    if man.get("mb_budget") == b:
        return False
    man["mb_budget"] = b
    f.write_text(json.dumps(man, indent=1))
    return True


class GraphSearch:
    """This run's graph-search switches and state; every hook is a no-op with the defaults."""

    def __init__(self, cfg, work: Path, irs: list | None = None):
        self.cfg, self.work = cfg, Path(work)
        g = lambda k: getattr(cfg, k, CFG_DEFAULTS[k])          # noqa: E731
        self.mode = (g("graph_search") or "rounds").lower()
        if self.mode not in ("rounds", "interleaved"):
            raise ValueError(f"--graph-search must be rounds|interleaved, not {self.mode!r}")
        self.inter = self.mode == "interleaved"
        pm = (g("graph_priors") or "auto").lower()
        if pm not in ("auto", "off", "rank", "preapply"):
            raise ValueError(f"--graph-priors must be auto|off|rank|preapply, not {pm!r}")
        self.priors_mode = ("preapply" if self.inter else "off") if pm == "auto" else pm
        self.prior_conf, self.prior_min_n = float(g("graph_prior_conf")), int(g("graph_prior_min_n"))
        self.explore_rate, self.prior_min_gain = float(g("graph_explore")), float(g("graph_prior_min_gain"))
        ch = int(g("graph_characterize"))
        self.char_steps = (1 if self.inter else 0) if ch < 0 else ch
        self.char_top, self.char_rollouts = int(g("graph_char_top")), int(g("graph_char_rollouts"))
        sv = (g("graph_survival") or "auto").lower()
        if sv not in ("auto", "on", "off"):
            raise ValueError(f"--graph-survival must be auto|on|off, not {sv!r}")
        self.survival_on = self.inter if sv == "auto" else sv == "on"
        self.survival_head = float(g("graph_survival_head"))
        self.rung_steps, self.eta = int(g("graph_rung_steps")), float(g("graph_eta"))
        self.cap_steps, self.defer = int(g("graph_cap_steps")), bool(int(g("graph_defer")))
        self.beam = int(g("graph_diverse_beam") or 0)
        lf = (g("graph_layout_fold") or "off").lower()
        if lf not in ("on", "off"):
            raise ValueError(f"--graph-layout-fold must be on|off, not {lf!r}")
        self.layout_fold = lf == "on"
        self.part = None
        self.priors: gp.Priors | None = None
        if self.active():
            from .kb_global import part_for
            self.part = part_for(cfg, irs or [])
            self.priors = gp.Priors.default(self.part, self.work, min_n=self.prior_min_n)
        self.log_path = self.work / "graph_decisions.jsonl"
        self.t_run = time.time()
        self.round_t0 = self.t_run
        self.round = None
        self.events: list[dict] = []

    # ---- switches
    def explicit(self) -> dict:
        """Every switch at a non-default value (raw flags)."""
        return {k: getattr(self.cfg, k) for k, v in CFG_DEFAULTS.items()
                if getattr(self.cfg, k, v) != v}

    def active(self) -> bool:
        return bool(self.explicit())

    def resolved(self) -> dict:
        return {"mode": self.mode, "priors": self.priors_mode, "prior_conf": self.prior_conf,
                "prior_min_n": self.prior_min_n, "explore": self.explore_rate, "prior_min_gain": self.prior_min_gain,
                "characterize": [self.char_steps, self.char_top, self.char_rollouts],
                "survival": [self.survival_on, self.survival_head],
                "rungs": [self.rung_steps, self.eta, self.cap_steps, self.defer],
                "beam": self.beam, "layout_fold": self.layout_fold}

    def round_key(self) -> dict:
        """Only when a flag is set: existing work dirs keep their key."""
        return {"graph_search": self.resolved()} if self.active() else {}

    # ---- execution mode (FEATURE_TRACKER H.1: decisions timed in the deployed graph execution, or flagged)
    def exec_mode(self, scope: str | None = None) -> dict:
        """How the timings behind a graph decision are executed, read from the process environment the exports and
        re-timings use: walker concurrency ($MB_GRAPH_CONCURRENCY, set by --graph-concurrency; ``auto`` starts ``on``
        and may switch after a round), SM-budget policy, CUDA-graph replay (kb_proposals.remeasure and kb_model_run
        both time the CUDA-graph replay of the partition), the routing stamp (sha of the kernel cache index + the
        hints doc as they are now) and the scope (``partition``: a hint's sub-graph timed in isolation; ``whole``)."""
        import hashlib
        conc = os.getenv("MB_GRAPH_CONCURRENCY") or "off"
        h = hashlib.sha1()
        for f in (self.work / "cache" / "index.json", self.work / "graph_hints.json"):
            try:
                h.update(f.read_bytes())
            except OSError:
                h.update(b"-")
        return {"concurrency": conc, "concurrency_flag": getattr(self.cfg, "graph_concurrency", "off"),
                "sm_budget_policy": os.getenv("MB_SM_BUDGET_POLICY") or ("split" if conc != "off" else None),
                "graph_replay": "cuda_graph", "routing_stamp": h.hexdigest()[:12],
                **({"scope": scope} if scope else {})}

    def mode_mismatches(self, final: dict | None = None) -> list[dict]:
        """Graph decisions whose execution mode (concurrency, SM-budget policy, replay) differs from the final one
        (default: the mode now, i.e. what the final verification / model run will use)."""
        final = final or self.exec_mode()
        keys = ("concurrency", "sm_budget_policy", "graph_replay")
        out = []
        for e in self.events:
            m = e.get("exec_mode") or {}
            if e.get("event") != "decision" or not m:
                continue
            diff = {k: [m.get(k), final.get(k)] for k in keys if m.get(k) != final.get(k)}
            if diff or m.get("scope") == "partition":
                out.append({"round": e.get("round"), "hint": e.get("hint"), "phase": e.get("phase"), "diff": diff,
                            "isolated": m.get("scope") == "partition"})
        return out

    # ---- telemetry
    def start_round(self, r: int) -> None:
        self.round, self.round_t0 = r, time.time()

    def event(self, event: str, **kw) -> dict:
        now = time.time()
        e = {"event": event, "ts": round(now, 3), "t_run_s": round(now - self.t_run, 1),
             "t_round_s": round(now - self.round_t0, 1), "round": self.round, **kw}
        if event in ("decision", "beam_select", "characterize") and "exec_mode" not in e:
            e["exec_mode"] = self.exec_mode()
        self.events.append(e)
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(e, default=str) + "\n")
        except OSError:
            pass
        return e

    def round_summary(self, r: int) -> dict:
        ev = [e for e in self.events if e.get("round") == r]
        dec = [e for e in ev if e["event"] == "decision"]
        return {"decisions": len(dec),
                "by_status": {s: sum(1 for e in dec if e.get("status") == s)
                              for s in sorted({e.get("status") for e in dec})},
                "preapplied": [e["hint"] for e in dec if e.get("phase") == "preapply"],
                "stopped": [x for e in ev if e["event"] == "stop" for x in e.get("problems", [])],
                "events": len(ev)}

    def summary(self) -> dict:
        dec = [e for e in self.events if e["event"] == "decision"]
        return {**self.resolved(), "part": self.part,
                "prior_sources": sorted(self.priors.sources) if self.priors else [],
                "decisions": [{k: e.get(k) for k in ("round", "phase", "hint", "status", "t_run_s", "measured",
                                                     "steps", "gain")} for e in dec],
                "stops": [e for e in self.events if e["event"] == "stop"],
                "exec_mode_final": self.exec_mode(),
                # H.1: decisions not timed in the final execution mode (concurrency / SM policy / replay differ), or
                # timed on an isolated sub-graph (every per-hint decision: the paired problems are partitions)
                "exec_mode_mismatch": self.mode_mismatches(),
                "log": str(self.log_path)}

    # ---- priors
    def prior(self, h: dict) -> dict | None:
        return self.priors.prior(h) if self.priors is not None else None

    def rank(self, h: dict, est: float) -> float:
        """Candidate order: est. gain, x P(accept) with priors (rank / preapply)."""
        if self.priors_mode == "off" or self.priors is None:
            return est
        return est * float(self.prior(h)["p_mean"])

    def eligible(self, pr: dict | None) -> bool:
        return bool(pr) and pr.get("n", 0) >= self.prior_min_n and pr.get("lb", 0.0) >= self.prior_conf

    def preapply(self, r: int, doc: dict, graphs: dict, profiles: list[dict], specs: dict, *, kinds=("fuse",)) -> list:
        """Accept the hints whose prior is confident enough, without a measurement (``--graph-priors preapply``):
        proposals of every fusable chain on the current graphs (gain threshold ``--graph-prior-min-gain``) plus the
        doc's proposed hints. -> the pre-applied hint ids. An exploration draw leaves a hint to the measured
        decision (``--graph-explore``)."""
        if self.priors_mode != "preapply" or self.beam > 1:
            return []
        cands: dict[str, dict] = {}
        for p in profiles:
            g = graphs.get(p["ir"])
            if g is None:
                continue
            for h in scan_pairs(g, p, specs, doc, ir=p["ir"], min_gain_pct=self.prior_min_gain):
                if h["id"] in cands:
                    gh.merge({"hints": [cands[h["id"]]]}, [h])
                else:
                    cands[h["id"]] = h
        for h in doc["hints"]:
            if h.get("status") == "proposed" and h.get("kind") in kinds and h.get("target", "cuda") == "cuda":
                cands.setdefault(h["id"], h)
        # most confident first; chains that overlap an already pre-applied one wait for the next round (the
        # fused op is then their producer / consumer: the chain grows under its own prior)
        pri = {hid: self.prior(h) for hid, h in cands.items()}
        order = sorted(cands, key=lambda hid: -(pri[hid] or {}).get("lb", 0.0))
        taken: set = set()
        done = []
        for hid in order:
            h = cands[hid]
            atoms_ = {a for c in (h.get("chains") or []) for a in c}
            if atoms_ & taken:
                continue
            if gh.get(doc, hid) is not None and gh.get(doc, hid).get("status") != "proposed":
                continue
            pr = pri[hid]
            if not self.eligible(pr):
                continue
            taken |= atoms_
            gh.merge(doc, [h])
            if gp.explore(int(getattr(self.cfg, "seed", 0)), hid, self.explore_rate):
                self.event("explore", hint=hid, prior=pr, why="exploration draw: decided by measurement")
                continue
            ir = next((s.get("ir") for s in h.get("source", []) if s.get("ir")), None)
            gh.record_decision(doc, hid, {"decision": "accepted", "by": "graph-prior", "prior": pr, "ir": ir,
                                          "round": r, "measured": False, "ts": gh._now(),
                                          "why": f"prior {pr['key']} {pr['accepted']:.0f}/{pr['rejected']:.0f}, "
                                                 f"lb {pr['lb']:.2f} >= {self.prior_conf}"})
            self.event("decision", phase="preapply", hint=hid, status="accepted", measured=False, prior=pr,
                       exec_mode=self.exec_mode("none: not measured"),
                       steps=0, llm_calls=0, usd=0.0, ir=ir,
                       evidence={"est_gain_pct": max((s.get("est_gain_pct") or 0.0) for s in h.get("source") or [{}])})
            done.append(hid)
        return done

    # ---- characterization (round 0)
    def characterize(self, R: Path, prog, profiles: list[dict], *, kb, cache, specs: dict, reprofile) -> list[dict]:
        """A cheap first pass before any graph decision: ``char_steps`` steps x ``char_rollouts`` on the
        ``char_top`` largest op items (seed shares), winners cached (not marked sent: the plan optimises them
        further from there), then the IRs re-profiled (realistic shares for proposals, priors and the plan) and the
        roofline headroom of every op recorded. -> the new profiles."""
        if self.char_steps <= 0:
            return profiles
        from .kb_ingest import ingest
        from .kb_plan import export_plan, make_plan

        def run():
            plan = make_plan(profiles, coverage=2.0, top_k=self.char_top or None, per_item=True, min_share=0.0)
            if not plan["problems"]:
                return {"items": 0}
            dirs = export_plan(plan, R / "char" / "problems", cache=cache, arch=self.cfg.arch, specs=specs,
                               names={p.get("id", p["ir"]): f"{p.get('id', p['ir'])}__char" for p in plan["problems"]})
            for d in dirs:
                set_budget(d, {"steps": self.char_steps, "rollouts": self.char_rollouts, "stop_patience": 0})
            res = kb.run(dirs, R / "char" / "run", steps=self.char_steps, winners=R / "char" / "winners",
                         parallel_problems=len(dirs))
            ing = []
            if (R / "char" / "winners" / "winners.json").exists():
                ing = ingest(R / "char" / "winners" / "winners.json", cache, specs)
            return {"items": len(dirs), "llm_calls": sum(x.get("llm_calls") or 0 for x in res),
                    "usd": round(sum(usd(x.get("llm_usage")) for x in res), 4),
                    "problems": [{k: x.get(k) for k in ("problem", "baseline_partition_us", "final_partition_us",
                                                         "speedup", "llm_calls")} for x in res],
                    "ingested": sum(1 for x in ing if x.get("status") in ("added", "replaced"))}
        rep = prog.once("gs_char", run)
        new = prog.get("gs_char_profile")
        if new is None:
            new = reprofile("profile_char")
            prog.mark("gs_char_profile", new)
        hr = self.headroom_table(new)
        self.event("characterize", steps=self.char_steps, rollouts=self.char_rollouts, top=self.char_top,
                   llm_calls=rep.get("llm_calls"), usd=rep.get("usd"), items=rep.get("items"),
                   partition_us_before={p["ir"]: p.get("partition_us") for p in profiles},
                   partition_us_after={p["ir"]: p.get("partition_us") for p in new},
                   headroom_top=hr[:12])
        (R / "char").mkdir(parents=True, exist_ok=True)
        (R / "char" / "headroom.json").write_text(json.dumps(hr, indent=1))
        return new

    def headroom_table(self, profiles: list[dict]) -> list[dict]:
        out = []
        for p in profiles:
            try:
                g = gh.unique_names(json.loads((Path(p["ir_dir"]) / "graph.json").read_text()))
            except (OSError, ValueError, KeyError):
                continue
            by = {o["name"]: o for o in g["ops"]}
            tot = p.get("e2e_us") or sum((o.get("us") or 0.0) for o in p["ops"]) or 1.0
            for r in p["ops"]:
                o = by.get(r["name"])
                if o is None or not r.get("us"):
                    continue
                h = vp.op_headroom(g, o, r["us"], arch=str(self.cfg.arch))
                out.append({"ir": p["ir"], "op": r["name"], "kind": r["op"], "us": r["us"],
                            "share": round(r["us"] / tot, 4), "bound_us": h["bound_us"], "ratio": h["ratio"]})
        return sorted(out, key=lambda x: -x["us"])

    # ---- survival-weighted plan budgets
    def pending(self, doc: dict, graphs: dict, specs: dict, scope_ops) -> dict[str, list]:
        """{IR: [{"hint", "p", "absorb", "heads"}]} for the proposed hints (fuse / extended kinds)."""
        out: dict[str, list] = {}
        for h in doc["hints"]:
            if h.get("status") != "proposed" or h.get("target", "cuda") != "cuda":
                continue
            p = float((self.prior(h) or {}).get("p_mean", 0.5))
            for ir, g in graphs.items():
                try:
                    names = scope_ops(h, g, specs)
                except Exception:  # noqa: BLE001
                    continue
                if not names:
                    continue
                heads, absorb = set(), set()
                if h["kind"] == "fuse":
                    for ch in (h.get("chains") or []):
                        if ch:
                            heads.add(ch[0])
                            absorb.update(ch[1:])
                    if not heads:                       # no chains: every op of a located chain
                        for n in names:
                            a = gh.atoms(n)
                            heads.add(a[0])
                            absorb.update(a[1:])
                elif h["kind"] == "split_slot":
                    continue                           # a split keeps its ops (shape variants)
                else:
                    absorb.update(a for n in names for a in gh.atoms(n))
                out.setdefault(ir, []).append({"hint": h["id"], "p": p, "absorb": absorb, "heads": heads - absorb})
        return out

    def plan_budgets(self, dirs: list[Path], plan: dict, doc: dict, graphs: dict, specs: dict, scope_ops, *,
                     steps: int, stop_patience) -> dict:
        if not self.survival_on or not dirs:
            return {}
        pend = self.pending(doc, graphs, specs, scope_ops)
        rep = {}
        for d, prob in zip(dirs, plan["problems"]):
            atoms = [a for n in prob["op_names"] for a in gh.atoms(n)]
            ps = vp.survival(atoms, pend.get(prob["ir"], []), head_weight=self.survival_head)
            st = vp.survival_steps(steps, ps)
            if st < steps:
                set_budget(d, {"steps": st, "stop_patience": stop_patience})
            rep[_manifest(d)["name"]] = {"p_survive": round(ps, 3), "steps": st,
                                         "by": sorted({x["hint"] for x in pend.get(prob["ir"], [])
                                                       if set(atoms) & (x["absorb"] | x["heads"])})}
        self.event("survival", steps=steps, problems=rep,
                   steps_saved=sum(steps - v["steps"] for v in rep.values()))
        return rep

    # ---- rungs (successive halving) for hint decisions
    def bc(self) -> vp.BudgetCfg:
        vs = int(getattr(self.cfg, "variant_steps", None) or getattr(self.cfg, "steps", 2) or 2)
        base = self.rung_steps or max(1, int(round(vs / self.eta)))
        return vp.BudgetCfg(base_steps=base, min_steps=1, max_steps=max(base, vs), cap_steps=self.cap_steps,
                            eta=self.eta)

    def hint_budget(self, h: dict) -> dict:
        d = h.get("gs_deferred") or {}
        rung = int(d.get("rung", -1)) + 1
        used = int(d.get("steps_used", 0))
        bc = self.bc()
        st = vp.rung_steps(rung, bc.base_steps, bc, used)
        return {"rung": rung, "steps": max(1, st) if st > 0 else 0, "used": used}


    # ---- hooks called by kb_graph_opt._run (each a no-op unless its flag is set)
    def decide_kinds(self) -> tuple[str, ...]:
        return ("fold_layout",) if self.layout_fold else ()

    def before_propose(self, r: int, R: Path, prog, profiles: list[dict], *, kb, cache, specs: dict,
                       reprofile) -> list[dict]:
        """Round 0: the characterization pass (re-profiled). Later rounds: unchanged."""
        self.start_round(r)
        if r == 0 and self.char_steps > 0:
            return self.characterize(R, prog, profiles, kb=kb, cache=cache, specs=specs,
                                     reprofile=lambda sub: reprofile(sub))
        return profiles

    def after_propose(self, r: int, R: Path, prog, profiles: list[dict], graphs: dict, doc: dict, specs: dict, *,
                      reprofile, hints_path: Path | None = None) -> tuple[list[dict], dict]:
        """After the round's proposals: fold_layout proposals; pre-applied high-prior hints (then the IRs are
        re-profiled and proposed on again, so chains grow); beam members and their profiles. -> (profiles, graphs)."""
        if self.layout_fold:
            gh.merge(doc, propose_folds(graphs, profiles, doc, self))
        if self.priors_mode == "preapply" and self.beam <= 1:
            prog.once("gs_preapply", lambda: self.preapply(r, doc, graphs, profiles, specs))
            if hints_path is not None:
                gh.save(doc, hints_path)
            # from the doc, not the phase's return value: robust to a kill between the doc save and the mark
            pre = [h["id"] for h in doc["hints"] if any(e.get("by") == "graph-prior" and e.get("round") == r
                                                        for e in h.get("evidence") or [])]
            if pre:
                new = prog.get("gs_pre_profile")
                if new is None:
                    new = reprofile("profile_pre")
                    prog.mark("gs_pre_profile", new)
                profiles = new
                graphs = propose_all(profiles, doc, specs, self.cfg)
                if hints_path is not None:
                    gh.save(doc, hints_path)
        self.plan_profiles = None
        if self.beam > 1:
            self.load_beam()
            b = self.beam_state
            if not b.get("done"):
                if not b.get("members"):
                    b["members"] = beam_members(doc, self, self.beam)
                    b["formed_round"] = r
                    self.event("beam_members", members=b["members"])
                    if len(b["members"]) <= 1:
                        b["done"] = True
                        b["why"] = "one member only (nothing to diversify)"
                    self.save_beam()
                if not b.get("done"):
                    mp = prog.get("gs_beam_profiles")
                    if mp is None:
                        mp = []
                        for m in b["members"]:
                            if not m.get("alive"):
                                continue
                            ps = profiles if not m["hints"] else reprofile(f"beam_{m['id']}", member_doc(doc, m))
                            mp += [relabel(p, m["id"]) for p in ps]
                        prog.mark("gs_beam_profiles", mp)
                    self.plan_profiles = mp
                    self.member_graphs = {p["ir"]: json.loads((Path(p["ir_dir"]) / "graph.json").read_text())
                                          for p in mp}
        return profiles, graphs

    def plan_input(self, profiles: list[dict]) -> list[dict]:
        return self.plan_profiles if getattr(self, "plan_profiles", None) else profiles

    def beam_active(self) -> bool:
        return self.beam > 1 and bool(getattr(self, "plan_profiles", None))

    def budget(self, dirs: list, plan: dict, doc: dict, graphs: dict, specs: dict, scope_ops, *, steps: int,
               stop_patience) -> list:
        """Export hook: survival-weighted steps written into the plan problems' manifests (``--graph-survival``)."""
        if self.survival_on and dirs:
            gs_graphs = dict(graphs)
            gs_graphs.update(getattr(self, "member_graphs", None) or {})
            self.last_budgets = self.plan_budgets([Path(d) for d in dirs], plan, doc, gs_graphs, specs, scope_ops,
                                                  steps=steps, stop_patience=stop_patience)
        return dirs

    def ingest_specs(self, specs: dict, graphs: dict) -> dict:
        """Specs for the winners' ingest with every fused kind of the round's graphs: a fused slot's plan winner is
        cached. Default ON also without graph search (the round-based flow used to drop these winners as
        ``fusion-not-accepted`` because its specs lacked the fused kinds; X-137). Kill switch: ``--ingest-fused off``
        or ``MB_INGEST_FUSED=0``."""
        off = str(getattr(self.cfg, "ingest_fused", "on")).lower() in ("0", "off", "false", "no") or \
            os.getenv("MB_INGEST_FUSED", "1").lower() in ("0", "off", "false", "no")
        if off:
            return specs
        if not self.active() and not graphs:
            return specs
        from .kb_proposals import with_fused_specs
        out = specs                                    # with_fused_specs returns its input when nothing is missing
        for g in list(graphs.values()) + list((getattr(self, "member_graphs", None) or {}).values()):
            out = with_fused_specs(g, out)
        return out

    def winners(self, R: Path) -> Path:
        w = Path(R) / "winners" / "winners.json"
        if not self.inter:
            return w
        return filter_winners(w, Path(R) / "winners" / "winners_plan.json") or w

    def candidates(self, doc: dict, profiles: list[dict], kinds: tuple, est_gain, hint_ir, cfg) -> dict:
        if self.beam_active():
            return {"n_todo": 0, "cands": []}
        todo = [h for h in doc["hints"] if h.get("status") == "proposed" and h.get("target", "cuda") == "cuda"
                and h["kind"] in kinds and est_gain(h) >= cfg.min_gain_pct]
        todo.sort(key=lambda h: -self.rank(h, est_gain(h)))
        return {"n_todo": len(todo), "cands": [(h["id"], p["ir"]) for h in todo[:cfg.max_variants]
                                               if (p := hint_ir(h, profiles)) is not None]}

    def run_interleaved(self, r: int, R: Path, prog, *, plan_dirs: list, plan: dict | None, cands_fn, doc: dict,
                        profiles: list[dict], graphs: dict, kb, cache, cfg, specs: dict, leases, width: int,
                        hints_path: Path, poll_s: float = 1.0) -> dict:
        """Plan problems + the round's first wave of hint decisions in one KernelBlaster run (see
        :class:`Interleaved`); stored for :meth:`interleaved_decide`. -> {"plan_results", ...}."""
        from . import kb_graph_opt as go
        from . import kb_resume
        from . import kb_telemetry as kbt
        cs = prog.once("cands", cands_fn)
        cands = [(gh.get(doc, hid), next(p for p in profiles if p["ir"] == ir)) for hid, ir in cs["cands"]]
        cands = [(h, p) for h, p in cands if h is not None]
        fps = [go.footprint(h, p["ir"], graphs[p["ir"]], specs) for h, p in cands]
        waves = go.schedule(fps, max(1, len(cands)))
        sel = waves[0] if waves else []
        st = dict(prog.get("gs_inter") or {})
        if "budgets" not in st:
            st["budgets"] = {cands[i][0]["id"]: self.hint_budget(cands[i][0]) for i in sel}
            st["sel"] = [cands[i][0]["id"] for i in sel]
            prog.mark("gs_inter", st)
        jobs = []
        for i in sel:
            h, p = cands[i]
            b = st["budgets"][h["id"]]
            job = kb_resume.prepared(go.prepare_hint, h, p, R / "variants" / go._safe(h["id"]), cache=cache, cfg=cfg,
                                     specs=specs)
            job["gs"] = b
            if "early" not in job:
                if b["steps"] <= 0:
                    job["early"] = f"graph search: hint budget exhausted ({b['used']}/{self.cap_steps} steps)"
                else:
                    for side in ("orig", "variant"):
                        set_budget(job[side], {"steps": b["steps"], "stop_patience": 0})
            jobs.append(job)
        plan_irs = [pp["ir"] for pp in (plan or {}).get("problems", [])] if plan_dirs else []
        parent = kbt.get().current()
        n_hint = 2 * sum(1 for j in jobs if "early" not in j)
        par = (cfg.plan_parallel + n_hint) if (cfg.plan_parallel and plan_dirs) else (n_hint + len(plan_dirs)) or None
        run_dir = R / "run"
        eng = Interleaved(
            self, run_dir=run_dir, prog=prog,
            start=lambda probs: kb.start(probs, run_dir, winners=R / "winners", parallel_problems=par,
                                         stop_patience=cfg.stop_patience),
            read=lambda d: go.read_result(run_dir, Path(d)),
            measure=lambda j, ro, rv, attempts=None: kbt.run_within(parent, "mb.remeasure", go.measure_hint, j, ro,
                                                                    rv, cfg, leases, attempts=attempts),
            finish=lambda j, ro, rv, times: decide_checkpoint(
                self, j, ro, rv, times, cfg=cfg, doc=doc,
                finish_std=lambda *a: go.finish_hint(*a, cache=cache, doc=doc, cfg=cfg)),
            collect=lambda probs, proc: kb.collect([Path(x) for x in probs], run_dir, proc),
            scope_after=lambda j: self.absorbed(j, doc, graphs, specs),
            save_hints=lambda: gh.save(doc, hints_path), poll_s=poll_s,
            workers=max(1, min(max(1, len(jobs)), 2 * (len(leases.devices) if leases is not None else 1))))
        self.event("interleaved_start", plan_problems=len(plan_dirs), hints=[j["hint"]["id"] for j in jobs],
                   waiting=[cands[i][0]["id"] for w in waves[1:] for i in w],
                   budgets={j["hint"]["id"]: j["gs"] for j in jobs})
        out = eng.run([Path(d) for d in plan_dirs], plan_irs, jobs)
        out["cs"], out["waves"] = cs, [[j["hint"]["id"] for j in jobs]]
        self.inter_out = getattr(self, "inter_out", {})
        self.inter_out[r] = out
        self.event("interleaved_end", decisions=[(d or {}).get("status") for d in out["decisions"]],
                   stopped=out["stopped"])
        return out

    def absorbed(self, job: dict, doc: dict, graphs: dict, specs: dict) -> set:
        """Atoms of the ops an accepted hint removes from its IR's current graph (every match of its pattern)."""
        from . import kb_graph_opt as go
        from .kb_proposals import with_fused_specs
        h = gh.get(doc, job["hint"]["id"]) or job["hint"]
        g = graphs.get(job["ev"].get("ir"))
        if g is None:
            return set()
        gu = gh.unique_names(copy.deepcopy(g))
        try:
            names = go._scope_ops(h, gu, with_fused_specs(g, specs))
        except Exception:  # noqa: BLE001
            names = []
        # the measured chains too (an accepted hint's chains move to tried_chains, and the kind's pattern is not
        # found where a stage is already fused, e.g. conv1 -> bn1+relu for conv+bn+relu)
        for ch in (h.get("chains") or []) + (h.get("tried_chains") or []):
            try:
                names += gh._locate(gu, ch)
            except ValueError:
                continue
        names += list(job.get("names") or [])
        return {a for n in names for a in gh.atoms(n)}

    def interleaved_decide(self, r: int, R: Path, prog, **kw) -> tuple[list, list]:
        """The decide step in interleaved mode: the decisions made inside the plan's run; if the round had no plan
        run, the hints alone (same engine)."""
        out = (getattr(self, "inter_out", None) or {}).get(r)
        if out is None:
            out = self.run_interleaved(r, R, prog, plan_dirs=[], plan=None, **kw)
        return out["waves"], [d for d in out["decisions"] if d is not None]

    def blocks_fixpoint(self, r: int, doc: dict) -> bool:
        if not self.active():
            return False
        if self.beam > 1 and not (getattr(self, "beam_state", None) or {}).get("done", True):
            return True
        return any(h.get("gs_deferred") and h.get("status") == "proposed" for h in doc["hints"])

    def end_round(self, r: int, R: Path, prog, *, profiles: list[dict], doc: dict, cache, specs: dict, kbeval=None,
                  gpu_model: str | None = None, hints_path: Path | None = None) -> dict:
        """Round end: the beam's checkpoint (paired whole-model re-timing of the members, successive halving,
        commit of the last member); the round's graph-search summary for round.json."""
        out = {}
        if self.beam_active():
            b = self.beam_state
            alive = [m["id"] for m in b["members"] if m.get("alive")]
            t = prog.once("gs_beam_time", lambda: beam_time(R, b["members"], profiles, doc, cache, specs,
                                                            arch=str(self.cfg.arch), kbeval=kbeval,
                                                            gpu_model=gpu_model, reps=max(3, int(self.cfg.remeasure))))
            keep = halve(t["times"], alive)
            for m in b["members"]:
                if m["id"] in alive and m["id"] not in keep:
                    m["alive"], m["dropped_round"] = False, r
            b.setdefault("history", []).append({"round": r, "times": t["times"], "kept": keep})
            ev = {"beam_times_us": t["times"], "kept": keep}
            self.event("beam_select", times=t["times"], alive=alive, kept=keep, exec_mode=self.exec_mode("whole"))
            if len(keep) == 1:
                win = next(m for m in b["members"] if m["id"] == keep[0])
                lost = beam_commit(doc, win, b["members"], {"evidence": ev, "member": win["id"]}, r)
                b.update(done=True, winner=win["id"], committed_round=r)
                em = self.exec_mode("whole")
                for hid in win["hints"]:
                    self.event("decision", phase="beam", hint=hid, status="accepted", measured=True, evidence=ev,
                               exec_mode=em)
                for hid in lost:
                    self.event("decision", phase="beam", hint=hid, status="rejected", measured=True, evidence=ev,
                               exec_mode=em)
                if hints_path is not None:
                    gh.save(doc, hints_path)
            self.save_beam()
            out["beam"] = {"times": t["times"], "kept": keep, "done": b.get("done", False)}
        out.update(self.round_summary(r))
        out["exec_mode"] = self.exec_mode()
        mm = [x for x in self.mode_mismatches() if x["round"] == r and x["diff"]]
        if mm:                                           # decided in another walker / SM mode than deployed now
            out["exec_mode_mismatch"] = mm
            print(json.dumps({"round": r, "graph_search_exec_mode_mismatch": mm})[:1500], flush=True)
        if getattr(self, "last_budgets", None):
            out["survival"] = self.last_budgets
            self.last_budgets = None
        return out

    # ---- beam state (graph_search_state.json in the work dir)
    def load_beam(self) -> None:
        if getattr(self, "beam_state", None) is None:
            f = self.work / "graph_search_state.json"
            try:
                self.beam_state = json.loads(f.read_text()).get("beam") or {}
            except (OSError, ValueError):
                self.beam_state = {}

    def save_beam(self) -> None:
        from .kb_resume import atomic_write_json
        atomic_write_json(self.work / "graph_search_state.json", {"beam": self.beam_state})


def scan_pairs(graph: dict, profile: dict, specs: dict, doc: dict, *, ir: str | None, min_gain_pct: float = 0.0
               ) -> list[dict]:
    """Every producer -> elementwise-consumer pair of the graph (propose_from_profile's criteria, but not greedy:
    overlapping pairs are all listed, so bn+relu is seen next to conv+bn), grouped by fused kind -> fuse hints with
    their chains and the consumers' measured time as the estimate."""
    from .kb_proposals import fuse_pair, with_fused_specs
    from .op_classes import op_class
    g = gh.unique_names(copy.deepcopy(graph))
    specs = with_fused_specs(g, specs)
    op_us = {o["name"]: o.get("us") or 0.0 for o in profile.get("ops", [])}
    total = profile.get("e2e_us") or sum(op_us.values()) or 1.0
    launch = profile.get("launch_us") or 0.0
    done = gh.decided(doc)
    outs = set(g["output"].get("tensors") or [g["output"]["tensor"]])
    readers: dict[str, list] = {}
    for o in g["ops"]:
        for t in o["inputs"]:
            readers.setdefault(t, []).append(o)
    by: dict[str, dict] = {}
    for o in g["ops"]:
        if len(o["outputs"]) != 1 or o["op"] not in specs or o["outputs"][0] in outs:
            continue
        r = readers.get(o["outputs"][0], [])
        if len(r) != 1 or r[0]["op"] not in specs or op_class(r[0]["op"]) != "elementwise":
            continue
        try:
            _, spec = fuse_pair(g, o["name"], r[0]["name"], specs)
        except (ValueError, StopIteration, KeyError):
            continue
        if f"fuse:{spec.op}" in done:
            continue
        c = by.setdefault(spec.op, {"chains": [], "gain_us": 0.0})
        c["chains"].append(gh.atoms(o["name"]) + gh.atoms(r[0]["name"]))
        c["gain_us"] += op_us.get(r[0]["name"], 0.0) + launch
    out = []
    for kind, c in by.items():
        pct = 100.0 * c["gain_us"] / total
        if pct < min_gain_pct:
            continue
        out.append(gh.fuse_hint(kind, c["chains"], {"by": "graph-prior-scan", "ir": ir, "ts": gh._now(),
                                                     "est_gain_us": round(c["gain_us"], 2),
                                                     "est_gain_pct": round(pct, 3)}))
    return out


def propose_all(profiles: list[dict], doc: dict, specs: dict, cfg) -> dict:
    """kb_graph_opt's proposal step (MB proposals + --fusion-extended) on new profiles -> {IR: graph}."""
    graphs = {}
    for p in profiles:
        g = graphs[p["ir"]] = json.loads((Path(p["ir_dir"]) / "graph.json").read_text())
        gh.merge(doc, gh.propose_from_profile(g, {o["name"]: o["us"] or 0.0 for o in p["ops"]}, specs,
                                               doc=doc, min_gain_pct=cfg.min_gain_pct, ir=p["ir"],
                                               launch_us=p.get("launch_us") or 0.0, total_us=p.get("e2e_us")))
    from . import cuda_learnings as cl
    e = cl.current()
    if e is not None:
        e.propose(graphs, profiles, doc, specs)
    return graphs


# ---------------------------------------------------------------- interleaved round

def _problem_atoms(d: Path) -> set:
    man = _manifest(d)
    return {a for o in man.get("ops", []) for a in gh.atoms(o["name"])}


class Interleaved:
    """One round in interleaved mode: the plan's problems and the hint decisions' (original, variant) problems in
    ONE KernelBlaster run; hints re-timed and decided as soon as their pair finishes; an acceptance applies at once
    (stop files for the plan problems whose ops it absorbs; fused slots cached by finish_hint).

    Injected (tests use stubs): ``start(problems) -> proc`` (``poll() / wait()``), ``read(problem) -> result|None``,
    ``measure(job, ro, rv, attempts=None) -> times``, ``finish(job, ro, rv, times) -> decision``, ``collect(problems, proc) ->
    results``. ``prog``: a kb_resume.RoundProgress (phase ``gs_inter``: budgets and finished decisions)."""

    def __init__(self, gs: GraphSearch, *, run_dir: Path, prog, start, read, measure, finish, collect,
                 scope_after=None, save_hints=None, poll_s: float = 1.0, workers: int = 2):
        self.gs, self.run_dir, self.prog = gs, Path(run_dir), prog
        self.start, self.read, self.measure, self.finish, self.collect = start, read, measure, finish, collect
        self.scope_after, self.save_hints = scope_after, save_hints
        self.poll_s, self.workers = poll_s, workers
        self.absorbed_so_far: dict[str, set] = {}

    def stop_absorbed(self, hid: str, absorbed: set, plan_dirs: list[Path], plan_irs: list[str], ir: str) -> list:
        """Stop files for the plan problems of ``ir`` whose ops are all absorbed by this round's accepted hints so far
        (their union: bn2 + downsample.1 in one problem may go to two different fusions)."""
        self.absorbed_so_far.setdefault(ir, set()).update(absorbed)
        absorbed = self.absorbed_so_far[ir]
        out = []
        for d, pir in zip(plan_dirs, plan_irs):
            if pir != ir:
                continue
            atoms = _problem_atoms(d)
            if atoms and atoms <= absorbed:
                name = _manifest(d)["name"]
                f = self.run_dir / name / "mb_stop.json"
                if not f.exists():
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_text(json.dumps({"by": "graph_search", "hint": hid, "ts": time.time(),
                                             "why": "ops absorbed by an accepted graph decision"}))
                    out.append(name)
        return out

    def run(self, plan_dirs: list[Path], plan_irs: list[str], jobs: list[dict]) -> dict:
        """jobs: prepared hint jobs (``orig`` / ``variant`` dirs, ``hint``, ``ev``, ``names``; ``early`` = nothing
        to run), each with ``gs`` = {rung, steps, used}. -> {"plan_results", "decisions", "stopped"}."""
        st = self.prog.get("gs_inter") or {}
        decided: dict[str, dict] = dict(st.get("decided") or {})
        live = [j for j in jobs if "early" not in j]
        hint_probs = [p for j in live for p in (j["orig"], j["variant"])]
        problems = hint_probs + list(plan_dirs)      # hint problems first: they start at once
        res_by: dict[str, tuple] = {}
        stopped: list[str] = list(st.get("stopped") or [])
        decisions: dict[str, dict] = {}
        for j in jobs:                               # nothing to run: decided at once (rejected / undecided)
            if "early" in j and j["hint"]["id"] not in decided:
                decided[j["hint"]["id"]] = self.finish(j, None, None, None)
                self._mark(decided, stopped)
        proc = self.start(problems) if problems else None
        futs: dict = {}
        retried: set = set()
        with ThreadPoolExecutor(max_workers=max(1, self.workers)) as ex:
            try:
                while proc is not None:
                    done = proc.poll() is not None
                    for j in live:
                        hid = j["hint"]["id"]
                        if hid in futs or hid in decided:
                            continue
                        ro, rv = self.read(j["orig"]), self.read(j["variant"])
                        if (ro is not None and rv is not None) or done:
                            ro = dict(ro or {"final_ok": False}, _dir=str(self.run_dir / _manifest(j["orig"])["name"]))
                            rv = dict(rv or {"final_ok": False},
                                      _dir=str(self.run_dir / _manifest(j["variant"])["name"]))
                            res_by[hid] = (ro, rv)
                            futs[hid] = ex.submit(self.measure, j, ro, rv)
                    # decide in completion order, in this thread (hints doc, cache, stop files)
                    for j in live:
                        hid = j["hint"]["id"]
                        f = futs.get(hid)
                        if hid in decided or f is None or not f.done():
                            continue
                        times = f.result()
                        if j.get("remeasure_infra") and hid not in retried:
                            if not done:             # lost while the run shares the GPUs: once more after it
                                continue
                            retried.add(hid)
                            futs[hid] = ex.submit(self.measure, j, *res_by[hid], 1)
                            continue
                        d = self.finish(j, *res_by[hid], times)
                        decided[hid] = d
                        if d.get("hint_status") == "accepted" and self.scope_after is not None:
                            absorbed = self.scope_after(j)
                            s = self.stop_absorbed(hid, absorbed, plan_dirs, plan_irs, j["ev"].get("ir"))
                            if s:
                                stopped += s
                                self.gs.event("stop", hint=hid, problems=s, absorbed=sorted(absorbed)[:50])
                        self._mark(decided, stopped)
                    if done and all(j["hint"]["id"] in decided for j in live):
                        break
                    time.sleep(self.poll_s)
            except BaseException:
                if proc is not None and proc.poll() is None:
                    proc.kill()
                raise
        results = self.collect(problems, proc) if proc is not None else []
        n = len(hint_probs)
        for j in jobs:
            decisions[j["hint"]["id"]] = decided.get(j["hint"]["id"])
        return {"plan_results": results[n:], "hint_results": results[:n],
                "decisions": [decisions[j["hint"]["id"]] for j in jobs], "stopped": stopped}

    def _mark(self, decided: dict, stopped: list) -> None:
        if self.save_hints is not None:
            self.save_hints()
        st = dict(self.prog.get("gs_inter") or {})
        st.update(decided=decided, stopped=stopped)
        self.prog.mark("gs_inter", st)


def decide_checkpoint(gs: GraphSearch, job: dict, ro: dict | None, rv: dict | None, times, *, cfg, doc: dict,
                      finish_std) -> dict:
    """A hint's decision at its rung checkpoint: the standard gate (``finish_std`` = kb_graph_opt.finish_hint) for
    accepted / failed / infrastructure cases; a variant that is not faster yet is ``promising`` (deferred to its next
    rung, a non-voting decision) when variant_policy's expectation test passes, else the standard rejection."""
    from .kb_proposals import _median
    h = job["hint"]
    b = job.get("gs") or {"rung": 0, "steps": 0, "used": 0}
    pr = gs.prior(h)
    t0 = time.time()
    ok = bool(ro and rv and ro.get("final_ok") and rv.get("final_ok") and times and times[0] and times[1])
    x = None
    if ok and gs.defer:
        o, v = _median(times[0]), _median(times[1])
        d = h.get("gs_deferred") or {}
        hr = next((s.get("headroom") for s in reversed(h.get("source") or []) if s.get("headroom")), None)
        x = vp.checkpoint(orig_us=o, var_us=v, accept_gain=cfg.accept_gain, rung=b["rung"],
                          steps_used=b["used"] + b["steps"], cap_steps=gs.cap_steps, hr=hr,
                          history=d.get("history") or [], traj=vp.trajectory(rv), prior=pr, defer=True)
    if x is not None and x["decision"] == "promising":
        rec = gh.get(doc, h["id"]) or h
        hist = list((rec.get("gs_deferred") or {}).get("history") or []) + [
            {"var_us": x["tests"]["var_us"], "orig_us": x["tests"]["orig_us"], "steps": b["steps"], "round": gs.round}]
        ev = dict(job["ev"], decision="promising", gain=x["tests"].get("gain"), why=x["why"], expectation=x,
                  prior=pr, rung=b["rung"], steps=b["steps"], original_us=x["tests"]["orig_us"],
                  variant_us=x["tests"]["var_us"],
                  remeasured={"original_us": times[0], "variant_us": times[1]})
        rec = gh.record_decision(doc, h["id"], ev)
        rec["gs_deferred"] = {"rung": b["rung"], "steps_used": b["used"] + b["steps"], "history": hist,
                              "sources_dir": str(Path(rv["_dir"]) / "best" / "slots"), "why": x["why"]}
        res = {"hint": h["id"], "status": "promising", "hint_status": rec["status"], "gain": x["tests"].get("gain"),
               "original_us": x["tests"]["orig_us"], "variant_us": x["tests"]["var_us"], "ops": job.get("names")}
    else:
        res = finish_std(job, ro, rv, times)
        rec = gh.get(doc, h["id"])
        if rec is not None and "gs_deferred" in rec and res.get("hint_status") in ("accepted", "rejected"):
            rec["gs_deferred_done"] = rec.pop("gs_deferred")
        if x is not None:
            res["expectation"] = x
    calls = ((ro or {}).get("llm_calls") or 0) + ((rv or {}).get("llm_calls") or 0)
    cost = usd((ro or {}).get("llm_usage")) + usd((rv or {}).get("llm_usage"))
    gs.event("decision", phase=f"rung{b['rung']}", hint=h["id"], exec_mode=gs.exec_mode("partition"), status=res.get("status"),
             hint_status=res.get("hint_status"), measured=bool(ok), gain=res.get("gain"), steps=b["steps"],
             steps_used=b["used"] + b["steps"], rung=b["rung"], llm_calls=calls, usd=round(cost, 4), prior=pr,
             evidence={"original_us": res.get("original_us"), "variant_us": res.get("variant_us"),
                       "why": (x or {}).get("why") or res.get("why"), "tests": (x or {}).get("tests")},
             decide_s=round(time.time() - t0, 2), ir=job["ev"].get("ir"))
    return res


def variant_seed_override(h: dict, vplan: dict, seeds: dict, tags: dict, specs: dict) -> list[str]:
    """A promising (deferred) hint continues where its last rung stopped: its variant's new slots are seeded from
    that rung's best sources. No-op otherwise. -> the slots overridden."""
    d = (h.get("gs_deferred") or {}).get("sources_dir")
    if not d or not Path(d).is_dir():
        return []
    from .kb_export import slot_id_of
    sids = set()
    for o in vplan["graph"]["ops"]:
        try:
            sids.add(slot_id_of(o, specs))
        except Exception:  # noqa: BLE001
            continue
    out = []
    for f in sorted(Path(d).glob("*.cu")):
        if f.stem in sids:
            seeds[f.stem], tags[f.stem] = f.read_text(), f"graph-search:rung{h['gs_deferred'].get('rung', 0)}"
            out.append(f.stem)
    return out


def filter_winners(winners: Path, out: Path, drop=("__variant",)) -> Path | None:
    """winners.json without the hint decisions' variant problems (a variant's slots are cached only on
    acceptance, by finish_hint); the originals' winners are ordinary kernels of the current graph."""
    if not Path(winners).exists():
        return None
    w = json.loads(Path(winners).read_text())
    w["winners"] = [r for r in w.get("winners", []) if not any(x in str(r.get("problem_id") or r.get("problem") or "")
                                                               for x in drop)]
    Path(out).write_text(json.dumps(w, indent=1))
    return Path(out)


# ---------------------------------------------------------------- graph-diverse beam

def beam_members(doc: dict, gs: GraphSearch, n: int) -> list[dict]:
    """Members from the proposed hints: m0 baseline (the current accepted set), m1 + high-prior fusions,
    m2 + the uncertain ones (p_mean >= 0.3), m3 + layout folds, m4 + attention fusions; identical sets merged,
    at most ``n``."""
    prop = [h for h in doc["hints"] if h.get("status") == "proposed" and h.get("target", "cuda") == "cuda"]
    fuse = [h for h in prop if h["kind"] == "fuse"]
    pri = {h["id"]: gs.prior(h) or gp.posterior(0, 0) for h in fuse}
    hi = [h["id"] for h in fuse if gs.eligible(pri[h["id"]])]
    unc = [h["id"] for h in fuse if h["id"] not in hi and pri[h["id"]]["p_mean"] >= 0.3]
    fold = [h["id"] for h in prop if h["kind"] == "fold_layout"]
    attn = [h["id"] for h in prop if h["kind"] == "fuse_attention"]
    cand = [("baseline", []), ("high-prior", hi), ("uncertain", hi + unc), ("layout-fold", hi + fold),
            ("attention", hi + attn)]
    out, seen = [], set()
    for origin, hs in cand:
        key = tuple(sorted(hs))
        if key in seen:
            continue
        seen.add(key)
        out.append({"id": f"m{len(out)}", "origin": origin, "hints": list(key), "alive": True})
    return out[:max(1, n)]


def member_doc(doc: dict, member: dict) -> dict:
    d = copy.deepcopy(doc)
    for hid in member["hints"]:
        h = gh.get(d, hid)
        if h is not None:
            h["status"] = "accepted"
            for key in ("chains", "groups"):              # applies wherever the pattern matches, as on acceptance
                if key in h:
                    h.setdefault("tried_" + key, []).extend(h.pop(key))
    return d


def relabel(profile: dict, mid: str) -> dict:
    p = dict(profile)
    p["ir_base"], p["ir"], p["member"] = profile["ir"], f"{profile['ir']}~{mid}", mid
    return p


def halve(times: dict[str, float | None], alive: list[str], *, keep_min: int = 1) -> list[str]:
    """Successive halving over members by paired whole-model time (None = failed: dropped first)."""
    ranked = sorted(alive, key=lambda m: (times.get(m) is None, times.get(m) or 0.0))
    k = max(keep_min, (len(ranked) + 1) // 2)
    return ranked[:k]


def beam_commit(doc: dict, member: dict, members: list[dict], evidence: dict, r: int) -> list[str]:
    """The surviving member's hints become accepted (decision ``beam``); hints only the losers carried are rejected."""
    won = set(member["hints"])
    for hid in sorted(won):
        if (gh.get(doc, hid) or {}).get("status") == "proposed":
            gh.record_decision(doc, hid, {"decision": "accepted", "by": "graph-beam", "round": r, "measured": True,
                                          "ts": gh._now(), **evidence})
    lost = sorted({x for m in members for x in m["hints"]} - won)
    for hid in lost:
        if (gh.get(doc, hid) or {}).get("status") == "proposed":
            gh.record_decision(doc, hid, {"decision": "rejected", "by": "graph-beam", "round": r, "measured": True,
                                          "ts": gh._now(), **evidence})
    return lost


def env_flag(name: str) -> bool:
    return os.getenv(name, "").lower() in ("1", "true", "yes", "on")


def beam_time(R: Path, members: list[dict], profiles: list[dict], doc: dict, cache, specs: dict, *, arch: str,
              kbeval=None, gpu_model: str | None = None, reps: int = 5) -> dict:
    """Paired whole-model re-timing of the alive members on one GPU, interleaved (kb_proposals.remeasure) ->
    {member id: median us summed over the IRs (None: a member failed)}, plus the raw runs."""
    from .kb_export import export_problem
    from .kb_ingest import cache_plan
    from .kb_plan import hinted_ir
    from .kb_proposals import _median, remeasure, with_fused_specs, write_variant_ir

    alive = [m for m in members if m.get("alive")]
    tot: dict[str, float | None] = {m["id"]: 0.0 for m in alive}
    raw: dict = {}
    for p in profiles:
        probs, ids = [], []
        for m in alive:
            w = Path(R) / "beam_time" / m["id"] / p["ir"]
            try:
                _, sp, g = hinted_ir(Path(p["source_ir"]), member_doc(doc, m), w / "hinted", dict(specs))
                sp = with_fused_specs(g, sp)
                plan = cache_plan(g, cache, f"sm_{arch}", sp)
                d = write_variant_ir(Path(p["source_ir"]), plan["graph"], w / "ir")
                probs.append((export_problem(d, w / "whole", slot_sources=plan["sources"],
                                             slot_source_tags=plan["tags"], sub_sources=plan["sub_sources"],
                                             specs=sp, strategy="whole"), {}))
                ids.append(m["id"])
            except Exception as e:  # noqa: BLE001 -- a member that cannot be built is out
                tot[m["id"]] = None
                raw.setdefault(p["ir"], {})[m["id"]] = {"error": f"{type(e).__name__}: {e}"[:300]}
        if not probs:
            continue
        t = remeasure(probs, arch=arch, reps=reps, kbeval=kbeval, gpu_model=gpu_model,
                      work=Path(R) / "beam_time" / "remeasure" / p["ir"] if kbeval is not None else None)
        for mid, xs in zip(ids, t):
            med = _median(xs)
            raw.setdefault(p["ir"], {})[mid] = {"runs": xs, "median_us": med}
            if med is None or tot.get(mid) is None:
                tot[mid] = None
            else:
                tot[mid] += med
    return {"times": tot, "raw": raw}


def fold_seeds(h: dict, g0: dict, vplan: dict, base: dict, seeds: dict, tags: dict, specs: dict, *,
               cap_mb: int = 8192) -> dict:
    """Composite seeds for a fold_layout variant (``--graph-layout-fold on``): the incumbent's copies + GEMM kernels
    back to back inside the new bmm_strided slot (composite_seed), so the variant starts at parity."""
    if h.get("kind") != "fold_layout":
        return {}
    from .composite_seed import composite_sources
    from .kb_export import DEFAULT_HAND_CU, CudaLibrary, _hand_source, _problem_dtype

    dtype = _problem_dtype([o["op"] for o in g0["ops"] if o["op"] not in ("view", "chunk2_c1")])
    lib = CudaLibrary(_hand_source(DEFAULT_HAND_CU, dtype))
    srcs, crep = composite_sources(g0, vplan["graph"], specs, sources=base["sources"], sub_sources=base["sub_sources"],
                                   lib=lib, verified_consts=set(), cap_bytes=int(cap_mb) << 20)
    rep = {}
    for sid, r in crep.items():
        if sid in vplan["sources"]:
            rep[sid] = {"seed": "cache", **r}
            continue
        if sid in srcs:
            seeds[sid] = srcs[sid]
            tags[sid] = ("composite:" + "+".join(r["stages"][0]))[:200] if r.get("stages") else "composite"
        rep[sid] = {"seed": "composite" if sid in srcs else "hand", **r}
    return rep


def propose_folds(graphs: dict, profiles: list[dict], doc: dict, gs: GraphSearch) -> list[dict]:
    """fold_layout proposals (layout_fold matcher) with a roofline estimate (variant_policy.fold_headroom) and the
    prior; merged into ``doc`` by the caller."""
    from .layout_fold import fold, fold_hint
    out = []
    done = gh.decided(doc)
    for p in profiles:
        g = graphs.get(p["ir"])
        if g is None:
            continue
        op_us = {o["name"]: o.get("us") or 0.0 for o in p.get("ops", [])}
        total = p.get("e2e_us") or sum(op_us.values()) or 1.0
        try:
            gu = gh.unique_names(copy.deepcopy(g))
            _, rep = fold(gu, op_us=op_us)
        except Exception as e:  # noqa: BLE001 -- a proposer never breaks a round
            gs.event("proposer_error", kind="fold_layout", error=f"{type(e).__name__}: {e}"[:300])
            continue
        if not rep.get("folded"):
            continue
        h = fold_hint({"by": "mb-graph-search", "ir": p["ir"], "ts": gh._now()}, [f["op"] for f in rep["folded"]])
        if h["id"] in done:
            continue
        try:
            hr = vp.fold_headroom(rep["folded"], op_us, arch=str(gs.cfg.arch))
            vp.annotate(h, hr=hr, total_us=total, prior=gs.prior(h))
        except Exception:  # noqa: BLE001 -- keep the hint, unranked
            pass
        out.append(h)
    return out
