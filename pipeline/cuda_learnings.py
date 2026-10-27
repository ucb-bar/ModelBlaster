"""The three CUDA-only-baseline learnings as opt-in hooks of the graph optimiser (``kb_graph_opt``).

l3_farm/analysis/cuda_learnings_impl.md has the design, flags and measurements. All defaults leave the flow as it
was (and existing work dirs keep their ``kb_resume`` round key):

    --reuse-winners off|siblings|global   winner reuse: canonical plan keys (flags such as ``reverse`` are not shape;
                                          --reuse-canon), a zero-LLM cross-application pass after every round
                                          (kb_reuse.cross_apply), sibling / global seeds for new slot problems
                                          (kb_reuse.add_seed_candidates); with it KernelBlaster rollouts keep their
                                          own verified gains (--rollout-accept local, default with reuse on)
    --graph-concurrency off|on            independent ops as concurrent CUDA-graph branches (kb_concurrency), with
    --sm-budget-policy split|none|strict  an SM budget per branch and a deadlock watchdog (sequential fallback)
    --fusion-extended off|siblings|epilogue|attention|all
                                          sibling / pool-epilogue / flash-attention fusions; fused variants keep their
                                          per-shape stage kernels (fusion_ext)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import fusion_ext
from . import kb_reuse

CFG_DEFAULTS = {"reuse_winners": "off", "reuse_canon": "default", "rollout_accept": "auto",
                "graph_concurrency": "off", "sm_budget_policy": "split", "conc_timeout_s": 30.0,
                "fusion_extended": "off"}


class Ext:
    """This run's switches; every method is a no-op with the defaults."""

    def __init__(self, cfg, work: Path):
        self.cfg, self.work = cfg, Path(work)
        g = lambda k: getattr(cfg, k, CFG_DEFAULTS[k])         # noqa: E731
        self.reuse = (g("reuse_winners") or "off").lower()
        if self.reuse not in kb_reuse.MODES:
            raise ValueError(f"--reuse-winners must be {'|'.join(kb_reuse.MODES)}, not {self.reuse!r}")
        self.canon_spec = g("reuse_canon")
        self.rollout = (g("rollout_accept") or "auto").lower()
        self.conc = (g("graph_concurrency") or "off").lower()
        self.policy = (g("sm_budget_policy") or "split").lower()
        self.conc_timeout = float(g("conc_timeout_s") or 30.0)
        self.fusion = g("fusion_extended") or "off"
        fusion_ext.enabled(self.fusion)                      # validates
        from .kb_concurrency import resolve
        resolve("on" if self.conc == "auto" else self.conc, self.policy)     # validates
        self.reports: dict = {"reuse": [], "seeds": {}, "fusion_proposals": []}

    # ---- process-wide settings (exports of every kind, KernelBlaster's subprocesses)
    def apply_env(self) -> None:
        if self.conc != "off":
            # auto: on until a round's back-to-back comparison says the sequential walker is faster (conc_auto)
            os.environ["MB_GRAPH_CONCURRENCY"] = self.reports.get("conc_auto_choice") or \
                ("on" if self.conc == "auto" else self.conc)
            os.environ["MB_SM_BUDGET_POLICY"] = self.policy
            os.environ["MB_CONC_TIMEOUT_S"] = str(self.conc_timeout)
        if self.rollout == "local" or (self.rollout == "auto" and self.reuse != "off"):
            os.environ["KERNELBLASTER_MULTISLOT_ROLLOUT_ACCEPT"] = "local"

    def round_key(self) -> dict:
        """Only the switches that are set: existing work dirs keep their key."""
        k = {}
        if self.reuse != "off":
            k["reuse_winners"] = [self.reuse, self.canon_spec]
        if self.conc != "off":
            k["graph_concurrency"] = [self.conc, self.policy]
        if fusion_ext.enabled(self.fusion):
            k["fusion_extended"] = self.fusion
        if self.rollout not in ("auto", "composite"):
            k["rollout_accept"] = self.rollout
        return k

    # ---- plan (reuse a): canonical item keys
    def plan_kw(self, specs: dict) -> dict:
        if self.reuse == "off":
            return {}
        from .kb_plan import parse_canon_flags
        return {"canon": parse_canon_flags(self.canon_spec), "specs": specs}

    # ---- seeds (reuse c)
    def impl_dirs(self) -> list[Path]:
        d = [self.work / "knowledge" / "impl"]
        seed = getattr(self.cfg, "knowledge_global_seed", "") or ""
        if seed:
            d.append(Path(seed) / "impl")
        return [x for x in d if x.exists()]

    def seed_problems(self, dirs: list[Path], cache, specs: dict) -> list[Path]:
        if self.reuse == "off":
            return dirs
        for d in dirs:
            try:
                r = kb_reuse.add_seed_candidates(d, cache, arch=self.cfg.arch, specs=specs, mode=self.reuse,
                                                 impl_dirs=self.impl_dirs())
            except Exception as e:  # noqa: BLE001 -- seeding is best effort
                r = {"error": f"{type(e).__name__}: {e}"[:300]}
            if r:
                self.reports["seeds"][Path(d).name] = r
        return dirs

    def wants_after_round(self) -> bool:
        return self.reuse != "off" or self.conc == "auto"

    # ---- after a round's ingest (reuse b): zero-LLM cross-application per IR; --graph-concurrency auto
    def after_round(self, R: Path, profiles: list[dict], doc: dict, cache, specs: dict, kbeval=None,
                    gpu_model: str | None = None) -> list[dict]:
        out = []
        if self.conc == "auto":
            out.append(self.conc_auto(R, profiles, doc, cache, specs, kbeval, gpu_model))
        if self.reuse == "off":
            return out
        for p in profiles:
            try:
                rep = kb_reuse.cross_apply(Path(p["source_ir"]), doc, cache, arch=self.cfg.arch, specs=specs,
                                           work=Path(R) / "reuse" / p["ir"], profile=p, mode=self.reuse,
                                           impl_dirs=self.impl_dirs(), kbeval=kbeval, gpu_model=gpu_model)
            except Exception as e:  # noqa: BLE001 -- never let the pass break a round
                rep = {"ir": p.get("ir"), "error": f"{type(e).__name__}: {e}"[:800]}
            out.append({k: rep.get(k) for k in ("ir", "changed", "reverted", "guard", "evals", "error")} |
                       {"kinds": [{k: v for k, v in x.items() if k not in ("table",)} for x in rep.get("kinds", [])]})
            print(json.dumps({"reuse": out[-1]}, default=str)[:2000], flush=True)
        self.reports["reuse"] += out
        return out

    def conc_auto(self, R: Path, profiles: list[dict], doc: dict, cache, specs: dict, kbeval=None,
                  gpu_model: str | None = None) -> dict:
        """--graph-concurrency auto: every IR as deployed now, sequential vs concurrent walker, built and re-timed back
        to back; concurrency stays on only if no IR is slower with it (8_ResNetBasicBlock's two convs: +2.4%)."""
        from .kb_export import export_problem
        from .kb_ingest import cache_plan
        from .kb_plan import hinted_ir
        from .kb_proposals import _median, remeasure, with_fused_specs, write_variant_ir
        rec = {"conc_auto": {}}
        choice = "on"
        for p in profiles:
            w = Path(R) / "conc_auto" / p["ir"]
            try:
                _, sp, g = hinted_ir(Path(p["source_ir"]), doc, w / "hinted", dict(specs))
                sp = with_fused_specs(g, sp)
                plan = cache_plan(g, cache, f"sm_{self.cfg.arch}", sp)
                d = write_variant_ir(Path(p["source_ir"]), plan["graph"], w / "ir")
                probs = [export_problem(d, w / f"p_{c}", slot_sources=plan["sources"], slot_source_tags=plan["tags"],
                                        sub_sources=plan["sub_sources"], specs=sp, strategy="whole", concurrency=c,
                                        sm_budget_policy=self.policy) for c in ("off", "on")]
                t = remeasure([(x, {}) for x in probs], arch=self.cfg.arch, reps=3, kbeval=kbeval, gpu_model=gpu_model,
                              work=w / "remeasure" if kbeval is not None else None)
                seq, conc = _median(t[0]), _median(t[1])
                r = {"seq_us": seq, "conc_us": conc}
                if conc is None or (seq is not None and conc > seq):
                    choice = "off"
            except Exception as e:  # noqa: BLE001 -- keep the current setting
                r = {"error": f"{type(e).__name__}: {e}"[:400]}
            rec["conc_auto"][p["ir"]] = r
        rec["conc_auto"]["choice"] = choice
        self.reports["conc_auto_choice"] = choice
        os.environ["MB_GRAPH_CONCURRENCY"] = choice
        print(json.dumps(rec, default=str)[:1500], flush=True)
        return rec

    # ---- proposals (fusion)
    def propose(self, graphs: dict, profiles: list[dict], doc: dict, specs: dict) -> None:
        if not fusion_ext.enabled(self.fusion):
            return
        from . import graph_hints as gh
        for p in profiles:
            g = graphs.get(p["ir"])
            if g is None:
                continue
            try:
                new = fusion_ext.propose(g, p, specs, doc, mode=self.fusion, min_gain_pct=self.cfg.min_gain_pct,
                                         ir=p["ir"])
            except Exception as e:  # noqa: BLE001 -- a proposer never breaks a round
                print(f"[kb_graph_opt] fusion_ext proposer: {type(e).__name__}: {e}", flush=True)
                continue
            gh.merge(doc, new)
            self.reports["fusion_proposals"] += [h["id"] for h in new]

    def decide_kinds(self) -> tuple[str, ...]:
        return fusion_ext.decide_kinds(self.fusion)

    def summary(self) -> dict:
        return {"reuse_winners": self.reuse, "graph_concurrency": self.conc, "sm_budget_policy": self.policy,
                "fusion_extended": self.fusion, **self.reports}


# the running flow's switches, for the module-level helpers of kb_graph_opt (scope / footprint / prepare_hint)
CURRENT: dict = {"ext": None}


def current() -> Ext | None:
    return CURRENT.get("ext")


def ext_kind(h: dict) -> bool:
    return h.get("kind") in ("fuse_siblings", "fuse_attention")


def variant_split() -> bool:
    """prepare_hint: build fused variants with the cache's coverage split (per-shape stage kernels)."""
    e = current()
    return bool(e is not None and fusion_ext.inherit(e.fusion))
