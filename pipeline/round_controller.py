"""Round controller: ``kb_graph_opt --round-controller off|fixed|graph-first|auto`` (docs/round_controller.md).

The flow's graph moves (``--merge-sibling-convs auto``, ``--region-windows auto``, ``--region-loop on``, the fusion / split hints,
``--graph-rewrite agent``) are separate flags, and the round always did the same thing: profile, propose everything, kernel steps
(KernelBlaster plan run), then the paired hint decisions. The controller makes the choice per round boundary explicit. At the start of
every round (after the profile and the cheap CPU proposals, before any GPU / LLM spend) it picks the MOVE SET of the round:

* ``kernel``       the plan's KernelBlaster rollouts (normal kernel steps);
* ``graph``        the graph proposals' paired decisions (hints, merge, windows) and the LLM proposer (``--graph-rewrite agent``);
* ``region_loop``  the region loop (KernelBlaster ``--region-loop on``) on this round's KernelBlaster runs: the plan's region slots and
                   the variants of the region hints decided this round; needs ``--region-loop on`` (the flag stays the capability switch);
* ``stop``         no round: the loop ends (recorded as fixpoint).

The flags still say WHAT MAY RUN; the controller says WHEN. ``off`` (default) runs none of this: no key, no record, every output
byte-identical to the flow without the controller. dtype handling, the correctness gates (graph_verify, region gate, hint
verification, the paired decision) and ``--quant`` / arch are never looked at: the controller only chooses which of them run when.

Policies
  ``fixed``        the flow's order: kernel + graph (+ region loop when its flag is on) every round;
  ``graph-first``  the first round of a run is graph-only (when it has decidable graph candidates), kernel steps start the next round on
                   the restructured graph; afterwards as ``fixed``;
  ``auto``         the rule in :meth:`RoundController.decide_auto` (no LLM): unfused small-op time share, live candidates and their
                   estimated gains, the last round's kernel vs graph gain, rounds / USD left.
  all non-off      the BUDGET GUARD: a move whose last two rounds (in which it had work) each gained < ``--round-controller-min-gain``
                   (default 3 % of the model's partition time) is stopped; every move stopped (or nothing left to decide) = stop.

Everything is recorded with its reasons under ``controller`` in ``rNN/round.json`` and ``round_controller`` in ``summary.json``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

ENV = "MB_ROUND_CONTROLLER"
ENV_MIN_GAIN = "MB_RC_MIN_GAIN"
ENV_BUDGET_USD = "MB_RC_BUDGET_USD"
ENV_SMALL_US = "MB_RC_SMALL_US"
POLICIES = ("off", "fixed", "graph-first", "auto")
MOVES = ("kernel", "graph", "region_loop")
GAIN_KEY = {"kernel": "kernel_pct", "graph": "graph_pct", "region_loop": "region_pct"}
DEFAULT_MIN_GAIN = 0.03          # fraction of the model's partition time
DEFAULT_SMALL_US = 15.0          # an unfused op at or below this is "small" (launch-bound on a B200)
SMALL_CLASSES = ("elementwise", "data_movement", "reduction")
GRAPH_FIRST_PCT = 8.0            # auto, first round: est. graph gain (%) that makes the round graph-only
SMALL_SHARE_HI = 0.25            # ... or this share of time in unfused small ops (with some candidate)
GRAPH_DOMINATES = 2.0            # auto: last graph gain >= 2 x last kernel gain -> graph-only round (restructure before tuning)
TOP_K = 3                        # candidates summed for the estimated gain of a class


def resolve_policy(cfg=None) -> str:
    """``--round-controller`` wins, else ``$MB_ROUND_CONTROLLER``, else off. Validated."""
    v = getattr(cfg, "round_controller", None)
    if v is None:
        v = os.environ.get(ENV, "off")
    v = str(v).strip().lower().replace("_", "-") or "off"
    if v in ("0", "false", "no"):
        v = "off"
    if v not in POLICIES:
        raise ValueError(f"--round-controller must be one of {POLICIES}, not {v!r}")
    return v


def _num(cfg, attr: str, env: str, default: float, *, lo: float = 0.0) -> float:
    v = getattr(cfg, attr, None)
    if v is None or v == 0 or v == "":
        v = os.environ.get(env, "") or default
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{attr.replace('_', '-')} / ${env}: {v!r} is not a number") from None
    if v < lo:
        raise ValueError(f"{attr.replace('_', '-')} / ${env}: {v} < {lo}")
    return v


@dataclass
class Move:
    kernel: bool = True
    graph: bool = True
    region_loop: bool = False
    stop: bool = False
    deferred: bool = False          # a component was held back to a later round (not "nothing to do"): never a fixpoint
    reasons: list = field(default_factory=list)
    inputs: dict = field(default_factory=dict)

    def label(self) -> str:
        if self.stop:
            return "stop"
        return "+".join(m for m, on in (("kernel", self.kernel), ("graph", self.graph), ("region_loop", self.region_loop)) if on) or "stop"

    def to_dict(self, policy: str) -> dict:
        return {"policy": policy, "move": self.label(), "kernel": self.kernel, "graph": self.graph, "region_loop": self.region_loop,
                "stop": self.stop, "deferred": self.deferred, "reasons": list(self.reasons), "inputs": self.inputs}

    @classmethod
    def from_dict(cls, d: dict) -> "Move":
        return cls(kernel=bool(d["kernel"]), graph=bool(d["graph"]), region_loop=bool(d["region_loop"]), stop=bool(d["stop"]), deferred=bool(d.get("deferred")),
                   reasons=list(d.get("reasons") or []), inputs=dict(d.get("inputs") or {}))


# ------------------------------------------------------------------------------------------------ the state the flow already has
def small_unfused_share(profiles: list, small_us: float) -> dict:
    """Share of the profiled time in unfused SMALL ops: not ``fused__`` / ``region__``, class elementwise / data-movement / reduction,
    at most ``small_us`` each. From the per-op profile (name, op, us) every round already takes."""
    from .op_classes import op_class
    small = total = 0.0
    n_small = n_ops = 0
    for p in profiles:
        for o in p.get("ops", []):
            us = float(o.get("us") or 0.0)
            kind = str(o.get("op", ""))
            total += us
            n_ops += 1
            if kind.startswith(("fused__", "region__")):
                continue
            if us <= small_us and op_class(kind) in SMALL_CLASSES:
                small += us
                n_small += 1
    return {"share": round(small / total, 4) if total else 0.0, "n_small": n_small, "n_ops": n_ops, "small_us": small_us}


def live_region_ops(profiles: list) -> int:
    return sum(1 for p in profiles for o in p.get("ops", []) if str(o.get("op", "")).startswith("region__"))


def _est(h: dict) -> float:
    return max((s.get("est_gain_pct") or 0.0) for s in h.get("source", [{}]) or [{}])


def candidate_classes(hints: list, min_gain_pct: float) -> dict:
    """The decidable hints (status proposed) by class: ``merge`` (merge_sibling_convs), ``region`` (fuse_region, windows included),
    ``graph`` (everything else). Per class: count, the estimated gain (sum of the best ``TOP_K`` est_gain_pct, <= 100) and the best id."""
    out = {c: {"n": 0, "est_gain_pct": 0.0, "best": None} for c in ("graph", "merge", "region")}
    gains: dict = {c: [] for c in out}
    for h in hints:
        e = _est(h)
        if e < min_gain_pct:
            continue
        c = "merge" if h["kind"] == "merge_sibling_convs" else "region" if h["kind"] == "fuse_region" else "graph"
        gains[c].append((e, h["id"]))
    for c, g in gains.items():
        g.sort(key=lambda x: -x[0])
        out[c] = {"n": len(g), "est_gain_pct": round(min(100.0, sum(x[0] for x in g[:TOP_K])), 2), "best": g[0][1] if g else None}
    return out


def realized(rec: dict, doc_kinds: dict) -> dict:
    """Per-move gain of a finished round, in percent of the round's profiled partition time (a lower bound of the model gain: the
    kernel gain is KernelBlaster's baseline -> final on the plan's problems, the graph / region gain the accepted paired decisions)."""
    total = sum((rec.get("partition_us") or {}).values()) or 0.0
    k = sum(max(0.0, (x.get("baseline_partition_us") or 0.0) - (x.get("final_partition_us") or 0.0))
            for x in rec.get("kb_runs", []) if x.get("final_ok"))
    g = r = 0.0
    n_g = n_r = 0
    for d in rec.get("decisions", []):
        if d.get("hint_status") != "accepted" or d.get("original_us") is None or d.get("variant_us") is None:
            continue
        delta = max(0.0, d["original_us"] - d["variant_us"])
        if doc_kinds.get(d["hint"]) == "fuse_region":
            r, n_r = r + delta, n_r + 1
        else:
            g, n_g = g + delta, n_g + 1
    pct = (lambda x: round(100.0 * x / total, 3) if total else 0.0)
    return {"kernel_pct": pct(k), "graph_pct": pct(g), "region_pct": pct(r), "accepted_graph": n_g, "accepted_region": n_r}


class RoundController:
    """The flow's handle (same shape as ``MergeHook`` / ``RegionLoopHook``). Inert when off."""

    def __init__(self, cfg, work=None, *, region_loop_active: bool = False):
        self.cfg, self.work = cfg, work
        self.policy = resolve_policy(cfg)
        self.rl_active = bool(region_loop_active)
        self.min_gain = _num(cfg, "round_controller_min_gain", ENV_MIN_GAIN, DEFAULT_MIN_GAIN * 100) / 100.0   # a percent on the CLI
        self.budget_usd = _num(cfg, "round_controller_budget_usd", ENV_BUDGET_USD, 0.0)
        self.small_us = _num(cfg, "round_controller_small_us", ENV_SMALL_US, DEFAULT_SMALL_US)
        self.log: list[dict] = []
        self.stop_reason: str | None = None

    # ---- switch
    @property
    def on(self) -> bool:
        return self.policy != "off"

    def active(self) -> bool:
        return self.on

    def round_key(self) -> dict:
        if not self.on:
            return {}
        return {"round_controller": self.policy,
                **({"rc_min_gain": self.min_gain} if abs(self.min_gain - DEFAULT_MIN_GAIN) > 1e-12 else {}),
                **({"rc_budget_usd": self.budget_usd} if self.budget_usd else {}),
                **({"rc_small_us": self.small_us} if self.small_us != DEFAULT_SMALL_US else {})}

    def summary(self) -> dict:
        return {"policy": self.policy, "min_gain_pct": round(100 * self.min_gain, 3), "budget_usd": self.budget_usd or None,
                "small_us": self.small_us, "stop_reason": self.stop_reason, "rounds": list(self.log)}

    # ---- history of this configuration (state["rounds"] entries with the controller's record)
    @staticmethod
    def history(state: dict, kh: str) -> list[dict]:
        return [x["controller"] for x in state.get("rounds", []) if x.get("key") == kh and isinstance(x.get("controller"), dict)]

    def stopped(self, hist: list[dict]) -> dict:
        """BUDGET GUARD: move -> reason for every move whose last two WORKING rounds each gained less than ``min_gain``."""
        out = {}
        for m in MOVES:
            last = [h for h in hist if (h.get("worked") or {}).get(m)][-2:]
            gains = [(h.get("gain") or {}).get(GAIN_KEY[m], 0.0) for h in last]
            if len(last) == 2 and all(g < 100 * self.min_gain for g in gains):
                out[m] = (f"budget guard: {m} gained {gains[0]}% and {gains[1]}% in its last two rounds "
                          f"(< {round(100 * self.min_gain, 3)}%)")
        return out

    def pre_stop(self, hist: list[dict], spent_usd: float) -> str | None:
        """Checked before the round's profile (no spend yet): USD budget used up, or every move already stopped by the guard."""
        if self.budget_usd and spent_usd >= self.budget_usd:
            return f"budget: spent ${spent_usd:.2f} of ${self.budget_usd:.2f}"
        st = self.stopped(hist)
        if "kernel" in st and "graph" in st:
            return "budget guard: kernel and graph moves both stopped (" + st["kernel"] + "; " + st["graph"] + ")"
        return None

    # ---- the features
    def features(self, *, r: int, first: bool, rounds_left: int, profiles: list, hints: list, min_gain_pct: float, hist: list[dict],
                 spent_usd: float, prev_total: float | None = None, cur_total: float | None = None) -> dict:
        cand = candidate_classes(hints, min_gain_pct)
        small = small_unfused_share(profiles, self.small_us)
        last = hist[-1] if hist else None
        lg = (last or {}).get("gain") or {}
        return {"round": r, "first_round": first, "rounds_left": rounds_left, "small_unfused": small, "candidates": cand,
                "live_region_ops": live_region_ops(profiles),
                "last_gain_pct": ({"kernel": lg.get("kernel_pct"), "graph": round((lg.get("graph_pct") or 0.0) + (lg.get("region_pct") or 0.0), 3)}
                                  if lg else None),
                "prev_round_realized_pct": (round(100.0 * (prev_total - cur_total) / prev_total, 3)
                                            if prev_total and cur_total is not None else None),
                "last_move": (last or {}).get("move"), "spent_usd": round(spent_usd, 4),
                "budget_left_usd": round(self.budget_usd - spent_usd, 4) if self.budget_usd else None}

    # ---- the decision
    def decide(self, f: dict, hist: list[dict]) -> Move:
        mv = Move(inputs=f)
        st = self.stopped(hist)
        cand = f["candidates"]
        g_est = cand["graph"]["est_gain_pct"] + cand["merge"]["est_gain_pct"] + cand["region"]["est_gain_pct"]
        mv.inputs = {**f, "graph_est_gain_pct": round(min(100.0, g_est), 2), "stopped": st}
        graph_ok = "graph" not in st and (g_est > 0 or self.policy == "fixed")
        kernel_ok = "kernel" not in st
        for m, why in st.items():
            mv.reasons.append(why)
        if self.policy == "fixed":
            mv.kernel, mv.graph = kernel_ok, "graph" not in st
            mv.reasons.append("fixed: the flow's order (kernel steps, then the graph decisions)")
        elif self.policy == "graph-first":
            mv.kernel, mv.graph = kernel_ok, "graph" not in st
            if f["first_round"] and graph_ok and g_est >= 0 and any(cand[c]["n"] for c in cand):
                mv.kernel, mv.deferred = False, True
                mv.reasons.append(f"graph-first: first round graph-only ({sum(cand[c]['n'] for c in cand)} candidate(s), est. gain "
                                  f"{mv.inputs['graph_est_gain_pct']}%); kernel steps start next round on the restructured graph")
            else:
                mv.reasons.append("graph-first: " + ("no decidable graph candidate in the first round, kernel + graph as fixed"
                                                     if f["first_round"] else "after the first round: kernel + graph as fixed"))
        else:
            self._auto(mv, f, hist, kernel_ok, graph_ok, g_est)
        # region loop: a capability of the flag; runs where it has something to act on
        if self.rl_active and not mv.stop:
            live = f["live_region_ops"] > 0
            pend = cand["region"]["n"] > 0
            if "region_loop" in st:
                mv.reasons.append(st["region_loop"])
            elif self.policy in ("fixed", "graph-first"):
                mv.region_loop = mv.kernel or mv.graph
                mv.reasons.append("region_loop: flag on, " + self.policy + " keeps it with every round")
            elif (live and mv.kernel) or (pend and mv.graph):
                mv.region_loop = True
                mv.reasons.append(f"region_loop: {f['live_region_ops']} live region op(s)" if live and mv.kernel
                                  else f"region_loop: {cand['region']['n']} region candidate(s) (est. {cand['region']['est_gain_pct']}%) decided this round")
            else:
                mv.reasons.append("region_loop: no live region op and no region candidate to decide: not run")
        if not mv.kernel and not mv.graph:
            mv.stop = True
            mv.region_loop = False
            if not any("stop" in x for x in mv.reasons):
                mv.reasons.append("stop: nothing left to run (kernel and graph moves are out)")
        return mv

    def _auto(self, mv: Move, f: dict, hist: list, kernel_ok: bool, graph_ok: bool, g_est: float) -> None:
        small = f["small_unfused"]["share"]
        lg = f["last_gain_pct"]
        mv.kernel, mv.graph = kernel_ok, graph_ok
        if not graph_ok and "graph" not in mv.inputs["stopped"]:
            mv.reasons.append("auto: no decidable graph candidate with est. gain >= min_gain_pct: graph move not run")
        if not (mv.kernel and mv.graph):
            if not mv.kernel:
                mv.reasons.append("auto: kernel move out (guard)")
            return
        prev_graph_only = bool(hist) and hist[-1].get("move") == "graph"
        if f["first_round"] and (g_est >= GRAPH_FIRST_PCT or (small >= SMALL_SHARE_HI and g_est > 0)):
            mv.kernel, mv.deferred = False, True
            mv.reasons.append(f"auto: first round graph-only: est. graph gain {round(g_est, 2)}% (>= {GRAPH_FIRST_PCT}%) or "
                              f"{round(100 * small, 1)}% of the time in unfused small ops (>= {round(100 * SMALL_SHARE_HI)}%) with a candidate: "
                              "restructure before tuning kernels")
        elif lg and lg["kernel"] is not None and lg["graph"] is not None and not prev_graph_only \
                and lg["graph"] >= GRAPH_DOMINATES * max(lg["kernel"], 100 * self.min_gain):
            mv.kernel, mv.deferred = False, True
            mv.reasons.append(f"auto: last graph gain {lg['graph']}% >= {GRAPH_DOMINATES}x the kernel gain {lg['kernel']}%: graph-only "
                              "round (a restructured graph changes the kernels worth tuning)")
        else:
            why = [f"auto: kernel + graph (est. graph gain {round(g_est, 2)}%, {round(100 * small, 1)}% unfused small-op time"]
            if lg:
                why.append(f", last gains kernel {lg['kernel']}% / graph {lg['graph']}%")
            if f["rounds_left"] <= 1:
                why.append(", last round")
            mv.reasons.append("".join(why) + ")")

    # ---- the round's record
    def begin(self, mv: Move) -> dict:
        return mv.to_dict(self.policy)

    def end_round(self, rec_ctl: dict, rec: dict, doc: dict, *, plan_problems: int, spent_usd: float) -> dict:
        kinds = {h["id"]: h["kind"] for h in doc.get("hints", [])}
        ran_graph = rec_ctl["graph"]
        decided = len(rec.get("decisions") or [])
        region_dec = sum(1 for d in rec.get("decisions", []) if kinds.get(d.get("hint")) == "fuse_region")
        rec_ctl["gain"] = realized(rec, kinds)
        rec_ctl["worked"] = {"kernel": bool(rec_ctl["kernel"] and plan_problems),
                             "graph": bool(ran_graph and decided),
                             "region_loop": bool(rec_ctl["region_loop"] and (region_dec or rec_ctl["inputs"].get("live_region_ops")))}
        rec_ctl["spent_usd"] = round(spent_usd, 4)
        tot = sum((rec.get("partition_us") or {}).values()) or None
        rec_ctl["start_partition_us"] = tot
        self.log.append({"round": rec["round"], "move": rec_ctl["move"], "reasons": rec_ctl["reasons"], "gain": rec_ctl["gain"],
                         "worked": rec_ctl["worked"]})
        return rec_ctl

    def stop_record(self, r: int, mv: Move | None, reason: str) -> dict:
        self.stop_reason = reason
        d = mv.to_dict(self.policy) if mv is not None else {"policy": self.policy, "move": "stop", "kernel": False, "graph": False,
                                                            "region_loop": False, "stop": True, "reasons": [reason], "inputs": {}}
        self.log.append({"round": r, "move": "stop", "reasons": d["reasons"]})
        return d
