"""Graph-variant policy: headroom, budgets with successive halving, the deferred-decision expectation test,
survival weights, structural-outcome knowledge.

Adapted from the variant-discovery prototype (``l3_farm/analysis/variant_discovery.patch``; its composite seeds,
attention / layout matchers are live as ``composite_seed`` / ``attention_fuse`` / ``layout_fold``) for the graph
search redesign (``graph_search``; l3_farm/analysis/graph_search_redesign.md). Pure functions, no I/O except
:class:`StructuralKnowledge`; nothing here runs unless ``kb_graph_opt`` is started with a ``--graph-*`` flag.

1. **Headroom** (:func:`headroom`, :func:`op_headroom`): an op's / sub-graph's measured time vs an optimistic
   roofline bound (io bytes / DRAM bandwidth vs flops / tensor-core peak at an achievable fraction ``eff``).
2. **Budget** (:func:`base_steps`, :func:`rung_steps`, :func:`allocate`): steps per hint ~ (1 + log2 headroom),
   blended with the kind's prior, successive halving over rungs (rung k gets ``eta^k`` x the base, survivors only).
3. **Checkpoint decision** (:func:`checkpoint`): after a rung's paired re-timing: ``accepted`` when the variant is
   ``accept_gain`` faster; else ``promising`` (continued at the next rung) only when the auditable expectation test
   (:func:`expectation`: bound, cap, trajectory, projection, prior) passes; else ``rejected`` with the failing test.
4. **Survival** (:func:`survival`, :func:`survival_steps`): P(an op survives into the final graph) from the pending
   hints that would absorb it and their priors; plan slot problems get steps in proportion.
5. **Knowledge** (:class:`StructuralKnowledge`): structural outcomes as jsonl (+ KernelBlaster v2 stats rows), read
   back as priors (kept from the prototype for ``fuse_attention`` / ``fold_layout``).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

STRUCTURAL_KINDS = ("fuse_attention", "fold_layout")
# (fp16 tensor-core GFLOP/s, DRAM GB/s), as KernelBlaster's opt_multislot roofline table
PEAKS = {"75": (65000.0, 672.0), "80": (312000.0, 1555.0), "86": (71000.0, 936.0), "89": (91600.0, 864.0),
         "90": (989000.0, 3350.0), "100": (2250000.0, 8000.0)}
SEED_PRIOR = {"fuse_attention": {"p_win": 0.5, "n": 1.0, "steps_to_win": 6.0},
              "fold_layout": {"p_win": 0.5, "n": 1.0, "steps_to_win": 3.0}}
_ESIZE = {"f16": 2, "fp16": 2, "float16": 2, "bf16": 2, "f32": 4, "fp32": 4, "float32": 4, "i8": 1, "s8": 1,
          "int8": 1, "u8": 1, "i32": 4, "int32": 4, "i64": 8, "int64": 8}


# ---------------------------------------------------------------- headroom

def headroom(incumbent_us: float, *, io_bytes: float, flops: float, arch: str = "89", eff: float = 0.6,
             floor_us: float = 0.0) -> dict:
    gf, bw = PEAKS.get(str(arch).removeprefix("sm_"), PEAKS["89"])
    bound = max(io_bytes / (bw * 1e3), flops / (gf * 1e3))           # us (GB/s -> bytes/us = 1e3 * GB/s)
    opt = max(bound / max(eff, 1e-3), floor_us)
    return {"incumbent_us": round(incumbent_us, 3), "bound_us": round(bound, 3), "optimistic_us": round(opt, 3),
            "io_bytes": int(io_bytes), "flops": float(flops), "eff": eff, "floor_us": round(floor_us, 3),
            "ratio": round(incumbent_us / opt, 3) if opt > 0 else None,
            "est_gain_us": round(max(0.0, incumbent_us - opt), 3)}


def tensor_bytes(graph: dict, t: str) -> int:
    info = graph.get("tensors", {}).get(t, {})
    n = 1
    for d in info.get("shape", []) or []:
        n *= max(int(d), 1)
    return n * _ESIZE.get(str(info.get("dtype", "f16")), 2)


def op_flops(op: dict) -> float:
    """A rough flop count from the op's shape dict (conv / gemm kinds); 0 otherwise (memory bound)."""
    s = op.get("shape") or {}
    if isinstance(s.get("a"), dict):                     # fused op: its head stage's work
        s = s["a"]
    try:
        if {"N", "OC", "OH", "OW", "KH", "KW"} <= set(s):
            ic = int(s.get("IC", 1)) // max(int(s.get("G", 1)), 1)
            return 2.0 * int(s["N"]) * int(s["OC"]) * int(s["OH"]) * int(s["OW"]) * ic * int(s["KH"]) * int(s["KW"])
        if {"M", "N", "K"} <= set(s):
            return 2.0 * int(s.get("batch", s.get("B", 1)) or 1) * int(s["M"]) * int(s["N"]) * int(s["K"])
    except (TypeError, ValueError):
        return 0.0
    return 0.0


def op_headroom(graph: dict, op: dict, us: float, arch: str = "89", eff: float = 0.6) -> dict:
    """Roofline headroom of one op instance (inputs + outputs once each, its flops)."""
    io = sum(tensor_bytes(graph, t) for t in list(op.get("inputs", [])) + list(op.get("outputs", [])))
    return headroom(us or 0.0, io_bytes=io, flops=op_flops(op), arch=arch, eff=eff)


def attention_headroom(rep_fused: list[dict], op_us: dict, arch: str = "89", esize: int = 2) -> dict:
    """Sum over the fused chains of attention_fuse.fuse's report."""
    inc = io = fl = 0.0
    for f in rep_fused:
        s = f["shape"]
        B = int(s["B0"]) * int(s["B1"])
        M, N, D, DV = int(s["M"]), int(s["N"]), int(s["D"]), int(s["DV"])
        io += esize * B * (M * D + N * D + N * DV + M * DV)
        fl += 2.0 * B * M * N * (D + DV) * (0.5 if int(s.get("causal", 0)) else 1.0)
        inc += sum(op_us.get(n, 0.0) for n in f["removed"])
    return headroom(inc, io_bytes=io, flops=fl, arch=arch)


def fold_headroom(rep_folded: list[dict], op_us: dict, arch: str = "89", esize: int = 2) -> dict:
    """The fold keeps the GEMM: its incumbent time is the floor; the copies are what can disappear."""
    inc = io = fl = floor = 0.0
    for f in rep_folded:
        s = f["shape"]
        s = s if "B0" in s else s.get("a", s)          # a fused group's head stage
        bt = int(s["B0"]) * int(s["B1"]) if "B0" in s else int(s.get("batch", 1))
        M, K, N = int(s["M"]), int(s["K"]), int(s["N"])
        io += esize * bt * (M * K + K * N + M * N)
        fl += 2.0 * bt * M * K * N
        g = op_us.get(f["op"], 0.0)
        floor += g
        inc += g + sum(op_us.get(n, 0.0) for n in f["copies"])
    return headroom(inc, io_bytes=io, flops=fl, arch=arch, floor_us=floor)


def annotate(hint: dict, *, hr: dict, total_us: float, prior: dict | None = None, trig: list | None = None) -> dict:
    """The optimistic estimate on the hint's newest source: est_gain_* from the bound (not the seed),
    ``rank`` = est_gain_pct x the prior's acceptance rate."""
    src = hint["source"][-1]
    pct = 100.0 * hr["est_gain_us"] / total_us if total_us else 0.0
    p = (prior or {}).get("p_mean", (prior or {}).get("p_win", 0.5))
    src.update(est_gain_us=hr["est_gain_us"], est_gain_pct=round(pct, 2), headroom=hr, prior=prior,
               triggers=[t["id"] for t in trig or []], rank=round(pct * p, 3), estimate="optimistic-bound")
    return hint


# ---------------------------------------------------------------- budget (successive halving)

@dataclass
class BudgetCfg:
    base_steps: int = 2           # cfg.variant_steps
    min_steps: int = 1
    max_steps: int = 8            # per rung (rung 0)
    cap_steps: int = 16           # per hint, over all rungs
    round_budget: int = 0         # steps over all hints of a round (0 = no limit)
    eta: float = 2.0              # rung k gets eta^k x the base


def base_steps(hr: dict | None, prior: dict | None, bc: BudgetCfg) -> int:
    """Rung-0 steps: base x (1 + log2 headroom ratio), blended with the prior: a kind that wins reliably and fast
    needs fewer steps to show it; a kind that rarely wins gets fewer steps too (it must show a win early)."""
    ratio = max(1.0, float((hr or {}).get("ratio") or 1.0))
    s = bc.base_steps * (1.0 + math.log2(ratio))
    if prior and (prior.get("n_obs") or prior.get("n")):
        if prior.get("steps_to_win"):
            s = 0.5 * s + 0.5 * 1.5 * float(prior["steps_to_win"])
        p = float(prior.get("p_mean", prior.get("p_win", 0.5)))
        s *= min(1.0, 0.5 + p)
    return int(max(bc.min_steps, min(bc.max_steps, math.ceil(s))))


def rung_steps(rung: int, base: int, bc: BudgetCfg, used: int = 0) -> int:
    s = int(round(base * bc.eta ** max(0, int(rung))))
    return max(0, min(s, int(bc.max_steps * bc.eta ** max(0, int(rung))), bc.cap_steps - int(used)))


def allocate(cands: list[dict], bc: BudgetCfg) -> list[dict]:
    """cands: [{"id", "hr", "prior", "rung", "used", "rank"}] -> same dicts with ``steps`` (0 = waits for a later
    round), in rank order; the round budget (if any) goes to the highest-ranked first."""
    left = bc.round_budget if bc.round_budget > 0 else 10 ** 9
    out = []
    for c in sorted(cands, key=lambda c: -(c.get("rank") or 0.0)):
        s = rung_steps(int(c.get("rung") or 0), base_steps(c.get("hr"), c.get("prior"), bc), bc, c.get("used") or 0)
        s = s if s <= left else 0
        left -= s
        out.append({**c, "steps": max(0, s)})
    return out


# ---------------------------------------------------------------- the expectation test

def trajectory(result: dict) -> list[float]:
    """Best-so-far end-to-end (else slot) time of a KernelBlaster problem, in commit order, starting at the seed."""
    start = result.get("baseline_e2e_us") or result.get("baseline_partition_us")
    pts = [start] if start else []
    best = start
    for s in result.get("steps") or []:
        v = s.get("e2e_after")
        if v is None and s.get("status") in ("improved", "accepted", "ok"):
            v = s.get("slot_us_after") if not start else None
        if v and (best is None or v < best):
            best = v
        if best is not None:
            pts.append(best)
    return pts


def expectation(*, orig_us: float, var_us: float, accept_gain: float, hr: dict | None, history: list[dict],
                traj: list[float], steps_used: int, cap_steps: int, prior: dict | None,
                min_improve: float = 0.05, plateau: float = 0.02, prior_min_p: float = 0.15,
                prior_min_n: float = 3.0, near: float = 0.0) -> dict:
    """``promising`` or ``rejected`` for a variant that did not win yet -> {"decision", "why", "tests"}.
    ``history``: earlier rungs' [{"var_us", "orig_us", "steps"}]; ``traj``: this rung's best-so-far curve;
    ``near``: a variant already within ``near`` of the target passes the trajectory test (a fused seed starts at
    parity and may not move on its own steps, while the original did not move either)."""
    target = orig_us * (1.0 - accept_gain)
    tests = {"target_us": round(target, 3), "var_us": round(var_us, 3), "orig_us": round(orig_us, 3)}
    if hr and hr.get("incumbent_us"):
        outside = max(0.0, orig_us - hr["incumbent_us"])
        best_possible = outside + hr["optimistic_us"]
        tests["bound_best_us"] = round(best_possible, 3)
        if best_possible >= target:
            return {"decision": "rejected", "why": f"bound falsified: even at the optimistic bound the variant "
                    f"({best_possible:.1f} us) would not beat the target {target:.1f} us", "tests": tests}
    if steps_used >= cap_steps:
        return {"decision": "rejected", "why": f"budget cap reached ({steps_used}/{cap_steps} steps) above the "
                f"target", "tests": tests}
    t0 = traj[0] if traj else None
    t1 = traj[-1] if traj else None
    imp = (t0 - t1) / t0 if t0 and t1 else 0.0
    if history:
        prev = history[-1]["var_us"] / history[-1]["orig_us"] if history[-1].get("orig_us") else None
        now = var_us / orig_us
        imp = max(imp, (prev - now) / prev if prev else 0.0)
    tests["improvement"] = round(imp, 4)
    gap = var_us / target - 1.0
    tests["gap_to_target"] = round(gap, 4)
    need_imp = min_improve if not history else plateau
    if imp < need_imp and not (near and gap <= near and not history):
        return {"decision": "rejected", "why": f"plateau: improved {100 * imp:.1f}% over its steps "
                f"(< {100 * need_imp:.0f}%) while still {100 * (var_us / orig_us - 1):+.1f}% vs the original",
                "tests": tests}
    n = max(1, len(traj) - 1)
    rate = (t1 / t0) ** (1.0 / n) if t0 and t1 and t1 < t0 else 1.0
    if rate < 1.0 and var_us > target:
        need = math.log(target / var_us) / math.log(rate)
    elif var_us > target:
        need = 0.0 if gap <= near else float("inf")
    else:
        need = 0.0
    left = cap_steps - steps_used
    p = float((prior or {}).get("p_mean", (prior or {}).get("p_win", 0.5)))
    tol = 1.0 + p                                                 # a confident prior tolerates a slower fit
    tests.update(rate_per_step=round(rate, 4), projected_steps=round(need, 2) if need != float("inf") else None,
                 steps_left=left, tolerance=tol)
    if need > tol * left:
        return {"decision": "rejected", "why": f"projected {need:.1f} more steps to beat the target at the fitted "
                f"rate {rate:.3f}/step > {tol:.2f} x the {left} steps left", "tests": tests}
    n_obs = float((prior or {}).get("n_obs", (prior or {}).get("n", 0)) or 0)
    if prior and n_obs >= prior_min_n and p < prior_min_p:
        return {"decision": "rejected", "why": f"prior: {prior.get('source')} acceptance {p:.2f} < {prior_min_p} "
                f"over {n_obs:.0f} decisions", "tests": tests}
    return {"decision": "promising", "why": f"improving {100 * imp:.1f}%, projected to reach the target in "
            f"~{need:.1f} of {left} steps left; bound allows it; prior {p:.2f}", "tests": tests}


def checkpoint(*, orig_us: float | None, var_us: float | None, accept_gain: float, rung: int, steps_used: int,
               cap_steps: int, hr: dict | None = None, history: list | None = None, traj: list | None = None,
               prior: dict | None = None, defer: bool = True, min_improve: float = 0.05,
               near: float = 0.02) -> dict:
    """A hint's decision at a successive-halving checkpoint (its paired re-timing done)."""
    if not orig_us or not var_us:
        return {"decision": "rejected", "why": "no paired time", "tests": {}}
    gain = (orig_us - var_us) / orig_us
    if gain >= accept_gain:
        return {"decision": "accepted", "why": f"variant {100 * gain:.1f}% faster at rung {rung} "
                f"({steps_used} steps)", "tests": {"gain": round(gain, 4)}}
    if not defer:
        return {"decision": "rejected", "why": f"gain {100 * gain:.1f}% < {100 * accept_gain:.1f}%",
                "tests": {"gain": round(gain, 4)}}
    x = expectation(orig_us=orig_us, var_us=var_us, accept_gain=accept_gain, hr=hr, history=history or [],
                    traj=traj or [], steps_used=steps_used, cap_steps=cap_steps, prior=prior,
                    min_improve=min_improve, near=near)
    x["tests"]["gain"] = round(gain, 4)
    return x


# ---------------------------------------------------------------- survival weights

def survival(op_atoms: list[str], pending: list[dict], *, head_weight: float = 0.5) -> float:
    """P(the ops of a plan item survive into the final graph) = mean over its op atoms of
    prod over pending hints h that absorb the atom of (1 - p_accept(h) x a), a = 1 for an absorbed (non-head)
    stage, ``head_weight`` for a fusion's head stage (its kernel seeds the fused op, so its work is partly kept).
    ``pending``: [{"p": acceptance probability, "absorb": set of atoms, "heads": set of atoms}]."""
    if not op_atoms:
        return 1.0
    tot = 0.0
    for a in op_atoms:
        s = 1.0
        for h in pending:
            if a in h.get("heads", ()):
                s *= 1.0 - float(h["p"]) * head_weight
            elif a in h.get("absorb", ()):
                s *= 1.0 - float(h["p"])
        tot += s
    return tot / len(op_atoms)


def survival_steps(steps: int, p_survive: float, *, min_steps: int = 1, floor: float = 0.0) -> int:
    """Steps for a slot problem in proportion to its survival probability (at least ``min_steps``; ``floor``
    = survival treated as at least this)."""
    p = max(float(floor), min(1.0, float(p_survive)))
    return int(max(min_steps, min(steps, round(steps * p))))


# ---------------------------------------------------------------- knowledge (structural kinds)

class StructuralKnowledge:
    """Structural-change outcomes as knowledge (prototype, kept): a jsonl (``path``) and optionally a knowledge v2
    ``KnowledgeDB`` (anything with ``add_stats`` / ``all_stats`` / ``audit``). :meth:`prior` reads either."""

    ITEM = "mb_struct:{kind}"

    def __init__(self, path: Path | None = None, kb_db=None, target_key: str = "cuda:sm_89"):
        self.path = Path(path) if path else None
        self.db = kb_db
        self.target_key = target_key

    def _nodes(self, trig: list[str]):
        scopes = [("target", self.target_key), ("universal", "")]
        return [(lvl, sc, st) for lvl, sc in scopes for st in ["*", *trig]]

    def record(self, rec: dict) -> dict:
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "target_key": self.target_key, **rec}
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        if self.db is not None and rec.get("final"):
            item = self.ITEM.format(kind=rec["kind"])
            win = 1.0 if rec["outcome"] == "accepted" else 0.0
            metrics = {"win": win, "steps": float(rec.get("steps_used") or 0)}
            if win:
                metrics["steps_to_win"] = float(rec.get("steps_used") or 0)
            if rec.get("gain") is not None:
                metrics["gain"] = float(rec["gain"])
            try:
                for m, v in metrics.items():
                    self.db.add_stats([(item, lvl, sc, m, st) for lvl, sc, st in self._nodes(rec.get("triggers") or [])],
                                      v)
                self.db.audit("mb_structural_outcome", rec.get("hint") or item, rec)
            except Exception as e:  # noqa: BLE001
                print(f"[variant_policy] knowledge write failed: {type(e).__name__}: {e}")
        return rec

    def _rows(self) -> list[dict]:
        if self.path is None or not self.path.exists():
            return []
        return [json.loads(x) for x in self.path.read_text().splitlines() if x.strip()]

    def prior(self, kind: str, trigger: str | None = None) -> dict:
        seed = SEED_PRIOR.get(kind, {"p_win": 0.5, "n": 1.0, "steps_to_win": 4.0})
        rows = [r for r in self._rows() if r.get("final") and r.get("kind") == kind]
        if trigger:
            tr = [r for r in rows if trigger in (r.get("triggers") or [])]
            rows = tr or rows
        if not rows:
            return {**seed, "p_mean": seed["p_win"], "n_obs": 0, "source": "seed"}
        wins = [r for r in rows if r["outcome"] == "accepted"]
        k = seed["n"]
        p = (len(wins) + k * seed["p_win"]) / (len(rows) + k)
        return {"p_win": p, "p_mean": p, "n": len(rows) + k, "n_obs": len(rows),
                "steps_to_win": (sum(r.get("steps_used") or 0 for r in wins) / len(wins)) if wins
                else seed["steps_to_win"], "source": "jsonl"}
