"""Cost-efficiency rules of the graph optimiser (E7 / X-101, ``l3_farm/analysis/efficiency_rules.md``).

Two rules, ON by default since X-129 (``--efficiency-rules on``):

* **R1 patience** (``stop_patience``, KernelBlaster ``run_multislot.py --stop-patience``): a plan slot problem
  stops after ``N`` consecutive *committed* steps without a new slot best (commit order across its parallel
  rollouts; in-flight steps finish; no new step starts). Applied to the plan's optimisation runs only, never to
  the fusion / split decision runs (their original and variant must get the same budget).
* **R2 late min-share** (``plan_min_share_late``): in rounds >= 1 only, plan items below this share of the
  round's profiled model time are not sent (round 0 keeps ``plan_min_share`` / ``min_share``: round-0 shares are
  measured on the naive seeds and mislead).

Presets (``efficiency_rules``): ``on`` = patience 8 + late min-share 0.005; ``safe`` = late min-share 0.01, no
patience; ``off`` = neither (exactly the behaviour before X-129). An explicit ``stop_patience`` /
``plan_min_share_late`` overrides the preset's value (0 = that rule off).

Resume: with no efficiency flag given, an existing work dir keeps the rules it was started with
(``state.json`` ``efficiency``); a work dir from before X-129 (no record, no round key entry) is a legacy run and
continues with the rules off, so its round key -- and its interrupted round -- stay valid. With the rules off the
round key has no ``efficiency`` entry at all (identical to the legacy key).
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

PRESETS = {
    "on": {"stop_patience": 8, "plan_min_share_late": 0.005},
    "safe": {"stop_patience": 0, "plan_min_share_late": 0.01},
    "off": {"stop_patience": 0, "plan_min_share_late": 0.0},
}
DEFAULT = "on"


def preset_values(mode: str | None) -> dict:
    mode = mode or DEFAULT
    if mode not in PRESETS:
        raise ValueError(f"efficiency_rules must be one of {sorted(PRESETS)}, not {mode!r}")
    return dict(PRESETS[mode])


def explicit(cfg) -> bool:
    """Any efficiency flag given on the command line / in the Config."""
    return any(getattr(cfg, k, None) is not None for k in ("efficiency_rules", "stop_patience", "plan_min_share_late"))


def from_flags(mode: str | None, stop_patience: int | None, plan_min_share_late: float | None) -> dict:
    """The rule parameters of a preset with the explicit per-rule values on top."""
    v = preset_values(mode)
    if stop_patience is not None:
        v["stop_patience"] = max(0, int(stop_patience))
    if plan_min_share_late is not None:
        v["plan_min_share_late"] = max(0.0, float(plan_min_share_late))
    return {"rules": mode or DEFAULT, **v}


def _recorded(work: Path, state: dict) -> tuple[dict | None, bool]:
    """(the rules recorded for the work dir, whether it holds an earlier run at all)."""
    rec = state.get("efficiency")
    if isinstance(rec, dict) and "stop_patience" in rec:
        return rec, True
    existing = bool(state.get("rounds")) or bool(state.get("sent"))
    for p in sorted(Path(work).glob("r[0-9][0-9]/progress.json")):
        existing = True
        try:
            key = json.loads(p.read_text()).get("key") or {}
        except (OSError, ValueError):
            continue
        if isinstance(key.get("efficiency"), dict):
            return dict(key["efficiency"]), True
    return None, existing


def resolve(cfg, work: Path, state: dict) -> dict:
    """The rule parameters this run uses, and why (``source``):

    * ``explicit``: a flag was given (a change against the work dir's record is a new configuration, as for any
      other round-key field);
    * ``default``: a fresh work dir, the default preset;
    * ``resumed``: the work dir's recorded rules (no flag given);
    * ``legacy``: the work dir predates the rules (no record): rules off, round key unchanged."""
    rec, existing = _recorded(work, state)
    if explicit(cfg):
        eff = from_flags(cfg.efficiency_rules, cfg.stop_patience, cfg.plan_min_share_late)
        eff["source"] = "explicit"
        if rec is None and existing and active(eff):
            eff["note"] = "legacy work dir (no efficiency record) continued with explicit rules: a new configuration"
        elif rec is not None and params(rec) != params(eff):
            eff["note"] = f"recorded rules {params(rec)} replaced by explicit flags: a new configuration"
        return eff
    if rec is not None:
        return {"rules": rec.get("rules", "custom"), "stop_patience": int(rec.get("stop_patience") or 0),
                "plan_min_share_late": float(rec.get("plan_min_share_late") or 0.0), "source": "resumed"}
    if existing:
        return {**from_flags("off", None, None), "rules": "off", "source": "legacy"}
    return {**from_flags(None, None, None), "source": "default"}


def params(eff: dict) -> dict:
    return {"stop_patience": int(eff.get("stop_patience") or 0),
            "plan_min_share_late": float(eff.get("plan_min_share_late") or 0.0)}


def active(eff: dict) -> bool:
    p = params(eff)
    return bool(p["stop_patience"] or p["plan_min_share_late"])


def apply(cfg, eff: dict):
    """``cfg`` with the resolved concrete values (round key, KernelBlaster command, plan)."""
    return replace(cfg, efficiency_rules=eff["rules"], stop_patience=params(eff)["stop_patience"],
                   plan_min_share_late=params(eff)["plan_min_share_late"])


def key_fields(cfg) -> dict:
    """Round-key entry: only when a rule is on (rules off = the legacy key, so pre-X-129 work dirs stay valid)."""
    sp = int(getattr(cfg, "stop_patience", None) or 0)
    late = float(getattr(cfg, "plan_min_share_late", None) or 0.0)
    if not sp and not late:
        return {}
    return {"efficiency": {"rules": getattr(cfg, "efficiency_rules", None), "stop_patience": sp,
                           "plan_min_share_late": late}}


def round_min_share(base: float, late: float | None, r: int) -> float:
    """The plan's min_share in round ``r``: ``base`` in round 0, max(base, late) from round 1 on."""
    return max(base or 0.0, late or 0.0) if r >= 1 else (base or 0.0)


def late_skipped(plan: dict, base: float) -> list[dict]:
    """Plan items the late min-share rule (not the base min_share) kept out of the round."""
    return [{"key": it.get("key"), "op": it.get("op"), "share": round(float(it.get("share") or 0.0), 6),
             "total_us": round(float(it.get("total_us") or 0.0), 3)}
            for it in plan.get("items", [])
            if it.get("why_not") == "below-min-share" and float(it.get("share") or 0.0) >= (base or 0.0)]


def kb_patience(results: list[dict]) -> dict:
    """Patience telemetry of one plan run's KernelBlaster results (``result.json`` ``efficiency``)."""
    stopped, committed, planned = [], 0, 0
    for r in results:
        e = r.get("efficiency") or {}
        committed += int(e.get("steps_committed") or 0)
        planned += int(e.get("steps_planned") or 0)
        if e.get("stopped") == "patience":
            stopped.append({"problem": r.get("problem"), "at_commit": e.get("trigger_commit"),
                            "steps_committed": e.get("steps_committed"), "steps_planned": e.get("steps_planned"),
                            "steps_after_trigger": e.get("steps_after_trigger")})
    return {"patience_stopped": stopped, "n_patience_stopped": len(stopped), "steps_committed": committed,
            "steps_planned": planned, "steps_not_run": max(planned - committed, 0)}


def kb_supports_patience(kb: Path) -> bool:
    """The KernelBlaster checkout has ``run_multislot.py --stop-patience`` (X-129)."""
    try:
        return "--stop-patience" in (Path(kb) / "scripts" / "run_multislot.py").read_text()
    except OSError:
        return False


def summarize(eff: dict, rounds: list[dict]) -> dict:
    """summary.json ``efficiency``: the rules and what they did per round (for E7-style attribution)."""
    per = []
    for rec in rounds:
        e = rec.get("efficiency")
        if not e:
            continue
        per.append({"round": rec.get("round"), "min_share": e.get("min_share"),
                    "late_skipped": len(e.get("late_skipped") or []),
                    "late_skipped_share": round(sum(x.get("share") or 0.0 for x in e.get("late_skipped") or []), 6),
                    "patience_stopped": e.get("n_patience_stopped", 0),
                    "steps_committed": e.get("steps_committed"), "steps_planned": e.get("steps_planned")})
    tot = {k: sum(int(x.get(k) or 0) for x in per) for k in ("late_skipped", "patience_stopped", "steps_committed",
                                                             "steps_planned")}
    tot["steps_not_run"] = max(tot["steps_planned"] - tot["steps_committed"], 0)
    return {**eff, "rounds": per, "totals": tot}


def describe(eff: dict) -> str:
    p = params(eff)
    return (f"efficiency rules: {eff.get('rules')} ({eff.get('source')}): "
            f"patience {p['stop_patience'] or 'off'}, late min-share (rounds>=1) "
            f"{p['plan_min_share_late'] or 'off'}" + (f"; {eff['note']}" if eff.get("note") else ""))
