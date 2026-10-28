"""ModelBlaster side of the region loop: ``kb_graph_opt --region-loop off|on`` (docs/region_loop.md).

The loop itself lives in KernelBlaster (``agents/region_loop.py``): a bounded, tool-using agent loop on the REGION slots of a slot problem
(show / profile / scratch / verify / time / commit / revert), driven through the flow's own LLM path (the relay) with a structured-text
protocol, replacing the generic per-slot rollout steps on those slots. This module is the flow's switch for it:

* ``--region-loop off|on`` (default off; ``$MB_REGION_LOOP`` when the flag is absent), ``--region-loop-budget "turns,tools,usd,wall_s"``
  (``$MB_REGION_LOOP_BUDGET``; the KernelBlaster default is 40,120,8,3600 per problem, shared by the problem's region slots by time
  share), ``--region-loop-gate on|off`` (default on when the loop is on);
* ``kb_args`` adds ``--region-loop on [--region-loop-budget ...]`` to every ``run_multislot.py`` command line (and so to its resume
  config: a run resumed with another setting is refused);
* ``round_key`` / ``summary`` carry the setting only when it is on (off: the round key, ``summary.json`` and every output are unchanged);
* the PRE-LOOP GATE (``gate``): before a hint variant containing region ops is exported for KernelBlaster, ``graph_verify`` checks the
  variant IR against the original on the CPU (structural: names, DAG, tensors, dtype contract, kind bindings; numerical: the torch walkers
  on random inputs, outputs and sampled intermediates). A variant that does not verify is a REJECTED hypothesis (``job["early"]``, never
  timed, no LLM spend): the loop starts only on graphs known to compute the same function.

Not here: the pilot's runtime checks (NaN-poisoned op-by-op run, replay with new inputs, in-place weight swap) are not in the flow driver;
see the RUNBOOK / docs ("Not done").
"""
from __future__ import annotations

import os
import time
from pathlib import Path

ENV = "MB_REGION_LOOP"
ENV_BUDGET = "MB_REGION_LOOP_BUDGET"
MODES = ("off", "on")
SOURCE = "region-loop"


def resolve_mode(cfg=None) -> str:
    """``--region-loop`` wins, else ``$MB_REGION_LOOP``, else off. Validated."""
    v = getattr(cfg, "region_loop", None)
    if v is None:
        v = os.environ.get(ENV, "off")
    v = str(v).strip().lower() or "off"
    if v in ("0", "false", "no"):
        v = "off"
    if v in ("1", "true", "yes"):
        v = "on"
    if v not in MODES:
        raise ValueError(f"--region-loop must be one of {MODES}, not {v!r}")
    return v


def _budget(cfg) -> str:
    b = str(getattr(cfg, "region_loop_budget", "") or os.environ.get(ENV_BUDGET, "") or "").strip()
    if b:                                                  # validate with the loop's own parser (without importing KernelBlaster)
        parts = [p.strip() for p in b.split(",")]
        if len(parts) > 4:
            raise ValueError(f"--region-loop-budget {b!r}: expected turns,tools,usd,wall_s")
        for p in parts:
            if p:
                try:
                    if float(p) <= 0:
                        raise ValueError
                except ValueError:
                    raise ValueError(f"--region-loop-budget {b!r}: {p!r} is not a positive number") from None
    return b


def kb_args(cfg) -> list:
    """The KernelBlaster ``run_multislot.py`` flags of the setting: ``[]`` when off (the command line is then what it was)."""
    if resolve_mode(cfg) != "on":
        return []
    b = _budget(cfg)
    return ["--region-loop", "on"] + (["--region-loop-budget", b] if b else [])


GATED: list = []                                       # one record per gated variant of this process's run (summary)


def gate_enabled(cfg) -> bool:
    gate = str(getattr(cfg, "region_loop_gate", "on") or "on").strip().lower()
    if gate not in ("on", "off"):
        raise ValueError(f"--region-loop-gate must be on|off, not {gate!r}")
    return resolve_mode(cfg) == "on" and gate == "on"


class RegionLoopHook:
    """The flag's plumbing; inert (no env, no key, no gate) when the mode is off."""

    def __init__(self, cfg, work: Path | None = None):
        self.cfg, self.work = cfg, work
        self.mode = resolve_mode(cfg)
        self.budget = _budget(cfg) if self.mode == "on" else ""
        self.gate_on = gate_enabled(cfg)
        GATED.clear()

    def active(self) -> bool:
        return self.mode == "on"

    def round_key(self) -> dict:
        if not self.active():
            return {}
        return {"region_loop": "on", **({"region_loop_budget": self.budget} if self.budget else {}),
                **({} if self.gate_on else {"region_loop_gate": "off"})}

    def summary(self) -> dict:
        return {"mode": self.mode, "budget": self.budget or "default", "gate": "on" if self.gate_on else "off", "gated": list(GATED)}


def gate(cfg, hint: dict, vgraph: dict, orig_dir, variant_dir, specs: dict) -> str | None:
    """None = go on; else the reason the variant is rejected (``kb_graph_opt.prepare_hint`` sets ``job["early"]``). Only variants that
    contain region ops are gated (the loop acts on those); the check is ``graph_verify.verify`` (CPU): structural + numerical
    equivalence of the variant IR against the original, one seed, one input distribution (cheap; the paired measurement is the real
    test of the hint). Off (the default): returns None at once."""
    if not gate_enabled(cfg) or not any(str(o.get("op", "")).startswith("region__") for o in vgraph.get("ops", [])):
        return None
    from . import graph_verify as gv
    t0 = time.time()
    try:
        v = gv.verify(orig_dir, variant_dir, specs=specs, seeds=(0,), variants=("uniform",), n_track=6)
    except Exception as e:  # noqa: BLE001 -- a gate that cannot run must not wave a variant through
        GATED.append({"hint": hint.get("id"), "ok": False, "error": f"{type(e).__name__}: {e}"[:300]})
        return f"region gate could not run: {type(e).__name__}: {e}"[:400]
    rec = {"hint": hint.get("id"), "ok": bool(v.ok), "seconds": round(time.time() - t0, 2), "checks": list(v.checks)[:12]}
    if not v.ok:
        rec["reasons"] = [f"[{r['code']}] {r['msg']}"[:200] for r in v.reasons][:4]
    GATED.append(rec)
    return None if v.ok else "region gate (graph_verify): " + "; ".join(rec["reasons"])[:500]
