"""The flow side of the SOL gate: seed probe, unreachable policy, accept / final guard with rollback (docs/sol_gate.md).

``--sol-gate off|final|accept`` (``$MB_SOL_GATE``; default off: nothing here runs, no output changes):

  final   the whole model of every IR with a SOL tolerance is checked against SOL's rule at the end -- ``verify_ir`` of
          the final state; when it fails, the cached winners the failure points at are rejected for that IR (the
          cache's per-IR ``rejected`` list: lookups then fall to the next best winner, or the seed kernel) and the model
          is verified again, until it passes. Nothing else changes during the rounds, so a winner the SOL check
          refuses costs its LLM effort, nothing more.
  accept  ``final`` plus: every export carries the SOL tables (the driver of a partition that produces a graph
          output gates it, so KernelBlaster's own candidate acceptance applies SOL's rule there), fp32 slots get a bound
          derived from SOL's tolerance (``sol_tolerance.slot_cap``: single-pass TF32 fails it), and each round's winners are
          checked in the whole model right after they are ingested -- before the hint decisions and the next round
          build on them -- with the same rollback.

Before anything runs the gate measures the *seed* (the hand-kernel original) once per IR (``probe``): the result decides what
the gate compares against (:func:`decide_policy`): SOL's own reference outputs when the seed passes them, else the MB golden
at SOL's tolerance (``sol_unreachable``: SOL's reference itself is off -- 92015), else no regression against the seed's
matched ratio; the result is then marked ``unverified_vs_sol``. The policy is what the driver is exported with.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from . import sol_tolerance as st

FORMAT = "mb-sol-gate/1"
NO_REGRESSION_SLACK = 0.999       # the seed's matched ratio x this is the floor of the no-regression policy


def blame(rel_err: dict, *, ratio: float = 0.5, max_n: int = 4, exclude=()) -> list:
    """Which cached slots a failed SOL check points at: those whose isolated-op error is within ``ratio`` of the largest
    among the slots not yet rolled back (at most ``max_n``). The isolated errors of a single-pass TF32 slot and an fp32 slot
    differ by orders of magnitude, so the class that loses precision separates; the next failure (if any) takes the next class.
    ``rel_err`` {slot key: error or None}."""
    cand = {k: (v if v is not None else 0.0) for k, v in rel_err.items() if k not in set(exclude)}
    if not cand:
        return []
    top = max(cand.values())
    ranked = sorted(cand, key=lambda k: -cand[k])
    sel = [k for k in ranked if cand[k] >= ratio * top] if top > 0 else ranked[:1]
    return sel[:max_n]


def blame_slots(res: dict, plan: dict) -> dict:
    """``kb_plan.failed_cached_slots``'s SOL case: a run that failed only SOL's check names no slot; the cache entries
    (``plan["used"]``) to reject are the blamed class (:func:`blame`) among the used slots."""
    v = st.verdict(res.get("sol"))
    if not v or v.get("ok") is not False:
        return {}
    used = plan.get("used") or {}
    sl = res.get("slots") or {}
    err = {k: (sl.get(k.split("/")[0]) or {}).get("rel_err") for k in used}
    return {k: used[k] for k in blame(err)}


def decide_policy(sol: dict, outputs: dict, base_ratio: dict, golden_margin: float) -> dict:
    """The comparison the gate uses for each output, from the seed's MB_SOL lines (``sol`` = parse_sol_lines, keyed by buffer
    name; ``outputs`` = the manifest's sol outputs {buffer: {index, ...}}; ``base_ratio`` {index: SOL's required ratio}).

    Ladder (first that the seed passes):
      1. ``sol``      SOL's own reference outputs (margin 1.0): a kernel that passes the gate passes SOL on this machine's torch;
      2. ``golden``   the MB golden at SOL's tolerance x ``golden_margin`` (no / unreachable SOL reference; the margin leaves room
                      for the reference's own error: a golden is a fp32 walk, SOL's reference runs cuDNN TF32 convs);
      3. ``golden``   at SOL's own tolerance (margin 1.0);
      4. ``golden``   margin 1.0 with the required ratio lowered to the seed's matched ratio x 0.999 (the no-regression policy:
                      the seed itself misses SOL's tolerance against even the golden).
    ``sol_unreachable`` is set when a SOL reference exists and the seed fails it (the result is then ``unverified_vs_sol``)."""
    out, status, unreachable = {}, "reachable", {}
    for name, o in outputs.items():
        i = str(o["index"])
        d = sol.get(name) or {}
        s, g, g1 = d.get("sol"), d.get("golden"), d.get("golden1") or d.get("golden")
        req = base_ratio.get(o["index"], st.DEFAULT_RATIO)
        if s is not None and s["ok"]:
            out[i] = {"ref": "sol", "margin": 1.0, "required": req, "step": 1}
            continue
        if s is not None:
            unreachable[name] = {"matched_ratio": s["matched_ratio"], "required_ratio": s["required_ratio"],
                                 "max_abs_err": s["max_abs_err"], "atol": s["atol"], "rtol": s["rtol"],
                                 "nonfinite_output": s["nonfinite_output"]}
        if g is not None and g["ok"]:
            out[i] = {"ref": "golden", "margin": golden_margin, "required": req, "step": 2}
        elif g1 is not None and g1["ok"]:
            out[i] = {"ref": "golden", "margin": 1.0, "required": req, "step": 3}
        elif g1 is not None:
            out[i] = {"ref": "golden", "margin": 1.0, "required": round(min(req, g1["matched_ratio"] * NO_REGRESSION_SLACK), 6),
                      "step": 4, "seed_matched_ratio": g1["matched_ratio"]}
        else:                                                       # nothing to compare the seed with
            out[i] = {"ref": "golden", "margin": golden_margin, "required": req, "step": 0}
    steps = {o["step"] for o in out.values()}
    if 4 in steps:
        status = "no_regression"
    elif 3 in steps:
        status = "golden_full_tolerance"
    elif 2 in steps:
        status = "golden_margin"
    elif 0 in steps:
        status = "unchecked"
    return {"status": status, "outputs": out, **({"sol_unreachable": unreachable} if unreachable else {})}


class SolGate:
    """The flow's SOL gate (inert unless ``--sol-gate final|accept``)."""

    def __init__(self, cfg, work: Path):
        self.cfg, self.work = cfg, Path(work)
        for k, env, attr in (("sol_gate", "MB_SOL_GATE", "mode_cfg"), ("sol_slot_frac", "MB_SOL_SLOT_FRAC", "slot_frac"),
                             ("sol_golden_margin", "MB_SOL_GOLDEN_MARGIN", "golden_margin"), ("sol_bound", "MB_SOL_BOUND", "bound")):
            v = getattr(cfg, k, None)
            if v not in (None, ""):
                setattr(st.STATE, attr, float(v) if attr in ("slot_frac", "golden_margin") else str(v))
        self.mode = st.STATE.mode
        self.max_rollbacks = int(getattr(cfg, "sol_max_rollbacks", 8) or 8)
        self.bench = str(getattr(cfg, "sol_bench", "") or "")
        self.path = self.work / "sol_gate.json"
        self.irs: dict[str, dict] = {}          # IR dir name -> record
        self.last_device = None                 # the GPU of the last seed probe (the flow's hardware check)
        self._load()

    # ------------------------------------------------------------ plumbing
    def active(self) -> bool:
        return self.mode != "off"

    def accepting(self) -> bool:
        return self.mode == "accept"

    def round_key(self) -> dict:
        if not self.active():
            return {}
        return {"sol_gate": self.mode, **({"sol_slot_frac": st.STATE.frac} if self.accepting() else {}),
                **({"sol_bound": "replace"} if st.STATE.bound_mode == "replace" else {}),
                "sol_golden_margin": st.STATE.margin}

    def apply_env(self) -> None:
        """The settings as env, for every subprocess (exports run in this process; KernelBlaster only reads MB_SOL lines)."""
        if not self.active():
            return
        os.environ["MB_SOL_GATE"] = self.mode
        os.environ["MB_SOL_SLOT_FRAC"] = repr(st.STATE.frac)
        os.environ["MB_SOL_GOLDEN_MARGIN"] = repr(st.STATE.margin)
        os.environ["MB_SOL_BOUND"] = st.STATE.bound_mode
        if self.bench:
            os.environ["MB_SOL_BENCH"] = self.bench

    def _load(self) -> None:
        try:
            d = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return
        self.irs = d.get("irs") or {}
        for rec in self.irs.values():
            if rec.get("policy") and rec.get("ir_name"):
                st.STATE.policy[rec["ir_name"]] = rec["policy"]
            if rec.get("reference") and rec.get("ir_name") and os.path.exists(rec["reference"].get("path", "")):
                st.STATE.refs[rec["ir_name"]] = rec["reference"]["path"]

    def _save(self) -> None:
        if not self.active():
            return
        self.work.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"format": FORMAT, "mode": self.mode, "irs": self.irs}, indent=1, default=str))
        os.replace(tmp, self.path)

    # ------------------------------------------------------------ start: tolerance, reference, seed probe
    def prepare(self, irs: list[Path]) -> None:
        """Per IR: its SOL tolerance (none: the IR is not gated), SOL's reference outputs (written beside the IR when the
        bench is known and a torch is here; else the gate compares with the MB golden)."""
        for d in irs:
            d = Path(d)
            g = json.loads((d / "graph.json").read_text())
            rec = self.irs.setdefault(d.name, {})
            rec.update(ir_name=g["name"], ir_dir=str(d))
            if rec.get("prepared"):                       # once per IR (a later round / a resumed run)
                if rec.get("reference"):
                    st.STATE.refs[g["name"]] = rec["reference"]["path"]
                continue
            tols = st.tolerances_for(g)
            if tols is None:
                rec["status"] = "no_sol_tolerance"
                continue
            rec["tolerance"] = tols
            ref = st.load_reference(d, g["name"])
            if ref is None and os.environ.get("MB_SOL_REF", "auto").strip().lower() not in ("0", "off", "no", "false"):
                if st.ensure_reference(d, self.bench or None) is not None:
                    ref = st.load_reference(d, g["name"])
            if ref is not None:
                st.STATE.refs[g["name"]] = ref["path"]
                rec["reference"] = {"path": ref["path"], "device": ref["device"], "dtypes": ref["dtypes"]}
                if ref["device"] != "cuda":
                    rec.setdefault("warnings", []).append(
                        "SOL's reference was computed on the CPU (no GPU here): no cuDNN TF32 noise in it, the gate treats it "
                        "like the golden")
            else:
                rec["reference"] = None
                rec.setdefault("warnings", []).append(
                    "no SOL reference outputs for this IR (no sol_ref.npz, no bench: set MB_SOL_REF=1 at extraction or "
                    "--sol-bench): the gate compares with the MB golden at SOL's tolerance x the golden margin")
            rec.setdefault("rollbacks", [])
            rec["prepared"] = True
        self._save()

    def start(self, irs: list[Path], *, specs: dict, arch: str, kbeval=None, gpu_model=None, allow_mismatch=False, hw_check=None) -> None:
        """prepare + probe every IR (each once; later calls are no-ops). ``hw_check(step, device, what)``: the flow's hardware record."""
        self.prepare(irs)
        for d in irs:
            self.probe(d, specs=specs, arch=arch, kbeval=kbeval, gpu_model=gpu_model, allow_mismatch=allow_mismatch,
                       work=self.work / "sol_gate" / "probe" / Path(d).name)
            if self.last_device and hw_check is not None:
                hw_check("sol_probe", self.last_device, Path(d).name)
            self.last_device = None

    def probe(self, d: Path, *, specs: dict, arch: str, kbeval=None, gpu_model=None, allow_mismatch=False, work: Path) -> dict | None:
        """Measure the seed (hand kernels, no cache, no hints) once: its MB_SOL verdicts decide the policy. Skipped when a
        policy is recorded (a resumed run) or the IR has no SOL tolerance."""
        from .kb_eval import builtin_device, run_problem
        from .kb_export import build_and_run, export_problem
        d = Path(d)
        rec = self.irs.get(d.name)
        if rec is None or rec.get("status") == "no_sol_tolerance":
            return None
        if rec.get("policy"):
            return rec["policy"]
        work = Path(work)
        t0 = time.time()
        sm = str(arch).removeprefix("sm_")
        shutil.rmtree(work, ignore_errors=True)
        with st.scope("report"):
            p = export_problem(d, work / "problem", strategy="whole")
        man = json.loads((p / "manifest.json").read_text())
        if "sol" not in man:
            rec["policy"] = {"status": "no_sol_block", "outputs": {}}
            rec["seed"] = {"note": "the seed export carries no SOL block"}
            st.STATE.policy[rec["ir_name"]] = rec["policy"]
            self._save()
            return rec["policy"]
        env = {"MB_SCREEN": "1"}
        if kbeval is not None:
            res = run_problem(kbeval, p, arch=sm, work=work / "kb_time", reps=1, gpu_model=gpu_model,
                              allow_mismatch=allow_mismatch, name="sol_seed", driver_env=env)
        else:
            res = build_and_run(p, arch=sm, runs=1, graph=False, env=env)
            res["device"] = builtin_device(sm)
        self.last_device = res.get("device")
        sol = res.get("sol") or {}
        base = {o["index"]: o["tolerance"]["required_matched_ratio"] for o in man["sol"]["outputs"].values()}
        pol = decide_policy(sol, man["sol"]["outputs"], base, st.STATE.margin)
        rec["seed"] = {"sol": sol, "verdict": st.verdict(sol), "driver_passed": res.get("passed"),
                       "seconds": round(time.time() - t0, 1), "device": (res.get("device") or {}).get("name")
                       if isinstance(res.get("device"), dict) else res.get("device")}
        rec["policy"] = pol
        st.STATE.policy[rec["ir_name"]] = pol
        if pol.get("sol_unreachable"):
            rec["sol_unreachable"] = pol["sol_unreachable"]
            print(json.dumps({"sol_gate": "SOL_UNREACHABLE", "ir": d.name, "policy": pol["status"],
                              "seed_vs_sol_reference": pol["sol_unreachable"],
                              "note": "the seed (hand kernels) already fails SOL's tolerance against SOL's own reference; the "
                                      "gate compares with the golden instead; the result is unverified vs SOL"}), flush=True)
        elif pol["status"] != "reachable":
            print(json.dumps({"sol_gate": pol["status"], "ir": d.name,
                              "note": "the gate compares with the MB golden at SOL's tolerance (no SOL reference outputs, or the "
                                      "seed does not pass them); the result is marked unverified vs SOL"}), flush=True)
        self._save()
        shutil.rmtree(work / "problem", ignore_errors=True)
        return pol

    # ------------------------------------------------------------ guard: verify with rollback
    def _reject(self, cache, ir_name: str, impl_ids: set) -> int:
        n = 0
        for e in cache.entries:
            if e.get("impl_id") in impl_ids and ir_name not in e["rejected"]:
                e["rejected"].append(ir_name)
                n += 1
        cache.save()
        return n

    def verify(self, d: Path, verify_fn, cache, *, where: str = "final") -> dict:
        """``verify_fn()`` (a ``kb_ingest.verify_ir`` of this IR) under the SOL gate; a SOL failure rejects the cached
        winners it points at and verifies again; at most ``--sol-max-rollbacks`` attempts, then everything used is reverted
        to the next-best / seed kernels at once. -> verify_ir's dict plus ``sol_gate`` (the attempts)."""
        d = Path(d)
        rec = self.irs.get(d.name)
        if not self.active() or rec is None or rec.get("status") == "no_sol_tolerance" or not rec.get("policy"):
            return verify_fn()
        ir_name, tried, steps = rec["ir_name"], [], []
        v: dict = {}
        revert_all = False
        for attempt in range(self.max_rollbacks + 2):
            with st.scope("enforce"):
                v = verify_fn()
            sv = st.verdict(v.get("sol"))
            steps.append({"attempt": attempt, "passed": v.get("passed"), "ok": None if sv is None else sv["ok"],
                          "status": v.get("status")})
            if v.get("passed") or v.get("status") == "no-cache-entries":
                break
            if sv is None or sv["ok"] is not False:
                break                                          # not a SOL failure: the flow's own fallback handles it
            slots = v.get("slots") or {}
            if revert_all:
                break
            if attempt >= self.max_rollbacks:
                pick = [k for k in slots if k not in tried]
                revert_all = True
            else:
                pick = blame({k: s.get("rel_err") for k, s in slots.items()}, exclude=tried)
            if not pick:
                break
            ids = {slots[k]["impl_id"] for k in pick}
            n = self._reject(cache, ir_name, ids)
            tried += pick
            steps[-1].update(rolled_back=pick, impl_ids=sorted(ids), rejected_entries=n,
                             rel_err={k: slots[k].get("rel_err") for k in pick},
                             verdict=sv["outputs"])
            rec["rollbacks"].append({"where": where, "attempt": attempt, "slots": pick, "impl_ids": sorted(ids),
                                     "rel_err": steps[-1]["rel_err"], "matched_ratio": {
                                         k: o.get("matched_ratio") for k, o in sv["outputs"].items()}})
            print(json.dumps({"sol_gate": "rollback", "ir": d.name, "where": where, "slots": pick, "impl_ids": sorted(ids),
                              "matched_ratio": {k: o.get("matched_ratio") for k, o in sv["outputs"].items()}}), flush=True)
        sv = st.verdict(v.get("sol"))
        rec["final"] = {"where": where, "passed": v.get("passed"), "verdict": sv, "attempts": len(steps)}
        v["sol_gate"] = {"where": where, "attempts": steps, "verdict": sv}
        self._save()
        return v

    def after_ingest(self, r: int, irs: list[Path], verify_for, cache) -> list[dict]:
        """accept: the round's ingested winners checked in the whole model (``verify_for(d, work)`` -> a verify_ir call of
        IR ``d`` into ``work``), with rollback, before the hint decisions and the next round build on them."""
        out = []
        if not self.accepting():
            return out
        for d in irs:
            d = Path(d)
            if (self.irs.get(d.name) or {}).get("status") == "no_sol_tolerance":
                continue
            w = self.work / f"r{r:02d}" / "sol_gate" / d.name
            shutil.rmtree(w, ignore_errors=True)
            t0 = time.time()
            v = self.verify(d, lambda d=d, w=w: (shutil.rmtree(w, ignore_errors=True), verify_for(d, w))[1], cache,
                            where=f"round {r}")
            sg = v.get("sol_gate") or {}
            out.append({"ir": d.name, "attempts": len(sg.get("attempts") or []), "passed": v.get("passed"),
                        "rolled_back": [s for a in sg.get("attempts") or [] for s in a.get("rolled_back", [])],
                        "seconds": round(time.time() - t0, 1)})
            shutil.rmtree(w, ignore_errors=True)
        return out

    # ------------------------------------------------------------ reporting
    def final_scope(self):
        """The block the final model's run is exported in: SOL's check enforced (a no-op context when the gate is off)."""
        import contextlib
        return st.scope("enforce") if self.active() else contextlib.nullcontext()

    def model_record(self, d_name: str, model: dict | None) -> dict | None:
        """The SOL verdict of the final model (kb_model_run's driver run) for model_runtime.json / summary.json."""
        rec = self.irs.get(d_name)
        if not self.active() or rec is None:
            return None
        if rec.get("status") == "no_sol_tolerance":
            return {"mode": self.mode, "status": "no_sol_tolerance"}
        sol = (model or {}).get("sol")
        sv = st.verdict(sol)
        pol = rec.get("policy") or {}
        gate_refs = {o.get("ref") for o in (sv or {}).get("outputs", {}).values()}
        unverified = bool(pol.get("sol_unreachable")) or gate_refs != {"sol"} or pol.get("status") not in (None, "reachable")
        out = {"mode": self.mode, "status": pol.get("status"), "policy": pol.get("outputs"),
               "pass": None if sv is None else sv["ok"], "unverified_vs_sol": unverified,
               "outputs": (sv or {}).get("outputs"), "tolerance": rec.get("tolerance"),
               "reference": rec.get("reference"), "rollbacks": rec.get("rollbacks") or [],
               "seed": {"verdict": (rec.get("seed") or {}).get("verdict")}}
        if pol.get("sol_unreachable"):
            out["sol_unreachable"] = pol["sol_unreachable"]
        if rec.get("warnings"):
            out["warnings"] = rec["warnings"]
        rec["model"] = {k: out[k] for k in ("pass", "unverified_vs_sol", "outputs")}
        self._save()
        return out

    def summary(self) -> dict:
        return {"mode": self.mode, "slot_frac": st.STATE.frac if self.accepting() else None,
                "golden_margin": st.STATE.margin, "bound": st.STATE.bound_mode, "max_rollbacks": self.max_rollbacks,
                "irs": {n: {k: r.get(k) for k in ("status", "policy", "sol_unreachable", "rollbacks", "final", "warnings")
                            if r.get(k) is not None} for n, r in self.irs.items()}}
