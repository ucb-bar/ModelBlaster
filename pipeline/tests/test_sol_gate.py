"""The flow side of the SOL gate (sol_gate.py), CPU only with a fake evaluator:

* ``blame`` classes and ``decide_policy``'s ladder (reachable / golden margin / SOL-unreachable / no-regression);
* ``SolGate.verify``: rollback of the winners a SOL failure points at, next-best entries, revert-all fallback, a failure that is not
  SOL's passes through, the gate off calls ``verify_fn`` once and writes nothing;
* ``kb_plan.failed_cached_slots`` blames SOL failures in a profile;
* the whole round loop (kb_graph_opt._run with every GPU step faked): seed probe -> policy, accept-mode checks after each round,
  rollback recorded in sol_gate.json / summary.json / model_runtime.json, resume reuses the policy, gate off = no trace.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import modelblaster.pipeline.kb_model_run as mr  # noqa: E402
from modelblaster.pipeline import graph_hints as gh  # noqa: E402
from modelblaster.pipeline import kb_export  # noqa: E402
from modelblaster.pipeline import kb_graph_opt as go  # noqa: E402
from modelblaster.pipeline import kb_plan, sol_gate, sol_tolerance as st  # noqa: E402
from modelblaster.pipeline.kb_ingest import CudaCache  # noqa: E402
from modelblaster.pipeline.tests import rewrite_fixtures as F  # noqa: E402
from modelblaster.pipeline.tests.test_graph_rewrite import _profile  # noqa: E402
from modelblaster.pipeline.tests.test_kb_graph_opt_parallel import FakeKB  # noqa: E402

TITAN = {"host": None, "label": "0", "index": 0, "uuid": "GPU-titan", "name": "NVIDIA TITAN RTX", "arch": "sm_75"}
TOL = {"max_atol": 0.05, "max_rtol": 1e-3}


def rec(matched, ok=None, required=0.99, atol=0.05, rtol=1e-3, max_abs=0.1):
    return {"matched_ratio": matched, "required_ratio": required, "max_abs_err": max_abs, "max_rel_err": 1.0, "max_excess": 0.0,
            "nonfinite_output": False, "all_zero_vs_nonzero_ref": False, "max_error_cap_exceeded": False, "n": 100,
            "atol": atol, "rtol": rtol, "ok": matched >= required if ok is None else ok}


def sol_of(name="out", sol=None, golden=None, golden1=None, gate="sol"):
    d = {"gate": gate}
    for k, v in (("sol", sol), ("golden", golden), ("golden1", golden1)):
        if v is not None:
            d[k] = v
    return {name: d}


class PolicyTest(unittest.TestCase):
    OUTS = {"out": {"index": 0}}
    BASE = {0: 0.99}

    def test_blame_takes_the_class_that_loses_precision(self):
        err = {"conv": 5e-4, "mm0": 4e-4, "mm1": 6e-4, "ewise": 1e-6, "red": None}
        self.assertEqual(sorted(sol_gate.blame(err)), ["conv", "mm0", "mm1"])
        self.assertEqual(sol_gate.blame(err)[0], "mm1")                    # worst first
        self.assertEqual(sol_gate.blame(err, exclude=["mm1", "conv", "mm0"]), ["ewise"])
        self.assertEqual(sol_gate.blame({"a": 1e-3}, exclude=["a"]), [])
        self.assertEqual(sol_gate.blame({"a": 0.0, "b": 0.0}), ["a"])       # all zero: one at a time
        many = {f"s{i}": 1e-3 for i in range(9)}
        self.assertEqual(len(sol_gate.blame(many)), 4)                     # at most 4 per attempt

    def test_ladder_reachable(self):
        p = sol_gate.decide_policy(sol_of(sol=rec(1.0), golden=rec(1.0, atol=0.025), golden1=rec(1.0)), self.OUTS, self.BASE, 0.5)
        self.assertEqual(p["status"], "reachable")
        self.assertEqual(p["outputs"]["0"]["ref"], "sol")
        self.assertEqual(p["outputs"]["0"]["margin"], 1.0)
        self.assertNotIn("sol_unreachable", p)

    def test_ladder_no_sol_reference_uses_the_golden_margin(self):
        p = sol_gate.decide_policy(sol_of(golden=rec(1.0, atol=0.025), golden1=rec(1.0), gate="golden"), self.OUTS, self.BASE, 0.5)
        self.assertEqual(p["status"], "golden_margin")
        self.assertEqual(p["outputs"]["0"], {"ref": "golden", "margin": 0.5, "required": 0.99, "step": 2})
        # the seed misses the golden at the margin but passes SOL's own tolerance
        p = sol_gate.decide_policy(sol_of(golden=rec(0.97, atol=0.025), golden1=rec(0.995), gate="golden"), self.OUTS, self.BASE, 0.5)
        self.assertEqual(p["status"], "golden_full_tolerance")
        self.assertEqual(p["outputs"]["0"]["margin"], 1.0)

    def test_ladder_sol_unreachable_92015_like(self):
        # the seed fails SOL's reference (0.0067) but is within tolerance of the golden (0.9928): the gate moves to the golden
        p = sol_gate.decide_policy(sol_of(sol=rec(0.0067), golden=rec(0.9928, atol=0.04), golden1=rec(0.9928)), self.OUTS, self.BASE, 0.5)
        self.assertEqual(p["status"], "golden_margin")
        self.assertEqual(p["outputs"]["0"]["ref"], "golden")
        self.assertAlmostEqual(p["sol_unreachable"]["out"]["matched_ratio"], 0.0067)       # recorded, never silent
        self.assertEqual(p["sol_unreachable"]["out"]["atol"], 0.05)

    def test_ladder_no_regression_when_even_the_golden_fails(self):
        p = sol_gate.decide_policy(sol_of(sol=rec(0.01), golden=rec(0.60, atol=0.025), golden1=rec(0.70)), self.OUTS, self.BASE, 0.5)
        self.assertEqual(p["status"], "no_regression")
        o = p["outputs"]["0"]
        self.assertEqual(o["ref"], "golden")
        self.assertEqual(o["margin"], 1.0)
        self.assertAlmostEqual(o["required"], 0.70 * 0.999, places=5)       # a kernel may not be worse than the seed
        self.assertIn("sol_unreachable", p)

    def test_nothing_measured(self):
        p = sol_gate.decide_policy({}, self.OUTS, self.BASE, 0.5)
        self.assertEqual(p["status"], "unchecked")

    def test_multi_output_policy_per_output(self):
        outs = {"a": {"index": 0}, "b": {"index": 1}}
        sol = {**sol_of("a", sol=rec(1.0)), **sol_of("b", sol=rec(0.2), golden=rec(1.0, atol=.02), golden1=rec(1.0))}
        p = sol_gate.decide_policy(sol, outs, {0: 0.99, 1: 0.99}, 0.5)
        self.assertEqual(p["outputs"]["0"]["ref"], "sol")
        self.assertEqual(p["outputs"]["1"]["ref"], "golden")
        self.assertEqual(list(p["sol_unreachable"]), ["b"])


def new_cache(root: Path, ir_name: str, impls: list[tuple[str, str]]) -> CudaCache:
    """A cache with one entry per (impl_id, op)."""
    c = CudaCache(root)
    for iid, op in impls:
        c.add(op=op, entry=f"kernel_{op}", signature=f"void kernel_{op}(float *x)", arch="sm_75", shapes=[{"n": 1}],
              code=f"// {iid}", slot_us=1.0, speedup=2.0, impl_id=iid, source={}, dtype="fp32")
    c.save()
    return c


class GateVerifyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mb_sol_gate_"))
        self.env = {k: os.environ.get(k) for k in ("MB_SOL_GATE", "MB_SOL_TOLERANCE", "MB_SOL_SLOT_FRAC", "MB_SOL_GOLDEN_MARGIN",
                                                    "MB_SOL_BOUND", "MB_SOL_BENCH")}
        for k in self.env:
            os.environ.pop(k, None)
        st.STATE.reset()
        (self.tmp / "ir").mkdir()
        (self.tmp / "ir" / "graph.json").write_text(json.dumps({"name": "m", "output": {"tensors": ["out"]}}))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        for k, v in self.env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        st.STATE.reset()

    def gate(self, mode="final", **kw):
        cfg = go.Config(kb=self.tmp, sol_gate=mode, **kw)
        g = sol_gate.SolGate(cfg, self.tmp / "work")
        g.irs["ir"] = {"ir_name": "m", "ir_dir": str(self.tmp / "ir"), "tolerance": [st.normalize(TOL)], "rollbacks": [],
                       "policy": {"status": "reachable", "outputs": {"0": {"ref": "sol", "margin": 1.0, "required": 0.99}}}}
        return g

    def scripted(self, cache, bad, errs, calls):
        """A verify_ir stand-in: SOL fails while any impl in ``bad`` is still usable for IR m."""
        def verify():
            calls.append(st.effective("m"))
            live = {iid: op for iid, op in errs.items()
                    if not any(iid == e["impl_id"] and "m" in e["rejected"] for e in cache.entries)}
            slots = {op: {"impl_id": iid, "ok": True, "rel_err": errs[iid]} for iid, op in
                     {i: o for i, o in zip(errs, ("conv", "mm", "ewise", "red", "soft", "x")) if i in live}.items()}
            fail = any(i in live for i in bad)
            sol = sol_of(sol=rec(0.98 if fail else 1.0))
            return {"ir": "m", "passed": not fail, "slots": slots, "sol": sol, "partition_us": 5.0}
        return verify

    def test_off_is_one_call_and_no_scope(self):
        calls = []
        cache = new_cache(self.tmp / "c", "m", [("a", "conv")])
        g = self.gate("off")
        v = g.verify(self.tmp / "ir", self.scripted(cache, ["a"], {"a": 5e-4}, calls), cache)
        self.assertEqual(calls, [None])                                    # off: no scope, one call
        self.assertFalse(v["passed"])                                      # and nothing was rolled back
        self.assertNotIn("sol_gate", v)
        self.assertFalse((self.tmp / "work" / "sol_gate.json").exists())
        self.assertEqual(cache.entries[0]["rejected"], [])
        self.assertEqual(g.round_key(), {})
        self.assertIsNone(g.model_record("ir", {}))

    def test_rollback_of_the_blamed_class_then_pass(self):
        calls = []
        cache = new_cache(self.tmp / "c", "m", [("tf32_conv", "conv"), ("tf32_mm", "mm"), ("fp32_ew", "ewise")])
        errs = {"tf32_conv": 5e-4, "tf32_mm": 4e-4, "fp32_ew": 1e-6}
        g = self.gate("final")
        v = g.verify(self.tmp / "ir", self.scripted(cache, ["tf32_conv", "tf32_mm"], errs, calls), cache, where="final")
        self.assertTrue(v["passed"])
        self.assertEqual(calls, ["enforce", "enforce"])                    # failed once, passed after one rollback
        rej = {e["impl_id"]: e["rejected"] for e in cache.entries}
        self.assertEqual(rej, {"tf32_conv": ["m"], "tf32_mm": ["m"], "fp32_ew": []})   # the whole class, not the fp32 slot
        sg = v["sol_gate"]
        self.assertEqual(len(sg["attempts"]), 2)
        self.assertEqual(sorted(sg["attempts"][0]["rolled_back"]), ["conv", "mm"])
        self.assertEqual(g.irs["ir"]["rollbacks"][0]["where"], "final")
        saved = json.loads((self.tmp / "work" / "sol_gate.json").read_text())
        self.assertEqual(saved["irs"]["ir"]["final"]["passed"], True)
        self.assertEqual(saved["irs"]["ir"]["rollbacks"][0]["impl_ids"], ["tf32_conv", "tf32_mm"])

    def test_one_class_per_attempt_when_errors_differ(self):
        calls = []
        cache = new_cache(self.tmp / "c", "m", [("a", "conv"), ("b", "mm"), ("c", "ewise")])
        errs = {"a": 1e-3, "b": 1e-5, "c": 1e-7}
        g = self.gate("accept")
        v = g.verify(self.tmp / "ir", self.scripted(cache, ["a", "b"], errs, calls), cache, where="round 0")
        self.assertTrue(v["passed"])
        self.assertEqual(len(calls), 3)                                    # a rolled back, then b, then pass
        self.assertEqual([e["rejected"] for e in cache.entries], [["m"], ["m"], []])

    def test_budget_exhausted_reverts_everything_used(self):
        calls = []
        ids = [(f"w{i}", op) for i, op in enumerate(("conv", "mm", "ewise", "red"))]
        cache = new_cache(self.tmp / "c", "m", ids)
        errs = {"w0": 1e-3, "w1": 1e-5, "w2": 1e-7, "w3": 1e-9}
        g = self.gate("final", sol_max_rollbacks=1)
        # only the last (tiny-error) one really breaks SOL: blame goes to the biggest errors first, the budget runs out
        v = g.verify(self.tmp / "ir", self.scripted(cache, ["w3"], errs, calls), cache)
        self.assertTrue(v["passed"])
        self.assertTrue(all(e["rejected"] == ["m"] for e in cache.entries))     # revert-all: every used winner is out
        self.assertEqual(len(calls), 3)

    def test_a_failure_that_is_not_sols_passes_through(self):
        cache = new_cache(self.tmp / "c", "m", [("a", "conv")])
        g = self.gate("final")
        calls = []

        def verify():
            calls.append(1)
            return {"ir": "m", "passed": False, "slots": {"conv": {"impl_id": "a", "ok": False, "rel_err": 1.0}},
                    "sol": sol_of(sol=rec(1.0))}                           # SOL fine; the flow's own check failed
        v = g.verify(self.tmp / "ir", verify, cache)
        self.assertEqual(len(calls), 1)
        self.assertEqual(cache.entries[0]["rejected"], [])                 # the flow's own fallback handles it, not the gate

    def test_no_cache_entries_means_the_seed(self):
        g = self.gate("final")
        v = g.verify(self.tmp / "ir", lambda: {"ir": "m", "status": "no-cache-entries"}, new_cache(self.tmp / "c", "m", []))
        self.assertEqual(v["status"], "no-cache-entries")

    def test_ir_without_a_tolerance_is_not_gated(self):
        g = self.gate("final")
        g.irs["ir"] = {"ir_name": "m", "status": "no_sol_tolerance"}
        calls = []
        g.verify(self.tmp / "ir", lambda: calls.append(st.effective("m")) or {"passed": True}, new_cache(self.tmp / "c", "m", []))
        self.assertEqual(calls, [None])
        self.assertEqual(g.model_record("ir", {})["status"], "no_sol_tolerance")

    def test_model_record_flags_unverified_results(self):
        g = self.gate("final")
        g.irs["ir"]["policy"] = {"status": "golden_margin", "outputs": {"0": {"ref": "golden", "margin": .5, "required": .99}},
                                 "sol_unreachable": {"out": {"matched_ratio": 0.0067}}}
        m = g.model_record("ir", {"sol": sol_of(golden=rec(0.995), gate="golden")})
        self.assertTrue(m["pass"])
        self.assertTrue(m["unverified_vs_sol"])                            # gated against the golden: never silently "verified"
        self.assertEqual(m["sol_unreachable"]["out"]["matched_ratio"], 0.0067)
        g.irs["ir"]["policy"] = {"status": "reachable", "outputs": {"0": {"ref": "sol"}}}
        m = g.model_record("ir", {"sol": sol_of(sol=rec(1.0))})
        self.assertFalse(m["unverified_vs_sol"])
        self.assertTrue(m["pass"])
        m = g.model_record("ir", {"sol": sol_of(sol=rec(0.5))})
        self.assertFalse(m["pass"])

    def test_profile_fallback_blames_a_sol_failure(self):
        res = {"passed": False, "slots": {"conv": {"ok": True, "rel_err": 5e-4}, "ew": {"ok": True, "rel_err": 1e-7}},
               "sol": sol_of(sol=rec(0.9))}
        ea, eb = {"impl_id": "a", "rejected": []}, {"impl_id": "b", "rejected": []}
        plan = {"used": {"conv": ea, "ew": eb}}
        self.assertEqual(kb_plan.failed_cached_slots(res, plan), {"conv": ea})
        res["sol"] = sol_of(sol=rec(1.0))
        self.assertEqual(kb_plan.failed_cached_slots(res, plan), {})       # SOL passed: nothing to blame
        res = {"passed": False, "slots": {"conv": {"ok": False, "rel_err": 1.0}, "ew": {"ok": True}}, "sol": sol_of(sol=rec(0.9))}
        self.assertEqual(kb_plan.failed_cached_slots(res, plan), {"conv": ea})   # a slot's own failure first: as before
        self.assertEqual(kb_plan.failed_cached_slots({"passed": False, "slots": {"conv": {"ok": False}}}, plan), {"conv": ea})


# ---------------------------------------------------------------- the round loop with every GPU step faked

class LoopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="mb_sol_loop_"))
        cls.o = F.fire_ir(cls.tmp / "ir", "fp16")
        (cls.tmp / "kb" / "scripts").mkdir(parents=True)
        (cls.tmp / "kb" / "scripts" / "time_problems.py").write_text("")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.saved = {k: getattr(go, k) for k in ("profile_ir", "verify_ir", "detect_gpus", "make_plan", "remeasure", "KBRunner")}
        self.saved_mr, self.saved_bar = mr.model_runtime, kb_export.build_and_run
        keys = ("KERNELBLASTER_GPU_FLOCK", "MB_SOL_GATE", "MB_SOL_TOLERANCE", "MB_SOL_SLOT_FRAC", "MB_SOL_GOLDEN_MARGIN",
                "MB_SOL_BOUND", "MB_SOL_BENCH", "MB_SOL_REF")
        self.env = {k: os.environ.get(k) for k in keys}
        for k in keys:
            os.environ.pop(k, None)
        os.environ["KERNELBLASTER_GPU_FLOCK"] = "0"
        os.environ["MB_SOL_TOLERANCE"] = json.dumps(TOL)
        os.environ["MB_SOL_REF"] = "0"
        st.STATE.reset()
        self.verifies, self.probes, self.models = [], [], []
        self.tf32_seen = {"n": 0}

        def profile(irdir, work, *, hints_doc=None, **kw):
            d, specs, prov = gh.apply_to_ir(Path(irdir), hints_doc or gh.new_doc(), Path(work) / "ir")
            return _profile(d, irdir)

        def remeasure(problems, *, arch, reps, lease=None, **kw):
            return [[100.0 if json.loads((Path(d) / "manifest.json").read_text())["name"].endswith("__orig") else 80.0] * reps
                    for d, _ in problems]

        def build_and_run(problem_dir, **kw):
            man = json.loads((Path(problem_dir) / "manifest.json").read_text())
            self.probes.append({"enforce": man["sol"]["enforce"], "outputs": list(man["sol"]["outputs"]), "env": kw.get("env")})
            name = next(iter(man["sol"]["outputs"]))
            return {"passed": True, "sol": sol_of(name, golden=rec(1.0, atol=0.025), golden1=rec(1.0), gate="golden"),
                    "device": TITAN}

        def verify_ir(ir, cache, **kw):
            # the first calls find a TF32-like winner in the cache that breaks SOL until it is rolled back
            if not any(e["impl_id"] == "tf32_conv" for e in cache.entries):
                for iid, op in (("tf32_conv", "conv2d_f16"), ("fp32_relu", "relu_f16")):
                    cache.add(op=op, entry=f"kernel_{op}", signature=f"void kernel_{op}(float *x)", arch="sm_75", shapes=[{"n": 1}],
                              code=f"// {iid}", slot_us=1.0, speedup=2.0, impl_id=iid, source={}, dtype="fp16")
                cache.save()
            live = [e for e in cache.entries if "fire" not in e["rejected"]]
            bad = any(e["impl_id"] == "tf32_conv" for e in live)
            self.verifies.append({"scope": st.effective("fire"), "bad": bad})
            return {"ir": "fire", "passed": not bad, "partition_us": 5.0, "device": TITAN,
                    "slots": {e["op"]: {"impl_id": e["impl_id"], "ok": True, "rel_err": 5e-4 if e["impl_id"] == "tf32_conv" else 1e-7}
                              for e in live},
                    **({"sol": sol_of(sol=rec(0.986))} if st.effective("fire") else {})}

        def model_runtime(ir, **kw):
            self.models.append(st.effective("fire"))
            return {"device": TITAN, "passed": True, "model_e2e_us": 4.0,
                    "model": {"sol": sol_of(golden=rec(1.0), gate="golden")} if st.effective("fire") else {}}
        go.profile_ir, go.verify_ir, go.remeasure, go.KBRunner = profile, verify_ir, remeasure, FakeKB
        mr.model_runtime = model_runtime
        kb_export.build_and_run = build_and_run
        go.make_plan = lambda *a, **k: {"problems": [], "selected_share": 0.0}
        go.detect_gpus = lambda cfg: {"local": [{"ident": "GPU-titan", "index": 0, "name": "NVIDIA TITAN RTX"}], "remote": []}

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(go, k, v)
        mr.model_runtime, kb_export.build_and_run = self.saved_mr, self.saved_bar
        for k, v in self.env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        st.STATE.reset()

    def run_loop(self, name, **kw):
        cfg = go.Config(kb=self.tmp / "kb", arch="75", **{"rounds": 2, "max_variants": 3, "remeasure": 3, "accept_gain": 0.03, **kw})
        w = self.tmp / name
        return go.run([self.o], w, cfg), w

    def test_accept_probe_rollback_and_reporting(self):
        s, w = self.run_loop("acc", sol_gate="accept")
        # the seed probe: one report-only export, the policy decided from its MB_SOL lines
        self.assertEqual(len(self.probes), 1)
        self.assertFalse(self.probes[0]["enforce"])
        self.assertEqual(self.probes[0]["env"], {"MB_SCREEN": "1"})
        g = json.loads((w / "sol_gate.json").read_text())["irs"]["ir"]
        self.assertEqual(g["policy"]["status"], "golden_margin")               # no SOL reference here: the golden at the margin
        self.assertEqual(g["policy"]["outputs"]["0"]["margin"], 0.5)
        self.assertEqual(g["reference"], None)
        self.assertTrue(any("no SOL reference" in x for x in g["warnings"]))
        # the first round's check found the winner breaking SOL's rule and rolled it back; later checks passed
        self.assertEqual(len(g["rollbacks"]), 1)
        self.assertEqual(g["rollbacks"][0]["impl_ids"], ["tf32_conv"])
        self.assertEqual(g["rollbacks"][0]["where"], "round 0")
        self.assertEqual(g["final"]["where"], "final")
        self.assertTrue(g["final"]["passed"])
        self.assertTrue(all(v["scope"] == "enforce" for v in self.verifies))
        cache = CudaCache(w / "cache")
        rej = {e["impl_id"]: e["rejected"] for e in cache.entries}
        self.assertEqual((rej["tf32_conv"], rej["fp32_relu"]), (["fire"], []))
        # reporting: summary.json, model_runtime.json (next to the existing fields)
        sg = s["sol_gate"]
        self.assertEqual(sg["mode"], "accept")
        self.assertEqual(sg["irs"]["ir"]["rollbacks"][0]["impl_ids"], ["tf32_conv"])
        ir = s["irs"]["fire"]
        self.assertTrue(ir["sol"]["pass"])
        self.assertTrue(ir["sol"]["unverified_vs_sol"])                         # gated against the golden: said so
        mrj = json.loads((w / "model_runtime.json").read_text())
        self.assertTrue(mrj["ir"]["sol"]["pass"])
        self.assertEqual(mrj["ir"]["sol"]["status"], "golden_margin")
        self.assertEqual(self.models, ["enforce"])
        self.assertIn("sol_gate", json.loads((w / "r00" / "round.json").read_text()))

    def test_final_mode_checks_only_the_end(self):
        s, w = self.run_loop("fin", sol_gate="final")
        self.assertTrue(all(v["scope"] == "enforce" for v in self.verifies))
        self.assertEqual(len(self.verifies), 2)                                 # the failure, then the pass after the rollback
        g = json.loads((w / "sol_gate.json").read_text())["irs"]["ir"]
        self.assertEqual(g["rollbacks"][0]["where"], "final")
        self.assertNotIn("sol_gate", json.loads((w / "r00" / "round.json").read_text()))   # nothing during the rounds

    def test_resume_keeps_the_policy_and_probes_once(self):
        self.run_loop("res", sol_gate="accept")
        self.assertEqual(len(self.probes), 1)
        st.STATE.reset()
        self.run_loop("res", sol_gate="accept", rounds=3)
        self.assertEqual(len(self.probes), 1)                                   # a resumed work dir does not probe again

    def test_off_leaves_no_trace(self):
        s, w = self.run_loop("off")
        self.assertNotIn("sol_gate", s)
        self.assertNotIn("sol", s["irs"]["fire"])
        self.assertEqual(self.probes, [])
        self.assertFalse((w / "sol_gate.json").exists())
        self.assertFalse((w / "sol_gate").exists())
        self.assertEqual([v["scope"] for v in self.verifies], [None])           # the final verification only, no scope
        self.assertEqual(self.models, [None])
        self.assertNotIn("sol", json.loads((w / "model_runtime.json").read_text())["ir"])
        self.assertNotIn("sol", " ".join(sorted(os.environ.get(k, "") for k in ("MB_SOL_GATE",))))
        for r in sorted(w.glob("r0*/round.json")):
            self.assertNotIn("sol_gate", json.loads(r.read_text()))

    def test_gate_flag_off_explicit_equals_default(self):
        a, wa = self.run_loop("o1")
        self.verifies.clear()
        b, wb = self.run_loop("o2", sol_gate="off")
        ka = json.loads((wa / "state.json").read_text())["rounds"][0]["key"]
        kb = json.loads((wb / "state.json").read_text())["rounds"][0]["key"]
        self.assertEqual(ka, kb)                                                # the round key is untouched
        self.assertEqual(sorted(a["irs"]["fire"]), sorted(b["irs"]["fire"]))

    def test_round_key_only_with_the_gate_on(self):
        self.run_loop("k0")
        self.run_loop("k1", sol_gate="final")
        k0 = json.loads((self.tmp / "k0" / "state.json").read_text())["rounds"][0]["key"]
        k1 = json.loads((self.tmp / "k1" / "state.json").read_text())["rounds"][0]["key"]
        self.assertNotEqual(k0, k1)


if __name__ == "__main__":
    unittest.main()
