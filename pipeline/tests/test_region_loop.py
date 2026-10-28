"""ModelBlaster side of the region loop (region_loop.py, ``--region-loop off|on``; docs/region_loop.md).

CPU only. The flag / env / budget resolution, the KernelBlaster command line (flags only when on), the round key / summary, the pre-loop
gate (graph_verify on a variant with region ops: a faithful windowed variant passes, a tampered one is rejected before any export), and
the whole round loop with fake GPU steps: gate recorded, variant exported with region slots, rejected hypothesis on a failing gate, and
flag-off identity (off == unset == env off: same hints, summary, rounds, round key).
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import modelblaster.pipeline.kb_model_run as mr  # noqa: E402
from modelblaster.pipeline import graph_hints as gh  # noqa: E402
from modelblaster.pipeline import graph_verify as gv  # noqa: E402
from modelblaster.pipeline import kb_graph_opt as go  # noqa: E402
from modelblaster.pipeline import region_loop as rlm  # noqa: E402
from modelblaster.pipeline import region_windows as rw  # noqa: E402
from modelblaster.pipeline.kb_proposals import _all_specs, with_fused_specs  # noqa: E402
from modelblaster.pipeline.tests import window_fixtures as wf  # noqa: E402
from modelblaster.pipeline.tests.test_graph_rewrite import TITAN, _profile  # noqa: E402
from modelblaster.pipeline.tests.test_kb_graph_opt_parallel import FakeKB  # noqa: E402


def _cfg(**kw):
    return mock.Mock(spec=[], **kw)


class FlagTest(unittest.TestCase):
    def setUp(self):
        self.env = {k: os.environ.pop(k, None) for k in (rlm.ENV, rlm.ENV_BUDGET)}

    def tearDown(self):
        for k, v in self.env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_mode(self):
        self.assertEqual(rlm.resolve_mode(_cfg()), "off")
        for v, want in (("on", "on"), ("off", "off"), ("1", "on"), ("0", "off"), ("", "off"), (" ON ", "on")):
            self.assertEqual(rlm.resolve_mode(_cfg(region_loop=v)), want)
        with self.assertRaises(ValueError):
            rlm.resolve_mode(_cfg(region_loop="always"))
        os.environ[rlm.ENV] = "on"
        self.assertEqual(rlm.resolve_mode(_cfg()), "on")
        self.assertEqual(rlm.resolve_mode(_cfg(region_loop="off")), "off")      # the flag wins over the env

    def test_budget_and_kb_args(self):
        self.assertEqual(rlm.kb_args(_cfg()), [])                                # off: the command line is what it was
        self.assertEqual(rlm.kb_args(_cfg(region_loop="off", region_loop_budget="9,9,9,9")), [])
        self.assertEqual(rlm.kb_args(_cfg(region_loop="on")), ["--region-loop", "on"])
        self.assertEqual(rlm.kb_args(_cfg(region_loop="on", region_loop_budget="10,,2.5")), ["--region-loop", "on", "--region-loop-budget", "10,,2.5"])
        os.environ[rlm.ENV_BUDGET] = "5,5,1,60"
        self.assertEqual(rlm.kb_args(_cfg(region_loop="on"))[-1], "5,5,1,60")
        for bad in ("a", "1,2,3,4,5", "0", "-1,2"):
            with self.assertRaises(ValueError):
                rlm.kb_args(_cfg(region_loop="on", region_loop_budget=bad))

    def test_command_line(self):
        base = dict(kb=Path("/kb"), python="py", model="m", arch="75")
        off = go.KBRunner(go.Config(**base), Path("/k")).command([Path("/p")], Path("/o"))
        explicit = go.KBRunner(go.Config(**base, region_loop="off"), Path("/k")).command([Path("/p")], Path("/o"))
        self.assertEqual(off, explicit)
        self.assertNotIn("--region-loop", off)
        on = go.KBRunner(go.Config(**base, region_loop="on", region_loop_budget="8,40,3,600"), Path("/k")).command([Path("/p")], Path("/o"))
        i = on.index("--region-loop")
        self.assertEqual(on[i:i + 4], ["--region-loop", "on", "--region-loop-budget", "8,40,3,600"])
        self.assertEqual([x for x in on if x not in ("--region-loop", "on", "--region-loop-budget", "8,40,3,600")], off)

    def test_hook(self):
        h = rlm.RegionLoopHook(_cfg())
        self.assertFalse(h.active())
        self.assertEqual(h.round_key(), {})
        h = rlm.RegionLoopHook(_cfg(region_loop="on", region_loop_budget="8,40,3,600"))
        self.assertEqual(h.round_key(), {"region_loop": "on", "region_loop_budget": "8,40,3,600"})
        self.assertEqual(h.summary()["gate"], "on")
        h = rlm.RegionLoopHook(_cfg(region_loop="on", region_loop_gate="off"))
        self.assertEqual(h.round_key(), {"region_loop": "on", "region_loop_gate": "off"})
        with self.assertRaises(ValueError):
            rlm.RegionLoopHook(_cfg(region_loop="on", region_loop_gate="maybe"))


class GateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="mb_rl_gate_"))
        cls.g = wf.rnn_unrolled(16, fused=False)
        cls.sp = with_fused_specs(cls.g, dict(_all_specs()))
        cls.o = wf.ir_dir_from_graph(cls.tmp / "orig", cls.g)
        cands, _ = rw.propose_windows(cls.g, cls.sp, rw.WindowConfig(mode="auto", sizes=(30,), min_ops=2))
        cls.c = next(c for c in cands if c.label == "chunk30")
        h = rw.hint_for(cls.c, "m")
        h["status"] = "accepted"
        cls.h = h
        with mock.patch.dict(os.environ, {"MB_REGION_MAX_MEMBERS": "300"}):
            cls.vg, _, _ = gh.apply_hints(copy.deepcopy(cls.g), [h], cls.sp, statuses=("accepted",))
        cls.v = wf.ir_dir_from_graph(cls.tmp / "var", cls.vg)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        rlm.GATED.clear()

    def test_off_and_non_region_are_inert(self):
        self.assertIsNone(rlm.gate(_cfg(), self.h, self.vg, self.o, self.v, self.sp))
        self.assertEqual(rlm.GATED, [])
        self.assertIsNone(rlm.gate(_cfg(region_loop="on"), self.h, self.g, self.o, self.o, self.sp))       # no region op in the variant
        self.assertIsNone(rlm.gate(_cfg(region_loop="on", region_loop_gate="off"), self.h, self.vg, self.o, self.v, self.sp))
        self.assertEqual(rlm.GATED, [])

    def test_faithful_variant_passes(self):
        why = rlm.gate(_cfg(region_loop="on"), self.h, self.vg, self.o, self.v, self.sp)
        self.assertIsNone(why, why)
        self.assertEqual(len(rlm.GATED), 1)
        rec = rlm.GATED[0]
        self.assertTrue(rec["ok"])
        self.assertIn("outputs", " ".join(rec["checks"]) + " outputs")                  # numeric checks ran
        self.assertTrue(rec["checks"])

    def test_tampered_variant_is_rejected(self):
        """A region whose member computes something else (a linear's N changed in one member): structural / numeric rejection."""
        bad = copy.deepcopy(self.vg)
        reg = next(o for o in bad["ops"] if o["op"].startswith("region__"))
        member = next(m for m in reg["sub_ops"] if m["op"] == "tanh_f16")
        member["op"] = "relu_f16"                                                       # tanh -> relu: same signature, other function
        for m in reg["sub_ops"]:                                                        # keep the same dtype contract
            pass
        d = wf.ir_dir_from_graph(self.tmp / "bad", bad)
        why = rlm.gate(_cfg(region_loop="on"), self.h, bad, self.o, d, self.sp)
        self.assertIsNotNone(why)
        self.assertIn("region gate (graph_verify)", why)
        self.assertFalse(rlm.GATED[-1]["ok"])
        self.assertTrue(rlm.GATED[-1]["reasons"])

    def test_gate_failure_to_run_is_a_rejection(self):
        with mock.patch.object(gv, "verify", side_effect=RuntimeError("walker died")):
            why = rlm.gate(_cfg(region_loop="on"), self.h, self.vg, self.o, self.v, self.sp)
        self.assertIn("could not run", why)
        self.assertFalse(rlm.GATED[-1]["ok"])


class LoopTest(unittest.TestCase):
    """kb_graph_opt._run with every GPU step faked, windows proposed, the region loop flag on / off."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="mb_rl_loop_"))
        cls.ir = wf.ir_dir_from_graph(cls.tmp / "ir", wf.rnn_unrolled(24, fused=False))
        (cls.tmp / "kb" / "scripts").mkdir(parents=True)
        (cls.tmp / "kb" / "scripts" / "time_problems.py").write_text("")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.saved = {k: getattr(go, k) for k in ("profile_ir", "verify_ir", "detect_gpus", "make_plan", "remeasure", "KBRunner")}
        self.saved_mr = mr.model_runtime
        keys = ("KERNELBLASTER_GPU_FLOCK", rlm.ENV, rlm.ENV_BUDGET, "MB_REGION_WINDOWS", "MB_REGION_MAX_MEMBERS")
        self.env = {k: os.environ.get(k) for k in keys}
        os.environ["KERNELBLASTER_GPU_FLOCK"] = "0"
        for k in keys[1:]:
            os.environ.pop(k, None)
        self.runs = []
        outer = self

        class KB(FakeKB):
            def start(self, problems, out, **kw):
                outer.runs.append(self.command(problems, out))
                return super().start(problems, out, **kw)

        def profile(irdir, work, *, hints_doc=None, **kw):
            d, specs, prov = gh.apply_to_ir(Path(irdir), hints_doc or gh.new_doc(), Path(work) / "ir")
            return _profile(d, irdir)

        def remeasure(problems, *, arch, reps, lease=None, **kw):
            return [[100.0 if json.loads((Path(d) / "manifest.json").read_text())["name"].endswith("__orig") else 70.0] * reps
                    for d, _ in problems]
        go.profile_ir = profile
        go.verify_ir = lambda ir, cache, **kw: {"ir": "synthetic", "passed": True, "partition_us": 5.0, "device": TITAN, "slots": {}}
        mr.model_runtime = lambda ir, **kw: {"device": TITAN, "passed": True, "model_e2e_us": 4.0}
        go.remeasure, go.KBRunner = remeasure, KB
        go.make_plan = lambda *a, **k: {"problems": [], "selected_share": 0.0}
        go.detect_gpus = lambda cfg: {"local": [{"ident": "GPU-titan", "index": 0, "name": "NVIDIA TITAN RTX"}], "remote": []}

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(go, k, v)
        mr.model_runtime = self.saved_mr
        for k, v in self.env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def _run(self, name, **kw):
        cfg = go.Config(kb=self.tmp / "kb", arch="75", rounds=2, max_variants=2, remeasure=3, accept_gain=0.03, **kw)
        s = go.run([self.ir], self.tmp / name, cfg)
        return s, gh.load(self.tmp / name / "graph_hints.json"), json.loads((self.tmp / name / "state.json").read_text())

    WIN = dict(region_windows="auto", region_window_sizes="40", region_window_min_ops=2)

    def test_on_gates_exports_and_passes_the_flag_to_kernelblaster(self):
        s, doc, st = self._run("on", region_loop="on", region_loop_budget="8,40,3,600", **self.WIN)
        rec = s["region_loop"]
        self.assertEqual((rec["mode"], rec["budget"], rec["gate"]), ("on", "8,40,3,600", "on"))
        self.assertTrue(rec["gated"] and all(g["ok"] for g in rec["gated"]), rec["gated"])
        wins = [h for h in doc["hints"] if h["id"].startswith("region:win:")]
        self.assertTrue(any(h["status"] == "accepted" for h in wins), [(h["id"], h["status"]) for h in wins])
        self.assertTrue(self.runs and all("--region-loop" in c and c[c.index("--region-loop") + 1] == "on" for c in self.runs))
        self.assertTrue(all("--region-loop-budget" in c for c in self.runs))
        var = next((self.tmp / "on" / "r00" / "variants").glob("*region*/variant/manifest.json"))
        self.assertTrue(any(x["id"].startswith("region__") for x in json.loads(var.read_text())["slots"]))

    def test_a_failing_gate_rejects_the_hypothesis_before_any_export(self):
        bad = gv.Verdict()
        bad.fail("outputs", "variant output differs from the original", "outputs")
        with mock.patch.object(gv, "verify", return_value=bad):
            s, doc, st = self._run("rej", region_loop="on", **self.WIN)
        wins = [h for h in doc["hints"] if h["id"].startswith("region:win:")]
        self.assertTrue(wins and all(h["status"] == "rejected" for h in wins))
        self.assertIn("region gate (graph_verify)", wins[0]["evidence"][-1]["why"])
        self.assertEqual(wins[0]["evidence"][-1]["decision"], "rejected")
        self.assertFalse(list((self.tmp / "rej" / "r00" / "variants").glob("*region*/variant/manifest.json")))     # never exported, never timed
        self.assertTrue(s["region_loop"]["gated"] and not s["region_loop"]["gated"][0]["ok"])

    def test_gate_off_skips_graph_verify(self):
        with mock.patch.object(gv, "verify", side_effect=AssertionError("must not run")):
            s, doc, st = self._run("nogate", region_loop="on", region_loop_gate="off", **self.WIN)
        self.assertEqual(s["region_loop"]["gate"], "off")
        self.assertEqual(s["region_loop"]["gated"], [])
        self.assertTrue(any(h["status"] == "accepted" for h in doc["hints"] if h["id"].startswith("region:win:")))

    def test_off_equals_unset_equals_env_off(self):
        a = self._run("a", **self.WIN)
        b = self._run("b", region_loop="off", **self.WIN)
        os.environ[rlm.ENV] = "off"
        c = self._run("c", **self.WIN)
        os.environ.pop(rlm.ENV)

        def norm(x):
            s, doc, st = x

            def strip(o):
                if isinstance(o, dict):
                    return {k: strip(v) for k, v in o.items() if k not in ("seconds", "decide_seconds", "ts", "telemetry", "key")}
                if isinstance(o, list):
                    return [strip(v) for v in o]
                if isinstance(o, str):
                    return o.replace(str(self.tmp), "<T>")
                return o
            return json.dumps(strip({"hints": doc, "summary": s, "rounds": st["rounds"]}), sort_keys=True, default=str)
        self.assertEqual(norm(a), norm(b))
        self.assertEqual(norm(a), norm(c))
        self.assertNotIn("region_loop", a[0])
        k = lambda n: json.loads((self.tmp / n / "state.json").read_text())["rounds"][0]["key"]      # noqa: E731
        self.assertEqual(k("a"), k("b"))
        self.assertEqual(k("a"), k("c"))
        self.assertNotIn("region_loop", k("a"))
        self.assertTrue(self.runs and all("--region-loop" not in c for c in self.runs))               # off: the command line has no flag

    def test_the_flag_changes_the_round_key_only_when_on(self):
        self._run("k_off", **self.WIN)
        self._run("k_on", region_loop="on", **self.WIN)
        k = lambda n: json.loads((self.tmp / n / "state.json").read_text())["rounds"][0]["key"]      # noqa: E731
        self.assertNotEqual(k("k_off"), k("k_on"))


if __name__ == "__main__":
    unittest.main()
