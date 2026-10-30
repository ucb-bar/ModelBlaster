"""Round controller (round_controller.py, ``--round-controller off|fixed|graph-first|auto``; docs/round_controller.md).

CPU only. Unit: flag / env resolution, the features (small unfused share, candidate classes), the budget guard, every policy's decision
(fixed / graph-first / auto cases with their reasons), the region-loop switch on the KernelBlaster command line. Flow (every GPU step
faked, as test_region_loop.LoopTest): off == unset == env off (same hints, summary, state, round.json: no controller key anywhere),
fixed / graph-first / auto records, a graph-only first round that defers the kernel plan, the guard stopping a move, the USD budget stop,
resume of a controller round, and the refusal with --graph-* search flags.
"""
from __future__ import annotations

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
from modelblaster.pipeline import kb_graph_opt as go  # noqa: E402
from modelblaster.pipeline import region_loop as rlm  # noqa: E402
from modelblaster.pipeline import round_controller as rc  # noqa: E402
from modelblaster.pipeline.tests import window_fixtures as wf  # noqa: E402
from modelblaster.pipeline.tests.test_graph_rewrite import TITAN, _profile  # noqa: E402
from modelblaster.pipeline.tests.test_kb_graph_opt_parallel import FakeKB  # noqa: E402


def _cfg(**kw):
    return mock.Mock(spec=[], **kw)


def _ctl(policy="auto", rl=False, **kw):
    return rc.RoundController(_cfg(round_controller=policy, **kw), region_loop_active=rl)


def _hint(i, kind, est):
    return {"id": i, "kind": kind, "status": "proposed", "target": "cuda", "source": [{"est_gain_pct": est, "ir": "m"}]}


def _prof(ops):
    return {"ir": "m", "ops": [{"name": f"o{i}", "op": k, "us": us} for i, (k, us) in enumerate(ops)]}


def _feat(c, *, first=True, hints=(), profiles=None, hist=(), left=3, spent=0.0, prev=None, cur=None):
    return c.features(r=len(hist), first=first, rounds_left=left, profiles=profiles or [_prof([("conv2d", 100.0)])], hints=list(hints),
                      min_gain_pct=2.0, hist=list(hist), spent_usd=spent, prev_total=prev, cur_total=cur)


def _h(move, kernel=0.0, graph=0.0, region=0.0, worked=("kernel", "graph")):
    return {"move": move, "kernel": "kernel" in move, "graph": "graph" in move, "region_loop": False,
            "gain": {"kernel_pct": kernel, "graph_pct": graph, "region_pct": region}, "worked": {m: m in worked for m in rc.MOVES}}


class FlagTest(unittest.TestCase):
    def setUp(self):
        self.env = {k: os.environ.pop(k, None) for k in (rc.ENV, rc.ENV_MIN_GAIN, rc.ENV_BUDGET_USD, rc.ENV_SMALL_US)}

    def tearDown(self):
        for k, v in self.env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_policy(self):
        self.assertEqual(rc.resolve_policy(_cfg()), "off")
        for v, want in (("off", "off"), ("0", "off"), ("", "off"), ("Fixed", "fixed"), ("graph-first", "graph-first"),
                        ("graph_first", "graph-first"), (" auto ", "auto")):
            self.assertEqual(rc.resolve_policy(_cfg(round_controller=v)), want)
        with self.assertRaises(ValueError):
            rc.resolve_policy(_cfg(round_controller="sometimes"))
        os.environ[rc.ENV] = "auto"
        self.assertEqual(rc.resolve_policy(_cfg()), "auto")
        self.assertEqual(rc.resolve_policy(_cfg(round_controller="off")), "off")      # the flag wins over the env

    def test_off_is_inert(self):
        c = rc.RoundController(_cfg())
        self.assertFalse(c.on)
        self.assertEqual(c.round_key(), {})

    def test_key_only_when_on_and_non_default(self):
        self.assertEqual(_ctl("fixed").round_key(), {"round_controller": "fixed"})
        k = _ctl("auto", round_controller_min_gain=5.0, round_controller_budget_usd=50.0, round_controller_small_us=9.0).round_key()
        self.assertEqual(k, {"round_controller": "auto", "rc_min_gain": 0.05, "rc_budget_usd": 50.0, "rc_small_us": 9.0})
        os.environ[rc.ENV_MIN_GAIN] = "4"
        self.assertAlmostEqual(_ctl("fixed").min_gain, 0.04)
        with self.assertRaises(ValueError):
            _ctl("fixed", round_controller_min_gain=-1.0)


class FeatureTest(unittest.TestCase):
    def test_small_unfused_share(self):
        p = _prof([("relu_f16", 5.0), ("add", 5.0), ("conv2d", 80.0), ("fused__conv2d__relu", 5.0), ("region__x", 3.0), ("softmax", 30.0)])
        s = rc.small_unfused_share([p], 15.0)
        self.assertEqual(s["n_small"], 2)
        self.assertAlmostEqual(s["share"], 10.0 / 128.0, places=3)      # fused / region ops, the big conv and the 30 us softmax are not small
        self.assertEqual(rc.small_unfused_share([{"ops": []}], 15.0)["share"], 0.0)
        self.assertEqual(rc.small_unfused_share([_prof([("relu", None)])], 15.0)["share"], 0.0)

    def test_candidate_classes(self):
        hs = [_hint("fuse:a", "fuse", 6.0), _hint("fuse:b", "fuse", 1.0), _hint("merge:x", "merge_sibling_convs", 4.0),
              _hint("region:win:c256", "fuse_region", 2.5), _hint("region:win:w", "fuse_region", 2.2), _hint("s", "split_slot", 3.0)]
        c = rc.candidate_classes(hs, 2.0)
        self.assertEqual((c["graph"]["n"], c["graph"]["est_gain_pct"], c["graph"]["best"]), (2, 9.0, "fuse:a"))      # est < 2 dropped
        self.assertEqual((c["merge"]["n"], c["merge"]["est_gain_pct"]), (1, 4.0))
        self.assertEqual((c["region"]["n"], c["region"]["est_gain_pct"]), (2, 4.7))
        self.assertEqual(rc.candidate_classes([], 2.0)["graph"], {"n": 0, "est_gain_pct": 0.0, "best": None})

    def test_realized(self):
        rec = {"partition_us": {"m": 200.0}, "kb_runs": [{"baseline_partition_us": 100.0, "final_partition_us": 90.0, "final_ok": True},
                                                          {"baseline_partition_us": 50.0, "final_partition_us": 40.0, "final_ok": False}],
               "decisions": [{"hint": "fuse:a", "hint_status": "accepted", "original_us": 100.0, "variant_us": 80.0},
                             {"hint": "region:win:c", "hint_status": "accepted", "original_us": 50.0, "variant_us": 45.0},
                             {"hint": "fuse:b", "hint_status": "rejected", "original_us": 50.0, "variant_us": 70.0}]}
        g = rc.realized(rec, {"fuse:a": "fuse", "region:win:c": "fuse_region", "fuse:b": "fuse"})
        self.assertEqual((g["kernel_pct"], g["graph_pct"], g["region_pct"]), (5.0, 10.0, 2.5))
        self.assertEqual(rc.realized({"partition_us": {}}, {})["kernel_pct"], 0.0)


class GuardTest(unittest.TestCase):
    def test_two_low_rounds_stop_a_move(self):
        c = _ctl("fixed")
        self.assertEqual(c.stopped([_h("kernel+graph", 1.0, 5.0), _h("kernel+graph", 2.9, 5.0)]).keys(), {"kernel"})
        self.assertEqual(c.stopped([_h("kernel+graph", 5.0, 1.0), _h("kernel+graph", 5.0, 0.0)]).keys(), {"graph"})
        self.assertEqual(c.stopped([_h("kernel+graph", 1.0, 5.0)]), {})                           # one round is not enough
        self.assertEqual(c.stopped([_h("kernel+graph", 1.0, 5.0), _h("kernel+graph", 3.0, 5.0)]), {})   # exactly 3% is not < 3%
        self.assertEqual(c.stopped([_h("kernel", 1.0, 0.0, worked=("kernel",)), _h("graph", 9.0, 0.0, worked=("graph",)),
                                    _h("kernel", 0.5, 0.0, worked=("kernel",))]).keys(), {"kernel"})   # the last two ROUNDS THE MOVE WORKED in
        self.assertEqual(c.stopped([_h("kernel+graph", 0.0, 0.0, worked=()), _h("kernel+graph", 0.0, 0.0, worked=())]), {})   # no work, no verdict

    def test_threshold_is_a_flag(self):
        c = _ctl("fixed", round_controller_min_gain=10.0)
        self.assertEqual(c.stopped([_h("kernel+graph", 8.0, 12.0), _h("kernel+graph", 9.0, 12.0)]).keys(), {"kernel"})

    def test_pre_stop(self):
        c = _ctl("fixed", round_controller_budget_usd=10.0)
        self.assertIn("budget", c.pre_stop([], 10.0))
        self.assertIsNone(c.pre_stop([], 9.99))
        low = [_h("kernel+graph", 0.1, 0.1), _h("kernel+graph", 0.1, 0.1)]
        self.assertIn("both stopped", _ctl("fixed").pre_stop(low, 0.0))
        self.assertIsNone(_ctl("fixed").pre_stop([_h("kernel+graph", 0.1, 5.0), _h("kernel+graph", 0.1, 5.0)], 0.0))


class DecideTest(unittest.TestCase):
    CANDS = [_hint("fuse:a", "fuse", 6.0), _hint("merge:x", "merge_sibling_convs", 4.0)]

    def test_fixed_runs_everything_with_a_reason(self):
        c = _ctl("fixed", rl=True)
        mv = c.decide(_feat(c, hints=self.CANDS), [])
        self.assertEqual((mv.kernel, mv.graph, mv.region_loop, mv.stop, mv.deferred), (True, True, True, False, False))
        self.assertEqual(mv.label(), "kernel+graph+region_loop")
        self.assertTrue(any(r.startswith("fixed:") for r in mv.reasons))
        self.assertFalse(_ctl("fixed").decide(_feat(_ctl("fixed")), []).region_loop)        # region loop flag off: never

    def test_graph_first_first_round_is_graph_only(self):
        c = _ctl("graph-first")
        mv = c.decide(_feat(c, hints=self.CANDS), [])
        self.assertEqual((mv.kernel, mv.graph, mv.deferred), (False, True, True))
        self.assertIn("graph-first", mv.reasons[-1])
        later = c.decide(_feat(c, first=False, hints=self.CANDS, hist=[_h("graph")]), [_h("graph")])
        self.assertEqual((later.kernel, later.graph), (True, True))
        none = c.decide(_feat(c), [])                                                       # nothing to restructure: kernel + graph
        self.assertEqual((none.kernel, none.graph, none.deferred), (True, True, False))

    def test_guard_in_every_policy(self):
        low = [_h("kernel+graph", 0.1, 5.0), _h("kernel+graph", 0.1, 5.0)]
        for pol in ("fixed", "graph-first", "auto"):
            c = _ctl(pol)
            mv = c.decide(_feat(c, first=False, hints=self.CANDS, hist=low), low)
            self.assertFalse(mv.kernel, pol)
            self.assertTrue(mv.graph, pol)
            self.assertTrue(any("budget guard" in r for r in mv.reasons))
        both = [_h("kernel+graph", 0.1, 0.1), _h("kernel+graph", 0.1, 0.1)]
        c = _ctl("fixed")
        mv = c.decide(_feat(c, first=False, hist=both), both)
        self.assertTrue(mv.stop)
        self.assertEqual(mv.label(), "stop")

    def test_auto_first_round_structural(self):
        c = _ctl("auto")
        mv = c.decide(_feat(c, hints=[_hint("fuse:a", "fuse", 9.0)]), [])                    # est 9% >= 8%
        self.assertEqual((mv.kernel, mv.graph, mv.deferred), (False, True, True))
        small = _prof([("relu", 10.0)] * 5 + [("conv2d", 50.0)])                             # 50% of the time in small unfused ops
        mv = c.decide(_feat(c, hints=[_hint("fuse:a", "fuse", 2.5)], profiles=[small]), [])
        self.assertEqual((mv.kernel, mv.graph), (False, True))
        self.assertIn("unfused small ops", mv.reasons[-1])
        mv = c.decide(_feat(c, hints=[_hint("fuse:a", "fuse", 2.5)]), [])                    # small gain, big ops: both
        self.assertEqual((mv.kernel, mv.graph, mv.deferred), (True, True, False))

    def test_auto_no_candidates_is_kernel_only_not_deferred(self):
        c = _ctl("auto")
        mv = c.decide(_feat(c), [])
        self.assertEqual((mv.kernel, mv.graph, mv.deferred), (True, False, False))
        self.assertTrue(any("no decidable graph candidate" in r for r in mv.reasons))

    def test_auto_graph_dominates_then_not_twice_in_a_row(self):
        c = _ctl("auto")
        hist = [_h("kernel+graph", 1.0, 9.0)]
        mv = c.decide(_feat(c, first=False, hints=self.CANDS, hist=hist), hist)
        self.assertEqual((mv.kernel, mv.graph, mv.deferred), (False, True, True))
        hist2 = hist + [_h("graph", 0.0, 9.0, worked=("graph",))]
        mv = c.decide(_feat(c, first=False, hints=self.CANDS, hist=hist2), hist2)
        self.assertTrue(mv.kernel and mv.graph)                                              # never two graph-only rounds in a row
        hist3 = [_h("kernel+graph", 6.0, 9.0)]
        mv = c.decide(_feat(c, first=False, hints=self.CANDS, hist=hist3), hist3)
        self.assertTrue(mv.kernel and mv.graph)

    def test_auto_stops_when_nothing_is_left(self):
        c = _ctl("auto")
        hist = [_h("kernel+graph", 0.1, 0.0), _h("kernel+graph", 0.1, 0.0)]
        mv = c.decide(_feat(c, first=False, hist=hist), hist)                                # kernel stopped by the guard, no graph candidate
        self.assertTrue(mv.stop)
        self.assertTrue(mv.reasons)

    def test_auto_region_loop_needs_something_to_act_on(self):
        c = _ctl("auto", rl=True)
        none = c.decide(_feat(c, hints=[_hint("fuse:a", "fuse", 2.5)]), [])
        self.assertFalse(none.region_loop)
        self.assertTrue(any(r.startswith("region_loop:") for r in none.reasons))
        pend = c.decide(_feat(c, hints=[_hint("region:win:c", "fuse_region", 3.0)]), [])      # region candidate decided this round
        self.assertTrue(pend.region_loop and pend.graph)
        live = c.decide(_feat(c, profiles=[_prof([("region__rnn", 90.0), ("relu", 1.0)])]), [])   # a live region op, kernel steps run
        self.assertTrue(live.region_loop and live.kernel)
        kernel_off = c.decide(_feat(c, hints=[_hint("region:win:c", "fuse_region", 3.0)], first=False, hist=[_h("kernel+graph", 0.1, 5.0)] * 2,
                                    profiles=[_prof([("region__rnn", 90.0)])]), [_h("kernel+graph", 0.1, 5.0)] * 2)
        self.assertTrue(kernel_off.region_loop and not kernel_off.kernel)                    # candidate decided via the graph move

    def test_move_round_trip_and_label(self):
        c = _ctl("auto")
        mv = c.decide(_feat(c, hints=self.CANDS), [])
        back = rc.Move.from_dict(json.loads(json.dumps(c.begin(mv))))
        self.assertEqual((back.kernel, back.graph, back.deferred, back.reasons), (mv.kernel, mv.graph, mv.deferred, mv.reasons))
        self.assertEqual(rc.Move(kernel=False, graph=False).label(), "stop")

    def test_budget_in_features(self):
        c = _ctl("auto", round_controller_budget_usd=20.0)
        f = _feat(c, spent=5.5, prev=200.0, cur=180.0)
        self.assertEqual((f["budget_left_usd"], f["prev_round_realized_pct"]), (14.5, 10.0))


class CommandLineTest(unittest.TestCase):
    def tearDown(self):
        rlm.set_round(None)

    def test_region_loop_flag_only_where_the_round_wants_it(self):
        cfg = _cfg(region_loop="on")
        self.assertEqual(rlm.kb_args(cfg), ["--region-loop", "on"])
        rlm.set_round(False)
        self.assertEqual(rlm.kb_args(cfg), [])
        rlm.set_round(True)
        self.assertEqual(rlm.kb_args(cfg), ["--region-loop", "on"])
        rlm.set_round(None)
        self.assertEqual(rlm.kb_args(_cfg(region_loop="off")), [])                           # never turned ON by the controller


class FlowTest(unittest.TestCase):
    """kb_graph_opt._run with every GPU step faked (a graph with windows + a real plan), the controller off / on."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="mb_rc_"))
        cls.ir = wf.ir_dir_from_graph(cls.tmp / "ir", wf.rnn_unrolled(24, fused=False))
        (cls.tmp / "kb" / "scripts").mkdir(parents=True)
        (cls.tmp / "kb" / "scripts" / "time_problems.py").write_text("")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.saved = {k: getattr(go, k) for k in ("profile_ir", "verify_ir", "detect_gpus", "make_plan", "remeasure", "KBRunner")}
        self.saved_mr = mr.model_runtime
        keys = ("KERNELBLASTER_GPU_FLOCK", rc.ENV, rc.ENV_MIN_GAIN, rc.ENV_BUDGET_USD, rc.ENV_SMALL_US, rlm.ENV, rlm.ENV_BUDGET,
                "MB_REGION_WINDOWS", "MB_REGION_MAX_MEMBERS")
        self.env = {k: os.environ.get(k) for k in keys}
        os.environ["KERNELBLASTER_GPU_FLOCK"] = "0"
        for k in keys[1:]:
            os.environ.pop(k, None)
        self.runs = []
        self.kernel_gain = 10.0
        outer = self

        class KB(FakeKB):
            def start(self, problems, out, **kw):
                outer.runs.append((str(out), self.command(problems, out)))
                return super().start(problems, out, **kw)

        def profile(irdir, work, *, hints_doc=None, **kw):
            from modelblaster.pipeline.kb_ingest import signature_sha
            from modelblaster.pipeline.kb_proposals import _all_specs, with_fused_specs
            from modelblaster.pipeline.kb_export import parse_signature
            from modelblaster.pipeline.kb_export import slot_id_of
            d, specs, prov = gh.apply_to_ir(Path(irdir), hints_doc or gh.new_doc(), Path(work) / "ir")
            p = _profile(d, irdir)
            sp = with_fused_specs(json.loads((Path(d) / "graph.json").read_text()), _all_specs())
            for o in p["ops"]:                                    # what make_plan ranks by (kb_plan.profile_ir adds these)
                if o["op"] not in sp:                             # views have no kernel
                    o["us"], o["entry"], o["signature_sha"] = None, "", ""
                    continue
                o["entry"], o["signature_sha"] = parse_signature(sp[o["op"]].signature)[0], signature_sha(sp[o["op"]].signature)
                o["slot"], o["seed"], o["slot_base"] = o["name"], "seed", slot_id_of({"op": o["op"]}, sp)
            return p

        def remeasure(problems, *, arch, reps, lease=None, **kw):
            return [[100.0 if json.loads((Path(d) / "manifest.json").read_text())["name"].endswith("__orig") else 70.0] * reps
                    for d, _ in problems]
        go.profile_ir = profile
        go.verify_ir = lambda ir, cache, **kw: {"ir": "synthetic", "passed": True, "partition_us": 5.0, "device": TITAN, "slots": {}}
        mr.model_runtime = lambda ir, **kw: {"device": TITAN, "passed": True, "model_e2e_us": 4.0}
        go.remeasure, go.KBRunner = remeasure, KB
        self.plan_calls = []

        def plan(profiles, **kw):                                  # a plan with a problem as long as there are unsent items
            self.plan_calls.append(1)
            return self.saved["make_plan"](profiles, **kw)
        go.make_plan = plan
        go.detect_gpus = lambda cfg: {"local": [{"ident": "GPU-titan", "index": 0, "name": "NVIDIA TITAN RTX"}], "remote": []}

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(go, k, v)
        mr.model_runtime = self.saved_mr
        rlm.set_round(None)
        for k, v in self.env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    WIN = dict(region_windows="auto", region_window_sizes="40", region_window_min_ops=2)

    def _run(self, name, rounds=2, **kw):
        cfg = go.Config(kb=self.tmp / "kb", arch="75", rounds=rounds, max_variants=1, remeasure=3, accept_gain=0.03, **kw)
        s = go.run([self.ir], self.tmp / name, cfg)
        w = self.tmp / name
        return s, gh.load(w / "graph_hints.json"), json.loads((w / "state.json").read_text()), \
            [json.loads(p.read_text()) for p in sorted(w.glob("r[0-9][0-9]/round.json"))]

    def test_off_equals_unset_equals_env_off(self):
        a = self._run("unset", **self.WIN)
        b = self._run("off", round_controller="off", **self.WIN)
        os.environ[rc.ENV] = "off"
        c = self._run("envoff", **self.WIN)
        def norm(r):                                            # timestamps / seconds are the only volatile parts: compare the rest
            return json.dumps({"hints": [(h["id"], h["status"], h.get("kind")) for h in r[1]["hints"]],
                               "summary_keys": sorted(r[0]), "accepted": r[0]["accepted"], "rejected": r[0]["rejected"],
                               "round_keys": [sorted(x) for x in r[3]], "state_round_keys": [sorted(x) for x in r[2]["rounds"]],
                               "decisions": [[(d["hint"], d["status"], d.get("gain")) for d in x["decisions"]] for x in r[3]],
                               "plans": [x["plan"] for x in r[3]], "waves": [x["waves"] for x in r[3]]}, sort_keys=True)
        self.assertEqual(norm(a), norm(b))
        self.assertEqual(norm(a), norm(c))
        for r in (a, b, c):
            self.assertNotIn("round_controller", r[0])
            self.assertTrue(all("controller" not in x for x in r[3]))
            self.assertTrue(all("controller" not in x for x in r[2]["rounds"]))
        self.assertTrue(all(not any("controller" in k for k in r[2]["rounds"][0]) for r in (a, b, c)))

    def test_fixed_records_each_round_and_decides_like_off(self):
        off = self._run("f_off", **self.WIN)
        fx = self._run("fixed", round_controller="fixed", **self.WIN)
        self.assertEqual([(h["id"], h["status"]) for h in off[1]["hints"]], [(h["id"], h["status"]) for h in fx[1]["hints"]])
        self.assertEqual(fx[0]["round_controller"]["policy"], "fixed")
        for x in fx[3]:
            self.assertEqual(x["controller"]["move"], "kernel+graph")
            self.assertTrue(x["controller"]["reasons"])
            self.assertIn("gain", x["controller"])
        self.assertEqual(len(fx[0]["round_controller"]["rounds"]), len(fx[3]))

    def test_graph_first_defers_the_kernel_plan_to_round_two(self):
        gf = self._run("gf", rounds=2, round_controller="graph-first", **self.WIN)
        r0, r1 = gf[3][0], gf[3][1]
        self.assertEqual(r0["controller"]["move"], "graph")
        self.assertEqual((r0["kb_runs"], r0["plan"]["problems"]), ([], []))                  # no kernel run in the graph round
        self.assertTrue(r0["decisions"])
        self.assertEqual(r1["controller"]["move"], "kernel+graph")
        self.assertTrue(r1["plan"]["problems"], r1["plan"])
        self.assertTrue(any(h["status"] == "accepted" for h in gf[1]["hints"]))

    def test_auto_prefers_graph_when_candidates_are_big_and_records_inputs(self):
        au = self._run("auto", round_controller="auto", **self.WIN)
        c0 = au[3][0]["controller"]
        self.assertEqual(c0["policy"], "auto")
        self.assertIn("candidates", c0["inputs"])
        self.assertIn("small_unfused", c0["inputs"])
        self.assertTrue(c0["reasons"])
        self.assertEqual(au[0]["round_controller"]["rounds"][0]["move"], c0["move"])

    def test_graph_off_round_keeps_hints_proposed(self):
        with mock.patch.object(rc.RoundController, "decide", lambda self, f, h: rc.Move(kernel=True, graph=False, reasons=["test"], inputs=f)):
            r = self._run("kernel_only", round_controller="fixed", rounds=1, **self.WIN)
        self.assertEqual(r[3][0]["decisions"], [])
        self.assertTrue(any(h["status"] == "proposed" for h in r[1]["hints"]))

    def test_stop_move_ends_the_loop_as_fixpoint(self):
        with mock.patch.object(rc.RoundController, "decide", lambda self, f, h: rc.Move(kernel=False, graph=False, stop=True, reasons=["test stop"], inputs=f)):
            r = self._run("stop", round_controller="auto", rounds=3, **self.WIN)
        self.assertEqual(len(r[3]), 1)
        self.assertEqual((r[3][0]["controller"]["move"], r[2]["fixpoint"]), ("stop", 0))
        self.assertEqual(r[0]["round_controller"]["stop_reason"], "test stop")
        self.assertEqual(r[3][0]["kb_runs"], [])

    def test_usd_budget_stops_before_the_profile(self):
        w = "budget"
        orig = FakeKB.collect

        def collect(kb, problems, out, proc):
            res = orig(kb, problems, out, proc)
            kb.usd += 4.0
            return res
        with mock.patch.object(FakeKB, "collect", collect):
            r = self._run(w, round_controller="fixed", rounds=4, round_controller_budget_usd=3.0, **self.WIN)
        last = r[2]["rounds"][-1]["controller"]
        self.assertEqual(last["move"], "stop")
        self.assertIn("budget", r[0]["round_controller"]["stop_reason"])
        self.assertLess(len(r[2]["rounds"]), 4)

    def test_region_loop_switch_reaches_the_command_line(self):
        r = self._run("rl", rounds=2, round_controller="fixed", region_loop="on", **self.WIN)
        self.assertTrue(self.runs)
        self.assertTrue(all("--region-loop" in c for _, c in self.runs))
        self.runs.clear()
        with mock.patch.object(rc.RoundController, "decide", lambda self, f, h: rc.Move(kernel=True, graph=True, region_loop=False,
                                                                                         reasons=["test"], inputs=f)):
            self._run("rl_off", rounds=2, round_controller="fixed", region_loop="on", **self.WIN)
        self.assertTrue(self.runs)
        self.assertTrue(all("--region-loop" not in c for _, c in self.runs))

    def test_resume_keeps_the_move(self):
        self._run("resume", rounds=1, round_controller="graph-first", **self.WIN)
        s = json.loads((self.tmp / "resume" / "r00" / "progress.json").read_text())
        self.assertIn("controller", s["phases"])
        self.assertEqual(s["phases"]["controller"]["value"]["move"], "graph")

    def test_refused_with_graph_search_flags(self):
        with self.assertRaises(ValueError):
            self._run("gs", round_controller="fixed", graph_search="interleaved")


if __name__ == "__main__":
    unittest.main()
