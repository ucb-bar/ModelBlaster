"""Search fix 3: region:all / survival (FEATURE_TRACKER H.9 item 2 / 3). All of it needs ``--graph-survival`` on (auto
with interleaved); with it off nothing changes. Region hints leave the survival cut (``MB_SURVIVAL_EXCLUDE_REGION``,
default 1), a region's decision spend is capped (``MB_REGION_DECISION_STEPS``, default 2; 0 = no cap), high-gain hints
get more steps (``MB_HIGHGAIN_PCT`` default 10, ``MB_HIGHGAIN_MULT`` default 2; pct 0 = off), the fusion decisions run
before the steps of the slots they would absorb (``MB_FUSION_FIRST``, default 1). GPU-, torch- and LLM-free (stubs)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from modelblaster.pipeline import graph_search as gsm  # noqa: E402
from modelblaster.pipeline.tests.test_graph_search import _Proc, _Prog, _cfg, _problem  # noqa: E402


def _env(**kw):
    """Set env vars for one test."""
    return mock.patch.dict(os.environ, {k: v for k, v in kw.items() if v is not None}, clear=False)


def _gs(root: Path, **kw):
    return gsm.GraphSearch(_cfg(graph_search="interleaved", graph_survival=kw.pop("survival", "on"), **kw), root, [])


def _region(est=0.3):
    return {"id": "region:all", "kind": "fuse_region", "pattern": "all", "target": "cuda", "status": "proposed",
            "source": [{"by": "mb-region-slots", "ir": "m", "est_gain_pct": est}]}


class SurvivalRegionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _budgets(self, gs):
        doc = {"hints": [_region()]}
        d = _problem(self.root / "p", "conv_r00", ["conv1"])
        plan = {"problems": [{"ir": "m", "op_names": ["conv1"]}]}
        steps = gs.plan_budgets([d], plan, doc, {"m": {}}, {}, lambda h, g, s: ["conv1", "bn1"], steps=4,
                                stop_patience=8)
        return steps["conv_r00"]

    def test_region_does_not_cut_round0_steps(self):
        with _env(MB_SURVIVAL_EXCLUDE_REGION="1"):
            b = self._budgets(_gs(self.root))
        self.assertEqual((b["steps"], b["by"]), (4, []))

    def test_old_behaviour_cut_steps_for_a_pending_region(self):
        with _env(MB_SURVIVAL_EXCLUDE_REGION="0"):
            b = self._budgets(_gs(self.root))
        self.assertEqual((b["steps"], b["by"]), (2, ["region:all"]))        # 45_UNetSoftmax: 4 -> 2

    def test_a_real_fusion_still_counts(self):
        gs = _gs(self.root)
        doc = {"hints": [_region(), {"id": "fuse:x", "kind": "fuse", "status": "proposed", "target": "cuda",
                                     "chains": [["conv0", "conv1"]], "source": [{"est_gain_pct": 5.0}]}]}
        with _env(MB_SURVIVAL_EXCLUDE_REGION="1"):
            pend = gs.pending(doc, {"m": {}}, {}, lambda h, g, s: ["conv0", "conv1"])
        self.assertEqual([x["hint"] for x in pend["m"]], ["fuse:x"])

    def test_survival_off_is_unchanged(self):
        gs = _gs(self.root, survival="off")
        doc = {"hints": [_region(est=20.0)]}
        with _env(MB_SURVIVAL_EXCLUDE_REGION="1", MB_REGION_DECISION_STEPS="2", MB_HIGHGAIN_PCT="10"):
            pend = gs.pending(doc, {"m": {}}, {}, lambda h, g, s: ["conv1"])
            self.assertEqual([x["hint"] for x in pend["m"]], ["region:all"])    # not excluded: no survival cut anyway
            self.assertEqual(gs.region_cap(_region()), 0)
            self.assertEqual(gs.high_gain_mult({"id": "fuse:y", "kind": "fuse", "source": [{"est_gain_pct": 50}]}), 1)
            self.assertEqual(gs.cap_for(_region()), gs.cap_steps)


class HintBudgetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.gs = _gs(Path(self.tmp.name), variant_steps=8, graph_cap_steps=8)

    def tearDown(self):
        self.tmp.cleanup()

    def test_region_decision_spend_is_capped_at_two_steps(self):
        with _env(MB_REGION_DECISION_STEPS="2"):
            b = self.gs.hint_budget(_region())
            self.assertLessEqual(b["steps"], 2)
            h = dict(_region(), gs_deferred={"rung": 0, "steps_used": 2})
            self.assertEqual(self.gs.hint_budget(h)["steps"], 0)              # used up: no later rung
            self.assertEqual(self.gs.cap_for(_region()), 2)
        with _env(MB_REGION_DECISION_STEPS="0"):                              # cap off
            self.assertGreater(self.gs.hint_budget(_region())["steps"], 2)

    def test_high_gain_hint_gets_more_steps_and_a_higher_cap(self):
        big = {"id": "fuse:big", "kind": "fuse", "source": [{"est_gain_pct": 15.0}]}
        small = {"id": "fuse:small", "kind": "fuse", "source": [{"est_gain_pct": 3.0}]}
        with _env(MB_HIGHGAIN_PCT="10", MB_HIGHGAIN_MULT="2"):
            self.assertGreater(self.gs.hint_budget(big)["steps"], self.gs.hint_budget(small)["steps"])
            self.assertEqual(self.gs.cap_for(big), 2 * self.gs.cap_for(small))
            deferred = dict(big, gs_deferred={"rung": 1, "steps_used": 8})    # past the default cap of 8: still room
            self.assertGreater(self.gs.hint_budget(deferred)["steps"], 0)
            self.assertEqual(self.gs.hint_budget(dict(small, gs_deferred={"rung": 1, "steps_used": 8}))["steps"], 0)
        with _env(MB_HIGHGAIN_PCT="0"):                                       # off: the default budget
            self.assertEqual(self.gs.hint_budget(big)["steps"], self.gs.hint_budget(small)["steps"])


class FusionFirstTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.gs = _gs(self.root)
        self.starts = []

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, accept: bool, fusion_first=True, kind="fuse"):
        t0 = time.time()
        jobs = []
        for hid in ("fuse:A",):
            o = _problem(self.root / "v", "A__orig", [])
            v = _problem(self.root / "v", "A__variant", [])
            jobs.append({"hint": {"id": hid, "kind": kind}, "orig": o, "variant": v, "ev": {"ir": "m"}, "names": [],
                         "gs": {"rung": 0, "steps": 1, "used": 0}})
        plan = [_problem(self.root / "p", "member_r00", ["bn1"]),            # absorbed if A is accepted
                _problem(self.root / "p", "free_r00", ["fc"])]               # not a member
        results = {"A__orig": t0, "A__variant": t0}

        def start(probs, append=False):
            self.starts.append(([Path(p).name for p in probs], append))
            return _Proc(time.time() + (0.02 if append else 0.1))

        def read(d):
            name = json.loads((Path(d) / "manifest.json").read_text())["name"]
            return {"final_ok": True} if name in results else None
        st = "accepted" if accept else "rejected"
        eng = gsm.Interleaved(self.gs, run_dir=self.root / "run", prog=_Prog(), start=start, read=read,
                              measure=lambda j, ro, rv, attempts=None: [[1.0], [0.5]],
                              finish=lambda j, ro, rv, tm: {"hint": "fuse:A", "status": st, "hint_status": st},
                              collect=lambda probs, proc: [{"problem": Path(p).name, "final_ok": True,
                                                            "baseline_partition_us": 1.0} for p in probs],
                              scope_after=lambda j: {"bn1"}, poll_s=0.01, fusion_first=fusion_first)
        return eng.run(plan, ["m", "m"], jobs)

    def test_old_behaviour_starts_the_member_at_once(self):
        out = self._run(accept=True, fusion_first=False)
        self.assertEqual(self.starts, [(["A__orig", "A__variant", "member_r00", "free_r00"], False)])
        self.assertEqual(out["stopped"], ["member_r00"])                      # started, then stopped: wasted steps

    def test_accepted_fusion_means_the_member_never_starts(self):
        out = self._run(accept=True)
        self.assertEqual(self.starts, [(["A__orig", "A__variant", "free_r00"], False)])   # one invocation only
        names = [r["problem"] for r in out["plan_results"]]
        self.assertEqual(names, ["member_r00", "free_r00"])                   # same order as the plan
        member = out["plan_results"][0]
        self.assertTrue(member["efficiency"]["external_stop"])                # consumed, not measured-and-failed

    def test_rejected_fusion_runs_the_member_afterwards(self):
        out = self._run(accept=False)
        self.assertEqual(self.starts, [(["A__orig", "A__variant", "free_r00"], False), (["member_r00"], True)])
        self.assertEqual([r["problem"] for r in out["plan_results"]], ["member_r00", "free_r00"])
        self.assertTrue(out["plan_results"][0]["final_ok"])

    def test_a_region_hint_does_not_hold_members(self):
        self._run(accept=False, kind="fuse_region")
        self.assertEqual(len(self.starts), 1)


if __name__ == "__main__":
    unittest.main()
