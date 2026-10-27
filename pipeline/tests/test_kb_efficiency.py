"""E7 cost-efficiency rules in the graph optimiser (pipeline/kb_efficiency.py, X-129).

No GPU: presets and per-rule overrides, resolution against a work dir (fresh / resumed / legacy / explicit),
round-key stability for a real pre-X-129 work dir (its stored key and hash, from the Level 3 sweep's 1_MLP),
late min-share selection in rounds >= 1 only, the KernelBlaster command (patience for plan runs, never for hint
decisions), the telemetry roll-ups and the generated CLI flags.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from modelblaster.pipeline import kb_efficiency as E  # noqa: E402
from modelblaster.pipeline import kb_resume as R  # noqa: E402
from modelblaster.pipeline.kb_graph_opt import Config, KBRunner  # noqa: E402
from modelblaster.pipeline.kb_plan import select  # noqa: E402

KB = ROOT / "third_party" / "KernelBlaster"

# r00/progress.json "key" / "key_hash" of l3_farm/results/1_MLP/work (Level 3 sweep, before X-129)
LEGACY_KEY = {"model": "azure/anthropic/claude-opus-5-5", "reflect_model": "azure/anthropic/claude-opus-5-5",
              "arch": "89", "rollouts": 4, "steps": 4, "variant_steps": 2, "max_fix": 2, "coverage": 0.8, "top_k": 6,
              "min_share": 0.01, "min_gain_pct": 0.5, "max_variants": 4, "remeasure": 5, "accept_gain": 0.03,
              "profile_runs": 3, "seed": 0, "plan_mode": "all", "irs": ["/tmp/krisd/farm/1_MLP/irs/1_MLP"]}
LEGACY_HASH = "14fd9d1617dd2819"


def legacy_cfg(**kw) -> Config:
    m = "azure/anthropic/claude-opus-5-5"
    return Config(kb=KB, model=m, reflect_model=m, arch="89", rollouts=4, steps=4, variant_steps=2, min_gain_pct=0.5,
                  max_variants=4, plan_mode="all", **kw)


def key_for(cfg: Config) -> dict:
    """kb_graph_opt._run's round key (global knowledge / cuda_learnings off add nothing) for the sweep's IR path."""
    key = R.round_key(cfg, [Path("/x")])
    key.update(E.key_fields(cfg))
    key["irs"] = list(LEGACY_KEY["irs"])
    return key


class PresetTest(unittest.TestCase):
    def test_presets(self):
        self.assertEqual(E.from_flags(None, None, None),
                         {"rules": "on", "stop_patience": 8, "plan_min_share_late": 0.005})
        self.assertEqual(E.from_flags("off", None, None),
                         {"rules": "off", "stop_patience": 0, "plan_min_share_late": 0.0})
        self.assertEqual(E.from_flags("safe", None, None),
                         {"rules": "safe", "stop_patience": 0, "plan_min_share_late": 0.01})
        with self.assertRaises(ValueError):
            E.from_flags("aggressive", None, None)

    def test_explicit_rules_override_the_preset(self):
        self.assertEqual(E.from_flags("on", 0, None)["stop_patience"], 0)
        self.assertEqual(E.from_flags("on", None, 0.0)["plan_min_share_late"], 0.0)
        self.assertEqual(E.from_flags("off", 2, None), {"rules": "off", "stop_patience": 2, "plan_min_share_late": 0.0})
        self.assertEqual(E.from_flags("safe", 6, 0.02)["stop_patience"], 6)
        self.assertEqual(E.from_flags(None, None, 0.01)["stop_patience"], 8)    # 'on' with a tuned late share


class ResolveTest(unittest.TestCase):
    def setUp(self):
        self.work = Path(tempfile.mkdtemp(prefix="mb_eff_"))

    def test_fresh_work_dir_gets_the_default(self):
        eff = E.resolve(Config(kb=KB), self.work, {"sent": [], "rounds": []})
        self.assertEqual((eff["source"], eff["stop_patience"], eff["plan_min_share_late"]), ("default", 8, 0.005))

    def test_legacy_state_resumes_with_rules_off(self):
        state = {"sent": ["k"], "rounds": [{"round": 0, "key": LEGACY_HASH}]}
        eff = E.resolve(Config(kb=KB), self.work, state)
        self.assertEqual((eff["source"], eff["rules"], E.active(eff)), ("legacy", "off", False))

    def test_legacy_interrupted_round_zero(self):
        # killed inside round 0 before state.json had a round: only r00/progress.json (legacy key) exists
        R.RoundProgress(self.work / "r00", dict(LEGACY_KEY)).mark("profile", [])
        eff = E.resolve(Config(kb=KB), self.work, {"sent": [], "rounds": []})
        self.assertEqual(eff["source"], "legacy")

    def test_recorded_rules_win_without_flags(self):
        state = {"sent": [], "rounds": [], "efficiency": {"rules": "safe", "stop_patience": 0,
                                                          "plan_min_share_late": 0.01}}
        eff = E.resolve(Config(kb=KB), self.work, state)
        self.assertEqual((eff["source"], eff["rules"], eff["plan_min_share_late"]), ("resumed", "safe", 0.01))
        # recorded only in an interrupted round's key (state.json not yet written)
        R.RoundProgress(self.work / "r00", {**LEGACY_KEY, "efficiency": {"rules": "on", "stop_patience": 8,
                                                                          "plan_min_share_late": 0.005}})
        eff = E.resolve(Config(kb=KB), self.work, {"sent": [], "rounds": []})
        self.assertEqual((eff["source"], eff["stop_patience"]), ("resumed", 8))

    def test_explicit_flags_win_and_say_so(self):
        state = {"sent": [], "rounds": [], "efficiency": {"rules": "on", "stop_patience": 8,
                                                          "plan_min_share_late": 0.005}}
        eff = E.resolve(Config(kb=KB, efficiency_rules="off"), self.work, state)
        self.assertEqual((eff["source"], E.active(eff)), ("explicit", False))
        self.assertIn("new configuration", eff["note"])
        eff = E.resolve(Config(kb=KB, stop_patience=4), self.work, {"sent": ["k"], "rounds": [{}]})
        self.assertEqual((eff["stop_patience"], eff["plan_min_share_late"]), (4, 0.005))
        self.assertIn("legacy work dir", eff["note"])


class RoundKeyTest(unittest.TestCase):
    def test_legacy_work_dir_key_is_unchanged(self):
        state = {"sent": ["k"], "rounds": [{"round": 0, "key": LEGACY_HASH}, {"round": 1, "key": LEGACY_HASH}]}
        cfg = E.apply(legacy_cfg(), E.resolve(legacy_cfg(), Path(tempfile.mkdtemp()), state))
        key = key_for(cfg)
        self.assertEqual(key, LEGACY_KEY)
        self.assertEqual(R.key_hash(key), LEGACY_HASH)
        # the same with an explicit --efficiency-rules off (ablations): rules off = the legacy key
        cfg_off = E.apply(legacy_cfg(efficiency_rules="off"), E.resolve(legacy_cfg(efficiency_rules="off"),
                                                                        Path(tempfile.mkdtemp()), state))
        self.assertEqual(R.key_hash(key_for(cfg_off)), LEGACY_HASH)

    def test_legacy_interrupted_round_resumes_instead_of_moving_aside(self):
        work = Path(tempfile.mkdtemp(prefix="mb_eff_"))
        p = R.RoundProgress(work / "r00", dict(LEGACY_KEY))
        p.mark("profile", [{"ir": "x"}])
        cfg = E.apply(legacy_cfg(), E.resolve(legacy_cfg(), work, {"sent": [], "rounds": []}))
        q = R.RoundProgress(work / "r00", key_for(cfg))
        self.assertTrue(q.resumed and q.done("profile") and q.moved is None)

    def test_rules_on_record_their_parameters_in_the_key(self):
        cfg = E.apply(legacy_cfg(), E.resolve(legacy_cfg(), Path(tempfile.mkdtemp()), {"sent": [], "rounds": []}))
        key = key_for(cfg)
        self.assertEqual(key["efficiency"], {"rules": "on", "stop_patience": 8, "plan_min_share_late": 0.005})
        self.assertNotEqual(R.key_hash(key), LEGACY_HASH)
        safe = E.apply(legacy_cfg(efficiency_rules="safe"), E.from_flags("safe", None, None))
        self.assertEqual(key_for(safe)["efficiency"]["plan_min_share_late"], 0.01)
        self.assertNotEqual(R.key_hash(key_for(safe)), R.key_hash(key))
        self.assertEqual(R.key_hash(key_for(cfg)), R.key_hash(key))           # stable


class LateMinShareTest(unittest.TestCase):
    def test_round_zero_keeps_the_base_share(self):
        self.assertEqual(E.round_min_share(0.002, 0.005, 0), 0.002)
        self.assertEqual(E.round_min_share(0.002, 0.005, 1), 0.005)
        self.assertEqual(E.round_min_share(0.002, 0.005, 2), 0.005)
        self.assertEqual(E.round_min_share(0.01, 0.005, 1), 0.01)            # never below the base
        self.assertEqual(E.round_min_share(0.002, 0.0, 1), 0.002)             # off
        self.assertEqual(E.round_min_share(0.002, None, 3), 0.002)

    def _items(self):
        us = [600.0, 300.0, 60.0, 30.0, 4.0, 3.0, 2.0, 1.0]                   # shares 0.6 ... 0.001
        return [{"key": f"k{i}", "op": f"op{i}", "total_us": u} for i, u in enumerate(us)]

    def test_selection_in_rounds(self):
        tot = 1000.0
        for r, want_out in ((0, {"k7"}), (1, {"k4", "k5", "k6", "k7"}), (2, {"k4", "k5", "k6", "k7"})):
            plan = select(self._items(), coverage=2.0, top_k=None,
                          min_share=E.round_min_share(0.002, 0.005, r))
            out = {it["key"] for it in plan["items"] if not it["selected"]}
            self.assertEqual(out, want_out, r)
            skipped = E.late_skipped(plan, 0.002)
            self.assertEqual({x["key"] for x in skipped}, want_out - {"k7"})  # k7 (0.1%) is the base rule's
            for x in skipped:
                self.assertTrue(0.002 <= x["share"] < 0.005)
                self.assertAlmostEqual(x["total_us"] / tot, x["share"])


class KBCommandTest(unittest.TestCase):
    def test_kb_checkout_supports_patience(self):
        self.assertTrue(E.kb_supports_patience(KB))
        self.assertFalse(E.kb_supports_patience(Path(tempfile.mkdtemp())))

    def _cmd(self, cfg, **kw):
        c = KBRunner(cfg, Path(tempfile.mkdtemp()) / "knowledge").command([Path("/p")], Path(tempfile.mkdtemp()), **kw)
        i = c.index("--stop-patience")
        return c[i + 1]

    def test_plan_runs_get_patience_hint_decisions_do_not(self):
        on = E.apply(Config(kb=KB), E.from_flags("on", None, None))
        self.assertEqual(self._cmd(on), "8")
        self.assertEqual(self._cmd(on, steps=2, stop_patience=0), "0")      # evaluate_hints
        off = E.apply(Config(kb=KB), E.from_flags("off", None, None))
        self.assertEqual(self._cmd(off), "0")                                 # explicit: KB's own default is on
        custom = E.apply(Config(kb=KB), E.from_flags("safe", 3, None))
        self.assertEqual(self._cmd(custom), "3")


class TelemetryTest(unittest.TestCase):
    def test_rollups(self):
        res = [{"problem": "a", "efficiency": {"stop_patience": 8, "stopped": "patience", "trigger_commit": 9,
                                               "steps_committed": 11, "steps_planned": 16, "steps_after_trigger": 2}},
               {"problem": "b", "efficiency": {"stop_patience": 8, "stopped": None, "steps_committed": 16,
                                               "steps_planned": 16}},
               {"problem": "c"}]
        k = E.kb_patience(res)
        self.assertEqual((k["n_patience_stopped"], k["steps_committed"], k["steps_planned"], k["steps_not_run"]),
                         (1, 27, 32, 5))
        self.assertEqual(k["patience_stopped"][0]["at_commit"], 9)
        rounds = [{"round": 0, "efficiency": {"min_share": 0.002, "late_skipped": [], **k}},
                  {"round": 1, "efficiency": {"min_share": 0.005, "late_skipped": [{"share": 0.003}, {"share": 0.004}],
                                              "n_patience_stopped": 0, "steps_committed": 4, "steps_planned": 4}},
                  {"round": 2}]
        s = E.summarize({"rules": "on", "stop_patience": 8, "plan_min_share_late": 0.005}, rounds)
        self.assertEqual(s["totals"], {"late_skipped": 2, "patience_stopped": 1, "steps_committed": 31,
                                       "steps_planned": 36, "steps_not_run": 5})
        self.assertEqual(s["rounds"][1]["late_skipped_share"], 0.007)
        self.assertIn("patience 8", E.describe({**s, "source": "default"}))


class CliTest(unittest.TestCase):
    def test_generated_flags(self):
        env = {**os.environ, "PYTHONPATH": str(ROOT.parent)}
        out = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.kb_graph_opt", "--help"], env=env,
                             capture_output=True, text=True, timeout=120).stdout
        for flag in ("--efficiency-rules", "--stop-patience", "--plan-min-share-late"):
            self.assertIn(flag, out)


if __name__ == "__main__":
    unittest.main()
