"""Search fix 1: sent-on-success + requeue (FEATURE_TRACKER H.9 item 1). A plan item is marked ``sent`` only when its
problem produced a measured outcome; lost / infra / never-evaluated items stay eligible, a bounded number of times
(``MB_SENT_ON_SUCCESS``, default 1; ``MB_SENT_RETRY_MAX``, default 2). GPU-, torch- and LLM-free (stubs only)."""
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

from modelblaster.pipeline import search_fixes as sfx  # noqa: E402

SCP = ("baseline failed (run): [infra] copy of the host binary / shared objects ... scp: Connection closed")


def _env(**kw):
    """Set env vars for one test."""
    return mock.patch.dict(os.environ, {k: v for k, v in kw.items() if v is not None}, clear=False)


def _plan(round_no=0, n=3):
    """n selected per-item problems ``p{i}`` (key ``k{i}``) + one item that was not selected."""
    items = [{"key": f"k{i}", "selected": True} for i in range(n)] + [{"key": "skip", "selected": False}]
    probs = [{"id": f"p{i}", "ir": "m", "keys": [f"k{i}"]} for i in range(n)]
    return {"items": items, "problems": probs}


def _res(i, round_no=0, **kw):
    return {"problem": f"p{i}__r{round_no:02d}", **kw}


OK = dict(final_ok=True, baseline_partition_us=10.0, final_partition_us=5.0)


class SentOnSuccessTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fx = sfx.Fixes(SimpleNamespace(), Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def _results(self):
        return [_res(0, **OK),                                                         # measured, accepted
                _res(1, final_ok=False, final_infra=True, final_error=SCP, baseline_partition_us=None),   # lost
                ]                                                                       # p2: never evaluated

    def test_lost_and_unevaluated_items_stay_eligible(self):
        with _env(MB_SENT_ON_SUCCESS="1"):
            st = {"sent": []}
            self.fx.mark_sent(_plan(), self._results(), st, 0)
        self.assertEqual(st["sent"], ["k0"])
        self.assertEqual(sorted(st["sent_retry"]), ["k1", "k2"])

    def test_old_rule_marks_everything_sent(self):
        with _env(MB_SENT_ON_SUCCESS="0"):
            st = {"sent": []}
            self.fx.mark_sent(_plan(), self._results(), st, 0)
        self.assertEqual(st["sent"], ["k0", "k1", "k2"])                    # the lost items are blocked later

    def test_measured_kernel_failure_is_sent_but_infra_text_without_tag_is_not(self):
        rs = [_res(0, final_ok=False, final_error="illegal memory access", baseline_partition_us=10.0),
              _res(1, final_ok=False, final_error=SCP, baseline_partition_us=10.0),            # tag missing
              _res(2, final_ok=False, final_error="no result.json", baseline_partition_us=None)]
        with _env(MB_SENT_ON_SUCCESS="1"):
            st = {"sent": []}
            self.fx.mark_sent(_plan(), rs, st, 0)
        self.assertEqual(st["sent"], ["k0"])

    def test_absorbed_run_counts_as_consumed(self):
        rs = [_res(0, final_ok=False, efficiency={"external_stop": {"by": "graph_search"}})]
        with _env(MB_SENT_ON_SUCCESS="1"):
            st = {"sent": []}
            self.fx.mark_sent(_plan(n=1), rs, st, 0)
        self.assertEqual(st["sent"], ["k0"])

    def test_retry_is_bounded_and_resume_safe(self):
        lost = [_res(0, final_ok=False, final_infra=True)]
        with _env(MB_SENT_ON_SUCCESS="1", MB_SENT_RETRY_MAX="2"):
            st = {"sent": []}
            for r in (0, 0, 1):                                               # round 0 resumed once: counted once
                self.fx.mark_sent(_plan(n=1), [_res(0, r, final_ok=False, final_infra=True)], st, r)
            self.assertEqual(st["sent"], [])                                  # 2 re-queues used
            self.assertEqual(st["sent_retry"]["k0"]["n"], 2)
            self.fx.mark_sent(_plan(n=1), [_res(0, 2, final_ok=False, final_infra=True)], st, 2)
            self.assertEqual(st["sent"], ["k0"])                              # gave up: a failing item ends
        self.assertEqual(st["sent_retry"]["k0"]["gave_up"], 2)
        del lost

    def test_next_round_plan_is_not_empty_while_an_item_is_held(self):
        """The failure the flag fixes: every item 'sent before' -> empty plan -> fixpoint."""
        for flag, empty in (("0", True), ("1", False)):
            with _env(MB_SENT_ON_SUCCESS=flag):
                st = {"sent": []}
                self.fx.mark_sent(_plan(n=1), [_res(0, final_ok=False, final_infra=True)], st, 0)
                eligible = [it["key"] for it in _plan(n=1)["items"] if it["key"] not in set(st["sent"])]
                self.assertEqual("k0" not in eligible, empty, flag)

    def test_round_record_reports_held_items(self):
        with _env(MB_SENT_ON_SUCCESS="1"):
            self.fx.mark_sent(_plan(), self._results(), {"sent": []}, 0)
        self.assertEqual(sorted(self.fx.round_record()["sent_held"]["0"]), ["k1", "k2"])


if __name__ == "__main__":
    unittest.main()
