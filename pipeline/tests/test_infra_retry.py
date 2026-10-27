"""Search fix 2: infra is not a kernel failure (FEATURE_TRACKER H.9 items 1 / 3 correction). Infra signatures without
the ``final_infra`` tag leave a hint undecided (``MB_INFRA_AWARE``, default 1); one automatic re-evaluation, then
``infra_unresolved``: neither accepted nor rejected, never a candidate again (``MB_INFRA_RETRY``, default 1; 0 = old).
GPU-, torch- and LLM-free (stubs only)."""
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

from modelblaster.pipeline import graph_hints as gh  # noqa: E402
from modelblaster.pipeline import kb_graph_opt as go  # noqa: E402
from modelblaster.pipeline import kb_proposals as kp  # noqa: E402

SCP = ("baseline failed (run): [infra] copy of the host binary / shared objects ... scp: Connection closed")


def _env(**kw):
    """Set env vars for one test."""
    return mock.patch.dict(os.environ, {k: v for k, v in kw.items() if v is not None}, clear=False)


def _job(tmp: Path, hint: dict):
    for n in ("orig", "variant"):
        (tmp / n).mkdir(exist_ok=True)
        (tmp / n / "manifest.json").write_text(json.dumps({"name": n, "slots": []}))
    return {"hint": hint, "ev": {"ir": "m"}, "names": ["a"], "stem": "s", "prov": {"applied": []},
            "orig": str(tmp / "orig"), "variant": str(tmp / "variant"), "specs": {}}


class InfraTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.h = gh.fuse_hint("fused__conv2d_f16__relu_f16", [["conv", "relu"]],
                              {"by": "t", "ir": "m", "est_gain_pct": 5.0})
        self.doc = gh.new_doc()
        gh.merge(self.doc, [self.h])
        self.cfg = SimpleNamespace(remeasure=0, accept_gain=0.03)

    def tearDown(self):
        self.tmp.cleanup()

    def _finish(self, rv):
        ro = {"final_ok": True, "final_partition_us": 100.0, "llm_calls": 1}
        return go.finish_hint(_job(self.t, self.doc["hints"][0]), ro, rv, None, cache=None, doc=self.doc, cfg=self.cfg)

    def test_signatures(self):
        for text in (SCP, "NoDevice: no usable GPU", "relay returned 503 Service Unavailable (queue_full)",
                     "[infra] x", "no result.json when the KernelBlaster process exited (rc=77)"):
            self.assertTrue(kp.is_infra_result({"final_ok": False, "final_error": text}), text)
        for text in ("illegal memory access", "wrong answer: max abs err 3.1", "compile error: expected ';'", None):
            self.assertFalse(kp.is_infra_result({"final_ok": False, "final_error": text}), text)
        self.assertTrue(kp.is_infra_result({"final_ok": False, "final_infra": True}))
        self.assertFalse(kp.is_infra_result({"final_ok": True, "final_infra": True}))
        with _env(MB_INFRA_AWARE="0"):
            self.assertFalse(kp.is_infra_result({"final_ok": False, "final_error": SCP}))     # the old rule

    def test_untagged_scp_failure_is_not_a_rejection(self):
        """43's sdpa: the variant's baseline died on `scp: Connection closed`, the hint was rejected 'measured False'."""
        with _env(MB_INFRA_AWARE="1", MB_INFRA_RETRY="1"):
            r = self._finish({"final_ok": False, "final_error": SCP})
        self.assertEqual((r["status"], r["hint_status"]), ("undecided", "proposed"))
        self.assertNotIn(self.h["id"], gh.decided(self.doc))

    def test_untagged_scp_failure_old_rule_rejects(self):
        with _env(MB_INFRA_AWARE="0"):
            r = self._finish({"final_ok": False, "final_error": SCP})
        self.assertEqual(r["status"], "rejected")

    def test_kernel_failure_is_still_a_rejection(self):
        with _env(MB_INFRA_AWARE="1"):
            r = self._finish({"final_ok": False, "final_error": "illegal memory access", "final_partition_us": None})
        self.assertEqual((r["status"], r["hint_status"]), ("rejected", "rejected"))

    def test_one_reevaluation_then_infra_unresolved(self):
        lost = {"final_ok": False, "final_infra": True, "final_error": "no result.json when the KernelBlaster "
                                                                          "process exited (rc=77)"}
        with _env(MB_INFRA_RETRY="1"):
            r1 = self._finish(lost)
            self.assertEqual((r1["status"], r1["hint_status"]), ("undecided", "proposed"))   # candidate again
            r2 = self._finish(lost)
        self.assertEqual((r2["status"], r2["hint_status"]), (go.INFRA_UNRESOLVED, go.INFRA_UNRESOLVED))
        self.assertEqual(r2["infra_failures"], 2)
        self.assertNotIn(self.h["id"], gh.decided(self.doc))                 # neither accepted nor rejected
        self.assertNotEqual(self.doc["hints"][0]["status"], "proposed")      # and no candidate filter picks it up

    def test_retry_zero_is_the_old_unbounded_undecided(self):
        lost = {"final_ok": False, "final_infra": True}
        with _env(MB_INFRA_RETRY="0"):
            for _ in range(4):
                self.assertEqual(self._finish(lost)["status"], "undecided")

    def test_a_later_measured_decision_still_votes(self):
        lost = {"final_ok": False, "final_infra": True}
        ok = {"final_ok": True, "final_partition_us": 50.0, "llm_calls": 1}
        with _env(MB_INFRA_RETRY="1"), mock.patch.object(go, "_ingest_slots", return_value=[]):
            self._finish(lost)
            r = self._finish(ok)
        self.assertEqual(r["status"], "accepted")


if __name__ == "__main__":
    unittest.main()
