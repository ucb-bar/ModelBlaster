"""A hint's re-measurement that fails: why it failed reaches the result file, the hint's
evidence and the trace; an infrastructure failure is retried on a fresh lease and, still
failing, leaves the hint undecided (never a rejection on merit); a candidate's own failure
rejects it with the error text.

The failure this guards against: the flow's KernelBlaster run and ModelBlaster's
re-measurement of a hint shared a remote dir, the re-measurement's copy of the
content-named persistent host binary over the one the KernelBlaster run was executing
failed (``scp: dest open ...: Failure``), the GPU was quarantined and a hint KernelBlaster
had measured 33% faster was rejected as "no passing run on a side".
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline import graph_hints as gh  # noqa: E402
from modelblaster.pipeline import kb_graph_opt as go  # noqa: E402
from modelblaster.pipeline import kb_proposals as kp  # noqa: E402

SCP_ERR = ('[infra] copy of the host binary / shared objects to u@h failed: scp: dest open '
           '"/r/kbhost_1d16bff26c55e4fd": Failure | scp: failed to upload file /w/x.host to /r/kbhost_1d16bff26c55e4fd')


def _run(ok, err=None, us=10.0):
    return {"ok": ok, "stage": "ok" if ok else "run", "partition_us": us if ok else None,
            "e2e_us": us if ok else None, "errors": [err] if err else [],
            "log": f"MB_ERROR {err}\nfailed\n" if err else "", "gpu": None}


class FakeKBEval:
    """kb_eval.KBEval stand-in: ``run`` writes and returns a time_problems result."""

    def __init__(self, runs_per_call):
        self.calls, self.runs_per_call = [], list(runs_per_call)

    def for_device(self, dev):
        return self

    def describe(self):
        return {"evaluator": "fake"}

    def run(self, jobs, *, arch, work, gpu_model=None, name):
        runs = self.runs_per_call.pop(0)
        work.mkdir(parents=True, exist_ok=True)
        self.calls.append(work)
        res = {"jobs": {j["id"]: {"id": j["id"], "runs": runs[k], "gpu": {"label": "u@h:2"}, "gpus": ["u@h:2"]}
                        for k, j in enumerate(jobs)}}
        (work / f"{name}.result.json").write_text(json.dumps(res))
        return res


class RemeasureFailureTextTest(unittest.TestCase):
    def test_failures_in_info_and_result_file(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            ev = FakeKBEval([[[_run(False, SCP_ERR)], [_run(True)]]])
            info = {}
            times = kp.remeasure([(t / "orig", {}), (t / "variant", {})], arch="89", reps=1, kbeval=ev,
                                 info=info, work=t / "rm")
            self.assertEqual(times, [[], [10.0]])
            (f,) = info["failures"]
            self.assertEqual((f["problem"], f["stage"], f["infra"]), (str(t / "orig"), "run", True))
            self.assertIn('dest open "/r/kbhost_1d16bff26c55e4fd": Failure', f["error"])
            doc = json.loads((t / "rm" / "remeasure.result.json").read_text())
            self.assertEqual(doc["failures"], info["failures"])

    def test_infra_classification(self):
        self.assertTrue(kp.is_infra_text(SCP_ERR))
        self.assertTrue(kp.is_infra_text("ssh: connect to host h port 22: Connection refused"))
        self.assertFalse(kp.is_infra_text("partition: output 0 mismatch rel_err 0.3"))
        self.assertFalse(kp.is_infra_text(None))


def _job(t: Path):
    for side in ("o", "v"):
        (t / side / "best" / "slots").mkdir(parents=True)
    (t / "variant").mkdir()
    (t / "variant" / "manifest.json").write_text(json.dumps({"slots": []}))
    hint = gh.fuse_hint("fused__ewise2__ewise2__a", [["add", "mul"]], {"from": "test"})
    return {"orig": t / "orig", "variant": t / "variant", "hint": hint, "ev": {"ir": "m"}, "names": ["add", "mul"],
            "stem": "m__h", "prov": {"applied": [hint["id"]]}, "specs": {}}


class MeasureHintTest(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.t = Path(self._td.name)
        self._old = go.kb_eval_for, go.use_kb_eval, go.remeasure, go._ingest_slots
        go.use_kb_eval = lambda cfg: True
        go._ingest_slots = lambda *a, **kw: []

    def tearDown(self):
        go.kb_eval_for, go.use_kb_eval, go.remeasure, go._ingest_slots = self._old
        self._td.cleanup()

    def _measure(self, calls, attempts=3):
        ev = FakeKBEval(calls)
        go.kb_eval_for = lambda cfg: ev
        job = _job(self.t)
        ro, rv = {"final_ok": True, "_dir": str(self.t / "o")}, {"final_ok": True, "_dir": str(self.t / "v")}
        cfg = go.Config(kb=self.t, remeasure=2, remeasure_attempts=attempts, remeasure_retry_s=0.0,
                        allow_gpu_mismatch=True, accept_gain=0.05)
        times = go.measure_hint(job, ro, rv, cfg, None)
        doc = gh.new_doc()
        res = go.finish_hint(job, dict(ro, final_partition_us=15.0), dict(rv, final_partition_us=10.0), times,
                             cache=None, doc=doc, cfg=cfg)
        return ev, job, res, doc

    def test_infra_failure_is_retried_on_a_fresh_lease(self):
        bad = [[_run(False, SCP_ERR)] * 2, [_run(False, SCP_ERR)] * 2]
        good = [[_run(True, us=15.0)] * 2, [_run(True, us=10.0)] * 2]
        ev, job, res, doc = self._measure([bad, good])
        self.assertEqual([w.name for w in ev.calls], ["remeasure", "remeasure_a2"])
        self.assertEqual([a["ok"] for a in job["remeasure_attempts"]], [False, True])
        self.assertTrue(job["remeasure_attempts"][0]["infra"])
        self.assertEqual(res["status"], "accepted")
        (e,) = gh.get(doc, job["hint"]["id"])["evidence"]
        self.assertEqual((e["original_us"], e["variant_us"], e["remeasure_attempts"]), (15.0, 10.0, 2))

    def test_persistent_infra_failure_leaves_the_hint_undecided(self):
        bad = [[_run(False, SCP_ERR)] * 2, [_run(False, SCP_ERR)] * 2]
        ev, job, res, doc = self._measure([bad, bad, bad])
        self.assertEqual(len(ev.calls), 3)
        self.assertEqual((res["status"], res["infra"]), ("undecided", True))
        h = gh.get(doc, job["hint"]["id"])
        self.assertEqual(h["status"], "proposed")                        # a candidate again next round
        self.assertNotIn(h["id"], gh.decided(doc))
        (e,) = h["evidence"]
        self.assertEqual(e["decision"], "undecided")
        self.assertIn("infrastructure failure", e["why"])
        self.assertIn('dest open "/r/kbhost_1d16bff26c55e4fd": Failure', e["why"])
        self.assertEqual({f["side"] for f in e["remeasure_errors"]}, {"original", "variant"})

    def test_a_candidate_failure_rejects_with_its_error(self):
        wrong = "partition: output 0 mismatch (rel_err 0.31 > 0.001)"
        calls = [[[_run(True, us=15.0)] * 2, [_run(False, wrong)] * 2]]
        ev, job, res, doc = self._measure(calls)
        self.assertEqual(len(ev.calls), 1)                              # not retried
        self.assertEqual(res["status"], "rejected")
        (e,) = gh.get(doc, job["hint"]["id"])["evidence"]
        self.assertIn("no passing run on a side", e["why"])
        self.assertIn("variant: [run] " + wrong, e["why"])

    def test_time_problems_not_running_is_infra(self):
        def boom(*a, **kw):
            raise RuntimeError("KernelBlaster time_problems failed (rc 1); see x.log: ssh: connect to host h "
                               "port 22: Connection timed out")
        go.remeasure = boom
        ev, job, res, doc = self._measure([], attempts=2)
        self.assertEqual(res["status"], "undecided")
        self.assertIn("Connection timed out", res["why"])


class KBFinalFailureTest(unittest.TestCase):
    """A KernelBlaster side whose final run failed: lost to the infrastructure (the FP16
    run's ``cannot load incumbent ./kbso_<sha>.so: No such file``, its shared object removed
    by another process) the hint is undecided; the composite's own failure rejects it."""

    def _finish(self, ro_extra, rv_extra):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            job = _job(t)
            cfg = go.Config(kb=t, remeasure=2, allow_gpu_mismatch=True)
            ro = {"final_ok": True, "final_partition_us": 15.0, "_dir": str(t / "o"), **ro_extra}
            rv = {"final_ok": True, "final_partition_us": 10.0, "_dir": str(t / "v"), **rv_extra}
            doc = gh.new_doc()
            res = go.finish_hint(job, ro, rv, None, cache=None, doc=doc, cfg=cfg)
            return res, gh.get(doc, job["hint"]["id"])

    def test_infra_final_failure_is_undecided(self):
        err = ("MB_ERROR cannot load incumbent ./kbso_b378842e59fe1919.so: No such file or directory "
               "MB_ERROR [infra] staged kbso_b378842e59fe1919.so missing on u@h (removed under this process)")
        res, h = self._finish({"final_ok": False, "final_partition_us": None, "final_error": err,
                               "final_infra": True}, {})
        self.assertEqual((res["status"], h["status"]), ("undecided", "proposed"))
        self.assertIn("kbso_b378842e59fe1919.so", res["why"])
        self.assertEqual(h["evidence"][0]["kb_errors"], {"original": err})

    def test_composite_failure_rejects_with_its_error(self):
        err = "output out0: max_abs_err=0.5 > bound=0.01"
        res, h = self._finish({}, {"final_ok": False, "final_error": err, "final_infra": False})
        self.assertEqual(res["status"], "rejected")
        self.assertIn("variant not final_ok (" + err, h["evidence"][0]["why"])

    def test_one_side_infra_other_side_broken_rejects(self):
        res, _ = self._finish({"final_ok": False, "final_error": "[infra] x", "final_infra": True},
                              {"final_ok": False, "error": "baseline failed (compile)"})
        self.assertEqual(res["status"], "rejected")


class RecordDecisionTest(unittest.TestCase):
    def test_undecided_does_not_vote(self):
        doc = gh.new_doc()
        h = gh.fuse_hint("fused__a__b", [["x", "y"]], {"from": "t"})
        self.assertEqual(gh.record_decision(doc, h, {"decision": "undecided"})["status"], "proposed")
        self.assertEqual(gh.record_decision(doc, h, {"decision": "accepted"})["status"], "accepted")
        self.assertEqual(gh.record_decision(doc, h, {"decision": "undecided"})["status"], "accepted")


if __name__ == "__main__":
    unittest.main()
