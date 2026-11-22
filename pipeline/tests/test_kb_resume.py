"""Resume of the graph optimiser (pipeline/kb_resume.py).

Unit tests (no GPU): round progress, stale rounds, resumed hint waves, kept hint jobs,
signal forwarding. GPU integration (skipped without torch / nvcc / a GPU / a
KernelBlaster checkout with resume): kb_graph_opt on a tiny model killed with SIGKILL
in the middle of a KernelBlaster run and in the middle of a hint decision, then
resumed: the same hints accepted and the same final model as an uninterrupted run,
with no journaled LLM call repeated.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from modelblaster.pipeline import kb_resume as R  # noqa: E402


class _Cfg:
    def __init__(self, **kw):
        for k in R.ROUND_FIELDS:
            setattr(self, k, kw.get(k, 1))


class RoundProgressTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mb_resume_"))

    def test_once_saves_and_resumes(self):
        key = R.round_key(_Cfg(), [self.tmp / "ir"])
        p = R.RoundProgress(self.tmp / "r00", key)
        self.assertFalse(p.resumed)
        calls = []
        self.assertEqual(p.once("plan", lambda: calls.append(1) or {"a": 1}), {"a": 1})
        q = R.RoundProgress(self.tmp / "r00", key)                  # a rerun after a kill
        self.assertTrue(q.resumed)
        self.assertEqual(q.once("plan", lambda: calls.append(2) or {"a": 2}), {"a": 1})
        self.assertEqual(calls, [1])
        self.assertEqual(len(q.data["attempts"]), 2)

    def test_other_config_moves_the_round_aside(self):
        p = R.RoundProgress(self.tmp / "r00", R.round_key(_Cfg(steps=1), [self.tmp / "ir"]))
        p.mark("profile", [1])
        (self.tmp / "r00" / "x").write_text("old")
        q = R.RoundProgress(self.tmp / "r00", R.round_key(_Cfg(steps=2), [self.tmp / "ir"]))
        self.assertFalse(q.resumed or q.done("profile"))
        self.assertTrue(q.moved and Path(q.moved, "x").exists())
        self.assertFalse((self.tmp / "r00" / "x").exists())
        # a round dir from before progress.json existed is an unknown attempt: moved too
        (self.tmp / "r01").mkdir()
        z = R.RoundProgress(self.tmp / "r01", q.key)
        self.assertTrue(z.moved)

    def test_decide_hints_skips_finished_waves(self):
        key = R.round_key(_Cfg(), [])
        prog = R.RoundProgress(self.tmp / "r00", key)
        cands = [({"id": f"h{i}"}, {"ir": "m"}) for i in range(4)]
        sched = lambda fps, w: [[0, 1], [2], [3]]                    # noqa: E731
        seen = []

        def wave(k, items, die_at=None):
            if k == die_at:
                raise KeyboardInterrupt                               # stand-in for a kill
            seen.append([h["id"] for h, _, _ in items])
            return [{"hint": h["id"], "status": "rejected"} for h, _, _ in items]

        saves = []
        kw = dict(schedule=sched, footprint=lambda *a: None, specs={}, hints_path=self.tmp / "h.json",
                  save_hints=lambda: saves.append(1))
        with self.assertRaises(KeyboardInterrupt):
            R.decide_hints(prog, cands, {"m": {}}, self.tmp / "v", 2,
                           decide_wave=lambda k, it: wave(k, it, die_at=1), **kw)
        self.assertEqual(seen, [["h0", "h1"]])
        prog2 = R.RoundProgress(self.tmp / "r00", key)
        waves, decs = R.decide_hints(prog2, cands, {"m": {}}, self.tmp / "v", 2, decide_wave=wave, **kw)
        self.assertEqual(seen, [["h0", "h1"], ["h2"], ["h3"]])       # wave 0 not redone
        self.assertEqual([d["hint"] for d in decs], ["h0", "h1", "h2", "h3"])
        self.assertEqual(waves, [["h0", "h1"], ["h2"], ["h3"]])

    def test_prepared_keeps_the_exported_job(self):
        out = self.tmp / "hint"
        for side in ("orig", "variant"):
            (out / side).mkdir(parents=True)
            (out / side / "manifest.json").write_text("{}")
        ir = self.tmp / "ir"
        ir.mkdir()
        (ir / "graph.json").write_text(json.dumps({"ops": []}))
        n = []

        def prep(h, prof, o, **kw):
            n.append(1)
            return {"hint": h, "specs": {}, "names": ["a"], "ev": {"ir": "m"}, "orig": out / "orig",
                    "variant": out / "variant", "stem": "s", "prov": {"applied": ["x"]}}

        h, prof = {"id": "fuse:x"}, {"ir_dir": str(ir)}
        j1 = R.prepared(prep, h, prof, out, specs={})
        j2 = R.prepared(prep, h, prof, out, specs={})
        self.assertEqual(n, [1])
        self.assertTrue(j2["resumed"] and j2["orig"] == j1["orig"] and j2["prov"] == j1["prov"])

    def test_kb_exit(self):
        R.kb_exit(0, Path("x"))
        with self.assertRaises(R.Interrupted):
            R.kb_exit(75, Path("x"))


class StopTest(unittest.TestCase):
    def test_first_signal_forwards_second_exits(self):
        code = (
            "import os, signal, subprocess, sys, time\n"
            f"sys.path.insert(0, {str(ROOT.parent)!r})\n"
            "from modelblaster.pipeline import kb_resume as R\n"
            "R.STOP.install()\n"
            "c = subprocess.Popen([sys.executable, '-c', 'import signal, time, sys\\n"
            "signal.signal(signal.SIGTERM, lambda *a: (print(\"child term\", flush=True), sys.exit(75)))\\n"
            "time.sleep(30)'], start_new_session=True)\n"
            "R.STOP.register(c)\n"
            "print('ready', flush=True)\n"
            "rc = c.wait()\n"
            "print('child rc', rc, flush=True)\n"
            "try:\n"
            "    R.kb_exit(rc, None)\n"
            "except R.Interrupted:\n"
            "    print('interrupted', os.environ.get('KB_STOP_WITH_PARENT'), flush=True)\n"
            "time.sleep(30)\n")
        p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
        self.assertEqual(p.stdout.readline().strip(), "ready")
        time.sleep(0.5)
        p.send_signal(signal.SIGTERM)
        out = [p.stdout.readline().strip() for _ in range(4)]
        self.assertIn("child term", out)
        self.assertIn("child rc 75", out)
        self.assertIn("interrupted 1", out)
        p.send_signal(signal.SIGTERM)
        self.assertEqual(p.wait(timeout=10), R.EXIT_ABORTED)


# ---------------------------------------------------------------- GPU integration

def _skip() -> str | None:
    try:
        from modelblaster.pipeline.tests import test_kb_graph_opt_flow as F
    except Exception as e:  # noqa: BLE001
        return f"flow test helpers: {e}"
    if F.SKIP:
        return F.SKIP
    if not R.kb_supports_resume(F.KB):
        return f"{F.KB} has no resumable runs"
    return None


SKIP = _skip()

COUNTING_PLUGIN = '''
import json, os
from kernelblaster.agents.utils import agentic_llm
exec(open({base!r}).read())
_inner = agentic_llm._stubs["graphopt"]


def counted(msgs):
    with open(os.environ["MB_TEST_LLM_LOG"], "a") as f:
        f.write("1\\n")
    return _inner(msgs)


agentic_llm.register_stub("graphopt", counted)
'''


@unittest.skipIf(SKIP is not None, SKIP or "")
class GraphOptResumeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from modelblaster.pipeline.tests import test_kb_graph_opt_flow as F
        cls.F = F
        base = Path(os.environ.get("MB_TEST_TMP", tempfile.gettempdir()))
        base.mkdir(parents=True, exist_ok=True)
        cls.tmp = Path(tempfile.mkdtemp(prefix="mb_resume_flow_", dir=base))
        (cls.tmp / "tiny.py").write_text(F.MODEL)
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT.parent), os.environ.get("PYTHONPATH", "")])}
        cp = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file",
                             str(cls.tmp / "tiny.py"), "--out-dir", str(cls.tmp / "ir"), "--quant", "fp32"],
                            cwd=ROOT, capture_output=True, text=True, timeout=600, env=env)
        assert cp.returncode == 0, cp.stderr[-2000:]
        (cls.tmp / "plugin_base.py").write_text(F.PLUGIN.format(fused=json.dumps(F.fused_sources())))
        (cls.tmp / "plugin.py").write_text(COUNTING_PLUGIN.format(base=str(cls.tmp / "plugin_base.py")))
        cls.arch = F._arch()
        cls.env = env

    @classmethod
    def tearDownClass(cls):
        if not os.environ.get("MB_TEST_KEEP"):
            import shutil
            shutil.rmtree(cls.tmp, ignore_errors=True)

    def _cmd(self, work):
        return [sys.executable, "-m", "modelblaster.pipeline.kb_graph_opt", str(self.tmp / "ir"), "--work", str(work),
                "--kb", str(self.F.KB), "--model", "stub:graphopt", "--plugin", str(self.tmp / "plugin.py"),
                "--python", self.F.KB_PY, "--arch", self.arch, "--rounds", "3", "--rollouts", "1", "--steps", "2",
                "--variant-steps", "1", "--max-fix", "1", "--top-k", "2", "--max-variants", "2", "--remeasure", "3",
                "--profile-runs", "1", "--accept-gain", "-1.0"]

    def _run(self, work, log, kill_when=None, timeout=1800):
        env = {**self.env, "MB_TEST_LLM_LOG": str(log)}
        p = subprocess.Popen(self._cmd(work), cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True)
        if kill_when is None:
            out, _ = p.communicate(timeout=timeout)
            return p.returncode, out
        deadline = time.time() + timeout
        while time.time() < deadline and p.poll() is None and not kill_when(work):
            time.sleep(0.1)
        self.assertIsNone(p.poll(), "finished before the kill point")
        p.send_signal(signal.SIGKILL)
        p.wait()
        return p.returncode, p.stdout.read()

    def _digest(self, work):
        from modelblaster.pipeline import graph_hints as gh
        s = json.loads((work / "summary.json").read_text())
        doc = gh.load(work / "graph_hints.json")
        ir = next(iter(s["irs"].values()))
        return {"status": {h["id"]: h["status"] for h in doc["hints"]},
                "n_evidence": {h["id"]: len(h.get("evidence", [])) for h in doc["hints"]},
                "fused": sorted(ir["fused"] or []), "passed": ir["passed"], "rounds": s["rounds"],
                "fixpoint": s["fixpoint"], "llm_calls": s["llm_calls"]}

    @staticmethod
    def _calls(log):
        return len(Path(log).read_text().splitlines()) if Path(log).exists() else 0

    def test_kill_mid_kb_run_and_mid_decision_then_resume(self):
        ref_w, ref_log = self.tmp / "ref", self.tmp / "ref.log"
        rc, out = self._run(ref_w, ref_log)
        self.assertEqual(rc, 0, out[-3000:])
        self.assertTrue(self._digest(ref_w)["passed"])

        work, log = self.tmp / "w", self.tmp / "w.log"

        def mid_kb(w):        # the round-0 KernelBlaster run has journaled a response
            return any(w.glob("r00/run/*/journal/llm.jsonl"))

        def mid_decision(w):  # a hint's variant run is under way (its problems exported)
            return any(w.glob("r00/variants/**/journal/state.json"))

        rc, out = self._run(work, log, mid_kb)
        self.assertEqual(rc, -signal.SIGKILL)
        rc, out = self._run(work, log, mid_decision)
        self.assertEqual(rc, -signal.SIGKILL)
        prog = json.loads((work / "r00" / "progress.json").read_text())
        self.assertIn("export", prog["phases"])
        rc, out = self._run(work, log)
        self.assertEqual(rc, 0, out[-3000:])
        got = self._digest(work)
        # no decision recorded twice, the final model verified
        self.assertTrue(got["passed"], got)
        self.assertTrue(all(n == 1 for n in got["n_evidence"].values()), got)
        # round 0 (the killed one) decided exactly its saved candidates, once each
        st = json.loads((work / "state.json").read_text())
        cands = [hid for hid, _ in prog["phases"]["cands"]["value"]["cands"]]
        self.assertEqual([d["hint"] for d in st["rounds"][0]["decisions"]], cands)
        # the GPU timings differ between runs, so later rounds may propose other hints; where
        # round 0 saw the same candidates as the uninterrupted run, it decided them the same way
        ref_r0 = json.loads((ref_w / "state.json").read_text())["rounds"][0]
        if [d["hint"] for d in ref_r0["decisions"]] == cands:
            self.assertEqual([(d["hint"], d["status"]) for d in st["rounds"][0]["decisions"]],
                             [(d["hint"], d["status"]) for d in ref_r0["decisions"]])
            self.assertEqual(st["rounds"][0]["llm_calls_total"], ref_r0["llm_calls_total"])
        # LLM calls: every call the stub answered is counted once in the results, i.e. no
        # journaled response was paid again (at most a call in flight at each of the two kills)
        paid = self._calls(log)
        self.assertLessEqual(got["llm_calls"], paid)
        self.assertLessEqual(paid, got["llm_calls"] + 2)
        # a rerun of a complete work dir does nothing
        rc, out = self._run(work, log)
        self.assertEqual(rc, 0, out[-3000:])
        self.assertIn("already complete", out)
        self.assertEqual(self._calls(log), paid)

    def test_sigterm_stops_and_resumes(self):
        work, log = self.tmp / "t", self.tmp / "t.log"
        env = {**self.env, "MB_TEST_LLM_LOG": str(log)}
        p = subprocess.Popen(self._cmd(work), cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True)
        deadline = time.time() + 900
        while time.time() < deadline and p.poll() is None and not any(work.glob("r00/run/*/journal/llm.jsonl")):
            time.sleep(0.1)
        p.send_signal(signal.SIGTERM)
        out, _ = p.communicate(timeout=600)
        self.assertEqual(p.returncode, R.EXIT_INTERRUPTED, out[-3000:])
        rc, out = self._run(work, log)
        self.assertEqual(rc, 0, out[-3000:])
        self.assertTrue(json.loads((work / "summary.json").read_text())["irs"])


if __name__ == "__main__":
    unittest.main()
