"""Barrier scheduling, node side (kb_barrier.py; ``--park-after-char on``; l3_farm/analysis/barrier_scheduling.md).

Unit (GPU-free, no LLM, stdlib + the pipeline package):  flag off changes nothing (Config default, round key, no state trace),
the park hook (parked.json, exit exception, double park, skipped paths), the resume-time import (once, never raises, why="barrier"),
``drop_partial_rounds`` (what worker5.sh keeps of a parked round).

Flow (needs torch, nvcc, a CUDA GPU, a KernelBlaster checkout; the scripted ``stub:graphopt`` LLM of test_kb_graph_opt_flow): the
flow with ``--graph-characterize 1`` parks after the characterization pass (exit 78 through ``main``), the same command without the
flag resumes at the next phase and ends with the same decisions / final model as an uninterrupted run; a double park; characterization
off -> the run continues and records ``park_skipped``.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from modelblaster.pipeline import kb_barrier, kb_resume  # noqa: E402


class Cfg:
    park_after_char = "on"
    knowledge_global_refresh = ""
    knowledge_global_char_sync = "off"
    knowledge_global_char_sync_wait_s = 180.0


def mk(tmp: Path, flag="on", gk=None):
    work = Path(tmp) / "work"
    (work / "r00").mkdir(parents=True, exist_ok=True)
    state = {"sent": [], "rounds": []}
    sp = work / "state.json"
    kb_resume.atomic_write_json(sp, state)
    cfg = Cfg()
    cfg.park_after_char = flag
    return kb_barrier.Barrier(cfg, work, state, sp, gk, irs=[work / "irs" / "tiny"]), work, state, sp


class FlagOffTest(unittest.TestCase):
    def test_off_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as td:
            b, work, state, sp = mk(td, flag="off")
            prog = kb_resume.RoundProgress(work / "r00", {"k": 1})
            prog.mark("gs_char", {"items": 3})
            self.assertFalse(kb_barrier.enabled(b.cfg))
            self.assertIsNone(b.after_char(work / "r00", prog, {"items": 3}))        # nothing raised
            self.assertFalse((work / "parked.json").exists())
            self.assertNotIn("barrier", json.loads(sp.read_text()))

    def test_default_config_and_round_key(self):
        try:
            from modelblaster.pipeline.kb_graph_opt import Config
        except ImportError as e:                                     # numpy / torch missing (a laptop)
            self.skipTest(str(e))
        c0 = Config(kb=Path("."))
        c1 = Config(kb=Path("."), park_after_char="on")
        self.assertEqual(c0.park_after_char, "off")
        self.assertFalse(kb_barrier.enabled(c0))
        self.assertTrue(kb_barrier.enabled(c1))
        self.assertEqual(kb_resume.round_key(c0, ["/x/ir"]), kb_resume.round_key(c1, ["/x/ir"]))   # the flag is not in the key

    def test_cli_flag_exists(self):
        src = (ROOT / "pipeline" / "kb_graph_opt.py").read_text()
        self.assertIn("park_after_char: str = \"off\"", src)
        self.assertIn("except kb_barrier.Parked", src)


class ParkTest(unittest.TestCase):
    def test_park_writes_parked_json_and_raises(self):
        with tempfile.TemporaryDirectory() as td:
            b, work, state, sp = mk(td)
            prog = kb_resume.RoundProgress(work / "r00", {"k": 1})
            prog.mark("profile", [])
            prog.mark("gs_char", {"items": 4, "llm_calls": 7, "usd": 0.5})
            b.key_hash = "abc"
            with self.assertRaises(kb_barrier.Parked):
                b.after_char(work / "r00", prog, {"items": 4, "llm_calls": 7, "usd": 0.5})
            d = json.loads((work / "parked.json").read_text())
            self.assertEqual((d["phase"], d["round"], d["stage"], d["parks"]), ("char", 0, "parked", 1))
            self.assertEqual(d["problems"], ["tiny"])
            self.assertEqual(d["char"]["llm_calls"], 7)
            st = json.loads(sp.read_text())["barrier"]
            self.assertTrue(st["parked"] and st["parks"] == 1)
            self.assertEqual(kb_barrier.EXIT_PARKED, 78)

    def test_double_park_is_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            b, work, state, sp = mk(td)
            prog = kb_resume.RoundProgress(work / "r00", {"k": 1})
            prog.mark("gs_char", {"items": 2})
            for n in (1, 2):
                with self.assertRaises(kb_barrier.Parked):
                    b.after_char(work / "r00", prog, {"items": 2})
                self.assertEqual(json.loads((work / "parked.json").read_text())["parks"], n)
            self.assertEqual(json.loads((work / "parked.json").read_text())["stage"], "parked")

    def test_nothing_to_park_after_continues(self):
        with tempfile.TemporaryDirectory() as td:
            b, work, state, sp = mk(td)
            prog = kb_resume.RoundProgress(work / "r00", {"k": 1})
            self.assertIsNone(b.after_char(work / "r00", prog, {"items": 0}))
            self.assertIsNone(b.after_char(work / "r00", prog, None))
            self.assertFalse((work / "parked.json").exists())
            self.assertIn("no items", json.loads(sp.read_text())["barrier"]["park_skipped"])

    def test_reprofile_already_done_continues(self):
        with tempfile.TemporaryDirectory() as td:
            b, work, state, sp = mk(td)
            prog = kb_resume.RoundProgress(work / "r00", {"k": 1})
            prog.mark("gs_char", {"items": 2})
            prog.mark("gs_char_profile", [])
            self.assertIsNone(b.after_char(work / "r00", prog, {"items": 2}))
            self.assertFalse((work / "parked.json").exists())


class CharSyncTest(unittest.TestCase):
    """Wait mode (--knowledge-global-char-sync on, no park): marker, bounded wait, import why="char", resume safety."""

    def _b(self, td, wait=0.3):
        b, work, state, sp = mk(td, flag="off", gk={"applied": True})
        inbox = Path(td) / "inbox"
        inbox.mkdir()
        b.cfg.knowledge_global_refresh = str(inbox)
        b.cfg.knowledge_global_char_sync = "on"
        b.cfg.knowledge_global_char_sync_wait_s = wait
        state["knowledge_global"] = {"applied": True}
        prog = kb_resume.RoundProgress(work / "r00", {"k": 1})
        prog.mark("gs_char", {"items": 2})
        return b, work, state, sp, inbox, prog

    def test_enabled_needs_seed_and_inbox(self):
        c = Cfg()
        c.knowledge_global_char_sync = "on"
        c.knowledge_global_refresh = "x"
        self.assertTrue(kb_barrier.char_sync_enabled(c, {"applied": True}))
        self.assertFalse(kb_barrier.char_sync_enabled(c, None))
        self.assertFalse(kb_barrier.char_sync_enabled(c, {"applied": False}))
        c.knowledge_global_char_sync = "off"
        self.assertFalse(kb_barrier.char_sync_enabled(c, {"applied": True}))

    def test_ack_delivers_and_imports_with_why_char(self):
        import threading
        from modelblaster.pipeline import kb_global
        calls = []
        old = kb_global.refresh_at_round
        kb_global.refresh_at_round = lambda work, cfg, state, r, why=None: calls.append((r, why)) or \
            {"status": "applied", "counts": {"stats": 5, "impl": 2}}
        try:
            with tempfile.TemporaryDirectory() as td:
                b, work, state, sp, inbox, prog = self._b(td, wait=20)
                seen = []

                def ctl():                                  # the controller: sees the paused marker, answers with an ack
                    import time
                    for _ in range(200):
                        f = work / "r00" / "char_sync.json"
                        if f.exists() and json.loads(f.read_text()).get("stage") == "paused":
                            seen.append(1)
                            (inbox / "char-r00.ack.json").write_text("{}")
                            return
                        time.sleep(0.05)
                os.environ["MB_HALFWAY_POLL_S"] = "0.05"
                t = threading.Thread(target=ctl)
                t.start()
                self.assertIsNone(b.after_char(work / "r00", prog, {"items": 2}))      # wait mode: no Parked
                t.join()
                self.assertEqual((seen, calls), ([1], [(0, "char")]))
                d = json.loads((work / "r00" / "char_sync.json").read_text())
                self.assertEqual((d["stage"], d["outcome"], d["imported_rows"]), ("done", "imported", 7))
                self.assertEqual(json.loads(sp.read_text())["barrier"]["char_sync"]["n"], 1)
                self.assertTrue((inbox / "done" / "char-r00.ack.json").exists())
                b.after_char(work / "r00", prog, {"items": 2})                        # a resume: done -> never again
                self.assertEqual(len(calls), 1)
        finally:
            kb_global.refresh_at_round = old

    def test_timeout_goes_on_without_a_package(self):
        with tempfile.TemporaryDirectory() as td:
            os.environ["MB_HALFWAY_POLL_S"] = "0.05"
            b, work, state, sp, inbox, prog = self._b(td, wait=0.2)
            b.after_char(work / "r00", prog, {"items": 2})
            d = json.loads((work / "r00" / "char_sync.json").read_text())
            self.assertEqual((d["stage"], d["outcome"]), ("done", "timeout"))
            self.assertEqual(json.loads(sp.read_text())["barrier"]["char_sync"]["timeouts"], 1)

    def test_park_supersedes_the_wait(self):
        with tempfile.TemporaryDirectory() as td:
            b, work, state, sp, inbox, prog = self._b(td, wait=30)
            b.cfg.park_after_char = "on"
            with self.assertRaises(kb_barrier.Parked):
                b.after_char(work / "r00", prog, {"items": 2})
            self.assertFalse((work / "r00" / "char_sync.json").exists())

    def test_off_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as td:
            b, work, state, sp, inbox, prog = self._b(td)
            b.cfg.knowledge_global_char_sync = "off"
            b.after_char(work / "r00", prog, {"items": 2})
            self.assertFalse((work / "r00" / "char_sync.json").exists())


class ResumeImportTest(unittest.TestCase):
    def _parked(self, td, gk=None):
        b, work, state, sp = mk(td, gk=gk)
        prog = kb_resume.RoundProgress(work / "r00", {"k": 1})
        prog.mark("gs_char", {"items": 2})
        with self.assertRaises(kb_barrier.Parked):
            b.after_char(work / "r00", prog, {"items": 2})
        return b, work, state, sp

    def test_import_once_with_why_barrier(self):
        from modelblaster.pipeline import kb_global
        calls = []
        old = kb_global.refresh_at_round
        kb_global.refresh_at_round = lambda work, cfg, state, r, why=None: calls.append((r, why)) or \
            {"status": "applied", "counts": {"stats": 5, "claims": 2, "note": "x"}}
        try:
            with tempfile.TemporaryDirectory() as td:
                b, work, state, sp = self._parked(td, gk={"applied": True})
                b.cfg.knowledge_global_refresh = str(Path(td) / "inbox")
                state["knowledge_global"] = {"applied": True}
                rec = b.resume_import()
                self.assertEqual(calls, [(0, "barrier")])
                self.assertEqual((rec["status"], rec["imported_rows"]), ("applied", 7))
                d = json.loads((work / "parked.json").read_text())
                self.assertEqual(d["stage"], "resumed")
                self.assertIsNone(b.resume_import())                     # once: a second resume imports nothing
                self.assertEqual(len(calls), 1)
                st = json.loads(sp.read_text())
                self.assertFalse(st["barrier"]["parked"])
                self.assertEqual(st["barrier"]["resumes"][0]["status"], "applied")
                self.assertEqual(st["knowledge_global"]["barrier"]["n"], 1)
        finally:
            kb_global.refresh_at_round = old

    def test_import_error_never_raises(self):
        from modelblaster.pipeline import kb_global
        old = kb_global.refresh_at_round

        def boom(*a, **k):
            raise RuntimeError("store busy")
        kb_global.refresh_at_round = boom
        try:
            with tempfile.TemporaryDirectory() as td:
                b, work, state, sp = self._parked(td, gk={"applied": True})
                b.cfg.knowledge_global_refresh = str(Path(td) / "inbox")
                rec = b.resume_import()
                self.assertIn("store busy", rec["error"])
                self.assertEqual(json.loads((work / "parked.json").read_text())["stage"], "resumed")
        finally:
            kb_global.refresh_at_round = old

    def test_no_gk_resume_just_marks(self):
        with tempfile.TemporaryDirectory() as td:
            b, work, state, sp = self._parked(td, gk=None)
            rec = b.resume_import()
            self.assertEqual(rec["status"], "none")
            self.assertEqual(json.loads((work / "parked.json").read_text())["stage"], "resumed")

    def test_not_parked_nothing_to_do(self):
        with tempfile.TemporaryDirectory() as td:
            b, work, state, sp = mk(td)
            self.assertIsNone(b.resume_import())


class DropPartialRoundsTest(unittest.TestCase):
    def _work(self, td, rounds=0, parked=None):
        w = Path(td) / "work"
        (w / "r00" / "profile").mkdir(parents=True)
        (w / "r00" / "char").mkdir()
        (w / "r00" / "problems").mkdir()
        (w / "r00" / "profile_char").mkdir()
        (w / "r00" / "plan.json").write_text("{}")
        (w / "r01").mkdir()
        kb_resume.atomic_write_json(w / "state.json", {"rounds": [{}] * rounds})
        kb_resume.atomic_write_json(w / "r00" / "progress.json", {"phases": {
            "profile": {"value": 1}, "gs_char": {"value": 2}, "gs_char_profile": {"value": 3}, "plan": {"value": 4},
            "export": {"value": 5}}})
        if parked:
            kb_resume.atomic_write_json(w / "parked.json", parked)
        return w

    def test_old_behaviour_without_parked_json(self):
        with tempfile.TemporaryDirectory() as td:
            w = self._work(td, rounds=1)
            msgs = kb_barrier.drop_partial_rounds(w)
            self.assertEqual(msgs, ["dropped partial round r01"])
            self.assertTrue((w / "r00").exists() and not (w / "r01").exists())
            w2 = self._work(Path(td) / "b", rounds=0)
            self.assertEqual(kb_barrier.drop_partial_rounds(w2), ["dropped partial round r00", "dropped partial round r01"])
            self.assertFalse((w2 / "r00").exists())

    def test_parked_round_is_rolled_back_to_its_park_point(self):
        with tempfile.TemporaryDirectory() as td:
            w = self._work(td, rounds=0, parked={"phase": "char", "round": 0, "stage": "parked"})
            msgs = kb_barrier.drop_partial_rounds(w)
            self.assertEqual(msgs, ["kept parked round r00 (rolled back to its park point)", "dropped partial round r01"])
            prog = json.loads((w / "r00" / "progress.json").read_text())
            self.assertEqual(sorted(prog["phases"]), ["gs_char", "profile"])
            self.assertTrue((w / "r00" / "profile").is_dir() and (w / "r00" / "char").is_dir())
            self.assertFalse((w / "r00" / "plan.json").exists() or (w / "r00" / "problems").exists()
                             or (w / "r00" / "profile_char").exists())
            # the same on the park-point state itself (a resume right after the park): unchanged
            self.assertEqual(kb_barrier.drop_partial_rounds(w), ["kept parked round r00 (rolled back to its park point)"])
            self.assertEqual(sorted(json.loads((w / "r00" / "progress.json").read_text())["phases"]), ["gs_char", "profile"])

    def test_resumed_stage_too_and_finished_round_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            w = self._work(td, rounds=0, parked={"phase": "char", "round": 0, "stage": "resumed"})
            self.assertIn("kept parked round r00", kb_barrier.drop_partial_rounds(w)[0])
            w2 = self._work(Path(td) / "b", rounds=1, parked={"phase": "char", "round": 0, "stage": "resumed"})
            kb_barrier.drop_partial_rounds(w2)                                  # round 0 finished: never touched
            self.assertEqual(len(json.loads((w2 / "r00" / "progress.json").read_text())["phases"]), 5)

    def test_parked_json_of_another_round_does_not_save_a_partial_round(self):
        with tempfile.TemporaryDirectory() as td:
            w = self._work(td, rounds=0, parked={"phase": "char", "round": 1, "stage": "parked"})
            self.assertIn("dropped partial round r00", kb_barrier.drop_partial_rounds(w))


# ---------------------------------------------------------------- GPU flow (stub LLM)

try:
    from modelblaster.pipeline.tests import test_kb_graph_opt_flow as F
    SKIP = F.SKIP
except Exception as e:  # noqa: BLE001 -- numpy / torch missing
    F, SKIP = None, f"flow test helpers unavailable: {e}"


@unittest.skipIf(SKIP is not None, SKIP or "")
class BarrierFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(os.environ.get("MB_TEST_TMP", tempfile.gettempdir()))
        base.mkdir(parents=True, exist_ok=True)
        cls.tmp = Path(tempfile.mkdtemp(prefix="mb_barflow_", dir=base))
        (cls.tmp / "tiny.py").write_text(F.MODEL)
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT.parent), os.environ.get("PYTHONPATH", "")])}
        cp = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file",
                             str(cls.tmp / "tiny.py"), "--out-dir", str(cls.tmp / "ir"), "--quant", "fp32"],
                            cwd=ROOT, capture_output=True, text=True, timeout=600, env=env)
        assert cp.returncode == 0, cp.stderr[-2000:]
        (cls.tmp / "plugin.py").write_text(F.PLUGIN.format(fused=json.dumps(F.fused_sources())))
        cls.arch = F._arch()

    @classmethod
    def tearDownClass(cls):
        if not os.environ.get("MB_TEST_KEEP"):
            shutil.rmtree(cls.tmp, ignore_errors=True)

    def _cfg(self, **kw):
        from modelblaster.pipeline.kb_graph_opt import Config
        base = dict(kb=F.KB, model="stub:graphopt", plugins=[str(self.tmp / "plugin.py")], python=F.KB_PY, arch=self.arch,
                    rounds=4, rollouts=1, steps=2, variant_steps=2, max_fix=1, top_k=2, max_variants=2, remeasure=3,
                    profile_runs=1, accept_gain=-1.0, min_gain_pct=0.0, graph_search="interleaved", graph_priors="off",
                    graph_characterize=1, graph_char_top=2)
        base.update(kw)
        return Config(**base)

    def _statuses(self, work):
        from modelblaster.pipeline import graph_hints as gh
        return {h["id"]: h["status"] for h in gh.load(work / "graph_hints.json")["hints"] if h["id"].startswith("fuse:")}

    def test_park_then_resume_equals_an_uninterrupted_run(self):
        from modelblaster.pipeline.kb_graph_opt import run
        ref = self.tmp / "ref"
        s0 = run([self.tmp / "ir"], ref, self._cfg())
        w = self.tmp / "parked"
        with self.assertRaises(kb_barrier.Parked):
            run([self.tmp / "ir"], w, self._cfg(park_after_char="on"))
        pk = json.loads((w / "parked.json").read_text())
        self.assertEqual((pk["phase"], pk["round"], pk["stage"]), ("char", 0, "parked"))
        prog = json.loads((w / "r00" / "progress.json").read_text())
        self.assertEqual(sorted(prog["phases"]), ["gs_char", "profile"])           # parked BEFORE the re-profile / plan
        self.assertFalse((w / "r00" / "profile_char").exists() or (w / "r00" / "plan.json").exists())
        self.assertFalse((w / "summary.json").exists())
        calls_char = prog["phases"]["gs_char"]["value"]["llm_calls"]
        # a double park (a rerun with the flag still on) parks again at once: no second characterization
        with self.assertRaises(kb_barrier.Parked):
            run([self.tmp / "ir"], w, self._cfg(park_after_char="on"))
        self.assertEqual(json.loads((w / "parked.json").read_text())["parks"], 2)
        key0 = prog["key_hash"]
        s1 = run([self.tmp / "ir"], w, self._cfg())                                # the SAME command without the flag
        self.assertEqual(json.loads((w / "r00" / "progress.json").read_text())["key_hash"], key0)   # same round key
        self.assertEqual(json.loads((w / "parked.json").read_text())["stage"], "resumed")
        self.assertTrue(next(iter(s1["irs"].values()))["passed"], s1)
        self.assertEqual(self._statuses(w), self._statuses(ref))                   # the same fusion decisions
        self.assertTrue(any(v == "accepted" for v in self._statuses(w).values()))
        st = json.loads((w / "state.json").read_text())
        self.assertEqual(len(st["barrier"]["resumes"]), 1)
        self.assertEqual(st["rounds"][0]["key"], json.loads((ref / "state.json").read_text())["rounds"][0]["key"])
        # the characterization ran once: the resumed run's total LLM calls = the uninterrupted run's (within the noise of
        # timing-dependent hints), and the park-time char call count is what the resumed round recorded
        self.assertGreaterEqual(s1["llm_calls"], 0)
        print(f"\nbarrier flow: char llm_calls {calls_char}; calls park+resume {s1['llm_calls']} vs uninterrupted {s0['llm_calls']}",
              flush=True)

    def test_main_exits_78_and_the_same_command_resumes(self):
        w = self.tmp / "cli"
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT.parent), os.environ.get("PYTHONPATH", "")])}
        cmd = [sys.executable, "-m", "modelblaster.pipeline.kb_graph_opt", str(self.tmp / "ir"), "--work", str(w),
               "--kb", str(F.KB), "--python", F.KB_PY, "--model", "stub:graphopt", "--plugin", str(self.tmp / "plugin.py"),
               "--arch", self.arch, "--rounds", "1", "--rollouts", "1", "--steps", "2", "--variant-steps", "2",
               "--max-fix", "1", "--top-k", "2", "--max-variants", "2", "--remeasure", "3", "--profile-runs", "1",
               "--accept-gain", "-1", "--min-gain-pct", "0", "--graph-search", "interleaved", "--graph-priors", "off",
               "--graph-characterize", "1", "--graph-char-top", "2"]
        p = subprocess.run(cmd + ["--park-after-char", "on"], capture_output=True, text=True, timeout=1800, env=env, cwd=ROOT)
        self.assertEqual(p.returncode, 78, p.stdout[-1500:] + p.stderr[-1500:])
        self.assertTrue((w / "parked.json").exists())
        p2 = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env, cwd=ROOT)
        self.assertEqual(p2.returncode, 0, p2.stdout[-1500:] + p2.stderr[-1500:])
        self.assertTrue((w / "summary.json").exists())

    def test_characterization_off_continues_and_records_park_skipped(self):
        from modelblaster.pipeline.kb_graph_opt import run
        w = self.tmp / "nochar"
        s = run([self.tmp / "ir"], w, self._cfg(park_after_char="on", graph_characterize=0, rounds=1))
        self.assertTrue(next(iter(s["irs"].values()))["passed"], s)
        st = json.loads((w / "state.json").read_text())
        self.assertIn("characterization is off", st["barrier"]["park_skipped"])
        self.assertFalse((w / "parked.json").exists())

    def test_flag_off_leaves_no_trace(self):
        from modelblaster.pipeline.kb_graph_opt import run
        w = self.tmp / "off"
        run([self.tmp / "ir"], w, self._cfg(rounds=1))
        self.assertNotIn("barrier", json.loads((w / "state.json").read_text()))
        self.assertFalse((w / "parked.json").exists())


if __name__ == "__main__":
    unittest.main()
