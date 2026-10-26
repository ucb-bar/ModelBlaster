"""Halfway sync of global knowledge, node side (kb_halfway.py, KBRunner, kb_graph_opt flags): GPU-free, no LLM.

Unit: flag off changes nothing (Config default, round key, KernelBlaster command line); on adds one constant marker to
the round key. Run (KernelBlaster's real ``run_multislot.py`` with its stub LLM / fake evaluator plugin, as its resume
tests): the plan run parks at half its steps, a scripted controller delivers (or does not deliver) the package, the run
continues with every rollout's own state and ends exactly as an uninterrupted run; timeout / error / crash-during-pause
paths; telemetry in state.json; a two-run exchange with the real merge / refresh machinery.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from modelblaster.pipeline import kb_global, kb_halfway, kb_resume  # noqa: E402
from modelblaster.pipeline.kb_graph_opt import Config, KBRunner  # noqa: E402

KB = Path(os.environ.get("MB_KB_ROOT", ROOT / "third_party" / "KernelBlaster"))
FIX = KB / "tests" / "fixtures" / "mb_lenet"
PLUGIN = KB / "tests" / "fixtures" / "resume_plugin.py"
PROB = "lenet__conv1__fc3"


def _kb_ok() -> bool:
    try:
        p = subprocess.run([sys.executable, "-c", "import loguru, kernelblaster.knowledge.v2.merge"], capture_output=True,
                           env=dict(os.environ, PYTHONPATH=str(KB / "src")), timeout=120)
        return p.returncode == 0 and FIX.exists() and PLUGIN.exists()
    except Exception:  # noqa: BLE001
        return False


KB_OK = _kb_ok()


def mkcfg(tmp: Path, **kw) -> Config:
    base = dict(kb=KB, model="stub:resume", plugins=[str(PLUGIN)], python=sys.executable, arch="89", rollouts=1, steps=4,
                kb_args=["--evaluator", "plugin", "--stop-grace", "30"], stop_patience=0, trace="off")
    base.update(kw)
    return Config(**base)


def gkrec(**kw) -> dict:
    return {"mode": "read", "applied": True, "snapshot": None, "part": "cuda:sm_89|fp16", "problem": "b", **kw}


class FlagOffTest(unittest.TestCase):
    def test_default_is_off_and_changes_nothing(self):
        cfg = Config(kb=KB, arch="89")
        self.assertEqual((cfg.knowledge_global_halfway, cfg.knowledge_global_halfway_wait_s), ("off", 180.0))
        self.assertEqual(kb_halfway.key_fields(cfg, gkrec()), {})
        self.assertEqual(kb_halfway.key_fields(cfg, None), {})
        k0 = {**kb_resume.round_key(cfg, []), **kb_global.key_fields(cfg, gkrec())}
        self.assertNotIn("knowledge_global_halfway", k0)
        a = KBRunner(cfg, Path("/tmp/k")).command([Path("/p")], Path("/o"))
        r = KBRunner(cfg, Path("/tmp/k"))
        self.assertIsNone(r.halfway)
        self.assertEqual(r.command([Path("/p")], Path("/o")), a)
        self.assertNotIn("--pause-spec", a)

    def test_on_without_seed_refresh_or_kb_support_is_inactive(self):
        cfg = Config(kb=KB, arch="89", knowledge_global_halfway="on")
        self.assertFalse(kb_halfway.enabled(cfg, gkrec()))                               # no refresh inbox
        cfg = Config(kb=KB, arch="89", knowledge_global_halfway="on", knowledge_global_refresh="/tmp/inbox")
        self.assertFalse(kb_halfway.enabled(cfg, None))                                  # no global knowledge
        self.assertFalse(kb_halfway.enabled(cfg, gkrec(applied=False)))                  # no seed applied
        self.assertTrue(kb_halfway.enabled(cfg, gkrec()) == (KB / "scripts" / "run_multislot.py").read_text()
                        .__contains__("--pause-spec"))
        old = Config(kb=Path("/nonexistent"), knowledge_global_halfway="on", knowledge_global_refresh="/x")
        self.assertFalse(kb_halfway.enabled(old, gkrec()))                               # a KernelBlaster without the hook

    @unittest.skipUnless(KB_OK and kb_halfway.kb_supports_pause(KB), "needs the KernelBlaster pause hook")
    def test_on_adds_one_constant_marker_to_the_round_key(self):
        off = Config(kb=KB, arch="89", knowledge_global_refresh="/tmp/inbox")
        on = Config(kb=KB, arch="89", knowledge_global_refresh="/tmp/inbox", knowledge_global_halfway="on",
                    knowledge_global_halfway_wait_s=7.0)                                # the wait is not part of the key
        k_off = {**kb_global.key_fields(off, gkrec()), **kb_halfway.key_fields(off, gkrec())}
        k_on = {**kb_global.key_fields(on, gkrec()), **kb_halfway.key_fields(on, gkrec())}
        self.assertEqual(k_on, {**k_off, "knowledge_global_halfway": True})
        self.assertEqual(kb_resume.key_hash(k_on), kb_resume.key_hash({**kb_global.key_fields(on, gkrec(snapshot=None)),
                                                                          "knowledge_global_halfway": True}))

    def test_pause_step_rule(self):
        self.assertEqual([kb_halfway.pause_at(n) for n in (1, 2, 3, 4, 5, 6)], [None, 1, 2, 2, 3, 3])


class _RunBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mb_halfway_"))
        os.environ["MB_HALFWAY_POLL_S"] = "0.05"
        os.environ["KB_TEST_LLM_LOG"] = str(self.tmp / "llm.log")
        os.environ["KB_TEST_EVAL_LOG"] = str(self.tmp / "eval.log")
        for k in ("KB_FAULT", "KB_TEST_SLOW"):
            os.environ.pop(k, None)

    def tearDown(self):
        for k in ("MB_HALFWAY_POLL_S", "KB_TEST_LLM_LOG", "KB_TEST_EVAL_LOG"):
            os.environ.pop(k, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    # -- helpers
    def llm_calls(self) -> int:
        f = self.tmp / "llm.log"
        return len(f.read_text().splitlines()) if f.exists() else 0

    def setup_run(self, name: str, wait_s: float = 30.0, **cfgkw):
        w = self.tmp / name
        (w / "knowledge").mkdir(parents=True)
        inbox = self.tmp / f"{name}_inbox"
        inbox.mkdir()
        cfg = mkcfg(self.tmp, knowledge_global_halfway="on", knowledge_global_halfway_wait_s=wait_s,
                    knowledge_global_refresh=str(inbox), **cfgkw)
        state = {"sent": [], "rounds": [], "knowledge_global": gkrec(problem=name)}
        sp = w / "state.json"
        kb_resume.atomic_write_json(sp, state)
        kb = KBRunner(cfg, w / "knowledge")
        kb.halfway = kb_halfway.Halfway(cfg, w, state, sp, state["knowledge_global"])
        return w, inbox, cfg, state, kb

    def reference(self, name="ref", **cfgkw):
        w = self.tmp / name
        (w / "knowledge").mkdir(parents=True)
        kb = KBRunner(mkcfg(self.tmp, **cfgkw), w / "knowledge")
        n0 = self.llm_calls()
        res = kb.run([FIX], w / "r00" / "run")
        return res, self.llm_calls() - n0, w / "r00" / "run"

    def run_halfway(self, kb, w, controller=None, rnd=0):
        out = w / "r00" / "run"
        kb.halfway.arm(rnd, out, [PROB])
        t = None
        if controller is not None:
            t = threading.Thread(target=controller, args=(w, out), daemon=True)
            t.start()
        res = kb.run([FIX], out)
        if t is not None:
            t.join(10)
        return res, out

    @staticmethod
    def wait_paused(w, timeout=60.0):
        f = w / "r00" / "halfway.json"
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                d = json.loads(f.read_text())
                if d.get("stage") == "paused":
                    return d
            except (OSError, ValueError):
                pass
            time.sleep(0.02)
        raise AssertionError("never paused")

    @staticmethod
    def sig(res):
        r = res[0]
        return (r["final_ok"], r["final_partition_us"], r["speedup"], r["llm_calls"])



@unittest.skipUnless(KB_OK and kb_halfway.kb_supports_pause(KB), "needs KernelBlaster (loguru, knowledge v2) with --pause-spec")
class HalfwayRunTest(_RunBase):
    # -- tests
    def test_pause_import_resume_equals_an_uninterrupted_run(self):
        ref, ref_calls, ref_out = self.reference()
        w, inbox, cfg, state, kb = self.setup_run("b")
        seen = {}

        def controller(w_, out):
            d = self.wait_paused(w_)
            seen["half"] = d
            seen["steps_at_pause"] = json.loads((out / PROB / "journal" / "state.json").read_text())["rollout_state"]
            kb_halfway.stub_ack(inbox, 0, package=None, status="up_to_date")

        n0 = self.llm_calls()
        res, out = self.run_halfway(kb, w, controller)
        calls = self.llm_calls() - n0
        self.assertEqual(self.sig(res), self.sig(ref))                                    # final model = no-halfway run
        self.assertEqual(calls, ref_calls)                                                 # no step / LLM call paid twice
        self.assertEqual(seen["half"]["round"], 0)
        self.assertEqual(seen["half"]["steps_done"], {PROB: 2})
        self.assertEqual(seen["steps_at_pause"]["0"]["next_step"], 2)                      # each rollout keeps its own state
        h = json.loads((w / "r00" / "halfway.json").read_text())
        self.assertEqual((h["stage"], h["outcome"]), ("done", "none"))
        t = state["knowledge_global"]["halfway"]
        self.assertEqual((t["n"], t["timeouts"], t["errors"]), (1, 0, 0))
        self.assertEqual(json.loads((w / "state.json").read_text())["knowledge_global"]["halfway"]["n"], 1)
        self.assertTrue((inbox / "done" / "halfway-r00.ack.json").exists())
        self.assertEqual(len(json.loads((out / PROB / "result.json").read_text())["steps"]), 4)
        self.assertTrue((out / "pause_spec.json").exists())                               # the first half carried the spec
        self.assertNotIn("--pause-spec", kb.commands[str(out)])                           # the second half did not
        self.assertEqual(json.loads((out / PROB / "result.json").read_text())["resume"]["segments"], 2)

    def test_timeout_resumes_without_the_package_and_records_it(self):
        ref, ref_calls, _ = self.reference()
        w, inbox, cfg, state, kb = self.setup_run("b", wait_s=0.3)
        t0 = time.time()
        res, out = self.run_halfway(kb, w)
        self.assertEqual(self.sig(res), self.sig(ref))
        t = state["knowledge_global"]["halfway"]
        self.assertEqual((t["n"], t["timeouts"]), (1, 1))
        self.assertTrue(t["halfway_timeout"])
        self.assertEqual(json.loads((w / "r00" / "halfway.json").read_text())["outcome"], "timeout")
        self.assertLess(t["wait_s"], 5.0)
        self.assertGreater(time.time() - t0, 0.3)

    def test_import_error_is_recorded_and_the_run_continues(self):
        ref, _, _ = self.reference()
        w, inbox, cfg, state, kb = self.setup_run("b")
        real = kb_global.refresh_at_round

        def boom(*a, **k):
            raise RuntimeError("import exploded")

        kb_global.refresh_at_round = boom
        try:
            res, out = self.run_halfway(kb, w, lambda w_, o: (self.wait_paused(w_), kb_halfway.stub_ack(inbox, 0, "x")))
        finally:
            kb_global.refresh_at_round = real
        self.assertEqual(self.sig(res), self.sig(ref))
        t = state["knowledge_global"]["halfway"]
        self.assertEqual((t["n"], t["errors"]), (1, 1))
        self.assertIn("import exploded", t["halfway_error"])

    def test_a_failed_import_status_is_an_error_not_a_failure(self):
        w, inbox, cfg, state, kb = self.setup_run("b")
        real = kb_global.refresh_at_round
        kb_global.refresh_at_round = lambda *a, **k: {"status": "refused", "reason": "lineage", "counts": {}}
        try:
            res, out = self.run_halfway(kb, w, lambda w_, o: (self.wait_paused(w_), kb_halfway.stub_ack(inbox, 0, "x")))
        finally:
            kb_global.refresh_at_round = real
        self.assertTrue(res[0]["final_ok"])
        self.assertEqual(state["knowledge_global"]["halfway"]["errors"], 1)

    def test_crash_during_the_pause_resumes_into_the_paused_state(self):
        ref, ref_calls, _ = self.reference()
        w, inbox, cfg, state, kb = self.setup_run("b")
        out = w / "r00" / "run"
        kb.halfway.arm(0, out, [PROB])

        def die(*a, **k):
            raise KeyboardInterrupt("killed during the pause")        # not an Exception: nothing catches it

        kb.halfway._wait_and_import = die
        n0 = self.llm_calls()
        with self.assertRaises(KeyboardInterrupt):
            kb.run([FIX], out)
        self.assertEqual(json.loads((w / "r00" / "halfway.json").read_text())["stage"], "paused")
        half_calls = self.llm_calls() - n0
        self.assertLess(half_calls, ref_calls)
        # a new process: fresh runner / Halfway on the same work dir; the controller's ack is already there
        kb_halfway.stub_ack(inbox, 0, None, status="up_to_date")
        state2 = json.loads((w / "state.json").read_text())
        kb2 = KBRunner(cfg, w / "knowledge")
        kb2.halfway = kb_halfway.Halfway(cfg, w, state2, w / "state.json", state2["knowledge_global"])
        kb2.halfway.arm(0, out, [PROB])
        self.assertEqual(kb2.halfway.mode(out), "paused")                                # no first half again
        res = kb2.run([FIX], out)
        self.assertEqual(self.sig(res), self.sig(ref))
        self.assertEqual(self.llm_calls() - n0, ref_calls)                               # the half was not redone
        self.assertEqual(kb2.halfway.mode(out), "normal")
        self.assertEqual(state2["knowledge_global"]["halfway"]["n"], 1)

    def test_a_killed_first_half_parks_at_the_same_step_after_a_restart(self):
        ref, ref_calls, _ = self.reference()
        w, inbox, cfg, state, kb = self.setup_run("b")
        out = w / "r00" / "run"
        kb.halfway.arm(0, out, [PROB])
        os.environ["KB_FAULT"] = "llm:2"                          # SIGKILL inside the 2nd LLM call of the first half
        try:
            with self.assertRaises(RuntimeError):
                kb.run([FIX], out)
        finally:
            os.environ.pop("KB_FAULT")
        self.assertFalse((w / "r00" / "halfway.json").exists())
        kb_halfway.stub_ack(inbox, 0, None)
        res, _ = self.run_halfway(kb, w)
        self.assertEqual(self.sig(res), self.sig(ref))
        self.assertEqual(json.loads((out / PROB / "result.json").read_text())["resume"]["segments"], 3)

    def test_unarmed_run_and_hint_runs_are_untouched(self):
        w, inbox, cfg, state, kb = self.setup_run("b")
        out = self.tmp / "other_run"
        res = kb.run([FIX], out)                                  # not armed (e.g. a hint decision wave): no pause
        self.assertTrue(res[0]["final_ok"])
        self.assertFalse((out / "paused.json").exists())
        self.assertNotIn("halfway", state["knowledge_global"])

    def test_problem_too_short_to_pause_runs_through(self):
        w, inbox, cfg, state, kb = self.setup_run("b", steps=1)
        res, out = self.run_halfway(kb, w)
        self.assertTrue(res[0]["final_ok"])
        self.assertNotIn("halfway", state["knowledge_global"])                        # nothing paused, nothing waited for

    def test_two_runs_exchange_knowledge_at_halfway(self):
        """Run A merged its first half into the store; run B's halfway package (built by a scripted controller with the
        real merge / refresh CLI) carries A's statistics, claims and the like; B's own work is otherwise untouched."""
        sys.path.insert(0, str(KB / "src"))
        from kernelblaster.knowledge.outcomes import OutcomeRecord
        from kernelblaster.knowledge.v2 import KnowledgeV2
        from kernelblaster.knowledge.v2.stats import path as stat_path
        store = str(self.tmp / "store")
        wa = self.tmp / "a"
        wa.mkdir()
        cfa = Config(kb=KB, arch="89", knowledge_global="readwrite", knowledge_global_store=store,
                     knowledge_global_problem="a", python=sys.executable)
        sa = {}
        ra = kb_global.apply_seed(wa, cfa, [self._ir()], sa)
        w, inbox, cfg, state, kb = self.setup_run("b")
        # b's seed: from the (empty) store, exactly as the farm's `seed`
        cseed = Config(kb=KB, arch="89", knowledge_global="read", knowledge_global_store=store,
                       knowledge_global_problem="b", python=sys.executable)
        shutil.rmtree(w / "knowledge")
        state.pop("knowledge_global")
        rb = kb_global.apply_seed(w, cseed, [self._ir()], state)
        self.assertTrue(rb["applied"])
        kb_resume.atomic_write_json(w / "state.json", state)
        # a's first half: three outcomes, a claim; merged (why round1h)
        kv = KnowledgeV2(wa / "knowledge")
        t = sorted(kv.ontology.techniques)[0]
        chain = [sorted(x.id for x in kv.hierarchy.states.values() if x.parent is None)[0]]
        ids = []
        for _ in range(3):
            rec = OutcomeRecord(technique=t, status="improved", target="cuda", arch="sm_89", problem_id="p", slot="s0",
                                op_class="gemm", dtype="fp16", speedup=1.2)
            kv.outcomes.append(rec, {"item_id": t, "reward": 0.2, "chain": chain})
            kv.stats.update(t, stat_path(kv.registry, target_key="cuda:sm_89", op_class="gemm", chain=chain,
                                         dtype="fp16"), 0.2)
            ids.append(rec.id)
        from kernelblaster.knowledge.claims import Claim
        kv.claims.add(Claim(kind="applicability", text=f"{t} helps small gemm on sm_89 (run a)", arch="sm_89",
                            op_class="gemm", technique=t, data={"effect": "helps"}), ids[:2])
        kv.claims.save()
        kv.db.conn.close()
        self.assertEqual(kb_global.merge_back(wa, cfa, ra)["status"], "merged")
        cb = Config(kb=KB, arch="89", knowledge_global="read", knowledge_global_store=store,
                    knowledge_global_problem="b", python=sys.executable, knowledge_global_halfway="on",
                    knowledge_global_refresh=str(inbox), knowledge_global_halfway_wait_s=60.0, rollouts=1, steps=4,
                    plugins=[str(PLUGIN)], model="stub:resume", kb_args=["--evaluator", "plugin", "--stop-grace", "30"],
                    stop_patience=0, trace="off")
        kb = KBRunner(cb, w / "knowledge")
        kb.halfway = kb_halfway.Halfway(cb, w, state, w / "state.json", state["knowledge_global"])
        rep = {}

        def controller(w_, out):
            d = self.wait_paused(w_)
            kd = w_ / "knowledge"
            before = sqlite3.connect(kd / "knowledge.db").execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
            rep["b_outcomes_before"] = before
            m = kb_global._merge_cli(cb, ["--store", store, "merge", "--run", str(kd), "--problem", "b", "--model", "stub",
                                          "--base", rb["seed"], "--label", "round1h b"])
            rep["merge"] = m
            pk = self.tmp / "pk"
            r = kb_global._merge_cli(cb, ["--store", store, "refresh", "--run-db", str(kd / "knowledge.db"),
                                          "--out", str(pk), "--problem", "b", "--part", rb["part"]])
            rep["refresh"] = r
            tb = inbox / f".{r['rid']}.part"
            with tarfile.open(tb, "w:gz") as tf:
                tf.add(pk, arcname=".")
            os.replace(tb, inbox / f"{r['rid']}.tar.gz")
            kb_halfway.stub_ack(inbox, 0, r["rid"], status="ready")

        res, out = self.run_halfway(kb, w, controller)
        self.assertTrue(res[0]["final_ok"], res)
        self.assertEqual(rep["merge"].get("status"), "merged", rep)
        self.assertEqual(rep["refresh"]["status"], "ready", rep)
        t = state["knowledge_global"]["halfway"]
        self.assertEqual((t["n"], t["timeouts"], t["errors"]), (1, 0, 0), t)
        self.assertGreater(t["imported_rows"], 0)
        ev = t["events"][0]
        self.assertEqual((ev["outcome"], ev["import_status"]), ("imported", "applied"))
        self.assertEqual(state["knowledge_global"]["refreshes"][-1]["at"], "halfway")
        c = sqlite3.connect(w / "knowledge" / "knowledge.db")
        texts = [json.loads(r[0]).get("text", "") for r in c.execute("SELECT data FROM claims")]
        c.close()
        self.assertTrue(any("(run a)" in x for x in texts), texts)                  # A's claim reached B before its 2nd half
        self.assertEqual(len(json.loads((out / PROB / "result.json").read_text())["steps"]), 4)

    def _ir(self) -> Path:
        d = self.tmp / "ir"
        d.mkdir(exist_ok=True)
        (d / "graph.json").write_text(json.dumps({"tensors": {"x": {"dtype": "f16"}, "w": {"dtype": "f32"}}}))
        return d


@unittest.skipUnless(KB_OK and kb_halfway.kb_supports_pause(KB), "needs KernelBlaster (loguru, knowledge v2) with --pause-spec")
class InterleavedHalfwayTest(_RunBase):
    """A4 interleaved graph search: the hint decisions run INSIDE the plan's KernelBlaster run; the halfway pause must
    not break them (hint problems are not paused, decisions are made once, an acceptance's ``mb_stop.json`` still stops
    the absorbed plan problem -- after the pause too)."""

    def _problem(self, name, **budget):
        d = self.tmp / "probs" / name
        shutil.copytree(FIX, d)
        m = json.loads((d / "manifest.json").read_text())
        m["name"] = name
        if budget:
            m["mb_budget"] = budget
        (d / "manifest.json").write_text(json.dumps(m))
        return d

    def _engine(self, kb, w, accept, scope):
        from modelblaster.pipeline import graph_search as gsm
        from modelblaster.pipeline import kb_graph_opt as go
        from modelblaster.pipeline.tests.test_graph_search import _Prog, _cfg
        run_dir = w / "r00" / "run"
        gs = gsm.GraphSearch(_cfg(graph_search="interleaved"), self.tmp, [])
        self.decided = []

        def measure(j, ro, rv, attempts=None):
            f = run_dir / "paused.json"                      # decide only once the plan problem is parked
            t0 = time.time()
            while not f.exists() and time.time() - t0 < 60:
                time.sleep(0.02)
            return [[10.0], [9.0]]

        def finish(j, ro, rv, tm):
            self.decided.append(j["hint"]["id"])
            st = "accepted" if accept else "rejected"
            return {"hint": j["hint"]["id"], "status": st, "hint_status": st}
        return gsm.Interleaved(gs, run_dir=run_dir, prog=_Prog(), start=lambda probs: kb.start(probs, run_dir, parallel_problems=3),
                               read=lambda d: go.read_result(run_dir, Path(d)), measure=measure, finish=finish,
                               collect=lambda probs, proc: kb.collect([Path(x) for x in probs], run_dir, proc),
                               scope_after=lambda j: scope, save_hints=None, poll_s=0.02, workers=2)

    def _run(self, accept, scope, pre_stop=False):
        w, inbox, cfg, state, kb = self.setup_run("b", steps=4)
        plan = self._problem("plan_r00")
        ho, hv = self._problem("fuse_A__orig", rollouts=1, steps=1, stop_patience=0), \
            self._problem("fuse_A__variant", rollouts=1, steps=1, stop_patience=0)
        run_dir = w / "r00" / "run"
        kb.halfway.arm(0, run_dir, ["plan_r00"])
        if pre_stop:
            (run_dir / "plan_r00").mkdir(parents=True)
            (run_dir / "plan_r00" / "mb_stop.json").write_text(json.dumps({"by": "test"}))
        job = {"hint": {"id": "fuse:A"}, "orig": ho, "variant": hv, "ev": {"ir": "lenet"}, "names": [],
               "gs": {"rung": 0, "steps": 1, "used": 0}}
        eng = self._engine(kb, w, accept, scope)
        if pre_stop:
            eng.measure = lambda j, ro, rv, attempts=None: [[10.0], [9.0]]
        threading.Thread(target=lambda: (self.wait_paused(w), kb_halfway.stub_ack(inbox, 0, None)), daemon=True).start() \
            if not pre_stop else None
        out = eng.run([plan], ["lenet"], [job])
        return w, state, out, run_dir

    def test_decision_inside_the_run_and_no_stop_means_the_plan_completes(self):
        w, state, out, run_dir = self._run(False, set())
        self.assertEqual(self.decided, ["fuse:A"])                                   # decided once
        self.assertEqual([d["status"] for d in out["decisions"]], ["rejected"])
        self.assertEqual(state["knowledge_global"]["halfway"]["n"], 1)
        plan = json.loads((run_dir / "plan_r00" / "result.json").read_text())
        self.assertEqual(len(plan["steps"]), 4)                                      # both halves ran
        hint = json.loads((run_dir / "fuse_A__orig" / "result.json").read_text())
        self.assertEqual(len(hint["steps"]), 1)                                      # the hint problems were never paused
        self.assertEqual(json.loads((run_dir / "paused.json").read_text())["problems"].keys(), {"plan_r00"})

    def test_acceptance_after_the_pause_stops_the_absorbed_plan_problem(self):
        w, state, out, run_dir = self._run(True, set(json.loads((FIX / "manifest.json").read_text())["partition"]["op_names"]))
        self.assertEqual(self.decided, ["fuse:A"])
        self.assertEqual(out["stopped"], ["plan_r00"])
        self.assertTrue((run_dir / "plan_r00" / "mb_stop.json").exists())
        plan = json.loads((run_dir / "plan_r00" / "result.json").read_text())
        self.assertEqual(len(plan["steps"]), 2)                                      # A4 semantics: no new steps after the stop
        self.assertEqual(plan["efficiency"]["stopped"], "external")
        self.assertTrue(plan["final_ok"])
        self.assertEqual(state["knowledge_global"]["halfway"]["n"], 1)

    def test_a_plan_problem_already_stopped_never_pauses(self):
        w, state, out, run_dir = self._run(False, set(), pre_stop=True)
        self.assertEqual(self.decided, ["fuse:A"])
        self.assertNotIn("halfway", state["knowledge_global"])                       # nothing to sync: the problem just stopped
        self.assertFalse((run_dir / "paused.json").exists())


if __name__ == "__main__":
    unittest.main()
