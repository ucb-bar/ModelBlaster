"""Telemetry in the graph optimiser (``kb_telemetry`` + ``kb_graph_opt``).

- the adapter: KernelBlaster's telemetry package loaded from the checkout,
  the no-op without it or with ``KB_TELEMETRY=0``; sections, worker-thread
  spans, reading a KernelBlaster run's telemetry back, the roll-up;
- the decide phase of a round (fake KernelBlaster, fake re-timing, as in
  test_kb_graph_opt_parallel): ModelBlaster's spans (prepare, kb_wait,
  remeasure in the worker threads, finish) cover the round's wall clock and
  the KernelBlaster runs' tokens are summed.

Needs a KernelBlaster checkout with ``src/kernelblaster/telemetry``
(``MB_KB_ROOT``, default ``third_party/KernelBlaster``); torch (CPU) for the
decide test.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from modelblaster.pipeline import kb_graph_opt as go  # noqa: E402
from modelblaster.pipeline import kb_telemetry as kbt  # noqa: E402

KB = Path(os.environ.get("MB_KB_ROOT", ROOT / "third_party" / "KernelBlaster"))
HAS_TEL = (KB / "src" / "kernelblaster" / "telemetry" / "__init__.py").exists()


def fake_kb_summary(calls=2, wins=1, trunc=1):
    """A KernelBlaster run-level telemetry summary (the shape run_multislot writes)."""
    tot = {"calls": calls, "errors": 0, "input_tokens": 1000 * calls, "uncached_input_tokens": 1000 * calls,
           "cache_read_input_tokens": 0, "cache_write_input_tokens": 0, "output_tokens": 200 * calls,
           "cost_usd": 0.0, "est_usd": 0.002 * calls, "truncated": trunc, "latency_s": 3.0 * calls,
           "server_latency_s": 0, "ttft_s": 0, "provider_retries": 0, "backoff_s": 0, "backoffs": 0,
           "latency_max_s": 3.0}
    return {"wall_s": 10.0, "time_s": {"llm.codegen_simple": 6.0, "build.nvcc": 3.0, "step": 1.0},
            "busy_s": {}, "counters": {"step": 2, "win": wins}, "gpu_busy_s": {"gpu0": 2.0},
            "evals": {"candidate": {"n": 2, "s": 3.0, "failed": 0}},
            "llm": {"total": tot, "by_role_model": [{"role": "codegen_simple", "model": "m", **tot}]}}


@unittest.skipUnless(HAS_TEL, "needs a KernelBlaster checkout with telemetry (MB_KB_ROOT)")
class AdapterTest(unittest.TestCase):
    def test_load_and_off(self):
        t = kbt.load(KB)
        self.assertTrue(getattr(t, "available", False))
        self.assertIs(kbt.get(), t)
        os.environ["KB_TELEMETRY"] = "0"
        try:
            self.assertIs(kbt.load(KB), kbt._OFF)
        finally:
            del os.environ["KB_TELEMETRY"]
        self.assertIs(kbt.load(Path("/nonexistent")), t)          # already loaded: reused
        off = kbt._OFF
        with off.span("x", collect=True) as sp:
            off.lap("y")
        self.assertIsNone(sp)
        self.assertIsNone(off.full_summary(sp))

    def test_sections_threads_and_rollup(self):
        t = kbt.load(KB)
        with t.span("mb.round", collect=True, laps=True) as rsp:
            t.lap("mb.decide")
            parent = t.current()
            wait = kbt.section("mb.kb_wait", kind="hints")
            with ThreadPoolExecutor(2) as ex:
                futs = [ex.submit(kbt.run_within, parent, "mb.remeasure", threading.Event().wait, 0.05)
                        for _ in range(2)]
                [f.result() for f in futs]
            wait.end()
            t.end_laps(rsp)
        s = t.full_summary(rsp)
        for k in ("mb.decide", "mb.kb_wait", "mb.remeasure"):
            self.assertIn(k, s["busy_s"])
        self.assertAlmostEqual(s["busy_s"]["mb.remeasure"], 0.1, delta=0.04)   # two threads
        self.assertAlmostEqual(sum(s["time_s"].values()), s["wall_s"], delta=0.01)
        r = kbt.kb_rollup([fake_kb_summary(), None, fake_kb_summary(calls=3, trunc=0)])
        self.assertEqual(r["runs"], 2)
        self.assertEqual(r["llm"]["total"]["calls"], 5)
        self.assertEqual(r["llm"]["by_role_model"][0]["calls"], 5)
        self.assertEqual(r["derived"]["accepted_wins"], 2)
        self.assertAlmostEqual(r["derived"]["usd"], 0.01)
        self.assertEqual(r["time_by_category_s"], {"llm_wait": 12.0, "compile": 6.0, "agent": 2.0})

    def test_kb_run_summary_reads_the_run_level_entry(self):
        d = Path(tempfile.mkdtemp())
        self.assertIsNone(kbt.kb_run_summary(d))
        (d / "summary.json").write_text(json.dumps([
            {"problem": "p", "telemetry": {"wall_s": 1.0}}, {"load_balance": {}}, {"telemetry": fake_kb_summary()}]))
        self.assertEqual(kbt.kb_run_summary(d)["wall_s"], 10.0)


@unittest.skipUnless(HAS_TEL and importlib.util.find_spec("torch") is not None, "needs telemetry + torch")
class DecideTelemetryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from modelblaster.pipeline.tests import test_kb_graph_opt_parallel as par
        cls.par = par
        base = Path(os.environ.get("MB_TEST_TMP", tempfile.gettempdir()))
        base.mkdir(parents=True, exist_ok=True)
        cls.tmp = Path(tempfile.mkdtemp(prefix="mb_tel_", dir=base))
        (cls.tmp / "m.py").write_text(par.MODEL)
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT.parent), os.environ.get("PYTHONPATH", "")])}
        cp = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file",
                             str(cls.tmp / "m.py"), "--out-dir", str(cls.tmp / "ir"), "--quant", "fp32"],
                            cwd=ROOT, capture_output=True, text=True, timeout=600, env=env)
        assert cp.returncode == 0, cp.stderr[-2000:]
        from modelblaster.pipeline import graph_hints as gh
        cls.graph = gh.unique_names(json.loads((cls.tmp / "ir" / "graph.json").read_text()))
        cls.ir = cls.graph["name"]

    @classmethod
    def tearDownClass(cls):
        if not os.environ.get("MB_TEST_KEEP"):
            import shutil
            shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_decide_spans_and_kb_rollup(self):
        par = self.par
        from modelblaster.pipeline import graph_hints as gh
        from modelblaster.pipeline.kb_ingest import CudaCache
        from modelblaster.pipeline.kb_proposals import _all_specs

        class TelKB(par.FakeKB):
            def collect(self, problems, out, proc):
                proc.wait()
                (out / "summary.json").write_text(json.dumps([{"telemetry": fake_kb_summary()}]))
                return super().collect(problems, out, proc)

        t = kbt.load(KB)
        work = self.tmp / "w"
        cfg = go.Config(kb=KB, arch="75", remeasure=3, accept_gain=0.0, parallel_hints=8)
        kb = TelKB(cfg, work / "knowledge")
        doc = gh.new_doc()
        hs = par.hints_for(self.ir)
        gh.merge(doc, hs)
        prof = {"ir": self.ir, "ir_dir": str(self.tmp / "ir"), "source_ir": str(self.tmp / "ir")}
        orig, go.remeasure = go.remeasure, par.fake_remeasure
        try:
            with t.span("mb.round", collect=True, laps=True) as rsp:
                t.lap("mb.decide")
                waves, _ = go.decide_hints([(h, prof) for h in hs], {self.ir: self.graph}, work / "variants", 8,
                                           kb=kb, cache=CudaCache(work / "cache"), doc=doc, cfg=cfg,
                                           specs=dict(_all_specs()), leases=go.GpuLeases([]))
                t.end_laps(rsp)
        finally:
            go.remeasure = orig
        s = t.full_summary(rsp)
        for k in ("mb.prepare", "mb.kb_wait", "mb.remeasure", "mb.finish"):
            self.assertIn(k, s["busy_s"], s["busy_s"])
        self.assertAlmostEqual(sum(s["time_s"].values()), s["wall_s"], delta=0.02)
        self.assertEqual(len(kb.telemetry), len(waves))                    # one per KernelBlaster run
        r = kbt.kb_rollup(kb.telemetry)
        self.assertEqual(r["runs"], len(waves))
        self.assertEqual(r["llm"]["total"]["calls"], 2 * len(waves))
        self.assertEqual(r["llm"]["total"]["truncated"], len(waves))


if __name__ == "__main__":
    unittest.main()
