"""Trace store in the graph optimiser (``kb_trace`` + ``kb_graph_opt``).

- the adapter: KernelBlaster's trace package loaded from the checkout, the
  no-op without it or with ``KB_TRACE=off``; ModelBlaster's build / driver
  output hooks (``kb_export``) and records;
- the decide phase of a round (fake KernelBlaster, fake re-timing, as in
  test_kb_graph_opt_parallel): one ``mb.hint`` record per hint with both
  problems' graphs / manifests and the decision, one ``mb.kb_run`` per run;
- the whole flow on the GPU (as test_kb_graph_opt_flow, two rounds): every
  ModelBlaster step is recorded and each KernelBlaster run's own trace (its
  steps' prompts, candidates, builds) is linked and readable.

Needs a KernelBlaster checkout with ``src/kernelblaster/trace`` (``MB_KB_ROOT``,
default ``third_party/KernelBlaster``); torch (CPU) for the decide test; nvcc,
a GPU and KernelBlaster's dependencies for the flow test.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from modelblaster.pipeline import kb_graph_opt as go  # noqa: E402
from modelblaster.pipeline import kb_trace  # noqa: E402

KB = Path(os.environ.get("MB_KB_ROOT", ROOT / "third_party" / "KernelBlaster"))
HAS_TRACE = (KB / "src" / "kernelblaster" / "trace" / "__init__.py").exists()


def _tmp(prefix):
    base = Path(os.environ.get("MB_TEST_TMP", tempfile.gettempdir()))
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=base))


def _recs(store: Path) -> list[dict]:
    return [json.loads(l) for l in (store / "index.jsonl").read_text().splitlines() if l.strip()]


@unittest.skipUnless(HAS_TRACE, "needs a KernelBlaster checkout with the trace store (MB_KB_ROOT)")
class AdapterTest(unittest.TestCase):
    def tearDown(self):
        kb_trace.close()

    def test_off_and_noop(self):
        os.environ["KB_TRACE"] = "off"
        try:
            work = _tmp("mb_trace_off_")
            self.assertIsNone(kb_trace.configure(KB, work))
            self.assertFalse((work / "trace").exists())
        finally:
            del os.environ["KB_TRACE"]
        self.assertIs(kb_trace.get(), kb_trace._OFF)
        kb_trace.record("plan", x=1)                       # no store: nothing, no error
        with kb_trace.get().scope(mb_stage="x") as d:
            self.assertEqual(d, {})
        if "_mb_kb_trace" not in sys.modules:             # a checkout without the package: the no-op
            self.assertIsNone(kb_trace.configure(Path("/nonexistent"), _tmp("mb_trace_nokb_")))

    def test_hooks_and_records(self):
        from modelblaster.pipeline import kb_export
        work = _tmp("mb_trace_")
        os.environ["MB_TEST_SECRET_TOKEN"] = "s3cr3t-value-0123456789"
        try:
            st = kb_trace.configure(KB, work, "full")
            self.assertIsNotNone(st)
            t = kb_trace.get()
            with t.scope(mb_stage="profile", ir="tiny"):
                kb_export.parse_driver_output("MB_PARTITION_US 4.0\npassed\n")
                rc, _ = kb_export._nvcc([sys.executable, "-c", "print('nvcc says s3cr3t-value-0123456789')"], 60)
            (work / "plan.json").write_text('{"problems": []}')
            kb_trace.record("plan", mb_stage="plan", files=kb_trace.files({"plan.json": work / "plan.json",
                                                                          "missing": work / "nope"}))
            summ = kb_trace.close()
        finally:
            del os.environ["MB_TEST_SECRET_TOKEN"]
        self.assertEqual(rc, 0)
        self.assertGreaterEqual(summ["records"], 3)
        recs = {r["t"]: r for r in _recs(work / "trace")}
        self.assertEqual(recs["driver"]["mb_stage"], "profile")
        self.assertEqual(recs["driver"]["ir"], "tiny")
        self.assertIn("***", recs["build"]["output"]["text"])
        self.assertNotIn("s3cr3t", (work / "trace" / "index.jsonl").read_text())
        self.assertEqual(list(recs["mb.plan"]["files"]), ["plan.json"])
        self.assertTrue(kb_trace.kb_supports_trace(KB))


@unittest.skipUnless(HAS_TRACE and importlib.util.find_spec("torch") is not None, "needs the trace store + torch")
class DecideTraceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from modelblaster.pipeline.tests import test_kb_graph_opt_parallel as par
        cls.par = par
        cls.tmp = _tmp("mb_trace_decide_")
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
        kb_trace.close()
        if not os.environ.get("MB_TEST_KEEP"):
            import shutil
            shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_hint_and_kb_run_records(self):
        par = self.par
        from modelblaster.pipeline import graph_hints as gh
        from modelblaster.pipeline.kb_ingest import CudaCache
        from modelblaster.pipeline.kb_proposals import _all_specs

        work = self.tmp / "w"
        kb_trace.configure(KB, work, "full")
        cfg = go.Config(kb=KB, arch="75", remeasure=3, accept_gain=0.0, parallel_hints=8)
        kb = par.FakeKB(cfg, work / "knowledge")
        doc = gh.new_doc()
        hs = par.hints_for(self.ir)
        gh.merge(doc, hs)
        prof = {"ir": self.ir, "ir_dir": str(self.tmp / "ir"), "source_ir": str(self.tmp / "ir")}
        orig, go.remeasure = go.remeasure, par.fake_remeasure
        try:
            waves, dec = go.decide_hints([(h, prof) for h in hs], {self.ir: self.graph}, work / "variants", 8,
                                         kb=kb, cache=CudaCache(work / "cache"), doc=doc, cfg=cfg,
                                         specs=dict(_all_specs()), leases=go.GpuLeases([]))
        finally:
            go.remeasure = orig
            kb_trace.close()
        recs = _recs(work / "trace")
        hints = {r["hint"]: r for r in recs if r["t"] == "mb.hint"}
        self.assertEqual(set(hints), {h["id"] for h in hs})
        for d in dec:
            r = hints[d["hint"]]
            self.assertEqual(r["decision"], d["status"])
            if "orig_ir/graph.json" in (r.get("files") or {}):
                self.assertIn("variant/manifest.json", r["files"])
                self.assertEqual(len(r["remeasured"][0]), 3)
        runs = [r for r in recs if r["t"] == "mb.kb_run"]
        self.assertEqual(len(runs), len(waves))
        self.assertTrue(all(r["rc"] == 0 and r["problems"] for r in runs))


def _flow_skip():
    from modelblaster.pipeline.tests import test_kb_graph_opt_flow as flow
    if not HAS_TRACE:
        return "needs the trace store in the KernelBlaster checkout"
    return flow.SKIP


FLOW_SKIP = _flow_skip()


@unittest.skipIf(FLOW_SKIP is not None, FLOW_SKIP or "")
class FlowTraceTest(unittest.TestCase):
    def test_flow_records_every_step_and_links_kb_traces(self):
        from modelblaster.pipeline.tests import test_kb_graph_opt_flow as flow
        tmp = _tmp("mb_trace_flow_")
        (tmp / "tiny.py").write_text(flow.MODEL)
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT.parent), os.environ.get("PYTHONPATH", "")])}
        cp = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file",
                             str(tmp / "tiny.py"), "--out-dir", str(tmp / "ir"), "--quant", "fp32"],
                            cwd=ROOT, capture_output=True, text=True, timeout=600, env=env)
        self.assertEqual(cp.returncode, 0, cp.stderr[-2000:])
        (tmp / "plugin.py").write_text(flow.PLUGIN.format(fused=json.dumps(flow.fused_sources())))
        cfg = go.Config(kb=KB, model="stub:graphopt", plugins=[str(tmp / "plugin.py")], python=flow.KB_PY,
                        arch=flow._arch(), rounds=2, rollouts=1, steps=1, variant_steps=1, max_fix=1, top_k=2,
                        max_variants=1, remeasure=2, profile_runs=1, accept_gain=-1.0, trace="full")
        work = tmp / "work"
        go.run([tmp / "ir"], work, cfg)
        recs = _recs(work / "trace")
        kinds = {r["t"] for r in recs}
        for k in ("mb.profile", "mb.plan", "mb.export", "mb.kb_run", "mb.hint", "mb.verify", "mb.model_run",
                  "driver", "build"):
            self.assertIn(k, kinds)
        prof = next(r for r in recs if r["t"] == "mb.profile")
        self.assertEqual(prof["mb_round"], 0)
        self.assertIn("profile.json", prof["files"])
        self.assertTrue(any(r["t"] == "driver" and r.get("mb_stage") == "profile" for r in recs))
        runs = [r for r in recs if r["t"] == "mb.kb_run"]
        self.assertTrue(all(r["kb_trace"] and "--trace" in r["cmd"] for r in runs), runs)
        kbr = [json.loads(l) for l in (Path(runs[0]["kb_trace"]) / "index.jsonl").read_text().splitlines()]
        self.assertTrue(any(r["t"] == "llm" for r in kbr) and any(r["t"] == "step" for r in kbr))
        # the viewer reads the work dir: MB's records and every KernelBlaster run's steps
        view = [sys.executable, str(KB / "scripts" / "trace_view.py"), str(work)]
        mb = subprocess.run(view + ["--mb"], capture_output=True, text=True, timeout=120)
        self.assertEqual(mb.returncode, 0, mb.stderr)
        self.assertIn("[mb.hint]", mb.stdout)
        ls = subprocess.run(view + ["--list"], capture_output=True, text=True, timeout=120)
        self.assertEqual(ls.returncode, 0, ls.stderr)
        self.assertIn("technique", ls.stdout)
        if not os.environ.get("MB_TEST_KEEP"):
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
