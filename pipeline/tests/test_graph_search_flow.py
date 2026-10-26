"""GPU integration of the graph search redesign (graph_search.py), LLM-free: the tiny conv -> bn -> relu (x2) model
and the scripted ``stub:`` KernelBlaster LLM of test_kb_graph_opt_flow (real fused kernels for fused slots).

- interleaved: the hint decisions run inside the plan's KernelBlaster run (rung budgets in the problems'
  manifests), are decided mid-run, the conv->bn(->relu) chain is accepted, the final model verifies, every
  decision is in graph_decisions.jsonl and round.json carries the graph_search record;
- pre-apply: with a prior table that is confident about bn+relu, the fusion is accepted before round 0's plan
  (no measurement) and the round-0 plan already holds the fused op;
- beam: two members (baseline, + the proposed fusions) share one plan, are re-timed back to back at the round's
  end, and the faster one's hints are committed.

Needs torch, nvcc, a CUDA GPU and a KernelBlaster checkout (as test_kb_graph_opt_flow); skipped otherwise.
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

from modelblaster.pipeline.tests import test_kb_graph_opt_flow as F  # noqa: E402


@unittest.skipIf(F.SKIP is not None, F.SKIP or "")
class GraphSearchFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(os.environ.get("MB_TEST_TMP", tempfile.gettempdir()))
        base.mkdir(parents=True, exist_ok=True)
        cls.tmp = Path(tempfile.mkdtemp(prefix="mb_gsflow_", dir=base))
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
        base = dict(kb=F.KB, model="stub:graphopt", plugins=[str(self.tmp / "plugin.py")], python=F.KB_PY,
                    arch=self.arch, rounds=4, rollouts=1, steps=2, variant_steps=2, max_fix=1, top_k=2,
                    max_variants=2, remeasure=3, profile_runs=1, accept_gain=-1.0, min_gain_pct=0.0)
        base.update(kw)
        return Config(**base)

    def _events(self, work):
        f = work / "graph_decisions.jsonl"
        return [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []

    def test_interleaved_fuses_the_chain_mid_round(self):
        from modelblaster.pipeline import graph_hints as gh
        from modelblaster.pipeline.kb_graph_opt import run
        work = self.tmp / "inter"
        s = run([self.tmp / "ir"], work, self._cfg(graph_search="interleaved", graph_priors="off",
                                                   graph_characterize=1, graph_char_top=2))
        doc = gh.load(work / "graph_hints.json")
        status = {h["id"]: h["status"] for h in doc["hints"]}
        self.assertTrue(any(v == "accepted" and k.startswith("fuse:") for k, v in status.items()), status)
        ir = next(iter(s["irs"].values()))
        self.assertTrue(ir["passed"], s)
        self.assertIn("graph_search", s)
        ev = self._events(work)
        kinds = [e["event"] for e in ev]
        self.assertIn("characterize", kinds)
        self.assertIn("interleaved_start", kinds)
        dec = [e for e in ev if e["event"] == "decision"]
        self.assertTrue(dec and all("t_run_s" in e and "steps" in e for e in dec), dec)
        rounds = json.loads((work / "state.json").read_text())["rounds"]
        self.assertIn("graph_search", rounds[0])
        # the hint problems ran inside the plan's run with their rung budget
        man = [json.loads(p.read_text()) for p in (work / "r00" / "variants").glob("*/variant/manifest.json")]
        self.assertTrue(man and all(m.get("mb_budget", {}).get("stop_patience") == 0 for m in man), man)
        self.assertTrue(any((work / "r00" / "run").glob("*__variant/result.json")))

    def test_preapply_from_a_confident_prior(self):
        from modelblaster.pipeline import graph_hints as gh
        from modelblaster.pipeline import graph_priors as gp
        from modelblaster.pipeline.kb_graph_opt import run
        work = self.tmp / "pre"
        tab = {"format": "mb-graph-priors/1", "part": f"cuda:sm_{self.arch}|fp32", "source": "test",
               "counts": {"batchnorm2d+relu": [9, 0], "conv2d+batchnorm2d": [2, 2]}}
        old = gp.BOOTSTRAP
        f = self.tmp / "priors.json"
        f.write_text(json.dumps(tab))
        gp.BOOTSTRAP = f
        try:
            run([self.tmp / "ir"], work, self._cfg(rounds=1, graph_search="interleaved", graph_explore=0.0,
                                                   graph_characterize=0))
        finally:
            gp.BOOTSTRAP = old
        doc = gh.load(work / "graph_hints.json")
        h = gh.get(doc, "fuse:fused__batchnorm2d__relu")
        self.assertIsNotNone(h, [x["id"] for x in doc["hints"]])
        self.assertEqual(h["status"], "accepted")
        self.assertEqual(h["evidence"][0]["by"], "graph-prior")
        plan = json.loads((work / "r00" / "plan.json").read_text())
        self.assertTrue(any(it["op"].startswith("fused__") for it in plan["items"]), [it["op"] for it in plan["items"]])

    def test_beam_commits_one_member(self):
        from modelblaster.pipeline import graph_hints as gh
        from modelblaster.pipeline.kb_graph_opt import run
        work = self.tmp / "beam"
        s = run([self.tmp / "ir"], work, self._cfg(rounds=2, graph_diverse_beam=3, graph_priors="off"))
        st = json.loads((work / "graph_search_state.json").read_text())["beam"]
        self.assertTrue(st.get("done"), st)
        ev = [e for e in self._events(work) if e["event"] == "beam_select"]
        self.assertTrue(ev and all(v is not None for v in ev[0]["times"].values()), ev)
        self.assertTrue(next(iter(s["irs"].values()))["passed"], s)
        doc = gh.load(work / "graph_hints.json")
        self.assertTrue(any(e.get("by") == "graph-beam" for h in doc["hints"] for e in h.get("evidence", [])))


if __name__ == "__main__":
    unittest.main()
