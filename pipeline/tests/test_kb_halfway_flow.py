"""GPU integration of the halfway sync (kb_halfway.py) through the whole flow: the tiny conv -> bn -> relu (x2) model, the scripted
``stub:graphopt`` KernelBlaster LLM of test_kb_graph_opt_flow, a global knowledge seed from a fresh local store and a scripted
controller (a thread that answers every parked round with an ack and no package). The flow with ``--knowledge-global-halfway
on`` must make the same graph decisions and verify the same final model as the flow without it, in both the rounds flow and the
A4 interleaved graph search (hint decisions run inside the plan's KernelBlaster run), and every round's plan run must have
paused once at half its steps.

Needs torch, nvcc, a CUDA GPU and a KernelBlaster checkout (as test_kb_graph_opt_flow); skipped otherwise."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from modelblaster.pipeline.tests import test_kb_graph_opt_flow as F  # noqa: E402


@unittest.skipIf(F.SKIP is not None, F.SKIP or "")
class HalfwayFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(os.environ.get("MB_TEST_TMP", tempfile.gettempdir()))
        base.mkdir(parents=True, exist_ok=True)
        cls.tmp = Path(tempfile.mkdtemp(prefix="mb_hwflow_", dir=base))
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

    def _cfg(self, name, halfway, **kw):
        from modelblaster.pipeline.kb_graph_opt import Config
        inbox = self.tmp / f"{name}_inbox"
        inbox.mkdir(exist_ok=True)
        base = dict(kb=F.KB, model="stub:graphopt", plugins=[str(self.tmp / "plugin.py")], python=F.KB_PY,
                    arch=self.arch, rounds=4, rollouts=1, steps=2, variant_steps=2, max_fix=1, top_k=2,
                    max_variants=2, remeasure=3, profile_runs=1, accept_gain=-1.0, min_gain_pct=0.0,
                    knowledge_global="read", knowledge_global_store=str(self.tmp / "store"),
                    knowledge_global_problem=name, knowledge_global_refresh=str(inbox),
                    knowledge_global_halfway="on" if halfway else "off", knowledge_global_halfway_wait_s=60.0)
        base.update(kw)
        return Config(**base), inbox

    def _controller(self, work, inbox, stop):
        from modelblaster.pipeline import kb_halfway
        seen = set()
        while not stop.is_set():
            for f in work.glob("r[0-9][0-9]/halfway.json"):
                try:
                    h = json.loads(f.read_text())
                except (OSError, ValueError):
                    continue
                if h.get("stage") == "paused" and h["round"] not in seen:
                    seen.add(h["round"])
                    kb_halfway.stub_ack(inbox, h["round"], None, status="up_to_date")
            time.sleep(0.05)

    def _run(self, name, halfway, **kw):
        from modelblaster.pipeline import graph_hints as gh
        from modelblaster.pipeline.kb_graph_opt import run
        os.environ["MB_HALFWAY_POLL_S"] = "0.05"
        work = self.tmp / name
        cfg, inbox = self._cfg(name, halfway, **kw)
        stop = threading.Event()
        t = threading.Thread(target=self._controller, args=(work, inbox, stop), daemon=True)
        t.start()
        t0 = time.time()
        try:
            s = run([self.tmp / "ir"], work, cfg)
        finally:
            stop.set()
            t.join(5)
        doc = gh.load(work / "graph_hints.json")
        return s, work, {h["id"]: h["status"] for h in doc["hints"]}, time.time() - t0

    def _check(self, kw):
        s0, w0, st0, dt0 = self._run(f"off_{kw.get('graph_search', 'rounds')}", False, **kw)
        s1, w1, st1, dt1 = self._run(f"on_{kw.get('graph_search', 'rounds')}", True, **kw)
        for s in (s0, s1):
            self.assertTrue(next(iter(s["irs"].values()))["passed"], s)
        # the same fusion decisions (a `split:` hint comes from the proposals of the end-of-run timings and appears or not
        # from run to run, with and without the halfway sync alike)
        fuse = lambda st: {k: v for k, v in st.items() if k.startswith("fuse:")}   # noqa: E731
        self.assertEqual(fuse(st1), fuse(st0))
        self.assertTrue(any(v == "accepted" and k.startswith("fuse:") for k, v in st1.items()), st1)
        st = json.loads((w1 / "state.json").read_text())
        hw = st["knowledge_global"]["halfway"]
        self.assertGreaterEqual(hw["n"], 1)
        self.assertEqual((hw["timeouts"], hw["errors"]), (0, 0), hw)
        self.assertNotIn("halfway", json.loads((w0 / "state.json").read_text())["knowledge_global"])   # off: no trace
        self.assertTrue(list(w1.glob("r[0-9][0-9]/halfway.json")))
        self.assertFalse(list(w0.glob("r[0-9][0-9]/halfway.json")))
        for f in w1.glob("r[0-9][0-9]/halfway.json"):
            self.assertEqual(json.loads(f.read_text())["stage"], "done")
        print(f"\nhalfway flow {kw}: off {dt0:.0f}s on {dt1:.0f}s, pauses {hw['n']}, wait {hw['wait_s']}s", flush=True)
        return s0, s1

    def test_rounds_flow(self):
        s0, s1 = self._check({})
        d0 = next(iter(s0["irs"].values())).get("final_e2e_us") or next(iter(s0["irs"].values())).get("final_us")
        d1 = next(iter(s1["irs"].values())).get("final_e2e_us") or next(iter(s1["irs"].values())).get("final_us")
        if d0 and d1:
            self.assertLess(abs(d1 - d0) / d0, 0.5)

    def test_interleaved_graph_search_flow(self):
        self._check(dict(graph_search="interleaved", graph_priors="off", graph_characterize=1, graph_char_top=2))


if __name__ == "__main__":
    unittest.main()
