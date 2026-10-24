"""The generated multi-slot driver's run modes (docs/gpu_harness_perf.md):
MB_ONLY_SLOTS, MB_SCREEN, MB_GRAPH and the MB_PHASE_MS breakdown.

The parser test runs everywhere; the build tests need torch, nvcc and a CUDA
GPU (one tiny model is extracted, exported and built once for all of them).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline.kb_export import parse_driver_output  # noqa: E402

MODEL = '''import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 32)
    def forward(self, x):
        return torch.relu(self.fc2(torch.relu(self.fc(x))))
def get_inputs():
    return [torch.randn(4, 64)]
def get_init_inputs():
    return []
'''


class ParseTest(unittest.TestCase):
    def test_new_lines(self):
        r = parse_driver_output("\n".join([
            "MB_ONLY_SLOTS relu", "MB_CHECK output y max_abs_err=0 bound=1e-05 ok=1",
            "MB_SLOT_CHECK relu rel_err=0 ok=1", "MB_PARTITION_US 12.5 reps=200",
            "MB_PARTITION_DIST min=11.000 med=12.000 max=20.000",
            "MB_PARTITION_GRAPH_US 9.25 reps=200 nodes=3", "MB_SLOT_US relu 2.0", "MB_OP_US relu 2.0",
            "MB_PHASE_MS cuda_init 150.0", "MB_PHASE_MS total 160.5", "passed"]))
        self.assertTrue(r["passed"])
        self.assertEqual(r["partition_us"], 12.5)
        self.assertEqual(r["partition_graph_us"], 9.25)
        self.assertEqual(r["partition_dist"], {"min": 11.0, "med": 12.0, "max": 20.0})
        self.assertEqual(r["phase_ms"], {"cuda_init": 150.0, "total": 160.5})
        self.assertIsNone(r["graph_skip"])
        r = parse_driver_output("MB_GRAPH_SKIP operation not permitted when stream is capturing\npassed\n")
        self.assertEqual(r["graph_skip"], "operation not permitted when stream is capturing")
        self.assertIsNone(r["partition_graph_us"])

    def test_e2e_and_attribution_lines(self):
        r = parse_driver_output("\n".join([
            "MB_PARTITION_US 12.0 reps=10", "MB_PARTITION_GRAPH_US 9.0 reps=10 nodes=4",
            "MB_E2E_US 8.900 basis=graph mean=9.000 reps=10", "MB_REF_E2E_US 10.000 basis=graph mean=10.0 reps=10",
            "MB_E2E_RATIO med=0.89000 mean=0.9 min=0.8 max=1.0 n=10",
            "MB_ATTR_SLOT_US relu 1.500 launches=2.0", "MB_ATTR_OP_US relu 1.500",
            "MB_ATTR_SUM_US 7.000 basis=cupti reps=5 launches=4.0 unmatched=0",
            "MB_LAUNCH_OVERHEAD_US eager=5.000 graph=2.000 frac_eager=0.4167", "passed"]))
        self.assertEqual((r["e2e_us"], r["e2e_basis"], r["ref_e2e_us"]), (8.9, "graph", 10.0))
        self.assertEqual(r["e2e_ratio"]["med"], 0.89)
        self.assertEqual(r["attr_slot_us"], {"relu": 1.5})
        self.assertEqual(r["attr_slot_launches"], {"relu": 2.0})
        self.assertEqual((r["attr_sum_us"], r["attr_basis"], r["attr_launches"]), (7.0, "cupti", 4.0))
        self.assertEqual((r["launch_overhead_us"], r["launch_overhead_graph_us"]), (5.0, 2.0))
        self.assertEqual(r["graph_nodes"], 4)
        self.assertIsNone(parse_driver_output("MB_ATTR_SKIP no cupti\npassed\n")["attr_sum_us"])


def _have_gpu() -> bool:
    if shutil.which("nvcc") is None or shutil.which("nvidia-smi") is None:
        return False
    try:
        import torch  # noqa: F401
        return subprocess.run(["nvidia-smi"], capture_output=True, timeout=30).returncode == 0
    except Exception:
        return False


@unittest.skipUnless(_have_gpu(), "needs torch, nvcc and a CUDA GPU")
class DriverModesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from modelblaster.pipeline.kb_export import build_and_run, export_problem

        root = Path(__file__).resolve().parents[3]
        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.d = Path(cls.tmp.name)
        (d / "tiny.py").write_text(MODEL)
        cp = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file",
                             str(d / "tiny.py"), "--out-dir", str(d / "ir"), "--quant", "fp32"],
                            cwd=root, capture_output=True, text=True, timeout=600,
                            env={**os.environ, "PYTHONPATH": os.pathsep.join(
                                [str(root), os.environ.get("PYTHONPATH", "")])})
        assert cp.returncode == 0, cp.stderr[-2000:]
        cls.prob = export_problem(d / "ir", d / "prob")
        cls.arch = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.split()[0].replace(".", "")
        cls.exe = build_and_run(cls.prob, arch=cls.arch, runs=0)["exe"]
        cls.exe_ptds = build_and_run(cls.prob, arch=cls.arch, runs=0, graph=True, work=d / "ptds")["exe"]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_exe(self, env=None, exe=None):
        from modelblaster.pipeline.kb_export import run_built
        return run_built(exe or self.exe, env=env)

    def test_full_run_reports_every_slot_and_phases(self):
        r = self.run_exe()
        self.assertTrue(r["passed"], r["log"])
        self.assertEqual(set(r["slots"]), {"linear", "relu"})
        self.assertEqual(len(r["op_us"]), 4)
        for p in ("cuda_init", "check_partition", "check_ops", "time_partition", "time_ops", "total"):
            self.assertIn(p, r["phase_ms"])
        self.assertIsNotNone(r["partition_dist"])

    def test_only_slots(self):
        r = self.run_exe({"MB_ONLY_SLOTS": "relu"})
        self.assertTrue(r["passed"], r["log"])
        self.assertEqual(set(r["slots"]), {"relu"})        # isolated checks: relu's ops only
        self.assertTrue(r["outputs"])                      # the partition is still checked
        self.assertIsNotNone(r["partition_us"])            # ... and timed, and every op
        self.assertEqual(set(r["slot_us"]), {"linear", "relu"})
        self.assertEqual(len(r["op_us"]), 4)
        self.assertLess(r["phase_ms"]["check_ops"], self.run_exe()["phase_ms"]["check_ops"] + 50)
        r = self.run_exe({"MB_ONLY_SLOTS": "relu,nope"})
        self.assertFalse(r["passed"])
        self.assertTrue(any("nope" in e for e in r["errors"]))

    def test_screen(self):
        full, scr = self.run_exe(), self.run_exe({"MB_SCREEN": "1"})
        self.assertTrue(scr["passed"], scr["log"])
        reps = lambda r: int(r["log"].split("MB_PARTITION_US")[1].split("reps=")[1].split()[0])
        self.assertLessEqual(reps(scr), reps(full))
        self.assertEqual(set(scr["slots"]), {"linear", "relu"})

    def test_gates_kept_with_only_slots(self):
        """A broken slot that is not the one being screened still fails the run
        (partition output check); screening the broken slot attributes it."""
        from modelblaster.pipeline.kb_export import build_and_run

        bad = (self.prob / "slots" / "relu.cu").read_text().replace("fmaxf(x[i], 0.0f)", "fmaxf(x[i], 1.0f)")
        exe = build_and_run(self.prob, arch=self.arch, sources={"relu": bad}, work=self.d / "bad", runs=0)["exe"]
        r = self.run_exe({"MB_ONLY_SLOTS": "linear"}, exe)
        self.assertFalse(r["passed"])
        self.assertFalse(all(o["ok"] for o in r["outputs"].values()))
        self.assertTrue(r["slots"]["linear"]["ok"])
        r = self.run_exe({"MB_ONLY_SLOTS": "relu", "MB_SCREEN": "1"}, exe)
        self.assertFalse(r["passed"])
        self.assertFalse(r["slots"]["relu"]["ok"])

    def test_graph(self):
        r = self.run_exe({"MB_GRAPH": "1"}, self.exe_ptds)
        self.assertTrue(r["passed"], r["log"])
        self.assertIsNotNone(r["partition_graph_us"], r["log"])
        self.assertIn("MB_GRAPH_CHECK output", r["log"])
        # legacy-stream build: not capturable, reported and skipped, run still passes
        r = self.run_exe({"MB_GRAPH": "1"})
        self.assertTrue(r["passed"], r["log"])
        self.assertIsNone(r["partition_graph_us"])
        self.assertTrue(r["graph_skip"])

    def test_graph_fallback_on_host_sync(self):
        """A launcher that synchronises cannot be captured: MB_GRAPH_SKIP, eager numbers stand."""
        from modelblaster.pipeline.kb_export import build_and_run

        src = (self.prob / "slots" / "relu.cu").read_text()
        sync = re.sub(r"(k_relu<<<[^;]*;)", r"\1 cudaDeviceSynchronize();", src)
        self.assertNotEqual(sync, src)
        res = build_and_run(self.prob, arch=self.arch, sources={"relu": sync}, work=self.d / "sync", graph=True)
        self.assertTrue(res["passed"], res.get("log"))
        self.assertIsNone(res["partition_graph_us"])
        self.assertTrue(res["graph_skip"])
        self.assertIsNotNone(res["partition_us"])

    # ---- the persistent host (driver.cpp -DMB_HOST + the kernels as a shared object)
    def test_host_matches_one_process_driver(self):
        """One host request prints what the one-process driver prints: same checks, same errors."""
        from modelblaster.pipeline.kb_export import build_and_run, supports_host

        self.assertTrue(supports_host(self.prob))
        for env in ({}, {"MB_ONLY_SLOTS": "relu", "MB_SCREEN": "1"}):
            h = build_and_run(self.prob, arch=self.arch, work=self.d / "host", runtime="host", env=env)
            q = self.run_exe({"MB_GRAPH": "1", **env}, self.exe_ptds)
            self.assertTrue(h["passed"] and q["passed"], h["log"])
            self.assertEqual(h["slots"], q["slots"])
            self.assertEqual(h["outputs"], q["outputs"])
            self.assertEqual(set(h["slot_us"]), set(q["slot_us"]))
            self.assertEqual((h["e2e_basis"], q["e2e_basis"]), ("graph", "graph"))
            self.assertIn("load", h["phase_ms"])
            self.assertNotIn("cuda_init", h["phase_ms"])          # paid once, at host start
        if h["attr_skip"] is None:
            self.assertEqual(set(h["attr_slot_us"]), {"linear", "relu"})
            self.assertGreaterEqual(h["attr_launches"], 4)

    def test_host_crash_recovery_and_pairing(self):
        """A crashing candidate is a run error and the host restarts; a wrong one fails its slot;
        ref= times the incumbent in the same request."""
        from modelblaster.pipeline.kb_export import DriverHost, build_host, build_so

        host, log = build_host(self.prob, arch=self.arch, work=self.d / "h2")
        self.assertIsNotNone(host, log)
        src = (self.prob / "slots" / "relu.cu").read_text()
        segv = ("#include <cuda_runtime.h>\n"
                "void kernel_relu(const float *input, float *output, int n) { *(volatile int *)nullptr = n; }\n")
        wrong = src.replace("fmaxf(x[i], 0.0f)", "fmaxf(x[i], 1.0f)")
        sos = {}
        for name, s in (("good", None), ("segv", segv), ("wrong", wrong)):
            so, log = build_so(self.prob, arch=self.arch, sources={"relu": s} if s else None, work=self.d / "h2",
                               name=f"{name}.so")
            self.assertIsNotNone(so, log)
            sos[name] = so
        h = DriverHost(host, timeout=120)
        try:
            env = {"MB_GRAPH": "1", "MB_SCREEN": "1"}
            r = h.evaluate(sos["good"], env=env)
            self.assertTrue(r["passed"], r["log"])
            r = h.evaluate(sos["segv"], env=env)
            self.assertFalse(r["passed"])
            self.assertTrue(r["host_crash"])
            r = h.evaluate(sos["wrong"], env={**env, "MB_ONLY_SLOTS": "relu"})
            self.assertFalse(r["passed"])
            self.assertFalse(r["slots"]["relu"]["ok"])
            self.assertIsNone(r["host_crash"])
            r = h.evaluate(sos["good"], ref=sos["good"], env=env)
            self.assertTrue(r["passed"], r["log"])
            self.assertIsNotNone(r["e2e_ratio"])
            self.assertEqual(h.starts, 2)
        finally:
            h.close()

    def test_model_runtime_composes_window_winners(self):
        """kb_model_run: windows' best slots composed into one whole-model runtime, checked and
        timed by graph replay against the hand-seeded model (paired), next to the windows' sum."""
        import json
        from modelblaster.pipeline.kb_export import export_problem, partition_ir
        from modelblaster.pipeline.kb_model_run import model_runtime

        ir = self.d / "ir"
        g = json.loads((ir / "graph.json").read_text())
        parts = partition_ir(g, "window", max_ops=2)
        self.assertEqual(len(parts), 2)
        for k, idx in enumerate(parts):
            pdir = export_problem(ir, self.d / "win" / f"p{k}", op_indices=idx, strategy="window")
            man = json.loads((pdir / "manifest.json").read_text())
            run = self.d / "win_runs" / man["name"]
            (run / "best" / "slots").mkdir(parents=True)
            for sl in man["slots"]:          # "winners": the seeds with a marker (same code per window)
                (run / "best" / "slots" / f"{sl['id']}.cu").write_text((pdir / sl["file"]).read_text() + "\n// win\n")
            (run / "result.json").write_text(json.dumps({"problem": man["name"], "problem_dir": str(pdir),
                                                         "final_ok": True, "run_id": "t", "slots": {}}))
        r = model_runtime(ir, work=self.d / "model_run", runs=[self.d / "win_runs"], arch=self.arch, reps=1,
                          time_partitions=True)
        self.assertTrue(r["passed"], r["model"].get("log"))
        self.assertEqual(r["model_basis"], "graph")
        self.assertTrue(r["paired"])                 # same slots as the seed model: timed in one process
        self.assertTrue(r["seed_e2e_us"] and r["model_e2e_us"] and r["speedup_paired"])
        self.assertTrue(all(t.startswith("cache:") for t in r["composed_slots"].values()))
        self.assertEqual(set(r["composed_slots"]), {"linear", "relu"})   # one slot each, not split by window
        self.assertEqual(len(r["partitions"]), 2)
        self.assertTrue(all(p["passed"] for p in r["partitions"]))
        self.assertIsNotNone(r["boundary_effect_us"])


if __name__ == "__main__":
    unittest.main()
