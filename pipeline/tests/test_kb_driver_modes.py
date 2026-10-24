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


if __name__ == "__main__":
    unittest.main()
