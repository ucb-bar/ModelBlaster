"""kb_export: data.bin size, streamed data.bin, the outputs' conditioning term, GPU lock path,
chunked bf16 narrowing.

CPU only (extract + export, no build): the reference-error term of an output
that cancels to ~0, data_bin_bytes against the file written, the float64 /
live-set walk, an empty partition, and KernelBlaster's GPU lock file name.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from modelblaster.pipeline import kb_export  # noqa: E402

# out[0] = sum_j (x - mean_j x): exactly 0, its fp32 value is rounding noise of
# |x| ~ 20 (as the gradient of a key bias through a softmax); out[1] = sum_j x
CANCEL_MODEL = '''
import torch
import torch.nn as nn
class Model(nn.Module):
    def forward(self, x):
        c = x - x.mean(dim=1, keepdim=True)
        return c.sum(dim=1), x.sum(dim=1)
def get_inputs():
    torch.manual_seed(0)
    return [torch.randn(16, 997) * 20 + 3]
def get_init_inputs():
    return []
'''


def _have_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_have_torch(), "needs torch")
class ConditioningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.d = Path(cls.tmp.name)
        (d / "cancel.py").write_text(CANCEL_MODEL)
        cp = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file",
                             str(d / "cancel.py"), "--out-dir", str(d / "ir"), "--quant", "fp32"],
                            cwd=ROOT.parent, capture_output=True, text=True, timeout=600,
                            env={**os.environ, "PYTHONPATH": os.pathsep.join(
                                [str(ROOT.parent), os.environ.get("PYTHONPATH", "")])})
        assert cp.returncode == 0, cp.stderr[-2000:]
        cls.prob = kb_export.export_problem(d / "ir", d / "prob", strategy="whole", conditioning=True)
        cls.man = json.loads((cls.prob / "manifest.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def outputs(self, man=None):
        man = man or self.man
        return [man["buffers"][i] for i in man["outputs"]]

    def test_cancelling_output_gets_its_reference_error(self):
        cancel, plain = self.outputs()
        self.assertGreater(cancel["ref_err"], 1e-6)              # the fp32 noise of sum(x - mean)
        # well-conditioned: the reference error is far below the relative bound
        self.assertLess(plain["ref_err"], 1e-3 * 1e-2 * 997 * 23)
        self.assertEqual(self.man["tolerance"]["output_cond"], kb_export.OUTPUT_COND)
        drv = (self.prob / "driver.cpp").read_text()
        cond = re.search(r"static const double B_COND\[\] = \{([^}]*)\};", drv).group(1).split(", ")
        k = self.man["outputs"][0]
        self.assertAlmostEqual(float(cond[k]), kb_export.OUTPUT_COND * cancel["ref_err"])
        self.assertIn("B_OUT_ABS[b] + B_COND[b]", drv)

    def test_cancelling_output_reference_exceeds_the_plain_bound(self):
        """Without the term the fp32 reference itself would fail its own output check
        (|ref - exact| > rel * max|ref| + abs): the bound was ill-posed for it."""
        import numpy as np
        cancel, _ = self.outputs()
        ref = np.fromfile(self.prob / "data.bin", dtype=np.uint8)
        vals = ref[cancel["ref"][0]:cancel["ref"][0] + cancel["ref"][1]].view(np.float32)
        plain = 1e-3 * float(np.abs(vals).max()) + 1e-5
        self.assertGreater(cancel["ref_err"], plain)

    def test_default_follows_the_ir_dtype_and_the_env(self):
        """conditioning=None (the default): off for this fp32 IR (the KernelBench default path keeps its old bound and
        cost), forced on by $MB_EXPORT_CONDITIONING=1; a bf16 / native IR defaults to on (the SOL problems)."""
        def refs(out):
            man = json.loads((out / "manifest.json").read_text())
            return [man["buffers"][i].get("ref_err") for i in man["outputs"]]
        p = kb_export.export_problem(self.d / "ir", self.d / "dflt", strategy="whole")
        self.assertTrue(all(r is None for r in refs(p)))
        old = os.environ.get("MB_EXPORT_CONDITIONING")
        os.environ["MB_EXPORT_CONDITIONING"] = "1"
        try:
            p = kb_export.export_problem(self.d / "ir", self.d / "dflt_env", strategy="whole")
            self.assertTrue(all(r is not None for r in refs(p)))
            os.environ["MB_EXPORT_CONDITIONING"] = "0"
            p = kb_export.export_problem(self.d / "ir", self.d / "dflt_env0", strategy="whole", conditioning=None)
            self.assertTrue(all(r is None for r in refs(p)))
        finally:
            if old is None:
                os.environ.pop("MB_EXPORT_CONDITIONING", None)
            else:
                os.environ["MB_EXPORT_CONDITIONING"] = old

    def test_no_conditioning(self):
        p = kb_export.export_problem(self.d / "ir", self.d / "nocond", strategy="whole", conditioning=False)
        man = json.loads((p / "manifest.json").read_text())
        self.assertTrue(all(b.get("ref_err") is None for b in self.outputs(man)))
        drv = (p / "driver.cpp").read_text()
        cond = re.search(r"B_COND\[\] = \{([^}]*)\}", drv).group(1).split(", ")
        self.assertTrue(all(float(c) == 0.0 for c in cond))

    def test_data_bin_streamed_and_sized(self):
        blob = (self.prob / "data.bin").read_bytes()
        self.assertEqual(hashlib.sha1(blob).hexdigest(), self.man["data_sha1"])
        last = max((b["ref"] for b in self.man["buffers"] if b["ref"]), key=lambda r: r[0])
        self.assertEqual(len(blob), last[0] + last[1])
        est = kb_export.data_bin_bytes(self.d / "ir")
        self.assertGreaterEqual(est, len(blob))                  # padded to 256 per buffer
        self.assertLess(est - len(blob), 256 * len(self.man["buffers"]))

    def test_walk_float64_live_set(self):
        import torch
        sys.path.insert(0, str(ROOT / "gpu_eval" / "harness"))
        import mbir
        import walk_torch
        ir = mbir.load_ir(self.d / "ir")
        out32, env32 = walk_torch.walk(ir, dev="cpu")
        out64, env64 = walk_torch.walk(ir, dev="cpu", dtype=torch.float64, store_rounding=False, keep=set())
        self.assertEqual(out64.dtype, torch.float64)
        self.assertEqual(set(env64), set(ir.output_names()))     # only the outputs survive
        self.assertTrue(torch.allclose(out32.double(), out64, rtol=1e-4, atol=1e-2))
        n = len(env32[ir.output_names()[0]].reshape(-1))
        self.assertLess(float(out64[:n].abs().max()), 1e-9)       # the cancelling output is ~0 exactly


class EmptyPartitionTest(unittest.TestCase):
    def test_no_ops_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            g = {"name": "empty", "ops": [], "tensors": {"x": {"shape": [4]}},
                 "input": {"tensor": "x"}, "output": {"tensor": "x"}}
            (d / "ir").mkdir()
            (d / "ir" / "graph.json").write_text(json.dumps(g))
            import numpy as np
            np.savez(d / "ir" / "weights.npz")
            np.savez(d / "ir" / "io.npz", input=np.zeros(4, np.float32), output=np.zeros(4, np.float32))
            with self.assertRaisesRegex(ValueError, "no ops to export"):
                kb_export.export_problem(d / "ir", d / "prob")


class Bf16ChunkTest(unittest.TestCase):
    """round_bf16 / bf16_bits in bounded chunks (uint32, no uint64 copies of a multi-GB
    array): the same bits across chunk edges, torch's RNE on finite values, quiet nans."""

    def test_chunks_and_specials(self):
        import numpy as np
        from modelblaster.pipeline import cuda_dtypes as cd
        rng = np.random.default_rng(1)
        u = np.concatenate([rng.integers(0, 2 ** 32, 10007, dtype=np.uint64).astype(np.uint32),
                            np.array([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7F7FFFFF, 0xFF7FFFFF,
                                      0x7F800001, 0xFFFFFFFF, 0x3F808000, 0x3F818000], np.uint32)])
        x = u.view(np.float32)
        whole = cd.bf16_bits(x)
        with mock.patch.object(cd, "_BF16_CHUNK", 97):
            self.assertTrue(np.array_equal(cd.bf16_bits(x), whole))
            self.assertTrue(np.array_equal(cd.round_bf16(x).view(np.uint32) >> 16, whole.astype(np.uint32)))
        fin = np.isfinite(x)
        if _have_torch():
            import torch
            ref = torch.from_numpy(x[fin]).bfloat16().view(torch.int16).numpy().view(np.uint16)
            self.assertTrue(np.array_equal(whole[fin], ref))
        nan = np.isnan(x)
        self.assertTrue(np.all(np.isnan(cd.round_bf16(x)[nan])))
        self.assertTrue(np.all(whole[nan] & 0x40))                       # quiet
        self.assertEqual(cd.round_bf16(np.ones((3, 4), np.float32)).shape, (3, 4))


FAKE_HOST = """#!/usr/bin/env python3
import sys, time
print("MB_HOST_READY abi=1 nb=0 nops=0 nslots=0 init_ms=0", flush=True)
for line in sys.stdin:
    f = line.rstrip("\\n").split("\\t")
    if f[0] != "EVAL":
        break
    time.sleep(1.5 if any(x.startswith("lock=") for x in f) else 0)    # 'waiting for the GPU lock'
    print("passed", flush=True)
    print(f"MB_HOST_DONE {f[1]} rc=0 poisoned=0 recycle=0", flush=True)
"""


class DriverHostLockTest(unittest.TestCase):
    def test_lock_wait_not_counted_against_the_run(self):
        """A request queued behind another user's lock gets its run timeout after the lock wait
        (as KernelBlaster's runtime_host): timeout 1 s, 1.5 s in the lock, still answered."""
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "host"
            exe.write_text(FAKE_HOST.replace("/usr/bin/env python3", sys.executable))
            exe.chmod(0o755)
            h = kb_export.DriverHost(exe, timeout=1)
            try:
                r = h.evaluate("x.so", lock=str(Path(d) / "gpu.lock"), lock_wait=30)
                self.assertIsNone(r["host_crash"], r)
                self.assertTrue(r["passed"])
            finally:
                h.close()


class GpuLockPathTest(unittest.TestCase):
    def test_specs(self):
        self.assertIsNone(kb_export.gpu_lock_path("none"))
        self.assertIsNone(kb_export.gpu_lock_path(None))
        self.assertEqual(kb_export.gpu_lock_path("/tmp/x.lock"), "/tmp/x.lock")
        env = {"CUDA_VISIBLE_DEVICES": "GPU-22350958-ee15,GPU-1", "KERNELBLASTER_GPU_LOCK_DIR": "/run/l"}
        with mock.patch.dict(os.environ, env):
            self.assertEqual(kb_export.gpu_lock_path("auto"), "/run/l/kernelblaster_gpu_GPU-22350958-ee15.lock")
            with mock.patch.dict(os.environ, {"KERNELBLASTER_GPU_FLOCK": "0"}):
                self.assertIsNone(kb_export.gpu_lock_path("auto"))

    def test_index_resolved_by_nvidia_smi(self):
        cp = subprocess.CompletedProcess([], 0, stdout="GPU-019a51ce-7238\n", stderr="")
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3", "KERNELBLASTER_GPU_LOCK_DIR": "/dev/shm"}), \
                mock.patch.object(kb_export.subprocess, "run", return_value=cp) as run:
            self.assertEqual(kb_export.gpu_lock_path("auto"), "/dev/shm/kernelblaster_gpu_GPU-019a51ce-7238.lock")
            self.assertIn("3", run.call_args[0][0])
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):     # no GPU visible
            self.assertIsNone(kb_export.gpu_lock_path("auto"))
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}), \
                mock.patch.object(kb_export.subprocess, "run", side_effect=OSError):   # no nvidia-smi
            self.assertIsNone(kb_export.gpu_lock_path("auto"))


if __name__ == "__main__":
    unittest.main()
