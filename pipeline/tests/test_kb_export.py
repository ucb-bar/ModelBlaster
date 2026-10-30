"""Tests for the KernelBlaster export / write-back / proposal pipeline.

Pure-python tests run everywhere; the end-to-end test (extract a tiny
model, export it, nvcc-build and run the driver) needs torch, nvcc and a
CUDA GPU and is skipped otherwise.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline.kb_export import CudaLibrary, partition_ir, wrap_slot  # noqa: E402
from modelblaster.pipeline.kb_ingest import CudaCache  # noqa: E402
from modelblaster.pipeline.op_classes import op_class  # noqa: E402

SRC = r'''#include <cuda_runtime.h>
#define MB_TID ((long)blockIdx.x * blockDim.x + threadIdx.x)

// ---------------------------------------------------------------- banner

// helper used by two kernels
__device__ float sq(float v) { return v * v; }

// squares
__global__ void k_sq(const float *x, float *y, int n) { long i = MB_TID; if (i < n) y[i] = sq(x[i]); }
__global__ void k_other(const float *x, float *y, int n) { long i = MB_TID; if (i < n) y[i] = sq(x[i]) + 1; }

extern "C" void kernel_sq(const float *x, float *y, int n) { k_sq<<<(n + 255) / 256, 256>>>(x, y, n); }
extern "C" void kernel_other(const float *x, float *y, int n) { k_other<<<(n + 255) / 256, 256>>>(x, y, n); }
'''


class CudaLibraryTest(unittest.TestCase):
    def test_closure_and_comments(self):
        lib = CudaLibrary(SRC)
        self.assertEqual(lib.closure("kernel_sq"), ["sq", "k_sq", "kernel_sq"])
        src = lib.slot_source("kernel_sq")
        self.assertIn("// helper used by two kernels", src)
        self.assertIn("// squares", src)
        self.assertNotIn("banner", src)              # separated by a blank line
        self.assertNotIn("k_other", src)
        self.assertIn("#define MB_TID", lib.head_source())

    def test_wrap_slot(self):
        w = wrap_slot("sq", '#include <cuda_fp16.h>\n#define T 4\nextern "C" void kernel_sq() {}\n')
        self.assertTrue(w.startswith("#include <cuda_fp16.h>"))
        self.assertIn("namespace mbslot_sq {", w)
        self.assertIn("#undef T", w)
        self.assertNotIn('extern "C"', w)


class PartitionTest(unittest.TestCase):
    def test_strategies(self):
        g = {"ops": [{"op": "conv2d"}, {"op": "view"}, {"op": "relu"}, {"op": "linear"}, {"op": "view"}]}
        self.assertEqual(partition_ir(g, "whole"), [[0, 1, 2, 3, 4]])
        self.assertEqual(partition_ir(g, "per_op"), [[0], [1, 2], [3, 4]])
        self.assertEqual(partition_ir(g, "window", 2), [[0, 1, 2], [3, 4]])

    def test_op_classes(self):
        self.assertEqual(op_class("conv2d"), "conv")
        self.assertEqual(op_class("linear_s8"), "gemm")
        self.assertEqual(op_class("batchnorm2d"), "elementwise")
        self.assertEqual(op_class("nonsense"), "other")


class FusionTest(unittest.TestCase):
    def test_fuse_pair_and_bind(self):
        from modelblaster.pipeline.kb_proposals import bind_fused, fuse_pair
        from modelblaster.pipeline.reference_kernels import KERNEL_SPECS

        g = {"name": "t", "input": {"tensor": "x"}, "output": {"tensor": "r"},
             "tensors": {"x": {"shape": [8]}, "y": {"shape": [8]}, "r": {"shape": [8]}},
             "ops": [{"name": "s", "op": "sigmoid", "inputs": ["x"], "outputs": ["y"], "shape": {"n": 8}},
                     {"name": "r", "op": "relu", "inputs": ["y"], "outputs": ["r"], "shape": {"n": 8}}]}
        g2, spec = fuse_pair(g, "s", "r", KERNEL_SPECS)
        self.assertEqual([o["op"] for o in g2["ops"]], ["fused__sigmoid__relu"])
        self.assertEqual(spec.signature,
                         "void kernel_fused__sigmoid__relu(const float* a_input, float* a_output, int a_n, "
                         "float* b_output, int b_n);")
        args = bind_fused(g2["ops"][0], KERNEL_SPECS)
        self.assertEqual([(a.kind, a.value) for a in args],
                         [("in", "x"), ("scratch", "y"), ("scalar", 8), ("out", "r"), ("scalar", 8)])
        with self.assertRaises(ValueError):           # y is a graph output -> can't be internal
            g3 = json.loads(json.dumps(g))
            g3["output"] = {"tensors": ["y", "r"]}
            fuse_pair(g3, "s", "r", KERNEL_SPECS)


class RegistryTest(unittest.TestCase):
    def test_accepted_fusions_round_trip(self):
        from modelblaster.pipeline.kb_proposals import apply_accepted, make_fused_spec, spec_from_kind
        from modelblaster.pipeline.reference_kernels import KERNEL_SPECS

        s = make_fused_spec("sigmoid", "relu", "input", KERNEL_SPECS)
        self.assertEqual(spec_from_kind(s.op, KERNEL_SPECS).signature, s.signature)
        g = {"name": "t", "input": {"tensor": "x"}, "output": {"tensor": "z"},
             "tensors": {k: {"shape": [8]} for k in "xyz"},
             "ops": [{"name": "s", "op": "sigmoid", "inputs": ["x"], "outputs": ["y"], "shape": {"n": 8}},
                     {"name": "r", "op": "relu", "inputs": ["y"], "outputs": ["z"], "shape": {"n": 8}}]}
        with tempfile.TemporaryDirectory() as d:
            reg = Path(d) / "r.json"
            reg.write_text(json.dumps({"format": "mb-fusion-registry/1", "fusions": {
                s.op: {"kind": s.op, "signature": s.signature, "status": "rejected", "evidence": []}}}))
            self.assertEqual(apply_accepted(g, reg, KERNEL_SPECS)[2], [])
            r = json.loads(reg.read_text())
            r["fusions"][s.op]["status"] = "accepted"
            reg.write_text(json.dumps(r))
            g2, used, done = apply_accepted(g, reg, KERNEL_SPECS)
            self.assertEqual(done, ["s+r"])
            self.assertEqual([o["op"] for o in g2["ops"]], [s.op])
            self.assertIn(s.op, used)


class DecideTest(unittest.TestCase):
    def test_decide_compares_only_touched_ops(self):
        from modelblaster.pipeline.kb_proposals import decide

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "o").mkdir()
            (d / "v").mkdir()
            man = lambda slots: {"slots": [{"id": s, "instances": [{"op_name": n} for n in ns]}
                                           for s, ns in slots.items()]}
            (d / "o" / "manifest.json").write_text(json.dumps(man({"gemm": ["g"], "sig": ["s"], "relu": ["r"]})))
            (d / "v" / "manifest.json").write_text(json.dumps(man({"gemm": ["g"], "f": ["s+r"]})))
            # the variant's partition is much faster only because its GEMM got lucky;
            # the fusion itself saves 40 of 1000 us (4%) -> accepted on its own merit
            o = {"problem": "p", "problem_dir": str(d / "o"), "final_ok": True, "final_partition_us": 1000.0,
                 "op_us": {"g": 900.0, "s": 50.0, "r": 50.0}, "slots": {}}
            v = {"problem": "p", "problem_dir": str(d / "v"), "final_ok": True, "final_partition_us": 560.0,
                 "op_us": {"g": 500.0, "s+r": 60.0}, "slots": {}}
            var = {"applied": [{"pair": ["s", "r"], "kind": "fused__sigmoid__relu", "signature": "x"}]}
            for n, x in (("o.json", o), ("v.json", v), ("var.json", var)):
                (d / n).write_text(json.dumps(x))
            r = decide(d / "o.json", d / "v.json", d / "var.json", registry=d / "reg.json")
            k = r["kinds"]["fused__sigmoid__relu"]
            self.assertAlmostEqual(k["gain"], 0.04)
            self.assertEqual(k["status"], "accepted")
            self.assertAlmostEqual(r["partition_gain"], 0.44)
            v["op_us"]["s+r"] = 90.0                         # fusion only saves 1% -> rejected
            (d / "v.json").write_text(json.dumps(v))
            r = decide(d / "o.json", d / "v.json", d / "var.json", registry=d / "reg2.json")
            self.assertEqual(r["kinds"]["fused__sigmoid__relu"]["status"], "rejected")


class CacheTest(unittest.TestCase):
    def test_shape_guard(self):
        with tempfile.TemporaryDirectory() as d:
            c = CudaCache(Path(d))
            sig = "void kernel_relu(const float *input, float *output, int n);"
            c.add(op="relu", entry="kernel_relu", signature=sig, arch="sm_75", shapes=[{"n": 8}, {"n": 16}],
                  code="// x", slot_us=1.0, speedup=2.0, impl_id="i1", source={})
            self.assertIsNotNone(c.lookup("kernel_relu", sig, "sm_75", [{"n": 8}]))
            self.assertIsNone(c.lookup("kernel_relu", sig, "sm_75", [{"n": 8}, {"n": 32}]))
            self.assertIsNone(c.lookup("kernel_relu", sig, "sm_89", [{"n": 8}]))
            self.assertIsNone(c.lookup("kernel_relu", sig.replace("int n", "long n"), "sm_75", [{"n": 8}]))
            _, st = c.add(op="relu", entry="kernel_relu", signature=sig, arch="sm_75",
                          shapes=[{"n": 16}, {"n": 8}], code="// y", slot_us=0.5, speedup=4.0,
                          impl_id="i2", source={})
            self.assertEqual(st, "replaced")
            c.save()
            self.assertEqual(CudaCache(Path(d)).lookup("kernel_relu", sig, "sm_75", [{"n": 8}])["impl_id"], "i2")


MODEL = '''import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 32)
    def forward(self, x):
        return torch.relu(self.fc(x))
def get_inputs():
    return [torch.randn(4, 64)]
def get_init_inputs():
    return []
'''


def _have_gpu() -> bool:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "0").strip() in ("", "-1"):   # GPUs hidden: skip
        return False
    if shutil.which("nvcc") is None or shutil.which("nvidia-smi") is None:
        return False
    try:
        import torch  # noqa: F401
        return subprocess.run(["nvidia-smi"], capture_output=True, timeout=30).returncode == 0
    except Exception:
        return False


ARGMAX_MODEL = '''import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 32)
    def forward(self, x):
        return torch.argmax(torch.selu(self.fc(x)), dim=1)
def get_inputs():
    return [torch.randn(8, 64)]
def get_init_inputs():
    return []
'''


@unittest.skipUnless(_have_gpu(), "needs torch, nvcc and a CUDA GPU")
class EndToEndTest(unittest.TestCase):
    def test_int64_argmax_output(self):
        """argmax writes int64 indices: an i64 buffer end to end."""
        from modelblaster.pipeline.kb_export import build_and_run, export_problem

        root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "am.py").write_text(ARGMAX_MODEL)
            cp = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file",
                                 str(d / "am.py"), "--out-dir", str(d / "ir"), "--quant", "fp32"],
                                cwd=root, capture_output=True, text=True, timeout=600,
                                env={**__import__("os").environ, "PYTHONPATH": str(root)})
            self.assertEqual(cp.returncode, 0, cp.stderr[-2000:])
            prob = export_problem(d / "ir", d / "prob")
            man = json.loads((prob / "manifest.json").read_text())
            self.assertIn("i64", {b["dtype"] for b in man["buffers"]})
            arch = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.split()[0].replace(".", "")
            res = build_and_run(prob, arch=arch)
            self.assertTrue(res["passed"], res.get("log"))

    def test_extract_export_build_run(self):
        from modelblaster.pipeline.kb_export import assemble, build_and_run, export_problem

        root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "tiny.py").write_text(MODEL)
            cp = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file",
                                 str(d / "tiny.py"), "--out-dir", str(d / "ir"), "--quant", "fp32"],
                                cwd=root, capture_output=True, text=True, timeout=600,
                                env={**os.environ, "PYTHONPATH": os.pathsep.join(
                                    [str(root), os.environ.get("PYTHONPATH", "")])})
            self.assertEqual(cp.returncode, 0, cp.stderr[-2000:])
            prob = export_problem(d / "ir", d / "prob")
            man = json.loads((prob / "manifest.json").read_text())
            self.assertEqual(sorted(s["id"] for s in man["slots"]), ["linear", "relu"])
            self.assertIn("namespace mbslot_linear", assemble(prob))
            arch = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.split()[0].replace(".", "")
            res = build_and_run(prob, arch=arch)
            self.assertTrue(res["passed"], res.get("log"))
            self.assertEqual(set(res["slot_us"]), {"linear", "relu"})
            # a wrong slot is caught by its isolated check and attributed to it
            bad = (prob / "slots" / "relu.cu").read_text().replace("fmaxf(x[i], 0.0f)", "fmaxf(x[i], 1.0f)")
            res = build_and_run(prob, arch=arch, sources={"relu": bad}, work=d / "bad")
            self.assertFalse(res["passed"])
            self.assertFalse(res["slots"]["relu"]["ok"])
            self.assertTrue(res["slots"]["linear"]["ok"])


if __name__ == "__main__":
    unittest.main()
