"""Tests for the OpenCL (Qualcomm Adreno) multi-slot target.

Pure-python tests (seed coverage, templates, the fp16 transform) run
everywhere. The host-compile test needs g++ and OpenCL headers; the device
tests (every seed program builds; extract -> export -> build -> run, a wrong
slot attributed) need an OpenCL GPU with coarse-grained SVM (the Adreno
board's role; any OpenCL 2.0 GPU) and torch, and are skipped otherwise.
"""
from __future__ import annotations

import functools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline.kb_export import wrap_slot  # noqa: E402
from modelblaster.pipeline.kb_export_opencl import HAND_LIB, RUNTIME_H  # noqa: E402
from modelblaster.pipeline.opencl_seeds import (OpenCLSeeds, all_entries, f16_device, template_source,  # noqa: E402
                                                to_f16)

LEVEL_KINDS = {  # every op kind KernelBench Level1-3 IRs use (view is an alias, not a kernel)
    "adaptive_avg_pool2d", "add", "argmax_dim", "argmin_dim", "avgpool2d", "avgpool3d", "batchnorm2d", "bmm",
    "cat2_c1", "cat3_c1", "cat4_c1", "catn", "conv2d", "conv2d_dw", "conv2d_grouped", "conv3d",
    "conv_transpose2d", "conv_transpose3d", "cross_entropy_loss", "cumprod", "cumsum", "diag_matmul", "elu",
    "ewise1", "ewise2", "ewise_scalar", "exclusive_cumsum", "flip", "frobenius_norm", "gather", "gelu",
    "gelu_exact", "group_norm", "gru_seq", "hardsigmoid", "hardtanh", "hinge_loss", "huber_loss", "kldiv_loss",
    "l2_norm", "layer_norm", "leaky_relu", "linear", "log", "log_softmax", "lstm_seq", "matmul", "matmul_ta",
    "matmul_tatb", "matmul_tb", "max_dim", "maxpool2d", "maxpool2d_ex", "maxpool3d", "mean_abs_norm",
    "mean_dim", "min_dim", "mse_loss", "mul", "mul_scalar", "reduce", "relu", "relu6", "rms_norm", "sdpa",
    "selu", "sigmoid", "softmax", "softmax_dim", "softplus", "softsign", "strided_copy", "sum_dim", "swish",
    "tanh", "tril", "triplet_loss", "triu"}


@functools.lru_cache(None)
def _seeds() -> OpenCLSeeds:
    return OpenCLSeeds(HAND_LIB)


class SeedTest(unittest.TestCase):
    def test_every_fp32_entry_and_level_kind_has_a_seed(self):
        from modelblaster.pipeline.kernel_abi import parse_signature
        from modelblaster.pipeline.reference_kernels import KERNEL_SPECS

        s = _seeds()
        missing = [op for e, op in all_entries("fp32").items() if not s.has(e)]
        self.assertEqual(missing, [])
        for kind in LEVEL_KINDS:
            entry = parse_signature(KERNEL_SPECS[kind].signature)[0]
            src = s.slot_source(entry)
            self.assertRegex(src, rf"void {entry}\(")
            self.assertIn("MB_CL(", src)
            self.assertNotRegex(src, r"\bdouble\b")              # Adreno has no fp64

    def test_fp16_entries_follow_their_fp32_seed(self):
        s = _seeds()
        have = [op for e, op in all_entries("fp16").items() if s.has(e)]
        self.assertGreater(len(have), 80)
        src = s.slot_source("kernel_conv2d_f16")
        self.assertIn("void kernel_conv2d_f16(const __half *input", src)
        self.assertIn("__global const half *x", src)
        self.assertIn("vstore_half(acc.s + (b ? vload_half(oc, b) : 0.0f), i, y)", src)
        self.assertEqual(s.origin("kernel_softmax_f16"), "hand-f16")
        self.assertEqual(s.origin("kernel_relu_f16"), "template-f16")
        self.assertFalse(s.has("kernel_scatter_dim_f16"))

    def test_templates_use_the_signature(self):
        from modelblaster.pipeline.reference_kernels import KERNEL_SPECS
        src = template_source("hardtanh", KERNEL_SPECS["hardtanh"].signature)
        self.assertIn("__kernel void k_hardtanh(__global const float *input, __global float *output, int n, "
                      "float min_val, float max_val)", src)
        self.assertIn("fmin(fmax(v, min_val), max_val)", src)
        src = template_source("argmax_dim", KERNEL_SPECS["argmax_dim"].signature)
        self.assertIn("__global long *output", src)
        src = template_source("mse_loss", KERNEL_SPECS["mse_loss"].signature)
        self.assertIn("k_mse_loss_partial", src)
        self.assertIsNone(template_source("conv2d", KERNEL_SPECS["conv2d"].signature))

    def test_f16_transform(self):
        cl = ("__kernel void k(__global const float *x, __global float *y, __global float *part, int n) {\n"
              "    long i = MB_TID; y[i] = x[i] * 2.0f; y[i + 1] += x[x[0] > 1 ? 1 : 0];\n"
              "    part[0] = x[i] <= y[i] ? 1.0f : 0.0f;\n}\n")
        out = f16_device(cl)
        self.assertIn("__global const half *x, __global half *y, __global float *part", out)
        self.assertIn("vstore_half(vload_half(i, x) * 2.0f, i, y)", out)
        self.assertIn("vstore_half(vload_half(i + 1, y) + (vload_half(vload_half(0, x) > 1 ? 1 : 0, x)), i + 1, y)", out)
        self.assertIn("part[0] = vload_half(i, x) <= vload_half(i, y) ? 1.0f : 0.0f", out)
        host = to_f16('void kernel_a(const float *p, float *q) { kernel_a(p, q);\n'
                      '    float *part = (float *)MB_SCRATCH(c, 4 * sizeof(float)); }', {"kernel_a": "kernel_a_f16"})
        self.assertIn("void kernel_a_f16(const __half *p, __half *q) { kernel_a_f16(p, q);", host)
        self.assertIn("float *part = (float *)MB_SCRATCH(c, 4 * sizeof(float))", host)


def _cxx_ok() -> bool:
    if shutil.which("g++") is None:
        return False
    return subprocess.run(["g++", "-std=c++11", "-fsyntax-only", "-x", "c++", "-"], input="#include <CL/cl.h>\n",
                          capture_output=True, text=True).returncode == 0


@functools.lru_cache(None)
def _opencl_gpu() -> bool:
    if not _cxx_ok():
        return False
    d = Path(tempfile.mkdtemp())
    try:
        (d / "q.cpp").write_text(f'#include "{RUNTIME_H}"\nint main() {{ return mb_cl_init(); }}\n')
        if subprocess.run(["g++", "-std=c++11", "-w", "-o", str(d / "q"), str(d / "q.cpp"), "-lOpenCL"],
                          capture_output=True).returncode:
            return False
        return subprocess.run([str(d / "q")], capture_output=True, timeout=60).returncode == 0
    except Exception:
        return False
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _all_slots_tu(dtype: str) -> str:
    s = _seeds()
    parts = [RUNTIME_H.read_text()]
    for entry, op in sorted(all_entries(dtype).items()):
        if s.has(entry):
            parts.append(wrap_slot(op, s.slot_source(entry)))
    return "\n".join(parts)


@unittest.skipUnless(_cxx_ok(), "needs g++ and OpenCL headers")
class HostCompileTest(unittest.TestCase):
    def test_all_seeds_compile_in_one_tu(self):
        for dtype in ("fp32", "fp16"):
            cp = subprocess.run(["g++", "-std=c++11", "-w", "-fsyntax-only", "-x", "c++", "-"],
                                input=_all_slots_tu(dtype), capture_output=True, text=True)
            self.assertEqual(cp.returncode, 0, cp.stderr[-3000:])


@unittest.skipUnless(_opencl_gpu(), "needs g++ and an OpenCL GPU with coarse-grained SVM")
class DeviceTest(unittest.TestCase):
    def test_every_seed_program_builds(self):
        progs = []
        for dtype in ("fp32", "fp16"):
            for m in re.finditer(r'static mb_program (\w+) = MB_CL(?:_OPTS\([^,]*,)?\(R"CL\((.*?)\)CL"\)',
                                 _all_slots_tu(dtype), re.S):
                progs.append((m.group(1), m.group(2)))
        self.assertGreater(len(progs), 170)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            lines = [f'#include "{RUNTIME_H}"', "int main() {", "  if (mb_cl_init()) return 2; int bad = 0;"]
            for k, (name, src) in enumerate(progs):
                (d / f"p{k}.cl").write_text(src)
                lines.append(f'  {{ FILE *f = fopen("{d}/p{k}.cl", "rb"); static char b[1 << 16]; size_t n = fread(b, 1, sizeof b - 1, f); b[n] = 0; fclose(f);'
                             f' const char *s[2] = {{mb_cl_prelude, b}}; cl_int e; cl_program p = clCreateProgramWithSource(mb_rt.ctx, 2, s, NULL, &e);'
                             f' if (clBuildProgram(p, 1, &mb_rt.device, "-cl-std=CL1.2", NULL, NULL)) {{ printf("FAIL p{k} {name}\\n"); bad++; }} }}')
            lines += ["  return bad;", "}"]
            (d / "b.cpp").write_text("\n".join(lines))
            cp = subprocess.run(["g++", "-std=c++11", "-w", "-o", str(d / "b"), str(d / "b.cpp"), "-lOpenCL"],
                                capture_output=True, text=True)
            self.assertEqual(cp.returncode, 0, cp.stderr[-2000:])
            rp = subprocess.run([str(d / "b")], capture_output=True, text=True, timeout=1200)
            self.assertEqual(rp.returncode, 0, rp.stdout[-2000:])


MODEL = '''import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 32)
    def forward(self, x):
        return torch.softmax(torch.relu(self.fc(x)), dim=-1)
def get_inputs():
    return [torch.randn(4, 64)]
def get_init_inputs():
    return []
'''


@unittest.skipUnless(_opencl_gpu(), "needs g++ and an OpenCL GPU with coarse-grained SVM")
class EndToEndTest(unittest.TestCase):
    def test_extract_export_build_run(self):
        from modelblaster.pipeline.kb_export import export_problem
        from modelblaster.pipeline.kb_export_opencl import assemble, build_and_run, single_tu

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
            prob = export_problem(d / "ir", d / "prob", target="opencl")
            man = json.loads((prob / "manifest.json").read_text())
            self.assertEqual(man["target"], "opencl")
            self.assertEqual(sorted(s["id"] for s in man["slots"]), ["linear", "relu", "softmax"])
            self.assertTrue(all(s["file"].endswith(".cpp") for s in man["slots"]))
            tu = assemble(prob)
            self.assertIn("namespace mbslot_linear", tu)
            self.assertIn("int main(", single_tu(tu, (prob / "driver.cpp").read_text()))
            res = build_and_run(prob)
            self.assertTrue(res["passed"], res.get("log"))
            self.assertEqual(set(res["slot_us"]), {"linear", "relu", "softmax"})
            self.assertIn("relu/k_relu", res["kernel_us"])
            # a wrong slot is caught by its isolated check and attributed to it
            bad = (prob / "slots" / "relu.cpp").read_text().replace("fmax(v, 0.0f)", "fmax(v, 1.0f)")
            res = build_and_run(prob, sources={"relu": bad}, work=d / "bad")
            self.assertFalse(res["passed"])
            self.assertFalse(res["slots"]["relu"]["ok"])
            self.assertTrue(res["slots"]["linear"]["ok"])
            # a slot that does not build is reported, not a crash
            res = build_and_run(prob, sources={"softmax": (prob / "slots" / "softmax.cpp").read_text()
                                               .replace("fmax(mx, xr[k])", "fmax(mx, xr[k]) +")}, work=d / "b2")
            self.assertFalse(res["passed"])
            self.assertTrue(any("cl_build" in e for e in res["errors"]), res.get("log"))


if __name__ == "__main__":
    unittest.main()
