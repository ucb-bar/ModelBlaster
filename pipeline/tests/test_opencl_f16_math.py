"""fp16-math OpenCL seeds (cl_khr_fp16, Adreno) and their switch: MB_OPENCL_F16_MATH.

CPU-only: seed generation, the switch and the legacy-identity guarantee, the runtime header's build-option logic (text),
the numerics policy of the conv (emulated with numpy float16 / float32), and -- when an Apple/LLVM clang with OpenCL
support is installed -- a syntax check of every generated OpenCL C program with cl_khr_fp16 enabled. The board validation
(correctness gate, times) is in the REPORT of the change, not here.
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
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline.kb_export_opencl import HAND_LIB, RUNTIME_H, OpenCLTarget  # noqa: E402
from modelblaster.pipeline.opencl_seeds import (F16_MATH_ARCHS, OpenCLSeeds, all_entries,  # noqa: E402
                                                f16_math_enabled)

F16_MATH_FILE = HAND_LIB.with_name("mb_opencl_kernels_f16_math.cpp")
HAND_COVERED = ("kernel_conv2d_f16", "kernel_matmul_f16", "kernel_linear_f16", "kernel_batchnorm2d_f16", "kernel_ewise2_f16")
MAP_COVERED = ("kernel_relu_f16", "kernel_add_f16", "kernel_mul_f16", "kernel_gelu_f16", "kernel_sigmoid_f16",
               "kernel_leaky_relu_f16", "kernel_hardtanh_f16", "kernel_mul_scalar_f16", "kernel_relu6_f16",
               "kernel_ewise1_f16", "kernel_ewise_scalar_f16")


def _clang_cl() -> str | None:
    cc = shutil.which("clang")
    if cc is None:
        return None
    r = subprocess.run([cc, "-x", "cl", "-cl-std=CL2.0", "-Xclang", "-finclude-default-header",
                        "-Xclang", "-cl-ext=+cl_khr_fp16", "-fsyntax-only", "-"], input="__kernel void k(){}\n",
                       capture_output=True, text=True)
    return cc if r.returncode == 0 else None


def _programs(src: str) -> list[str]:
    return re.findall(r'R"CL\((.*?)\)CL"', src, re.S)


class SwitchTest(unittest.TestCase):
    def test_default_on_for_adreno_only(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MB_OPENCL_F16_MATH", None)
            os.environ.pop("MB_OPENCL_ARCH", None)
            self.assertFalse(f16_math_enabled())                       # no device label: the NVIDIA stand-in path
            self.assertFalse(f16_math_enabled("titanrtx"))
            self.assertFalse(f16_math_enabled("l40s"))
            self.assertTrue(f16_math_enabled("adreno650"))
            os.environ["MB_OPENCL_ARCH"] = "adreno650"
            self.assertTrue(f16_math_enabled())
            os.environ["MB_OPENCL_ARCH"] = "titanrtx"
            self.assertFalse(f16_math_enabled())
        self.assertIn("adreno650", F16_MATH_ARCHS)

    def test_explicit_values_win(self):
        for v in ("0", "off", "false", "no"):
            with mock.patch.dict(os.environ, {"MB_OPENCL_F16_MATH": v}):
                self.assertFalse(f16_math_enabled("adreno650"))
        for v in ("1", "on", "true", "yes"):
            with mock.patch.dict(os.environ, {"MB_OPENCL_F16_MATH": v}):
                self.assertTrue(f16_math_enabled("titanrtx"))
        with mock.patch.dict(os.environ, {"MB_OPENCL_F16_MATH": "auto"}):
            self.assertTrue(f16_math_enabled("adreno650"))
            self.assertFalse(f16_math_enabled("titanrtx"))


class SeedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.off = OpenCLSeeds(HAND_LIB, f16_math=False)
        cls.on = OpenCLSeeds(HAND_LIB, f16_math=True)

    def test_off_is_the_storage_only_seed(self):
        for e in HAND_COVERED + MAP_COVERED:
            src = self.off.slot_source(e)
            self.assertIn("vload_half", src, e) if e not in ("kernel_ewise_scalar_f16",) else None
            self.assertNotIn("cl_khr_fp16", src, e)
            self.assertFalse(self.off.origin(e).endswith("f16math"), e)

    def test_default_without_arch_is_off(self):
        with mock.patch.dict(os.environ, {}):
            os.environ.pop("MB_OPENCL_F16_MATH", None)
            os.environ.pop("MB_OPENCL_ARCH", None)
            d = OpenCLSeeds(HAND_LIB)
            for e in HAND_COVERED + MAP_COVERED:
                self.assertEqual(d.slot_source(e), self.off.slot_source(e), e)
            os.environ["MB_OPENCL_ARCH"] = "adreno650"
            self.assertNotEqual(OpenCLSeeds(HAND_LIB).slot_source("kernel_conv2d_f16"),
                                self.off.slot_source("kernel_conv2d_f16"))

    def test_on_uses_half_math_and_direct_vectors(self):
        for e in HAND_COVERED + MAP_COVERED:
            src = self.on.slot_source(e)
            self.assertIn("#pragma OPENCL EXTENSION cl_khr_fp16 : enable", src, e)
            self.assertRegex(src, r"vload8|vload4", e)
            self.assertNotIn("vload_half", src, e)
            self.assertNotIn("vstore_half", src, e)
            self.assertTrue(self.on.origin(e).endswith("f16math"), e)
            self.assertRegex(src, rf"void {e}\(", e)                    # the KernelSpec entry point is unchanged
        conv = self.on.slot_source("kernel_conv2d_f16")
        self.assertIn("MB_ACC_HALF", conv)                              # the documented fp16-accumulate switch
        self.assertIn("MB_ACC_FLUSH", conv)
        self.assertIn("fvec a0", conv)                                  # fp32 accumulators by default (fvec = float8 / float4)
        self.assertIn("-DMB_OT=8", conv)

    def test_signatures_match_the_storage_only_seed(self):
        """Same host signature => same slot ABI / hash as the CUDA fp16 problems and the legacy seeds."""
        sig = re.compile(r"^void (kernel_\w+_f16)\(([^)]*)\)", re.M)
        for e in HAND_COVERED + MAP_COVERED:
            a = sig.findall(self.on.slot_source(e))
            b = sig.findall(self.off.slot_source(e))
            norm = lambda l: [(n, re.sub(r"\s*\*\s*", "*", re.sub(r"\s+", " ", p)).strip()) for n, p in l if n == e]
            self.assertEqual(norm(a), norm(b), e)

    def test_uncovered_ops_keep_their_storage_only_seed(self):
        for e, op in all_entries("fp16").items():
            if e in HAND_COVERED + MAP_COVERED or not self.on.has(e):
                continue
            if self.on.origin(e).endswith("f16math"):
                continue                                                # another math template: fine
            self.assertEqual(self.on.slot_source(e), self.off.slot_source(e), e)

    def test_every_fp16_entry_still_has_a_seed(self):
        have_off = {e for e in all_entries("fp16") if self.off.has(e)}
        have_on = {e for e in all_entries("fp16") if self.on.has(e)}
        self.assertEqual(have_off, have_on)
        for e in have_on:
            self.on.slot_source(e)                                      # generation does not raise

    def test_fp32_entries_are_untouched(self):
        for e in ("kernel_conv2d", "kernel_matmul", "kernel_relu", "kernel_batchnorm2d", "kernel_add"):
            self.assertEqual(self.on.slot_source(e), self.off.slot_source(e), e)

    def test_fused_seed_follows_the_switch(self):
        from modelblaster.pipeline.kb_proposals import _all_specs
        specs = _all_specs()
        self.assertIn("conv2d_f16", specs)                              # stages come from the per-stage seed sources
        src = self.on.slot_source("kernel_conv2d_f16")
        self.assertIn("MB_SCRATCH(cl_conv2d", src)


@unittest.skipUnless(_clang_cl(), "needs an LLVM clang with OpenCL C support (-x cl)")
class OpenCLCSyntaxTest(unittest.TestCase):
    """Every generated fp16-math program compiles as OpenCL C 2.0 with cl_khr_fp16 (clang's front end, not Adreno's)."""

    def test_programs_parse(self):
        rt = RUNTIME_H.read_text()
        prelude = re.search(r'mb_cl_prelude = R"CL\((.*?)\)CL";', rt, re.S).group(1)
        on = OpenCLSeeds(HAND_LIB, f16_math=True)
        bad = []
        for e, op in sorted(all_entries("fp16").items()):
            if not on.has(e):
                continue
            for k, code in enumerate(_programs(on.slot_source(e))):
                with tempfile.NamedTemporaryFile("w", suffix=".cl", delete=False) as t:
                    t.write("#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n#define MB_HAS_FP16 1\n" + prelude + code)
                r = subprocess.run([_clang_cl(), "-x", "cl", "-cl-std=CL2.0", "-Xclang", "-finclude-default-header",
                                    "-Xclang", "-cl-ext=+cl_khr_fp16", "-fsyntax-only", t.name],
                                   capture_output=True, text=True)
                os.unlink(t.name)
                if r.returncode:
                    bad.append((e, k, r.stderr[:400]))
        self.assertEqual(bad, [])

    def test_fp16_accumulate_switch_parses(self):
        rt = RUNTIME_H.read_text()
        prelude = re.search(r'mb_cl_prelude = R"CL\((.*?)\)CL";', rt, re.S).group(1)
        code = _programs(OpenCLSeeds(HAND_LIB, f16_math=True).slot_source("kernel_conv2d_f16"))[0]
        for opts in (["-DMB_ACC_HALF=1", "-DMB_ACC_FLUSH=16"], ["-DMB_ACC_HALF=1", "-DMB_ACC_FLUSH=0"]):
            with tempfile.NamedTemporaryFile("w", suffix=".cl", delete=False) as t:
                t.write("#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" + prelude + code)
            r = subprocess.run([_clang_cl(), "-x", "cl", "-cl-std=CL2.0", "-Xclang", "-finclude-default-header",
                                "-Xclang", "-cl-ext=+cl_khr_fp16", "-fsyntax-only", *opts, t.name],
                               capture_output=True, text=True)
            os.unlink(t.name)
            self.assertEqual(r.returncode, 0, r.stderr[:600])


class RuntimeHeaderTest(unittest.TestCase):
    def test_std_is_no_longer_hard_coded(self):
        rt = RUNTIME_H.read_text()
        self.assertNotIn('"-cl-std=CL1.2 "', rt)
        self.assertIn("MB_CL_STD", rt)                                  # $MB_CL_STD override
        self.assertIn("CL_DEVICE_OPENCL_C_VERSION", rt)                 # auto: the device's OpenCL C
        self.assertIn("MB_CL_FP16", rt)                                 # $MB_CL_FP16 override of the prelude
        self.assertIn("MB_SCRATCH_IMAGE", rt)                           # the slot-owned image2d helper
        # no direct enqueue / create calls were added for slots: images go through the runtime helper
        self.assertIn("mb_scratch_image", rt)

    def test_head_source_follows_the_seed_switch(self):
        tgt = OpenCLTarget()
        legacy = tgt.head_source(OpenCLSeeds(HAND_LIB, f16_math=False))
        new = tgt.head_source(OpenCLSeeds(HAND_LIB, f16_math=True))
        self.assertIn('#define MB_CL_STD_DEFAULT "CL1.2"', legacy)
        self.assertIn("#define MB_NO_FP16_PRELUDE 1", legacy)
        self.assertNotIn("#define MB_NO_FP16_PRELUDE", new)
        self.assertNotIn("#define MB_CL_STD_DEFAULT", new.split("static const char *mb_cl_fp16_prelude")[0])
        self.assertTrue(legacy.endswith(RUNTIME_H.read_text()))
        self.assertTrue(new.endswith(RUNTIME_H.read_text()))

    def test_slot_text_still_has_no_direct_enqueue(self):
        on = OpenCLSeeds(HAND_LIB, f16_math=True)
        for e in HAND_COVERED + MAP_COVERED:
            self.assertNotRegex(on.slot_source(e), r"clEnqueue|clFinish|clCreate", e)


class NumericsPolicyTest(unittest.TestCase):
    """The accumulation policy of the conv seed, emulated on the CPU with the flow's gate (about two fp16 ulps of the
    largest magnitude: slot_rel 2e-3). fp16 operands + fp32 accumulators (default) must pass; pure fp16 accumulation is
    allowed only where it measures within the bound -- this test pins the ORDER of the three policies, not a hard pass."""

    def test_accumulation_policies(self):
        try:
            import numpy as np
        except ImportError:                                             # pragma: no cover
            self.skipTest("numpy")
        rng = np.random.RandomState(0)
        K, P = 576, 64                                                  # taps of a 64-channel 3x3 conv, 64 outputs
        x = rng.uniform(-1, 1, (P, K)).astype(np.float16)
        w = (rng.uniform(-1, 1, K) * 0.1).astype(np.float16)
        ref = (x.astype(np.float64) * w.astype(np.float64)).sum(1)
        scale = np.abs(ref).max()
        ref16 = ref.astype(np.float16).astype(np.float64)

        def fp32_acc():
            acc = np.zeros(P, np.float32)
            for k in range(K):
                acc = (acc + x[:, k].astype(np.float32) * w[k].astype(np.float32)).astype(np.float32)
            return acc.astype(np.float16).astype(np.float64)

        def fp16_acc(flush):
            acc, part, n = np.zeros(P, np.float32), np.zeros(P, np.float16), 0
            for k in range(K):
                part = (part + (x[:, k] * w[k]).astype(np.float16)).astype(np.float16)   # mad: product + add in half
                n += 1
                if flush and n == flush:
                    acc += part.astype(np.float32)
                    part, n = np.zeros(P, np.float16), 0
            return (acc + part.astype(np.float32)).astype(np.float16).astype(np.float64)

        e32 = np.abs(fp32_acc() - ref16).max() / scale
        e16f = np.abs(fp16_acc(16) - ref16).max() / scale
        e16 = np.abs(fp16_acc(0) - ref16).max() / scale
        self.assertLess(e32, 1e-3)                                      # default: within one fp16 ulp-ish
        self.assertLessEqual(e32, e16f + 1e-12)
        self.assertLess(e16f, e16)                                      # flushing bounds the drift of pure fp16 accumulation
        self.assertLess(e32, 2e-3)                                      # the flow's per-slot bound


if __name__ == "__main__":
    unittest.main()
