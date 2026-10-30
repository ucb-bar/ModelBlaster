"""fp16 (IEEE half) on ModelBlaster's CUDA path: the dtype contract, the
generated `_f16` hand kernels, dtype-keyed cache entries, and an fp16
extract -> export -> build -> run with a wrong slot caught.

The end-to-end test needs torch, nvcc and a CUDA GPU and is skipped otherwise.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline import cuda_dtypes  # noqa: E402
from modelblaster.pipeline.kb_export import DEFAULT_HAND_CU_F16, HARNESS, CudaLibrary  # noqa: E402
from modelblaster.pipeline.kb_ingest import CudaCache, signature_sha  # noqa: E402
from modelblaster.pipeline.tests.test_kb_export import MODEL, _have_gpu  # noqa: E402


class ContractTest(unittest.TestCase):
    def test_dtypes_and_spelling(self):
        self.assertEqual(cuda_dtypes.op_dtype("conv2d_f16"), "fp16")
        self.assertEqual(cuda_dtypes.op_dtype("conv2d"), "fp32")
        # fused kinds follow their stages, also when the name ends in the param that reads
        # the intermediate (these were fp32: the variant compiled without cuda_fp16.h)
        for k in ("fused__ewise2_f16__sigmoid_f16", "fused__ewise2_f16__ewise2_f16__a",
                  "fused__conv2d_f16__ewise2_f16__a", "kernel_fused__ewise2_f16__ewise2_f16__a"):
            self.assertEqual(cuda_dtypes.op_dtype(k), "fp16", k)
        for k in ("fused__ewise2__ewise2__a", "fused__add__relu", "kernel_fused__ewise2__ewise2__a"):
            self.assertEqual(cuda_dtypes.op_dtype(k), "fp32", k)
        self.assertEqual(cuda_dtypes.cuda_signature("void kernel_relu_f16(const _Float16 *x, _Float16 *y, int n)"),
                         "void kernel_relu_f16(const __half *x, __half *y, int n)")
        import numpy as np
        self.assertEqual([cuda_dtypes.buffer_dtype(np.dtype(t)) for t in ("float16", "float32", "int64")],
                         ["f16", "f32", "i32"])
        # the fp32 bounds are the Level2/3 gates', unchanged
        self.assertEqual(cuda_dtypes.TOLERANCE["fp32"]["slot_rel"], 1e-3)
        self.assertGreater(cuda_dtypes.TOLERANCE["fp16"]["slot_rel"], 2.0 ** -10)   # >= 1 fp16 ulp

    def test_f16_spec_signatures_bind(self):
        """Every `_f16` op Level3 uses has a hand CUDA entry with its spec's signature."""
        from modelblaster.pipeline.reference_kernels import KERNEL_SPECS
        src = DEFAULT_HAND_CU_F16.read_text()
        lib = CudaLibrary(src)
        for op in ("conv2d_f16", "linear_f16", "softmax_f16", "adaptive_avg_pool2d_f16", "cat2_c1_f16",
                   "lstm_seq_f16", "gru_seq_f16", "strided_copy_f16", "gather_f16", "catn_f16"):
            sig = cuda_dtypes.cuda_signature(KERNEL_SPECS[op].signature)
            entry = sig.split("(")[0].split()[-1]
            self.assertIn(entry, lib.items, op)
            hand = " ".join(lib.items[entry].text.split("{")[0].replace('extern "C"', "").split())
            norm = lambda s: s.replace(" *", "*").replace("* ", "*").rstrip(";").strip()  # noqa: E731
            self.assertEqual(norm(hand), norm(" ".join(sig.split())), op)


class GeneratorTest(unittest.TestCase):
    def test_generated_file_is_current_and_fp32_math(self):
        sys.path.insert(0, str(HARNESS))
        import f16_kernels
        text = f16_kernels.generate(f16_kernels.FP32_CU.read_text(), f16_kernels.HAND_CU.read_text())
        self.assertEqual(text, DEFAULT_HAND_CU_F16.read_text(), "regenerate: python gpu_eval/harness/f16_kernels.py")
        self.assertNotIn("float *", text.replace("float *sh", ""))       # no fp32 buffers left
        self.assertIn("y[i] = mb_ld(a[i]) + mb_ld(b[i]);", text)          # loads widen, math in fp32

    def test_transform_rules(self):
        sys.path.insert(0, str(HARNESS))
        import f16_kernels
        src = ("#include <math.h>\n__global__ void k(const float *x, float *y, int n) {\n"
               "    float *v = y + 1, *w = y + 2; v[0] *= x[w[1] > 0]; w[0] = x[0];\n}\n"
               'extern "C" void kernel_k(const float *x, float *y, int n) { k<<<1, 1>>>(x, y, n); }\n')
        out = f16_kernels.generate(src, "")
        self.assertIn("v[0] = mb_ld(v[0]) * (mb_ld(x[mb_ld(w[1]) > 0]));", out)
        self.assertIn("w[0] = mb_ld(x[0]);", out)
        self.assertIn('extern "C" void kernel_k_f16(const __half *x, __half *y, int n)', out)


class CacheDtypeTest(unittest.TestCase):
    def test_fp16_and_fp32_entries_never_mix(self):
        sig16 = "void kernel_relu_f16(const _Float16 *input, _Float16 *output, int n)"
        with tempfile.TemporaryDirectory() as d:
            c = CudaCache(Path(d))
            e, _ = c.add(op="relu_f16", entry="kernel_relu_f16", signature=sig16, arch="sm_75",
                         shapes=[{"n": 8}], code="// h", slot_us=1.0, speedup=2.0, impl_id="h1", source={})
            self.assertEqual(e["dtype"], "fp16")
            # keyed by the CUDA spelling, which is what KernelBlaster hashes from the manifest
            self.assertEqual(e["signature_sha"], signature_sha(cuda_dtypes.cuda_signature(sig16)))
            self.assertIsNotNone(c.lookup("kernel_relu_f16", sig16, "sm_75", [{"n": 8}]))
            self.assertIsNone(c.lookup("kernel_relu", "void kernel_relu(const float *input, float *output, int n)",
                                       "sm_75", [{"n": 8}]))


@unittest.skipUnless(_have_gpu(), "needs torch, nvcc and a CUDA GPU")
class EndToEndFp16Test(unittest.TestCase):
    def test_extract_export_build_run_fp16(self):
        import numpy as np

        from modelblaster.pipeline.kb_export import build_and_run, export_problem

        root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "tiny.py").write_text(MODEL)
            cp = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file",
                                 str(d / "tiny.py"), "--out-dir", str(d / "ir"), "--quant", "fp16"],
                                cwd=root, capture_output=True, text=True, timeout=600,
                                env={**os.environ, "PYTHONPATH": ":".join(
                                    filter(None, [str(root), os.environ.get("PYTHONPATH")]))})
            self.assertEqual(cp.returncode, 0, cp.stderr[-2000:])
            prob = export_problem(d / "ir", d / "prob")
            man = json.loads((prob / "manifest.json").read_text())
            self.assertEqual(man["dtype"], "fp16")
            self.assertEqual(man["tolerance"]["slot_rel"], cuda_dtypes.TOLERANCE["fp16"]["slot_rel"])
            self.assertEqual(sorted(s["id"] for s in man["slots"]), ["linear_f16", "relu_f16"])
            self.assertTrue(all("__half" in s["signature"] and "_Float16" not in s["signature"]
                                for s in man["slots"]))
            self.assertEqual({b["dtype"] for b in man["buffers"]}, {"f16"})
            # data.bin holds the fp16 values: 2 bytes per element
            b = next(b for b in man["buffers"] if b["kind"] == "weight")
            self.assertEqual(b["ref"][1], 2 * b["numel"])
            w = np.frombuffer((prob / "data.bin").read_bytes()[b["ref"][0]:b["ref"][0] + b["ref"][1]], np.float16)
            ir_w = np.load(d / "ir" / "weights.npz")[b["name"][2:]].reshape(-1)
            self.assertTrue(np.array_equal(w, ir_w))
            arch = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.split()[0].replace(".", "")
            res = build_and_run(prob, arch=arch)
            self.assertTrue(res["passed"], res.get("log"))
            self.assertTrue(all(s["rel_err"] <= cuda_dtypes.TOLERANCE["fp16"]["slot_rel"]
                                for s in res["slots"].values()))
            # a wrong fp16 slot is caught by its isolated check and attributed to it
            bad = (prob / "slots" / "relu_f16.cu").read_text()
            self.assertIn("fmaxf(mb_ld(x[i]), 0.0f)", bad)
            res = build_and_run(prob, arch=arch, sources={"relu_f16": bad.replace("0.0f)", "0.01f)")},
                                work=d / "bad")
            self.assertFalse(res["passed"])
            self.assertFalse(res["slots"]["relu_f16"]["ok"])
            self.assertTrue(res["slots"]["linear_f16"]["ok"])
            # fp16 accumulation (the contract's main pitfall) is measurably worse than the
            # seed; at this K=64 it still fits the bound (which judges the result, not the method)
            lin = (prob / "slots" / "linear_f16.cu").read_text()
            acc16 = lin.replace("double acc = b ? mb_ld(b[n]) : 0.0f;", "__half acc = b ? b[n] : __float2half(0.0f);") \
                       .replace("acc += mb_ld(x[(long)m * K + k]) * mb_ld(w[(long)n * K + k]);",
                                "acc = __hadd(acc, __hmul(x[(long)m * K + k], w[(long)n * K + k]));")
            self.assertNotEqual(acc16, lin)
            res = build_and_run(prob, arch=arch, sources={"linear_f16": acc16}, work=d / "acc16")
            self.assertEqual(res.get("stage"), "run", res.get("log"))
            seed_err = build_and_run(prob, arch=arch, work=d / "seed")["slots"]["linear_f16"]["rel_err"]
            self.assertGreater(res["slots"]["linear_f16"]["rel_err"], seed_err)


if __name__ == "__main__":
    unittest.main()
