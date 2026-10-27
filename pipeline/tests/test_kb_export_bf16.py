"""bfloat16 on ModelBlaster's CUDA path (docs/bf16_cuda.md): the dtype contract,
the `_bf16` KernelSpecs, the generated `_bf16` hand kernels, `extract_graph
--quant bf16` (fp32 shapes, bf16-exact npz values), the host-C refusal, the
bf16-storage reference walk, dtype-keyed cache entries, and a bf16 extract ->
export -> build -> run with a wrong slot caught.

The end-to-end test needs torch, nvcc and a CUDA GPU and is skipped otherwise
(sm_75 has no bf16 arithmetic: the kernels widen to fp32, which compiles there).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline import cuda_dtypes  # noqa: E402
from modelblaster.pipeline.kb_export import DEFAULT_HAND_CU, DEFAULT_HAND_CU_BF16, HARNESS, CudaLibrary  # noqa: E402
from modelblaster.pipeline.kb_ingest import CudaCache, signature_sha  # noqa: E402
from modelblaster.pipeline.tests.test_kb_export import MODEL, _have_gpu  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]


def _extract(d: Path, quant: str, model: str = MODEL) -> Path:
    (d / "tiny.py").write_text(model)
    out = d / f"ir_{quant}"
    cp = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file",
                         str(d / "tiny.py"), "--out-dir", str(out), "--quant", quant],
                        cwd=ROOT, capture_output=True, text=True, timeout=600,
                        env={**os.environ, "PYTHONPATH": ":".join(
                            filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))})
    if cp.returncode != 0:
        raise AssertionError(cp.stderr[-2000:])
    return out


class ContractTest(unittest.TestCase):
    def test_dtypes_and_spelling(self):
        self.assertEqual(cuda_dtypes.op_dtype("conv2d_bf16"), "bf16")
        self.assertEqual(cuda_dtypes.op_dtype("kernel_relu_bf16"), "bf16")
        self.assertEqual(cuda_dtypes.op_dtype("conv2d_f16"), "fp16")          # "_bf16" is not "_f16"
        for k in ("fused__ewise2_bf16__sigmoid_bf16", "fused__ewise2_bf16__ewise2_bf16__a",
                  "kernel_fused__conv2d_bf16__ewise2_bf16__a"):
            self.assertEqual(cuda_dtypes.op_dtype(k), "bf16", k)
        self.assertEqual(cuda_dtypes.cuda_signature("void kernel_relu_bf16(const __bf16 *x, __bf16 *y, int n)"),
                         "void kernel_relu_bf16(const __nv_bfloat16 *x, __nv_bfloat16 *y, int n)")
        self.assertEqual(cuda_dtypes.ir_dtype({"quant": "bf16"}), "bf16")
        self.assertEqual(cuda_dtypes.ir_dtype({"quant": "int8"}), "fp32")
        # a bf16 IR's float arrays are float32 containers: their buffers are bf16
        self.assertEqual([cuda_dtypes.buffer_dtype(np.dtype(t), "bf16") for t in ("float32", "int64")],
                         ["bf16", "i32"])
        self.assertEqual(cuda_dtypes.buffer_dtype(np.dtype("float32")), "f32")
        self.assertEqual(cuda_dtypes.BUFFER_BYTES["bf16"], 2)
        tol = cuda_dtypes.TOLERANCE["bf16"]
        self.assertGreater(tol["slot_rel"], 2.0 ** -7)                         # >= 1 bf16 ulp of [1, 2)
        self.assertGreater(tol["golden_rel"], cuda_dtypes.TOLERANCE["fp16"]["golden_rel"])
        self.assertEqual(cuda_dtypes.TOLERANCE["fp32"]["slot_rel"], 1e-3)       # unchanged

    def test_rounding_matches_torch(self):
        import torch
        rng = np.random.default_rng(0)
        x = (rng.standard_normal(20000) * 10.0 ** rng.integers(-40, 38, 20000)).astype(np.float32)
        x = np.concatenate([x, np.array([np.inf, -np.inf, 3.39e38, -0.0, 1e-41, 2.0 ** -133], np.float32)])
        ref = torch.from_numpy(x).to(torch.bfloat16)
        r = cuda_dtypes.round_bf16(x)
        self.assertTrue(np.array_equal(r.view(np.uint32), ref.float().numpy().view(np.uint32)))
        self.assertTrue(np.array_equal(cuda_dtypes.bf16_bits(x), ref.view(torch.int16).numpy().view(np.uint16)))
        self.assertTrue(cuda_dtypes.bf16_exact(r) and not cuda_dtypes.bf16_exact(x))
        self.assertTrue(np.isnan(cuda_dtypes.round_bf16(np.array([np.nan], np.float32))).all())


class SpecTest(unittest.TestCase):
    def test_every_fp32_spec_has_a_derived_bf16_spec(self):
        from modelblaster.pipeline.kernel_abi import parse_signature
        from modelblaster.pipeline.reference_kernels import KERNEL_SPECS
        fp32 = [k for k in KERNEL_SPECS if cuda_dtypes.op_dtype(k) == "fp32" and "_s8" not in k
                and not k.startswith("cast_")]
        self.assertGreater(len(fp32), 80)
        for k in fp32:
            b = KERNEL_SPECS[k + "_bf16"]
            self.assertNotIn("float *", b.signature, k)
            self.assertNotIn("_Float16", b.signature + b.reference_impl, k)
            # same parameter names as the fp32 op (so the same IR binding), with bf16 pointers
            n32 = [p.name for p in parse_signature(KERNEL_SPECS[k].signature)[1]]
            self.assertEqual([p.name for p in parse_signature(b.signature)[1]], n32, k)
        # ops with a hand-written `_f16` spec still derive `_bf16` from fp32 (softmax_f16 has input_scale)
        self.assertNotIn("input_scale", KERNEL_SPECS["softmax_bf16"].signature)
        self.assertIn("input_scale", KERNEL_SPECS["softmax_f16"].signature)

    def test_bf16_reference_c_compiles(self):
        """The `_bf16` references are C with GCC/Clang's `__bf16` (the slots' .ref.c context)."""
        import shutil
        if shutil.which("gcc") is None:
            self.skipTest("no gcc")
        from modelblaster.pipeline.reference_kernels import KERNEL_SPECS
        ops = ("conv2d_bf16", "linear_bf16", "softmax_bf16", "layer_norm_bf16", "lstm_seq_bf16", "gather_bf16")
        src = "#include <math.h>\n#include <stdint.h>\n#include <string.h>\n#include <float.h>\n" + \
            "\n".join(KERNEL_SPECS[o].reference_impl for o in ops)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "r.c").write_text(src)
            cp = subprocess.run(["gcc", "-fsyntax-only", "-std=gnu11", str(Path(d) / "r.c")],
                                capture_output=True, text=True)
        self.assertEqual(cp.returncode, 0, cp.stderr[-2000:])

    def test_bf16_spec_signatures_bind_to_hand_kernels(self):
        """Every fp32 hand entry has a `_bf16` hand entry with its spec's (CUDA-spelled) signature."""
        from modelblaster.pipeline.reference_kernels import KERNEL_SPECS
        lib32, lib = CudaLibrary(DEFAULT_HAND_CU.read_text()), CudaLibrary(DEFAULT_HAND_CU_BF16.read_text())
        norm = lambda s: " ".join(s.replace(" *", "*").replace("* ", "*").rstrip(";").split())  # noqa: E731
        n = 0
        for op, spec in KERNEL_SPECS.items():
            if cuda_dtypes.op_dtype(op) != "bf16":
                continue
            sig = cuda_dtypes.cuda_signature(spec.signature)
            entry = sig.split("(")[0].split()[-1]
            if entry.removesuffix("_bf16") not in lib32.items:
                continue
            self.assertIn(entry, lib.items, op)
            hand = lib.items[entry].text.split("{")[0].replace('extern "C"', "")
            hand = hand[hand.index("void"):]
            self.assertEqual(norm(hand), norm(sig), op)
            n += 1
        self.assertGreater(n, 60)


class GeneratorTest(unittest.TestCase):
    def test_generated_file_is_current_and_fp32_math(self):
        sys.path.insert(0, str(HARNESS))
        import f16_kernels
        t, _, _, hand, out = f16_kernels.DTYPES["bf16"]
        text = f16_kernels.generate(f16_kernels.FP32_CU.read_text(), hand.read_text(), "bf16")
        self.assertEqual(out, DEFAULT_HAND_CU_BF16)
        self.assertEqual(text, DEFAULT_HAND_CU_BF16.read_text(), "regenerate: python gpu_eval/harness/f16_kernels.py")
        self.assertNotIn("float *", text.replace("float *sh", ""))       # no fp32 buffers left
        self.assertNotIn("__half", text)
        self.assertIn("#include <cuda_bf16.h>", text)
        self.assertIn("y[i] = mb_ld(a[i]) + mb_ld(b[i]);", text)          # loads widen, math in fp32
        # the two kernels whose fp32 code rounds an intermediate in the output are hand-written
        lib = CudaLibrary(text)
        self.assertIn("k_softmax_bf", lib.items["kernel_softmax_bf16"].text)
        self.assertIn("k_sdpa_bf", lib.items["kernel_sdpa_bf16"].text)
        # the fp16 file is still the fp16 transform
        t16 = f16_kernels.generate(f16_kernels.FP32_CU.read_text(), f16_kernels.HAND_CU.read_text())
        self.assertEqual(t16, f16_kernels.generate(f16_kernels.FP32_CU.read_text(),
                                                   f16_kernels.HAND_CU.read_text(), "f16"))

    def test_transform_rules(self):
        sys.path.insert(0, str(HARNESS))
        import f16_kernels
        src = ("#include <math.h>\n__global__ void k(const float *x, float *y, int n) {\n"
               "    float *v = y + 1, *w = y + 2; v[0] *= x[w[1] > 0]; w[0] = x[0];\n}\n"
               'extern "C" void kernel_k(const float *x, float *y, int n) { k<<<1, 1>>>(x, y, n); }\n')
        out = f16_kernels.generate(src, "", "bf16")
        self.assertIn("v[0] = mb_ld(v[0]) * (mb_ld(x[mb_ld(w[1]) > 0]));", out)
        self.assertIn("w[0] = mb_ld(x[0]);", out)
        self.assertIn('extern "C" void kernel_k_bf16(const __nv_bfloat16 *x, __nv_bfloat16 *y, int n)', out)
        self.assertIn("__bfloat162float(v)", out)


class ExtractTest(unittest.TestCase):
    def test_extract_bf16_keeps_fp32_shapes(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            ir32, irb = _extract(d, "fp32"), _extract(d, "bf16")
            g32, gb = (json.loads((p / "graph.json").read_text()) for p in (ir32, irb))
            self.assertEqual(gb["quant"], "bf16")
            self.assertEqual([o["op"] for o in gb["ops"]], [o["op"] + "_bf16" if o["op"] != "view" else "view"
                                                            for o in g32["ops"]])
            self.assertEqual([o["shape"] for o in gb["ops"]], [o["shape"] for o in g32["ops"]])
            self.assertEqual({k: v["shape"] for k, v in gb["tensors"].items()},
                             {k: v["shape"] for k, v in g32["tensors"].items()})
            self.assertEqual({v["dtype"] for v in gb["tensors"].values()}, {"bf16"})
            w32, wb = np.load(ir32 / "weights.npz"), np.load(irb / "weights.npz")
            self.assertEqual(sorted(wb.files), sorted(w32.files))
            for k in wb.files:                                    # float32 containers, bf16 values
                self.assertEqual((wb[k].dtype, wb[k].shape), (np.float32, w32[k].shape), k)
                self.assertTrue(cuda_dtypes.bf16_exact(wb[k]) and not cuda_dtypes.bf16_exact(w32[k]), k)
            io32, iob = np.load(ir32 / "io.npz"), np.load(irb / "io.npz")
            for k in ("input", "output"):
                self.assertEqual((iob[k].dtype, iob[k].shape), (np.float32, io32[k].shape))
                self.assertTrue(cuda_dtypes.bf16_exact(iob[k]), k)
            # the golden: the (fp32-weight) model on the rounded inputs, rounded once; against the
            # bf16 weights it is within the weight rounding (u * sqrt-ish K) plus one rounding
            x = iob["input"].reshape(4, 64).astype(np.float64)
            ref = np.maximum(x @ wb["fc.weight"].T.astype(np.float64) + wb["fc.bias"], 0)
            err = np.abs(iob["output"].reshape(4, 32) - ref).max() / np.abs(ref).max()
            self.assertLess(err, 2.0 ** -5)

    def test_walk_rounds_every_output_to_bf16_and_matches_the_golden(self):
        import torch
        sys.path.insert(0, str(HARNESS))
        import mbir
        import walk_torch
        with tempfile.TemporaryDirectory() as d:
            ir = mbir.load_ir(_extract(Path(d), "bf16"))
            self.assertEqual(ir.dtype, "bf16")
            with torch.no_grad():
                out, env = walk_torch.walk(ir, dev="cpu")
            for t, v in env.items():
                if torch.is_tensor(v) and v.is_floating_point():
                    self.assertTrue(cuda_dtypes.bf16_exact(v.numpy()), t)
            self.assertTrue(mbir.compare(out, ir.golden_out, ir.dtype)["ok"])

    def test_host_c_targets_refuse_bf16(self):
        from modelblaster.pipeline import generate_skeleton
        with tempfile.TemporaryDirectory() as d:
            ir = _extract(Path(d), "bf16")
            with self.assertRaises(SystemExit) as c:
                generate_skeleton.generate(str(ir / "graph.json"), str(ir / "weights.npz"), str(ir / "io.npz"),
                                           str(Path(d) / "gen"), backend="scalar", platform="host")
            self.assertIn("CUDA path only", str(c.exception))
            self.assertFalse((Path(d) / "gen" / "weights.c").exists())
            from modelblaster.pipeline.kb_export import export_problem
            with self.assertRaises(NotImplementedError):
                export_problem(ir, Path(d) / "ocl", target="opencl")


class CacheDtypeTest(unittest.TestCase):
    def test_bf16_fp16_and_fp32_entries_never_mix(self):
        sigb = "void kernel_relu_bf16(const __bf16 *input, __bf16 *output, int n)"
        with tempfile.TemporaryDirectory() as d:
            c = CudaCache(Path(d))
            e, _ = c.add(op="relu_bf16", entry="kernel_relu_bf16", signature=sigb, arch="sm_89",
                         shapes=[{"n": 8}], code="// b", slot_us=1.0, speedup=2.0, impl_id="b1", source={})
            self.assertEqual(e["dtype"], "bf16")
            self.assertEqual(e["signature_sha"], signature_sha(cuda_dtypes.cuda_signature(sigb)))
            self.assertIsNotNone(c.lookup("kernel_relu_bf16", sigb, "sm_89", [{"n": 8}]))
            self.assertIsNone(c.lookup("kernel_relu_f16", "void kernel_relu_f16(const _Float16 *input, "
                                       "_Float16 *output, int n)", "sm_89", [{"n": 8}]))


@unittest.skipUnless(_have_gpu(), "needs torch, nvcc and a CUDA GPU")
class EndToEndBf16Test(unittest.TestCase):
    def test_extract_export_build_run_bf16(self):
        from modelblaster.pipeline.kb_export import build_and_run, export_problem

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            ir = _extract(d, "bf16")
            prob = export_problem(ir, d / "prob")
            man = json.loads((prob / "manifest.json").read_text())
            self.assertEqual(man["dtype"], "bf16")
            self.assertEqual(man["tolerance"]["slot_rel"], cuda_dtypes.TOLERANCE["bf16"]["slot_rel"])
            self.assertEqual(sorted(s["id"] for s in man["slots"]), ["linear_bf16", "relu_bf16"])
            self.assertTrue(all("__nv_bfloat16" in s["signature"] for s in man["slots"]))
            self.assertEqual({b["dtype"] for b in man["buffers"]}, {"bf16"})
            self.assertIn("#include <cuda_bf16.h>", (prob / "walker_head.cu").read_text())
            # data.bin holds the bf16 bit patterns: 2 bytes per element
            b = next(b for b in man["buffers"] if b["kind"] == "weight")
            self.assertEqual(b["ref"][1], 2 * b["numel"])
            w = np.frombuffer((prob / "data.bin").read_bytes()[b["ref"][0]:b["ref"][0] + b["ref"][1]], np.uint16)
            ir_w = np.load(ir / "weights.npz")[b["name"][2:]].reshape(-1)
            self.assertTrue(np.array_equal((w.astype(np.uint32) << 16).view(np.float32), ir_w))
            arch = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.split()[0].replace(".", "")
            res = build_and_run(prob, arch=arch)
            self.assertTrue(res["passed"], res.get("log"))
            self.assertTrue(all(s["rel_err"] <= cuda_dtypes.TOLERANCE["bf16"]["slot_rel"]
                                for s in res["slots"].values()))
            # a wrong bf16 slot is caught by its isolated check and attributed to it
            bad = (prob / "slots" / "relu_bf16.cu").read_text()
            self.assertIn("fmaxf(mb_ld(x[i]), 0.0f)", bad)
            res = build_and_run(prob, arch=arch, sources={"relu_bf16": bad.replace("0.0f)", "0.1f)")},
                                work=d / "bad")
            self.assertFalse(res["passed"])
            self.assertFalse(res["slots"]["relu_bf16"]["ok"])
            self.assertTrue(res["slots"]["linear_bf16"]["ok"])
            # bf16 accumulation (the contract's main pitfall) is far worse than the fp32-math seed
            lin = (prob / "slots" / "linear_bf16.cu").read_text()
            acc = lin.replace("double acc = b ? mb_ld(b[n]) : 0.0f;",
                              "__nv_bfloat16 acc = b ? b[n] : __float2bfloat16(0.0f);") \
                     .replace("acc += mb_ld(x[(long)m * K + k]) * mb_ld(w[(long)n * K + k]);",
                              "acc = __float2bfloat16(__bfloat162float(acc) + mb_ld(x[(long)m * K + k]) "
                              "* mb_ld(w[(long)n * K + k]));") \
                     .replace("y[i] = acc;", "y[i] = __bfloat162float(acc);")
            self.assertNotEqual(acc, lin)
            res = build_and_run(prob, arch=arch, sources={"linear_bf16": acc}, work=d / "accb")
            self.assertEqual(res.get("stage"), "run", res.get("log"))
            seed_err = build_and_run(prob, arch=arch, work=d / "seed")["slots"]["linear_bf16"]["rel_err"]
            self.assertGreater(res["slots"]["linear_bf16"]["rel_err"], seed_err)


if __name__ == "__main__":
    unittest.main()
