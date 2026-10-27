"""Per-tensor dtypes (docs/tensor_dtypes.md): `extract_graph --quant native` on a
mixed-dtype model (bf16 / fp32 / int64 tensors), its casts, integer io and
unrounded fp32 constants; the reference walk; the multi-slot export of a
mixed IR (buffer dtypes, data.bin layout, per-buffer / per-op bounds, slot
dtypes, the shared head); the cast kernels; fusion refusing a dtype change;
and that the single-dtype presets are unchanged.

The build test needs nvcc (compile only; nothing runs on a GPU) and is skipped
otherwise.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline import cuda_dtypes  # noqa: E402
from modelblaster.pipeline.kb_export import DEFAULT_HAND_CU, CudaLibrary  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
CAST_CU = DEFAULT_HAND_CU.with_name("mb_cuda_kernels_cast.cu")

# A SOL-style bench: the native model works in bf16 and fp32 and reads int64
# indices past bf16's exact range (256); NativeModel / get_native_inputs() are
# what --quant native traces, Model / get_inputs() the fp32 variant.
MIXED = '''
import torch
import torch.nn as nn

N, D = 6, 16


class NativeModel(nn.Module):
    LOW = torch.bfloat16

    def forward(self, x, w, idx):
        inv = 1.0 / (10000.0 ** (torch.arange(0, D, dtype=torch.float32) / D))   # folded, fp32
        h = x.float() * w + inv                      # fp32 math on a bf16 input
        y = h.to(self.LOW)                           # stored bf16
        z = torch.matmul(y, y.transpose(0, 1))       # a bf16 op (fp32 accumulation, one rounding)
        p = idx.float() * 0.5                        # integer-valued input as fp32
        return z, h.sum(dim=-1), idx + 1, p


class Model(NativeModel):                            # the fp32 variant (dtypes rewritten)
    LOW = torch.float32


def get_native_inputs():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(N, D, generator=g).to(torch.bfloat16)
    w = torch.randn(N, D, generator=g)
    idx = torch.arange(N, dtype=torch.int64) * 1001 + 300   # 300 .. 5305: not bf16-exact
    return [x, w, idx]


def get_inputs():
    return [t.float() if t.is_floating_point() else t for t in get_native_inputs()]


def get_init_inputs():
    return []
'''


def _extract(d: Path, quant: str, model: str = MIXED) -> Path:
    (d / "mixed.py").write_text(model)
    out = d / f"ir_{quant}"
    cp = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file",
                         str(d / "mixed.py"), "--bench-target-mb", "0", "--out-dir", str(out), "--quant", quant],
                        cwd=ROOT, capture_output=True, text=True, timeout=600,
                        env={**os.environ, "PYTHONPATH": ":".join(
                            filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))})
    if cp.returncode != 0:
        raise AssertionError(cp.stderr[-3000:])
    return out


class ContractTest(unittest.TestCase):
    def test_helpers(self):
        g = {"quant": "native", "tensors": {"a": {"dtype": "bf16"}, "b": {"dtype": "f32"}, "i": {"dtype": "i64"}},
             "weight_dtypes": {"w": "f32"}, "ops": [
                 {"op": "cast_bf16_to_f32", "inputs": ["a"], "outputs": ["a32"]},
                 {"op": "add", "inputs": ["a32", "b"], "outputs": ["c"]},
                 {"op": "relu", "inputs": ["b"], "outputs": ["d"]}]}
        g["tensors"].update({"a32": {"dtype": "f32"}, "c": {"dtype": "f32"}, "d": {"dtype": "f32"}})
        self.assertEqual(cuda_dtypes.ir_dtype(g), "bf16")                # its narrowest float storage
        self.assertEqual(cuda_dtypes.ir_dtype({"quant": "native", "tensors": {"x": {"dtype": "f32"}}}), "fp32")
        self.assertTrue(cuda_dtypes.is_mixed(g) and not cuda_dtypes.is_mixed({"quant": "bf16"}))
        # an fp32 output computed through a bf16 tensor gets bf16's bound, one of fp32 math fp32's
        self.assertEqual(cuda_dtypes.cone_dtypes(g, ["c", "d"]), {"c": "bf16", "d": "fp32"})
        self.assertEqual(cuda_dtypes.parse_cast("cast_bf16_to_f32"), ("bf16", "f32"))
        self.assertEqual(cuda_dtypes.parse_cast("kernel_cast_f32_to_i64"), ("f32", "i64"))
        self.assertIsNone(cuda_dtypes.parse_cast("cast_i8_to_f16"))         # the int8 path's
        self.assertIsNone(cuda_dtypes.parse_cast("relu_bf16"))
        # a cast's kind ends in its destination: op_dtype is what it stores
        self.assertEqual(cuda_dtypes.op_dtype("cast_f32_to_bf16"), "bf16")
        self.assertEqual(cuda_dtypes.op_dtype("cast_bf16_to_f32"), "fp32")
        self.assertEqual(cuda_dtypes.op_dtype("cast_bf16_to_f16"), "fp16")
        self.assertEqual(cuda_dtypes.weight_dtype(g, "w", np.dtype("float32")), "f32")
        self.assertEqual(cuda_dtypes.weight_dtype({"quant": "bf16"}, "w", np.dtype("float32")), "bf16")
        self.assertEqual(cuda_dtypes.weight_dtype({"quant": "fp32"}, "k", np.dtype("int64")), "i32")
        self.assertEqual(cuda_dtypes.BUFFER_BYTES["b8"], 1)
        x = np.array([1.0 + 2.0 ** -12, 300.0, -0.0], np.float32)
        self.assertEqual(cuda_dtypes.np_storage(x, "bf16").tolist(), [1.0, 300.0, -0.0])
        self.assertEqual(cuda_dtypes.np_storage(x, "f32").tolist(), x.tolist())
        self.assertEqual(cuda_dtypes.np_storage(np.array([300, 5305]), "i64").dtype, np.int64)

    def test_cast_specs_and_kernels(self):
        """Every cast has a spec (excluded from the `_f16` / `_bf16` derivation) and a hand entry with
        its CUDA-spelled signature; the reference C compiles with GCC's __bf16."""
        from modelblaster.pipeline.reference_kernels import KERNEL_SPECS
        lib = CudaLibrary(CAST_CU.read_text())
        norm = lambda s: " ".join(s.replace(" *", "*").replace("* ", "*").rstrip(";").split())  # noqa: E731
        for a, b in cuda_dtypes.CASTS:
            k = cuda_dtypes.cast_kind(a, b)
            spec = KERNEL_SPECS[k]
            self.assertNotIn(k + "_bf16", KERNEL_SPECS)
            self.assertNotIn(k + "_f16", KERNEL_SPECS)
            entry = f"kernel_{k}"
            text = lib.items[entry].text
            sig = re.search(r'extern "C" (void kernel_\w+\([^)]*\))', text).group(1)
            self.assertEqual(norm(sig), norm(cuda_dtypes.cuda_signature(spec.signature)), k)
        from modelblaster.pipeline.op_classes import op_class
        self.assertEqual(op_class("cast_f32_to_bf16"), "elementwise")
        if shutil.which("gcc") is None:
            self.skipTest("no gcc")
        src = "#include <stdint.h>\n" + "\n".join(
            KERNEL_SPECS[cuda_dtypes.cast_kind(a, b)].reference_impl for a, b in cuda_dtypes.CASTS)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "c.c").write_text(src)
            cp = subprocess.run(["gcc", "-fsyntax-only", "-std=gnu11", str(Path(d) / "c.c")],
                                capture_output=True, text=True)
        self.assertEqual(cp.returncode, 0, cp.stderr[-2000:])


class ExtractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = Path(cls.tmp.name)
        cls.ir = _extract(cls.d, "native")
        cls.g = json.loads((cls.ir / "graph.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _dt(self, t):
        return self.g["tensors"][t]["dtype"]

    def test_io_dtypes_are_the_sources(self):
        g = self.g
        ins = [p["name"] for p in g["input"]["packed_inputs"]]
        self.assertEqual([self._dt(t) for t in ins], ["bf16", "f32", "i64"])
        self.assertEqual([self._dt(t) for t in g["output"]["tensors"]], ["bf16", "f32", "i64", "f32"])
        self.assertEqual(g["quant"], "native")

    def test_ops_have_one_storage_dtype_and_casts_between(self):
        g = self.g
        kinds = [o["op"] for o in g["ops"]]
        for k in ("cast_bf16_to_f32", "cast_f32_to_bf16", "cast_i64_to_f32", "cast_f32_to_i64"):
            self.assertIn(k, kinds)
        self.assertTrue(any(k.endswith("_bf16") and k.startswith(("bmm", "matmul")) for k in kinds), kinds)
        for o in g["ops"]:
            if o["op"] in ("view", "chunk2_c1"):
                self.assertEqual({self._dt(t) for t in o["inputs"] + o["outputs"]}.__len__(), 1, o)
                continue
            src_dst = cuda_dtypes.parse_cast(o["op"])
            if src_dst:
                self.assertEqual((self._dt(o["inputs"][0]), self._dt(o["outputs"][0])), src_dst)
                continue
            want = {"fp32": "f32", "fp16": "f16", "bf16": "bf16"}[cuda_dtypes.op_dtype(o["op"])]
            for t in o["inputs"] + o["outputs"]:
                if not g["tensors"][t].get("const") and self._dt(t) not in ("i64",):
                    self.assertEqual(self._dt(t), want, (o["op"], t))

    def test_integer_io_is_exact_and_constants_are_not_rounded(self):
        io = np.load(self.ir / "io.npz")
        g = self.g
        idx = g["input"]["packed_inputs"][2]["name"]
        a = io[f"in.{idx}"]
        self.assertEqual(a.dtype, np.int64)
        self.assertEqual(a.tolist(), [i * 1001 + 300 for i in range(6)])           # > 256: bf16 would round
        out_idx = g["output"]["tensors"][2]
        self.assertEqual(io[f"out.{out_idx}"].dtype, np.int64)
        self.assertEqual(io[f"out.{out_idx}"].tolist(), [i * 1001 + 301 for i in range(6)])
        # the flat float32 io holds the same values (the legacy readers' view)
        self.assertEqual(io["input"].size, 6 * 16 * 2 + 6)
        w = np.load(self.ir / "weights.npz")
        fl = [k for k in w.files if w[k].dtype.kind == "f"]
        self.assertTrue(fl)
        for k in fl:
            self.assertEqual(g["weight_dtypes"][k], "f32")
            self.assertFalse(cuda_dtypes.bf16_exact(w[k]), k)                    # inv_freq kept fp32

    def test_walk_matches_golden_and_bf16_preset_rounds_ints(self):
        import torch
        sys.path.insert(0, str(ROOT / "gpu_eval" / "harness"))
        import mbir
        import walk_torch
        ir = mbir.load_ir(self.ir)
        with torch.no_grad():
            out, env = walk_torch.walk(ir, "cpu")
        r = mbir.compare(out, ir.golden_out, ir.dtype)
        self.assertTrue(r["ok"], r)
        self.assertEqual(ir.dtype, "bf16")
        # the integer output is exact; the fp32 sum is fp32-exact against the golden (no bf16 store)
        n = [int(np.prod(ir.tshape(t))) for t in ir.output_names()]
        o = out.numpy()
        off = np.cumsum([0] + n)
        np.testing.assert_array_equal(o[off[2]:off[3]], ir.golden_out[off[2]:off[3]])
        s32 = np.abs(o[off[1]:off[2]] - ir.golden_out[off[1]:off[2]]).max()
        self.assertLess(s32, 1e-4)
        # the single-dtype preset stores every input in its bf16 io buffer (indices > 256 change)
        g16 = json.loads((_extract(self.d, "bf16") / "graph.json").read_text())
        self.assertNotIn("weight_dtypes", g16)
        self.assertFalse(any(cuda_dtypes.parse_cast(o["op"]) for o in g16["ops"]))
        io16 = np.load(self.d / "ir_bf16" / "io.npz")
        self.assertNotIn(f"in.{ir.graph['input']['packed_inputs'][2]['name']}", io16.files)
        self.assertNotEqual(io16["input"][-6:].tolist(), [i * 1001 + 300 for i in range(6)])

    def test_fp32_preset_unchanged(self):
        g32 = json.loads((_extract(self.d, "fp32") / "graph.json").read_text())
        self.assertEqual(g32["quant"], "fp32")
        self.assertFalse(any(cuda_dtypes.parse_cast(o["op"]) for o in g32["ops"]))
        self.assertTrue(all(m["dtype"] == "f32" for m in g32["tensors"].values()))
        self.assertEqual(set(np.load(self.d / "ir_fp32" / "io.npz").files), {"input", "output"})

    def test_fusion_refuses_a_dtype_change(self):
        from modelblaster.pipeline.kb_proposals import _all_specs, fuse_pair
        g = self.g
        prod = {t: o for o in g["ops"] for t in o["outputs"]}
        cast = next(o for o in g["ops"] if o["op"] == "cast_f32_to_bf16")
        src = prod[cast["inputs"][0]]
        with self.assertRaisesRegex(ValueError, "cast|dtype"):
            fuse_pair(g, src["name"], cast["name"], _all_specs())

    def test_export_layout(self):
        from modelblaster.pipeline.kb_export import export_problem
        p = export_problem(self.ir, self.d / "kbx", name="mixed")
        man = json.loads((p / "manifest.json").read_text())
        self.assertEqual(man["dtype"], "mixed")
        self.assertEqual(man["tolerance"]["slot_rel"], cuda_dtypes.TOLERANCE["bf16"]["slot_rel"])
        sd = {s["id"]: s for s in man["slots"]}
        mm = next(s for s in man["slots"] if s["op"].startswith(("bmm", "matmul")))
        self.assertEqual((mm["dtype"], mm["tolerance"]["slot_rel"]), ("bf16", 1.6e-2))
        self.assertEqual(sd["cast_bf16_to_f32"]["dtype"], "fp32")
        self.assertEqual(sd["cast_bf16_to_f32"]["tolerance"]["slot_rel"], 1e-3)
        self.assertIn("__nv_bfloat16", sd["cast_bf16_to_f32"]["signature"])
        bufs = {b["name"]: b for b in man["buffers"]}
        ins = [p_["name"] for p_ in self.g["input"]["packed_inputs"]]
        self.assertEqual([bufs[t]["dtype"] for t in ins], ["bf16", "f32", "i64"])
        # data.bin: each buffer's bytes at its own element size; the int64 input exact
        blob = (p / "data.bin").read_bytes()
        for t, (dt, es) in zip(ins, (("bf16", 2), ("f32", 4), ("i64", 8))):
            off, n = bufs[t]["ref"]
            self.assertEqual(n, bufs[t]["numel"] * es, t)
        off, n = bufs[ins[2]]["ref"]
        self.assertEqual(np.frombuffer(blob[off:off + n], np.int64).tolist(), [i * 1001 + 300 for i in range(6)])
        off, n = bufs[ins[0]]["ref"]
        io = np.load(self.ir / "io.npz")
        self.assertTrue(np.array_equal(np.frombuffer(blob[off:off + n], np.uint16),
                                       cuda_dtypes.bf16_bits(io[f"in.{ins[0]}"])))
        # per-output bounds: an output computed through bf16 storage gets bf16's, fp32 math fp32's
        outs = self.g["output"]["tensors"]
        self.assertEqual(bufs[outs[0]]["tolerance"]["output_rel"], cuda_dtypes.TOLERANCE["bf16"]["output_rel"])
        self.assertEqual(bufs[outs[2]]["tolerance"]["output_rel"], cuda_dtypes.TOLERANCE["fp32"]["output_rel"])
        drv = (p / "driver.cpp").read_text()
        for tab in ("B_I32", "B_BYTE", "B_OUT_REL", "B_OUT_ABS", "OP_SLOT_REL", "OP_SLOT_ABS"):
            self.assertRegex(drv, rf"static const \w+ {tab}\[\]")
        rel = re.search(r"OP_SLOT_REL\[\] = \{([^}]*)\}", drv).group(1).split(", ")
        ops = man["ops"]
        for r_, v in zip(ops, rel):
            want = cuda_dtypes.TOLERANCE[cuda_dtypes.op_dtype(r_["op"])]["slot_rel"]
            self.assertEqual(float(v), want, r_["op"])
        head = (p / "walker_head.cu").read_text()
        self.assertIn("#include <cuda_bf16.h>", head)
        self.assertEqual(head.count("#define MB_THREADS"), 1)

    @unittest.skipUnless(shutil.which("nvcc"), "needs nvcc (compile only)")
    def test_export_builds_one_process_and_host(self):
        from modelblaster.pipeline.kb_export import build_and_run, export_problem
        p = export_problem(self.ir, self.d / "kbx_build", name="mixed")
        for kw in ({}, {"runtime": "host"}):
            r = build_and_run(p, arch="75", runs=0, **kw)
            self.assertEqual(r["stage"], "built", (kw, str(r.get("log", ""))[-2000:]))


class PresetTest(unittest.TestCase):
    def test_single_dtype_problem_dtypes(self):
        from modelblaster.pipeline.kb_export import _problem_dtype
        self.assertEqual(_problem_dtype(["relu", "add"]), "fp32")
        self.assertEqual(_problem_dtype(["relu_bf16", "add_bf16"]), "bf16")
        self.assertEqual(_problem_dtype(["relu_bf16", "add"]), "fp32")             # a legacy IR: as before
        self.assertEqual(_problem_dtype(["relu_bf16", "cast_bf16_to_f32", "add"], mixed=True), "mixed")
        self.assertEqual(_problem_dtype(["cast_bf16_to_f32", "add"], mixed=True), "fp32")


if __name__ == "__main__":
    unittest.main()
