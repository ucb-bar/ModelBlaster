"""Derived weights (pipeline/weight_derivations.py): record / replay round trips for every storage dtype, the rejections
(not derivable, wrong dtype, bad expressions), and the integration points (write_variant_ir, the harness loader).

CPU only; numpy (+ torch for the IR fixtures).
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline import cuda_dtypes  # noqa: E402
from modelblaster.pipeline import weight_derivations as wd  # noqa: E402
from modelblaster.pipeline.kb_proposals import write_variant_ir  # noqa: E402


def _g(quant="fp32", **kw):
    return {"name": "t", "quant": quant, "tensors": {}, "ops": [], **kw}


def _base(dt):
    r = np.random.default_rng(0)
    return {"a": r.standard_normal((2, 3, 1, 1)).astype(dt), "b": r.standard_normal((4, 3, 3, 3)).astype(dt),
            "v": r.standard_normal(5).astype(dt)}


class RoundTripTest(unittest.TestCase):
    def test_replay_is_bit_identical_fp32_fp16(self):
        for quant, dt in (("fp32", np.float32), ("fp16", np.float16)):
            base = _base(dt)
            g = _g(quant)
            D = wd.Recorder(g, base)
            D.pad("d.ap", "a", [[0, 0], [0, 0], [1, 1], [1, 1]])
            D.slice("d.bs", "b", 0, 1, 3)
            D.concat("d.w", ["d.ap", "d.bs"], 0)
            D.transpose("d.wt", "d.w", [1, 0, 2, 3])
            D.reshape("d.flat", "d.wt", [-1])
            D.stack("d.stk", ["v", "v"], 0)
            D.scale("d.sv", "v", 0.5)
            D.const("d.z", [3], 0.0, "f16" if dt == np.float16 else "f32")
            g2 = json.loads(json.dumps(g))                            # through graph.json
            self.assertEqual(wd.validate(g2, base), [])
            rep = wd.replay(g2, base)
            self.assertEqual(sorted(rep), sorted(D.arrays))
            for k, a in D.arrays.items():
                self.assertEqual(rep[k].dtype, a.dtype, k)
                self.assertTrue(np.array_equal(rep[k], a), k)
                self.assertEqual(rep[k].dtype, dt, k)               # fp16 IR: fp16 storage; fp32: fp32
            # structural derivations are exact copies of base values
            self.assertTrue(np.array_equal(rep["d.ap"][:, :, 1, 1], base["a"][:, :, 0, 0]))
            self.assertEqual(float(np.abs(rep["d.ap"].astype(np.float64)).sum()), float(np.abs(base["a"].astype(np.float64)).sum()))   # padding adds zeros only
            self.assertFalse(wd.is_exact(g2))                          # scale is arithmetic
            self.assertEqual(rep["d.sv"].dtype, dt)
            self.assertTrue(np.array_equal(rep["d.sv"], (base["v"].astype(np.float32) * np.float32(0.5)).astype(dt)))

    def test_exactness_class(self):
        base = _base(np.float32)
        g = _g()
        wd.Recorder(g, base).pad("d.ap", "a", [[0, 0], [0, 0], [1, 1], [1, 1]])
        self.assertTrue(wd.is_exact(g))

    def test_bf16_ir_holds_bf16_exact_float32(self):
        base = {k: cuda_dtypes.round_bf16(v) for k, v in _base(np.float32).items()}
        g = _g("bf16")
        D = wd.Recorder(g, base)
        D.scale("d.s", "v", 0.3)
        D.add("d.t", "v", "d.s")
        D.cast("d.c", "v", "bf16")
        rep = wd.replay(json.loads(json.dumps(g)), base)
        for k in ("d.s", "d.t", "d.c"):
            self.assertEqual(rep[k].dtype, np.float32)
            self.assertTrue(cuda_dtypes.bf16_exact(rep[k]), k)         # rounded to bf16, not just float32
            self.assertTrue(np.array_equal(rep[k], D.arrays[k]))
        self.assertEqual(g["weight_derivations"]["d.s"]["dtype"], "bf16")
        # float32 scale of a bf16 value is NOT bf16-exact before the rounding: the rounding is what the derivation does
        raw = base["v"] * np.float32(0.3)
        self.assertFalse(cuda_dtypes.bf16_exact(raw))

    def test_native_ir_declares_weight_dtypes(self):
        base = {"w16": _base(np.float16)["v"], "w32": _base(np.float32)["v"]}
        g = _g("native", weight_dtypes={"w16": "f16", "w32": "f32"})
        D = wd.Recorder(g, base)
        D.scale("d.a", "w16", 2.0)
        D.cast("d.b", "w16", "f32")
        D.cast("d.c", "w32", "bf16")
        self.assertEqual({k: g["weight_dtypes"][k] for k in ("d.a", "d.b", "d.c")}, {"d.a": "f16", "d.b": "f32", "d.c": "bf16"})
        self.assertEqual(wd.validate(g, base), [])
        rep = wd.replay(g, base)
        self.assertEqual((rep["d.a"].dtype, rep["d.b"].dtype, rep["d.c"].dtype), (np.float16, np.float32, np.float32))
        self.assertTrue(cuda_dtypes.bf16_exact(rep["d.c"]))
        # a native IR whose weight_dtypes does not name the derived weight violates the dtype contract
        g2 = json.loads(json.dumps(g))
        del g2["weight_dtypes"]["d.b"]
        self.assertEqual([e["code"] for e in wd.validate(g2, base)], ["dtype_contract"])

    def test_derivations_can_read_each_other_and_materialize(self):
        base = _base(np.float16)
        g = _g("fp16")
        wd.Recorder(g, base).concat("d.v2", ["v", "v"], 0)
        full = wd.materialize(g, base)
        self.assertEqual(sorted(full), sorted(list(base) + ["d.v2"]))
        self.assertTrue(np.array_equal(full["d.v2"], np.concatenate([base["v"], base["v"]])))
        g["weight_derivations"]["d.v4"] = {"dtype": "f16", "shape": [20], "expr": {"op": "concat", "axis": 0, "args": [{"w": "d.v2"}, {"w": "d.v2"}]}}
        full = wd.materialize(g, base)
        self.assertEqual(full["d.v4"].shape, (20,))


class RejectionTest(unittest.TestCase):
    def setUp(self):
        self.base = _base(np.float16)

    def _codes(self, g, base=None):
        return [e["code"] for e in wd.validate(g, set(base or self.base))]

    def test_unknown_leaf_is_unmapped(self):
        g = _g("fp16", weight_derivations={"d.x": {"dtype": "f16", "shape": [5], "expr": {"w": "not.a.weight"}}})
        self.assertEqual(self._codes(g), ["unmapped"])
        with self.assertRaises(wd.DerivationError) as cm:
            wd.replay(g, self.base)
        self.assertEqual(cm.exception.code, "unmapped")
        self.assertIn("IR weights not mapped to the Model", str(cm.exception))

    def test_arbitrary_code_is_not_a_node(self):
        for expr in ({"op": "eval", "args": [{"w": "v"}]}, {"op": "__import__", "args": []}, {"fn": "lambda x: x"},
                     {"op": "scale", "by": "os.system('x')", "args": [{"w": "v"}]}, {"w": "v", "op": "scale"}):
            g = _g("fp16", weight_derivations={"d.x": {"dtype": "f16", "shape": [5], "expr": expr}})
            self.assertTrue(self._codes(g), expr)

    def test_cycle_shadowing_and_prefix(self):
        g = _g("fp16", weight_derivations={"d.a": {"dtype": "f16", "shape": [5], "expr": {"op": "scale", "by": 2, "args": [{"w": "d.b"}]}},
                                           "d.b": {"dtype": "f16", "shape": [5], "expr": {"op": "scale", "by": 2, "args": [{"w": "d.a"}]}}})
        self.assertIn("cycle", self._codes(g))
        g = _g("fp16", weight_derivations={"v": {"dtype": "f16", "shape": [5], "expr": {"w": "v"}}})
        self.assertTrue({"bad_key", "shadows_base"} <= set(self._codes(g)))           # may not replace an original weight
        g = _g("fp16", weight_derivations={"x.y": {"dtype": "f16", "shape": [5], "expr": {"w": "v"}}})
        self.assertEqual(self._codes(g), ["bad_key"])

    def test_dtype_contract_of_a_single_dtype_ir(self):
        g = _g("fp16", weight_derivations={"d.x": {"dtype": "f32", "shape": [5], "expr": {"op": "cast", "dtype": "f32", "args": [{"w": "v"}]}}})
        self.assertEqual(self._codes(g), ["dtype_contract"])      # an fp16 IR stores float weights as f16

    def test_mixed_dtype_concat_and_record_mismatch(self):
        base = {"h": np.ones(3, np.float16), "f": np.ones(3, np.float32)}
        g = _g("native", weight_dtypes={"h": "f16", "f": "f32"},
               weight_derivations={"d.x": {"dtype": "f16", "shape": [6], "expr": {"op": "concat", "axis": 0, "args": [{"w": "h"}, {"w": "f"}]}}})
        g["weight_dtypes"]["d.x"] = "f16"
        with self.assertRaises(wd.DerivationError) as cm:
            wd.replay(g, base)
        self.assertEqual(cm.exception.code, "dtype")
        g = _g("fp16", weight_derivations={"d.x": {"dtype": "f16", "shape": [9], "expr": {"w": "v"}}})
        with self.assertRaises(wd.DerivationError) as cm:                # recorded shape is wrong
            wd.replay(g, self.base)
        self.assertEqual(cm.exception.code, "shape")

    def test_shape_errors(self):
        g = _g("fp16")
        D = wd.Recorder(g, self.base)
        for bad in (lambda: D.slice("d.s", "v", 0, 3, 9), lambda: D.reshape("d.r", "v", [2, 3]), lambda: D.concat("d.c", ["a", "b"], 0),
                    lambda: D.transpose("d.t", "v", [1, 0]), lambda: D.pad("d.p", "v", [[0, 1], [0, 1]])):
            with self.assertRaises(wd.DerivationError):
                bad()
        self.assertNotIn("weight_derivations", g)                        # nothing half-recorded: the graph is untouched
        with self.assertRaises(wd.DerivationError):
            D.scale("v", "v", 2.0)                                       # name taken
        with self.assertRaises(wd.DerivationError):
            D.scale("noprefix", "v", 2.0)


class IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mb_wd_"))
        from modelblaster.pipeline.tests import rewrite_fixtures as F
        self.F = F
        self.ir = F.fire_ir(self.tmp / "ir", "fp16")

    def test_default_ir_is_untouched(self):
        """No derivation: write_variant_ir writes graph.json as before and symlinks the weights (byte for byte)."""
        g = json.loads((self.ir / "graph.json").read_text())
        out = write_variant_ir(self.ir, g, self.tmp / "v0")
        self.assertEqual((out / "graph.json").read_text(), json.dumps(g, indent=1))
        self.assertTrue((out / "weights.npz").is_symlink() and (out / "io.npz").is_symlink())
        self.assertEqual((out / "weights.npz").resolve(), (self.ir / "weights.npz").resolve())

    def test_write_variant_ir_materializes_derived_weights(self):
        e, _ = self.F.merge_expand(self.ir, self.tmp / "e")
        g = json.loads((e / "graph.json").read_text())
        self.assertIn("weight_derivations", g)
        out = write_variant_ir(self.ir, g, self.tmp / "v1")
        self.assertFalse((out / "weights.npz").is_symlink())
        z, z0 = np.load(out / "weights.npz"), np.load(self.ir / "weights.npz")
        self.assertTrue(set(z0.files) < set(z.files))                    # base weights kept (the Model mapping needs them)
        self.assertEqual(z["d.fire.w"].dtype, np.float16)
        self.assertEqual(z["d.fire.w"].shape, (5, 3, 3, 3))
        for k in z0.files:
            self.assertTrue(np.array_equal(z[k], z0[k]))
        self.assertTrue(np.array_equal(z["d.fire.w"][2:], z0["e3.weight"]))
        self.assertTrue(np.array_equal(z["d.fire.w"][:2, :, 1, 1], z0["e1.weight"][:, :, 0, 0]))
        self.assertEqual(float(np.abs(z["d.fire.w"][:2]).astype(np.float64).sum()), float(np.abs(z0["e1.weight"]).astype(np.float64).sum()))

    def test_harness_loader_rebuilds_missing_derived_weights(self):
        from modelblaster.pipeline.kb_export import _harness
        mbir, _ = _harness()
        e, _ = self.F.merge_expand(self.ir, self.tmp / "e")
        full = np.load(e / "weights.npz")
        self.assertIn("d.fire.w", full.files)
        stale = self.tmp / "stale"                                        # a variant dir written before the derivation existed
        stale.mkdir()
        (stale / "graph.json").write_text((e / "graph.json").read_text())
        for f in ("weights.npz", "io.npz"):
            (stale / f).symlink_to((self.ir / f).resolve())
        self.assertNotIn("d.fire.w", np.load(stale / "weights.npz").files)
        ir = mbir.load_ir(stale)
        self.assertTrue(np.array_equal(ir.weights["d.fire.w"], full["d.fire.w"]))
        # an IR with no derivations loads exactly as before
        ir0 = mbir.load_ir(self.ir)
        self.assertEqual(sorted(ir0.weights), sorted(np.load(self.ir / "weights.npz").files))


if __name__ == "__main__":
    unittest.main()
