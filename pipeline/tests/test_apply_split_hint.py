"""Unit tests for pipeline/apply_split_hint.py (Phase 1e axis-C split)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.apply_split_hint import apply_split_hint, SplitHintError


def _linear_op(did, name, M, K, N, inputs=("x",), outputs=("y",),
               depends_on=()):
    return {
        "name": name, "op": "linear_s8",
        "inputs": list(inputs), "outputs": list(outputs),
        "weight": f"{name}.weight_q", "bias": f"{name}.bias_q",
        "shape": {"M": M, "K": K, "N": N},
        "quant": {
            "input_offset": 0, "filter_offset": 0, "output_offset": 0,
            "output_multiplier": 1845733646, "output_shift": 7,
            "activation_min": -128, "activation_max": 127,
        },
        "dispatch_id": did, "hardware_target": "any",
        "depends_on": list(depends_on),
    }


def _g(*ops, name="t"):
    return {
        "name": name,
        "input": {"tensor": "x"},
        "output": {"tensor": ops[-1]["outputs"][0],
                   "tensors": [ops[-1]["outputs"][0]]},
        "tensors": {},
        "ops": list(ops),
    }


class LinearSplitBasicTest(unittest.TestCase):

    def test_splits_one_op_into_two_tiles(self):
        g = _g(_linear_op(0, "lin0", M=1, K=32, N=64))
        out = apply_split_hint(g, [{"op": 0, "n_splits": 2}])
        ops = [o for o in out["ops"] if o.get("dispatch_id") is not None]
        self.assertEqual(len(ops), 2)
        t0, t1 = ops
        self.assertEqual(t0["dispatch_id"], 0)
        self.assertEqual(t1["dispatch_id"], 1)
        self.assertEqual(t0["shape"]["N"], 32)
        self.assertEqual(t1["shape"]["N"], 32)
        self.assertEqual(t0["outputs"], ["y.tile_0"])
        self.assertEqual(t1["outputs"], ["y.tile_1"])
        self.assertEqual(t0["split_from"]["tile"], 0)
        self.assertEqual(t0["split_from"]["tile_offset_N"], 0)
        self.assertEqual(t1["split_from"]["tile_offset_N"], 32)
        self.assertEqual(t1["split_from"]["n_splits"], 2)

    def test_downstream_depends_on_all_tiles(self):
        g = _g(
            _linear_op(0, "lin0", M=1, K=32, N=64),
            _linear_op(1, "lin1", M=1, K=64, N=4,
                       inputs=["y"], outputs=["z"], depends_on=[0]),
        )
        out = apply_split_hint(g, [{"op": 0, "n_splits": 2}])
        ops = [o for o in out["ops"] if o.get("dispatch_id") is not None]
        self.assertEqual(len(ops), 3)
        tail = ops[-1]
        self.assertEqual(tail["name"], "lin1")
        self.assertEqual(set(tail["depends_on"]), {0, 1})


class LinearSplitRejectTest(unittest.TestCase):

    def test_reject_unknown_dispatch_id(self):
        g = _g(_linear_op(0, "lin0", 1, 32, 64))
        with self.assertRaises(SplitHintError):
            apply_split_hint(g, [{"op": 99, "n_splits": 2}])

    def test_reject_non_splittable_op_kind(self):
        g = _g({
            "name": "e", "op": "elu_s8",
            "inputs": ["x"], "outputs": ["y"],
            "shape": {"n": 4},
            "dispatch_id": 0, "hardware_target": "any", "depends_on": [],
        })
        with self.assertRaises(SplitHintError):
            apply_split_hint(g, [{"op": 0, "n_splits": 2}])

    def test_reject_n_not_dividing(self):
        g = _g(_linear_op(0, "lin0", 1, 32, 63))
        with self.assertRaises(SplitHintError):
            apply_split_hint(g, [{"op": 0, "n_splits": 2}])

    def test_reject_n_splits_less_than_2(self):
        g = _g(_linear_op(0, "lin0", 1, 32, 64))
        with self.assertRaises(SplitHintError):
            apply_split_hint(g, [{"op": 0, "n_splits": 1}])


if __name__ == "__main__":
    unittest.main()
