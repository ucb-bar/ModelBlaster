"""Unit tests for pipeline/apply_fusion_hint.py.

These run on hand-built IR dicts (no torch / no codegen) so they're
fast and don't pull in any heavy deps. The fixtures mirror the shape
of `examples/<model>/<quant>/generated/graph.json` (only the fields
the rewrite actually inspects).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.apply_fusion_hint import (
    FusionHintError,
    apply_hint,
)


def _op(did, op, name, inputs, outputs, depends_on=None, **kw):
    base = {
        "name": name,
        "op": op,
        "inputs": list(inputs),
        "outputs": list(outputs),
        "dispatch_id": did,
        "hardware_target": kw.pop("hardware_target", "any"),
        "depends_on": list(depends_on or []),
    }
    base.update(kw)
    return base


def _mlp3_graph():
    """3-op linear/elu/linear chain: x -> mlp_0 -> mlp_1 -> mlp_2."""
    return {
        "name": "tiny",
        "version": 1,
        "input": {"tensor": "x"},
        "output": {"tensor": "mlp_2", "tensors": ["mlp_2"]},
        "tensors": {
            "x": {"shape": [1, 8], "dtype": "i8"},
            "mlp_0": {"shape": [1, 16], "dtype": "i8"},
            "mlp_1": {"shape": [1, 16], "dtype": "i8"},
            "mlp_2": {"shape": [1, 4], "dtype": "i8"},
        },
        "ops": [
            _op(0, "linear_s8", "lin0", ["x"], ["mlp_0"]),
            _op(1, "elu_s8", "elu0", ["mlp_0"], ["mlp_1"], depends_on=[0]),
            _op(2, "linear_s8", "lin1", ["mlp_1"], ["mlp_2"], depends_on=[1]),
        ],
    }


class FuseTwoOpChainTest(unittest.TestCase):
    """A 2-op fuse_group should collapse to one fused op + the trailing op."""

    def test_basic_shape(self):
        g = _mlp3_graph()
        out = apply_hint(g, [[0, 1]])
        self.assertEqual(len(out["ops"]), 2)

        fused, tail = out["ops"]
        # fused op — when sub_ops are exactly [linear_s8, elu_s8] the
        # rewrite now emits the registered KernelSpec key
        # `linear_s8_elu_s8` (Phase 1d) instead of the synthetic chain
        # name; that routes codegen through the registered kernel
        # (with LLM-codegen seeds) rather than the chained-call fallback.
        self.assertEqual(fused["fused_from"], [0, 1])
        self.assertEqual(fused["op"], "linear_s8_elu_s8")
        self.assertEqual(fused["dispatch_id"], 0)
        self.assertEqual(fused["depends_on"], [])
        self.assertEqual(fused["inputs"], ["x"])
        self.assertEqual(fused["outputs"], ["mlp_1"])
        # mlp_0 produced inside the chain, consumed inside — stack-local
        self.assertEqual(fused["internal_tensors"], ["mlp_0"])
        # sub_ops verbatim
        self.assertEqual([s["op"] for s in fused["sub_ops"]],
                         ["linear_s8", "elu_s8"])

        # trailing op: dispatch_id shifted 2 -> 1, depends_on rewired 1 -> 0
        self.assertEqual(tail["op"], "linear_s8")
        self.assertEqual(tail["dispatch_id"], 1)
        self.assertEqual(tail["depends_on"], [0])

    def test_input_unmutated(self):
        g = _mlp3_graph()
        before = [op["dispatch_id"] for op in g["ops"]]
        _ = apply_hint(g, [[0, 1]])
        after = [op["dispatch_id"] for op in g["ops"]]
        self.assertEqual(before, after)


class FuseFullChainTest(unittest.TestCase):
    """A fuse_group covering the entire chain produces one fused op."""

    def test_all_three(self):
        g = _mlp3_graph()
        out = apply_hint(g, [[0, 1, 2]])
        self.assertEqual(len(out["ops"]), 1)
        fused = out["ops"][0]
        self.assertEqual(fused["inputs"], ["x"])
        # mlp_2 is the model output → must stay in outputs even though
        # no downstream op consumes it inside this graph.
        self.assertEqual(fused["outputs"], ["mlp_2"])
        self.assertEqual(set(fused["internal_tensors"]), {"mlp_0", "mlp_1"})
        self.assertEqual(fused["depends_on"], [])


class FuseMultipleGroupsTest(unittest.TestCase):
    """Two disjoint groups should produce two fused ops."""

    def test_two_pairs(self):
        # 5-op chain: pair up [0,1] and [3,4], leave op 2 alone.
        g = {
            "name": "tiny",
            "input": {"tensor": "x"},
            "output": {"tensor": "t4", "tensors": ["t4"]},
            "tensors": {n: {"shape": [1, 4], "dtype": "i8"}
                        for n in ["x", "t0", "t1", "t2", "t3", "t4"]},
            "ops": [
                _op(0, "linear_s8", "a", ["x"], ["t0"]),
                _op(1, "elu_s8", "b", ["t0"], ["t1"], depends_on=[0]),
                _op(2, "linear_s8", "c", ["t1"], ["t2"], depends_on=[1]),
                _op(3, "elu_s8", "d", ["t2"], ["t3"], depends_on=[2]),
                _op(4, "linear_s8", "e", ["t3"], ["t4"], depends_on=[3]),
            ],
        }
        out = apply_hint(g, [[0, 1], [3, 4]])
        self.assertEqual(len(out["ops"]), 3)

        fused_a, mid, fused_b = out["ops"]
        self.assertEqual(fused_a["dispatch_id"], 0)
        self.assertEqual(fused_a["fused_from"], [0, 1])
        self.assertEqual(fused_a["outputs"], ["t1"])
        self.assertEqual(fused_a["depends_on"], [])

        self.assertEqual(mid["dispatch_id"], 1)
        self.assertEqual(mid["op"], "linear_s8")
        self.assertEqual(mid["depends_on"], [0])  # was [1] → fused_a

        self.assertEqual(fused_b["dispatch_id"], 2)
        self.assertEqual(fused_b["fused_from"], [3, 4])
        self.assertEqual(fused_b["depends_on"], [1])  # was [2] → mid


class FuseRejectsTest(unittest.TestCase):

    def test_unknown_id(self):
        g = _mlp3_graph()
        with self.assertRaises(FusionHintError):
            apply_hint(g, [[0, 99]])

    def test_duplicate(self):
        g = _mlp3_graph()
        with self.assertRaises(FusionHintError):
            apply_hint(g, [[0, 0, 1]])

    def test_out_of_order(self):
        # 1 depends on 0; [1, 0] is not topo-sorted.
        g = _mlp3_graph()
        with self.assertRaises(FusionHintError):
            apply_hint(g, [[1, 0]])

    def test_overlapping_groups(self):
        g = _mlp3_graph()
        with self.assertRaises(FusionHintError):
            apply_hint(g, [[0, 1], [1, 2]])

    def test_empty_group(self):
        g = _mlp3_graph()
        with self.assertRaises(FusionHintError):
            apply_hint(g, [[]])


class FuseEmptyHintTest(unittest.TestCase):
    """An empty fuse_groups list is a no-op (copy through)."""

    def test_passthrough(self):
        g = _mlp3_graph()
        out = apply_hint(g, [])
        self.assertEqual(len(out["ops"]), 3)
        self.assertEqual([o["dispatch_id"] for o in out["ops"]], [0, 1, 2])
        self.assertEqual([o["op"] for o in out["ops"]],
                         ["linear_s8", "elu_s8", "linear_s8"])


class FuseBranchingOutputTest(unittest.TestCase):
    """If a group member's output is consumed by an op OUTSIDE the group
    AND the group's tail, the tensor must escape as a fused output."""

    def test_intermediate_consumed_outside(self):
        # 4 ops: 0 -> 1 -> 2, but op 1's output is ALSO consumed by op 3.
        # Fuse [0, 1, 2]. The fused op's outputs must include both
        # `t1` (consumed by op 3, OUTSIDE) and `t2` (model output).
        g = {
            "name": "tiny",
            "input": {"tensor": "x"},
            "output": {"tensor": "t3", "tensors": ["t3"]},
            "tensors": {n: {"shape": [1, 4], "dtype": "i8"}
                        for n in ["x", "t0", "t1", "t2", "t3"]},
            "ops": [
                _op(0, "linear_s8", "a", ["x"], ["t0"]),
                _op(1, "elu_s8", "b", ["t0"], ["t1"], depends_on=[0]),
                _op(2, "linear_s8", "c", ["t1"], ["t2"], depends_on=[1]),
                _op(3, "linear_s8", "d", ["t1"], ["t3"], depends_on=[1]),
            ],
        }
        out = apply_hint(g, [[0, 1, 2]])
        self.assertEqual(len(out["ops"]), 2)
        fused, tail = out["ops"]
        # `t1` escapes because op 3 consumes it; `t2` does not escape
        # (only the fused op produced it and no one outside consumes
        # it — but if `t2` is unreferenced downstream it's not in
        # outputs at all). Op `c` is the last writer of `t2`; since
        # tail consumes `t1` not `t2`, and `t2` isn't the model output
        # in this fixture, it's purely internal.
        self.assertIn("t1", fused["outputs"])
        self.assertNotIn("t2", fused["outputs"])
        self.assertEqual(set(fused["internal_tensors"]), {"t0", "t2"})
        # tail = original op 3
        self.assertEqual(tail["op"], "linear_s8")
        self.assertEqual(tail["depends_on"], [0])  # rewired


if __name__ == "__main__":
    unittest.main()
