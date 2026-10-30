"""Undoing a fusion must restore a graph the codegen can still build.

The failure modes here are structural and silent, so each has a test.

WHAT IT KEYS ON. `apply_fusion_hint` records `fused_from` and
`internal_tensors`; almost no fused op on disk came from it. 0 of yolov8_nano's
57 fused convs carry either field -- they are built at export time by
`extract_graph`'s conv->BN(->act) recognizer. A rewriter keyed on `fused_from`
would refuse 100% of the ops that matter, so `sub_ops` is the only field this
may rely on.

THE TWO CONSEQUENCES, both tested below:

* The internal tensors were never registered (`extract_graph`: "the conv/bn
  intermediates live inside the single kernel and need no global buffer"), so
  they must be SYNTHESIZED. For ops that did come from `apply_fusion_hint` they
  are already present, so both cases must work.

* A `conv2d_s8` sub_op has `output_multiplier`/`output_shift` and no
  `scale_out`, so its output scale exists only on the CONSUMER's `scale_in`.
  Defaulting instead would make everything reading the tensor report raw int8
  counts as physical values.

AND THE ONE THAT IS EASIEST TO GET WRONG. A split fans a downstream consumer
out to ALL tiles, because the output is their concatenation. An unfuse fans it
in to the TAIL piece only, because the tail produces the fused op's output.
Deriving `depends_on` from tensor producers -- the rule
`extract_graph._annotate_dispatches` already uses -- is correct for both the
restored chain and its consumers in one pass, with no id translation to get
wrong.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pipeline.apply_unfuse_hint import (  # noqa: E402
    UnfuseHintError, apply_unfuse_hint,
)


def _fused_op(did=0, out="y_act"):
    return {
        "name": "l0", "op": "conv2d_batchnorm2d_silu_s8", "dispatch_id": did,
        "inputs": ["x"], "outputs": [out], "depends_on": [],
        "hardware_target": "any",
        "sub_ops": [
            {"name": "l0.conv", "op": "conv2d_s8", "inputs": ["x"],
             "outputs": ["y_conv"], "weight": "w", "bias": "b",
             "shape": {"N": 1, "IC": 3, "IH": 8, "IW": 8, "OC": 4,
                       "OH": 4, "OW": 4, "KH": 3, "KW": 3,
                       "SH": 2, "SW": 2, "PH": 1, "PW": 1},
             "quant": {"output_multiplier": 1, "output_shift": 0}},
            {"name": "l0.bn", "op": "batchnorm2d_s8", "inputs": ["y_conv"],
             "outputs": ["y_bn"], "weight": "s", "bias": "o",
             "shape": {"N": 1, "C": 4, "H": 4, "W": 4},
             "quant": {"scale_in": 0.25, "scale_out": 0.5}},
            {"name": "l0.act", "op": "silu_s8", "inputs": ["y_bn"],
             "outputs": [out], "shape": {"n": 64},
             "quant": {"scale_in": 0.5, "scale_out": 0.75}},
        ],
    }


def _graph(*ops, out="y_act"):
    return {
        "name": "probe", "input": {"tensor": "x"}, "output": {"tensor": out},
        "tensors": {"x": {"shape": [1, 3, 8, 8], "dtype": "i8",
                          "quant": {"scale": 1.0, "zero_point": 0}},
                    out: {"shape": [1, 4, 4, 4], "dtype": "i8",
                          "quant": {"scale": 0.75, "zero_point": 0}}},
        "ops": list(ops),
        "dispatches": [o["dispatch_id"] for o in ops
                       if o.get("dispatch_id") is not None],
    }


class TheRestoration(unittest.TestCase):

    def test_one_op_becomes_its_constituents(self):
        g = _graph(_fused_op())
        out = apply_unfuse_hint(g, [{"op": 0}])
        kinds = [o["op"] for o in out["ops"]]
        self.assertEqual(kinds, ["conv2d_s8", "batchnorm2d_s8", "silu_s8"])
        self.assertEqual([o["dispatch_id"] for o in out["ops"]], [0, 1, 2])

    def test_the_chain_depends_in_order(self):
        out = apply_unfuse_hint(_graph(_fused_op()), [{"op": 0}])
        deps = {o["op"]: o["depends_on"] for o in out["ops"]}
        self.assertEqual(deps["conv2d_s8"], [])
        self.assertEqual(deps["batchnorm2d_s8"], [0])
        self.assertEqual(deps["silu_s8"], [1])

    def test_a_downstream_consumer_attaches_to_the_TAIL_only(self):
        """The property a split gets the opposite way round.

        A split's consumer depends on every tile; an unfuse's consumer depends
        on the tail alone. Getting this wrong yields a graph that is
        structurally valid and schedules a false dependency chain.
        """
        tail = {"name": "next", "op": "conv2d_s8", "dispatch_id": 1,
                "inputs": ["y_act"], "outputs": ["z"], "depends_on": [0],
                "shape": {"N": 1, "IC": 4, "IH": 4, "IW": 4, "OC": 4,
                          "OH": 4, "OW": 4, "KH": 1, "KW": 1,
                          "SH": 1, "SW": 1, "PH": 0, "PW": 0},
                "quant": {}}
        g = _graph(_fused_op(), tail)
        out = apply_unfuse_hint(g, [{"op": 0}])
        nxt = [o for o in out["ops"] if o["name"] == "next"][0]
        self.assertEqual(nxt["depends_on"], [2],
                         "must depend on the silu tail alone, not on all three")

    def test_outputs_are_not_renamed(self):
        """Restoration, not synthesis -- so consumers need no rewiring."""
        out = apply_unfuse_hint(_graph(_fused_op()), [{"op": 0}])
        self.assertEqual(out["ops"][-1]["outputs"], ["y_act"])

    def test_stale_sub_op_metadata_is_stripped(self):
        """apply_fusion_hint's sub_ops carry PRE-rewrite ids and deps."""
        f = _fused_op()
        for i, s in enumerate(f["sub_ops"]):
            s["dispatch_id"] = 90 + i
            s["depends_on"] = [88]
        out = apply_unfuse_hint(_graph(f), [{"op": 0}])
        self.assertEqual([o["dispatch_id"] for o in out["ops"]], [0, 1, 2])
        self.assertEqual(out["ops"][0]["depends_on"], [])


class TheInternalTensors(unittest.TestCase):

    def test_they_are_synthesized_when_absent(self):
        g = _graph(_fused_op())
        self.assertNotIn("y_conv", g["tensors"])
        out = apply_unfuse_hint(g, [{"op": 0}])
        self.assertIn("y_conv", out["tensors"])
        self.assertIn("y_bn", out["tensors"])

    def test_a_conv_output_scale_comes_from_its_consumer(self):
        """A conv sub_op has no scale_out; the consumer's scale_in is the
        only source, and defaulting would misreport every value."""
        out = apply_unfuse_hint(_graph(_fused_op()), [{"op": 0}])
        self.assertEqual(out["tensors"]["y_conv"]["quant"]["scale"], 0.25)

    def test_shape_preserving_ops_take_the_fused_output_shape(self):
        out = apply_unfuse_hint(_graph(_fused_op()), [{"op": 0}])
        self.assertEqual(out["tensors"]["y_bn"]["shape"], [1, 4, 4, 4])

    def test_an_unrecoverable_scale_is_refused(self):
        f = _fused_op()
        f["sub_ops"][1]["quant"] = {"scale_out": 0.5}     # no scale_in
        with self.assertRaises(UnfuseHintError) as cm:
            apply_unfuse_hint(_graph(f), [{"op": 0}])
        self.assertIn("scale", str(cm.exception))

    def test_an_already_registered_internal_tensor_is_kept(self):
        """The apply_fusion_hint case: they were never removed."""
        g = _graph(_fused_op())
        g["tensors"]["y_conv"] = {"shape": [1, 4, 4, 4], "dtype": "i8",
                                  "quant": {"scale": 0.25, "zero_point": 0}}
        out = apply_unfuse_hint(g, [{"op": 0}])
        self.assertEqual(out["tensors"]["y_conv"]["quant"]["scale"], 0.25)

    def test_a_conflicting_registration_is_refused_not_overwritten(self):
        g = _graph(_fused_op())
        g["tensors"]["y_conv"] = {"shape": [9, 9], "dtype": "i8"}
        with self.assertRaises(UnfuseHintError):
            apply_unfuse_hint(g, [{"op": 0}])


class TheRemapAndBookkeeping(unittest.TestCase):

    def test_id_remap_is_list_valued_like_a_split(self):
        out = apply_unfuse_hint(_graph(_fused_op()), [{"op": 0}])
        self.assertEqual(out["id_remap"], {"0": [0, 1, 2]})

    def test_untouched_ops_get_singleton_entries(self):
        tail = {"name": "n", "op": "add_s8", "dispatch_id": 1,
                "inputs": ["y_act"], "outputs": ["z"], "depends_on": [0],
                "shape": {"n": 64}, "quant": {}}
        out = apply_unfuse_hint(_graph(_fused_op(), tail), [{"op": 0}])
        self.assertEqual(out["id_remap"]["1"], [3])

    def test_the_dispatch_list_tracks_the_rewrite(self):
        out = apply_unfuse_hint(_graph(_fused_op()), [{"op": 0}])
        ids = [o["dispatch_id"] for o in out["ops"]]
        self.assertEqual(out["dispatches"], ids)


class TheRefusals(unittest.TestCase):

    def test_an_unfused_op_is_refused(self):
        plain = {"name": "p", "op": "conv2d_s8", "dispatch_id": 0,
                 "inputs": ["x"], "outputs": ["y_act"], "depends_on": [],
                 "shape": {}, "quant": {}}
        with self.assertRaises(UnfuseHintError) as cm:
            apply_unfuse_hint(_graph(plain), [{"op": 0}])
        self.assertIn("not a fused op", str(cm.exception))

    def test_an_unknown_dispatch_id_is_refused(self):
        with self.assertRaises(UnfuseHintError):
            apply_unfuse_hint(_graph(_fused_op()), [{"op": 99}])

    def test_sub_ops_that_do_not_compose_are_refused(self):
        f = _fused_op()
        f["sub_ops"][1]["inputs"] = ["somewhere_else"]
        with self.assertRaises(UnfuseHintError) as cm:
            apply_unfuse_hint(_graph(f), [{"op": 0}])
        self.assertIn("do not compose", str(cm.exception))

    def test_a_tail_that_renames_the_output_is_refused(self):
        f = _fused_op()
        f["sub_ops"][-1]["outputs"] = ["something_new"]
        with self.assertRaises(UnfuseHintError) as cm:
            apply_unfuse_hint(_graph(f), [{"op": 0}])
        self.assertIn("rename", str(cm.exception))

    def test_a_shapeless_sub_op_is_refused(self):
        f = _fused_op()
        f["sub_ops"][1].pop("shape")
        with self.assertRaises(UnfuseHintError) as cm:
            apply_unfuse_hint(_graph(f), [{"op": 0}])
        self.assertIn("shape", str(cm.exception))

    def test_nothing_is_written_when_any_target_is_invalid(self):
        """Validate all, then rewrite -- a bad hint leaves the IR untouched."""
        g = _graph(_fused_op())
        before = [o["op"] for o in g["ops"]]
        with self.assertRaises(UnfuseHintError):
            apply_unfuse_hint(g, [{"op": 0}, {"op": 99}])
        self.assertEqual([o["op"] for o in g["ops"]], before)


if __name__ == "__main__":
    unittest.main()
