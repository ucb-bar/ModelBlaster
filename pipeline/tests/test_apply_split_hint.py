"""Unit tests for pipeline/apply_split_hint.py (Phase 1e axis-C split)."""

from __future__ import annotations

import json
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


def _conv2d_op(did, name, N=1, IC=3, IH=8, IW=8, OC=16, KH=3, KW=3,
               inputs=("x",), outputs=("y",), depends_on=()):
    return {
        "name": name, "op": "conv2d_s8",
        "inputs": list(inputs), "outputs": list(outputs),
        "weight": f"{name}.weight_q", "bias": f"{name}.bias_q",
        "shape": {"N": N, "IC": IC, "IH": IH, "IW": IW, "OC": OC,
                  "KH": KH, "KW": KW, "SH": 1, "SW": 1, "PH": 1, "PW": 1},
        "quant": {
            "input_offset": 0, "filter_offset": 0, "output_offset": 0,
            "output_multiplier": 1845733646, "output_shift": 7,
            "activation_min": -128, "activation_max": 127,
        },
        "dispatch_id": did, "hardware_target": "any",
        "depends_on": list(depends_on),
    }


class Conv2dSplitBasicTest(unittest.TestCase):

    def test_splits_one_conv2d_into_two_tiles_along_oc(self):
        g = _g(_conv2d_op(0, "c0", OC=16))
        out = apply_split_hint(g, [{"op": 0, "n_splits": 2}])
        ops = [o for o in out["ops"] if o.get("dispatch_id") is not None]
        self.assertEqual(len(ops), 2)
        t0, t1 = ops
        self.assertEqual(t0["shape"]["OC"], 8)
        self.assertEqual(t1["shape"]["OC"], 8)
        self.assertEqual(t0["outputs"], ["y.tile_0"])
        self.assertEqual(t1["outputs"], ["y.tile_1"])
        self.assertEqual(t0["split_from"]["axis"], "OC")
        self.assertEqual(t0["split_from"]["tile_offset_OC"], 0)
        self.assertEqual(t1["split_from"]["tile_offset_OC"], 8)
        # IC/IH/IW unchanged
        self.assertEqual(t0["shape"]["IC"], 3)

    def test_reject_oc_not_dividing(self):
        g = _g(_conv2d_op(0, "c0", OC=15))
        with self.assertRaises(SplitHintError):
            apply_split_hint(g, [{"op": 0, "n_splits": 2}])

    def test_downstream_depends_on_all_conv_tiles(self):
        g = _g(
            _conv2d_op(0, "c0", OC=16),
            _conv2d_op(1, "c1", IC=16, OC=8,
                       inputs=["y"], outputs=["z"], depends_on=[0]),
        )
        out = apply_split_hint(g, [{"op": 0, "n_splits": 2}])
        ops = [o for o in out["ops"] if o.get("dispatch_id") is not None]
        self.assertEqual(len(ops), 3)
        tail = ops[-1]
        self.assertEqual(set(tail["depends_on"]), {0, 1})


class IdRemapTest(unittest.TestCase):
    """`id_remap` must let a consumer recover any op's new identity.

    Regression for the dropped-mapping bug: inserting N tiles in place
    of one op shifts every id after the split point, so ops that were
    never split still change identity. Any artifact keyed on
    dispatch_id (the profile CSV, the cost DB, SchedulerReport rows,
    Gantt labels) then joins against the wrong op.
    """

    def _chain(self):
        # lin0 (splittable) followed by two ops that are NOT touched.
        return _g(
            _linear_op(0, "lin0", M=1, K=32, N=64),
            _linear_op(1, "lin1", M=1, K=64, N=16,
                       inputs=["y"], outputs=["z"], depends_on=[0]),
            _linear_op(2, "lin2", M=1, K=16, N=8,
                       inputs=["z"], outputs=["w"], depends_on=[1]),
        )

    def test_untouched_op_identity_is_recoverable(self):
        out = apply_split_hint(self._chain(), [{"op": 0, "n_splits": 2}])
        remap = out["id_remap"]
        by_new = {o["dispatch_id"]: o for o in out["ops"]
                  if o.get("dispatch_id") is not None}
        # lin2 was id 2 and was NOT split, yet the two inserted tiles
        # pushed it to id 3.
        self.assertEqual(remap["2"], [3])
        self.assertEqual(by_new[remap["2"][0]]["name"], "lin2")
        self.assertEqual(by_new[remap["1"][0]]["name"], "lin1")

    def test_split_op_maps_to_all_its_tiles(self):
        out = apply_split_hint(self._chain(), [{"op": 0, "n_splits": 4}])
        remap = out["id_remap"]
        # One-to-many must be expressed honestly: keeping only the first
        # tile would make a consumer attribute the whole op's cost to a
        # quarter of the work.
        self.assertEqual(remap["0"], [0, 1, 2, 3])
        by_new = {o["dispatch_id"]: o for o in out["ops"]
                  if o.get("dispatch_id") is not None}
        for tile_idx, new_id in enumerate(remap["0"]):
            self.assertEqual(by_new[new_id]["split_from"]["tile"], tile_idx)

    def test_remap_values_are_always_lists(self):
        # Untouched ops get single-element lists so consumers never have
        # to branch on the value type.
        out = apply_split_hint(self._chain(), [{"op": 0, "n_splits": 2}])
        for old, new in out["id_remap"].items():
            self.assertIsInstance(new, list, f"id_remap[{old!r}]")
            self.assertTrue(new)

    def test_remap_covers_every_input_id(self):
        g = self._chain()
        out = apply_split_hint(g, [{"op": 0, "n_splits": 2}])
        original_ids = {str(o["dispatch_id"]) for o in g["ops"]
                        if o.get("dispatch_id") is not None}
        self.assertEqual(set(out["id_remap"]), original_ids)

    def test_remap_is_json_round_trippable(self):
        # The IR is written with json.dump, so keys must already be
        # strings — int keys would silently stringify on write and
        # mismatch anything comparing keys across the file round trip.
        out = apply_split_hint(self._chain(), [{"op": 0, "n_splits": 2}])
        self.assertEqual(out["id_remap"],
                         json.loads(json.dumps(out))["id_remap"])
        self.assertTrue(all(isinstance(k, str) for k in out["id_remap"]))

    def test_remap_agrees_with_rewired_depends_on(self):
        # The mapping isn't a separate bookkeeping channel: it must be
        # the same renumbering the rewrite applied to depends_on, or a
        # consumer translating through it lands on a different graph
        # than the scheduler sees.
        out = apply_split_hint(self._chain(), [{"op": 0, "n_splits": 2}])
        remap = out["id_remap"]
        by_new = {o["dispatch_id"]: o for o in out["ops"]
                  if o.get("dispatch_id") is not None}
        lin1_new = remap["1"][0]
        self.assertEqual(by_new[lin1_new]["depends_on"], remap["0"])


if __name__ == "__main__":
    unittest.main()


class TheDispatchListTracksTheRewrite(unittest.TestCase):
    """`dispatches` must describe the graph it ships with.

    extract_graph documents this field as "the ordered list of dispatch_ids"
    (:57), and both rewriters used to carry it through untouched -- so every
    rewritten graph on disk described the PRE-rewrite dispatch set. Measured
    before the fix: a fused mlp_control graph with 4 ops still claimed 7
    dispatches; a split DroNet graph with 22 ops claimed 21; a 213-op split
    claimed 212.

    Harmless on the main path, because emit_dispatch_graph walks `ops`. Not
    harmless in general: scripts/plot_frequency_sweep_v2.py:105 reads
    `len(fx["dispatches"])` as the op count, so it was wrong by exactly the
    rewrite delta, silently, inside a plot.
    """

    def test_split_grows_the_list_by_the_tiles_it_added(self):
        g = _g(_linear_op(0, "lin", 1, 32, 64))
        g["dispatches"] = [o["dispatch_id"] for o in g["ops"]]
        before = len(g["dispatches"])
        out = apply_split_hint(g, [{"op": 0, "n_splits": 2}])
        ids = [o["dispatch_id"] for o in out["ops"]
               if o.get("dispatch_id") is not None]
        self.assertEqual(out["dispatches"], ids)
        self.assertEqual(len(out["dispatches"]), before + 1,
                         "one op became two")

    def test_a_graph_without_the_field_does_not_gain_one(self):
        """Only maintained where it already existed -- the rewriters must not
        invent structure the input did not have."""
        g = _g(_linear_op(0, "lin", 1, 32, 64))
        g.pop("dispatches", None)
        out = apply_split_hint(g, [{"op": 0, "n_splits": 2}])
        self.assertNotIn("dispatches", out)


def _fused_conv_op(did, name, kind="conv2d_batchnorm2d_silu_s8",
                   N=1, IC=3, IH=8, IW=8, OC=16, OH=8, OW=8, KH=3, KW=3,
                   inputs=("x",), outputs=("y",), depends_on=()):
    """A fused conv exactly as `extract_graph` writes one.

    The two properties that matter here and that a hand-written fixture is
    easy to get wrong: the op itself carries NO `shape` key at all, and the
    constituents name their channel count three different ways -- `OC` on the
    conv, `C` on the batchnorm, a flat `n` on the activation.
    """
    conv = {
        "name": f"{name}.conv", "op": "conv2d_s8",
        "inputs": list(inputs), "outputs": [f"{name}_conv"],
        "weight": f"{name}.conv.weight_q", "bias": f"{name}.conv.bias_q",
        "shape": {"N": N, "IC": IC, "IH": IH, "IW": IW, "OC": OC,
                  "OH": OH, "OW": OW, "KH": KH, "KW": KW,
                  "SH": 1, "SW": 1, "PH": 1, "PW": 1},
        "quant": {"input_offset": 0, "filter_offset": 0, "output_offset": 0,
                  "output_multiplier": 1845733646, "output_shift": 7,
                  "activation_min": -128, "activation_max": 127},
    }
    bn = {
        "name": f"{name}.bn", "op": "batchnorm2d_s8",
        "inputs": [f"{name}_conv"], "outputs": [f"{name}_bn"],
        "weight": f"{name}.bn.scale", "bias": f"{name}.bn.bias_fused",
        "shape": {"N": N, "C": OC, "H": OH, "W": OW},
        "quant": {"scale_in": 0.04, "scale_out": 1.07,
                  "activation_min": -128, "activation_max": 127},
    }
    subs = [conv, bn]
    if kind == "conv2d_batchnorm2d_silu_s8":
        subs.append({
            "name": f"{name}.act", "op": "silu_s8",
            "inputs": [f"{name}_bn"], "outputs": list(outputs),
            "shape": {"n": N * OC * OH * OW},
            "quant": {"scale_in": 1.07, "scale_out": 1.07,
                      "activation_min": -128, "activation_max": 127},
        })
    else:
        bn["outputs"] = list(outputs)
    return {
        "name": name, "op": kind,
        "inputs": list(inputs), "outputs": list(outputs),
        "sub_ops": subs,
        "dispatch_id": did, "hardware_target": "any",
        "depends_on": list(depends_on),
    }


class FusedConvSplitTest(unittest.TestCase):
    """Splitting the ops that carry the runtime.

    `conv2d_batchnorm2d_silu_s8` is 97% of yolov8n and `conv2d_batchnorm2d_s8`
    is 29% of DroNet. A split path that refuses them can only tile ops too
    cheap for the tiling to matter.
    """

    def test_splits_a_triple_fused_conv_along_oc(self):
        g = _g(_fused_conv_op(0, "l0", OC=16))
        out = apply_split_hint(g, [{"op": 0, "n_splits": 2}])
        ops = [o for o in out["ops"] if o.get("dispatch_id") is not None]
        self.assertEqual(len(ops), 2)
        t0, t1 = ops
        self.assertEqual(t0["sub_ops"][0]["shape"]["OC"], 8)
        self.assertEqual(t1["sub_ops"][0]["shape"]["OC"], 8)
        self.assertEqual(t0["outputs"], ["y.tile_0"])
        self.assertEqual(t1["outputs"], ["y.tile_1"])
        self.assertEqual(t0["split_from"]["axis"], "OC")
        self.assertEqual(t0["split_from"]["tile_offset_OC"], 0)
        self.assertEqual(t1["split_from"]["tile_offset_OC"], 8)
        self.assertEqual(t1["split_from"]["tile_oc"], 8)

    def test_splits_a_pair_fused_conv_too(self):
        g = _g(_fused_conv_op(0, "c1", kind="conv2d_batchnorm2d_s8", OC=32))
        out = apply_split_hint(g, [{"op": 0, "n_splits": 4}])
        ops = [o for o in out["ops"] if o.get("dispatch_id") is not None]
        self.assertEqual(len(ops), 4)
        self.assertEqual([o["sub_ops"][0]["shape"]["OC"] for o in ops],
                         [8, 8, 8, 8])
        self.assertEqual([o["split_from"]["tile_offset_OC"] for o in ops],
                         [0, 8, 16, 24])

    def test_every_constituent_agrees_about_the_tile_width(self):
        """The conv's OC, the batchnorm's C and the activation's flat n all
        describe the SAME tensor. Narrowing only `sub_ops[0]` would ship a
        graph whose batchnorm claims to process 16 channels from an
        8-channel conv -- and `diff_dispatch_graph` prints exactly that
        signature, so it would be visible in the gate output and wrong."""
        g = _g(_fused_conv_op(0, "l0", OC=16, OH=8, OW=8))
        out = apply_split_hint(g, [{"op": 0, "n_splits": 2}])
        conv, bn, act = out["ops"][0]["sub_ops"]
        self.assertEqual(conv["shape"]["OC"], 8)
        self.assertEqual(bn["shape"]["C"], 8)
        self.assertEqual(act["shape"]["n"], 8 * 8 * 8)

    def test_internal_tensors_are_per_tile_but_the_input_is_not(self):
        """Every tile deep-copies the constituents, so without renaming, both
        tiles claim to produce `l0_conv` -- one tensor, two producers. The
        op's real INPUT is produced elsewhere and must stay shared: all tiles
        read the same input."""
        g = _g(_fused_conv_op(0, "l0", OC=16))
        out = apply_split_hint(g, [{"op": 0, "n_splits": 2}])
        t0, t1 = out["ops"][0], out["ops"][1]
        self.assertEqual(t0["sub_ops"][0]["inputs"], ["x"])
        self.assertEqual(t1["sub_ops"][0]["inputs"], ["x"])
        self.assertEqual(t0["sub_ops"][0]["outputs"], ["l0_conv.tile_0"])
        self.assertEqual(t1["sub_ops"][0]["outputs"], ["l0_conv.tile_1"])
        self.assertEqual(t0["sub_ops"][1]["inputs"], ["l0_conv.tile_0"])
        self.assertEqual(t1["sub_ops"][2]["outputs"], ["y.tile_1"])
        produced = [nm for t in (t0, t1) for s in t["sub_ops"]
                    for nm in s["outputs"]]
        self.assertEqual(len(produced), len(set(produced)))

    def test_tile_output_tensors_are_registered_with_a_narrowed_channel_dim(self):
        """The registration used to be gated on `"shape" in tile_ops[0]`. A
        fused op has no `shape` key, so the gate was False and no tile tensor
        was declared -- surfacing much later as an undeclared
        `buf_<net>_y_tile_0` at link time."""
        g = _g(_fused_conv_op(0, "l0", OC=16, OH=8, OW=8))
        g["tensors"] = {"y": {"shape": [1, 16, 8, 8], "dtype": "i8",
                              "quant": {"scale": 0.05, "zero_point": 0}}}
        out = apply_split_hint(g, [{"op": 0, "n_splits": 2}])
        for t in (0, 1):
            entry = out["tensors"][f"y.tile_{t}"]
            self.assertEqual(entry["shape"], [1, 8, 8, 8])
            self.assertEqual(entry["quant"]["scale"], 0.05)
        self.assertEqual(out["tensors"]["y"]["shape"], [1, 16, 8, 8])

    def test_downstream_depends_on_every_tile(self):
        g = _g(_fused_conv_op(0, "l0", OC=16),
               _fused_conv_op(1, "l1", IC=16, OC=16,
                              inputs=("y",), outputs=("z",), depends_on=(0,)))
        out = apply_split_hint(g, [{"op": 0, "n_splits": 4}])
        tail = out["ops"][-1]
        self.assertEqual(tail["depends_on"], [0, 1, 2, 3])
        self.assertEqual(out["id_remap"]["0"], [0, 1, 2, 3])
        self.assertEqual(out["id_remap"]["1"], [4])

    def test_reject_oc_not_dividing(self):
        g = _g(_fused_conv_op(0, "l0", OC=18))
        with self.assertRaises(SplitHintError):
            apply_split_hint(g, [{"op": 0, "n_splits": 4}])

    def test_reject_a_fused_op_with_no_conv_geometry(self):
        """Refusing loudly beats tiling along an OC read as 0."""
        op = _fused_conv_op(0, "l0", OC=16)
        del op["sub_ops"][0]["shape"]
        with self.assertRaises(SplitHintError):
            apply_split_hint(_g(op), [{"op": 0, "n_splits": 2}])


class UnevenSplit(unittest.TestCase):
    """`tile_sizes` partitions an axis into tiles that are NOT all equal.

    The reason this exists: the two backends a tile can land on do not have
    the same cost curve, so the partition that balances them is generally not
    the even one. Everything downstream keys off `tile_offset_OC` /
    `tile_offset_N` rather than `tile * tile_width`, which is only the even
    special case.
    """

    def _conv_graph(self, oc=32):
        return {
            "name": "net",
            "tensors": {"out": {"shape": [1, oc, 4, 4], "dtype": "i8"}},
            "ops": [
                {"dispatch_id": 0, "op": "conv2d_s8", "name": "c0",
                 "inputs": ["in"], "outputs": ["out"], "depends_on": [],
                 "weight": "w", "bias": "b",
                 "shape": {"N": 1, "IC": 3, "IH": 8, "IW": 8, "OC": oc,
                           "OH": 4, "OW": 4, "KH": 3, "KW": 3, "SH": 2,
                           "SW": 2, "PH": 1, "PW": 1}},
            ],
            "dispatches": [0],
        }

    def test_uneven_conv_tiles_get_cumulative_offsets(self):
        g = apply_split_hint(self._conv_graph(32),
                             [{"op": 0, "tile_sizes": [20, 7, 5]}])
        ops = g["ops"]
        self.assertEqual([o["shape"]["OC"] for o in ops], [20, 7, 5])
        self.assertEqual([o["split_from"]["tile_offset_OC"] for o in ops],
                         [0, 20, 27])
        self.assertEqual([o["split_from"]["tile_oc"] for o in ops], [20, 7, 5])
        self.assertEqual([o["split_from"]["n_splits"] for o in ops], [3, 3, 3])
        self.assertEqual(g["id_remap"], {"0": [0, 1, 2]})

    def test_uneven_tile_tensors_carry_per_tile_shape_and_offset(self):
        g = apply_split_hint(self._conv_graph(32),
                             [{"op": 0, "tile_sizes": [20, 7, 5]}])
        t = g["tensors"]
        # NCHW output [1, OC, 4, 4]; each tile narrows dim 1 and starts at the
        # cumulative element offset, NOT tile_idx * prod(tile_shape).
        self.assertEqual(t["out.tile_0"]["shape"], [1, 20, 4, 4])
        self.assertEqual(t["out.tile_1"]["shape"], [1, 7, 4, 4])
        self.assertEqual(t["out.tile_2"]["shape"], [1, 5, 4, 4])
        self.assertEqual([t[f"out.tile_{i}"]["elem_offset"] for i in range(3)],
                         [0, 20 * 16, 27 * 16])

    def test_even_split_is_unchanged_by_the_uneven_path(self):
        a = apply_split_hint(self._conv_graph(32), [{"op": 0, "n_splits": 2}])
        b = apply_split_hint(self._conv_graph(32),
                             [{"op": 0, "tile_sizes": [16, 16]}])
        self.assertEqual([o["split_from"] for o in a["ops"]],
                         [o["split_from"] for o in b["ops"]])

    def test_partition_that_does_not_sum_to_the_axis_is_rejected(self):
        with self.assertRaises(SplitHintError):
            apply_split_hint(self._conv_graph(32),
                             [{"op": 0, "tile_sizes": [20, 7]}])
        with self.assertRaises(SplitHintError):
            apply_split_hint(self._conv_graph(32),
                             [{"op": 0, "tile_sizes": [32, 0]}])


def _conv2d_oh_op(did, name, N=1, IC=3, IH=8, IW=8, OC=16, KH=3, KW=3,
                  SH=1, SW=1, PH=1, PW=1,
                  inputs=("x",), outputs=("y",), depends_on=()):
    """`_conv2d_op` with stride/padding as free parameters -- the OH split's
    geometry is a function of exactly those, and the OC split's is not, which is
    why the older fixture pins them."""
    op = _conv2d_op(did, name, N=N, IC=IC, IH=IH, IW=IW, OC=OC, KH=KH, KW=KW,
                    inputs=inputs, outputs=outputs, depends_on=depends_on)
    op["shape"].update({"SH": SH, "SW": SW, "PH": PH, "PW": PW})
    op["shape"]["OH"] = (IH + 2 * PH - KH) // SH + 1
    op["shape"]["OW"] = (IW + 2 * PW - KW) // SW + 1
    return op


class Conv2dOhSplitTest(unittest.TestCase):
    """The OH (spatial row) axis. Every assertion here is about geometry, and
    every one of them is load-bearing: the tiles read OVERLAPPING input and
    write DISJOINT output, so an off-by-one in the window silently produces a
    plausible answer rather than a crash."""

    def test_splits_conv_into_two_row_bands(self):
        g = _g(_conv2d_oh_op(0, "c0", IH=8, OC=16, KH=3, SH=1, PH=1))  # OH=8
        out = apply_split_hint(g, [{"op": 0, "n_splits": 2, "axis": "OH"}])
        ops = [o for o in out["ops"] if o.get("dispatch_id") is not None]
        self.assertEqual(len(ops), 2)
        t0, t1 = ops
        self.assertEqual(t0["split_from"]["axis"], "OH")
        self.assertEqual([t0["split_from"]["tile_oh"], t1["split_from"]["tile_oh"]],
                         [4, 4])
        self.assertEqual(t0["split_from"]["tile_offset_OH"], 0)
        self.assertEqual(t1["split_from"]["tile_offset_OH"], 4)
        self.assertEqual(t0["outputs"], ["y.tile_0"])
        self.assertEqual(t1["outputs"], ["y.tile_1"])
        # Every tile keeps ALL output channels -- that is the entire point of
        # this axis, so assert it rather than trusting the deepcopy.
        self.assertEqual(t0["shape"]["OC"], 16)
        self.assertEqual(t1["shape"]["OC"], 16)

    def test_tile_shape_is_a_zero_padded_conv_over_its_own_window(self):
        """IH/PH/OH are rewritten so `(IH + 2*PH - KH)/SH + 1 == OH` holds for
        the TILE. A cost model that multiplies the shape out then gets the
        tile's real work without knowing this axis exists."""
        g = _g(_conv2d_oh_op(0, "c0", IH=112, IW=112, OC=32, KH=3, SH=2, SW=2,
                             PH=1, PW=1))  # OH=56, OW=56
        out = apply_split_hint(g, [{"op": 0, "n_splits": 2, "axis": "OH"}])
        for t in out["ops"]:
            sh = t["shape"]
            self.assertEqual(sh["PH"], 0)
            self.assertEqual(sh["PW"], 0)
            self.assertEqual((sh["IH"] + 2 * sh["PH"] - sh["KH"]) // sh["SH"] + 1,
                             sh["OH"])
            self.assertEqual((sh["IW"] + 2 * sh["PW"] - sh["KW"]) // sh["SW"] + 1,
                             sh["OW"])
            self.assertEqual(sh["OW"], 56)      # width is NOT split
            self.assertEqual(sh["IW"], 112 + 2)  # ... it is PRE-PADDED

    def test_halo_and_per_tile_padding(self):
        """KH=3, SH=2, PH=1 over IH=112: tile 0's window starts one row ABOVE
        the tensor (that row is the conv's top padding) and tile 1's ends flush
        with it. Getting this backwards is the classic per-tile padding bug."""
        g = _g(_conv2d_oh_op(0, "c0", IH=112, IW=112, OC=32, KH=3, SH=2, PH=1))
        out = apply_split_hint(g, [{"op": 0, "n_splits": 2, "axis": "OH"}])
        a, b = [o["split_from"] for o in out["ops"]]
        self.assertEqual((a["in_row_lo"], a["window_rows"]), (-1, 57))
        self.assertEqual((a["pad_top"], a["pad_bot"], a["in_rows"]), (1, 0, 56))
        self.assertEqual((b["in_row_lo"], b["window_rows"]), (55, 57))
        self.assertEqual((b["pad_top"], b["pad_bot"], b["in_rows"]), (0, 0, 57))
        # The windows overlap by KH-SH rows; that overlap IS the halo cost.
        first_end = a["in_row_lo"] + a["window_rows"]
        self.assertGreater(first_end, b["in_row_lo"])
        self.assertEqual(first_end - b["in_row_lo"], 1)

    def test_window_covers_every_input_row_each_output_row_needs(self):
        """Exhaustive over a range of geometries: for every output row the tile
        owns, every tap the conv would read must be inside the tile's window."""
        for IH, KH, SH, PH, k in [(8, 3, 1, 1, 2), (112, 3, 2, 1, 2),
                                  (112, 3, 2, 1, 4), (27, 3, 2, 1, 2),
                                  (27, 1, 2, 0, 2), (14, 3, 1, 1, 2),
                                  (7, 3, 1, 1, 7), (9, 5, 2, 2, 3)]:
            OH = (IH + 2 * PH - KH) // SH + 1
            if OH % k:
                continue
            with self.subTest(IH=IH, KH=KH, SH=SH, PH=PH, k=k):
                g = _g(_conv2d_oh_op(0, "c0", IH=IH, IW=IH, KH=KH, KW=KH,
                                     SH=SH, SW=SH, PH=PH, PW=PH))
                out = apply_split_hint(g, [{"op": 0, "n_splits": k,
                                            "axis": "OH"}])
                covered = []
                for t in out["ops"]:
                    sf = t["split_from"]
                    lo, hi = sf["in_row_lo"], sf["in_row_lo"] + sf["window_rows"]
                    for oh in range(sf["tile_offset_OH"],
                                    sf["tile_offset_OH"] + sf["tile_oh"]):
                        for kh in range(KH):
                            self.assertTrue(lo <= oh * SH - PH + kh < hi)
                    covered += list(range(sf["tile_offset_OH"],
                                          sf["tile_offset_OH"] + sf["tile_oh"]))
                # ... and the tiles partition the output rows exactly once.
                self.assertEqual(covered, list(range(OH)))

    def test_uneven_row_partition(self):
        g = _g(_conv2d_oh_op(0, "c0", IH=14, IW=14, OC=32, KH=3, SH=1, PH=1))
        out = apply_split_hint(g, [{"op": 0, "tile_sizes": [3, 5, 6],
                                    "axis": "OH"}])
        sfs = [o["split_from"] for o in out["ops"]]
        self.assertEqual([s["tile_oh"] for s in sfs], [3, 5, 6])
        self.assertEqual([s["tile_offset_OH"] for s in sfs], [0, 3, 8])
        self.assertEqual([o["shape"]["OH"] for o in out["ops"]], [3, 5, 6])

    def test_tile_tensors_alias_the_parent_at_zero(self):
        """An OH band is NOT contiguous in NCHW, so unlike an OC tile it gets
        offset 0 and the codegen scatters. A non-zero offset here would put
        each tile's channel 0 plane on top of the previous tile's data."""
        g = _g(_conv2d_oh_op(0, "c0", IH=8, IW=8, OC=16, KH=3, SH=1, PH=1))
        g["tensors"]["y"] = {"shape": [1, 16, 8, 8], "dtype": "i8"}
        out = apply_split_hint(g, [{"op": 0, "n_splits": 2, "axis": "OH"}])
        for t in (0, 1):
            e = out["tensors"][f"y.tile_{t}"]
            self.assertEqual(e["shape"], [1, 16, 4, 8])   # H narrowed, not C
            self.assertEqual(e["elem_offset"], 0)
            self.assertEqual(e["alias_kind"], "row_window")

    def test_oc_split_is_still_the_default_for_a_conv(self):
        """Every split hint already on disk omits `axis`; they must keep
        getting OC."""
        g = _g(_conv2d_oh_op(0, "c0", IH=8, OC=16))
        out = apply_split_hint(g, [{"op": 0, "n_splits": 2}])
        self.assertEqual(out["ops"][0]["split_from"]["axis"], "OC")
        self.assertEqual(out["ops"][0]["shape"]["OC"], 8)

    def test_rejects_oh_on_a_kind_with_no_row_window_emitter(self):
        g = _g(_fused_conv_op(0, "f0", OC=16))
        with self.assertRaises(SplitHintError) as cm:
            apply_split_hint(g, [{"op": 0, "n_splits": 2, "axis": "OH"}])
        self.assertIn("cannot be split along OH", str(cm.exception))

    def test_rejects_an_unknown_axis(self):
        g = _g(_conv2d_oh_op(0, "c0", IH=8, OC=16))
        with self.assertRaises(SplitHintError):
            apply_split_hint(g, [{"op": 0, "n_splits": 2, "axis": "OW"}])

    def test_rejects_a_partition_that_is_not_the_output_height(self):
        g = _g(_conv2d_oh_op(0, "c0", IH=8, OC=16))     # OH=8
        with self.assertRaises(SplitHintError):
            apply_split_hint(g, [{"op": 0, "tile_sizes": [3, 3], "axis": "OH"}])
        with self.assertRaises(SplitHintError):
            apply_split_hint(g, [{"op": 0, "n_splits": 3, "axis": "OH"}])

    def test_rejects_dilation(self):
        """The conv2d_s8 kernel signature has no dilation, so a dilated conv
        would need a wider window than (tile_oh-1)*SH+KH -- refuse rather than
        silently understate the halo."""
        g = _g(_conv2d_oh_op(0, "c0", IH=8, OC=16))
        g["ops"][0]["shape"]["DH"] = 2
        with self.assertRaises(SplitHintError):
            apply_split_hint(g, [{"op": 0, "n_splits": 2, "axis": "OH"}])

    def test_a_bad_hint_leaves_the_graph_untouched(self):
        """Validation runs over EVERY spec before any op is rewritten."""
        g = _g(_conv2d_oh_op(0, "c0", IH=8, OC=16),
               _fused_conv_op(1, "f1", OC=16, inputs=("y",), outputs=("z",)))
        before = json.dumps(g, sort_keys=True)
        with self.assertRaises(SplitHintError):
            apply_split_hint(g, [{"op": 0, "n_splits": 2, "axis": "OH"},
                                 {"op": 1, "n_splits": 2, "axis": "OH"}])
        self.assertEqual(json.dumps(g, sort_keys=True), before)

    def test_id_remap_and_depends_on_across_an_oh_split(self):
        g = _g(_conv2d_oh_op(0, "c0", IH=8, OC=16),
               _conv2d_oh_op(1, "c1", IH=8, OC=16, inputs=("y",),
                             outputs=("z",), depends_on=(0,)))
        out = apply_split_hint(g, [{"op": 0, "n_splits": 2, "axis": "OH"}])
        self.assertEqual(out["id_remap"], {"0": [0, 1], "1": [2]})
        consumer = out["ops"][2]
        self.assertEqual(consumer["depends_on"], [0, 1])
