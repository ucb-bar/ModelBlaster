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
