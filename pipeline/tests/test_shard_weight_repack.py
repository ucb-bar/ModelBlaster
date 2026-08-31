"""An OC shard must get its OWN re-packed weights, not a pointer offset.

Why this file exists
--------------------
`parallel_conv2d_s8` shards a convolution across `modelblaster_pool` by
splitting OC and handing shard `t` the pointer `w + t*oc_per*IC*KH*KW`. That is
an OIHW offset, and the rvv backends pack conv weights IHWOC (OC innermost), so
the slice is strided and no offset can express it. The wrapper was therefore
compiled out entirely on a repacking backend, which is why intra-op sharding has
never run on rvv at all.

`shard_conv_weights` gives each shard its own array -- sliced in OIHW, then
packed -- so it strides by `oc_per`, exactly the `OC` the kernel is called with,
and reads from element 0. Same mechanism `split_conv_tile_weights` uses for
split tiles, reached from the other direction.

Three properties carry the design, and each has a test below because getting
any of them wrong is silent:

  * a pointer offset into the PACKED parent is not the shard's data (so the
    repack is necessary at all);
  * `concat(pack(shards)) != pack(full)` (so the no-pool path must loop the
    shards rather than make one full-OC call);
  * the shards partition the output channels exactly (so looping them is
    numerically identical, not an approximation -- OC is an output axis, so
    slicing it changes no accumulation order).
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pipeline import generate_skeleton as G  # noqa: E402


def _conv_ir(oc=8, ic=3, kh=3, kw=3, op="conv2d_s8"):
    conv = {
        "name": "c0", "op": "conv2d_s8", "inputs": ["x"], "outputs": ["y"],
        "weight": "c0.weight_q", "bias": "c0.bias_q",
        "shape": {"N": 1, "IC": ic, "IH": 8, "IW": 8, "OC": oc,
                  "OH": 4, "OW": 4, "KH": kh, "KW": kw,
                  "SH": 2, "SW": 2, "PH": 1, "PW": 1},
        "quant": {},
    }
    if op == "conv2d_s8":
        node = dict(conv, dispatch_id=0)
    else:                       # fused: the conv lives in sub_ops[0]
        node = {"name": "c0", "op": op, "inputs": ["x"], "outputs": ["y"],
                "dispatch_id": 0, "sub_ops": [conv, {"op": "batchnorm2d_s8"}]}
    return {"name": "probe", "ops": [node]}


class ThePlan(unittest.TestCase):

    def test_a_repacking_backend_gets_a_plan(self):
        p = G.shard_conv_weights(_conv_ir(oc=8), "rvv_x60", 4)
        self.assertEqual(len(p), 1)
        self.assertEqual(p[0]["oc_per"], 2)
        self.assertEqual(p[0]["n_shards"], 4)

    def test_an_oihw_backend_gets_none(self):
        """The pointer arithmetic is already right there; duplicating the
        arrays would only waste space."""
        self.assertEqual(G.shard_conv_weights(_conv_ir(), "scalar", 4), {})

    def test_factor_one_is_off(self):
        self.assertEqual(G.shard_conv_weights(_conv_ir(), "rvv_x60", 1), {})

    def test_oc_not_divisible_is_skipped_rather_than_special_cased(self):
        self.assertEqual(G.shard_conv_weights(_conv_ir(oc=6), "rvv_x60", 4), {})

    def test_a_fused_conv_is_planned_too(self):
        """The fused ops are where the time is -- 97% of yolov8n. Their shape
        lives in sub_ops[0], which is the same trap that made the profile
        write `noshape` for 57 dispatches."""
        p = G.shard_conv_weights(
            _conv_ir(oc=8, op="conv2d_batchnorm2d_silu_s8"), "rvv_x60", 4)
        self.assertEqual(len(p), 1)
        self.assertEqual(p[0]["oc_per"], 2)

    def test_a_split_tile_is_never_also_sharded(self):
        ir = _conv_ir(oc=8)
        ir["ops"][0]["split_from"] = {"axis": "OC", "tile": 0, "n_splits": 2}
        self.assertEqual(G.shard_conv_weights(ir, "rvv_x60", 4), {})


class ThePackingPropertiesTheDesignRestsOn(unittest.TestCase):

    def setUp(self):
        self.w = np.arange(8 * 3 * 2 * 2, dtype=np.int8).reshape(8, 3, 2, 2)
        self.full, _ = G._backend_pack_weight(self.w, "rvv_x60")

    def _packed_shards(self, n):
        oc = self.w.shape[0] // n
        return [G._backend_pack_weight(
            np.ascontiguousarray(self.w[i * oc:(i + 1) * oc]), "rvv_x60")[0]
            for i in range(n)]

    def test_a_pointer_offset_into_the_packed_parent_is_wrong(self):
        """If this ever passes, the repack is unnecessary and this whole
        mechanism should be deleted."""
        s0 = self._packed_shards(4)[0]
        self.assertFalse(
            np.array_equal(self.full.ravel()[:s0.size], s0.ravel()),
            "an offset into the packed parent must NOT yield the shard")

    def test_concatenating_the_shards_does_not_reconstruct_the_parent(self):
        """Why the no-pool path loops the shards instead of one full call."""
        cat = np.concatenate([s.ravel() for s in self._packed_shards(4)])
        self.assertFalse(np.array_equal(cat, self.full.ravel()))

    def test_the_shards_partition_the_output_channels_exactly(self):
        """Why looping them is exact rather than approximate."""
        oc = self.w.shape[0] // 4
        back = np.concatenate([self.w[i * oc:(i + 1) * oc] for i in range(4)])
        self.assertTrue(np.array_equal(back, self.w))


class TheEmittedArrays(unittest.TestCase):

    def test_each_shard_array_is_its_own_oc_slice(self):
        w = {"c0.weight_q": np.arange(8 * 3 * 2 * 2,
                                      dtype=np.int8).reshape(8, 3, 2, 2)}
        plan = G.shard_conv_weights(_conv_ir(oc=8), "rvv_x60", 4)
        extra, drop = G._apply_shard_weight_plan(w, plan)
        self.assertEqual(len(extra), 4)
        self.assertIn("c0.weight_q", drop, "the parent has no consumer left")
        for t in range(4):
            got = extra[G._shard_key("c0.weight_q", t)]
            self.assertTrue(np.array_equal(got, w["c0.weight_q"][t * 2:(t + 1) * 2]))

    def test_arrays_are_returned_in_oihw_not_pre_packed(self):
        """`emit_weights` packs everything it emits; packing here too would
        permute twice."""
        w = {"c0.weight_q": np.arange(8 * 3 * 2 * 2,
                                      dtype=np.int8).reshape(8, 3, 2, 2)}
        extra, _ = G._apply_shard_weight_plan(
            w, G.shard_conv_weights(_conv_ir(oc=8), "rvv_x60", 4))
        self.assertEqual(extra[G._shard_key("c0.weight_q", 0)].shape,
                         (2, 3, 2, 2), "still OIHW")


class TheEnvKnob(unittest.TestCase):

    def test_default_is_off(self):
        saved = os.environ.pop("MB_SHARD_FACTOR", None)
        try:
            self.assertEqual(G.shard_factor(), 1)
        finally:
            if saved is not None:
                os.environ["MB_SHARD_FACTOR"] = saved

    def test_a_bad_value_is_refused_not_defaulted(self):
        saved = os.environ.get("MB_SHARD_FACTOR")
        os.environ["MB_SHARD_FACTOR"] = "four"
        try:
            with self.assertRaises(SystemExit):
                G.shard_factor()
        finally:
            if saved is None:
                os.environ.pop("MB_SHARD_FACTOR", None)
            else:
                os.environ["MB_SHARD_FACTOR"] = saved


if __name__ == "__main__":
    unittest.main()
