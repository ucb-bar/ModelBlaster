"""Pin what conv weight packing does and does not apply to.

`_backend_pack_weight` used to derive one layout per backend and apply it to
every 4-D weight tensor. Packing is per-op now, and these tests hold the
three cases apart, because the failure mode of getting them wrong is a
transposed tensor and a plausible-looking wrong answer:

  * a tensor an op claims  -> packed to THAT op's contract
  * a tensor two ops claim with different contracts -> hard error
  * a 4-D tensor no op claims -> left alone, because rank 4 is not the same
    question as "is a conv filter"
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from pipeline import generate_skeleton as G  # noqa: E402

# A backend where conv2d_s8 packs away from OIHW, so "was it packed?" is
# observable as a shape change.
_REPACKING = "rvv_x60"


class WhatGetsPacked(unittest.TestCase):

    def setUp(self):
        self.w = np.arange(8 * 3 * 2 * 2, dtype=np.int8).reshape(8, 3, 2, 2)

    def test_the_backend_repacks_a_claimed_conv2d_s8_weight(self):
        out, tag = G._backend_pack_weight(self.w, _REPACKING, {"conv2d_s8"})
        self.assertIsNotNone(tag)
        self.assertNotEqual(out.shape, self.w.shape)

    def test_an_unclaimed_4d_tensor_is_left_alone_when_the_ir_is_known(self):
        # octo's IR carries a (1,1,690,690) attention mask and three
        # positional-embedding tables. They are rank 4 and they are not conv
        # filters; packing them IHWOC would transpose a position embedding.
        out, tag = G._backend_pack_weight(
            self.w, _REPACKING, owner_ops=set(), ir_known=True)
        self.assertIsNone(tag)
        np.testing.assert_array_equal(out, self.w)

    def test_no_ir_still_falls_back_to_the_conv2d_s8_layout(self):
        # The caller has asserted "this is a conv weight" and there is
        # nothing better to go on. This is the path the retired cross-op
        # guard used to kill.
        out, tag = G._backend_pack_weight(self.w, _REPACKING)
        self.assertIsNotNone(tag)
        self.assertNotEqual(out.shape, self.w.shape)

    def test_one_tensor_claimed_by_two_disagreeing_ops_is_an_error(self):
        # This is the real protection: one tensor cannot be packed two ways.
        s8 = G._conv_weight_layout_for_op("conv2d_s8", _REPACKING)
        dw = G._conv_weight_layout_for_op("depthwise_conv2d_s8", _REPACKING)
        if s8 == dw:
            self.skipTest(f"conv2d_s8 and depthwise_conv2d_s8 agree "
                          f"({s8!r}) on {_REPACKING}; nothing to disagree on")
        with self.assertRaises(SystemExit):
            G._backend_pack_weight(
                self.w, _REPACKING, {"conv2d_s8", "depthwise_conv2d_s8"})

    def test_a_non_4d_tensor_is_never_touched(self):
        b = np.arange(8, dtype=np.int32)
        out, tag = G._backend_pack_weight(b, _REPACKING, {"conv2d_s8"})
        self.assertIsNone(tag)
        np.testing.assert_array_equal(out, b)


class PerOpLayouts(unittest.TestCase):

    def test_int8_and_fp16_convs_may_disagree_on_the_same_backend(self):
        # Explicitly legitimate: they are different dtypes and never share a
        # tensor. Firing on this pair is what made rvv_f16 unbuildable.
        s8 = G._conv_weight_layout_for_op("conv2d_s8", "rvv_f16")
        f16 = G._conv_weight_layout_for_op("conv2d_f16", "rvv_f16")
        self.assertEqual(s8, "ihwoc")
        self.assertIn(f16, (None, "oihw"))

    def test_an_fp16_conv_weight_is_not_packed_ihwoc_on_rvv_f16(self):
        w = np.arange(4 * 3 * 2 * 2, dtype=np.float16).reshape(4, 3, 2, 2)
        out, tag = G._backend_pack_weight(w, "rvv_f16", {"conv2d_f16"})
        self.assertIsNone(tag)
        np.testing.assert_array_equal(out, w)


if __name__ == "__main__":
    unittest.main()
