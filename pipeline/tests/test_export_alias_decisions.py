"""Pin the export extractor's alias-vs-copy decisions.

Each of these predicates decides whether a view op is free or has to become a
kernel call. Getting one wrong does not fail a build -- it hands the next
kernel the right bytes in the wrong order, or the wrong number of them.
"""

from __future__ import annotations

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from pipeline import extract_graph_export as ege  # noqa: E402



class PermOrderTests(unittest.TestCase):

    def test_identity_is_free(self):
        self.assertTrue(ege._perm_preserves_order((0, 1, 2, 3), (2, 3, 4, 5)))

    def test_swapping_unit_axes_is_free(self):
        # pos_readout[:, :w] shaped (1, 2, 1, 384): the moved axes have
        # extent 1, so the flat order is unchanged.
        self.assertTrue(ege._perm_preserves_order((0, 2, 1, 3), (1, 2, 1, 384)))

    def test_head_split_is_not_free(self):
        # (1, S, H, D) -> (1, H, S, D) reorders every element.
        self.assertFalse(ege._perm_preserves_order((0, 2, 1, 3), (1, 690, 6, 64)))

    def test_nchw_to_nhwc_is_not_free(self):
        self.assertFalse(ege._perm_preserves_order((0, 2, 3, 1), (2, 512, 16, 16)))

    def test_last_two_axis_swap_is_not_free(self):
        self.assertFalse(ege._perm_preserves_order((0, 1, 3, 2), (1, 6, 690, 64)))


class BroadcastRoleTests(unittest.TestCase):

    def test_channel_gate_is_recognized_in_both_operand_orders(self):
        nchw = (2, 32, 128, 128)
        for gate in ((32,), (1, 32, 1, 1), (2, 32, 1, 1)):
            with self.subTest(gate=gate):
                idx, shape = ege._ExportWalker._channel_gate_roles(gate, nchw)
                self.assertEqual((idx, shape), (0, nchw))
                idx, shape = ege._ExportWalker._channel_gate_roles(nchw, gate)
                self.assertEqual((idx, shape), (1, nchw))

    def test_equal_shapes_are_not_a_gate(self):
        idx, _ = ege._ExportWalker._channel_gate_roles((2, 32, 4, 4), (2, 32, 4, 4))
        self.assertIsNone(idx)

    def test_shared_attention_mask_is_a_tile(self):
        # (1,1,S,S) added to (1,heads,S,S): one S*S block, repeated per head.
        idx, outer, inner = ege._ExportWalker._tile_roles(
            (1, 6, 690, 690), (1, 1, 690, 690))
        self.assertEqual((idx, outer, inner), (1, 6, 690 * 690))
        idx, outer, inner = ege._ExportWalker._tile_roles(
            (1, 1, 690, 690), (1, 6, 690, 690))
        self.assertEqual((idx, outer, inner), (0, 6, 690 * 690))

    def test_channel_gate_is_not_a_tile(self):
        # A length-C gate is NOT a repeated trailing block, so the tile path
        # must decline it and leave it to add_c1.
        idx, _, _ = ege._ExportWalker._tile_roles((2, 32, 128, 128), (32,))
        self.assertIsNone(idx)

    def test_equal_element_counts_are_not_a_tile(self):
        idx, _, _ = ege._ExportWalker._tile_roles((6, 4), (6, 4))
        self.assertIsNone(idx)


if __name__ == "__main__":
    unittest.main()
