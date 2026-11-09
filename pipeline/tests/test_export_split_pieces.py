"""torch.split pieces and elided tail ops in the export extractor.

A getitem of torch.split is a range of its source. It may alias the source
only when the consuming cat reassembles the whole split byte for byte; any
other use needs a copy of the range (slice_c). A tail op that changes the
element count (F.normalize's norm) is elided, never aliased, and no emitted op
may read it. Each case here is a tiny exported module run through the walker.
"""

from __future__ import annotations

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from pipeline import extract_graph_export as ege  # noqa: E402


def _walk(module, *inputs):
    module.eval()
    ep = torch.export.export(module, tuple(inputs))
    names = [s.arg.name for s in ep.graph_signature.input_specs
             if s.kind.name == "USER_INPUT"]
    calib = [{n: t for n, t in zip(names, inputs)}]
    w = ege._ExportWalker(ep, calib, names, per_channel=True, quant="int8",
                          op_precision={})
    w.calibrate()
    w.emit()
    return w


class _SplitCatConv(torch.nn.Module):
    """ViNT's frame unstack: (B, 3k, H, W) -> (B*k, 3, H, W), then a conv."""

    def __init__(self, order=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1)
        self.order = order

    def forward(self, x):
        pieces = torch.split(x, 3, dim=1)
        if self.order is not None:
            pieces = [pieces[i] for i in self.order]
        return self.conv(torch.cat(pieces, dim=0))


class _PieceToConv(torch.nn.Module):
    """One piece of a split feeding a conv directly."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1)

    def forward(self, x):
        return self.conv(torch.split(x, 3, dim=1)[1])


class SplitPieceTests(unittest.TestCase):

    def test_batch1_in_order_reassembly_is_an_alias(self):
        x = torch.randn(1, 9, 4, 5)
        w = _walk(_SplitCatConv(), x)
        kinds = [op["op"] for op in w.ops]
        self.assertNotIn("slice_c_s8", kinds)
        self.assertFalse(any(k.startswith("cat") for k in kinds), kinds)
        conv = next(op for op in w.ops if op["op"].startswith("conv2d"))
        self.assertEqual(conv["inputs"], ["x"])

    def test_batch2_reassembly_copies_the_pieces(self):
        # (2, 9, H, W) split on dim 1: each piece interleaves the two batch
        # rows, so the cat output is NOT the source's bytes.
        x = torch.randn(2, 9, 4, 5)
        w = _walk(_SplitCatConv(), x)
        slices = [op for op in w.ops if op["op"] == "slice_c_s8"]
        self.assertEqual(len(slices), 3)
        self.assertEqual([(s["shape"]["C_start"], s["shape"]["C_end"])
                          for s in slices], [(0, 3), (3, 6), (6, 9)])
        self.assertTrue(all(s["shape"]["N"] == 2 for s in slices))
        cat = next(op for op in w.ops if op["op"].startswith("cat"))
        self.assertEqual(cat["inputs"], [s["outputs"][0] for s in slices])

    def test_reordered_reassembly_copies_the_pieces(self):
        x = torch.randn(1, 9, 4, 5)
        w = _walk(_SplitCatConv(order=(2, 0, 1)), x)
        slices = {op["outputs"][0]: (op["shape"]["C_start"], op["shape"]["C_end"])
                  for op in w.ops if op["op"] == "slice_c_s8"}
        self.assertEqual(len(slices), 3)
        cat = next(op for op in w.ops if op["op"].startswith("cat"))
        self.assertEqual([slices[t] for t in cat["inputs"]],
                         [(6, 9), (0, 3), (3, 6)])

    def test_piece_into_conv_is_a_copy_of_its_range(self):
        x = torch.randn(1, 9, 4, 5)
        w = _walk(_PieceToConv(), x)
        slices = [op for op in w.ops if op["op"] == "slice_c_s8"]
        self.assertEqual(len(slices), 1)  # the unused pieces copy nothing
        sl = slices[0]
        self.assertEqual((sl["shape"]["C_start"], sl["shape"]["C_end"]), (3, 6))
        # A copy keeps the source's scale, so it cannot round.
        self.assertEqual(sl["quant"]["scale_in"], sl["quant"]["scale_out"])
        conv = next(op for op in w.ops if op["op"].startswith("conv2d"))
        self.assertEqual(conv["inputs"], sl["outputs"])


class _NormTail(torch.nn.Module):
    def __init__(self, reuse_norm=False):
        super().__init__()
        self.fc = torch.nn.Linear(8, 4)
        self.reuse_norm = reuse_norm

    def forward(self, x):
        y = self.fc(x)
        n = torch.linalg.vector_norm(y, dim=-1, keepdim=True)
        if self.reuse_norm:
            # A compute op reading the norm: nothing would compute it.
            return torch.relu(n)
        return y / n.clamp_min(1e-12).expand_as(y)


class TailTests(unittest.TestCase):

    def test_shrinking_tail_is_elided(self):
        w = _walk(_NormTail(), torch.randn(3, 8))
        self.assertTrue(any("vector_norm" in t for t in w.elided), w.elided)
        self.assertTrue(any("expand" in t for t in w.elided), w.elided)
        for t in w.elided:
            self.assertNotIn(t, w.name_map)
        self.assertEqual([op["op"] for op in w.ops], ["linear_s8_pc"])

    def test_reading_an_elided_tail_is_an_error(self):
        with self.assertRaises(NotImplementedError):
            _walk(_NormTail(reuse_norm=True), torch.randn(3, 8))


if __name__ == "__main__":
    unittest.main()


class _PadRelu(torch.nn.Module):
    """Ops whose kernels move or clamp codes without requantizing."""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 8)
        with torch.no_grad():
            self.fc.bias.fill_(-3.0)  # largest |relu input| is negative
        self.conv0 = torch.nn.Conv2d(2, 2, 1)
        self.conv = torch.nn.Conv2d(2, 2, 3, stride=2)

    def forward(self, x, img):
        y = torch.relu(self.fc(x))
        z = self.conv(F.pad(self.conv0(img), (0, 1, 0, 1)))
        return y, z


class CodeCopyScaleTests(unittest.TestCase):

    def test_pad_and_relu_carry_their_input_scale(self):
        w = _walk(_PadRelu(), torch.randn(3, 8), torch.randn(1, 2, 8, 8))
        checked = 0
        for op in w.ops:
            if op["op"] in ("pad_s8", "relu_s8"):
                i, o = op["inputs"][0], op["outputs"][0]
                self.assertEqual(op["quant"]["scale_in"], op["quant"]["scale_out"])
                self.assertEqual(w.tensors_meta[o]["quant"]["scale"],
                                 w.tensors_meta[i]["quant"]["scale"])
                checked += 1
        self.assertEqual(checked, 2)
