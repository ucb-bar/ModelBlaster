"""YOLOv8's input size must be able to match the camera's aspect ratio.

Why this exists
---------------
`MODELBLASTER_YOLOV8N_INPUT` was a single int, so the model was always square.
The FPV camera on this project is 90x60 (W x H), aspect 3:2, so a square input
forces a letterbox -- and the padding is then convolved at full cost through
the whole backbone. Against the fitted per-shape cost model
(XPU-RT/scripts/predict_conv_cost.py) a 96x96 letterbox of that frame is
88.5 ms, of which roughly 26 ms is spent convolving grey bars. The matching
64x96 rect is 62.6 ms with zero padding.

That is not a speed/accuracy trade: the content box inside a letterboxed 96x96
IS 64x96, so the rect keeps every real pixel. It is declining to do arithmetic
on constant input.

The multiple-of-32 rule is not cosmetic -- YOLOv8's deepest level downsamples
by 32, so a non-multiple has no valid stride-32 feature map. It now applies per
DIMENSION, which is what catches a plausible-looking `96x48`.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from modelblaster.models.yolov8_nano import parse_input_size
    _IMPORT_ERR = None
except Exception as exc:                     # torch absent in some envs
    parse_input_size = None
    _IMPORT_ERR = exc


@unittest.skipIf(parse_input_size is None, f"yolov8_nano unimportable: {_IMPORT_ERR}")
class ParsingTheSize(unittest.TestCase):

    def test_a_bare_number_is_still_square(self):
        """Backward compatibility: every existing caller passes one int."""
        self.assertEqual(parse_input_size("160"), (160, 160))
        self.assertEqual(parse_input_size("96"), (96, 96))

    def test_a_pair_is_height_then_width(self):
        """H x W, matching torch's NCHW and transforms.Resize((h, w)).

        Getting this order backwards would silently train on a transposed
        frame, which looks like a bad model rather than a bad config.
        """
        self.assertEqual(parse_input_size("64x96"), (64, 96))

    def test_the_separator_is_forgiving_but_the_order_is_not(self):
        for raw in ("64x96", "64,96", " 64X96 "):
            self.assertEqual(parse_input_size(raw), (64, 96), raw)

    def test_every_dimension_must_be_a_multiple_of_32(self):
        """`96x48` is the dangerous one: half of it is legal."""
        for raw in ("48", "96x48", "48x96", "0", "64x0"):
            with self.assertRaises(SystemExit, msg=raw):
                parse_input_size(raw)

    def test_the_refusal_names_the_offending_dimension(self):
        with self.assertRaises(SystemExit) as cm:
            parse_input_size("96x48")
        self.assertIn("48", str(cm.exception))
        self.assertNotIn("96,", str(cm.exception).replace("96x48", ""))

    def test_nonsense_is_refused_with_the_accepted_forms(self):
        for raw in ("abc", "64x96x32", ""):
            with self.assertRaises(SystemExit, msg=raw):
                parse_input_size(raw)


@unittest.skipIf(parse_input_size is None, "yolov8_nano unimportable")
class TheModelActuallyBuilds(unittest.TestCase):
    """Parsing a pair is worth nothing if the network cannot consume it."""

    def _forward(self, size):
        import torch
        os.environ["MODELBLASTER_YOLOV8N_INPUT"] = size
        os.environ["MODELBLASTER_YOLOV8N_PRETRAINED"] = "0"
        # Re-import under the new env: _cfg() reads it at call time, but the
        # module may already be cached with a different sample input.
        from modelblaster.models import yolov8_nano as y
        x = y.get_sample_input(seed=1)
        m = y.get_model(seed=0)
        with torch.no_grad():
            out = m(x)
        return tuple(x.shape), out

    def test_a_rectangular_input_produces_a_rectangular_feature_map(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed in this env")
        shape, out = self._forward("64x96")
        self.assertEqual(shape[2:], (64, 96))
        o = out[0] if isinstance(out, (tuple, list)) else out
        # stride 8 at the shallowest head level: 64/8 x 96/8
        self.assertEqual(tuple(o.shape)[-2:], (8, 12))

    def test_square_is_unchanged(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed in this env")
        shape, out = self._forward("160")
        self.assertEqual(shape[2:], (160, 160))
        o = out[0] if isinstance(out, (tuple, list)) else out
        self.assertEqual(tuple(o.shape)[-2:], (20, 20))


if __name__ == "__main__":
    unittest.main()
