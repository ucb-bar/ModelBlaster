"""Pin the calibration loaders and the window composer.

The composer distinction is the point of this file: `rolling_window` and
`window_stack` produce tensors with the SAME element count from the same
frames, so picking the wrong one traces cleanly and then feeds a
per-timestep ViT six channels of one frame instead of three channels of two.
"""

from __future__ import annotations

import os
import pickle
import sys
import tempfile
import unittest

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(_ROOT))

from modelblaster.mb_datasets import base as mb_base            # noqa: E402
from modelblaster.mb_datasets import bridge_episodes            # noqa: E402,F401
from modelblaster.mb_datasets import synthetic                  # noqa: E402,F401


def _pickle_episodes(path, n_ep=2, n_frames=5, size=64):
    eps = []
    rng = np.random.default_rng(0)
    for i in range(n_ep):
        eps.append({
            "images": rng.integers(0, 256, (n_frames, size, size, 3),
                                   dtype=np.uint8),
            "instr": f"episode {i}",
            "actions": rng.standard_normal((n_frames, 7)).astype(np.float32),
        })
    with open(path, "wb") as fh:
        pickle.dump(eps, fh)
    return n_ep * n_frames


class BridgeEpisodesTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "eps.pkl")
        self.n = _pickle_episodes(self.path)

    def tearDown(self):
        self._tmp.cleanup()

    def test_loads_every_frame_as_nchw(self):
        items = mb_base.load_dataset({"loader": "bridge_episodes",
                                      "path": self.path,
                                      "image_size": [64, 64]})
        self.assertEqual(len(items), self.n)
        self.assertEqual(tuple(items[0].data.shape), (3, 64, 64))

    def test_unit_domain_is_the_default_and_matches_normalize_images(self):
        items = mb_base.load_dataset({"loader": "bridge_episodes",
                                      "path": self.path,
                                      "image_size": [64, 64]})
        d = items[0].data
        # normalize_images(x) = x/127.5 - 1 maps [0,255] onto [-1, 1].
        self.assertGreaterEqual(float(d.min()), -1.0)
        self.assertLessEqual(float(d.max()), 1.0)

    def test_raw_domain_keeps_pixel_values(self):
        items = mb_base.load_dataset({"loader": "bridge_episodes",
                                      "path": self.path,
                                      "image_size": [64, 64],
                                      "domain": "raw"})
        self.assertGreater(float(items[0].data.max()), 1.0)

    def test_resizes_when_asked_for_a_smaller_camera(self):
        items = mb_base.load_dataset({"loader": "bridge_episodes",
                                      "path": self.path,
                                      "image_size": [32, 32]})
        self.assertEqual(tuple(items[0].data.shape), (3, 32, 32))

    def test_single_camera_episodes_serve_a_wrist_request(self):
        # These episodes carry only "images". Reusing them for the wrist stem
        # is closer to right than calibrating it on noise, and the meta says
        # which key was actually used.
        items = mb_base.load_dataset({"loader": "bridge_episodes",
                                      "path": self.path, "key": "image_wrist",
                                      "image_size": [32, 32]})
        self.assertEqual(len(items), self.n)
        self.assertEqual(items[0].meta["requested_key"], "image_wrist")
        self.assertEqual(items[0].meta["key"], "images")

    def test_missing_file_is_an_error_not_an_empty_set(self):
        with self.assertRaises(FileNotFoundError):
            mb_base.load_dataset({"loader": "bridge_episodes",
                                  "path": self.path + ".nope",
                                  "image_size": [32, 32]})


class ComposerTests(unittest.TestCase):

    def setUp(self):
        self.items = [mb_base.DatasetItem(data=torch.full((3, 8, 8), float(i)))
                      for i in range(6)]

    def test_window_stack_gives_the_window_its_own_axis(self):
        out = mb_base._compose_window_stack(
            self.items, {"frames_per_sample": 2}, 3)
        self.assertEqual(len(out), 3)
        self.assertEqual(tuple(out[0].shape), (1, 2, 3, 8, 8))

    def test_rolling_window_stacks_the_same_frames_on_channels(self):
        out = mb_base._compose_rolling_window(
            self.items, {"frames_per_sample": 2}, 3)
        self.assertEqual(tuple(out[0].shape), (1, 6, 8, 8))
        # Same element count as window_stack -- which is exactly why picking
        # the wrong one is silent.
        self.assertEqual(out[0].numel(),
                         mb_base._compose_window_stack(
                             self.items, {"frames_per_sample": 2}, 3)[0].numel())

    def test_window_stack_frames_are_consecutive_and_ordered(self):
        out = mb_base._compose_window_stack(
            self.items, {"frames_per_sample": 3}, 2)
        # Last frame of the window is the anchor; earlier ones precede it.
        vals = [float(out[1][0, k, 0, 0, 0]) for k in range(3)]
        self.assertEqual(vals, sorted(vals))
        self.assertEqual(vals[2] - vals[0], 2.0)


class SyntheticTests(unittest.TestCase):

    def test_shapes_carry_no_batch_dim(self):
        items = mb_base.load_dataset({"loader": "synthetic",
                                      "shape": [16, 768], "n_items": 4})
        self.assertEqual(len(items), 4)
        self.assertEqual(tuple(items[0].data.shape), (16, 768))

    def test_seed_makes_it_reproducible(self):
        a = mb_base.load_dataset({"loader": "synthetic", "shape": [4],
                                  "n_items": 2, "seed": 3})
        b = mb_base.load_dataset({"loader": "synthetic", "shape": [4],
                                  "n_items": 2, "seed": 3})
        torch.testing.assert_close(a[0].data, b[0].data)

    def test_const_kind_is_exactly_the_scale(self):
        items = mb_base.load_dataset({"loader": "synthetic", "shape": [2, 1],
                                      "kind": "const", "scale": 7.0,
                                      "n_items": 1})
        self.assertTrue(bool((items[0].data == 7.0).all()))

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            mb_base.load_dataset({"loader": "synthetic", "shape": [2],
                                  "kind": "poisson", "n_items": 1})


if __name__ == "__main__":
    unittest.main()
