"""A fused op must not collapse into a shapeless join key.

Why this matters more than it looks
-----------------------------------
Cross-rung comparison in this project has one hard rule: join on `module_name`,
never on `dispatch_id`, because a split or a `realize-hint` renumbers every
dispatch downstream. `module_name` is built from the op kind plus its shape.

A FUSED op carries no `shape` of its own -- the dimensions live on its sub_ops --
so `_shape_str` returned "" and the name rendered as `..._<op>_noshape`. Every
fused convolution in a model then shared ONE key. Measured on the real
authoritative profile:

    yolov8_nano   57 of 90 dispatches -> conv2d_batchnorm2d_silu_s8_noshape
                  spanning 0.605 ms to 17.465 ms, a 29x range
    dronet         3 of 21 -> conv2d_batchnorm2d_s8_noshape

A key 57 dispatches share is not a key. Nothing was broken while no rewrite
touched that family, which is exactly why it needed a test: the failure would
have appeared as an unjoinable model at the first fuse or split, one layer away
from where the cause lived.

This is the second bug of the same shape. The first was `_shape_concise`
returning the literal string "scalar" for a shapeless op, which put the word
"scalar" in module_name where a reader looks for the implementation and made a
correct vector build look like it had fallen back. Same root cause: a fused op
has no shape and the code assumed one.
"""

from __future__ import annotations

import os
import sys
import unittest

# `src` FIRST, then the repo root. The venv carries an editable install that
# resolves `modelblaster` to a SIBLING clone at /scratch2/agustin/ModelBlaster,
# so importing without this runs that checkout's generate_skeleton and the
# assertions below silently describe the wrong tree -- which is exactly how the
# first version of this test "failed" against a fix that was already applied.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", ".."))
for _p in (os.path.join(_ROOT, "src"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from modelblaster.pipeline.generate_skeleton import (  # noqa: E402
    _shape_str, __file__ as _gs_file,
)

if not os.path.abspath(_gs_file).startswith(_ROOT):
    raise RuntimeError(
        f"imported generate_skeleton from {_gs_file}, not this checkout "
        f"({_ROOT}) -- a sibling clone is shadowing it; run with "
        f"PYTHONPATH=src:.")


def _fused(*shapes):
    return {"op": "conv2d_batchnorm2d_silu_s8", "shape": None,
            "sub_ops": [{"op": f"s{i}", "shape": s}
                        for i, s in enumerate(shapes)]}


class FusedOpsKeepTheirProducingShape(unittest.TestCase):

    def test_a_fused_op_reports_its_convolution_shape(self):
        got = _shape_str(_fused({"N": 1, "IC": 3, "OC": 16},
                                {"N": 1, "C": 16}))
        self.assertEqual(got, "N=1;IC=3;OC=16")

    def test_two_fused_convs_at_different_shapes_do_not_collide(self):
        """The actual defect: distinct work under one key."""
        a = _shape_str(_fused({"N": 1, "IC": 3, "OC": 16}))
        b = _shape_str(_fused({"N": 1, "IC": 64, "OC": 128}))
        self.assertNotEqual(a, b)
        self.assertTrue(a and b)

    def test_the_first_sub_op_with_a_shape_wins(self):
        """Not the first sub_op unconditionally -- it may carry none."""
        got = _shape_str(_fused(None, {"N": 1, "C": 8}))
        self.assertEqual(got, "N=1;C=8")

    def test_an_unfused_op_is_unaffected(self):
        self.assertEqual(_shape_str({"op": "conv2d_s8",
                                     "shape": {"N": 1, "IC": 3}}),
                         "N=1;IC=3")

    def test_a_genuinely_shapeless_op_still_returns_empty(self):
        """A `view` has no shape anywhere; inventing one would be worse."""
        self.assertEqual(_shape_str({"op": "view"}), "")
        self.assertEqual(_shape_str({"op": "view", "sub_ops": []}), "")

    def test_list_valued_dims_still_use_the_pipe_separator(self):
        """Shapes land in a comma-delimited CSV; a list must not add commas."""
        got = _shape_str(_fused({"C_inputs": [16, 16, 16]}))
        self.assertEqual(got, "C_inputs=16|16|16")
        self.assertNotIn(",", got)


class AgainstTheRealGraphs(unittest.TestCase):
    """Skips when the extracted graphs are not present."""

    def _graph(self, model):
        import json
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "build", "k1", model, "int8", "graph.json")
        if not os.path.exists(p):
            self.skipTest(f"no extracted graph for {model}")
        with open(p) as fh:
            return json.load(fh)

    def test_yolov8_nanos_fused_convs_are_distinguishable(self):
        """57 of 90 shared one key before this fix."""
        import collections
        ops = self._graph("yolov8_nano")["ops"]
        keys = collections.Counter(
            f'{o["op"]}_{_shape_str(o) or "noshape"}' for o in ops)
        shapeless = sum(c for k, c in keys.items() if k.endswith("_noshape"))
        self.assertEqual(shapeless, 0,
                         "no yolov8_nano op should key to _noshape")
        self.assertLess(keys.most_common(1)[0][1], 20,
                        "a key shared by dozens of dispatches is not a key")


if __name__ == "__main__":
    unittest.main()
