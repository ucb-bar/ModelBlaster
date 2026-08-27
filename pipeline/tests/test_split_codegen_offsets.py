"""The generated C for a split tile must address ITS OWN slice of the weights.

Why this file exists
--------------------
`apply_split_hint` produced perfectly good IR for a split `linear_s8` -- correct
tile shapes, correct `split_from` lineage, correct consumer rewiring -- and
`generate_skeleton` then emitted two kernel calls that differed only in their
OUTPUT pointer:

    parallel_linear_s8(..., w, b, buf_y,        1, 32, 32, ...);
    parallel_linear_s8(..., w, b, (buf_y + 32), 1, 32, 32, ...);

Weights are `[N, K]` and the kernel indexes `weight[n*K + k]`, so both tiles
computed output rows [0, 32) and the second wrote those duplicates into
y[32:64]. Every element above the first tile was wrong. No crash, no build
error, no failing test -- because the whole existing suite checks the IR and
nothing checked the emitted C.

The bug survived a long time for a specific reason worth remembering: the only
model ever split in anger was DroNet, whose linear layers are N=1. Those are
rejected by the divisibility check before codegen is reached, so only the
`conv2d_s8` arm -- which does emit the offset -- was ever exercised. The comment
in that arm records this exact failure being diagnosed and fixed for conv, and
the linear arm beside it was left alone.

So these tests assert on the generated source text. That is unusual and slightly
brittle, and it is the point: an IR-level assertion cannot see this class of
defect at all.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.apply_split_hint import apply_split_hint  # noqa: E402
from pipeline import generate_skeleton  # noqa: E402


def _linear_graph(M, K, N):
    """One linear_s8, so a split produces exactly two comparable calls."""
    return {
        "name": "probe",
        "input": {"tensor": "x"},
        "output": {"tensors": ["y"]},
        "tensors": {
            "x": {"shape": [M, K], "dtype": "i8",
                  "quant": {"scale": 0.02, "zero_point": 0}},
            "y": {"shape": [M, N], "dtype": "i8",
                  "quant": {"scale": 0.05, "zero_point": 0}},
        },
        "ops": [{
            "name": "lin0", "op": "linear_s8",
            "inputs": ["x"], "outputs": ["y"],
            "weight": "lin0.weight_q", "bias": "lin0.bias_q",
            "shape": {"M": M, "K": K, "N": N},
            "quant": {
                "input_offset": 0, "filter_offset": 0, "output_offset": 0,
                "output_multiplier": 1845733646, "output_shift": 7,
                "activation_min": -128, "activation_max": 127,
            },
            "dispatch_id": 0, "hardware_target": "any", "depends_on": [],
        }],
    }


def _emit(graph, M, K, N):
    """Run the skeleton generator over `graph`; return model.c as text."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ir = td / "graph.json"
        ir.write_text(json.dumps(graph))
        np.savez(td / "weights.npz",
                 **{"lin0.weight_q": np.zeros((N, K), dtype=np.int8),
                    "lin0.bias_q": np.zeros((N,), dtype=np.int32)})
        np.savez(td / "io.npz",
                 input=np.zeros((M, K), dtype=np.int8),
                 output=np.zeros((M, N), dtype=np.int8))
        out = td / "generated"
        argv = sys.argv
        sys.argv = ["generate_skeleton",
                    "--ir", str(ir), "--weights", str(td / "weights.npz"),
                    "--io", str(td / "io.npz"), "--out-dir", str(out),
                    "--backend", "scalar", "--platform", "linux"]
        try:
            generate_skeleton.main()
        finally:
            sys.argv = argv
        return (out / "model.c").read_text()


def _linear_calls(src):
    return re.findall(r"parallel_linear_s8\([^;]*\);", src, re.S)


class SplitLinearWeightOffset(unittest.TestCase):

    def test_each_tile_addresses_its_own_weight_rows(self):
        """The regression. Tile t must read weights from row t*tile_N."""
        K, N, n_splits = 32, 64, 2
        g = apply_split_hint(
            _linear_graph(1, K, N),
            [{"op": 0, "n_splits": n_splits}])
        calls = _linear_calls(_emit(g, 1, K, N))
        self.assertEqual(len(calls), n_splits,
                         f"expected {n_splits} tile calls, got {len(calls)}")

        tile_n = N // n_splits
        # Tile 0 starts at row 0. Assert the OFFSET IS ZERO rather than that no
        # offset expression is present -- `(w + 0)` is equally correct and the
        # generator happens to emit it, so a syntactic assertion would fail on
        # working code.
        m0 = re.search(r"weight_q \+ (\d+)", calls[0])
        self.assertEqual(int(m0.group(1)) if m0 else 0, 0,
                         f"tile 0 must read from weight row 0; call was:\n{calls[0]}")
        # Tile 1 must skip tile_n rows of K elements each.
        want_w = f"+ {tile_n * K}"
        want_b = f"+ {tile_n}"
        self.assertIn(want_w, calls[1],
                      f"tile 1 must offset the weight pointer by {tile_n*K} "
                      f"(tile_N * K); without it both tiles compute output "
                      f"rows [0,{tile_n}) and tile 1 writes duplicates.\n"
                      f"call was: {calls[1]}")
        self.assertIn(want_b, calls[1],
                      f"tile 1 must offset the bias pointer by {tile_n}")

    def test_the_broken_version_would_have_failed_this(self):
        """Pins the exact defect: two calls differing ONLY in the output ptr.

        Written as its own test because that is what the emitted C looked like,
        and an assertion phrased any more loosely would have passed on it.
        """
        K, N = 32, 64
        g = apply_split_hint(
            _linear_graph(1, K, N),
            [{"op": 0, "n_splits": 2}])
        calls = _linear_calls(_emit(g, 1, K, N))
        # Strip the output pointer argument from each call and compare the rest.
        def _without_out_ptr(c):
            args = c[c.index("(") + 1:c.rindex(")")].split(",")
            del args[4]          # pool, in, w, b, OUT, M, K, N, ...
            return ",".join(a.strip() for a in args)
        self.assertNotEqual(
            _without_out_ptr(calls[0]), _without_out_ptr(calls[1]),
            "the two tile calls differ only in their output pointer -- they "
            "read the same weight rows and therefore compute the same values")

    def test_four_way_split_offsets_are_tile_parametric(self):
        """n_splits is not hardcoded to 2 anywhere; check a 4-way split."""
        K, N, n = 32, 64, 4
        g = apply_split_hint(
            _linear_graph(1, K, N),
            [{"op": 0, "n_splits": n}])
        calls = _linear_calls(_emit(g, 1, K, N))
        self.assertEqual(len(calls), n)
        tile_n = N // n
        for t in range(1, n):
            self.assertIn(f"+ {t * tile_n * K}", calls[t],
                          f"tile {t} weight offset")


class SplitLinearRejectsUnsupportedShapes(unittest.TestCase):

    def test_M_greater_than_one_is_refused_not_silently_wrong(self):
        """M>1 makes an N-tile a STRIDED column slice, not a contiguous block.

        With M=4, N=64, 2 tiles the contiguous assumption has tile 0 writing
        [0,128) and tile 1 writing [32,160): they overlap, and tile 1 runs past
        the parent buffer. Refusing is the only safe answer until strided tile
        emission exists.
        """
        M, K, N = 4, 32, 64
        g = apply_split_hint(
            _linear_graph(M, K, N),
            [{"op": 0, "n_splits": 2}])
        with self.assertRaises(SystemExit) as cm:
            _emit(g, M, K, N)
        self.assertIn("strided", str(cm.exception).lower())


if __name__ == "__main__":
    unittest.main()
