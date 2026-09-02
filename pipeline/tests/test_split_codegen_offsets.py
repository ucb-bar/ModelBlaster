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

# Repo root, for `pipeline.*`. And src/, because generate_skeleton's conv
# weight-layout query imports `modelblaster.pipeline.reference_kernels` by its
# installed name -- reachable only through the src/modelblaster namespace shim.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

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


def _conv_graph(N, IC, IH, IW, OC, KH, KW):
    """One conv2d_s8, so a split produces exactly two comparable calls."""
    OH, OW = IH, IW      # stride 1, pad 1, 3x3 -> same spatial size
    return {
        "name": "probe",
        "input": {"tensor": "x"},
        "output": {"tensors": ["y"]},
        "tensors": {
            "x": {"shape": [N, IC, IH, IW], "dtype": "i8",
                  "quant": {"scale": 0.02, "zero_point": 0}},
            "y": {"shape": [N, OC, OH, OW], "dtype": "i8",
                  "quant": {"scale": 0.05, "zero_point": 0}},
        },
        "ops": [{
            "name": "c0", "op": "conv2d_s8",
            "inputs": ["x"], "outputs": ["y"],
            "weight": "c0.weight_q", "bias": "c0.bias_q",
            "shape": {"N": N, "IC": IC, "IH": IH, "IW": IW, "OC": OC,
                      "OH": OH, "OW": OW, "KH": KH, "KW": KW,
                      "SH": 1, "SW": 1, "PH": 1, "PW": 1},
            "quant": {
                "input_offset": 0, "filter_offset": 0, "output_offset": 0,
                "output_multiplier": 1845733646, "output_shift": 7,
                "activation_min": -128, "activation_max": 127,
            },
            "dispatch_id": 0, "hardware_target": "any", "depends_on": [],
        }],
    }


def _emit_conv(graph, N, IC, IH, IW, OC, KH, KW, backend):
    """Run the skeleton generator; return (model.c, weights.c, weights.npz dict)."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ir = td / "graph.json"
        ir.write_text(json.dumps(graph))
        rng = np.random.default_rng(0)
        w = rng.integers(-127, 127, size=(OC, IC, KH, KW), dtype=np.int8)
        b = rng.integers(-1000, 1000, size=(OC,)).astype(np.int32)
        np.savez(td / "weights.npz", **{"c0.weight_q": w, "c0.bias_q": b})
        np.savez(td / "io.npz",
                 input=np.zeros((N, IC, IH, IW), dtype=np.int8),
                 output=np.zeros((N, OC, IH, IW), dtype=np.int8))
        out = td / "generated"
        argv = sys.argv
        sys.argv = ["generate_skeleton",
                    "--ir", str(ir), "--weights", str(td / "weights.npz"),
                    "--io", str(td / "io.npz"), "--out-dir", str(out),
                    "--backend", backend, "--platform", "linux"]
        try:
            generate_skeleton.main()
        finally:
            sys.argv = argv
        return ((out / "model.c").read_text(),
                (out / "weights.c").read_text(),
                {"c0.weight_q": w, "c0.bias_q": b})


def _c_array(src, name, backend=None):
    """Find the emitted array for a LOGICAL weight key.

    `_weight_name` suffixes every symbol with its backend, unconditionally, so
    that two backends' weights.c cannot define the same non-static symbol with
    differently-packed data -- a silent wrong answer (max_abs_err=51 on dronet),
    not a link error, because only one definition ever got compiled in. These
    tests care which TENSOR was emitted, not how the symbol is spelled, so the
    suffix is matched here rather than written into every assertion.
    """
    pat = rf"\b{name}(?:_{re.escape(backend)})?\[\d+\] = \{{(.*?)\n\}};" \
        if backend else rf"\b{name}\[\d+\] = \{{(.*?)\n\}};"
    m = re.search(pat, src, re.S)
    if not m:
        return None
    return np.array([int(x) for x in re.findall(r"-?\d+", m.group(1))],
                    dtype=np.int64)


class SplitConvUnderPackedWeightLayout(unittest.TestCase):
    """An OC split must respect the backend's conv weight LAYOUT.

    `_backend_pack_weight` permutes conv weights to IHWOC `(IC, KH, KW, OC)`
    for the rvv backends, because their kernels vectorise over OC and index
    `weight[((ic*KH + kh)*KW + kw)*OC + oc]`. Under that layout OC is the
    innermost axis, so an OC slice is STRIDED -- `weight + t*tile_oc*IC*KH*KW`
    names a different element set entirely, and the kernel additionally strides
    by `tile_oc` where the packed parent strides by the full OC. Both errors are
    invisible in the IR, in the build, and in the emitted C read on its own.
    """

    N, IC, IH, IW, OC, KH, KW = 1, 3, 8, 8, 16, 3, 3

    def _split(self, backend, n_splits=2):
        g = apply_split_hint(
            _conv_graph(self.N, self.IC, self.IH, self.IW,
                        self.OC, self.KH, self.KW),
            [{"op": 0, "n_splits": n_splits}])
        return _emit_conv(g, self.N, self.IC, self.IH, self.IW,
                          self.OC, self.KH, self.KW, backend)

    def test_rvv_tile_weights_are_the_packed_slice_not_a_slice_of_the_packed(self):
        """The regression. Each tile array must equal pack(w[tile slice])."""
        n = 2
        _, weights_c, npz = self._split("rvv_x60", n)
        tile_oc = self.OC // n
        w = npz["c0.weight_q"]
        packed_parent = np.ascontiguousarray(
            np.transpose(w, (1, 2, 3, 0))).ravel().astype(np.int64)
        for t in range(n):
            got = _c_array(weights_c, f"probe_c0_weight_q_tile_{t}", "rvv_x60")
            self.assertIsNotNone(
                got, f"no per-tile weight array emitted for tile {t}; the "
                     f"rvv backends pack conv weights IHWOC, where a pointer "
                     f"offset cannot express an OC slice")
            want = np.ascontiguousarray(np.transpose(
                w[t * tile_oc:(t + 1) * tile_oc], (1, 2, 3, 0))
            ).ravel().astype(np.int64)
            np.testing.assert_array_equal(
                got, want,
                f"tile {t} weight array is not pack(oihw_slice)")
            # And pin what the pre-fix codegen would have produced: a window
            # into the packed PARENT at t*tile_oc*IC*KH*KW. Asserting it is
            # different is what makes the test above mean something.
            off = t * tile_oc * self.IC * self.KH * self.KW
            old = packed_parent[off:off + want.size]
            self.assertFalse(
                np.array_equal(old, want),
                f"tile {t}: the old pointer-offset view coincides with the "
                f"correct slice at this shape, so this test cannot detect the "
                f"bug -- pick a shape where it does")

    def test_rvv_tile_call_reads_its_own_array_from_element_zero(self):
        model_c, _, _ = self._split("rvv_x60", 2)
        calls = re.findall(r"parallel_conv2d_s8\([^;]*\);", model_c, re.S)
        self.assertEqual(len(calls), 2)
        for t, call in enumerate(calls):
            self.assertIn(f"probe_c0_weight_q_tile_{t}", call)
            self.assertIn(f"probe_c0_bias_q_tile_{t}", call)
            self.assertNotIn("weight_q + ", call,
                             "a per-tile array must be read from element 0, "
                             "not offset again")

    def test_rvv_parent_array_is_not_also_emitted(self):
        """A conv weight has one consumer, so once tiles own it the parent is
        dead. Emitting both would double that conv's weight storage."""
        _, weights_c, _ = self._split("rvv_x60", 2)
        self.assertIsNone(_c_array(weights_c, "probe_c0_weight_q", "rvv_x60"),
                          "parent weight array emitted alongside the tiles")

    def test_scalar_backend_keeps_the_pointer_offset_and_no_tile_arrays(self):
        """OIHW makes an OC slice contiguous, so offsetting is correct AND
        free. The fix must not start duplicating arrays on that path."""
        model_c, weights_c, _ = self._split("scalar", 2)
        tile_oc = self.OC // 2
        per_filter = self.IC * self.KH * self.KW
        calls = re.findall(r"parallel_conv2d_s8\([^;]*\);", model_c, re.S)
        self.assertEqual(len(calls), 2)
        self.assertIn(f"+ {tile_oc * per_filter}", calls[1])
        self.assertIsNone(_c_array(weights_c, "probe_c0_weight_q_tile_0", "scalar"),
                          "scalar backend should not need per-tile arrays")
        self.assertIsNotNone(_c_array(weights_c, "probe_c0_weight_q", "scalar"),
                            "scalar backend must still emit the parent array")

    def test_linear_split_is_unaffected_by_conv_packing(self):
        """Linear weights are 2D; _backend_pack_weight only touches 4D
        tensors, so `[N, K]` stays row-major and the N-tile offset stays
        valid even on rvv."""
        K, N, n = 32, 64, 2
        g = apply_split_hint(_linear_graph(1, K, N), [{"op": 0, "n_splits": n}])
        calls = _linear_calls(_emit(g, 1, K, N))
        self.assertEqual(len(calls), n)
        self.assertIn(f"+ {(N // n) * K}", calls[1])


if __name__ == "__main__":
    unittest.main()


def _fused_conv_graph(N, IC, IH, IW, OC, KH, KW,
                      kind="conv2d_batchnorm2d_silu_s8"):
    """One fused conv->BN[->SiLU], written the way `extract_graph` writes one:
    no `shape` on the op itself, geometry and weights under `sub_ops`."""
    OH, OW = IH, IW      # stride 1, pad 1, 3x3 -> same spatial size
    conv = {
        "name": "c0.conv", "op": "conv2d_s8",
        "inputs": ["x"], "outputs": ["c0_conv"],
        "weight": "c0.weight_q", "bias": "c0.bias_q",
        "shape": {"N": N, "IC": IC, "IH": IH, "IW": IW, "OC": OC,
                  "OH": OH, "OW": OW, "KH": KH, "KW": KW,
                  "SH": 1, "SW": 1, "PH": 1, "PW": 1},
        "quant": {"input_offset": 0, "filter_offset": 0, "output_offset": 0,
                  "output_multiplier": 1845733646, "output_shift": 7,
                  "activation_min": -128, "activation_max": 127},
    }
    bn = {
        "name": "c0.bn", "op": "batchnorm2d_s8",
        "inputs": ["c0_conv"], "outputs": ["c0_bn"],
        "weight": "c0.bn.scale", "bias": "c0.bn.bias_fused",
        "shape": {"N": N, "C": OC, "H": OH, "W": OW},
        "quant": {"scale_in": 0.04, "scale_out": 1.07,
                  "activation_min": -128, "activation_max": 127},
    }
    subs = [conv, bn]
    if kind == "conv2d_batchnorm2d_silu_s8":
        subs.append({
            "name": "c0.act", "op": "silu_s8",
            "inputs": ["c0_bn"], "outputs": ["y"],
            "shape": {"n": N * OC * OH * OW},
            "quant": {"scale_in": 1.07, "scale_out": 1.07,
                      "activation_min": -128, "activation_max": 127},
        })
    else:
        bn["outputs"] = ["y"]
    return {
        "name": "probe",
        "input": {"tensor": "x"},
        "output": {"tensors": ["y"]},
        "tensors": {
            "x": {"shape": [N, IC, IH, IW], "dtype": "i8",
                  "quant": {"scale": 0.02, "zero_point": 0}},
            "y": {"shape": [N, OC, OH, OW], "dtype": "i8",
                  "quant": {"scale": 0.05, "zero_point": 0}},
        },
        "ops": [{
            "name": "c0", "op": kind,
            "inputs": ["x"], "outputs": ["y"], "sub_ops": subs,
            "dispatch_id": 0, "hardware_target": "any", "depends_on": [],
        }],
    }


def _emit_fused_conv(graph, N, IC, IH, IW, OC, KH, KW, backend):
    """As `_emit_conv`, plus the 1-D per-output-channel epilogue arrays."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ir = td / "graph.json"
        ir.write_text(json.dumps(graph))
        rng = np.random.default_rng(0)
        w = rng.integers(-127, 127, size=(OC, IC, KH, KW), dtype=np.int8)
        b = rng.integers(-1000, 1000, size=(OC,)).astype(np.int32)
        bn_s = rng.random(OC).astype(np.float32)
        bn_b = rng.random(OC).astype(np.float32)
        arrays = {"c0.weight_q": w, "c0.bias_q": b,
                  "c0.bn.scale": bn_s, "c0.bn.bias_fused": bn_b}
        np.savez(td / "weights.npz", **arrays)
        np.savez(td / "io.npz",
                 input=np.zeros((N, IC, IH, IW), dtype=np.int8),
                 output=np.zeros((N, OC, IH, IW), dtype=np.int8))
        out = td / "generated"
        argv = sys.argv
        sys.argv = ["generate_skeleton",
                    "--ir", str(ir), "--weights", str(td / "weights.npz"),
                    "--io", str(td / "io.npz"), "--out-dir", str(out),
                    "--backend", backend, "--platform", "linux"]
        try:
            generate_skeleton.main()
        finally:
            sys.argv = argv
        return ((out / "model.c").read_text(),
                (out / "weights.c").read_text(), arrays)


class SplitFusedConvOperands(unittest.TestCase):
    """A fused conv tile must retarget FOUR operands, not two.

    The plain conv path slices a weight and a bias. A fused conv adds
    `bn_scale` and `bn_bias`, which are 1-D per-output-channel floats.
    `_backend_pack_weight` only permutes 4-D tensors, so those two stay
    contiguous under every backend layout and `+ oc0` is correct for them on
    rvv as well as scalar -- while the 4-D weight still needs its own
    re-packed array on rvv. Getting one of the four right and the others
    wrong is bit-exact on tile 0 and silently wrong on every other tile.

    This is the same arithmetic `parallel_cbs_shard_fn` performs at runtime,
    which is what makes it trustworthy: that path's OC slicing was measured
    bit-exact on the board.
    """

    N, IC, IH, IW, OC, KH, KW = 1, 3, 8, 8, 16, 3, 3

    def _split(self, backend, n_splits=2, kind="conv2d_batchnorm2d_silu_s8"):
        g = apply_split_hint(
            _fused_conv_graph(self.N, self.IC, self.IH, self.IW,
                              self.OC, self.KH, self.KW, kind),
            [{"op": 0, "n_splits": n_splits}])
        return _emit_fused_conv(g, self.N, self.IC, self.IH, self.IW,
                                self.OC, self.KH, self.KW, backend)

    def _calls(self, model_c):
        return re.findall(r"kernel_conv2d_batchnorm2d_\w*s8_probe\([^;]*\);",
                          model_c, re.S)

    def test_scalar_offsets_every_operand_including_the_epilogue(self):
        model_c, weights_c, _ = self._split("scalar", 2)
        calls = self._calls(model_c)
        self.assertEqual(len(calls), 2)
        tile_oc = self.OC // 2
        per_filter = self.IC * self.KH * self.KW
        # `_scalar` is the backend suffix `_weight_name` puts on every
        # emitted symbol, so two backends' weights.c cannot define the same
        # name with differently-packed data. Asserted here rather than
        # stripped, because the offset is only correct on the array the
        # suffix names.
        self.assertIn(f"weight_q_scalar + {tile_oc * per_filter}", calls[1])
        self.assertIn(f"bias_q_scalar + {tile_oc}", calls[1])
        self.assertIn(f"bn_scale_scalar + {tile_oc}", calls[1])
        self.assertIn(f"bn_bias_fused_scalar + {tile_oc}", calls[1])
        # Tile 0 is the trap: every offset is 0, so a codegen that forgot the
        # epilogue entirely still produces a correct tile 0.
        self.assertIn("bn_scale_scalar + 0", calls[0])
        self.assertIn("bn_bias_fused_scalar + 0", calls[0])

    def test_each_tile_is_called_with_the_narrowed_oc(self):
        model_c, _, _ = self._split("scalar", 4)
        calls = self._calls(model_c)
        self.assertEqual(len(calls), 4)
        for call in calls:
            args = call.split("(", 1)[1].split(",")
            # in, w, b, bn_scale, bn_bias, out, N, IC, IH, IW, OC, ...
            self.assertEqual(args[10].strip(), str(self.OC // 4))

    def test_each_tile_writes_its_own_slice_of_the_output_buffer(self):
        """Without the offset alias every tile writes at element 0 and they
        trample each other -- the failure reads as `max_abs_err > 0` from a
        rewrite the granularity gate correctly passed."""
        model_c, _, _ = self._split("scalar", 2)
        calls = self._calls(model_c)
        plane = self.IH * self.IW
        # `y` is the model's output, so its buffer is `s->output`; a tile of
        # an interior tensor would read `buf_probe_<name> + ...` instead.
        self.assertIn(f"s->output + {self.OC // 2 * plane}", calls[1])
        self.assertNotIn("+ ", calls[0].split(",")[5],
                         "tile 0 must address the output at element 0")

    def test_rvv_weight_gets_its_own_packed_array_but_the_epilogue_does_not(self):
        """The whole point of the split: the 4-D weight cannot be offset under
        IHWOC, the 1-D epilogue can and must be."""
        n = 2
        model_c, weights_c, arrays = self._split("rvv_x60", n)
        tile_oc = self.OC // n
        w = arrays["c0.weight_q"]
        for t in range(n):
            got = _c_array(weights_c, f"probe_c0_weight_q_tile_{t}", "rvv_x60")
            self.assertIsNotNone(got, f"no per-tile weight array for tile {t}")
            want = np.ascontiguousarray(np.transpose(
                w[t * tile_oc:(t + 1) * tile_oc], (1, 2, 3, 0))
            ).ravel().astype(np.int64)
            np.testing.assert_array_equal(got, want)
        self.assertIsNone(_c_array(weights_c, "probe_c0_weight_q", "rvv_x60"),
                          "parent weight emitted alongside the tiles")
        calls = self._calls(model_c)
        for t, call in enumerate(calls):
            self.assertIn(f"probe_c0_weight_q_tile_{t}", call)
            self.assertIn(f"bn_scale_rvv_x60 + {t * tile_oc}", call,
                          "the 1-D epilogue arrays stay whole and are offset")
        # bn arrays are NOT split into per-tile copies.
        self.assertIsNone(_c_array(weights_c, "probe_c0_bn_scale_tile_0", "rvv_x60"))

    def test_pair_fused_conv_takes_the_same_path(self):
        model_c, _, _ = self._split("scalar", 2, kind="conv2d_batchnorm2d_s8")
        calls = self._calls(model_c)
        self.assertEqual(len(calls), 2)
        tile_oc = self.OC // 2
        self.assertIn(f"bn_scale_scalar + {tile_oc}", calls[1])
        self.assertIn(f"bn_bias_fused_scalar + {tile_oc}", calls[1])


# ── Pointwise (E) and pool-channel (C) tiles ───────────────────────────────
#
# Same reason as everything above this line: the IR can be perfect and the
# emitted C still wrong. Two specific defects are in scope here.
#
#   * A pointwise tile gets its output pointer from its alias and its `n` from
#     the splitter, so a tile whose INPUT pointer was left alone still builds,
#     still runs, and computes tile 0's answer into every tile's slot. The E
#     axis moves all 47 pointwise kinds at once through `_shim_override`, so
#     one missed redirect would be 47 wrong kernels, not one.
#
#   * A pool tile's input and output offsets are DIFFERENT numbers -- `c0*IH*IW`
#     and `c0*OH*OW` -- and are equal only when the pool does not subsample.
#     Every unit test that used a stride-1 pool would pass on code that used one
#     for both. DroNet's is 56x56 -> 27x27, so the test below subsamples.

def _emit_bare(graph, weights=None, io_in=None, io_out=None, backend="scalar"):
    """Emit model.c for a graph with no weight tensors of its own."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ir = td / "graph.json"
        ir.write_text(json.dumps(graph))
        np.savez(td / "weights.npz", **(weights or {}))
        np.savez(td / "io.npz",
                 input=np.zeros(io_in, dtype=np.int8),
                 output=np.zeros(io_out, dtype=np.int8))
        out = td / "generated"
        argv = sys.argv
        sys.argv = ["generate_skeleton",
                    "--ir", str(ir), "--weights", str(td / "weights.npz"),
                    "--io", str(td / "io.npz"), "--out-dir", str(out),
                    "--backend", backend, "--platform", "linux"]
        try:
            generate_skeleton.main()
        finally:
            sys.argv = argv
        return (out / "model.c").read_text()


def _calls(src, kernel):
    return re.findall(rf"{kernel}[a-z0-9_]*\([^;]*\);", src, re.S)


def _pointwise_graph(kind, n, shape, n_inputs=1):
    ins = ["x"] if n_inputs == 1 else ["x", "x2"]
    tensors = {"x": {"shape": list(shape), "dtype": "i8",
                     "quant": {"scale": 0.02, "zero_point": 0}},
               "y": {"shape": list(shape), "dtype": "i8",
                     "quant": {"scale": 0.02, "zero_point": 0}}}
    if n_inputs == 2:
        tensors["x2"] = dict(tensors["x"])
    return {
        "name": "probe",
        "input": {"tensor": "x"},
        "output": {"tensors": ["y"]},
        "tensors": tensors,
        "ops": [{
            "name": "pw", "op": kind, "inputs": ins, "outputs": ["y"],
            "shape": {"n": n},
            "quant": {"scale_in": 0.02, "scale_out": 0.02,
                      "scale_a": 0.02, "scale_b": 0.02,
                      "activation_min": -128, "activation_max": 127},
            "dispatch_id": 0, "hardware_target": "any", "depends_on": [],
        }],
    }


def _pool_graph(C, IH, IW, KH, SH):
    OH = (IH - KH) // SH + 1
    return {
        "name": "probe",
        "input": {"tensor": "x"},
        "output": {"tensors": ["y"]},
        "tensors": {
            "x": {"shape": [1, C, IH, IW], "dtype": "i8",
                  "quant": {"scale": 0.02, "zero_point": 0}},
            "y": {"shape": [1, C, OH, OH], "dtype": "i8",
                  "quant": {"scale": 0.02, "zero_point": 0}},
        },
        "ops": [{
            "name": "mp", "op": "maxpool2d_s8",
            "inputs": ["x"], "outputs": ["y"],
            "shape": {"N": 1, "C": C, "IH": IH, "IW": IW, "OH": OH, "OW": OH,
                      "KH": KH, "KW": KH, "SH": SH, "SW": SH,
                      "PH": 0, "PW": 0, "DH": 1, "DW": 1},
            "dispatch_id": 0, "hardware_target": "any", "depends_on": [],
        }],
    }


class SplitPointwiseElementOffsets(unittest.TestCase):

    def test_unary_tile_offsets_both_pointers(self):
        n, shape = 1024, [1, 16, 8, 8]
        g = apply_split_hint(_pointwise_graph("silu_s8", n, shape),
                             [{"op": 0, "n_splits": 2}])
        calls = _calls(_emit_bare(g, io_in=shape, io_out=shape), "kernel_silu_s8")
        self.assertEqual(len(calls), 2, calls)
        self.assertNotIn("+ 512", calls[0], "tile 0 starts at element 0")
        # Input AND output. Only the output is wired by the alias machinery;
        # the input is the one this axis had to add, and without it both tiles
        # read elements [0, 512) and tile 1 writes tile 0's answer.
        self.assertEqual(calls[1].count("+ 512"), 2,
                         f"tile 1 must offset BOTH its input and its output by "
                         f"512 elements; call was:\n{calls[1]}")
        self.assertIn(", 512,", calls[1], "tile 1's n must be the tile's, not "
                                          "the parent's")

    def test_binary_tile_offsets_all_three_pointers(self):
        """`add_s8` has two inputs, and a redirect that missed one would read
        the right `a` against the wrong `b`."""
        n, shape = 1024, [1, 16, 8, 8]
        g = apply_split_hint(_pointwise_graph("add_s8", n, shape, n_inputs=2),
                             [{"op": 0, "n_splits": 2}])
        calls = _calls(_emit_bare(g, io_in=shape, io_out=shape), "kernel_add_s8")
        self.assertEqual(len(calls), 2, calls)
        self.assertEqual(calls[1].count("+ 512"), 3,
                         f"tile 1 must offset a, b and out; call was:\n{calls[1]}")

    def test_the_broken_version_would_have_failed_this(self):
        """Two tile calls that differ only in the OUTPUT pointer.

        That is exactly what an E split emits if the input redirect is dropped,
        and it is the same defect the linear tests above pin -- restated here
        because the mechanism that prevents it is completely different
        (`_shim_override` redirection, not a per-emitter offset).
        """
        n, shape = 1024, [1, 16, 8, 8]
        g = apply_split_hint(_pointwise_graph("silu_s8", n, shape),
                             [{"op": 0, "n_splits": 2}])
        calls = _calls(_emit_bare(g, io_in=shape, io_out=shape), "kernel_silu_s8")

        def _without_out_ptr(c):
            args = c[c.index("(") + 1:c.rindex(")")].split(",")
            del args[1]                      # in, OUT, n, ...
            return ",".join(a.strip() for a in args)
        self.assertNotEqual(_without_out_ptr(calls[0]), _without_out_ptr(calls[1]))

    def test_uneven_partition_emits_the_recorded_offsets(self):
        n, shape = 1000, [1000]
        g = apply_split_hint(_pointwise_graph("silu_s8", n, shape),
                             [{"op": 0, "n_splits": 3,
                               "tile_sizes": [200, 300, 500]}])
        calls = _calls(_emit_bare(g, io_in=shape, io_out=shape), "kernel_silu_s8")
        self.assertEqual(len(calls), 3)
        self.assertIn("+ 200", calls[1])
        self.assertIn("+ 500", calls[2])
        self.assertIn(", 500,", calls[2])


class SplitPoolChannelOffsets(unittest.TestCase):

    def test_input_and_output_offsets_are_different_numbers(self):
        """The trap. A subsampling pool's planes are not the same size."""
        C, IH, KH, SH = 32, 56, 3, 2
        OH = (IH - KH) // SH + 1                      # 27
        g = apply_split_hint(_pool_graph(C, IH, IH, KH, SH),
                             [{"op": 0, "n_splits": 2}])
        src = _emit_bare(g, io_in=[1, C, IH, IH], io_out=[1, C, OH, OH])
        calls = _calls(src, "kernel_maxpool2d_s8")
        self.assertEqual(len(calls), 2, calls)
        c0 = C // 2
        self.assertIn(f"+ {c0 * IH * IH}", calls[1],      # 50176
                      f"tile 1's INPUT must skip {c0} planes of {IH}x{IH}; "
                      f"call was:\n{calls[1]}")
        self.assertIn(f"+ {c0 * OH * OH}", calls[1],      # 11664
                      f"tile 1's OUTPUT must skip {c0} planes of {OH}x{OH}")
        self.assertNotEqual(c0 * IH * IH, c0 * OH * OH,
                            "this fixture must subsample, or it cannot "
                            "distinguish the two offsets at all")
        # And the tile's own channel count, not the parent's.
        self.assertIn(f", 1, {c0}, {IH}, {IH},", calls[1])

    def test_tile_zero_starts_at_the_base_pointers(self):
        C, IH, KH, SH = 32, 56, 3, 2
        OH = (IH - KH) // SH + 1
        g = apply_split_hint(_pool_graph(C, IH, IH, KH, SH),
                             [{"op": 0, "n_splits": 2}])
        src = _emit_bare(g, io_in=[1, C, IH, IH], io_out=[1, C, OH, OH])
        calls = _calls(src, "kernel_maxpool2d_s8")
        self.assertNotIn("+ ", calls[0], f"tile 0 needs no offset:\n{calls[0]}")


class SplitRegisteredWithoutAnEmitterIsFatal(unittest.TestCase):
    """The loud failure, restated for the two new axes.

    Three of the four registration sites a split kind needs fail by producing a
    plausible wrong answer rather than an error. This is the one that does not:
    a tile whose (axis, kind) pair has no codegen path stops the build. Without
    it, adding a kind to `_POINTWISE_KINDS` and forgetting
    `_E_SLICEABLE_ELTWISE_OPS` would ship tiles that all write at offset 0.
    """

    def test_unemittable_pointwise_kind_stops_the_build(self):
        n, shape = 1024, [1, 16, 8, 8]
        g = apply_split_hint(_pointwise_graph("silu_s8", n, shape),
                             [{"op": 0, "n_splits": 2}])
        saved = set(generate_skeleton._E_SLICEABLE_ELTWISE_OPS)
        generate_skeleton._E_SLICEABLE_ELTWISE_OPS.discard("silu_s8")
        try:
            with self.assertRaises(SystemExit) as cm:
                _emit_bare(g, io_in=shape, io_out=shape)
            self.assertIn("no codegen path", str(cm.exception))
        finally:
            generate_skeleton._E_SLICEABLE_ELTWISE_OPS.clear()
            generate_skeleton._E_SLICEABLE_ELTWISE_OPS.update(saved)
