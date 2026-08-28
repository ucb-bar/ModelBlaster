"""A conv shard must not slice an IHWOC-packed weight with an OIHW formula.

Why this file exists
--------------------
`generate_skeleton` emits `parallel_conv2d_s8`, which shards a convolution
across `modelblaster_pool` helpers by splitting OC and handing each shard

    c->w + (size_t)oc0 * (IC * KH * KW)

That is an OIHW offset, correct for a `[OC, IC, KH, KW]` weight -- which is
what the scalar reference backend gets. The rvv backends do not get OIHW.
`_backend_pack_weight` permutes conv weights to IHWOC `(IC, KH, KW, OC)` at
codegen time, OC innermost, because the curated kernels index
`weight[((ic*KH + kh)*KW + kw)*OC + oc]`.

Under that layout an OC slice is strided, and two things go wrong at once --
the same two `split_conv_tile_weights` documents for the split path:

  * the base offset is wrong by a factor of IC*KH*KW, and
  * the kernel is handed `OC = oc1 - oc0` and strides the packed array by that
    instead of by the full OC.

The failure is invisible from an x86 host: scalar really is OIHW, so a
host-side bit-exactness check passes while every vector backend computes the
wrong answer.

It has never fired, which is why it survived. No run has enabled a pool with
helpers -- only `scripts/profile_firesim.sh` sets MODELBLASTER_POOL_THREADS,
and to 1, which is caller-only. So nothing measured in this tree is affected;
the point is that the sharding rung could not be run without hitting it.

These tests assert on the generated source text, like
`test_split_codegen_offsets.py` beside them and for the same reason: an
IR-level assertion cannot see this class of defect at all.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Repo root, for `pipeline.*`. And src/, because generate_skeleton's conv
# weight-layout query imports `modelblaster.pipeline.reference_kernels` by its
# installed name -- reachable only through the src/modelblaster namespace shim.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pipeline import generate_skeleton  # noqa: E402

_GUARD = "#if 0 /* OC slice is strided"


def _emit(backend, ops=("conv2d_s8", "linear_s8")):
    return generate_skeleton._emit_parallel_wrappers("m", set(ops), backend)


def _conv_section(src):
    """Just the conv2d_s8 wrapper.

    Bounded at BOTH ends: the wrappers are emitted in sorted op order, so
    linear_s8 follows conv2d_s8 and its own `#ifdef MODELBLASTER_USE_POOL`
    would otherwise be read as the conv wrapper's.
    """
    # Start exactly at the conv wrapper, with no backtrack: the file header
    # carries its own `#ifdef MODELBLASTER_USE_POOL` around the pool include,
    # and reading backwards would pick that up as the conv wrapper's.
    for marker in ("/* parallel_conv2d_s8", "/* ---- conv2d_s8",
                   "parallel_conv2d_s8_ctx_t"):
        start = src.find(marker)
        if start != -1:
            break
    tail = src.find("parallel_linear_s8_ctx_t", start)
    return src[start:tail if tail != -1 else len(src)]


class TheCheckedProperty(unittest.TestCase):

    def test_the_repo_still_packs_rvv_conv_weights_ihwoc(self):
        """The premise. If this ever changes, the rest of the file is moot and
        should be revisited rather than silently passing."""
        self.assertEqual(
            generate_skeleton._conv_weight_layout_for_backend("rvv_x60"),
            "ihwoc")

    def test_scalar_is_not_repacked_so_an_oc_slice_is_contiguous(self):
        self.assertIn(
            generate_skeleton._conv_weight_layout_for_backend("scalar"),
            (None, "oihw"))


class ConvShardingOnAPackedBackend(unittest.TestCase):

    def test_the_pool_path_is_compiled_out_for_conv(self):
        src = _emit("rvv_x60")
        self.assertIn(_GUARD, src)
        self.assertNotIn("#ifdef MODELBLASTER_USE_POOL", _conv_section(src),
                         "the conv wrapper must not reach the pool path on a "
                         "backend whose OC slice is strided")

    def test_the_oihw_offset_never_runs_there(self):
        """The specific arithmetic that would read the wrong bytes.

        `per_filter` is the OIHW stride `IC*KH*KW`. It lives in the shard
        worker, which is emitted BEFORE its caller, so it is not enough to
        guard the call: the worker itself must be inside a disabled region.
        """
        conv = _conv_section(_emit("rvv_x60"))
        self.assertIn("per_filter", conv, "sanity: the arithmetic is emitted")
        self.assertLess(conv.index(_GUARD), conv.index("per_filter"),
                        "the guard must precede the offset arithmetic")

    def test_the_public_entry_point_still_exists(self):
        """model.c calls `parallel_conv2d_s8` unconditionally, so disabling
        the shard machinery must not take the function with it."""
        conv = _conv_section(_emit("rvv_x60"))
        self.assertIn("static inline void parallel_conv2d_s8(", conv)
        self.assertIn("kernel_conv2d_s8_m(", conv,
                      "the serial arm must still call the kernel")

    def test_a_backend_without_repacking_keeps_its_shard_path(self):
        src = _emit("scalar")
        self.assertNotIn(_GUARD, src)
        self.assertIn("#ifdef MODELBLASTER_USE_POOL", _conv_section(src))

    def test_linear_is_unaffected_on_every_backend(self):
        """`_backend_pack_weight` only touches 4D tensors, so a linear weight
        stays `[N, K]` row-major and an N slice stays contiguous. Disabling it
        too would cost real parallelism for no correctness reason."""
        for backend in ("scalar", "rvv_x60"):
            src = _emit(backend, ops=("linear_s8",))
            self.assertIn("modelblaster_pool_parallelize_1d("
                          "pool, parallel_linear_s8_fn", src, backend)
            self.assertNotIn(_GUARD, src, backend)


if __name__ == "__main__":
    unittest.main()
