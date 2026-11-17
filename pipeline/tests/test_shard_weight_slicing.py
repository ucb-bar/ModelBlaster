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

    def test_the_pool_path_shards_a_repacked_copy_never_the_packed_weight(self):
        """On an IHWOC backend the OC-split pool path is enabled (hosted POSIX
        builds) but hands the shard worker a repacked copy whose OC chunks are
        contiguous, never the packed weight itself."""
        conv = _conv_section(_emit("rvv_x60"))
        self.assertNotIn(_GUARD, conv)
        self.assertIn("#if defined(MODELBLASTER_USE_POOL) && defined(MB_PLATFORM_POSIX)", conv)
        self.assertIn(".w = wpk,", conv)
        self.assertNotIn(".w = w,", conv)
        self.assertLess(conv.index("mb_ihwoc_shards_m("), conv.index(".w = wpk,"))

    def test_the_repack_puts_each_chunk_where_the_oihw_offset_points(self):
        """The shard worker still computes `w + oc0 * per_filter`; the repack
        must lay chunk [oc0, oc1) out exactly there, OC-innermost with stride
        oc1 - oc0."""
        conv = _conv_section(_emit("rvv_x60"))
        self.assertIn("int8_t *dst = p + per_filter * (size_t)oc0;", conv)
        self.assertIn("memcpy(dst + f * (size_t)ocn, w + f * (size_t)OC + oc0, (size_t)ocn);", conv)

    def test_the_public_entry_point_still_exists(self):
        """model.c calls `parallel_conv2d_s8` unconditionally, so disabling
        the shard machinery must not take the function with it."""
        conv = _conv_section(_emit("rvv_x60"))
        self.assertIn("static inline void parallel_conv2d_s8(", conv)
        self.assertIn("kernel_conv2d_s8_m(", conv,
                      "the serial arm must still call the kernel")

    def test_the_repacked_shards_are_bit_exact_with_the_serial_call(self):
        """Compile the emitted rvv_x60 wrapper with a reference IHWOC conv
        kernel and run it on pools of 1..4 threads against the serial call."""
        import os
        import shutil
        import subprocess
        import tempfile
        cc = shutil.which("cc") or shutil.which("gcc")
        if cc is None or sys.platform != "linux":
            self.skipTest("needs a C compiler on Linux")
        root = Path(__file__).resolve().parents[2]
        kernel = r"""
#include <stdint.h>
#include <stddef.h>
/* reference conv over IHWOC weights [IC][KH][KW][OC], NCHW activations */
void kernel_conv2d_s8_m(const int8_t *in, const int8_t *w, const int32_t *b, int8_t *out,
    int N, int IC, int IH, int IW, int OC, int KH, int KW, int SH, int SW, int PH, int PW,
    int in_off, int f_off, int out_off, int mult, int shift, int amin, int amax) {
    (void)N; (void)mult;
    int OH = (IH + 2*PH - KH)/SH + 1, OW = (IW + 2*PW - KW)/SW + 1;
    for (int oc = 0; oc < OC; oc++) for (int oh = 0; oh < OH; oh++) for (int ow = 0; ow < OW; ow++) {
        int32_t acc = b ? b[oc] : 0;
        for (int ic = 0; ic < IC; ic++) for (int kh = 0; kh < KH; kh++) for (int kw = 0; kw < KW; kw++) {
            int ih = oh*SH - PH + kh, iw = ow*SW - PW + kw;
            if (ih < 0 || ih >= IH || iw < 0 || iw >= IW) continue;
            acc += (in[(ic*IH + ih)*IW + iw] + in_off) * (w[((ic*KH + kh)*KW + kw)*OC + oc] + f_off);
        }
        acc = (acc >> shift) + out_off;
        out[(oc*OH + oh)*OW + ow] = (int8_t)(acc < amin ? amin : acc > amax ? amax : acc);
    }
}
"""
        main = r"""
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include "modelblaster_pool.h"
void kernel_conv2d_s8_m(const int8_t *, const int8_t *, const int32_t *, int8_t *,
    int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int);
@WRAPPER@
int main(void) {
    enum { IC = 5, IH = 9, IW = 7, OC = 13, KH = 3, KW = 3 };
    static int8_t in[IC*IH*IW], w[IC*KH*KW*OC], ref[OC*IH*IW], got[OC*IH*IW];
    static int32_t b[OC];
    for (int i = 0; i < (int)sizeof in; i++) in[i] = (int8_t)(i * 37 % 251 - 125);
    for (int i = 0; i < (int)sizeof w; i++) w[i] = (int8_t)(i * 53 % 241 - 120);
    for (int i = 0; i < OC; i++) b[i] = i * 17 - 90;
    parallel_conv2d_s8(NULL, in, w, b, ref, 1, IC, IH, IW, OC, KH, KW, 1, 1, 1, 1, 3, -2, 1, 0, 6, -128, 127);
    for (int t = 1; t <= 4; t++) {
        modelblaster_pool_t p = modelblaster_pool_create(t);
        memset(got, 0, sizeof got);
        parallel_conv2d_s8(p, in, w, b, got, 1, IC, IH, IW, OC, KH, KW, 1, 1, 1, 1, 3, -2, 1, 0, 6, -128, 127);
        if (memcmp(ref, got, sizeof ref) != 0) { printf("MISMATCH T=%d\n", t); return 1; }
        if (p) modelblaster_pool_destroy(p);
    }
    printf("OK\n");
    return 0;
}
""".replace("@WRAPPER@", generate_skeleton._emit_parallel_wrappers("m", {"conv2d_s8"}, "rvv_x60"))
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "k.c").write_text(kernel)
            (d / "m.c").write_text(main)
            exe = d / "t"
            r = subprocess.run([cc, "-O1", "-pthread", "-D_GNU_SOURCE", "-DMB_PLATFORM_POSIX",
                                "-DMODELBLASTER_USE_POOL=1", f"-I{root / 'runtime'}",
                                f"-I{root / 'runtime' / 'modelblaster_pool'}", "-o", str(exe),
                                str(d / "m.c"), str(d / "k.c"),
                                str(root / "runtime" / "modelblaster_pool" / "modelblaster_pool.c"),
                                str(root / "runtime" / "mb_placement.c")],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
            self.assertEqual(r.stdout.strip(), "OK", r.stdout + r.stderr)

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



class TheSplitPointIsDeclaredNotInferred(unittest.TestCase):
    """The marker replaces a match on the entry point's C signature.

    The first version of `_serial_only_wrapper` partitioned the template on
    `static inline void parallel_<op>(`. That works until someone reformats a
    signature across lines, at which point the partition silently fails and the
    function falls back to guarding only the call -- leaving the shard worker,
    with its wrong-for-this-backend offset arithmetic, compiled in. Silent
    degradation of a safety mechanism is the worst failure shape available, so
    a missing marker is now a hard error.
    """

    def test_every_offset_wrapper_template_declares_the_marker(self):
        for op, tmpl in generate_skeleton._PARALLEL_WRAPPER_TEMPLATES.items():
            self.assertIn(generate_skeleton._SHARD_MACHINERY_END, tmpl,
                          f"{op} template lost its split marker")

    def test_the_marker_precedes_the_entry_point(self):
        for op, tmpl in generate_skeleton._PARALLEL_WRAPPER_TEMPLATES.items():
            self.assertLess(
                tmpl.index(generate_skeleton._SHARD_MACHINERY_END),
                tmpl.index(f"static inline void parallel_{op}("),
                f"{op}: the marker must come before the entry point, or the "
                f"entry point would be disabled with the machinery")

    def test_a_template_without_the_marker_is_refused_loudly(self):
        saved = dict(generate_skeleton._PARALLEL_WRAPPER_TEMPLATES)
        try:
            generate_skeleton._PARALLEL_WRAPPER_TEMPLATES["conv2d_s8"] = (
                "static inline void parallel_conv2d_s8(void) {{}}")
            with self.assertRaises(SystemExit) as cm:
                generate_skeleton._serial_only_wrapper("conv2d_s8", "m")
            self.assertIn("marker", str(cm.exception))
        finally:
            generate_skeleton._PARALLEL_WRAPPER_TEMPLATES.clear()
            generate_skeleton._PARALLEL_WRAPPER_TEMPLATES.update(saved)
