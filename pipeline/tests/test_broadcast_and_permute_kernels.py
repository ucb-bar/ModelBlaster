"""Compile the reordering / broadcasting int8 reference kernels and check them
against numpy.

These four kernels exist because the export extractor used to lower each of
their cases as something cheaper and wrong -- a permute as a free alias, a
batched matmul as a single one, a channel- or tile-broadcast add as a flat
elementwise add. Each of those produced a plausible-looking wrong answer with
nothing to catch it, so the kernels that replaced them are pinned here rather
than only exercised end-to-end.

The reference_impl strings are compiled with the host cc and driven through
ctypes, so this test needs a working compiler but no board and no model.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from pipeline import reference_kernels as rk  # noqa: E402

_I8P = ctypes.POINTER(ctypes.c_int8)


def _p(a: np.ndarray):
    return a.ctypes.data_as(_I8P)


def _quantize(v: np.ndarray, amin: int = -128, amax: int = 127) -> np.ndarray:
    """The kernels' round-half-away-from-zero, then clamp."""
    r = np.where(v >= 0, np.floor(v + 0.5), np.ceil(v - 0.5))
    return np.clip(r, amin, amax).astype(np.int8)


class ReferenceKernelTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls._libs: dict[str, ctypes.CDLL] = {}
        for spec in (rk.PERMUTE4_S8, rk.MATMUL_B_S8,
                     rk.ADD_TILE_S8, rk.ADD_C1_S8):
            src = os.path.join(cls._tmp.name, spec.op + ".c")
            so = os.path.join(cls._tmp.name, spec.op + ".so")
            with open(src, "w") as fh:
                fh.write(spec.reference_impl)
            try:
                subprocess.run(["cc", "-O2", "-fPIC", "-shared", "-o", so,
                                src, "-lm"], check=True,
                               capture_output=True)
            except (OSError, subprocess.CalledProcessError) as e:
                raise unittest.SkipTest(f"cannot build {spec.op}: {e}") from e
            cls._libs[spec.op] = ctypes.CDLL(so)
        cls._rng = np.random.default_rng(0)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_permute4_reorders_like_numpy_transpose(self):
        f = self._libs["permute4_s8"].kernel_permute4_s8
        f.argtypes = [_I8P] * 2 + [ctypes.c_int] * 8 + \
                     [ctypes.c_float] * 2 + [ctypes.c_int] * 2
        cases = [
            ((1, 690, 6, 64), (0, 2, 1, 3)),   # ViT head split
            ((2, 512, 16, 16), (0, 2, 3, 1)),  # stem NCHW -> NHWC
            ((1, 6, 690, 64), (0, 1, 3, 2)),   # K transpose
            ((3, 4, 5, 7), (2, 0, 3, 1)),      # no axis left in place
        ]
        for dims, perm in cases:
            with self.subTest(dims=dims, perm=perm):
                a = self._rng.integers(-128, 128, size=dims, dtype=np.int8)
                out = np.zeros(int(np.prod(dims)), dtype=np.int8)
                f(_p(a), _p(out), *dims, *perm, 0.02, 0.02, -128, 127)
                np.testing.assert_array_equal(out, np.transpose(a, perm).ravel())

    def test_permute4_requantizes_when_the_scales_differ(self):
        f = self._libs["permute4_s8"].kernel_permute4_s8
        f.argtypes = [_I8P] * 2 + [ctypes.c_int] * 8 + \
                     [ctypes.c_float] * 2 + [ctypes.c_int] * 2
        a = self._rng.integers(-128, 128, size=(2, 3, 4, 5), dtype=np.int8)
        out = np.zeros(a.size, dtype=np.int8)
        f(_p(a), _p(out), 2, 3, 4, 5, 0, 2, 1, 3, 0.05, 0.02, -128, 127)
        want = _quantize(np.transpose(a, (0, 2, 1, 3)).astype(np.float32)
                         * np.float32(0.05 / 0.02)).ravel()
        np.testing.assert_array_equal(out, want)

    def test_matmul_b_matches_numpy_per_batch(self):
        f = self._libs["matmul_b_s8"].kernel_matmul_b_s8
        f.argtypes = [_I8P] * 3 + [ctypes.c_int] * 4 + \
                     [ctypes.c_float] * 3 + [ctypes.c_int, ctypes.c_float] + \
                     [ctypes.c_int] * 2
        sa, sb, so, sdiv = 0.02, 0.03, 0.5, 2.0
        for B, M, K, N, tb in [(6, 20, 8, 20, 0), (6, 20, 8, 20, 1),
                               (1, 7, 5, 3, 0), (3, 4, 4, 4, 1)]:
            with self.subTest(B=B, M=M, K=K, N=N, transpose_b=tb):
                a = self._rng.integers(-8, 8, size=(B, M, K), dtype=np.int8)
                b = self._rng.integers(-8, 8,
                                       size=(B, N, K) if tb else (B, K, N),
                                       dtype=np.int8)
                out = np.zeros(B * M * N, dtype=np.int8)
                f(_p(a), _p(b), _p(out), B, M, K, N,
                  sa, sb, so, tb, sdiv, -128, 127)
                bm = np.transpose(b, (0, 2, 1)) if tb else b
                acc = np.matmul(a.astype(np.int32), bm.astype(np.int32))
                want = _quantize(np.round(
                    acc.astype(np.float32)
                    * np.float32((sa * sb) / (so * sdiv)))).ravel()
                np.testing.assert_array_equal(out, want)

    def test_add_tile_repeats_the_trailing_block(self):
        f = self._libs["add_tile_s8"].kernel_add_tile_s8
        f.argtypes = [_I8P] * 3 + [ctypes.c_int] * 2 + \
                     [ctypes.c_float] * 3 + [ctypes.c_int] * 2
        outer, inner = 6, 97
        st, sx, so = 0.0787, 0.0133, 0.09
        tile = self._rng.integers(-128, 128, size=inner, dtype=np.int8)
        x = self._rng.integers(-128, 128, size=outer * inner, dtype=np.int8)
        out = np.zeros(outer * inner, dtype=np.int8)
        f(_p(tile), _p(x), _p(out), outer, inner, st, sx, so, -128, 127)
        want = _quantize((np.tile(tile, outer).astype(np.float32) * np.float32(st)
                          + x.astype(np.float32) * np.float32(sx))
                         / np.float32(so))
        np.testing.assert_array_equal(out, want)

    def test_add_c1_broadcasts_one_value_per_channel(self):
        f = self._libs["add_c1_s8"].kernel_add_c1_s8
        f.argtypes = [_I8P] * 3 + [ctypes.c_int] * 3 + \
                     [ctypes.c_float] * 3 + [ctypes.c_int] * 2
        N, C, HW = 2, 32, 101
        sg, sx, so = 0.0079, 0.0358, 0.0358
        gate = self._rng.integers(-128, 128, size=C, dtype=np.int8)
        x = self._rng.integers(-128, 128, size=N * C * HW, dtype=np.int8)
        out = np.zeros(N * C * HW, dtype=np.int8)
        f(_p(gate), _p(x), _p(out), N, C, HW, sg, sx, so, -128, 127)
        per_c = np.repeat(gate.astype(np.float32) * np.float32(sg), HW)
        want = _quantize((np.tile(per_c, N)
                          + x.astype(np.float32) * np.float32(sx))
                         / np.float32(so))
        np.testing.assert_array_equal(out, want)


if __name__ == "__main__":
    unittest.main()
