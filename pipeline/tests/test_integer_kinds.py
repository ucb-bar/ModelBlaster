"""Integer-closed kinds (integer_kernels.py), integer_close.py's constants, apply_qparams.py's refusals.

The C reference kernels are compiled on the host and checked against an independent numpy transliteration of the
integer contract (exact int64 / Python-int arithmetic), on random data and on the edges: negative shifts,
accumulators near 2^47, ties, saturation.
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

from pipeline import apply_qparams as aq  # noqa: E402
from pipeline import integer_close as ic  # noqa: E402
from pipeline import integer_kernels as ik  # noqa: E402


def requant_ref(t, M, s):
    """The contract's requant on Python ints (unbounded), elementwise."""
    out = []
    for v in np.asarray(t, np.int64).ravel().tolist():
        if s < 0:
            v <<= -s
        q = (v * int(M) + (1 << 30)) >> 31
        if s > 0:
            q = (q + (1 << (s - 1))) >> s
        out.append(q)
    return np.array(out, dtype=object)


def sat(v, bits):
    lo, hi = (-16384, 16383) if bits == 15 else (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
    return np.clip(np.asarray(v, dtype=object).astype(np.int64), lo, hi)


_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        src = "".join(s.reference_impl for s in ik.SPECS) + r"""
int64_t mbi_requant_x(int64_t t, int32_t M, int32_t s) { return mbi_requant(t, M, s); }
"""
        d = tempfile.mkdtemp()
        c, so = os.path.join(d, "k.c"), os.path.join(d, "k.so")
        open(c, "w").write(src)
        subprocess.run(["gcc", "-O2", "-shared", "-fPIC", c, "-o", so], check=True)
        _LIB = ctypes.CDLL(so)
        _LIB.mbi_requant_x.restype = ctypes.c_int64
        _LIB.mbi_requant_x.argtypes = [ctypes.c_int64, ctypes.c_int32, ctypes.c_int32]
    return _LIB


def P(a):
    return a.ctypes.data_as(ctypes.c_void_p)


class RequantTests(unittest.TestCase):

    def test_requant_matches_the_contract(self):
        rng = np.random.default_rng(0)
        cases = [(int(t), int(M), int(s)) for t, M, s in zip(
            rng.integers(-(1 << 44), 1 << 44, 4000), rng.integers(1 << 30, 1 << 31, 4000), rng.integers(-3, 40, 4000))]
        cases += [((1 << 46) - 1, (1 << 31) - 1, 0), (-(1 << 46), (1 << 31) - 1, 0), (-1, 1 << 30, 1), (1, 1 << 30, 1),
                  (-3, 1 << 30, 1), ((1 << 43) - 1, (1 << 31) - 1, -3), (-(1 << 43), 1 << 30, -3), (7, 0, 0)]
        L = lib()
        for t, M, s in cases:
            self.assertEqual(L.mbi_requant_x(t, M, s), int(requant_ref([t], M, s)[0]), (t, M, s))


class KernelTests(unittest.TestCase):

    def test_gap_q_int8_negative_shift_and_int16(self):
        rng = np.random.default_rng(1)
        for bits_in, bits_out, s in ((8, 8, -1), (16, 16, 5), (16, 15, 3)):
            N, C, HW = 2, 5, 37
            dt_in = np.int8 if bits_in == 8 else np.int16
            x = rng.integers(np.iinfo(dt_in).min, np.iinfo(dt_in).max, (N, C, HW)).astype(dt_in)
            y = np.zeros((N, C), np.int8 if bits_out == 8 else np.int16)
            M = 1_500_000_000
            lib().kernel_gap_q(P(x), bits_in, P(y), bits_out, N, C, HW, ctypes.c_int32(M), ctypes.c_int32(s))
            exp = sat(requant_ref(x.astype(np.int64).sum(-1), M, s), bits_out).reshape(N, C)
            np.testing.assert_array_equal(y, exp)

    def test_mul_c1_q(self):
        rng = np.random.default_rng(2)
        N, C, HW = 2, 4, 9
        x = rng.integers(-32768, 32767, (N, C, HW)).astype(np.int16)
        g = rng.integers(0, 32767, (N, C)).astype(np.int16)
        y = np.zeros_like(x)
        M, s = 1_234_567_890, 14
        lib().kernel_mul_c1_q(P(x), 16, P(g), 16, P(y), 16, N, C, HW, ctypes.c_int32(M), ctypes.c_int32(s))
        exp = sat(requant_ref(x.astype(np.int64) * g.astype(np.int64)[..., None], M, s), 16).reshape(N, C, HW)
        np.testing.assert_array_equal(y, exp)

    def test_add_q_per_row_and_broadcast(self):
        rng = np.random.default_rng(3)
        rows, cols, S = 14, 33, 22
        a = rng.integers(-32768, 32767, (rows, cols)).astype(np.int16)
        b = rng.integers(-32767, 32767, (7, cols)).astype(np.int16)
        ma = rng.integers(0, 1 << 23, 7).astype(np.int32)
        mb = 3_000_000
        y = np.zeros_like(a)
        lib().kernel_add_q(P(a), 16, P(b), 16, P(y), 16, rows, cols, P(ma), 7, ctypes.c_int32(mb), 7, S)
        r = np.arange(rows) % 7
        v = (a.astype(np.int64) * ma[r, None] + b[r].astype(np.int64) * mb + (1 << (S - 1))) >> S
        np.testing.assert_array_equal(y, sat(v, 16))

    def test_lut_s8(self):
        x = np.arange(-128, 128, dtype=np.int8)
        T = ic.lut8_table(ic._silu, 0.05, 0.03)
        y = np.zeros_like(x)
        lib().kernel_lut_s8(P(x), P(y), 256, P(T))
        np.testing.assert_array_equal(y, T)


class BuildTimeTests(unittest.TestCase):

    def test_quant_mult(self):
        for m in (1e-6, 0.37, 0.5, 0.9999999, 1.0, 3.7):
            M, s = ic.quant_mult(m)
            self.assertTrue((1 << 30) <= M < (1 << 31))
            self.assertAlmostEqual(M * 2.0 ** -(31 + s), m, delta=m * 2 ** -30)
        self.assertEqual(ic.quant_mult(3.7)[1], -2)            # a pre-shift

    def test_add_mults_bound(self):
        (ma, mb), S = ic.add_mults([0.3, 1.7])
        self.assertLessEqual(max(ma, mb), (1 << 23) - 1)
        self.assertGreater(int(ic.rha(1.7 * 2 ** (S + 1))), (1 << 23) - 1)


class ApplyQparamsTests(unittest.TestCase):

    def _ir(self):
        ir = {"ops": [{"name": "c", "op": "conv2d_s8_pc", "weight": "c.w", "bias": "c.b",
                       "quant": {"output_multiplier_per_oc_key": "c.m", "S": 3}}],
              "tensors": {"t": {"shape": [1], "dtype": "i8", "quant": {"scale": 1.0}}}}
        w = {"c.w": np.zeros((2, 1), np.int8), "c.b": np.zeros(2, np.int32), "c.m": np.zeros(2, np.int32)}
        return ir, w

    def test_applies_verbatim(self):
        ir, w = self._ir()
        res = aq.apply(ir, w, {"ops": {"c": {"op": "conv2d_s8_pc", "arrays": {"weight": "W", "bias": "B"},
                                             "quant": {"S": 7}}}, "tensors": {"t": {"scale": 0.5}}},
                       {"W": np.array([[3], [-4]], np.int64), "B": np.array([1, 2], np.int64)})
        np.testing.assert_array_equal(res["weights"]["c.w"], [[3], [-4]])
        self.assertEqual(res["weights"]["c.w"].dtype, np.int8)
        self.assertEqual(ir["ops"][0]["quant"]["S"], 7)
        self.assertEqual(ir["tensors"]["t"]["quant"]["scale"], 0.5)

    def test_refusals(self):
        bad = [({"ops": {"x": {}}}, {}),                                                  # unknown op
               ({"ops": {"c": {"op": "linear_s8_pc"}}}, {}),                              # wrong kind
               ({"ops": {"c": {"arrays": {"weight": "W"}}}}, {"W": np.zeros(2, np.int8)}),  # reshape refused
               ({"ops": {"c": {"arrays": {"weight": "W"}}}}, {"W": np.full((2, 1), 300)}),  # out of int8 range
               ({"ops": {"c": {"arrays": {"weight": "W"}}}}, {"W": np.zeros((2, 1))}),     # float
               ({"ops": {"c": {"quant": {"nope": 1}}}}, {}),                                # unknown quant field
               ({"tensors": {"nope": {"scale": 1.0}}}, {})]                                 # unknown tensor
        for patch, arrays in bad:
            ir, w = self._ir()
            with self.assertRaises(aq.QparamsError, msg=str(patch)):
                aq.apply(ir, w, patch, arrays)


def _mb(module):
    """Import modelblaster.pipeline.<module> (the package name the pipeline imports through)."""
    import importlib
    try:
        return importlib.import_module(f"modelblaster.pipeline.{module}")
    except ModuleNotFoundError:
        pkg = tempfile.mkdtemp()
        os.symlink(_ROOT, os.path.join(pkg, "modelblaster"))
        sys.path.insert(0, pkg)
        return importlib.import_module(f"modelblaster.pipeline.{module}")


class IntegerGateTests(unittest.TestCase):

    def test_parse_layers(self):
        pk = _mb("profile_kernel")
        out = ("=== MODELBLASTER_LAYER [a] ok 1 1 ===\nnoise\n=== MODELBLASTER_LAYER [b.c] BAD 2 3 ===\n"
               "=== MODELBLASTER_LAYER [d] BAD 4 5 ===\n")
        self.assertEqual(pk.parse_layers(out), (1, 2, "b.c"))
        self.assertEqual(pk.parse_layers("nothing"), (0, 0, None))

    def test_io_is_exact(self):
        pk = _mb("profile_kernel")
        d = tempfile.mkdtemp()
        cases = {"f.npz": (dict(output=np.zeros(3, np.float32)), False),
                 "i.npz": (dict(output=np.zeros(3, np.int64)), True),
                 "x.npz": (dict(output=np.zeros(3, np.float32), exact=np.array(1)), True)}
        for name, (arrs, want) in cases.items():
            np.savez(os.path.join(d, name), input=np.zeros(1, np.int8), **arrs)
            self.assertEqual(pk.io_is_exact(os.path.join(d, name)), want, name)
        self.assertFalse(pk.io_is_exact(None))


class LayerCheckTests(unittest.TestCase):

    def test_crc_matches_zlib(self):
        import importlib
        import zlib
        try:
            gs = importlib.import_module("modelblaster.pipeline.generate_skeleton")
        except ModuleNotFoundError:
            # generate_skeleton imports through the package name; give the
            # checkout one under a temporary directory.
            pkg = tempfile.mkdtemp()
            os.symlink(_ROOT, os.path.join(pkg, "modelblaster"))
            sys.path.insert(0, pkg)
            gs = importlib.import_module("modelblaster.pipeline.generate_skeleton")
        src = gs._layer_check_src("t").replace("static void", "void")
        d = tempfile.mkdtemp()
        c, so = os.path.join(d, "l.c"), os.path.join(d, "l.so")
        open(c, "w").write(src)
        subprocess.run(["gcc", "-O2", "-shared", "-fPIC", c, "-o", so], check=True)
        L = ctypes.CDLL(so)
        data = np.random.default_rng(4).integers(-128, 127, 1000).astype(np.int8)
        crc = zlib.crc32(data.tobytes())
        r, w = os.pipe()
        saved = os.dup(1)
        os.dup2(w, 1)
        L.mb_layer_check_t(b"a", P(data), ctypes.c_ulong(data.size), ctypes.c_ulong(crc))
        L.mb_layer_check_t(b"b", P(data), ctypes.c_ulong(data.size), ctypes.c_ulong(crc ^ 1))
        ctypes.CDLL(None).fflush(None)
        os.dup2(saved, 1)
        os.close(w)
        out = os.read(r, 4096).decode()
        self.assertIn("[a] ok", out)
        self.assertIn("[b] BAD", out)


if __name__ == "__main__":
    unittest.main()
