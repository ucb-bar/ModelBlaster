"""ime2_a100 weights packed at build time in the A100 matrix tile order (TASK-002).

The A100 linear, conv and fused-conv kernels used to repack their weights into
8x16 matrix tiles on every call. An ime2_a100 build now emits those weights
already tiled, behind a 64-byte header ("MBIME2TW", rows, K, zeros) that the
kernels check (MB_IME2_TW), in a 64-byte-aligned array so the tiles start on a
cache line; weights without the header are packed per call as before.

What is pinned here:
  * the build-time tiles are byte-for-byte what the kernels pack per call
    (checked against a direct port of the kernels' packing loops);
  * which weights are tiled: ime2_a100 only; linear/conv/fused only; never a
    split/sharded/shared weight, a batch-1 or tiny linear (a pool splits M == 1
    by columns; M*N <= 16 is the kernel's scalar path) or a weight under 16
    bytes (the kernel does not look for a header there);
  * weights.h declares the tiled size, weights.c aligns tiled arrays to 64, and
    the manifest is written or removed;
  * MODELBLASTER_IME2_TILES=0 turns tiling off;
  * generate_kernels refuses a kernel that cannot read tiled weights.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from pipeline import generate_kernels  # noqa: E402
from pipeline import generate_skeleton as gs  # noqa: E402

RNG = np.random.default_rng(7)


def kernel_linear_tiles(w: np.ndarray) -> np.ndarray:
    """Port of kernel_linear_s8's per-call B packing: per 32-column group,
    pb[ks*512 + p*128 + r*16 + q] = weight[(n0 + 8p + r)*K + 16ks + q], zero padded."""
    n_rows, k = w.shape
    ks_n = (k + 15) // 16
    out = np.zeros(((n_rows + 31) // 32) * ks_n * 512, np.int8)
    for g in range((n_rows + 31) // 32):
        for ks in range(ks_n):
            for p in range(4):
                for r in range(8):
                    n = g * 32 + p * 8 + r
                    for q in range(16):
                        kk = ks * 16 + q
                        if n < n_rows and kk < k:
                            out[((g * ks_n + ks) * 4 + p) * 128 + r * 16 + q] = w[n, kk]
    return out


def kernel_conv_tiles(w_oihw: np.ndarray) -> np.ndarray:
    """Port of kernel_conv2d_s8's per-call A packing from IHWOC weight[k*OC + oc]:
    pa[(ot*k_slabs + ks)*128 + r*16 + q], oc = 8ot + r, k = 16ks + q, zero padded."""
    oc_n = w_oihw.shape[0]
    ihwoc = np.transpose(w_oihw, (1, 2, 3, 0)).reshape(-1)   # what the rvv lineage emits
    k = ihwoc.size // oc_n
    ks_n = (k + 15) // 16
    out = np.zeros(((oc_n + 7) // 8) * ks_n * 128, np.int8)
    for ot in range((oc_n + 7) // 8):
        for ks in range(ks_n):
            for r in range(8):
                oc = ot * 8 + r
                for q in range(16):
                    kk = ks * 16 + q
                    if oc < oc_n and kk < k:
                        out[(ot * ks_n + ks) * 128 + r * 16 + q] = ihwoc[kk * oc_n + oc]
    return out


def header(rows: int, k: int) -> bytes:
    return b"MBIME2TW" + np.array([rows, k], "<u4").tobytes() + bytes(48)


class TheTilesAreWhatTheKernelsPack(unittest.TestCase):
    def test_linear(self):
        for n, k in [(1024, 256), (256, 1024), (40, 33), (9, 17), (32, 16), (70, 100)]:
            with self.subTest(n=n, k=k):
                w = RNG.integers(-128, 128, (n, k), dtype=np.int8)
                got = gs.ime2_tile_pack(w, "linear", n, k)
                self.assertEqual(got[:64].tobytes(), header(n, k))
                np.testing.assert_array_equal(got[64:], kernel_linear_tiles(w))

    def test_conv(self):
        for oc, ic, kh, kw in [(32, 3, 3, 3), (64, 32, 1, 1), (80, 80, 3, 3), (12, 5, 3, 3), (8, 2, 1, 1)]:
            with self.subTest(shape=(oc, ic, kh, kw)):
                w = RNG.integers(-128, 128, (oc, ic, kh, kw), dtype=np.int8)
                got = gs.ime2_tile_pack(w, "conv", oc, ic * kh * kw)
                self.assertEqual(got[:64].tobytes(), header(oc, ic * kh * kw))
                np.testing.assert_array_equal(got[64:], kernel_conv_tiles(w))


def _ir():
    lin = lambda did, w, m, k, n: {"op": "linear_s8", "dispatch_id": did, "name": f"l{did}",
                                   "weight": w, "shape": {"M": m, "K": k, "N": n}}
    conv = lambda did, w, oc, ic, extra=None: {"op": "conv2d_s8", "dispatch_id": did, "name": f"c{did}",
                                               "weight": w, "shape": {"N": 1, "IC": ic, "IH": 8, "IW": 8,
                                                                      "OC": oc, "KH": 3, "KW": 3},
                                               **(extra or {})}
    fused = {"op": "conv2d_batchnorm2d_silu_s8", "dispatch_id": 4, "name": "f4",
             "sub_ops": [{"op": "conv2d_s8", "weight": "f.w",
                          "shape": {"N": 1, "IC": 16, "IH": 8, "IW": 8, "OC": 32, "KH": 1, "KW": 1}}]}
    return {"name": "m", "ops": [
        lin(0, "fc.w", 128, 256, 1024),            # tiled
        lin(1, "head.w", 1, 2048, 1),              # tiny path: plain
        conv(2, "c.w", 32, 16),                    # tiled
        conv(3, "split.w", 32, 16, {"split_from": {"axis": "OC", "tile": 0}}),   # split: plain
        fused,                                     # tiled (weight on sub_ops)
        lin(5, "shared.w", 64, 64, 64), lin(6, "shared.w", 64, 64, 64),           # two readers: plain
        conv(7, "small.w", 1, 1, {"shape": {"N": 1, "IC": 1, "IH": 4, "IW": 4, "OC": 1, "KH": 3, "KW": 3}}),
        {"op": "conv2d_batchnorm2d_s8", "dispatch_id": 8, "name": "cb8", "weight": "cb.w",
         "shape": {"N": 1, "IC": 16, "IH": 8, "IW": 8, "OC": 32, "KH": 3, "KW": 3}},   # RVV-only op: plain
        {**lin(9, "asym.w", 64, 64, 64), "quant": {"input_offset": -3, "filter_offset": 0}},  # asymmetric: plain
        lin(10, "m1.w", 1, 256, 128),             # batch 1 (a pool splits it by columns): plain
    ]}


class WhichWeightsAreTiled(unittest.TestCase):
    def test_plan(self):
        plan = gs.ime2_tile_plan(_ir(), "ime2_a100")
        self.assertEqual(plan, {
            "fc.w": ("linear_s8", "linear", 1024, 256),
            "c.w": ("conv2d_s8", "conv", 32, 16 * 9),
            "f.w": ("conv2d_batchnorm2d_silu_s8", "conv", 32, 16),
        })

    def test_only_ime2_a100(self):
        for backend in ("rvv_x60", "ime_x60", "rvv", None):
            with self.subTest(backend=backend):
                self.assertEqual(gs.ime2_tile_plan(_ir(), backend), {})

    def test_sharded_ops_are_excluded(self):
        self.assertNotIn("c.w", gs.ime2_tile_plan(_ir(), "ime2_a100", frozenset({2})))

    def test_can_be_turned_off(self):
        from unittest import mock
        with mock.patch.dict(os.environ, {"MODELBLASTER_IME2_TILES": "0"}):
            self.assertEqual(gs.ime2_tile_plan(_ir(), "ime2_a100"), {})


class TheEmittedFiles(unittest.TestCase):
    def _weights(self):
        return {"fc.w": RNG.integers(-128, 128, (1024, 256), dtype=np.int8),
                "head.w": RNG.integers(-128, 128, (1, 2048), dtype=np.int8),
                "c.w": RNG.integers(-128, 128, (32, 16, 3, 3), dtype=np.int8),
                "cb.w": RNG.integers(-128, 128, (32, 16, 3, 3), dtype=np.int8)}

    def _ir(self):
        ir = _ir()
        keep = {"fc.w", "head.w", "c.w", "cb.w"}
        ir["ops"] = [op for op in ir["ops"] if (op.get("weight") or "") in keep]
        return ir

    def test_ime2_a100(self):
        with tempfile.TemporaryDirectory() as d:
            gs.emit_weights("m", self._weights(), d, backend="ime2_a100", ir=self._ir())
            h = open(os.path.join(d, "weights.h")).read()
            c = open(os.path.join(d, "weights.c")).read()
            man = json.load(open(os.path.join(d, gs.WEIGHT_LAYOUTS_JSON)))
        fc = gs.ime2_tile_pack(self._weights()["fc.w"], "linear", 1024, 256).size
        self.assertIn(f"[{fc}];", h)                       # tiled size, not 1024*256
        self.assertIn("A100 linear tiles", c)
        self.assertIn("77, 66, 73, 77, 69, 50, 84, 87", c)  # "MBIME2TW" as int8 literals
        self.assertEqual(man["ime2_tile"]["ops"], ["conv2d_s8", "linear_s8"])
        self.assertEqual(set(man["ime2_tile"]["weights"]), {"fc.w", "c.w"})
        self.assertIn(f"[{32 * 16 * 9}];", h)              # cb.w (RVV op) keeps its plain size
        self.assertIn("__attribute__((aligned(64))) const int8_t m_fc_w_ime2_a100[", c)
        self.assertNotIn("__attribute__((aligned(64))) const int8_t m_cb_w_ime2_a100[", c)

    def test_other_backends_unchanged_and_stale_manifest_removed(self):
        with tempfile.TemporaryDirectory() as d:
            stale = os.path.join(d, gs.WEIGHT_LAYOUTS_JSON)
            open(stale, "w").write("{}")
            gs.emit_weights("m", self._weights(), d, backend="rvv_x60", ir=self._ir())
            self.assertFalse(os.path.exists(stale))
            self.assertNotIn("MBIME2TW", open(os.path.join(d, "weights.c")).read())


class TheKernelCheck(unittest.TestCase):
    def _manifest(self, d, ops):
        with open(os.path.join(d, "weight_layouts.json"), "w") as f:
            json.dump({"ime2_tile": {"ops": ops, "weights": {}}}, f)

    def test_refuses_a_kernel_that_cannot_read_tiles(self):
        with tempfile.TemporaryDirectory() as d:
            self._manifest(d, ["linear_s8"])
            with self.assertRaises(SystemExit):
                generate_kernels._check_tiled_weights({"linear_s8": "void kernel_linear_s8(void) {}"}, d)

    def test_accepts_the_curated_kernel(self):
        src = (_ROOT / "kernels" / "ime2_a100" / "ime2_a100_linear_s8_ime2_vmadot_8x8x16.c").read_text()
        with tempfile.TemporaryDirectory() as d:
            self._manifest(d, ["linear_s8"])
            generate_kernels._check_tiled_weights({"linear_s8": src}, d)

    def test_no_manifest_no_check(self):
        with tempfile.TemporaryDirectory() as d:
            generate_kernels._check_tiled_weights({"linear_s8": "anything"}, d)

    def test_installed_kernels_read_tiles(self):
        for op in ("linear_s8", "conv2d_s8", "conv2d_batchnorm2d_silu_s8"):
            with self.subTest(op=op):
                self.assertTrue(gs._ime2_kernel_reads_tiles(op))


if __name__ == "__main__":
    unittest.main()
