"""Locks in the only-if-better IME rule (ime_cost) + the ffn/attn verdicts.

Dependency-free (ime_cost imports only stdlib), so it runs anywhere. Doubles as
the ffn/attn "does IME actually win?" verification: ffn (M=128) -> IME, attn
(M=8) -> RVV, conv -> per-dispatch (deferred), all from MEASURED data.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline import ime_cost  # noqa: E402


class TestIMESpeedup(unittest.TestCase):
    def test_matmul_small_M_loses(self):
        sp, prov = ime_cost.ime_speedup_for("linear_s8", {"M": 8, "K": 32, "N": 32})
        self.assertLess(sp, 1.0)                 # attention regime: IME slower
        self.assertEqual(prov, "measured-anchor")

    def test_matmul_large_M_wins(self):
        sp, _ = ime_cost.ime_speedup_for("linear_s8", {"M": 128, "K": 256, "N": 1024})
        self.assertGreater(sp, 1.5)              # ffn regime: IME faster (~2.3x)

    def test_conv_measured_lookup(self):
        # yolov8 l6.cv2 (IC256 OC128 1x1) measured win; l0 (IC3 OC16 3x3) measured loss
        win, prov = ime_cost.ime_speedup_for(
            "conv2d_s8", {"IC": 256, "IH": 10, "IW": 10, "OC": 128, "KH": 1, "KW": 1})
        self.assertEqual(prov, "measured")
        self.assertGreater(win, 1.0)
        lose, prov2 = ime_cost.ime_speedup_for(
            "conv2d_s8", {"IC": 3, "IH": 160, "IW": 160, "OC": 16, "KH": 3, "KW": 3})
        self.assertEqual(prov2, "measured")
        self.assertLess(lose, 1.0)

    def test_conv_unmeasured_is_unknown(self):
        sp, prov = ime_cost.ime_speedup_for(
            "conv2d_s8", {"IC": 999, "IH": 7, "IW": 7, "OC": 999, "KH": 9, "KW": 9})
        self.assertIsNone(sp)                    # never guessed
        self.assertEqual(prov, "unmeasured")


class TestFusedConvUsesItsOwnTable(unittest.TestCase):
    """A speedup is IME-vs-the-RVV-kernel-that-would-otherwise-run, so the fused
    conv must be costed against the FUSED RVV kernel the deployed build runs --
    not against the standalone conv, which nothing in that graph executes."""

    # deployed yolov8_nano_64x96 l2.m0.cv1: 0.41x against the standalone RVV
    # conv, 1.53x against the fused one. The wrong table is the difference
    # between "stays RVV" and "goes to the matrix engine".
    SHAPE = {"IC": 16, "IH": 16, "IW": 24, "OC": 16, "KH": 3, "KW": 3}

    def test_fused_op_reads_the_fused_table(self):
        sp, prov = ime_cost.ime_speedup_for("conv2d_batchnorm2d_silu_s8", self.SHAPE)
        self.assertEqual(prov, "measured-fused")
        self.assertGreater(sp, 1.0)

    def test_plain_conv_still_reads_the_standalone_table(self):
        sp, prov = ime_cost.ime_speedup_for("conv2d_s8", self.SHAPE)
        self.assertEqual(prov, "measured")
        self.assertLess(sp, 1.0)
        # ... and the two tables really do disagree on this shape, which is the
        # whole reason the op-kind has to pick its own.
        fused, _ = ime_cost.ime_speedup_for("conv2d_batchnorm2d_silu_s8", self.SHAPE)
        self.assertGreater(fused, sp)

    def test_conv_op_with_no_table_stays_rvv(self):
        # conv2d_batchnorm2d_s8 has no IME kernel and no measurement; borrowing
        # another op's table is exactly the bug. Only-if-better => None.
        sp, why = ime_cost.ime_speedup_for(
            "conv2d_batchnorm2d_s8", {"IC": 32, "IH": 27, "IW": 27, "OC": 32, "KH": 3, "KW": 3})
        self.assertIsNone(sp)
        self.assertIn("no measured", why)

    def test_unmeasured_fused_shape_stays_rvv(self):
        sp, prov = ime_cost.ime_speedup_for(
            "conv2d_batchnorm2d_silu_s8",
            {"IC": 999, "IH": 7, "IW": 7, "OC": 999, "KH": 9, "KW": 9})
        self.assertIsNone(sp)
        self.assertEqual(prov, "unmeasured")

    def test_fused_conv_kept_in_the_ime_table(self):
        # ime_useful is the multi-impl build's inclusion guard: with the fused
        # table the op is KNOWN faster on at least one shape, so the ime_x60
        # build keeps the kernel and the per-dispatch scheduler can route to it.
        keep, why = ime_cost.ime_useful("conv2d_batchnorm2d_silu_s8", [self.SHAPE])
        self.assertTrue(keep, why)
        # against the standalone table the same shape loses, i.e. the old
        # lookup excluded the kernel from the build outright.
        keep_wrong, _ = ime_cost.ime_useful("conv2d_s8", [self.SHAPE])
        self.assertFalse(keep_wrong)

    def test_repeated_shape_keeps_its_worst_row(self):
        # The fused table has one row per DISPATCH, so a shape can appear more
        # than once. The guard must not be decided by the luckiest instance.
        table = ime_cost.measured_conv_table("conv2d_batchnorm2d_silu_s8")
        self.assertIsNotNone(table)
        import csv as _csv
        rows = list(_csv.DictReader(open(ime_cost.CONV_TABLES["conv2d_batchnorm2d_silu_s8"])))
        worst = {}
        for r in rows:
            k = tuple(int(r[c]) for c in ("IC", "IH", "IW", "OC", "KH", "KW"))
            worst[k] = min(worst.get(k, 9e9), float(r["speedup"]))
        self.assertEqual(table, worst)


class TestAggregateVerdict(unittest.TestCase):
    def test_attention_stays_rvv(self):
        shapes = [{"M": 8, "K": 32, "N": 32}] * 4 + [{"M": 8, "K": 32, "N": 8}]
        win, _ = ime_cost.ime_wins_aggregate("matmul_s8", shapes)
        self.assertFalse(win)                    # the attn mispick this fixes

    def test_ffn_goes_ime_despite_a_tiny_linear(self):
        # mixed: two big M=128 GEMMs dominate, one tiny M=1 does not veto
        shapes = [{"M": 128, "K": 256, "N": 1024}, {"M": 128, "K": 1024, "N": 256},
                  {"M": 1, "K": 16, "N": 256}]
        win, _ = ime_cost.ime_wins_aggregate("linear_s8", shapes)
        self.assertTrue(win)

    def test_conv_is_deferred_to_scheduler(self):
        win, why = ime_cost.ime_wins_aggregate(
            "conv2d_s8", [{"IC": 256, "IH": 10, "IW": 10, "OC": 128, "KH": 1, "KW": 1}])
        self.assertFalse(win)                    # per-op-kind picker leaves conv on RVV
        self.assertIn("per-dispatch", why)


class TestFP16IsAccuracyGated(unittest.TestCase):
    """The K1 IME is int8-only: fp16 reaches it only via int8 requant, and only
    when the accuracy contract permits. Never a free/silent win."""

    def test_fp16_stays_rvv_by_default(self):
        sp, why = ime_cost.ime_speedup_for("linear_f16", {"M": 128, "K": 512, "N": 512})
        self.assertIsNone(sp)                    # even at favorable M
        self.assertIn("int8 requant", why)

    def test_fp16_eligible_only_when_accuracy_permits(self):
        sp, why = ime_cost.ime_speedup_for(
            "linear_f16", {"M": 128, "K": 512, "N": 512}, allow_int8_requant=True)
        self.assertGreater(sp, 1.0)
        self.assertIn("requant", why)
        win, _ = ime_cost.ime_wins_aggregate(
            "linear_f16", [{"M": 128, "K": 512, "N": 512}], allow_int8_requant=True)
        self.assertTrue(win)


if __name__ == "__main__":
    unittest.main()
