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
