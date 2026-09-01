"""A vector build that runs scalar code must fail, and must not cry wolf.

The defect this guards: curated kernels are looked up by EXACT op name, so a
FUSED op (`conv2d_batchnorm2d_silu_s8`) matches nothing even when every
constituent (`conv2d_s8`, `batchnorm2d_s8`, `silu_s8`) has a kernel. Selection
falls back to the scalar reference, records it in kernel_picks.json, and the
build reports success. On the K1 that made yolov8_nano measure 0.81x against
the scalar build -- slower -- while looking like a finding about RVV.

The second class of test here matters as much as the first. kernel_picks.json is
written at generate time and outlives its sources, so a stale one claimed
mlp_control's `linear_s8` was on the reference when the measured run showed a
curated kernel. A gate that reported that would manufacture a regression, and
would be switched off the first time someone checked it by hand.
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPT = os.path.join(_HERE, "..", "scripts", "check_kernel_coverage.py")
_spec = importlib.util.spec_from_file_location("check_kernel_coverage", _SCRIPT)
cov = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cov)

_CSV_HEAD = "dispatch_id,module_name,mean_time_ns,op,implementation\n"


class _Case:
    """A generated dir plus an optional profile, on disk."""

    def __init__(self, picks, ops=None, profile_rows=None):
        self.dir = tempfile.mkdtemp()
        gen = os.path.join(self.dir, "rvv_x60")
        os.makedirs(gen)
        self.gen = gen
        with open(os.path.join(gen, "kernel_picks.json"), "w") as fh:
            json.dump({"schema_version": 1, "target": "rvv_x60",
                       "picks": picks}, fh)
        with open(os.path.join(self.dir, "graph.json"), "w") as fh:
            json.dump({"ops": [{"op": o} for o in (ops or [])]}, fh)
        self.profile = None
        if profile_rows is not None:
            self.profile = os.path.join(self.dir, "results.csv")
            with open(self.profile, "w") as fh:
                fh.write(_CSV_HEAD)
                for i, row in enumerate(profile_rows):
                    mod, ns, op = row[0], row[1], row[2]
                    impl = row[3] if len(row) > 3 else ""
                    fh.write(f"{i},{mod},{ns},{op},{impl}\n")

    def check(self, **kw):
        return cov.check(self.gen, os.path.join(self.dir, "graph.json"),
                         self.profile, **kw)


_CURATED = {"source": "curated[rvv]", "algorithm": "direct", "path": "x.c"}
_REF = {"source": "reference", "algorithm": None, "path": None}


class FusedOpFallingBackFails(unittest.TestCase):

    def test_the_real_yolov8_shape_fails(self):
        """Constituents present, fused op absent -- the exact production bug."""
        c = _Case(
            picks={"conv2d_s8": _CURATED, "batchnorm2d_s8": _CURATED,
                   "silu_s8": _CURATED, "conv2d_batchnorm2d_silu_s8": _REF},
            ops=["conv2d_batchnorm2d_silu_s8"] * 57 + ["conv2d_s8"] * 3)
        self.assertFalse(c.check())

    def test_a_clean_build_passes(self):
        c = _Case(picks={"conv2d_s8": _CURATED}, ops=["conv2d_s8"] * 10)
        self.assertTrue(c.check())

    def test_an_op_with_no_dispatches_cannot_fail_the_build(self):
        """Weight is what separates a defect from an unused op."""
        c = _Case(picks={"conv2d_s8": _CURATED, "lstm_s8": _REF},
                  ops=["conv2d_s8"] * 10)
        self.assertTrue(c.check())

    def test_a_benign_reshape_is_not_a_defect(self):
        c = _Case(picks={"conv2d_s8": _CURATED, "reshape": _REF},
                  ops=["conv2d_s8"] * 5 + ["reshape"] * 5)
        self.assertTrue(c.check())

    def test_an_explicit_allow_is_honoured(self):
        c = _Case(picks={"add_s8": _REF}, ops=["add_s8"] * 10)
        self.assertFalse(c.check())
        self.assertTrue(c.check(allow={"add_s8"}))

    def test_a_scalar_target_is_not_judged(self):
        c = _Case(picks={"conv2d_s8": _REF}, ops=["conv2d_s8"] * 10)
        with open(os.path.join(c.gen, "kernel_picks.json"), "w") as fh:
            json.dump({"target": "scalar", "picks": {"conv2d_s8": _REF}}, fh)
        self.assertTrue(c.check())


class TheMeasurementOutranksTheStalePicksFile(unittest.TestCase):

    def test_a_stale_reference_claim_does_not_invent_a_regression(self):
        """mlp_control's case: picks says reference, the run says curated."""
        c = _Case(
            picks={"linear_s8": _REF},
            profile_rows=[("m$dispatch_0_rvv_x60_linear_s8_M1xK16xN256",
                           50_000_000, "linear_s8")])
        self.assertTrue(c.check())

    def test_a_stale_curated_claim_does_not_hide_a_real_regression(self):
        """The dangerous direction: picks looks clean, the run ran scalar."""
        c = _Case(
            picks={"conv2d_batchnorm2d_s8": _CURATED},
            profile_rows=[("m$dispatch_0_rvv_x60_conv2d_batchnorm2d_s8_scalar",
                           54_000_000, "conv2d_batchnorm2d_s8")])
        self.assertFalse(c.check())

    def test_time_is_weighted_not_dispatch_count(self):
        """One slow scalar op outranks many fast curated ones.

        DroNet: 3 of 21 dispatches on the reference is 13% by count and 86.7%
        by time. Counting dispatches would have rated this below the threshold.
        """
        rows = [("m$d_rvv_x60_conv2d_batchnorm2d_s8_scalar", 54_000_000,
                 "conv2d_batchnorm2d_s8")]
        rows += [(f"m$d{i}_rvv_x60_conv2d_s8_N1xIC3", 400_000, "conv2d_s8")
                 for i in range(18)]
        c = _Case(picks={"conv2d_s8": _CURATED,
                         "conv2d_batchnorm2d_s8": _REF},
                  profile_rows=rows)
        self.assertFalse(c.check())


if __name__ == "__main__":
    unittest.main()


class TheImplementationColumnIsTheGroundTruth(unittest.TestCase):
    """module_name cannot answer this, and reading it as if it could is a bug.

    profile_writer builds module_name as `<model>$dispatch_<id>_<backend>_<op>_
    <shape>`, and `_shape_concise()` returned the literal "scalar" for an op
    with no recorded shape. The fused convs had no shape AND ran the reference,
    so `..._conv2d_batchnorm2d_silu_s8_scalar` looked like a reliable
    "ran scalar" marker. It was a coincidence. Once real RVV kernels landed
    (22.9x measured on the K1) the name still ended in `_scalar`, and a gate
    trusting it would have failed a build that was correct.

    The `implementation` column is filled from kernel_picks.json at PROFILE
    time, so it describes the build that was actually measured.
    """

    def test_a_shapeless_op_with_a_real_kernel_passes(self):
        """The exact post-fix state: name says _scalar, impl says curated."""
        c = _Case(
            picks={"conv2d_batchnorm2d_silu_s8": _CURATED},
            profile_rows=[
                ("y$dispatch_0_rvv_x60_conv2d_batchnorm2d_silu_s8_scalar",
                 217_000_000, "conv2d_batchnorm2d_silu_s8",
                 "curated[rvv]/rvv_oc_blocked_bn_silu_epilogue")])
        self.assertTrue(c.check())

    def test_the_column_still_catches_a_genuine_fallback(self):
        c = _Case(
            picks={"conv2d_batchnorm2d_silu_s8": _CURATED},
            profile_rows=[
                ("y$dispatch_0_rvv_x60_conv2d_batchnorm2d_silu_s8_noshape",
                 4_962_000_000, "conv2d_batchnorm2d_silu_s8", "reference")])
        self.assertFalse(c.check())

    def test_the_column_outranks_a_stale_picks_file(self):
        """picks claims curated; the measured run says reference."""
        c = _Case(
            picks={"conv2d_s8": _CURATED},
            profile_rows=[("m$dispatch_0_rvv_x60_conv2d_s8_N1xIC3",
                           54_000_000, "conv2d_s8", "reference")])
        self.assertFalse(c.check())

    def test_an_empty_column_falls_back_to_the_legacy_heuristic(self):
        """Profiles written before the column existed must still be judged."""
        c = _Case(
            picks={"conv2d_batchnorm2d_s8": _REF},
            profile_rows=[("m$dispatch_0_rvv_x60_conv2d_batchnorm2d_s8_scalar",
                           54_000_000, "conv2d_batchnorm2d_s8", "")])
        self.assertFalse(c.check())
