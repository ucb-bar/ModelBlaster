"""Costs the scheduler plans against must be a median of N, not one sample.

Why this exists
---------------
Every cost in the authoritative profile DB was a SINGLE sample. The harness
printed one PROFILE block, the host read it, and that number became both the
service time the scheduler plans against and the figure the advisor compares to
a free slot. Two closed-loop recommendations have already been rejected on n=1
comparisons; the gaps were large (41%, 36%) so those conclusions stand, but the
project's own stated criterion asks for a median of N and there was none.

The binary could already do the work: MODELBLASTER_ITERS>1 runs the model N
times and prints a per-iteration profile block. Nothing parsed them, so the
repetitions were paid for on the board and thrown away.

The warmup rule is not cosmetic. First-touch faulting of a const weight array
lands entirely in iteration 0 -- measured on vitfly_lstm, whose 1.7 MB of weights
cost 21.9 ms cold against 2.88 ms warm. Including iteration 0 in a median of 3
would drag the reported cost toward a number that only ever happens once.
"""

from __future__ import annotations

import os
import sys
import unittest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     ".."))
for _p in (os.path.join(_ROOT, "src"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from modelblaster.validation.runner_common import (  # noqa: E402
    parse_profile, parse_profile_reps,
)


def _block(it, vals, names=None):
    names = names or [f"op{i}" for i in range(len(vals))]
    out = [f"=== MODELBLASTER_ITER_PROFILE_BEGIN [{it}] ===",
           "dispatch_id,name,op,shape,cycles"]
    for i, (n, v) in enumerate(zip(names, vals)):
        out.append(f"{i},{n},conv2d_s8,N=1,{v}")
    out.append(f"=== MODELBLASTER_ITER_PROFILE_END [{it}] ===")
    return "\n".join(out)


class MedianAcrossReps(unittest.TestCase):

    def test_the_median_is_reported_not_the_first_sample(self):
        txt = "\n".join(_block(i, [v]) for i, v in
                        enumerate([1000, 1200, 1010, 1005, 1008]))
        r = parse_profile_reps(txt)
        self.assertEqual(len(r), 1)
        # Warm reps are [1200, 1010, 1005, 1008]; statistics.median averages the
        # two middle values of an even sample, so (1008 + 1010) / 2 = 1009.
        # Note this is NOT 1000, the first sample, which is the whole point --
        # and it is not 1200 either, so one slow rep does not set the cost.
        self.assertEqual(r[0]["cycles"], 1009)
        self.assertEqual(r[0]["cycles_n"], 4)

    def test_iteration_zero_is_dropped_as_warmup(self):
        """The cold sample is a page-fault measurement, not a cost."""
        txt = "\n".join([_block(0, [9000]), _block(1, [1000]),
                         _block(2, [1010]), _block(3, [1005]),
                         _block(4, [1002])])
        r = parse_profile_reps(txt)
        self.assertEqual(r[0]["cycles_n"], 4)
        self.assertLess(r[0]["cycles"], 1100,
                        "the 9000-cycle cold run must not pull the median")

    def test_too_few_reps_keeps_iteration_zero(self):
        """Dropping the only warm sample would leave nothing to report.

        With 3 or fewer blocks the warmup is kept and `cycles_n` says so, which
        is honest; silently returning a 1-sample median labelled n=2 would not
        be.
        """
        txt = "\n".join([_block(0, [9000]), _block(1, [1000])])
        r = parse_profile_reps(txt)
        self.assertEqual(r[0]["cycles_n"], 2)

    def test_spread_is_reported(self):
        txt = "\n".join(_block(i, [1000 + 10 * i]) for i in range(6))
        r = parse_profile_reps(txt)
        self.assertIn("cycles_cv_pct", r[0])
        self.assertGreater(r[0]["cycles_cv_pct"], 0.0)
        self.assertEqual(r[0]["cycles_min"], 1010)
        self.assertEqual(r[0]["cycles_max"], 1050)

    def test_a_stable_op_reports_zero_spread(self):
        txt = "\n".join(_block(i, [1000]) for i in range(5))
        self.assertEqual(parse_profile_reps(txt)[0]["cycles_cv_pct"], 0.0)

    def test_ops_are_keyed_by_identity_not_position(self):
        """Averaging unrelated ops together would be worse than n=1.

        If an iteration ever emitted a different op order or count, a
        positional zip would pair a convolution's cycles with an activation's.
        """
        txt = "\n".join([
            _block(0, [1000, 50], names=["conv", "relu"]),
            _block(1, [1010, 52], names=["conv", "relu"]),
            _block(2, [1005, 51], names=["conv", "relu"]),
            _block(3, [1002, 49], names=["conv", "relu"]),
        ])
        r = {x["name"]: x["cycles"] for x in parse_profile_reps(txt)}
        self.assertLess(r["relu"], 100)
        self.assertGreater(r["conv"], 900)


class TheSingleBlockPathIsUnchanged(unittest.TestCase):

    def test_no_iteration_blocks_returns_none(self):
        """So the caller falls back to parse_profile and nothing regresses."""
        self.assertIsNone(parse_profile_reps("no blocks here"))

    def test_the_ordinary_profile_block_still_parses(self):
        txt = ("=== MODELBLASTER_PROFILE_BEGIN ===\n"
               "dispatch_id,name,op,shape,cycles\n"
               "0,conv,conv2d_s8,N=1,1234\n"
               "=== MODELBLASTER_PROFILE_END ===")
        r = parse_profile(txt)
        self.assertEqual(r[0]["cycles"], 1234)
        self.assertIsNone(parse_profile_reps(txt))


if __name__ == "__main__":
    unittest.main()


class CVIsNotReportedForDispatchesTooShortToMeasure(unittest.TestCase):
    """rdtime ticks at 24 MHz, so a sub-microsecond dispatch is a few ticks.

    The hybrid fused_full profile reported "worst per-dispatch CV 244.8%" -- on
    a `cast_i8_to_f16` costing 36 ticks (1.5 us). The spread there is
    quantisation of the 41.7 ns clock, not instability in the kernel. Quoting it
    as the run's headline made the whole profile look untrustworthy and buried
    the figures that do matter: the 0.5-0.7 ms convolutions in the SAME run sit
    at 15-22%, which is real and worth knowing.

    The values stay in the CSV -- discarding a measurement because it is noisy
    would be worse -- but the summary line must not lead with one.
    """

    def test_a_tiny_dispatch_does_not_set_the_headline(self):
        import io as _io
        import contextlib
        # One ~36-tick op with wild spread, one ~24000-tick op with a real ~3%.
        txt = "\n".join(
            _block(i, [v_small, v_big])
            for i, (v_small, v_big) in enumerate(zip(
                [10, 60, 20, 90, 30, 15, 40],
                [24000, 26000, 24500, 25000, 24200, 24800, 24100])))
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = parse_profile_reps(txt)
        out = buf.getvalue()
        self.assertNotIn("244", out)
        # The headline must come from the op that is long enough to trust.
        self.assertIn("ticks", out)
        # Both rows keep their numbers regardless.
        self.assertEqual(len(r), 2)
        for rec in r:
            self.assertIn("cycles_cv_pct", rec)

    def test_an_all_tiny_profile_says_so_rather_than_quoting_a_number(self):
        import io as _io
        import contextlib
        txt = "\n".join(_block(i, [v]) for i, v in
                        enumerate([10, 60, 20, 90, 30, 15, 40]))
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            parse_profile_reps(txt)
        self.assertIn("timer quantisation", buf.getvalue())
