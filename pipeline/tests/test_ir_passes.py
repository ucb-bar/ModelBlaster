"""The IR rewrite passes, and the check that they still agree with the IR they rewrite.

These seven passes (ir_cse, ir_concat, ir_lmsplit, ir_vperm, ir_vlayout, ir_batch,
ir_regbatch) rewrite graph.json between `extract_graph` and `generate_skeleton`.  Until
2026-09-22 they lived in a *different* repository from the extractor and the codegen, with
no test on either side that the two still agreed -- so an op kind renamed here and not
there, or there and not here, was discoverable only by a board run producing wrong tokens.
This file is the check that shape of drift cannot pass.

Two things are asserted, and the second is the one the move was for:

  1. every pass's own `selftest()` still returns 0.  Each is a hand-built toy graph with
     an assertion per rule, and they are what the lab scripts run before every real graph.

  2. EVERY OP KIND THESE PASSES NAME IS AN OP KIND THIS REPOSITORY HAS A KernelSpec FOR.
     A pass that keys on `permute4_s8` while reference_kernels.py has renamed it does not
     fail -- ir_cse simply collapses nothing, ir_vperm flips nothing, and the graph goes
     to the board a little slower and entirely plausible.  That is the failure this test
     exists to make loud.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

# The passes import their siblings relatively (`from . import ir_cse`, the way extract_q16
# imports extract_graph), so the repo root is all this file needs -- no PYTHONPATH, no
# assumption that the checkout directory is named `modelblaster`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline import (ir_batch, ir_concat, ir_cse, ir_lmsplit,   # noqa: E402
                      ir_regbatch, ir_vlayout, ir_vperm)
from pipeline.reference_kernels import KERNEL_SPECS               # noqa: E402

PASSES = (ir_cse, ir_concat, ir_lmsplit, ir_vperm, ir_vlayout, ir_batch, ir_regbatch)
_KIND = re.compile(r'"([a-z][a-z0-9_]*_s8)"')


class TestIRPassSelftests(unittest.TestCase):
    def test_every_pass_selftest_passes(self):
        for mod in PASSES:
            with self.subTest(pass_=mod.__name__):
                self.assertEqual(mod.selftest(), 0)


class TestIRPassOpKinds(unittest.TestCase):
    def test_named_op_kinds_exist_in_this_repo(self):
        """Read the kinds out of the source, not out of a list somebody has to maintain."""
        seen = 0
        for mod in PASSES:
            src = Path(mod.__file__).read_text()
            for kind in sorted(set(_KIND.findall(src))):
                seen += 1
                with self.subTest(pass_=mod.__name__, kind=kind):
                    self.assertIn(kind, KERNEL_SPECS,
                                  "%s keys on op kind %r, which reference_kernels.py no "
                                  "longer defines -- the pass would silently do nothing"
                                  % (mod.__name__, kind))
        self.assertGreater(seen, 0, "no op kinds found: the extractor pattern has changed")

    def test_ir_batch_row_key_covers_only_real_kinds(self):
        """ir_batch REFUSES a kind it has no rule for, so its table is load-bearing."""
        for kind in ir_batch.ROW_KEY:
            if kind == "view":          # a graph-level reshape, not a dispatched kernel
                continue
            with self.subTest(kind=kind):
                self.assertIn(kind, KERNEL_SPECS)


if __name__ == "__main__":
    unittest.main()
