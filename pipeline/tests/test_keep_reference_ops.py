"""`--keep-reference-ops`: the consumer of XPU-RT's `choose_implementation`.

A build is one target, so "use implementation X for dispatch 7" has no
representation in this codegen. What does have one is the curated SWAP
decision, per op kind -- and "the curated kernel is slower than the reference
here" is exactly what the advice measures.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline import generate_kernels


def _ir(ops):
    return {"name": "probe", "input": {"tensor": "x"},
            "output": {"tensors": ["y"]}, "tensors": {}, "ops": ops,
            "quant": "int8"}


def _picks(out_dir):
    return json.load(open(Path(out_dir) / "kernel_picks.json"))["picks"]


class ThePinKeepsTheReferenceKernel(unittest.TestCase):

    #: A real graph, so the curated library actually has something to swap in
    #: -- a pin that "works" because no curated kernel existed proves nothing.
    IR_PATH = (Path(__file__).resolve().parents[2] / "build" / "k1_xpurt"
               / "yolov8_nano" / "int8" / "graph.json")
    KERNELS = Path(__file__).resolve().parents[2] / "kernels"

    def _generate(self, keep):
        if not self.IR_PATH.exists():
            self.skipTest(f"no graph at {self.IR_PATH}")
        # A curated kernel is only ACCEPTED once it has been cross-compiled and
        # verified against the reference; with no target toolchain the picker
        # correctly refuses it and falls back, so every "curated[rvv]"
        # assertion below would fail for a reason that is about the
        # environment rather than about the pin. Skip, loudly.
        cross = os.environ.get("CROSS", "")
        if not (cross and shutil.which(cross + "gcc")):
            self.skipTest(
                'no cross toolchain: set CROSS to a riscv64 prefix '
                '(eval "$(scripts/setup_spacemit_toolchain.sh)") -- without it '
                'the curated kernels cannot be verified, so none is picked')
        # The curated verify cross-compiles with repo-relative include paths,
        # so it silently fails to build -- and the picker correctly falls back
        # to the reference -- when pytest is invoked from a different
        # directory. That made this test pass from ModelBlaster/ and fail from
        # the XPU-RT root, which is a worse failure than either outcome.
        cwd = os.getcwd()
        os.chdir(Path(__file__).resolve().parents[2])
        try:
            with tempfile.TemporaryDirectory() as td:
                generate_kernels.generate(
                    str(self.IR_PATH), td, "reference", "rvv_x60",
                    quant="int8", global_curated_dir=str(self.KERNELS),
                    keep_reference_ops=keep)
                return _picks(td)
        finally:
            os.chdir(cwd)

    def test_without_the_pin_the_curated_kernel_is_swapped_in(self):
        """The control. Without it, the test below cannot distinguish 'pinned'
        from 'there was never a curated kernel to begin with'."""
        picks = self._generate(None)
        self.assertEqual(picks["maxpool2d_s8"]["source"], "curated[rvv]")

    def test_the_pinned_op_keeps_the_reference_and_says_why(self):
        picks = self._generate({"maxpool2d_s8"})
        self.assertEqual(picks["maxpool2d_s8"]["source"], "reference")
        self.assertTrue(picks["maxpool2d_s8"]["pinned_to_reference"])

    def test_only_the_pinned_op_is_affected(self):
        """A pin is per op kind, not per build. Everything else must still get
        its curated kernel -- `conv2d_batchnorm2d_silu_s8` alone is 97% of
        this model's runtime."""
        picks = self._generate({"maxpool2d_s8"})
        for op in ("conv2d_s8", "conv2d_batchnorm2d_silu_s8"):
            self.assertEqual(picks[op]["source"], "curated[rvv]",
                             f"{op} lost its curated kernel to an unrelated pin")


if __name__ == "__main__":
    unittest.main()
