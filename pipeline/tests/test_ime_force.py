"""`MB_IME_FORCE=1`: the blanket-IME build, and the fact that it is opt-in.

The default picker is table-guided -- an (op, shape) reaches the K1 matrix
engine only where a MEASURED table says the engine beats the RVV kernel that
would otherwise run (`pipeline/ime_cost.py`). `MB_IME_FORCE=1` builds the other
deployment: the accelerator turned on for everything that has a kernel for it,
tables ignored. It exists so that deployment can be MEASURED rather than
argued about, so what these tests pin down is (a) that the default does not
move, (b) that the switch really does place a table-excluded op on the engine,
and (c) that the resulting build says in its own metadata that it is not
table-guided.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline import generate_kernels, ime_cost


class TheSwitchIsOffUnlessAskedFor(unittest.TestCase):

    def setUp(self):
        self._saved = os.environ.pop(ime_cost.FORCE_ENV, None)

    def tearDown(self):
        os.environ.pop(ime_cost.FORCE_ENV, None)
        if self._saved is not None:
            os.environ[ime_cost.FORCE_ENV] = self._saved

    def test_unset_is_table_guided(self):
        self.assertFalse(ime_cost.force_requested())

    def test_the_usual_ways_of_saying_no_mean_no(self):
        for off in ("0", "", "false", "False", "no", "off"):
            os.environ[ime_cost.FORCE_ENV] = off
            self.assertFalse(ime_cost.force_requested(), f"{off!r} turned it on")

    def test_asking_for_it_turns_it_on(self):
        for on in ("1", "yes", "true"):
            os.environ[ime_cost.FORCE_ENV] = on
            self.assertTrue(ime_cost.force_requested(), f"{on!r} left it off")

    def test_the_note_states_the_placement_constraint(self):
        """Forcing the kernel does not make `smt.vmadot` legal on cluster 1 --
        it raises SIGILL there (artifacts/ime_isa_probe/FINDINGS.md). A build
        metadata note that omits that is how a forced build ends up scheduled
        on harts 4-7."""
        note = ime_cost.FORCE_NOTE
        self.assertIn("SIGILL", note)
        self.assertIn("cluster 0", note)
        self.assertIn("not table-guided", note.lower())


class TheForcedBuildPutsTableLosersOnTheEngine(unittest.TestCase):
    """End to end through the picker, on the deployed graph.

    `conv2d_s8` is the op that makes this observable: the deployed
    yolov8_nano_64x96 has six of them (the detect heads), an IME kernel exists
    for the op kind, and the measured table excludes it -- so under the default
    it is `curated[rvv]` and under the switch it must be `curated[ime]`.

    Kernel verification is switched off here because what is under test is the
    PICK, not the kernel: verifying cross-compiles every candidate and needs a
    riscv toolchain, which would make this a test about the environment.
    """

    IR_PATH = (Path(__file__).resolve().parents[2] / "build" / "k1_xpurt"
               / "yolov8_nano_64x96" / "int8" / "graph.json")
    KERNELS = Path(__file__).resolve().parents[2] / "kernels"

    def _generate(self, force):
        if not self.IR_PATH.exists():
            self.skipTest(f"no graph at {self.IR_PATH}")
        saved = {k: os.environ.get(k)
                 for k in (ime_cost.FORCE_ENV, "MODELBLASTER_CURATED_VERIFY")}
        os.environ["MODELBLASTER_CURATED_VERIFY"] = "0"
        os.environ.pop(ime_cost.FORCE_ENV, None)
        if force:
            os.environ[ime_cost.FORCE_ENV] = "1"
        cwd = os.getcwd()
        os.chdir(Path(__file__).resolve().parents[2])
        try:
            with tempfile.TemporaryDirectory() as td:
                generate_kernels.generate(
                    str(self.IR_PATH), td, "reference", "ime_x60",
                    quant="int8", global_curated_dir=str(self.KERNELS))
                return json.load(open(Path(td) / "kernel_picks.json"))
        finally:
            os.chdir(cwd)
            for k, v in saved.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v

    def test_the_default_leaves_the_table_loser_on_rvv(self):
        """The control: without it, the test below cannot tell 'forced' from
        'it was going to be picked anyway'."""
        doc = self._generate(force=False)
        self.assertEqual(doc["picks"]["conv2d_s8"]["source"], "curated[rvv]")
        self.assertIn("ime_skipped_reason", doc["picks"]["conv2d_s8"])
        self.assertNotIn("ime_force", doc)

    def test_the_fused_conv_is_on_the_engine_either_way(self):
        """The table WANTS the fused conv on the engine, so the switch must not
        be what puts it there -- otherwise 'forced' and 'table-guided' would be
        indistinguishable on the op that dominates this net."""
        for force in (False, True):
            doc = self._generate(force=force)
            self.assertEqual(
                doc["picks"]["conv2d_batchnorm2d_silu_s8"]["source"],
                "curated[ime]", f"force={force}")

    def test_forcing_moves_the_table_loser_onto_the_engine(self):
        doc = self._generate(force=True)
        pick = doc["picks"]["conv2d_s8"]
        self.assertEqual(pick["source"], "curated[ime]")
        self.assertIn("ime_forced_over_table", pick,
                      "the pick does not record that a table said otherwise")

    def test_the_forced_build_declares_itself_not_table_guided(self):
        doc = self._generate(force=True)
        self.assertIs(doc["table_guided"], False)
        self.assertIs(doc["ime_force"], True)
        self.assertEqual(doc["ime_force_note"], ime_cost.FORCE_NOTE)
        self.assertIn("conv2d_s8", doc["ime_forced_ops"])
        self.assertIn("conv2d_batchnorm2d_silu_s8", doc["ime_forced_ops"])
        self.assertEqual(doc["ime_forced_over_table_ops"], ["conv2d_s8"])

    def test_forcing_does_not_invent_kernels_for_ops_that_have_none(self):
        """There is no IME kernel for maxpool/concat/upsample, and the switch
        must not pretend otherwise -- it removes a GUARD, it does not add
        implementations."""
        doc = self._generate(force=True)
        for op in ("maxpool2d_s8", "upsample_nearest_s8", "add_s8"):
            self.assertEqual(doc["picks"][op]["source"], "curated[rvv]", op)


if __name__ == "__main__":
    unittest.main()
