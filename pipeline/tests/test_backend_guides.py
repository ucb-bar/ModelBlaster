"""Locks in which optimization guide the ime2_a100 LLM prompt receives.

ime2_a100 uses its own guide (the A100 matrix tile, registers, vlenb check and
measured costs). The K1 ime_x60 prompt must not see it: same instruction
encoding, different tile.
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline import backends as backends_mod  # noqa: E402
from pipeline import generate_kernels  # noqa: E402

A100_GUIDE = "optimization_guide_ime2_a100.md"


class TestBackendGuides(unittest.TestCase):
    def test_every_backend_guide_file_exists(self):
        for backend in backends_mod.BACKENDS.values():
            path = os.path.join(generate_kernels.PROMPTS_DIR, backend.optimization_guide)
            self.assertTrue(os.path.isfile(path), f"{backend.name}: {path} missing")

    def test_ime2_a100_prompt_gets_the_a100_guide(self):
        guide = generate_kernels._load_backend_guide(backends_mod.get("ime2_a100"))
        self.assertIn("SpaceMiT K3 A100 matrix engine", guide)
        self.assertIn(".insn r 0x2b, 3, 0x71", guide)
        self.assertIn("8 x 16 x 8", guide)
        self.assertIn("vlenb", guide)

    def test_k1_prompt_does_not_get_the_a100_guide(self):
        self.assertNotEqual(backends_mod.get("ime_x60").optimization_guide, A100_GUIDE)
        guide = generate_kernels._load_backend_guide(backends_mod.get("ime_x60"))
        self.assertNotIn("SpaceMiT K3 A100 matrix engine", guide)


if __name__ == "__main__":
    unittest.main()
