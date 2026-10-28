"""kb_export.arch_opt: Blackwell -> -gencode=arch=compute_100a,code=sm_100a (nvcc 13.3 drops the suffix of -arch=sm_100a); others unchanged."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline import kb_export  # noqa: E402


class ArchOptTest(unittest.TestCase):
    def test_blackwell(self):
        for a in (100, "100", "sm_100", "sm_100a"):
            self.assertEqual(kb_export.arch_opt(a), "-gencode=arch=compute_100a,code=sm_100a")

    def test_others_keep_arch(self):
        for a, f in ((75, "sm_75"), (89, "sm_89"), ("sm_89", "sm_89"), (90, "sm_90a")):
            self.assertEqual(kb_export.arch_opt(a), f"-arch={f}")


if __name__ == "__main__":
    unittest.main()
