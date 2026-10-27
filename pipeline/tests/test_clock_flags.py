"""X-149: ``--ncu-clock-control`` / ``--clock-guard`` of kb_graph_opt: env only, nothing set by default."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline import kb_graph_opt as go  # noqa: E402

KEYS = ("KERNELBLASTER_NCU_CLOCK_CONTROL", "KERNELBLASTER_CLOCK_GUARD")


class ClockFlagsTest(unittest.TestCase):
    def setUp(self):
        self.env = dict(os.environ)
        for k in KEYS:
            os.environ.pop(k, None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.env)

    def main(self, *flags):
        d = tempfile.mkdtemp()
        with mock.patch.object(go, "run", return_value={}), mock.patch.object(
                sys, "argv", ["kb_graph_opt", "ir", "--work", d, "--kb", d, *flags]):
            go.main()

    def test_default_sets_nothing(self):
        self.main()
        self.assertFalse([k for k in KEYS if k in os.environ])

    def test_flags_set_env(self):
        self.main("--ncu-clock-control", "none", "--clock-guard", "on")
        self.assertEqual(os.environ["KERNELBLASTER_NCU_CLOCK_CONTROL"], "none")
        self.assertEqual(os.environ["KERNELBLASTER_CLOCK_GUARD"], "1")
        self.main("--clock-guard", "off")
        self.assertNotIn("KERNELBLASTER_CLOCK_GUARD", os.environ)

    def test_bad_value_rejected(self):
        with self.assertRaises(SystemExit):
            self.main("--ncu-clock-control", "turbo")


if __name__ == "__main__":
    unittest.main()
