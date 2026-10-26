"""X-145: the ``--experiments`` switch (capabilities.py): off by default (no env, no round key), only effective with
``--multiturn on``, validated."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline.capabilities import DEFAULTS, Caps  # noqa: E402

KEYS = ("KERNELBLASTER_MULTITURN", "KERNELBLASTER_EXPERIMENT_RECORD", "KERNELBLASTER_EXPERIMENT_INGEST",
        "KERNELBLASTER_EXPERIMENT_PROMPT")


class ExperimentsFlagTest(unittest.TestCase):
    def cfg(self, **kw):
        return SimpleNamespace(**{**DEFAULTS, **kw})

    def setUp(self):
        self.env = dict(os.environ)
        for k in KEYS:
            os.environ.pop(k, None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.env)

    def test_default_off_changes_nothing(self):
        c = Caps(self.cfg())
        self.assertEqual(c.round_key(), {})
        c.apply_env()
        self.assertFalse([k for k in KEYS if k in os.environ])
        Caps(self.cfg(multiturn="on")).apply_env()                    # multi-turn on, experiments off
        self.assertEqual(os.environ["KERNELBLASTER_MULTITURN"], "1")
        self.assertNotIn("KERNELBLASTER_EXPERIMENT_RECORD", os.environ)
        self.assertNotIn("KERNELBLASTER_EXPERIMENT_INGEST", os.environ)

    def test_modes(self):
        Caps(self.cfg(multiturn="on", experiments="record")).apply_env()
        self.assertEqual(os.environ["KERNELBLASTER_EXPERIMENT_RECORD"], "1")
        self.assertNotIn("KERNELBLASTER_EXPERIMENT_INGEST", os.environ)
        Caps(self.cfg(multiturn="on", experiments="ingest+prompt")).apply_env()
        self.assertEqual(os.environ["KERNELBLASTER_EXPERIMENT_INGEST"], "1")
        self.assertEqual(os.environ["KERNELBLASTER_EXPERIMENT_PROMPT"], "3")
        self.assertEqual(Caps(self.cfg(experiments="record")).round_key(), {"caps_experiments": "record"})

    def test_needs_multiturn_and_validates(self):
        Caps(self.cfg(experiments="ingest")).apply_env()               # multi-turn off: nothing is set
        self.assertFalse([k for k in KEYS if k in os.environ])
        with self.assertRaises(ValueError):
            Caps(self.cfg(experiments="sometimes"))


if __name__ == "__main__":
    unittest.main()
