"""Guard fp16 operations whose skeleton ABI differs from their fp32 name."""

from __future__ import annotations

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from pipeline import generate_skeleton  # noqa: E402


class ExplicitFp16OperationTests(unittest.TestCase):

    def test_lstm_f16_keeps_its_bespoke_state_and_weight_schema(self):
        # If absent, emit_model strips the `_f16` suffix and routes this record
        # to the fp32 `lstm` branch, which expects weight_ih + state.{h,c}.
        # Extracted lstm_f16 records instead carry weight + state[h,c].
        self.assertIn("lstm_f16", generate_skeleton._EXPLICIT_F16_OPS)


if __name__ == "__main__":
    unittest.main()
