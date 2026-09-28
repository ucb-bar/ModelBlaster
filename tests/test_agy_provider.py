"""Tests for the Antigravity CLI LLM provider."""

from __future__ import annotations

import json
import os
import sys
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (os.path.join(_ROOT, "src"), _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from modelblaster.pipeline import llm_client  # noqa: E402
from modelblaster.pipeline.agy_client import AgyClient  # noqa: E402


def _probe_ok():
    return mock.Mock(returncode=0, stdout="--input-format", stderr="")


class AgyProviderTests(unittest.TestCase):
    def _client(self, **kwargs):
        with mock.patch("shutil.which", return_value="/usr/bin/agy"):
            with mock.patch("subprocess.run", return_value=_probe_ok()):
                return AgyClient(**kwargs)

    def test_agy_is_selected_by_factory(self):
        with mock.patch("shutil.which", return_value="/usr/bin/agy"):
            with mock.patch("subprocess.run", return_value=_probe_ok()):
                client = llm_client.make_llm_client(provider="agy")
        self.assertIsInstance(client, AgyClient)

    def test_null_usage_is_treated_as_zero(self):
        client = self._client()
        payload = {
            "status": "SUCCESS",
            "response": "ok",
            "conversation_id": "conv-1",
            "usage": None,
        }
        proc = mock.Mock(
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )
        with mock.patch("subprocess.run", return_value=proc):
            result = client.converse(user="hello")

        self.assertEqual(result.text, "ok")
        self.assertEqual(result.input_tokens, 0)
        self.assertEqual(result.output_tokens, 0)
        self.assertEqual(result.cache_read_input_tokens, 0)
        self.assertEqual(result.request_id, "conv-1")


if __name__ == "__main__":
    unittest.main()
