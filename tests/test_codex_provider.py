"""The K1 kernel path must go through Codex, and must never reach Bedrock.

This is a hard project constraint, not a preference, so it is pinned by tests
rather than by documentation. Two distinct things are checked:

1. `LLM_PROVIDER=codex` actually selects the Codex client.
2. There is no path -- silent or otherwise -- from an unavailable Codex to
   Bedrock. When Codex is missing the correct behaviour is a loud failure, after
   which the caller falls back to *reference and curated kernels*, which are
   deterministic artifacts already in the tree, not to a different model.

The mislabelling test exists because bedrock_client._append_call_log hardcodes
`"provider": "bedrock"`. Reusing that helper would have stamped every Codex call
as a Bedrock call, and for a workflow whose entire claim is "these kernels came
from Codex", a wrong audit trail is worse than no audit trail.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (os.path.join(_ROOT, "src"), _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from modelblaster.pipeline import llm_client  # noqa: E402
from modelblaster.pipeline.codex_client import (  # noqa: E402
    CodexClient, _parse_usage,
)


class ProviderSelectionTests(unittest.TestCase):
    def test_codex_is_selected_by_env(self):
        with mock.patch.dict(os.environ, {"LLM_PROVIDER": "codex"}, clear=False):
            with mock.patch("shutil.which", return_value="/usr/bin/codex"):
                c = llm_client.make_llm_client()
        self.assertIsInstance(c, CodexClient)

    def test_codex_is_selected_by_argument(self):
        with mock.patch("shutil.which", return_value="/usr/bin/codex"):
            c = llm_client.make_llm_client(provider="codex")
        self.assertIsInstance(c, CodexClient)

    def test_unknown_provider_raises_rather_than_defaulting(self):
        """A typo must not silently land on some other provider."""
        with self.assertRaises(RuntimeError) as ctx:
            llm_client.make_llm_client(provider="codexx")
        self.assertIn("codexx", str(ctx.exception))

    def test_missing_codex_binary_fails_loudly(self):
        with mock.patch("shutil.which", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                llm_client.make_llm_client(provider="codex")
        msg = str(ctx.exception).lower()
        self.assertIn("codex", msg)
        self.assertNotIn("bedrock", msg,
                         "an unavailable Codex must not mention falling back")

    def test_no_bedrock_fallback_path_exists_in_the_factory(self):
        """Static guard: only an explicit 'bedrock' request may reach Bedrock."""
        import inspect
        src = inspect.getsource(llm_client.make_llm_client)
        for line in src.splitlines():
            if "BedrockClient" in line and "import" not in line:
                # the only construction must be guarded by name == "bedrock"
                self.assertIn("return", line)
        self.assertIn('name == "bedrock"', src)


class CodexClientTests(unittest.TestCase):
    def _client(self, **kw):
        with mock.patch("shutil.which", return_value="/usr/bin/codex"):
            return CodexClient(**kw)

    def test_defaults_to_a_read_only_sandbox(self):
        """Generating a kernel body is a text task; it must not write the tree."""
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CODEX_SANDBOX", None)
            c = self._client()
        self.assertEqual(c.sandbox, "read-only")

    def test_exposes_model_id_for_the_kernel_logger(self):
        c = self._client(model="some-model")
        self.assertEqual(c.model_id, "some-model")

    def test_prompt_goes_on_stdin_not_argv(self):
        """Kernel prompts embed whole reference implementations; argv overflows."""
        c = self._client()
        big = "X" * 300_000
        captured = {}

        def fake_run(args, **kwargs):
            captured["args"] = args
            captured["input"] = kwargs.get("input")
            with open(args[args.index("--output-last-message") + 1], "w") as f:
                f.write("ok")
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch("subprocess.run", side_effect=fake_run):
            c.converse(user=big)
        self.assertIn(big, captured["input"])
        self.assertTrue(all(big not in a for a in captured["args"]))
        self.assertEqual(captured["args"][-1], "-")

    def test_empty_reply_is_an_error(self):
        c = self._client()

        def fake_run(args, **kwargs):
            with open(args[args.index("--output-last-message") + 1], "w") as f:
                f.write("   ")
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch("subprocess.run", side_effect=fake_run):
            with self.assertRaises(RuntimeError):
                c.converse(user="hi")

    def test_nonzero_exit_is_an_error(self):
        c = self._client()
        with mock.patch("subprocess.run",
                        return_value=mock.Mock(returncode=3, stdout="",
                                               stderr="boom")):
            with self.assertRaises(RuntimeError):
                c.converse(user="hi")

    def test_call_log_records_codex_not_bedrock(self):
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "calls.jsonl")
            c = self._client(log_path=log)

            def fake_run(args, **kwargs):
                with open(args[args.index("--output-last-message") + 1], "w") as f:
                    f.write("kernel body")
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch("subprocess.run", side_effect=fake_run):
                c.converse(user="hi", phase="synth:matmul")
            rec = json.loads(open(log).read().strip())
        self.assertEqual(rec["provider"], "codex")
        self.assertNotEqual(rec["provider"], "bedrock")
        self.assertEqual(rec["phase"], "synth:matmul")


class UsageParsingTests(unittest.TestCase):
    def test_reads_usage_from_the_event_stream(self):
        stream = "\n".join([
            "not json",
            json.dumps({"msg": {"usage": {"input_tokens": 11,
                                          "output_tokens": 22}}}),
        ])
        self.assertEqual(_parse_usage(stream), (11, 22))

    def test_unknown_shape_reports_zero_rather_than_guessing(self):
        self.assertEqual(_parse_usage('{"something": "else"}'), (0, 0))
        self.assertEqual(_parse_usage(""), (0, 0))

    def test_last_usage_event_wins(self):
        stream = "\n".join([
            json.dumps({"usage": {"input_tokens": 1, "output_tokens": 2}}),
            json.dumps({"usage": {"input_tokens": 30, "output_tokens": 40}}),
        ])
        self.assertEqual(_parse_usage(stream), (30, 40))


if __name__ == "__main__":
    unittest.main()
