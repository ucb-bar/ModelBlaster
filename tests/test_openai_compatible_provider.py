"""Tests for the OpenAI-compatible LLM provider."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from modelblaster.pipeline import llm_client
from modelblaster.pipeline.openai_compatible_client import OpenAICompatibleClient


def _response(payload):
    return mock.Mock(
        status_code=200,
        text="",
        headers={"x-request-id": "request-1"},
        json=mock.Mock(return_value=payload),
    )


class OpenAICompatibleProviderTests(unittest.TestCase):
    def test_factory_selects_provider(self):
        env = {
            "OPENAI_BASE_URL": "http://localhost:8000/v1",
            "OPENAI_MODEL": "test-model",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            client = llm_client.make_llm_client(provider="openai_compatible")
        self.assertIsInstance(client, OpenAICompatibleClient)

    def test_model_falls_back_to_generic_model_env(self):
        env = {
            "OPENAI_BASE_URL": "http://localhost:8000/v1",
            "MODEL": "generic-model",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            client = OpenAICompatibleClient()
        self.assertEqual(client.model_id, "generic-model")

    def test_openai_model_takes_precedence_over_generic_model(self):
        env = {
            "OPENAI_BASE_URL": "http://localhost:8000/v1",
            "OPENAI_MODEL": "provider-model",
            "MODEL": "generic-model",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            client = OpenAICompatibleClient()
        self.assertEqual(client.model_id, "provider-model")

    def test_chat_completions_request(self):
        client = OpenAICompatibleClient(
            base_url="http://localhost:8000/v1",
            model_id="model-a",
            api_key="secret",
        )
        payload = {
            "id": "completion-1",
            "choices": [{
                "message": {"content": "hello"},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 2},
            },
        }

        with mock.patch(
            "modelblaster.pipeline.openai_compatible_client.requests.post",
            return_value=_response(payload),
        ) as post:
            result = client.converse("user", system="system")

        self.assertEqual(post.call_args.args[0], "http://localhost:8000/v1/chat/completions")
        request = post.call_args.kwargs
        self.assertEqual(request["json"]["model"], "model-a")
        self.assertEqual(
            request["json"]["messages"],
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
        )
        self.assertEqual(request["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.input_tokens, 10)
        self.assertEqual(result.cache_read_input_tokens, 2)
        self.assertEqual(result.output_tokens, 4)

    def test_api_key_is_optional(self):
        client = OpenAICompatibleClient(
            base_url="http://localhost:8000/v1/chat/completions",
            model_id="model-a",
        )
        payload = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {},
        }
        with mock.patch(
            "modelblaster.pipeline.openai_compatible_client.requests.post",
            return_value=_response(payload),
        ) as post:
            client.converse("hello")
        self.assertNotIn("Authorization", post.call_args.kwargs["headers"])


if __name__ == "__main__":
    unittest.main()
