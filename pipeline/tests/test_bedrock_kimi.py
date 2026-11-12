"""Bedrock settings for cheap first runs (Kimi K2.5) with a spending cap.

- BEDROCK_MODEL picks the model. MODEL is also read, but run_model_k1.sh
  reuses MODEL for the network name (ffn_block), so it cannot carry a model id.
- BEDROCK_MAX_TOKENS caps maxTokens: Kimi K2.5 allows at most 16K output
  tokens and the generator asks for 32768.
- A cap (MODELBLASTER_MAX_USD) with a model missing from pricing.yaml would
  never trip, so the client refuses to start instead.
"""
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline import bedrock_client  # noqa: E402

KIMI = "moonshotai.kimi-k2.5"


def _env(**kv):
    base = {"AWS_BEARER_TOKEN_BEDROCK": "test-token"}
    base.update(kv)
    return mock.patch.dict(os.environ, base, clear=True)


class _Reply:
    status_code = 200
    headers = {}

    def json(self):
        return {"output": {"message": {"content": [{"text": "ok"}]}},
                "usage": {"inputTokens": 1_000_000, "outputTokens": 1_000_000}}


class TestBedrockKimi(unittest.TestCase):
    def test_bedrock_model_wins_over_model(self):
        with _env(BEDROCK_MODEL=KIMI, MODEL="ffn_block"):
            self.assertEqual(bedrock_client.BedrockClient().model_id, KIMI)

    def test_model_still_used_when_bedrock_model_unset(self):
        with _env(MODEL=KIMI):
            self.assertEqual(bedrock_client.BedrockClient().model_id, KIMI)

    def test_max_tokens_capped_and_kimi_priced(self):
        with _env(BEDROCK_MODEL=KIMI, BEDROCK_MAX_TOKENS="16000",
                  MODELBLASTER_MAX_USD="100"):
            client = bedrock_client.BedrockClient()
            with mock.patch.object(bedrock_client.requests, "post",
                                   return_value=_Reply()) as post:
                client.converse(user="hi", max_tokens=32768)
            sent = json.loads(post.call_args.kwargs["data"])
            self.assertEqual(sent["inferenceConfig"]["maxTokens"], 16000)
            # 1M input at $0.60 + 1M output at $3.00 (AWS, us-east-1)
            self.assertAlmostEqual(client.budget.cumulative_cost_usd, 3.60)

    def test_cap_with_unpriced_model_refuses_to_start(self):
        with _env(BEDROCK_MODEL="some.unpriced-model", MODELBLASTER_MAX_USD="1"):
            with self.assertRaises(RuntimeError):
                bedrock_client.BedrockClient()


if __name__ == "__main__":
    unittest.main()
