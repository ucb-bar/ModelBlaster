"""Bedrock Converse client using AWS_BEARER_TOKEN_BEDROCK."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import requests


# HTTP statuses worth retrying — server-side flaps, throttling, brief outages.
_TRANSIENT_STATUSES = {408, 429, 500, 502, 503, 504}


DEFAULT_MODEL = "us.meta.llama4-maverick-17b-instruct-v1:0"
DEFAULT_REGION = "us-east-1"

# Models that Bedrock only serves through cross-region inference profiles
# (i.e. on-demand invocation by the bare model id is rejected). For these we
# silently prepend the us. prefix if missing.
_INFERENCE_PROFILE_REQUIRED = (
    "meta.llama4-",
)


@dataclass
class ConverseResult:
    text: str
    stop_reason: str
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _normalize_model_id(model_id: str) -> str:
    if any(model_id.startswith(p) for p in _INFERENCE_PROFILE_REQUIRED):
        return f"us.{model_id}"
    return model_id


class BedrockClient:
    def __init__(
        self,
        model_id: Optional[str] = None,
        region: Optional[str] = None,
        token: Optional[str] = None,
    ):
        self.model_id = _normalize_model_id(
            model_id or os.environ.get("MODEL", DEFAULT_MODEL)
        )
        self.region = region or os.environ.get("AWS_REGION", DEFAULT_REGION)
        self.token = token or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        if not self.token:
            raise RuntimeError(
                "AWS_BEARER_TOKEN_BEDROCK not set. "
                "Source set_api_keys.sh before running."
            )
        self.endpoint = f"https://bedrock-runtime.{self.region}.amazonaws.com"

    def converse(
        self,
        user: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        timeout: float = 180.0,
    ) -> ConverseResult:
        url = f"{self.endpoint}/model/{self.model_id}/converse"
        body = {
            "messages": [{"role": "user", "content": [{"text": user}]}],
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            body["system"] = [{"text": system}]

        last_err: Optional[str] = None
        for attempt in range(3):
            r = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                },
                data=json.dumps(body),
                timeout=timeout,
            )
            if r.status_code < 400:
                break
            last_err = f"Bedrock {r.status_code}: {r.text[:500]}"
            if r.status_code not in _TRANSIENT_STATUSES:
                raise RuntimeError(last_err)
            time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"Bedrock retries exhausted: {last_err}")
        data = r.json()
        msg = data["output"]["message"]
        text = "".join(part.get("text", "") for part in msg["content"])
        usage = data.get("usage", {})
        return ConverseResult(
            text=text,
            stop_reason=data.get("stopReason", ""),
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
        )


def extract_code_block(text: str, lang: str = "c") -> str:
    """Pull the first ```lang ... ``` (or ``` ... ```) fenced block out of text."""
    fence_open = f"```{lang}"
    i = text.find(fence_open)
    if i < 0:
        i = text.find("```")
        if i < 0:
            return text.strip()
        start = text.find("\n", i) + 1
    else:
        start = text.find("\n", i) + 1
    end = text.find("```", start)
    if end < 0:
        return text[start:].strip()
    return text[start:end].strip()
