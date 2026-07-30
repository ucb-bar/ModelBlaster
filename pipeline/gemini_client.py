"""Gemini API client.

Mirrors `pipeline.bedrock_client` shape: a `GeminiClient` exposing a
single `converse()` method that returns a `ConverseResult`, plus
per-call JSONL logging when `log_path` (or the GEMINI_CALLS_LOG env
var) is set. The benchmark harness uses this for Arm B's Gemini
provider variant without touching the kernel-generation code path.

API key resolution order:
  1. constructor `api_key` arg
  2. `GOOGLE_API_KEY` env var
  3. `GEMINI_API_KEY` env var
  4. `GEMMINI_API` env var (legacy CompGen spelling)
  5. `.env` file at the repo root, if present
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from modelblaster.pipeline.bedrock_client import ConverseResult
from modelblaster.pipeline.llm_budget import BudgetTracker


DEFAULT_MODEL = "gemini-2.5-flash"

_API_KEY_ENV_VARS = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMMINI_API")


def _resolve_api_key(passed: Optional[str]) -> str:
    if passed:
        return passed
    for var in _API_KEY_ENV_VARS:
        v = os.environ.get(var)
        if v:
            return v
    # Last-resort .env discovery so an interactive caller does not need
    # to source it; matches the pattern CompGen uses.
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"):
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() in _API_KEY_ENV_VARS:
                    return v.strip().strip('"').strip("'")
    raise RuntimeError(
        "No Gemini API key. Set GOOGLE_API_KEY (or GEMINI_API_KEY) "
        "in the environment, or drop the key into a .env file at the "
        "repo root."
    )


class GeminiClient:
    """Drop-in replacement for `BedrockClient` for kernel synthesis
    and beam-search reranking. Same `converse()` signature, same
    `ConverseResult` shape, same JSONL log schema (provider field is
    `"gemini"`)."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        log_path: Optional[str] = None,
        max_usd: Optional[float] = None,
        pricing: Optional[dict] = None,
    ):
        self.model_id = model_id or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        self.api_key = _resolve_api_key(api_key)
        self.log_path = log_path or os.environ.get("GEMINI_CALLS_LOG") or None
        self._client = None  # lazy
        # Shared budget tracker -- same code path used across the
        # bedrock / gemini / claude_code clients. Reads
        # MODELBLASTER_MAX_USD env when max_usd kwarg is None.
        self.budget = BudgetTracker(max_usd=max_usd, pricing=pricing,
                                     label="gemini")

    def _ensure_client(self, *, timeout_s: float) -> Any:
        if self._client is None:
            try:
                from google import genai  # type: ignore[import]
                from google.genai import types as genai_types  # type: ignore[import]
            except ImportError as e:
                raise RuntimeError(
                    "google-genai is not installed. Install the `llm` "
                    "extra: `uv sync --extra llm`"
                ) from e
            http_options = genai_types.HttpOptions(timeout=int(timeout_s * 1000))
            self._client = genai.Client(api_key=self.api_key, http_options=http_options)
        return self._client

    def converse(
        self,
        user: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        timeout: float = 600.0,
        phase: Optional[str] = None,
        parent_call_id: Optional[str] = None,
    ) -> ConverseResult:
        # Pre-call budget check: refuse if a previous call already
        # crossed the cap.
        self.budget.check_before_call()
        client = self._ensure_client(timeout_s=timeout)
        from google.genai import types as genai_types  # type: ignore[import]

        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system:
            config_kwargs["system_instruction"] = system
        config = genai_types.GenerateContentConfig(**config_kwargs)

        response = client.models.generate_content(
            model=self.model_id,
            contents=user,
            config=config,
        )

        text = response.text or ""
        finish_reason = ""
        try:
            finish_reason = str(response.candidates[0].finish_reason or "")
        except (AttributeError, IndexError):
            pass

        usage = getattr(response, "usage_metadata", None)
        # Gemini's `prompt_token_count` is the TOTAL input (cached +
        # uncached) -- opposite of Bedrock's `inputTokens` (uncached
        # only). To keep ConverseResult.input_tokens semantically
        # identical across providers (uncached-only), subtract the
        # cached portion. cost_monitor.price_call + BudgetTracker
        # both rely on this convention.
        prompt_tokens_total = int(getattr(usage, "prompt_token_count", 0) or 0)
        output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        cache_read = int(getattr(usage, "cached_content_token_count", 0) or 0)
        input_tokens_uncached = max(0, prompt_tokens_total - cache_read)

        result = ConverseResult(
            text=text,
            stop_reason=finish_reason,
            input_tokens=input_tokens_uncached,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_write_input_tokens=0,
            request_id=getattr(response, "response_id", None),
        )
        if self.log_path:
            _append_call_log(self.log_path, self.model_id, result,
                             phase=phase, parent_call_id=parent_call_id)
        # Post-call budget accumulation; same code path as bedrock_client.
        self.budget.account_usage(
            self.model_id,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cache_read_input_tokens=result.cache_read_input_tokens,
            cache_write_input_tokens=result.cache_write_input_tokens,
        )
        return result


def _append_call_log(
    path: str,
    model_id: str,
    result: ConverseResult,
    *,
    phase: Optional[str],
    parent_call_id: Optional[str],
) -> None:
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "provider": "gemini",
        "model_id": model_id,
        "request_id": result.request_id,
        "parent_call_id": parent_call_id,
        "phase": phase,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cache_read_input_tokens": result.cache_read_input_tokens,
        "cache_write_input_tokens": result.cache_write_input_tokens,
        "stop_reason": result.stop_reason,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(record) + "\n")
