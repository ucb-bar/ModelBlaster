"""Minimal OpenAI-compatible chat-completions client."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

from modelblaster.pipeline.bedrock_client import ConverseResult
from modelblaster.pipeline.llm_budget import BudgetTracker


_TRANSIENT_STATUSES = {408, 429, 500, 502, 503, 504}


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        log_path: Optional[str] = None,
        max_usd: Optional[float] = None,
        pricing: Optional[dict] = None,
    ):
        self.base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or ""
        ).rstrip("/")
        self.model_id = (
            model_id
            or os.environ.get("OPENAI_MODEL")
            or os.environ.get("MODEL")
            or ""
        )
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self.log_path = log_path or os.environ.get("OPENAI_CALLS_LOG") or None

        if not self.base_url:
            raise RuntimeError("OPENAI_BASE_URL is required for openai_compatible")
        if not self.model_id:
            raise RuntimeError(
                "OPENAI_MODEL or MODEL is required for openai_compatible"
            )

        self.endpoint = (
            self.base_url
            if self.base_url.endswith("/chat/completions")
            else self.base_url + "/chat/completions"
        )
        self.budget = BudgetTracker(
            max_usd=max_usd,
            pricing=pricing,
            label="openai_compatible",
        )

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
        self.budget.check_before_call()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        last_error = None
        for attempt in range(3):
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=body,
                timeout=timeout,
            )
            if response.status_code < 400:
                break
            last_error = (
                f"OpenAI-compatible {response.status_code}: "
                f"{response.text[:500]}"
            )
            if response.status_code not in _TRANSIENT_STATUSES:
                raise RuntimeError(last_error)
            time.sleep(2 ** attempt)
        else:
            raise RuntimeError(
                f"OpenAI-compatible retries exhausted: {last_error}"
            )

        data = response.json()
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "OpenAI-compatible response is missing choices[0].message.content"
            ) from exc

        if isinstance(content, list):
            text = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )
        else:
            text = str(content or "")

        usage = data.get("usage") or {}
        prompt_total = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        cached_tokens = int(details.get("cached_tokens") or 0)

        result = ConverseResult(
            text=text,
            stop_reason=str(choice.get("finish_reason") or ""),
            input_tokens=max(0, prompt_total - cached_tokens),
            output_tokens=completion_tokens,
            cache_read_input_tokens=cached_tokens,
            cache_write_input_tokens=0,
            request_id=(
                data.get("id")
                or response.headers.get("x-request-id")
                or response.headers.get("request-id")
            ),
        )

        if self.log_path:
            _append_call_log(
                self.log_path,
                self.model_id,
                result,
                phase=phase,
                parent_call_id=parent_call_id,
            )

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
        "provider": "openai_compatible",
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
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a") as file:
        file.write(json.dumps(record) + "\n")
