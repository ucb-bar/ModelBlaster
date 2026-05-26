"""Bedrock Converse client using AWS_BEARER_TOKEN_BEDROCK.

Per-call token usage is written to a JSONL file when `log_path` (or
the BEDROCK_CALLS_LOG env var) is set. The benchmark harness uses
this to attribute costs back to specific Arm B runs without changing
the kernel-generation / optimize-loop code paths."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests


# HTTP statuses worth retrying — server-side flaps, throttling, brief outages.
_TRANSIENT_STATUSES = {408, 429, 500, 502, 503, 504}


DEFAULT_MODEL = "us.meta.llama4-maverick-17b-instruct-v1:0"
DEFAULT_REGION = "us-east-1"

# Models that Bedrock only serves through cross-region inference profiles
# (i.e. on-demand invocation by the bare model id is rejected). For these we
# silently prepend the us. prefix if missing. The Anthropic Claude family
# from sonnet-4 onward joined this list — older models (3.5 sonnet, opus,
# haiku) still accept on-demand by bare id.
_INFERENCE_PROFILE_REQUIRED = (
    "meta.llama4-",
    "anthropic.claude-sonnet-4",
    "anthropic.claude-opus-4",
    "anthropic.claude-haiku-4",
)


@dataclass
class ConverseResult:
    text: str
    stop_reason: str
    input_tokens: int
    output_tokens: int
    # Prompt-caching counters. Populated when the model + region support
    # it and the request opted in; zero otherwise.
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    # Server-assigned request id from the x-amzn-requestid header.
    # Useful as a stable key when correlating with CloudTrail / billing
    # downstream of a benchmark run.
    request_id: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _normalize_model_id(model_id: str) -> str:
    if any(model_id.startswith(p) for p in _INFERENCE_PROFILE_REQUIRED):
        return f"us.{model_id}"
    return model_id


class BudgetExceeded(RuntimeError):
    """Raised by BedrockClient.converse() when the cumulative spend
    has crossed the budget cap. The arm driver catches this and
    writes exit_status=budget_exceeded into run.json. Carries the
    cumulative cost + cap for the message so the trace is legible
    in the captured stderr."""


class BedrockClient:
    def __init__(
        self,
        model_id: Optional[str] = None,
        region: Optional[str] = None,
        token: Optional[str] = None,
        log_path: Optional[str] = None,
        max_usd: Optional[float] = None,
        pricing: Optional[dict] = None,
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
        self.log_path = log_path or os.environ.get("BEDROCK_CALLS_LOG") or None
        # Hard budget cap. The arm driver passes --max-usd as the
        # MODELBLASTER_MAX_USD env var; the client refuses further
        # calls once the cumulative cost crosses the cap. Pricing is
        # loaded lazily from config/pricing.yaml on the first call
        # that needs it, unless an explicit dict was passed in.
        env_max = os.environ.get("MODELBLASTER_MAX_USD")
        if max_usd is None and env_max:
            try:
                max_usd = float(env_max)
            except ValueError:
                max_usd = None
        self.max_usd: Optional[float] = max_usd
        self._pricing_override = pricing
        self._pricing_cache: Optional[dict] = None
        self.cumulative_cost_usd: float = 0.0
        self._budget_tripped: bool = False

    def converse(
        self,
        user: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        # Sonnet 4.6 at max_tokens=32k can stream for several minutes
        # (especially with `--cache-aware-prompt` adding context). The
        # earlier 180s default cut off mid-stream on the longer
        # generations. 600s is generous; the connection idle-timeouts
        # in urllib3 will trip first if the server actually wedged.
        timeout: float = 600.0,
        # The benchmark harness records `phase` and `parent_call_id`
        # alongside the per-call usage so a beam-search reranker call
        # can be reattributed to the kernel-synthesis call that
        # spawned it. Both are passthrough; the client does not
        # interpret them.
        phase: Optional[str] = None,
        parent_call_id: Optional[str] = None,
    ) -> ConverseResult:
        # Pre-call budget check: refuse if a previous call already
        # crossed the cap. Cheaper to fail fast than to make another
        # paid call.
        if self._budget_tripped:
            raise BudgetExceeded(
                f"cumulative cost ${self.cumulative_cost_usd:.4f} "
                f"already exceeds --max-usd ${self.max_usd:.4f}; "
                f"aborting before next call."
            )
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
        result = ConverseResult(
            text=text,
            stop_reason=data.get("stopReason", ""),
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
            cache_read_input_tokens=(
                usage.get("cacheReadInputTokenCount")
                or usage.get("cacheReadInputTokens")
                or 0
            ),
            cache_write_input_tokens=(
                usage.get("cacheWriteInputTokenCount")
                or usage.get("cacheWriteInputTokens")
                or 0
            ),
            request_id=r.headers.get("x-amzn-requestid"),
        )
        if self.log_path:
            _append_call_log(self.log_path, self.model_id, result,
                             phase=phase, parent_call_id=parent_call_id)
        # Post-call budget accumulation. If this call pushed the
        # running total past the cap, mark the client tripped so the
        # NEXT call raises BudgetExceeded -- we don't void the response
        # we just paid for, but we refuse to keep spending.
        if self.max_usd is not None:
            self.cumulative_cost_usd += self._price_call(result)
            if self.cumulative_cost_usd >= self.max_usd:
                self._budget_tripped = True
                # Loud single-line stderr marker so callers / aggregators
                # can grep for the trip event after the fact.
                import sys as _sys
                print(
                    f"MODELBLASTER_BUDGET_EXCEEDED: "
                    f"cumulative=${self.cumulative_cost_usd:.4f} "
                    f"cap=${self.max_usd:.4f} "
                    f"model={self.model_id}",
                    file=_sys.stderr, flush=True,
                )
        return result

    def _price_call(self, result: "ConverseResult") -> float:
        """Same math as benchmarks/tools/cost_monitor.price_call -- per
        the AWS prompt-caching docs, inputTokens is the non-cached
        portion only; cache_read / cache_write are billed separately."""
        rates = self._rates_for_model()
        if rates is None:
            return 0.0
        r_in = rates.get("input_uncached")
        r_cache_read = rates.get("cache_read")
        r_cache_write_5m = rates.get("cache_write_5m")
        r_out = rates.get("output")
        if r_in is None or r_out is None:
            return 0.0
        uncached = float(result.input_tokens)
        cached_read = float(result.cache_read_input_tokens)
        cached_write = float(result.cache_write_input_tokens)
        out_t = float(result.output_tokens)
        total = uncached * r_in / 1_000_000.0
        if cached_read:
            total += cached_read * (r_cache_read if r_cache_read is not None
                                    else r_in) / 1_000_000.0
        if cached_write:
            total += cached_write * (r_cache_write_5m if r_cache_write_5m is
                                     not None else r_in) / 1_000_000.0
        total += out_t * r_out / 1_000_000.0
        return total

    def _rates_for_model(self) -> Optional[dict]:
        """Lookup rates for self.model_id in pricing.yaml. Cached so
        we don't re-parse on every call. Returns None when there's no
        match or the entry is a placeholder; the budget check then
        treats this call as free (the cap can only ever be enforced
        for priced models)."""
        if self._pricing_override is not None:
            table = self._pricing_override.get("models") or {}
            entry = table.get(self.model_id)
            if entry and not entry.get("placeholder"):
                return entry
            return None
        if self._pricing_cache is None:
            try:
                import yaml as _yaml
                from pathlib import Path as _Path
                candidate = (
                    _Path(__file__).resolve().parents[1]
                    / "benchmarks" / "config" / "pricing.yaml"
                )
                if candidate.exists():
                    self._pricing_cache = _yaml.safe_load(
                        candidate.read_text()
                    )
                else:
                    self._pricing_cache = {}
            except Exception:
                self._pricing_cache = {}
        table = (self._pricing_cache or {}).get("models") or {}
        entry = table.get(self.model_id)
        if entry and not entry.get("placeholder"):
            return entry
        return None


def _append_call_log(
    path: str,
    model_id: str,
    result: ConverseResult,
    *,
    phase: Optional[str],
    parent_call_id: Optional[str],
) -> None:
    """Append one JSON record describing this call. Each line is a
    standalone JSON object so the file can be read incrementally and
    tail-rotated without parsing the whole thing."""
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "provider": "bedrock",
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
