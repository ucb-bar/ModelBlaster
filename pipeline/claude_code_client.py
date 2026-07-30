"""Claude Code subprocess client.

Mirrors `pipeline.bedrock_client` / `pipeline.gemini_client` shape: a
single `converse()` method returning a `ConverseResult`, plus per-call
JSONL logging. The underlying call is a subprocess invocation of the
`claude` CLI in `--print --output-format json` mode, so each
`converse()` is one short-lived Claude Code session.

The harness uses this for Arm B-claude's per-op kernel synthesis.
Each subprocess pays Claude Code's startup cost (~1-2 seconds), so
this arm is naturally slower than Arms B-bedrock / B-gemini; the
trade is that it goes through Claude Code's prompt-caching and
billing as if a human typed the prompt, which is what the experiment
intends to measure.

Authentication comes from the user's `claude` CLI state (login
keychain or `ANTHROPIC_API_KEY`). The client does not manage auth;
a missing-or-expired login surfaces as `is_error: true` with
"Please run /login" in the response.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from modelblaster.pipeline.bedrock_client import ConverseResult
from modelblaster.pipeline.llm_budget import BudgetTracker


DEFAULT_MODEL = "sonnet"


class ClaudeCodeClient:
    """Each `converse()` shells out to `claude --print --output-format
    json --bare ...`. Token usage and `total_cost_usd` are read out of
    the JSON response and logged to the per-call JSONL when
    `log_path` (or CLAUDE_CODE_CALLS_LOG) is set."""

    def __init__(
        self,
        model: Optional[str] = None,
        log_path: Optional[str] = None,
        command: str = "claude",
        max_budget_usd: Optional[float] = None,
        max_usd: Optional[float] = None,
        extra_args: Optional[list[str]] = None,
        pricing: Optional[dict] = None,
    ):
        self.model = model or os.environ.get("CLAUDE_CODE_MODEL", DEFAULT_MODEL)
        self.command = command
        self.log_path = log_path or os.environ.get("CLAUDE_CODE_CALLS_LOG") or None
        # `max_budget_usd` (legacy) forwards to claude --max-budget-usd
        # (the CLI itself enforces). `max_usd` (new) hooks the shared
        # BudgetTracker so we get the same MODELBLASTER_BUDGET_EXCEEDED
        # marker + raise behavior as bedrock / gemini. Both can be set
        # for belt-and-suspenders.
        self.max_budget_usd = max_budget_usd
        self.extra_args = list(extra_args or [])
        self.budget = BudgetTracker(max_usd=max_usd, pricing=pricing,
                                     label="claude_code")
        if shutil.which(self.command) is None:
            raise RuntimeError(
                f"'{self.command}' is not on PATH. Install Claude Code "
                f"(https://docs.claude.com/en/docs/claude-code) or set "
                f"the `command` arg to the binary path."
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
        # Pre-call budget check: refuse if a previous call already
        # crossed the cap. Belt-and-suspenders with claude's own
        # --max-budget-usd flag.
        self.budget.check_before_call()
        # The Claude Code CLI takes a single prompt argument. There is
        # no separate user/system split in --print mode; we concatenate
        # so the system instruction lands at the top of the user turn.
        # `max_tokens` and `temperature` are not configurable from the
        # `claude --print` surface; they get whatever Claude Code's
        # current model defaults are. Document the gap rather than
        # silently dropping the kwargs.
        del max_tokens, temperature

        prompt = f"{system}\n\n{user}" if system else user
        args = [
            self.command,
            "--print",
            "--output-format", "json",
            "--bare",
            "--model", self.model,
        ]
        if self.max_budget_usd is not None:
            args += ["--max-budget-usd", str(self.max_budget_usd)]
        args += self.extra_args
        args.append(prompt)

        try:
            proc = subprocess.run(
                args, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"claude --print timed out after {timeout}s"
            ) from e
        if proc.returncode != 0:
            raise RuntimeError(
                f"claude exited {proc.returncode}: {proc.stderr[:500]}"
            )

        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"claude returned non-JSON output: {proc.stdout[:500]}"
            ) from e

        if data.get("is_error"):
            # The CLI sometimes returns an error payload while still
            # printing usage=0; treat that as a hard failure so callers
            # don't silently see "OK" with no tokens.
            raise RuntimeError(
                f"claude error: {data.get('result', '')[:500]}"
            )

        usage = data.get("usage") or {}
        text = data.get("result", "")
        # Resolve the actual model id from modelUsage if present, so the
        # JSONL records the real model rather than the alias the caller
        # passed (e.g. 'sonnet' -> 'claude-sonnet-4-6').
        model_usage = data.get("modelUsage") or {}
        resolved_model = next(iter(model_usage), self.model) if model_usage else self.model

        result = ConverseResult(
            text=text,
            stop_reason=str(data.get("stop_reason", "")),
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            cache_read_input_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
            cache_write_input_tokens=int(usage.get("cache_creation_input_tokens", 0) or 0),
            request_id=data.get("session_id"),
        )
        if self.log_path:
            _append_call_log(
                self.log_path, resolved_model, result,
                phase=phase,
                parent_call_id=parent_call_id,
                total_cost_usd=data.get("total_cost_usd"),
                duration_ms=data.get("duration_ms"),
                num_turns=data.get("num_turns"),
            )
        # Post-call budget accumulation. Claude Code returns its own
        # priced cost in `total_cost_usd`; use it directly via
        # account_prepriced rather than re-deriving from tokens +
        # pricing.yaml (the CLI already knows the negotiated rate).
        total_cost = data.get("total_cost_usd")
        if total_cost is not None:
            self.budget.account_prepriced(resolved_model,
                                           float(total_cost))
        else:
            # Fall back to token-based pricing when the CLI didn't
            # surface a cost (older claude versions).
            self.budget.account_usage(
                resolved_model,
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
    total_cost_usd: Optional[float] = None,
    duration_ms: Optional[int] = None,
    num_turns: Optional[int] = None,
) -> None:
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "provider": "claude_code",
        "model_id": model_id,
        "request_id": result.request_id,
        "parent_call_id": parent_call_id,
        "phase": phase,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cache_read_input_tokens": result.cache_read_input_tokens,
        "cache_write_input_tokens": result.cache_write_input_tokens,
        "stop_reason": result.stop_reason,
        # Claude-Code-specific telemetry: the CLI computes its own cost
        # estimate and tracks wall time + num_turns per call. We log
        # all three so the dashboard can cross-check our pricing.yaml
        # math against Claude Code's own billing view.
        "total_cost_usd_self_reported": total_cost_usd,
        "duration_ms": duration_ms,
        "num_turns": num_turns,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(record) + "\n")
