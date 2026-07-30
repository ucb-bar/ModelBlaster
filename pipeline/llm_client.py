"""Provider-agnostic LLM client factory.

The kernel-generation and beam-search optimize code paths only need
two things from their LLM client: a `.converse(user, system=..., ...)`
method returning a `ConverseResult`, and the option to log each call
to a JSONL file. Both `BedrockClient` and `GeminiClient` honor that
contract; the factory below picks one based on the `LLM_PROVIDER`
env var so the optimize loop stays provider-agnostic.

  LLM_PROVIDER=gemini   (default) -> GeminiClient
  LLM_PROVIDER=bedrock            -> BedrockClient

Per-call logging path is selected by the provider's own env var
(`GEMINI_CALLS_LOG` / `BEDROCK_CALLS_LOG`); the factory threads any
explicit `log_path` arg to whichever client it instantiates.
"""

from __future__ import annotations

import os
from typing import Optional, Protocol

from modelblaster.pipeline.bedrock_client import BedrockClient, ConverseResult


class LLMClient(Protocol):
    """The minimum interface the optimize loop relies on. Both
    `BedrockClient` and `GeminiClient` implement it; the type alias
    keeps `pipeline/generate_kernels.py` provider-neutral."""

    def converse(
        self,
        user: str,
        system: Optional[str] = ...,
        max_tokens: int = ...,
        temperature: float = ...,
        timeout: float = ...,
        phase: Optional[str] = ...,
        parent_call_id: Optional[str] = ...,
    ) -> ConverseResult: ...


def make_llm_client(
    *,
    provider: Optional[str] = None,
    log_path: Optional[str] = None,
) -> LLMClient:
    """Return the LLM client matching `provider` (or `LLM_PROVIDER`
    env var, default `gemini`). `log_path` is threaded into the
    client's constructor and overrides any provider-specific env
    fallback."""
    name = (provider or os.environ.get("LLM_PROVIDER") or "gemini").lower()
    if name == "bedrock":
        return BedrockClient(log_path=log_path)
    if name == "gemini":
        # Imported lazily so a bedrock-only run does not require
        # google-genai to be installed.
        from modelblaster.pipeline.gemini_client import GeminiClient
        return GeminiClient(log_path=log_path)
    if name in ("claude_code", "claude-code", "claudecode"):
        # Lazy import: the subprocess client checks for the `claude`
        # CLI on PATH at construction time, which would fail at module
        # import in shells that don't have Claude Code installed.
        from modelblaster.pipeline.claude_code_client import ClaudeCodeClient
        return ClaudeCodeClient(log_path=log_path)
    raise RuntimeError(
        f"unknown LLM_PROVIDER={name!r} "
        "(expected 'gemini', 'bedrock', or 'claude_code')"
    )
