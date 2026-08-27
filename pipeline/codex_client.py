"""Codex CLI subprocess client.

Kernel synthesis for the SpaceMiT K1 work must go through the user's Codex
subscription, never Bedrock. This is the provider that makes that possible;
`LLM_PROVIDER=codex` selects it.

Shaped like `claude_code_client` because `codex exec` is the same kind of thing:
a non-interactive CLI that takes a prompt and prints a result. Differences worth
knowing:

* The prompt goes in on **stdin** (`-`), not as an argv element. Kernel prompts
  carry whole reference implementations and blow past ARG_MAX otherwise.
* The reply is read from `--output-last-message`, a file, rather than scraped
  out of the event stream. `--json` gives JSONL *events*; the final assistant
  message is what we actually want, and asking for it directly avoids guessing
  which event terminates the turn.
* The sandbox defaults to **read-only**. Generating a kernel body is a text
  task; it has no business writing to the tree. Override deliberately via
  `CODEX_SANDBOX` if some future use genuinely needs it.

There is no Bedrock fallback here, and there must never be one. If Codex is
unavailable this raises, and the caller is expected to fall back to *reference
and curated kernels* -- not to another model provider.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from modelblaster.pipeline.bedrock_client import ConverseResult
from modelblaster.pipeline.llm_budget import BudgetTracker

DEFAULT_MODEL = "gpt-5.6-sol"
PROVIDER = "codex"


class CodexClient:
    """Each `converse()` shells out to `codex exec`.

    Token counts come from the JSONL event stream when Codex reports them; they
    are left at 0 when it does not, rather than being invented. Cost is not
    computed: a Codex subscription is not billed per token, so a per-token price
    would be fiction. `BudgetTracker` is still constructed so the budget plumbing
    behaves uniformly across providers.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        log_path: Optional[str] = None,
        command: str = "codex",
        sandbox: Optional[str] = None,
        cwd: Optional[str] = None,
        max_usd: Optional[float] = None,
        extra_args: Optional[list[str]] = None,
        pricing: Optional[dict] = None,
    ):
        self.model = model or os.environ.get("CODEX_MODEL", DEFAULT_MODEL)
        self.model_id = self.model  # generate_kernels logs client.model_id
        self.command = command
        self.log_path = log_path or os.environ.get("CODEX_CALLS_LOG") or None
        self.sandbox = sandbox or os.environ.get("CODEX_SANDBOX", "read-only")
        self.cwd = cwd or os.environ.get("CODEX_CWD") or os.getcwd()
        self.extra_args = list(extra_args or [])
        self.budget = BudgetTracker(max_usd=max_usd, pricing=pricing,
                                    label=PROVIDER)
        if shutil.which(self.command) is None:
            raise RuntimeError(
                f"'{self.command}' is not on PATH. Install the Codex CLI, or "
                f"pass command=<path>. This workflow must not fall back to "
                f"another provider."
            )

    def converse(
        self,
        user: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        timeout: float = 900.0,
        phase: Optional[str] = None,
        parent_call_id: Optional[str] = None,
    ) -> ConverseResult:
        self.budget.check_before_call()
        # codex exec has no separate system channel and no knobs for
        # max_tokens/temperature. Concatenate so the system instruction leads,
        # and document the dropped kwargs rather than pretending they applied.
        del max_tokens, temperature
        prompt = f"{system}\n\n{user}" if system else user

        out_fd, out_path = tempfile.mkstemp(prefix="codex_msg_", suffix=".txt")
        os.close(out_fd)
        args = [
            self.command, "exec",
            "--json",
            "--color", "never",
            "--skip-git-repo-check",
            "--sandbox", self.sandbox,
            "--model", self.model,
            "--cd", self.cwd,
            "--output-last-message", out_path,
        ] + self.extra_args + ["-"]

        t0 = time.time()
        try:
            proc = subprocess.run(args, input=prompt, capture_output=True,
                                  text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            os.unlink(out_path)
            raise RuntimeError(f"codex exec timed out after {timeout}s") from e

        try:
            if proc.returncode != 0:
                raise RuntimeError(
                    f"codex exec exited {proc.returncode}: {proc.stderr[:500]}")
            try:
                with open(out_path, encoding="utf-8") as f:
                    text = f.read()
            except OSError as e:
                raise RuntimeError(
                    f"codex exec wrote no last message: {proc.stderr[:500]}"
                ) from e
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass

        if not text.strip():
            raise RuntimeError("codex exec returned an empty message")

        in_tok, out_tok = _parse_usage(proc.stdout)
        duration_ms = int((time.time() - t0) * 1000)

        result = ConverseResult(
            text=text,
            stop_reason="end_turn",
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_read_input_tokens=0,
            cache_write_input_tokens=0,
            request_id=None,
        )
        if self.log_path:
            _append_codex_call_log(self.log_path, self.model, result,
                                   phase=phase, parent_call_id=parent_call_id,
                                   duration_ms=duration_ms)
        return result


def _parse_usage(stdout: str) -> tuple[int, int]:
    """Pull the last token-usage numbers out of the JSONL event stream.

    Codex's event schema is not stable across versions, so this looks for any
    object carrying a usage-shaped payload and takes the last one. Unknown shape
    means (0, 0) -- reporting zero is honest; guessing is not.
    """
    in_tok = out_tok = 0
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        usage = None
        for key in ("usage", "token_usage"):
            if isinstance(ev.get(key), dict):
                usage = ev[key]
                break
            msg = ev.get("msg")
            if isinstance(msg, dict) and isinstance(msg.get(key), dict):
                usage = msg[key]
                break
        if not usage:
            continue
        for k in ("input_tokens", "prompt_tokens"):
            if isinstance(usage.get(k), int):
                in_tok = usage[k]
                break
        for k in ("output_tokens", "completion_tokens"):
            if isinstance(usage.get(k), int):
                out_tok = usage[k]
                break
    return in_tok, out_tok


def _append_codex_call_log(path, model_id, result, *, phase, parent_call_id,
                           duration_ms):
    """Append one JSONL record for a Codex call.

    Deliberately NOT bedrock_client._append_call_log: that helper hardcodes
    "provider": "bedrock", so reusing it would stamp every Codex call as a
    Bedrock call. For a workflow whose whole point is that kernels came from
    Codex and not from Bedrock, a mislabelled audit trail is worse than none.
    """
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "provider": PROVIDER,
        "model_id": model_id,
        "request_id": result.request_id,
        "parent_call_id": parent_call_id,
        "phase": phase,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cache_read_input_tokens": result.cache_read_input_tokens,
        "cache_write_input_tokens": result.cache_write_input_tokens,
        "stop_reason": result.stop_reason,
        "duration_ms": duration_ms,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(record) + "\n")
