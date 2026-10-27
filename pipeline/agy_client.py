"""Official Antigravity headless CLI adapter; never substitutes another provider."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .bedrock_client import ConverseResult
from .llm_budget import BudgetTracker


class AgyClient:
    def __init__(self, model=None, log_path=None, command=None):
        self.command = command or os.environ.get("AGY_COMMAND", "agy")
        self.model_id = model or os.environ.get("AGY_MODEL") or "agy-default"
        self.log_path = log_path or os.environ.get("AGY_CALLS_LOG")
        self.budget = BudgetTracker(label="agy")
        if shutil.which(self.command) is None:
            raise RuntimeError(f"Antigravity CLI not found: {self.command}; set AGY_COMMAND")
        probe = subprocess.run(
            [self.command, "--help"], capture_output=True, text=True, timeout=15, check=False
        )
        if probe.returncode or "--input-format" not in probe.stdout + probe.stderr:
            raise RuntimeError(
                "AGY_COMMAND must name the headless Antigravity CLI, not the IDE launcher"
            )

    def converse(
        self,
        user,
        system=None,
        max_tokens=4096,
        temperature=0.2,
        timeout=900,
        phase=None,
        parent_call_id=None,
    ):
        # The CLI has no max_tokens/temperature controls. Do not pretend to apply them.
        del max_tokens, temperature
        self.budget.check_before_call()
        prompt = f"{system}\n\n{user}" if system else user
        if len(prompt.encode()) > 64000:
            raise ValueError("AGY prompt exceeds 64 KB; summarize evidence before calling the CLI")
        args = [
            self.command,
            "--output-format",
            "json",
            "--print-timeout",
            f"{timeout}s",
            "-p",
            prompt,
        ]
        if self.model_id != "agy-default":
            args += ["--model", self.model_id]
        start = time.monotonic()
        # A proposal is a text-only task. Keep the CLI away from the source workspace
        # and do not bypass its tool permissions or alter the user's CLI settings.
        with tempfile.TemporaryDirectory(prefix="modelblaster-agy-") as cwd:
            try:
                proc = subprocess.run(
                    args,
                    stdin=subprocess.DEVNULL,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=timeout + 15,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"agy timed out after {timeout}s") from exc
        try:
            data = json.loads(proc.stdout)
        except (ValueError, TypeError):
            data = {"status": "INVALID_RESPONSE", "error": "malformed JSON"}
        if not isinstance(data, dict):
            data = {"status": "INVALID_RESPONSE", "error": "JSON result is not an object"}
        if self.log_path:
            path = Path(self.log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "provider": "agy",
                            "model_id": self.model_id,
                            "phase": phase,
                            "parent_call_id": parent_call_id,
                            "request_id": data.get("conversation_id"),
                            "usage": data.get("usage"),
                            "duration_seconds": time.monotonic() - start,
                            "status": data.get("status"),
                            "returncode": proc.returncode,
                            "error": data.get("error"),
                        }
                    )
                    + "\n"
                )
        if proc.returncode:
            raise RuntimeError(
                f"agy exited {proc.returncode}: {data.get('error') or proc.stderr[-1000:]}"
            )
        if data.get("status") != "SUCCESS":
            raise RuntimeError(f"agy failed ({data.get('status')}): {data.get('error')}")
        if not isinstance(data.get("response"), str) or not data["response"].strip():
            raise RuntimeError("agy returned an empty response")
        usage = data.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}
        result = ConverseResult(
            text=data["response"],
            stop_reason="end_turn",
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_read_input_tokens=usage.get("cache_read_tokens", 0),
            request_id=data.get("conversation_id"),
        )
        return result
