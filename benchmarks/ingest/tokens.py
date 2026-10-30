"""Extractors for the per-run LLM token summary.

`llm_tokens.json` is provider-agnostic: Arm B writes it from Bedrock's
per-call usage records (synthesized at end-of-run by the arm driver),
Arm C writes it from the Claude Code session JSONL. Both arms produce
the same fields:

    provider               "bedrock" or "claude_code"
    tokens_input_cached    sum of cached-read tokens billed
    tokens_input_uncached  sum of standard input tokens billed
    tokens_output          sum of output tokens billed
    n_calls                number of distinct LLM invocations
    by_model               per-model breakdown {model_id -> {...}}

`decision_log.jsonl` is Arm C only — one line per agent decision, with
fields:

    decision_id, subagent, decision_type, payload, validator_verdict

`count_rejections` counts lines where `validator_verdict != "accepted"`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def sum_input_cached(path: Path) -> Optional[int]:
    v = _load_json(path).get("tokens_input_cached")
    return int(v) if v is not None else None


def sum_input_uncached(path: Path) -> Optional[int]:
    v = _load_json(path).get("tokens_input_uncached")
    return int(v) if v is not None else None


def sum_output(path: Path) -> Optional[int]:
    v = _load_json(path).get("tokens_output")
    return int(v) if v is not None else None


def by_model(path: Path) -> dict[str, dict[str, int]]:
    return _load_json(path).get("by_model", {})


def count_lines(path: Path) -> int:
    with open(path) as f:
        return sum(1 for ln in f if ln.strip())


def count_rejections(path: Path) -> int:
    n = 0
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rec = json.loads(ln)
            if rec.get("validator_verdict", "accepted") != "accepted":
                n += 1
    return n
