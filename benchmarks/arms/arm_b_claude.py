"""Arm B-claude: BACKEND=llm + beam-search optimize, Claude Code provider.

Stub. The Bedrock and Gemini variants drive per-call LLM synthesis
through an HTTP-based client (`pipeline/bedrock_client.py`,
`pipeline/gemini_client.py`). Driving Claude Code through the same
seam means invoking `claude --print --session-id <uuid>` as a
subprocess per converse() call, with token usage harvested from the
Claude Code session JSONL at end-of-run rather than per call. That
client (`pipeline/claude_code_client.py`) lands in a separate commit
together with the driver below.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Arm B-claude driver: LLM kernel synthesis through Claude Code."
    )
    ap.add_argument("--workload", required=True)
    args = ap.parse_args(argv)
    del args  # consumed only to validate the CLI shape
    print(
        "arm_b_claude is not implemented yet. The Claude Code "
        "subprocess client (pipeline/claude_code_client.py) lands in "
        "a follow-up commit; see config/arms.yaml gated_by: "
        "phase_1_6_implementation.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
