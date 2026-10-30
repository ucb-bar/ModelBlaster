"""Demo writer for visually verifying the cost monitor TUI.

Spawns a tiny producer that appends realistic-looking llm_calls.jsonl
records to a scratch file once per --interval seconds. Pair it with
the cost monitor running in a second terminal pointed at the same
file -- you'll watch tokens + $ tick up live, without making a single
real Bedrock call.

Usage (in two terminals):

  # terminal 1 -- watcher
  uv run python -m modelblaster.benchmarks.tools.cost_monitor \\
      --paths /tmp/cost_monitor_demo.jsonl --refresh-hz 4

  # terminal 2 -- producer
  uv run python -m modelblaster.benchmarks.tools.cost_monitor_demo \\
      --out /tmp/cost_monitor_demo.jsonl --interval 1 --calls 30

Ctrl-C either side to stop. The producer's calls match the schema
``pipeline.bedrock_client.append_call_log`` emits, so the math the
monitor renders matches what a real run would show -- the only
fakery is that no HTTP call leaves the box. Lets you eyeball that:

  * the summary panel updates per call,
  * the per-model breakdown sorts by spend,
  * the recent-calls table scrolls,
  * resizing the terminal reflows the layout.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path


_MODEL_POOL = [
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
]


def _one_call(i: int) -> dict:
    """Realistic mix: Sonnet draws kernel-synthesis size prompts;
    Haiku used for cheaper rerank-class calls. The first call has
    zero cache_read (cold); subsequent ones see meaningful cached
    input."""
    model = _MODEL_POOL[0] if i % 3 != 0 else _MODEL_POOL[1]
    is_sonnet = "sonnet" in model
    cached = 0 if i == 0 else random.randint(800, 1600)
    in_tot = random.randint(1500, 2500) if is_sonnet else random.randint(600, 1200)
    out_tot = random.randint(400, 800) if is_sonnet else random.randint(150, 350)
    phase = "kernel_synthesis" if i % 2 == 0 else "beam_rerank"
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "provider": "bedrock",
        "model_id": model,
        "request_id": f"demo-{i:04d}",
        "parent_call_id": None,
        "phase": phase,
        "input_tokens": in_tot,
        "output_tokens": out_tot,
        "cache_read_input_tokens": min(cached, in_tot),
        "cache_write_input_tokens": 0,
        "stop_reason": "end_turn",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Demo: append fake LLM call records so the cost "
                    "monitor can be eyeball-tested without spending money.",
    )
    ap.add_argument("--out", type=Path,
                    default=Path("/tmp/cost_monitor_demo.jsonl"),
                    help="path to append records to (default: %(default)s)")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="seconds between records (default: 1)")
    ap.add_argument("--calls", type=int, default=30,
                    help="how many records to emit then stop (default: 30; "
                         "0 = run forever)")
    ap.add_argument("--reset", action="store_true",
                    help="truncate the output file before starting")
    args = ap.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.reset and args.out.exists():
        args.out.unlink()

    i = 0
    print(f"writing to {args.out} every {args.interval}s "
          f"({args.calls if args.calls else 'unlimited'} calls)...")
    print("Ctrl-C to stop.")
    try:
        while args.calls == 0 or i < args.calls:
            rec = _one_call(i)
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            i += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    print(f"\nstopped after {i} record(s); file at {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
