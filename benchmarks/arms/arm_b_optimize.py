"""Arm B driver: BACKEND=llm + beam-search optimize, Bedrock-billed.

Same orchestration as Arm A (shell out to examples/<m>/run.sh, parse
the harness markers into the aggregator's artifact schema), with two
differences:

  * The harness env opts into LLM kernel synthesis (`BACKEND=llm`,
    `OPTIMIZE=1`) and, on FireSim, the re-rank step
    (`FIRESIM_EVAL=1`). Beam-search knobs (BEAM, EXPANSIONS,
    ITERATIONS) come from CLI flags with the same defaults as
    `pipeline/generate_kernels.py`.
  * The Bedrock client writes per-call usage records into
    `<run-dir>/llm_calls.jsonl` via the `BEDROCK_CALLS_LOG` env var.
    After the shell pipeline finishes, this driver rolls those records
    up into `<run-dir>/llm_tokens.json` in the aggregator's
    provider-agnostic shape so dollars_equivalent prices it directly
    out of `config/pricing.yaml`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from modelblaster.benchmarks.arms import _common


ARM_ID = "B"


def _build_env(
    workload: _common.Workload,
    *,
    beam: int,
    expansions: int,
    iterations: int,
    firesim_eval: bool,
    calls_log_path: Path,
) -> dict[str, str]:
    env = os.environ.copy()
    env["MODEL_NAME"] = workload.model
    env["TARGET"] = workload.target
    env["QUANT"] = workload.quant
    env["BACKEND"] = "llm"
    env["RUNNER"] = workload.runner
    env["OPTIMIZE"] = "1"
    env["BEAM"] = str(beam)
    env["EXPANSIONS"] = str(expansions)
    env["ITERATIONS"] = str(iterations)
    env["GLOBAL_CURATED_DIR"] = str(_common.REPO_ROOT / "kernels")
    if firesim_eval:
        env["FIRESIM_EVAL"] = "1"
    env["BEDROCK_CALLS_LOG"] = str(calls_log_path)
    return env


def synthesize_llm_tokens(calls_log: Path, out_path: Path) -> None:
    """Roll up the per-call Bedrock JSONL into the aggregator's
    `llm_tokens.json` schema. Sums are computed per model_id so the
    cost extractor can apply per-model rates from pricing.yaml."""
    if not calls_log.exists():
        # No LLM calls happened (e.g. cache hits + no synthesis); still
        # emit an empty rollup so the aggregator does not flag the cell
        # as "missing tokens" when the run was legitimately free.
        empty: dict[str, Any] = {
            "schema_version": 1,
            "provider": "bedrock",
            "tokens_input_cached": 0,
            "tokens_input_uncached": 0,
            "tokens_output": 0,
            "n_calls": 0,
            "by_model": {},
        }
        out_path.write_text(json.dumps(empty, indent=2))
        return

    per_model: dict[str, dict[str, int]] = defaultdict(
        lambda: {"input_cached": 0, "input_uncached": 0, "output": 0,
                 "calls": 0}
    )
    total_in_cached = total_in_uncached = total_out = total_calls = 0
    with open(calls_log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            mid = rec.get("model_id") or "unknown"
            slot = per_model[mid]
            cached = int(rec.get("cache_read_input_tokens") or 0)
            input_t = int(rec.get("input_tokens") or 0)
            output_t = int(rec.get("output_tokens") or 0)
            # Bedrock's `inputTokens` is the billable input total
            # (cache-read tokens are billed separately at the discounted
            # cache_read rate). Subtract them out so per-rate math holds.
            uncached = max(0, input_t - cached)
            slot["input_cached"] += cached
            slot["input_uncached"] += uncached
            slot["output"] += output_t
            slot["calls"] += 1
            total_in_cached += cached
            total_in_uncached += uncached
            total_out += output_t
            total_calls += 1

    rollup: dict[str, Any] = {
        "schema_version": 1,
        "provider": "bedrock",
        "tokens_input_cached": total_in_cached,
        "tokens_input_uncached": total_in_uncached,
        "tokens_output": total_out,
        "n_calls": total_calls,
        "by_model": {k: dict(v) for k, v in per_model.items()},
    }
    out_path.write_text(json.dumps(rollup, indent=2))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Arm B driver: LLM kernel synthesis + beam-search optimize."
    )
    ap.add_argument("--workload", required=True,
                    help="workload id from config/workloads.yaml")
    ap.add_argument("--run-id", default=None,
                    help="override run-id (default: UTC timestamp)")
    ap.add_argument("--beam", type=int, default=2,
                    help="beam width per iteration (default: 2)")
    ap.add_argument("--expansions", type=int, default=3,
                    help="candidate expansions per beam (default: 3)")
    ap.add_argument("--iterations", type=int, default=2,
                    help="beam-search iterations (default: 2)")
    ap.add_argument("--no-firesim-eval", action="store_true",
                    help="suppress FIRESIM_EVAL=1 even when the workload "
                         "would normally request it")
    args = ap.parse_args(argv)

    workload = _common.load_workload(args.workload)
    if workload.blocked_by:
        print(f"workload {workload.id} is blocked_by: {workload.blocked_by}",
              file=sys.stderr)
        return 2

    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        print(
            "AWS_BEARER_TOKEN_BEDROCK not set. Source set_api_keys.sh "
            "before running.",
            file=sys.stderr,
        )
        return 2

    firesim_eval = (workload.firesim_eval
                    and workload.runner == "firesim"
                    and not args.no_firesim_eval)

    run_id = args.run_id or _common.new_run_id()
    run_dir = _common.RESULTS_DIR / ARM_ID / workload.id / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    calls_log = run_dir / "llm_calls.jsonl"

    env = _build_env(
        workload,
        beam=args.beam,
        expansions=args.expansions,
        iterations=args.iterations,
        firesim_eval=firesim_eval,
        calls_log_path=calls_log,
    )

    outcome = _common.execute_run_sh(
        arm=ARM_ID, workload=workload, env=env, run_id=run_id,
    )

    # Roll the per-call records into the aggregator's tokens schema
    # regardless of subprocess exit code — partial runs still consumed
    # real budget and the dashboard should attribute it.
    synthesize_llm_tokens(calls_log, run_dir / "llm_tokens.json")

    return _common.finalize(
        outcome, arm=ARM_ID, workload=workload, run_id=run_id,
        extra_run_json={
            "beam": args.beam,
            "expansions": args.expansions,
            "iterations": args.iterations,
            "firesim_eval": firesim_eval,
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
