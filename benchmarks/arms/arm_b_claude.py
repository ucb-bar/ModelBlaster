"""Arm B-claude: BACKEND=llm + beam-search optimize, Claude Code provider.

Same orchestration as Arm B-bedrock / B-gemini. The LLM provider is
Claude Code (`claude --print --output-format json --bare`); each
`converse()` call in `pipeline.generate_kernels` spawns one
subprocess. The per-call response JSON carries token usage and
self-reported cost; those land in `<run-dir>/llm_calls.jsonl` and are
rolled up into `llm_tokens.json` by the shared synthesizer.

Authentication for the `claude` CLI is whatever it's configured with
(login keychain or ANTHROPIC_API_KEY). The driver does not manage
that; a missing login surfaces as a run error.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import Any, Optional

from modelblaster.benchmarks.arms import _common


ARM_ID = "B-claude"
PROVIDER = "claude_code"


def _build_env(
    workload: _common.Workload,
    *,
    beam: int,
    expansions: int,
    iterations: int,
    firesim_eval: bool,
    calls_log_path,
    claude_code_model: str,
    max_usd: Optional[float] = None,
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
    env["LLM_PROVIDER"] = PROVIDER
    env["CLAUDE_CODE_MODEL"] = claude_code_model
    if firesim_eval:
        env["FIRESIM_EVAL"] = "1"
    env["CLAUDE_CODE_CALLS_LOG"] = str(calls_log_path)
    # Hard budget cap via the shared BudgetTracker plumbed through
    # MODELBLASTER_MAX_USD. claude_code_client uses total_cost_usd
    # from the CLI response (pre-priced by Claude Code) and accumulates
    # via account_prepriced.
    if max_usd is not None:
        env["MODELBLASTER_MAX_USD"] = str(max_usd)
    return env


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Arm B-claude driver: LLM kernel synthesis through Claude Code."
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
    ap.add_argument("--claude-model", default="sonnet",
                    help="--model arg passed to `claude --print` (alias "
                         "like 'sonnet'/'opus'/'haiku' or full id like "
                         "'claude-sonnet-4-6'; default: sonnet)")
    ap.add_argument("--max-usd", type=float, default=None,
                    metavar="N",
                    help="hard kill: claude_code_client stops calling "
                         "once cumulative spend >= N USD. Writes "
                         "exit_status=budget_exceeded to run.json.")
    args = ap.parse_args(argv)

    workload = _common.load_workload(args.workload)
    if workload.blocked_by:
        print(f"workload {workload.id} is blocked_by: {workload.blocked_by}",
              file=sys.stderr)
        return 2

    if shutil.which("claude") is None:
        print(
            "`claude` CLI not on PATH. Install Claude Code "
            "(https://docs.claude.com/en/docs/claude-code) before "
            "running Arm B-claude.",
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
        workload, beam=args.beam, expansions=args.expansions,
        iterations=args.iterations, firesim_eval=firesim_eval,
        calls_log_path=calls_log, claude_code_model=args.claude_model,
        max_usd=args.max_usd,
    )

    auto_session = _common.maybe_auto_session(ARM_ID, workload, run_id)
    try:
        outcome = _common.execute_run_sh(
            arm=ARM_ID, workload=workload, env=env, run_id=run_id,
        )
    finally:
        _common.end_auto_session(auto_session)

    _common.synthesize_llm_tokens(
        calls_log, run_dir / "llm_tokens.json", provider=PROVIDER,
    )

    extra: dict[str, Any] = {
        "beam": args.beam,
        "expansions": args.expansions,
        "iterations": args.iterations,
        "firesim_eval": firesim_eval,
        "llm_provider": PROVIDER,
        "claude_code_model": args.claude_model,
    }
    if args.max_usd is not None:
        extra["max_usd"] = args.max_usd
    budget_tripped = _common.detect_budget_trip(run_dir, extra)
    rc = _common.finalize(
        outcome, arm=ARM_ID, workload=workload, run_id=run_id,
        extra_run_json=extra,
    )
    return 2 if budget_tripped else rc


if __name__ == "__main__":
    raise SystemExit(main())
