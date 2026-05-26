"""Arm B-gemini: BACKEND=llm + beam-search optimize, Gemini provider.

Identical orchestration to Arm B-bedrock; the only differences are
`LLM_PROVIDER=gemini` (so the kernel-generation factory picks the
Gemini client) and `GEMINI_CALLS_LOG` (instead of
`BEDROCK_CALLS_LOG`). Per-call usage is rolled into
`llm_tokens.json` in the same provider-agnostic shape the aggregator
expects.

Default model is `gemini-2.5-flash`; override per call with the
`GEMINI_MODEL` env var.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Optional

from modelblaster.benchmarks.arms import _common


ARM_ID = "B-gemini"
PROVIDER = "gemini"


def _build_env(
    workload: _common.Workload,
    *,
    beam: int,
    expansions: int,
    iterations: int,
    firesim_eval: bool,
    calls_log_path,
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
    if firesim_eval:
        env["FIRESIM_EVAL"] = "1"
    env["GEMINI_CALLS_LOG"] = str(calls_log_path)
    # Hard budget cap, plumbed via env so gemini_client picks it up.
    # gemini_client raises BudgetExceeded once cumulative spend
    # crosses the cap; arm driver detects via stderr marker below.
    if max_usd is not None:
        env["MODELBLASTER_MAX_USD"] = str(max_usd)
    return env


def _have_gemini_key() -> bool:
    for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMMINI_API"):
        if os.environ.get(var):
            return True
    return False


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Arm B-gemini driver: LLM kernel synthesis on Gemini."
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
    ap.add_argument("--max-usd", type=float, default=None,
                    metavar="N",
                    help="hard kill: gemini_client stops calling once "
                         "cumulative spend >= N USD. Writes "
                         "exit_status=budget_exceeded to run.json.")
    args = ap.parse_args(argv)

    workload = _common.load_workload(args.workload)
    if workload.blocked_by:
        print(f"workload {workload.id} is blocked_by: {workload.blocked_by}",
              file=sys.stderr)
        return 2

    if not _have_gemini_key():
        print(
            "No Gemini API key. Set GOOGLE_API_KEY (or GEMINI_API_KEY) "
            "in the environment, or source a .env file containing one.",
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
        calls_log_path=calls_log,
        max_usd=args.max_usd,
    )

    outcome = _common.execute_run_sh(
        arm=ARM_ID, workload=workload, env=env, run_id=run_id,
    )

    _common.synthesize_llm_tokens(
        calls_log, run_dir / "llm_tokens.json", provider=PROVIDER,
    )

    extra: dict[str, Any] = {
        "beam": args.beam,
        "expansions": args.expansions,
        "iterations": args.iterations,
        "firesim_eval": firesim_eval,
        "llm_provider": PROVIDER,
        "gemini_model": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
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
