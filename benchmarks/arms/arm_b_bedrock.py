"""Arm B-bedrock: BACKEND=llm + beam-search optimize, AWS Bedrock provider.

Shells out to `examples/<model>/run.sh` with `BACKEND=llm`,
`OPTIMIZE=1`, `LLM_PROVIDER=bedrock`, the workload's TARGET/QUANT/
RUNNER, and `BEDROCK_CALLS_LOG` pointing at the cell's
`llm_calls.jsonl`. After the shell pipeline finishes, the per-call
records are rolled up into `llm_tokens.json` in the aggregator's
provider-agnostic shape so `dollars_equivalent` prices it directly
out of `config/pricing.yaml`.

The shared orchestration lives in `arms/_common.py`; this file is
the env-policy + CLI.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Optional

from modelblaster.benchmarks.arms import _common


ARM_ID = "B-bedrock"
PROVIDER = "bedrock"


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
    env["BEDROCK_CALLS_LOG"] = str(calls_log_path)
    # Hard kill: bedrock_client reads MODELBLASTER_MAX_USD on init and
    # raises BudgetExceeded once cumulative spend crosses the cap.
    # Without this var, the client tracks no budget and never trips.
    if max_usd is not None:
        env["MODELBLASTER_MAX_USD"] = str(max_usd)
    return env


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Arm B-bedrock driver: LLM kernel synthesis on AWS Bedrock."
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
    ap.add_argument("--runner-override", default=None,
                    choices=["spike", "firesim"],
                    help="swap the workload's runner. On hetero workloads "
                         "this routes through spike-hetero (functional-"
                         "only); use the workload's default for the "
                         "baseline capture.")
    ap.add_argument("--max-usd", type=float, default=None,
                    metavar="N",
                    help="hard kill: BedrockClient stops calling once "
                         "cumulative spend >= N USD. Writes "
                         "exit_status=budget_exceeded to run.json and the "
                         "trip event lands as a MODELBLASTER_BUDGET_EXCEEDED "
                         "line in stderr.log. Default: no cap.")
    args = ap.parse_args(argv)

    workload = _common.load_workload(args.workload)
    workload = _common.apply_runner_override(workload, args.runner_override)
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
        workload, beam=args.beam, expansions=args.expansions,
        iterations=args.iterations, firesim_eval=firesim_eval,
        calls_log_path=calls_log,
        max_usd=args.max_usd,
    )

    outcome = _common.execute_run_sh(
        arm=ARM_ID, workload=workload, env=env, run_id=run_id,
    )

    # Roll the per-call records into the aggregator's tokens schema
    # regardless of subprocess exit code -- partial runs still consumed
    # real budget and the dashboard should attribute it.
    _common.synthesize_llm_tokens(
        calls_log, run_dir / "llm_tokens.json", provider=PROVIDER,
    )

    # Budget trip: bedrock_client writes a MODELBLASTER_BUDGET_EXCEEDED
    # line to stderr right before raising. Detect that here and override
    # the run.json exit_status so the dashboard can flag the cell
    # distinctly from a generic run failure.
    extra: dict[str, Any] = {
        "beam": args.beam,
        "expansions": args.expansions,
        "iterations": args.iterations,
        "firesim_eval": firesim_eval,
        "llm_provider": PROVIDER,
    }
    if args.max_usd is not None:
        extra["max_usd"] = args.max_usd
    budget_tripped = False
    stderr_path = run_dir / "stderr.log"
    if stderr_path.exists():
        for line in stderr_path.read_text().splitlines():
            if line.startswith("MODELBLASTER_BUDGET_EXCEEDED:"):
                budget_tripped = True
                extra["budget_trip_marker"] = line
                break
    if budget_tripped:
        # Forcefully relabel the outcome so finalize() writes the
        # right exit_status. The subprocess will have exited non-zero
        # (BudgetExceeded propagated through run.sh as a Python
        # exception -> non-zero exit) but "budget_exceeded" is the
        # more useful label than "exit_1".
        outcome = _common.RunOutcome(
            out_dir=outcome.out_dir,
            returncode=outcome.returncode,
            started_at=outcome.started_at,
            ended_at=outcome.ended_at,
            wall_clock_s=outcome.wall_clock_s,
            peak_rss_mb=outcome.peak_rss_mb,
        )
        extra["exit_status_override"] = "budget_exceeded"

    rc = _common.finalize(
        outcome, arm=ARM_ID, workload=workload, run_id=run_id,
        extra_run_json=extra,
    )
    # Budget trip exits 2 (vs 0 OK / 1 generic run failure) so the
    # caller of a loop over workloads can distinguish "skip rest of
    # baseline" from "this cell broke."
    return 2 if budget_tripped else rc


if __name__ == "__main__":
    raise SystemExit(main())
