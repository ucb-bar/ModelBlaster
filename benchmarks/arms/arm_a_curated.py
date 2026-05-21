"""Arm A driver: deterministic pipeline, curated kernels, no LLM.

For a given workload row in `config/workloads.yaml`, invokes the
matching `examples/<model>/run.sh` with BACKEND=reference and the
workload's TARGET/QUANT/RUNNER, then writes the aggregator's per-run
artifacts under `benchmarks/results/A/<workload-id>/<run-id>/`. The
heavy lifting lives in `_common`; this file is just the env policy
and the CLI.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from modelblaster.benchmarks.arms import _common


ARM_ID = "A"


def _build_env(workload: _common.Workload) -> dict[str, str]:
    env = os.environ.copy()
    env["MODEL_NAME"] = workload.model
    env["TARGET"] = workload.target
    env["QUANT"] = workload.quant
    env["BACKEND"] = "reference"
    env["RUNNER"] = workload.runner
    env["OPTIMIZE"] = "0"
    env["GLOBAL_CURATED_DIR"] = str(_common.REPO_ROOT / "kernels")
    return env


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Arm A driver: curated kernels, no LLM."
    )
    ap.add_argument("--workload", required=True,
                    help="workload id from config/workloads.yaml")
    ap.add_argument("--run-id", default=None,
                    help="override run-id (default: UTC timestamp)")
    args = ap.parse_args(argv)

    workload = _common.load_workload(args.workload)
    if workload.blocked_by:
        print(f"workload {workload.id} is blocked_by: {workload.blocked_by}",
              file=sys.stderr)
        return 2

    run_id = args.run_id or _common.new_run_id()
    outcome = _common.execute_run_sh(
        arm=ARM_ID, workload=workload, env=_build_env(workload), run_id=run_id,
    )
    return _common.finalize(
        outcome, arm=ARM_ID, workload=workload, run_id=run_id,
    )


if __name__ == "__main__":
    raise SystemExit(main())
