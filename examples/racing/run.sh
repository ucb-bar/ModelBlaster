#!/usr/bin/env bash
# End-to-end runner for the RiskyBird racing nets (gate detector, vision local
# planner) from a pre-extracted int8 IR in examples/<EXAMPLE_NAME>/int8/generated
# (the extract stage is skipped when the IR is on disk; there is no
# models/racing.py). See mlp/run.sh for env-var documentation; the shared body
# lives in examples/_run_lib.sh.
set -euo pipefail
export MB_DRIFT_ATOL="${MB_DRIFT_ATOL:-0}"
MODEL_NAME="${MODEL_NAME:-racing}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MODEL_NAME REPO_ROOT
source "${REPO_ROOT}/examples/_run_lib.sh"
