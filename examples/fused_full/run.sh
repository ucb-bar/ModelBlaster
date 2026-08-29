#!/usr/bin/env bash
# End-to-end runner for FusedFull (fused_full). See mlp/run.sh for env-var
# documentation; the shared body lives in modelblaster/examples/_run_lib.sh.
set -euo pipefail
MODEL_NAME=fused_full
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MODEL_NAME REPO_ROOT
source "${REPO_ROOT}/examples/_run_lib.sh"
