#!/usr/bin/env bash
# End-to-end runner for the FULL FusedSensorNet (CNN v12), multi-input.
# See mlp_generic/run.sh for env-var docs; shared body in examples/_run_lib.sh.
set -euo pipefail
MODEL_NAME=fused_full
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MODEL_NAME REPO_ROOT
source "${REPO_ROOT}/examples/_run_lib.sh"
