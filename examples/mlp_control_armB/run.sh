#!/usr/bin/env bash
# ISOLATED mlp_control tree for the ARM B (int8-drain conv, MB_DRIFT_ATOL=2) A/B.
# Shares nothing with examples/mlp_control/ -- a concurrent agent runs ARM A there and
# _run_lib.sh skips extract when graph.json exists, so a re-extract in a shared
# tree would silently corrupt the other arm's build. IR copied in verbatim from
# examples/mlp_control/fp32/generated so both arms compare the SAME graph.
set -euo pipefail
MODEL_NAME=mlp_control_armB
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MODEL_NAME REPO_ROOT
source "${REPO_ROOT}/examples/_run_lib.sh"
