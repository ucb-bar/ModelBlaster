#!/usr/bin/env bash
# ISOLATED dronet tree for the ARM B (int8-drain conv, MB_DRIFT_ATOL=2) A/B.
# Shares nothing with examples/dronet/ -- a concurrent agent runs ARM A there and
# _run_lib.sh skips extract when graph.json exists, so a re-extract in a shared
# tree would silently corrupt the other arm's build. IR copied in verbatim from
# examples/dronet/int8/generated so both arms compare the SAME graph.
set -euo pipefail
MODEL_NAME=dronet_armB
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MODEL_NAME REPO_ROOT
source "${REPO_ROOT}/examples/_run_lib.sh"
