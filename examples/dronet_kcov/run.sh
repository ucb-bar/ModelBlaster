#!/usr/bin/env bash
# ISOLATED dronet tree for the curated-kernel coverage work (kernel_opt_log
# experiment "kcov"). Shares nothing with examples/dronet/ -- the
# coordinator runs sharding sweeps there and _run_lib.sh skips extract when
# graph.json exists, so a re-extract in the shared tree would corrupt it. IR
# copied verbatim from examples/dronet/int8/generated so this compares the
# SAME graph.
set -euo pipefail
MODEL_NAME=dronet_kcov
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MODEL_NAME REPO_ROOT
source "${REPO_ROOT}/examples/_run_lib.sh"
