#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# MB_DRIFT_ATOL=2 is the DEFINING property of arm B and is exported here, not
# left to the caller. Without it generate_kernels' verify gate rejects
# gemmini_tiled_conv (HARDWARE im2col via tiled_conv_auto, max_abs_err=2 from
# the mvout float-scale requantize) and silently falls back to
# gemmini_im2col_full_C -- which is CPU/SOFTWARE im2col and ~3.45x slower on
# conv. That fallback is why this arm kept "resetting" to sw im2col: the intent
# lived in a comment while the env var lived in whoever's shell last ran it.
# Override with MB_DRIFT_ATOL=0 for an exact-kernel build.
export MB_DRIFT_ATOL="${MB_DRIFT_ATOL:-2}"
# ---------------------------------------------------------------------------
# ISOLATED dronet tree for the ARM B (int8-drain conv, MB_DRIFT_ATOL=2) A/B.
# Shares nothing with examples/dronet/ -- a concurrent agent runs ARM A there and
# _run_lib.sh skips extract when graph.json exists, so a re-extract in a shared
# tree would silently corrupt the other arm's build. IR copied in verbatim from
# examples/dronet/int8/generated so both arms compare the SAME graph.
set -euo pipefail
MODEL_NAME=dronet_ohT1
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MODEL_NAME REPO_ROOT
source "${REPO_ROOT}/examples/_run_lib.sh"
