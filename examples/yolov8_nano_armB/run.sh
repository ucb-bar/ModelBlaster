#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# ARM B's defining property, exported here rather than left to the caller.
# Without it generate_kernels' verify gate rejects gemmini_tiled_conv (HARDWARE
# im2col via tiled_conv_auto, max_abs_err=2 from the mvout float-scale
# requantize) and silently falls back to gemmini_im2col_full_C -- CPU/SOFTWARE
# im2col, ~3.45x slower on conv. The intent used to live in the comment above
# while the value lived in whoever's shell ran last, which is how this arm kept
# "resetting" to software im2col.
export MB_DRIFT_ATOL="${MB_DRIFT_ATOL:-2}"
# ---------------------------------------------------------------------------
# ISOLATED yolov8_nano tree for the ARM B (int8-drain conv, MB_DRIFT_ATOL=2) A/B.
# Shares nothing with examples/yolov8_nano/ -- a concurrent agent runs ARM A there and
# _run_lib.sh skips extract when graph.json exists, so a re-extract in a shared
# tree would silently corrupt the other arm's build. IR copied in verbatim from
# examples/yolov8_nano/int8/generated so both arms compare the SAME graph.
set -euo pipefail
MODEL_NAME=yolov8_nano_armB
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MODEL_NAME REPO_ROOT
source "${REPO_ROOT}/examples/_run_lib.sh"
