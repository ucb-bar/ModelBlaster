#!/usr/bin/env bash
# ISOLATED yolov8n tree for the gemmini_q31 mcause=1 crash investigation.
# Shares nothing with examples/yolov8_nano/ (an E2E profile owns that one);
# the IR was snapshotted in so stage 1 skips extract.
set -euo pipefail
MODEL_NAME=yolov8n_gemdbg
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MODEL_NAME REPO_ROOT
source "${REPO_ROOT}/examples/_run_lib.sh"
