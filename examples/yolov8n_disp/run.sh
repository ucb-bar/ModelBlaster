#!/usr/bin/env bash
# ISOLATED yolov8n tree for the conv2d_s8 1x1-dispatch A/B. Shares nothing
# with examples/yolov8_nano/ or examples/yolov8_nano_kopt/ (other agents own
# those); the IR was snapshotted in so stage 1 skips extract and never needs
# the model registry name.
set -euo pipefail
MODEL_NAME=yolov8n_disp
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MODEL_NAME REPO_ROOT
source "${REPO_ROOT}/examples/_run_lib.sh"
