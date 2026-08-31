#!/usr/bin/env bash
# ISOLATED yolov8n tree for kernel-opt A/B measurement. Shares nothing with
# examples/yolov8_nano/ (another agent owns that one); the IR was copied in
# so stage 1 skips extract and never needs the model registry name.
set -euo pipefail
MODEL_NAME=yolov8_nano_kopt
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MODEL_NAME REPO_ROOT
source "${REPO_ROOT}/examples/_run_lib.sh"
