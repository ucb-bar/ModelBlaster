#!/usr/bin/env bash
# End-to-end runner for YOLOv8-nano at 64x64 input (smaller variant for
# qrb5165 head-to-head). Shared body in _run_lib.sh.
set -euo pipefail
MODEL_NAME=yolov8_nano_64
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MODEL_NAME REPO_ROOT
source "${REPO_ROOT}/examples/_run_lib.sh"
