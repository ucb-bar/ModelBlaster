#!/usr/bin/env bash
# End-to-end runner for YOLOv8-nano. Shared body in _run_lib.sh.
set -euo pipefail
MODEL_NAME=yolov8_nano
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export MODEL_NAME REPO_ROOT
source "${REPO_ROOT}/modelblaster/examples/_run_lib.sh"
