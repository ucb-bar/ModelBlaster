#!/usr/bin/env bash
# ISOLATED yolov8n tree for the BN-folding A/B (arm: nobnfold).
set -euo pipefail
MODEL_NAME=yolov8_nano_nobnfold
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MODEL_NAME REPO_ROOT
source "${REPO_ROOT}/examples/_run_lib.sh"
