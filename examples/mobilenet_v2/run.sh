#!/usr/bin/env bash
# End-to-end runner for MobileNetV2 (torchvision impl, configurable
# width_mult / input via env). See agents/examples/_run_lib.sh for the
# shared flow body and the supported env knobs (BACKEND, TARGET, QUANT,
# OPTIMIZE, ALGORITHMS, BEAM/EXPANSIONS/ITERATIONS).
#
# Model-specific knobs (consumed by agents/models/mobilenet_v2.py):
#   AGENTS_MOBILENETV2_WIDTH_MULT   default 0.25
#   AGENTS_MOBILENETV2_INPUT        default 96
#   AGENTS_MOBILENETV2_NUM_CLASSES  default 1000
set -euo pipefail
MODEL_NAME=mobilenet_v2
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export MODEL_NAME REPO_ROOT
source "${REPO_ROOT}/agents/examples/_run_lib.sh"
