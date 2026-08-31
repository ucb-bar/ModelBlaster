#!/usr/bin/env bash
# ISOLATED dronet tree for the gemmini_q31_rvv conv2d_s8 kernel-opt A/B.
# Shares nothing with examples/dronet/ (other agents re-extract that IR
# mid-campaign -- it changed from fused conv2d_pool_s8 to unfused
# conv2d_s8+maxpool2d_s8 between two of this session's builds, which
# silently invalidated an A/B pair). The IR was copied in, so stage 1
# skips extract and never needs the model registry name.
set -euo pipefail
MODEL_NAME=dronet_hwcB
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MODEL_NAME REPO_ROOT
source "${REPO_ROOT}/examples/_run_lib.sh"
