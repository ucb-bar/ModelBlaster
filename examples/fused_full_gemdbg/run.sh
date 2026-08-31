#!/usr/bin/env bash
# ISOLATED fused_full tree for the Gemmini (gemmini_q31 / gemmini_q31_rvv)
# measurement. Shares nothing with examples/fused_full/ (an E2E profile owns
# that one); the IR was snapshotted in so stage 1 skips extract.
set -euo pipefail
MODEL_NAME=fused_full_gemdbg
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MODEL_NAME REPO_ROOT
source "${REPO_ROOT}/examples/_run_lib.sh"
