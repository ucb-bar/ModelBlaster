#!/usr/bin/env bash
# ISOLATED fused_full tree for the rvv_f16 conv2d_s8 weight-layout fix.
# Shares nothing with examples/fused_full/ (other agents build in that tree,
# and _run_lib.sh SKIPS extract when graph.json is present, so a concurrent
# re-extract of a shared tree silently corrupts in-flight A/Bs). The IR was
# copied in, so stage 1 skips extract and never needs the model registry name.
set -euo pipefail
MODEL_NAME=fused_full_fix
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MODEL_NAME REPO_ROOT
source "${REPO_ROOT}/examples/_run_lib.sh"
