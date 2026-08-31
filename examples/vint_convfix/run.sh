#!/usr/bin/env bash
# convfix investigation copy of examples/vint/run.sh (stage-1 extract skipped;
# IR is copied in from examples/vint/int8/generated).
set -euo pipefail
MODEL_NAME=vint_convfix
QUANT="${QUANT:-int8}"
TARGET="${TARGET:-rvv_f16}"
BACKEND="${BACKEND:-reference}"
RUNNER="${RUNNER:-firesim}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
MB_ROOT="${REPO_ROOT}"
source "${REPO_ROOT}/../scripts/set_envvars_sdk.sh"
REPO_ROOT="${MB_ROOT}"
unset FORCE_EXTRACT
export MODEL_NAME REPO_ROOT QUANT TARGET BACKEND RUNNER
source "${REPO_ROOT}/examples/_run_lib.sh"
