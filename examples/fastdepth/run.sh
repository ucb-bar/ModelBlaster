#!/usr/bin/env bash
# End-to-end runner for FastDepth (MobileNet encoder + NNConv5 decoder,
# monocular dense depth). See mlp/run.sh for env-var documentation; the shared
# body lives in modelblaster/examples/_run_lib.sh.
#
# MB_DRIFT_ATOL=2 matches the convention the other conv models use here:
# without it generate_kernels' verify gate rejects the curated tiled convs and
# silently falls back to the much slower software im2col.
set -euo pipefail
export MB_DRIFT_ATOL="${MB_DRIFT_ATOL:-2}"
MODEL_NAME=fastdepth
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MODEL_NAME REPO_ROOT
source "${REPO_ROOT}/examples/_run_lib.sh"
