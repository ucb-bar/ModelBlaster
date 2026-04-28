#!/usr/bin/env bash
# End-to-end runner for MLP. Orchestration body lives in
# agents/examples/_run_lib.sh; this script just sets MODEL_NAME and execs in.
#
# Env vars (forwarded to the shared lib):
#   BACKEND={reference,llm}       default reference
#   TARGET={scalar,rvv}           default scalar
#   QUANT={fp32}                  default fp32  (int8 etc. land later)
#   OPTIMIZE={0,1}                default 0
#   ALGORITHMS={all,default,csv}  default all
#   BEAM/EXPANSIONS/ITERATIONS    optimize-loop knobs
#
# Prereqs (run once per shell):
#   source tools/miniforge3/etc/profile.d/conda.sh && conda activate zephyr
#   source scripts/set_envvars_sdk.sh
#   source ../set_api_keys.sh   # required for BACKEND=llm
#
# Run from the zephyr-chipyard-sw repo root:
#   bash agents/examples/mlp/run.sh
#   BACKEND=llm bash agents/examples/mlp/run.sh
#   TARGET=rvv BACKEND=llm OPTIMIZE=1 bash agents/examples/mlp/run.sh
set -euo pipefail
MODEL_NAME=mlp_generic
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export MODEL_NAME REPO_ROOT
source "${REPO_ROOT}/agents/examples/_run_lib.sh"
