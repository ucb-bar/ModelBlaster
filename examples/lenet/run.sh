#!/usr/bin/env bash
# End-to-end runner for LeNet. See mlp/run.sh for env-var documentation; the
# shared body lives in modelblaster/examples/_run_lib.sh.
set -euo pipefail
MODEL_NAME=lenet
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export MODEL_NAME REPO_ROOT
source "${REPO_ROOT}/modelblaster/examples/_run_lib.sh"
