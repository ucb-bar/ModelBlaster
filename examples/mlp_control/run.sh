#!/usr/bin/env bash
# End-to-end runner for the trained drone-control MLP policy. Loads the
# rsl_rl PPO actor (16 obs -> 256/128/64 ELU -> 4 actions) from the latest
# crazyflie_steering_tracking checkpoint. Override the checkpoint path via
# AGENTS_MLP_CONTROL_CKPT.
#
# See mlp_generic/run.sh for the full env-var docs.
set -euo pipefail
MODEL_NAME=mlp_control
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export MODEL_NAME REPO_ROOT
source "${REPO_ROOT}/agents/examples/_run_lib.sh"
