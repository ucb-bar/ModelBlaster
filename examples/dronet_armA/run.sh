#!/usr/bin/env bash
# ISOLATED dronet tree for ARM A (int32-drain exact conv) of the conv2d_s8
# A/B. Shares nothing with examples/dronet/ -- a concurrent agent
# re-extracts / rebuilds that tree with a different kernel-selection gate.
# The IR was copied in, so stage 1 skips extract and never needs the model
# registry name. graph.json's "name" field is still "dronet", so every
# generated symbol stays model_dronet_* and the xpurt schedule's network
# name matches.
set -euo pipefail
MODEL_NAME=dronet_armA
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MODEL_NAME REPO_ROOT
source "${REPO_ROOT}/examples/_run_lib.sh"
