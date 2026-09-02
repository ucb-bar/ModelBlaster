#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# ARM-A CONTROL: MB_DRIFT_ATOL is UNSET here, deliberately and defensively.
# The control's whole purpose is the exact-kernel pick, so an MB_DRIFT_ATOL
# inherited from the caller's shell would silently turn this arm into arm B and
# make the A/B meaningless -- with no error and a plausible-looking result.
# Unset it rather than trust the environment.
unset MB_DRIFT_ATOL
# ---------------------------------------------------------------------------
# ISOLATED dronet tree for the ARM-A CONTROL (no MB_DRIFT_ATOL) pick probe A/B.
# Shares nothing with examples/dronet/ -- a concurrent agent runs ARM A there and
# _run_lib.sh skips extract when graph.json exists, so a re-extract in a shared
# tree would silently corrupt the other arm's build. IR copied in verbatim from
# examples/dronet/int8/generated so both arms compare the SAME graph.
set -euo pipefail
MODEL_NAME=dronet_armXref
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MODEL_NAME REPO_ROOT
source "${REPO_ROOT}/examples/_run_lib.sh"
