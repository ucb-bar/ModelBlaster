#!/usr/bin/env bash
# End-to-end runner for ViNT (visual-navigation transformer).
#
# ViNT can't be traced via torch.fx.symbolic_trace (EfficientNet's
# len(...) plus nn.TransformerEncoder internals defeat it). We use
# torch.export through extract_graph_export.py instead — see
# agents/notes/vint_zephyr_plan.md.
#
# Two-env split: extract_graph_export runs in the xpurt conda env
# (where vint_train + efficientnet_pytorch live); codegen + west
# build + spike run in zephyr like every other example.
#
# Defaults:
#   QUANT=int8 TARGET=scalar BACKEND=reference RUNNER=spike
set -euo pipefail

MODEL_NAME=vint
QUANT="${QUANT:-int8}"
TARGET="${TARGET:-scalar}"
BACKEND="${BACKEND:-reference}"
RUNNER="${RUNNER:-spike}"
# Default to 16 IDSIA samples for activation-scale calibration.
# Override via AGENTS_VINT_CALIB_DIR for the data source.
N_CALIB="${VINT_NUM_CALIBRATION:-16}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
EXAMPLE_DIR="${REPO_ROOT}/agents/examples/${MODEL_NAME}"
IR_DIR="${EXAMPLE_DIR}/${QUANT}/generated"

# Stage 1 — extract IR via torch.export. Runs in the xpurt env where
# ViNT's deps live. Re-run when FORCE_EXTRACT=1 or when IR is missing.
if [[ "${FORCE_EXTRACT:-0}" == "1" \
      || ! -f "${IR_DIR}/graph.json" \
      || ! -f "${IR_DIR}/weights.npz" \
      || ! -f "${IR_DIR}/io.npz" ]]; then
    echo "[vint] extract_graph_export (xpurt env) -> ${IR_DIR}"
    mkdir -p "${IR_DIR}"
    XPURT_PY="${XPURT_PY:-/scratch2/dima/miniforge3/envs/xpurt/bin/python}"
    PYTHONPATH="${REPO_ROOT}" "${XPURT_PY}" -m agents.pipeline.extract_graph_export \
        --model "${MODEL_NAME}" --quant "${QUANT}" \
        --num-calibration "${N_CALIB}" --out-dir "${IR_DIR}"
fi

# Stage 2–5 — delegate to _run_lib.sh. Re-source the Zephyr SDK env
# unconditionally before this hand-off: the xpurt python in stage 1
# is invoked via full path (no `conda activate`), so the shell's
# active env is whatever the caller was in — could be xpurt without
# the build-aware west extension, could be base, etc. Forcing
# set_envvars_sdk here means run.sh works from any starting env as
# long as the conda zephyr binaries are reachable.
source "${REPO_ROOT}/scripts/set_envvars_sdk.sh"
export MODEL_NAME REPO_ROOT QUANT TARGET BACKEND RUNNER
source "${REPO_ROOT}/agents/examples/_run_lib.sh"
