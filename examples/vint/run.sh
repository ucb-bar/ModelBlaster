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
        --model "${MODEL_NAME}" --quant "${QUANT}" --out-dir "${IR_DIR}"
fi

# Stage 2–5 — delegate to _run_lib.sh. It checks for the IR triple
# on disk and skips its own extract_graph step when they're present.
export MODEL_NAME REPO_ROOT QUANT TARGET BACKEND RUNNER
source "${REPO_ROOT}/agents/examples/_run_lib.sh"
