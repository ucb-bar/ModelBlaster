#!/usr/bin/env bash
# ISOLATED dronet tree for the TACIT basic-block profiling run.
#
# Why this does NOT source examples/_run_lib.sh: the shared lib hardcodes
# `west build ... harness`, and this experiment needs the TACIT-bracketed
# harness (harness_tacit/, a copy of harness/ whose main() wraps
# model_run_test() in l_trace_encoder_start/stop).  Everything else --
# stage order, cmake args, the firesim overlay pick -- mirrors _run_lib.sh.
#
# The IR was COPIED in from examples/dronet_kopt/int8/generated, so stage 1
# never re-extracts and this tree shares nothing writable with
# examples/{dronet,dronet_kopt,yolov8_nano,...} that other agents are
# building in concurrently.
set -euo pipefail

MODEL_NAME=dronet_tacit
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PATH="/usr/bin:${PATH}"

TARGET="${TARGET:-rvv}"
QUANT="${QUANT:-int8}"
BACKEND="${BACKEND:-reference}"
GEN_TARGET="${TARGET}"

EXAMPLE_DIR="${REPO_ROOT}/examples/${MODEL_NAME}"
IR_DIR="${EXAMPLE_DIR}/${QUANT}/generated"
GEN_DIR="${IR_DIR}/${TARGET}"
VERIFY_BUILD_DIR="${EXAMPLE_DIR}/${QUANT}/build/${TARGET}"
BUILD_DIR="${EXAMPLE_DIR}/${QUANT}/build/${TARGET}_firesim"
CACHE_DIR="${EXAMPLE_DIR}/${QUANT}/cache/${TARGET}"
mkdir -p "${GEN_DIR}" "${BUILD_DIR%/*}" "${CACHE_DIR}"

[[ -f "${IR_DIR}/graph.json" ]] || { echo "ERROR: no IR at ${IR_DIR}" >&2; exit 1; }
echo "[1/4] extract: SKIPPED (IR copied in from dronet_kopt)"

echo "[2/4] generate_skeleton (backend=${GEN_TARGET}) -> ${GEN_DIR}"
python -m modelblaster.pipeline.generate_skeleton \
    --ir "${IR_DIR}/graph.json" --weights "${IR_DIR}/weights.npz" \
    --io "${IR_DIR}/io.npz" --out-dir "${GEN_DIR}" --backend "${GEN_TARGET}"

echo "[3/4] generate_kernels (backend=${BACKEND} target=${GEN_TARGET}) -> ${GEN_DIR}"
GEN_KERNELS_ARGS=(
    --ir "${IR_DIR}/graph.json" --out-dir "${GEN_DIR}"
    --backend "${BACKEND}" --target "${GEN_TARGET}" --quant "${QUANT}"
    --io "${IR_DIR}/io.npz" --repo-root "${REPO_ROOT}"
    --build-dir "${VERIFY_BUILD_DIR}"
    --harness-dir "harness"
    --cache-dir "${CACHE_DIR}" --algorithms "${ALGORITHMS:-all}"
)
if [[ -n "${GLOBAL_CURATED_DIR:-}" ]]; then
    GEN_KERNELS_ARGS+=(--global-curated-dir "${GLOBAL_CURATED_DIR}")
fi
python -m modelblaster.pipeline.generate_kernels "${GEN_KERNELS_ARGS[@]}"

echo "[4/4] west build (board=chipyard_riscv64, harness=harness_tacit) -> ${BUILD_DIR}"
KERNEL_CFLAGS=$(python -c "
from modelblaster.pipeline.backends import get
b = get('${GEN_TARGET}')
print(';'.join(b.resolved_kernel_cflags('${REPO_ROOT}')))
")
FS_CONF="${FIRESIM_CONF_PATH:-${REPO_ROOT}/harness_tacit/backends/firesim_chipyard.conf}"
WEST_CMAKE_ARGS=(-DMODEL_DIR="${GEN_DIR}" -DMODELBLASTER_BACKEND="${GEN_TARGET}")
[[ -n "${KERNEL_CFLAGS}" ]] && WEST_CMAKE_ARGS+=(-DMODELBLASTER_KERNEL_CFLAGS="${KERNEL_CFLAGS}")

west build -p -b chipyard_riscv64/rocketchip_virt_riscv64 harness_tacit \
    --build-dir "${BUILD_DIR}" \
    -- "${WEST_CMAKE_ARGS[@]}" -DEXTRA_CONF_FILE="${FS_CONF}"

echo "ELF: ${BUILD_DIR}/zephyr/zephyr.elf"
ls -l "${BUILD_DIR}/zephyr/zephyr.elf"
