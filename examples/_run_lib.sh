#!/usr/bin/env bash
# Shared orchestration body for all examples.
#
# Caller responsibilities (set before sourcing/exec'ing this script):
#   MODEL_NAME   the model identifier passed to extract_graph (--model)
#   REPO_ROOT    repo root path; the script cd's into it
#
# Optional env vars (with defaults applied here):
#   BACKEND      reference (default) | llm
#   TARGET       scalar (default) | rvv
#   QUANT        fp32 (default)              # int8 etc. land later
#   OPTIMIZE     0 (default) | 1
#   ALGORITHMS   all (default) | comma list  # forwarded to generate_kernels
#   BEAM/EXPANSIONS/ITERATIONS               # optimize-loop knobs
#
# Layout (per-model, per-quant, per-target):
#   agents/examples/<model>/<quant>/generated/             # IR (target-indep)
#   agents/examples/<model>/<quant>/generated/<target>/    # generated C
#   agents/examples/<model>/<quant>/build/<target>/        # west build
#   agents/examples/<model>/<quant>/cache/<target>/        # kernel cache

set -euo pipefail

: "${MODEL_NAME:?MODEL_NAME must be set by the caller}"
: "${REPO_ROOT:?REPO_ROOT must be set by the caller}"

BACKEND="${BACKEND:-reference}"
TARGET="${TARGET:-scalar}"
QUANT="${QUANT:-fp32}"
OPTIMIZE="${OPTIMIZE:-0}"

EXAMPLE_DIR_REL="agents/examples/${MODEL_NAME}"
EXAMPLE_DIR="${REPO_ROOT}/${EXAMPLE_DIR_REL}"

cd "${REPO_ROOT}"
# Avoid the stale Vitis cmake on PATH (it's 3.3.2 and breaks west).
export PATH="/usr/bin:${PATH}"

IR_DIR="${EXAMPLE_DIR}/${QUANT}/generated"
GEN_DIR="${IR_DIR}/${TARGET}"
BUILD_DIR="${EXAMPLE_DIR}/${QUANT}/build/${TARGET}"
CACHE_DIR="${EXAMPLE_DIR}/${QUANT}/cache/${TARGET}"
mkdir -p "${GEN_DIR}" "${BUILD_DIR%/*}" "${CACHE_DIR}"

echo "[1/5] extract_graph (quant=${QUANT}) -> ${IR_DIR}"
python -m agents.pipeline.extract_graph \
    --model "${MODEL_NAME}" \
    --out-dir "${IR_DIR}" \
    --quant "${QUANT}"

echo "[2/5] generate_skeleton -> ${GEN_DIR}"
python -m agents.pipeline.generate_skeleton \
    --ir "${IR_DIR}/graph.json" \
    --weights "${IR_DIR}/weights.npz" \
    --io "${IR_DIR}/io.npz" \
    --out-dir "${GEN_DIR}"

echo "[3/5] generate_kernels (backend=${BACKEND} target=${TARGET} quant=${QUANT} optimize=${OPTIMIZE}) -> ${GEN_DIR}"
GEN_KERNELS_ARGS=(
    --ir "${IR_DIR}/graph.json"
    --out-dir "${GEN_DIR}"
    --backend "${BACKEND}"
    --target "${TARGET}"
    --quant "${QUANT}"
    --io "${IR_DIR}/io.npz"
    --repo-root "${REPO_ROOT}"
    --build-dir "${BUILD_DIR}"
    --harness-dir "agents/harness"
    --cache-dir "${CACHE_DIR}"
    --algorithms "${ALGORITHMS:-all}"
)
if [[ "${OPTIMIZE}" == "1" ]]; then
    GEN_KERNELS_ARGS+=(
        --optimize
        --beam "${BEAM:-2}"
        --expansions "${EXPANSIONS:-3}"
        --iterations "${ITERATIONS:-2}"
    )
fi
python -m agents.pipeline.generate_kernels "${GEN_KERNELS_ARGS[@]}"

echo "[4/5] west build -> ${BUILD_DIR}"
KERNEL_CFLAGS=$(python -c "
from agents.pipeline.backends import get
b = get('${TARGET}')
print(';'.join(b.kernel_cflags))
")
WEST_CMAKE_ARGS=(
    -DMODEL_DIR="${GEN_DIR}"
    -DAGENTS_BACKEND="${TARGET}"
)
if [[ -n "${KERNEL_CFLAGS}" ]]; then
    WEST_CMAKE_ARGS+=(-DAGENTS_KERNEL_CFLAGS="${KERNEL_CFLAGS}")
fi
west build -p -b spike_riscv64 agents/harness \
    --build-dir "${BUILD_DIR}" \
    -- "${WEST_CMAKE_ARGS[@]}"

echo "[5/5] spike + compare"
SPIKE_ARGS=$(python -c "
from agents.pipeline.backends import get
b = get('${TARGET}')
print(' '.join(b.spike_args))
")
SPIKE_FLAGS=()
for a in ${SPIKE_ARGS}; do
    # Use --spike-arg=<...> form so argparse accepts values starting with --.
    SPIKE_FLAGS+=("--spike-arg=${a}")
done

# Optional IREE-shape per-dispatch profile (PROFILE_OUT_ROOT env). The
# single-model case treats the run as a 1-hart "topo_0" trace by default
# — overrideable via PROFILE_CORES.
PROFILE_FLAGS=()
if [[ -n "${PROFILE_OUT_ROOT:-}" ]]; then
    if [[ -z "${PROFILE_BACKEND:-}" ]]; then
        case "${TARGET}" in
            rvv) PROFILE_BACKEND="RVV" ;;
            *)   PROFILE_BACKEND="${TARGET}" ;;
        esac
    fi
    PROFILE_FLAGS+=(
        "--profile-out-root=${PROFILE_OUT_ROOT}"
        "--profile-source=${PROFILE_SOURCE:-spike}"
        "--profile-backend=${PROFILE_BACKEND}"
        "--profile-cores=${PROFILE_CORES:-0}"
        "--profile-clock-mhz=${PROFILE_CLOCK_MHZ:-1000.0}"
    )
    if [[ -n "${PROFILE_CPU:-}" ]]; then
        PROFILE_FLAGS+=("--profile-cpu=${PROFILE_CPU}")
    fi
fi

python -m agents.validation.spike_runner \
    --elf "${BUILD_DIR}/zephyr/zephyr.elf" \
    --io "${IR_DIR}/io.npz" \
    "${SPIKE_FLAGS[@]}" \
    "${PROFILE_FLAGS[@]}"
