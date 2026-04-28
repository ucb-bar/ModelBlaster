#!/usr/bin/env bash
# Multi-model demo: run two existing models in one Zephyr binary.
#
# Pre-req: each constituent model must be extracted + skeleton + kernels
# generated under its own example dir. By default this runs:
#   bash agents/examples/mlp_generic/run.sh
#   bash agents/examples/mlp_control/run.sh
# first to ensure the per-model artifacts exist for the chosen TARGET / QUANT.
#
# Env vars:
#   MODELS=mlp_generic,mlp_control     comma list of constituent models
#   TARGET={scalar,rvv}                shared HW backend for all models
#   QUANT={fp32,int8}                  shared quant for all models
#   FORCE_REGEN={0,1}                  re-run each constituent's run.sh first
set -euo pipefail

MODELS="${MODELS:-mlp_generic,mlp_control}"
TARGET="${TARGET:-scalar}"
QUANT="${QUANT:-fp32}"
FORCE_REGEN="${FORCE_REGEN:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
export PATH="/usr/bin:${PATH}"

EXAMPLE_DIR="${REPO_ROOT}/agents/examples/multi_demo"
BUILD_DIR="${EXAMPLE_DIR}/${QUANT}/build/${TARGET}"
mkdir -p "${BUILD_DIR%/*}"

# Split MODELS into an array, dedupe nothing (caller controls order).
IFS=',' read -ra MODEL_LIST <<< "${MODELS}"

# Stage every constituent model's artifacts. Each run.sh re-extracts + emits
# generated/<target>/{model.c,kernels.c,weights.c,model.h,test_io.h} for the
# requested QUANT. We pass through TARGET/QUANT only — BACKEND stays at the
# default (reference) and OPTIMIZE is left off; this script is about
# multi-model wiring, not kernel-gen.
MODEL_DIRS=""
MODEL_NAMES=""
for m in "${MODEL_LIST[@]}"; do
    m_gen_dir="${REPO_ROOT}/agents/examples/${m}/${QUANT}/generated/${TARGET}"
    if [[ "${FORCE_REGEN}" == "1" || ! -f "${m_gen_dir}/model.h" ]]; then
        echo "[stage] running agents/examples/${m}/run.sh (TARGET=${TARGET} QUANT=${QUANT} BACKEND=${BACKEND:-reference} OPTIMIZE=${OPTIMIZE:-0})"
        TARGET="${TARGET}" QUANT="${QUANT}" \
        BACKEND="${BACKEND:-reference}" OPTIMIZE="${OPTIMIZE:-0}" \
        BEAM="${BEAM:-2}" EXPANSIONS="${EXPANSIONS:-3}" ITERATIONS="${ITERATIONS:-2}" \
            bash "${REPO_ROOT}/agents/examples/${m}/run.sh" >/dev/null
    fi
    MODEL_DIRS+="${MODEL_DIRS:+;}${m_gen_dir}"
    MODEL_NAMES+="${MODEL_NAMES:+;}${m}"
done

echo "[multi] west build (TARGET=${TARGET} QUANT=${QUANT})"
echo "[multi]   models: ${MODEL_NAMES}"

KERNEL_CFLAGS=$(python -c "
from agents.pipeline.backends import get
b = get('${TARGET}')
print(';'.join(b.kernel_cflags))
")
WEST_CMAKE_ARGS=(
    "-DAGENTS_BACKEND=${TARGET}"
    "-DMODEL_NAMES=${MODEL_NAMES}"
    "-DMODEL_DIRS=${MODEL_DIRS}"
)
if [[ -n "${KERNEL_CFLAGS}" ]]; then
    WEST_CMAKE_ARGS+=("-DAGENTS_KERNEL_CFLAGS=${KERNEL_CFLAGS}")
fi
# Optional: bake a fixed pool worker count into multi_main.c so the
# profile sweep can compare 1- vs N-thread per-dispatch costs without
# rebuilding Zephyr's CONFIG_MP_MAX_NUM_CPUS.
if [[ -n "${AGENTS_POOL_THREADS:-}" ]]; then
    WEST_CMAKE_ARGS+=("-DAGENTS_POOL_THREADS=${AGENTS_POOL_THREADS}")
fi

west build -p -b spike_riscv64 agents/harness_multi \
    --build-dir "${BUILD_DIR}" \
    -- "${WEST_CMAKE_ARGS[@]}"

echo "[multi] spike"
SPIKE_ARGS=$(python -c "
from agents.pipeline.backends import get
b = get('${TARGET}')
print(' '.join(b.spike_args))
")
SPIKE_FLAGS=("--spike-arg=-p${SPIKE_HARTS:-4}")
for a in ${SPIKE_ARGS}; do
    SPIKE_FLAGS+=("--spike-arg=${a}")
done

# IREE-shape per-dispatch profile emission (opt-in via PROFILE_OUT_ROOT
# env). When set, spike_runner additionally writes results.csv per model
# under <root>/<backend>/<cpu>/<model>/.../topo_<cores>/. See
# agents/pipeline/profile_writer.py.
PROFILE_FLAGS=()
if [[ -n "${PROFILE_OUT_ROOT:-}" ]]; then
    # Codegen uses lowercase backend names (matches agents/pipeline/backends.py);
    # XPU-RT's directory lookup expects "RVV" uppercase. Default the label to
    # an uppercased TARGET, overrideable via PROFILE_BACKEND.
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
        "--profile-cores=${PROFILE_CORES:-0,1,2,3}"
        "--profile-clock-mhz=${PROFILE_CLOCK_MHZ:-1000.0}"
    )
    if [[ -n "${PROFILE_CPU:-}" ]]; then
        PROFILE_FLAGS+=("--profile-cpu=${PROFILE_CPU}")
    fi
fi

python -m agents.validation.spike_runner \
    --elf "${BUILD_DIR}/zephyr/zephyr.elf" \
    --io  "${REPO_ROOT}/agents/examples/${MODEL_LIST[0]}/${QUANT}/generated/io.npz" \
    --models "${MODELS}" \
    --quant "${QUANT}" \
    --timeout "${SPIKE_TIMEOUT:-600}" \
    "${SPIKE_FLAGS[@]}" \
    "${PROFILE_FLAGS[@]}"
