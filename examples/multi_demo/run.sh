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
#   RUNNER={spike,firesim}             simulator (default spike)
#   POOL_SIZES=1,2,4                   when set, build pool-sweep harness
#                                      that runs each model once per pool
#                                      size; emits topo_<cores> profiles
#                                      reflecting that hart layout in a
#                                      single binary (amortizes infrasetup
#                                      on firesim).
set -euo pipefail

MODELS="${MODELS:-mlp_generic,mlp_control}"
TARGET="${TARGET:-scalar}"
QUANT="${QUANT:-fp32}"
FORCE_REGEN="${FORCE_REGEN:-0}"
RUNNER="${RUNNER:-spike}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
export PATH="/usr/bin:${PATH}"

EXAMPLE_DIR="${REPO_ROOT}/agents/examples/multi_demo"
BUILD_TAG="${TARGET}"
if [[ -n "${POOL_SIZES:-}" ]]; then
    # Distinct build dir for sweep — different multi_main.c, different
    # AGENTS_POOL_SIZES define. Avoids stomping on the single-pool build.
    BUILD_TAG="${TARGET}_sweep_${POOL_SIZES//,/_}"
fi
if [[ "${RUNNER}" == "firesim" ]]; then
    BUILD_TAG="${BUILD_TAG}_firesim"
fi
BUILD_DIR="${EXAMPLE_DIR}/${QUANT}/build/${BUILD_TAG}"
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
        # `var=value cmd` env assignments must be lexically on the
        # command line — `${VAR:+VAR=value}` expands AFTER tokenization
        # so the result is treated as a command word, not an env
        # assignment ("GLOBAL_CURATED_DIR=...: No such file or directory").
        # Export GLOBAL_CURATED_DIR explicitly so the child run.sh sees it
        # via the environment.
        if [[ -n "${GLOBAL_CURATED_DIR:-}" ]]; then
            export GLOBAL_CURATED_DIR
        fi
        TARGET="${TARGET}" QUANT="${QUANT}" \
        BACKEND="${BACKEND:-reference}" OPTIMIZE="${OPTIMIZE:-0}" \
        BEAM="${BEAM:-2}" EXPANSIONS="${EXPANSIONS:-3}" ITERATIONS="${ITERATIONS:-2}" \
            bash "${REPO_ROOT}/agents/examples/${m}/run.sh" >/dev/null
    fi
    MODEL_DIRS+="${MODEL_DIRS:+;}${m_gen_dir}"
    MODEL_NAMES+="${MODEL_NAMES:+;}${m}"
done

# Pick a board target up front: spike uses spike_riscv64; firesim uses the
# chipyard_riscv64 quad-rocket-saturn board with its prj.conf overlay.
case "${RUNNER}" in
    spike)
        BOARD_TARGET="spike_riscv64"
        ;;
    firesim)
        BOARD_TARGET="chipyard_riscv64/rocketchip_virt_riscv64"
        ;;
    *)
        echo "ERROR: unsupported RUNNER=${RUNNER} (expected spike|firesim)" >&2
        exit 1
        ;;
esac

echo "[multi] west build (TARGET=${TARGET} QUANT=${QUANT} RUNNER=${RUNNER} BOARD=${BOARD_TARGET})"
echo "[multi]   models: ${MODEL_NAMES}"
if [[ -n "${POOL_SIZES:-}" ]]; then
    echo "[multi]   pool-sweep: ${POOL_SIZES}"
fi

KERNEL_CFLAGS=$(python -c "
from agents.pipeline.backends import get
b = get('${TARGET}')
print(';'.join(b.resolved_kernel_cflags('${REPO_ROOT}')))
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
# Pool-sweep mode: tell the harness CMake to forward --pool-sizes to the
# multi_main.c generator. The emitted main re-creates the pool per size
# inside one binary.
if [[ -n "${POOL_SIZES:-}" ]]; then
    WEST_CMAKE_ARGS+=("-DAGENTS_POOL_SIZES=${POOL_SIZES}")
fi

WEST_BUILD_EXTRA=()
if [[ "${RUNNER}" == "firesim" ]]; then
    if [[ -n "${FIRESIM_CONF:-}" ]]; then
        FS_CONF="${REPO_ROOT}/agents/harness/backends/${FIRESIM_CONF}"
    elif [[ "${TARGET}" == "gemmini" ]]; then
        FS_CONF="${REPO_ROOT}/agents/harness/backends/firesim_chipyard_dual_gemmini.conf"
    else
        FS_CONF="${REPO_ROOT}/agents/harness/backends/firesim_chipyard.conf"
    fi
    WEST_BUILD_EXTRA+=(
        -DEXTRA_CONF_FILE="${FS_CONF}"
    )
fi

west build -p -b "${BOARD_TARGET}" agents/harness_multi \
    --build-dir "${BUILD_DIR}" \
    -- "${WEST_CMAKE_ARGS[@]}" "${WEST_BUILD_EXTRA[@]}"

# IREE-shape per-dispatch profile emission (opt-in via PROFILE_OUT_ROOT
# env). When set, the runner additionally writes results.csv per model
# under <root>/<backend>/<cpu>/<model>/.../topo_<cores>/. See
# agents/pipeline/profile_writer.py.
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
        "--profile-source=${PROFILE_SOURCE:-${RUNNER}}"
        "--profile-backend=${PROFILE_BACKEND}"
        "--profile-cores=${PROFILE_CORES:-0,1,2,3}"
        "--profile-clock-mhz=${PROFILE_CLOCK_MHZ:-1000.0}"
    )
    if [[ -n "${PROFILE_CPU:-}" ]]; then
        PROFILE_FLAGS+=("--profile-cpu=${PROFILE_CPU}")
    fi
fi

POOL_FLAGS=()
if [[ -n "${POOL_SIZES:-}" ]]; then
    POOL_FLAGS+=("--pool-sizes=${POOL_SIZES}")
fi

if [[ "${RUNNER}" == "spike" ]]; then
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
    SPIKE_BIN_FLAGS=()
    if [[ "${TARGET}" == "gemmini" ]]; then
        _GEMMINI_SPIKE="${AGENTS_GEMMINI_SPIKE:-/scratch2/dima/chipyard-fsim/.conda-env/riscv-tools/bin/spike}"
        _GEMMINI_LIB_DIR="${AGENTS_GEMMINI_LIB_DIR:-/scratch2/dima/chipyard-fsim/.conda-env/riscv-tools/lib}"
        if [[ -f "${_GEMMINI_SPIKE}" ]]; then
            SPIKE_BIN_FLAGS+=(--spike "${_GEMMINI_SPIKE}")
            export LD_LIBRARY_PATH="${_GEMMINI_LIB_DIR}:${LD_LIBRARY_PATH:-}"
        fi
    fi
    python -m agents.validation.spike_runner \
        --elf "${BUILD_DIR}/zephyr/zephyr.elf" \
        --io  "${REPO_ROOT}/agents/examples/${MODEL_LIST[0]}/${QUANT}/generated/io.npz" \
        --models "${MODELS}" \
        --quant "${QUANT}" \
        --timeout "${SPIKE_TIMEOUT:-600}" \
        "${SPIKE_BIN_FLAGS[@]}" \
        "${SPIKE_FLAGS[@]}" \
        "${POOL_FLAGS[@]}" \
        "${PROFILE_FLAGS[@]}"
else
    echo "[multi] firesim"
    FIRESIM_FLAGS=()
    if [[ -n "${FIRESIM_ROOT:-}" ]]; then
        FIRESIM_FLAGS+=("--firesim-root=${FIRESIM_ROOT}")
    fi
    if [[ -n "${FIRESIM_ENV:-}" ]]; then
        FIRESIM_FLAGS+=("--firesim-env=${FIRESIM_ENV}")
    fi
    if [[ -n "${FIRESIM_SLOT:-}" ]]; then
        FIRESIM_FLAGS+=("--firesim-slot=${FIRESIM_SLOT}")
    fi
    if [[ -n "${FIRESIM_TIMEOUT:-}" ]]; then
        FIRESIM_FLAGS+=("--timeout=${FIRESIM_TIMEOUT}")
    fi
    python -m agents.validation.firesim_runner \
        --elf "${BUILD_DIR}/zephyr/zephyr.elf" \
        --io  "${REPO_ROOT}/agents/examples/${MODEL_LIST[0]}/${QUANT}/generated/io.npz" \
        --models "${MODELS}" \
        --quant "${QUANT}" \
        "${FIRESIM_FLAGS[@]}" \
        "${POOL_FLAGS[@]}" \
        "${PROFILE_FLAGS[@]}"
fi
