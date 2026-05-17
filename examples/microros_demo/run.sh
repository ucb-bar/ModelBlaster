#!/usr/bin/env bash
# microros_demo — fixed-HW per-network ROS-node baseline runtime.
#
# Each network is one ROS node pinned to one hart with one rclc executor
# + one rmw session. No per-op scheduling: each network runs its full
# dispatch graph sequentially through whichever backend is assigned to
# its hart. Used as a baseline to compare against xpurt_demo's per-op
# scheduler.
#
# Env:
#   MODELS         comma list of 2-3 networks (default: dronet,yolov8_nano)
#   BACKENDS       comma list of HW backends to BUILD (default: gemmini_q31,rvv)
#   PIN_BACKENDS   comma list — which backend each network's executor
#                  uses for kernels (parallel to MODELS, default
#                  matches MODELS positionally to the first len(MODELS)
#                  entries of BACKENDS).
#   PIN_HARTS      comma list — which hart each network pins to.
#                  Default 0,1.
#   PERIODS_MS     comma list — timer period per network (0 = one-shot).
#                  Default 50,0.
#   QUANT          quant mode (default: int8). Must match what's
#                  generated under each model's <quant>/generated/<bs>/.
#   MODELBLASTER_POOL_THREADS  modelblaster_pool worker count (default: 1).
#                  Baseline is sequential; >1 turns on intra-op parallel
#                  IF the kernels emit parallel_<op> primitives.
#   FORCE_REGEN    {0,1}  re-run each model's run.sh first (default: 1).
#                  Set 0 to re-use existing artifacts (faster iteration).
#   MICROROS_BROKER_HART  hart for the broker (default: highest CPU).
#   RUNNER         {spike,firesim} default spike.
#
# Pre-reqs (from repo root):
#   source scripts/activate_conda.sh && conda activate zephyr
#   source scripts/set_envvars_sdk.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
export PATH="/usr/bin:${PATH}"

MODELS="${MODELS:-dronet,yolov8_nano}"
BACKENDS="${BACKENDS:-gemmini_q31,rvv}"
PIN_BACKENDS="${PIN_BACKENDS:-${BACKENDS}}"
PIN_HARTS="${PIN_HARTS:-0,1}"
PERIODS_MS="${PERIODS_MS:-50,0}"
QUANT="${QUANT:-int8}"
QUANTS="${QUANTS:-}"
MODELBLASTER_POOL_THREADS="${MODELBLASTER_POOL_THREADS:-1}"
FORCE_REGEN="${FORCE_REGEN:-1}"
RUNNER="${RUNNER:-spike}"

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

EXAMPLE_DIR="${REPO_ROOT}/modelblaster/examples/microros_demo"
GEN_DIR="${EXAMPLE_DIR}/${QUANT}/generated"
BUILD_TAG="$(echo "${BACKENDS}" | tr ',' '_')"
if [[ "${RUNNER}" == "firesim" ]]; then
    BUILD_TAG="${BUILD_TAG}_firesim"
fi
BUILD_DIR="${EXAMPLE_DIR}/${QUANT}/build/${BUILD_TAG}"
mkdir -p "${GEN_DIR}" "${BUILD_DIR%/*}"

IFS=',' read -ra MODEL_LIST   <<< "${MODELS}"
IFS=',' read -ra BACKEND_LIST <<< "${BACKENDS}"
IFS=',' read -ra PIN_BS_LIST  <<< "${PIN_BACKENDS}"
IFS=',' read -ra PIN_HART_LIST <<< "${PIN_HARTS}"
IFS=',' read -ra PERIOD_LIST  <<< "${PERIODS_MS}"

if [[ ${#MODEL_LIST[@]} -lt 2 || ${#MODEL_LIST[@]} -gt 3 ]]; then
    echo "ERROR: microros_demo supports 2 or 3 networks; got ${#MODEL_LIST[@]}" >&2
    exit 1
fi

# Resolve per-model quant list (parallel to MODELS).
if [[ -n "${QUANTS}" ]]; then
    IFS=',' read -ra QUANT_LIST <<< "${QUANTS}"
    if [[ "${#QUANT_LIST[@]}" -ne "${#MODEL_LIST[@]}" ]]; then
        echo "ERROR: QUANTS must have one entry per MODELS (got ${#QUANT_LIST[@]} vs ${#MODEL_LIST[@]})" >&2
        exit 1
    fi
else
    QUANT_LIST=()
    for _ in "${MODEL_LIST[@]}"; do
        QUANT_LIST+=("${QUANT}")
    done
fi

for arr in PIN_BS_LIST PIN_HART_LIST PERIOD_LIST; do
    declare -n a="${arr}"
    if [[ ${#a[@]} -ne ${#MODEL_LIST[@]} ]]; then
        echo "ERROR: ${arr} length ${#a[@]} != MODELS length ${#MODEL_LIST[@]}" >&2
        exit 1
    fi
done

# 1) Stage each (model, backend) pair's per-target artifacts. We need
# every backend's generated/<bs>/{model.c,kernels.c,weights.c} to exist
# under each model's example dir before we hand the harness MODEL_DIRS_BASE.
# Mirrors xpurt_demo's loop verbatim.
for idx in "${!MODEL_LIST[@]}"; do
    m="${MODEL_LIST[$idx]}"
    m_quant="${QUANT_LIST[$idx]}"
    for bs in "${BACKEND_LIST[@]}"; do
        if [[ "${FORCE_REGEN}" == "1" ]]; then
            echo "[microros_demo] regen ${m}/${bs} (quant=${m_quant})"
            TARGET="${bs}" QUANT="${m_quant}" \
            BACKEND=llm OPTIMIZE=0 FORCE_REGEN=1 \
            GLOBAL_CURATED_DIR="${REPO_ROOT}/modelblaster/kernels" \
                bash "${REPO_ROOT}/modelblaster/examples/${m}/run.sh"
        fi
        gen="${REPO_ROOT}/modelblaster/examples/${m}/${m_quant}/generated/${bs}"
        for f in model.c kernels.c weights.c model.h test_io.h buffers.c; do
            if [[ ! -f "${gen}/${f}" ]]; then
                echo "ERROR: ${gen} missing ${f} (run with FORCE_REGEN=1 or run modelblaster/examples/${m}/run.sh first)" >&2
                exit 1
            fi
        done
    done
done

# 2) Assemble harness CMake args.
MODEL_NAMES=""
MODEL_DIRS_BASE=""
for idx in "${!MODEL_LIST[@]}"; do
    m="${MODEL_LIST[$idx]}"
    m_quant="${QUANT_LIST[$idx]}"
    MODEL_NAMES="${MODEL_NAMES};${m}"
    MODEL_DIRS_BASE="${MODEL_DIRS_BASE};${REPO_ROOT}/modelblaster/examples/${m}/${m_quant}/generated"
done
MODEL_NAMES="${MODEL_NAMES#;}"
MODEL_DIRS_BASE="${MODEL_DIRS_BASE#;}"

PIN_BACKENDS_LIST="$(IFS=';'; echo "${PIN_BS_LIST[*]}")"
PIN_HARTS_LIST="$(IFS=';'; echo "${PIN_HART_LIST[*]}")"
PERIODS_LIST="$(IFS=';'; echo "${PERIOD_LIST[*]}")"

WEST_CMAKE_ARGS=(
    "-DMODEL_BACKENDS=${BACKENDS}"
    "-DMODEL_NAMES=${MODEL_NAMES}"
    "-DMODEL_DIRS_BASE=${MODEL_DIRS_BASE}"
    "-DMODEL_PIN_BACKENDS=${PIN_BACKENDS_LIST}"
    "-DMODEL_PIN_HARTS=${PIN_HARTS_LIST}"
    "-DMODEL_PERIODS_MS=${PERIODS_LIST}"
    "-DMODELBLASTER_POOL_THREADS=${MODELBLASTER_POOL_THREADS}"
)
if [[ -n "${MICROROS_BROKER_HART:-}" ]]; then
    WEST_CMAKE_ARGS+=("-DMICROROS_BROKER_HART=${MICROROS_BROKER_HART}")
fi
# run_graph_b interrupt-mask debug knob (default MICROROS_MASK_TIMER).
# See modelblaster/harness_microros/src/main.c::run_graph_b.
for _knob in MICROROS_MASK_ALL MICROROS_MASK_TIMER MICROROS_MASK_IPI MICROROS_MASK_EXT \
             MICROROS_SINGLE_EXECUTOR MICROROS_NO_BROKER MICROROS_NO_MICROROS \
             MICROROS_NO_PUBLISH MICROROS_SKIP_TRACE MICROROS_NO_LOCK_A \
             MICROROS_NO_FPREGS_C MICROROS_2EXEC_BC \
             MICROROS_2EXEC_FIRE_FAST MICROROS_2EXEC_NORCLC \
             MICROROS_2EXEC_FUSE_BC MICROROS_FUSE_BC_NO_C; do
    if [[ "${!_knob:-0}" == "1" ]]; then
        WEST_CMAKE_ARGS+=("-D${_knob}=ON")
    fi
done

# 3) Per-backend kernel cflags (same plumbing as xpurt_demo).
for bs in "${BACKEND_LIST[@]}"; do
    BS_UPPER="$(echo "${bs}" | tr '[:lower:]' '[:upper:]')"
    KERNEL_CFLAGS=$(python -c "
from modelblaster.pipeline.backends import get
b = get('${bs}')
print(';'.join(b.resolved_kernel_cflags('${REPO_ROOT}')))
")
    if [[ -n "${KERNEL_CFLAGS}" ]]; then
        WEST_CMAKE_ARGS+=("-DMODELBLASTER_KERNEL_CFLAGS_${BS_UPPER}=${KERNEL_CFLAGS}")
    fi
done

# 4) Per-target overlay (same as xpurt_demo). FireSim Q31 uses the
# dual-gemmini config; spike uses spike_quad.conf.
WEST_BUILD_EXTRA=()
if [[ "${RUNNER}" == "firesim" ]]; then
    if [[ -n "${FIRESIM_CONF:-}" ]]; then
        EXTRA_CONF="${REPO_ROOT}/modelblaster/harness/backends/${FIRESIM_CONF}"
    elif [[ ",${BACKENDS}," == *,gemmini,* || ",${BACKENDS}," == *,gemmini_q31,* ]]; then
        EXTRA_CONF="${REPO_ROOT}/modelblaster/harness/backends/firesim_chipyard_dual_gemmini.conf"
    else
        EXTRA_CONF="${REPO_ROOT}/modelblaster/harness/backends/firesim_chipyard.conf"
    fi
elif [[ "${RUNNER}" == "spike" ]]; then
    EXTRA_CONF="${REPO_ROOT}/modelblaster/harness/backends/${SPIKE_CONF:-spike_quad.conf}"
fi
if [[ -z "${EXTRA_CONF:-}" || ! -f "${EXTRA_CONF}" ]]; then
    echo "ERROR: per-target overlay not found (RUNNER=${RUNNER}, EXTRA_CONF=${EXTRA_CONF:-<unset>})" >&2
    exit 1
fi
WEST_BUILD_EXTRA+=(-DEXTRA_CONF_FILE="${EXTRA_CONF}")

# 5) west build.
echo "[microros_demo] west build (BACKENDS=${BACKENDS}, MODELS=${MODELS})"
west build -p -b "${BOARD_TARGET}" modelblaster/harness_microros \
    --build-dir "${BUILD_DIR}" \
    -- "${WEST_CMAKE_ARGS[@]}" "${WEST_BUILD_EXTRA[@]}"

# 6) Run.
echo "[microros_demo] ${RUNNER} run"
if [[ "${RUNNER}" == "spike" ]]; then
    SPIKE_BIN="${SPIKE_BIN:-}"
    if [[ -z "${SPIKE_BIN}" ]]; then
        if [[ ",${BACKENDS}," == *,gemmini,* || ",${BACKENDS}," == *,gemmini_q31,* ]]; then
            # gemmini ROCC ops → chipyard spike with --extension=gemmini
            SPIKE_BIN="${MODELBLASTER_GEMMINI_SPIKE:-/scratch2/dima/chipyard-fsim/.conda-env/riscv-tools/bin/spike}"
            export LD_LIBRARY_PATH="${MODELBLASTER_GEMMINI_LIB_DIR:-/scratch2/dima/chipyard-fsim/.conda-env/riscv-tools/lib}:${LD_LIBRARY_PATH:-}"
        fi
    fi
    SPIKE_CMD=()
    if [[ -n "${SPIKE_BIN}" ]]; then
        SPIKE_CMD+=("${SPIKE_BIN}")
    else
        SPIKE_CMD+=(spike)
    fi
    SPIKE_CMD+=(-p"${SPIKE_HARTS:-4}")
    _has_gemmini=0
    _has_rvv=0
    [[ ",${BACKENDS}," == *,gemmini,*    || ",${BACKENDS}," == *,gemmini_q31,* ]] && _has_gemmini=1
    [[ ",${BACKENDS}," == *,rvv,*        || ",${BACKENDS}," == *,V256D128_rvv,* || ",${BACKENDS}," == *,V512D256_rvv,* ]] && _has_rvv=1
    if [[ "${_has_gemmini}" == "1" ]]; then
        SPIKE_CMD+=(--extension=gemmini)
    fi
    if [[ "${_has_rvv}" == "1" ]]; then
        SPIKE_CMD+=(--isa=rv64gcv_zicntr)
    elif [[ "${_has_gemmini}" == "1" ]]; then
        SPIKE_CMD+=(--isa=rv64gc_zicntr)
    fi
    SPIKE_CMD+=("${BUILD_DIR}/zephyr/zephyr.elf")
    echo "[microros_demo] ${SPIKE_CMD[*]}"
    timeout "${SPIKE_TIMEOUT:-900}" "${SPIKE_CMD[@]}"
else
    FIRESIM_FLAGS=()
    [[ -n "${FIRESIM_ROOT:-}"   ]] && FIRESIM_FLAGS+=("--firesim-root=${FIRESIM_ROOT}")
    [[ -n "${FIRESIM_ENV:-}"    ]] && FIRESIM_FLAGS+=("--firesim-env=${FIRESIM_ENV}")
    [[ -n "${FIRESIM_SLOT:-}"   ]] && FIRESIM_FLAGS+=("--firesim-slot=${FIRESIM_SLOT}")
    [[ -n "${FIRESIM_TIMEOUT:-}" ]] && FIRESIM_FLAGS+=("--timeout=${FIRESIM_TIMEOUT}")
    python -m modelblaster.validation.firesim_runner \
        --elf "${BUILD_DIR}/zephyr/zephyr.elf" \
        --io  "${REPO_ROOT}/modelblaster/examples/${MODEL_LIST[0]}/${QUANT_LIST[0]}/generated/io.npz" \
        --models "${MODELS}" \
        --quant "${QUANT_LIST[0]}" \
        "${FIRESIM_FLAGS[@]}"
fi
