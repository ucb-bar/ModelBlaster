#!/usr/bin/env bash
# Round-trip demo: run an XPU-RT-emitted schedule.json through our flow.
#
#   schedule.json + per-network IRs + core registry
#       -> ingest_xpurt_schedule -> dispatch_table.{h,c}
#       -> generate_xpurt_main   -> xpurt_main.c
#       -> west build (harness_xpurt)
#       -> spike (-p4)
#       -> spike_runner verifies each network's output against PyTorch goldens.
#
# Env knobs:
#   SCHEDULE_JSON   path to scheduled_*.json
#   MODELS          comma list of constituent network names (default: dronet,mlp_control)
#   REGISTRY        path to agents/cores/*.json (default: chipyard_hetero_example.json)
#   TARGET          scalar | rvv (default: scalar)
#   QUANT           fp32 (default)
#   CPU_P_KIND      registry kind for CPU_P slots (default: rvv)
#   CPU_E_KIND      registry kind for CPU_E slots (default: scalar)
#   AGENTS_POOL_THREADS  pthreadpool worker count (default: 4)
#   FORCE_REGEN     {0,1}   re-run each model's run.sh first (default: 1)
#
# Pre-reqs (from repo root):
#   source tools/miniforge3/etc/profile.d/conda.sh && conda activate zephyr
#   source scripts/set_envvars_sdk.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
export PATH="/usr/bin:${PATH}"

SCHEDULE_JSON="${SCHEDULE_JSON:-/scratch2/dima/misc_sw/FreshScheduler/schedules/scheduled_networks_mlp_dronet_profile_zephyr_profiled.json}"
MODELS="${MODELS:-dronet,mlp_control}"
REGISTRY="${REGISTRY:-${REPO_ROOT}/agents/cores/chipyard_hetero_example.json}"
TARGET="${TARGET:-scalar}"
QUANT="${QUANT:-fp32}"
CPU_P_KIND="${CPU_P_KIND:-rvv}"
CPU_E_KIND="${CPU_E_KIND:-scalar}"
AGENTS_POOL_THREADS="${AGENTS_POOL_THREADS:-4}"
FORCE_REGEN="${FORCE_REGEN:-1}"
SCHED_NAME="${SCHED_NAME:-xpurt_demo_schedule}"

EXAMPLE_DIR="${REPO_ROOT}/agents/examples/xpurt_demo"
GEN_DIR="${EXAMPLE_DIR}/${QUANT}/generated"
BUILD_DIR="${EXAMPLE_DIR}/${QUANT}/build/${TARGET}"
mkdir -p "${GEN_DIR}" "${BUILD_DIR%/*}"

IFS=',' read -ra MODEL_LIST <<< "${MODELS}"

# Stage each constituent model's per-target artifacts.
MODEL_DIRS=""
MODEL_NAMES=""
IR_ARGS=()
for m in "${MODEL_LIST[@]}"; do
    m_gen_dir="${REPO_ROOT}/agents/examples/${m}/${QUANT}/generated/${TARGET}"
    if [[ "${FORCE_REGEN}" == "1" || ! -f "${m_gen_dir}/model.h" ]]; then
        echo "[stage] running agents/examples/${m}/run.sh"
        TARGET="${TARGET}" QUANT="${QUANT}" \
            bash "${REPO_ROOT}/agents/examples/${m}/run.sh" >/dev/null
    fi
    MODEL_DIRS+="${MODEL_DIRS:+;}${m_gen_dir}"
    MODEL_NAMES+="${MODEL_NAMES:+;}${m}"
    IR_ARGS+=(--ir "${m}:${REPO_ROOT}/agents/examples/${m}/${QUANT}/generated/graph.json")
done

# 1) Ingest the schedule into a C dispatch table.
echo "[xpurt] ingest schedule"
SCHED_C="${GEN_DIR}/${SCHED_NAME}.c"
SCHED_H="${GEN_DIR}/${SCHED_NAME}.h"
python -m agents.pipeline.ingest_xpurt_schedule \
    --schedule "${SCHEDULE_JSON}" \
    --registry "${REGISTRY}" \
    "${IR_ARGS[@]}" \
    --out "${SCHED_C}" \
    --name "${SCHED_NAME}" \
    --cpu-p-kind "${CPU_P_KIND}" \
    --cpu-e-kind "${CPU_E_KIND}"

# 2) Generate the schedule-driven main.
echo "[xpurt] generate main"
MAIN_C="${GEN_DIR}/${SCHED_NAME}_main.c"
python -m agents.pipeline.generate_xpurt_main \
    --schedule "${SCHEDULE_JSON}" \
    --out "${MAIN_C}" \
    --name "${SCHED_NAME}" \
    --dispatch-table-header "$(basename "${SCHED_H}")" \
    --core-kinds "${CPU_P_KIND},${CPU_E_KIND}"

# 3) west build harness_xpurt with the generated sources.
echo "[xpurt] west build (TARGET=${TARGET}, pool=${AGENTS_POOL_THREADS})"
KERNEL_CFLAGS=$(python -c "
from agents.pipeline.backends import get
b = get('${TARGET}')
print(';'.join(b.kernel_cflags))
")
WEST_CMAKE_ARGS=(
    "-DAGENTS_BACKEND=${TARGET}"
    "-DMODEL_NAMES=${MODEL_NAMES}"
    "-DMODEL_DIRS=${MODEL_DIRS}"
    "-DXPURT_SCHEDULE_C=${SCHED_C}"
    "-DXPURT_MAIN_C=${MAIN_C}"
    "-DXPURT_INCLUDE_DIR=${GEN_DIR}"
    "-DAGENTS_POOL_THREADS=${AGENTS_POOL_THREADS}"
)
if [[ -n "${KERNEL_CFLAGS}" ]]; then
    WEST_CMAKE_ARGS+=("-DAGENTS_KERNEL_CFLAGS=${KERNEL_CFLAGS}")
fi
# Optional execution-trace capture (XPURT_TRACE={0,1}, default 0). When
# enabled, the binary emits an AGENTS_XPURT_TRACE_BEGIN..END CSV block
# that agents/scripts/plot_xpurt_trace.py renders into a Gantt chart.
if [[ "${XPURT_TRACE:-0}" == "1" ]]; then
    WEST_CMAKE_ARGS+=("-DAGENTS_XPURT_TRACE=ON")
fi
west build -p -b spike_riscv64 agents/harness_xpurt \
    --build-dir "${BUILD_DIR}" \
    -- "${WEST_CMAKE_ARGS[@]}"

# 4) Run on spike + verify.
echo "[xpurt] spike + verify"
SPIKE_ARGS=$(python -c "
from agents.pipeline.backends import get
b = get('${TARGET}')
print(' '.join(b.spike_args))
")
SPIKE_FLAGS=("--spike-arg=-p${SPIKE_HARTS:-4}")
for a in ${SPIKE_ARGS}; do
    SPIKE_FLAGS+=("--spike-arg=${a}")
done
python -m agents.validation.spike_runner \
    --elf "${BUILD_DIR}/zephyr/zephyr.elf" \
    --io  "${REPO_ROOT}/agents/examples/${MODEL_LIST[0]}/${QUANT}/generated/io.npz" \
    --models "${MODELS}" \
    --quant "${QUANT}" \
    --timeout "${SPIKE_TIMEOUT:-900}" \
    "${SPIKE_FLAGS[@]}"
