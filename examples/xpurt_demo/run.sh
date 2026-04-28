#!/usr/bin/env bash
# Round-trip demo: run an XPU-RT-emitted schedule.json through our flow,
# linking MULTIPLE backends per model so the walker can pick the right
# kernel implementation based on the schedule's core_kind for each entry.
#
#   schedule.json + per-network IRs + core registry
#       -> ingest_xpurt_schedule -> dispatch_table.{h,c}
#       -> generate_xpurt_main   -> xpurt_main.c
#       -> west build (harness_xpurt) [linking each model x each backend]
#       -> spike (-p4)
#       -> spike_runner verifies each network's output against PyTorch goldens.
#
# Env knobs:
#   SCHEDULE_JSON   path to scheduled_*.json
#   MODELS          comma list of constituent network names (default: dronet,mlp_control)
#   REGISTRY        path to agents/cores/*.json (default: chipyard_hetero_example.json)
#   BACKENDS        comma list of HW backends to BUILD into the binary
#                   (default: scalar,rvv). Must cover every core_kind in
#                   the schedule.
#   QUANT           fp32 (default)
#   CPU_P_KIND      registry kind for CPU_P slots (default: rvv)
#   CPU_E_KIND      registry kind for CPU_E slots (default: scalar)
#   AGENTS_POOL_THREADS  pthreadpool worker count (default: 4)
#   FORCE_REGEN     {0,1}   re-run each model's run.sh first (default: 1)
#   XPURT_TRACE     {0,1}   enable execution-trace capture (default: 0)
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
BACKENDS="${BACKENDS:-scalar,rvv}"
QUANT="${QUANT:-fp32}"
CPU_P_KIND="${CPU_P_KIND:-rvv}"
CPU_E_KIND="${CPU_E_KIND:-scalar}"
AGENTS_POOL_THREADS="${AGENTS_POOL_THREADS:-4}"
FORCE_REGEN="${FORCE_REGEN:-1}"
SCHED_NAME="${SCHED_NAME:-xpurt_demo_schedule}"

EXAMPLE_DIR="${REPO_ROOT}/agents/examples/xpurt_demo"
GEN_DIR="${EXAMPLE_DIR}/${QUANT}/generated"
# Build dir tag carries the full backend set so cross-backend builds
# don't clobber each other.
BUILD_TAG="$(echo "${BACKENDS}" | tr ',' '_')"
BUILD_DIR="${EXAMPLE_DIR}/${QUANT}/build/${BUILD_TAG}"
mkdir -p "${GEN_DIR}" "${BUILD_DIR%/*}"

IFS=',' read -ra MODEL_LIST <<< "${MODELS}"
IFS=',' read -ra BACKEND_LIST <<< "${BACKENDS}"

# Stage each (model, backend) pair's per-target artifacts. We need
# every backend's generated/<bs>/{model.c,kernels.c,weights.c} to exist
# so the harness can link them all.
MODEL_NAMES=""
MODEL_DIRS_BASE=""
IR_ARGS=()
for m in "${MODEL_LIST[@]}"; do
    m_base="${REPO_ROOT}/agents/examples/${m}/${QUANT}/generated"
    for bs in "${BACKEND_LIST[@]}"; do
        m_gen_dir="${m_base}/${bs}"
        if [[ "${FORCE_REGEN}" == "1" || ! -f "${m_gen_dir}/model.h" ]]; then
            echo "[stage] running agents/examples/${m}/run.sh (TARGET=${bs})"
            TARGET="${bs}" QUANT="${QUANT}" \
                bash "${REPO_ROOT}/agents/examples/${m}/run.sh" >/dev/null
        fi
    done
    MODEL_NAMES+="${MODEL_NAMES:+;}${m}"
    MODEL_DIRS_BASE+="${MODEL_DIRS_BASE:+;}${m_base}"
    IR_ARGS+=(--ir "${m}:${m_base}/graph.json")
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

# 2) Generate the schedule-driven main. core-kinds == backends here
#    (the schedule's core_kind values map 1:1 to our backend tags).
echo "[xpurt] generate main"
MAIN_C="${GEN_DIR}/${SCHED_NAME}_main.c"
python -m agents.pipeline.generate_xpurt_main \
    --schedule "${SCHEDULE_JSON}" \
    --out "${MAIN_C}" \
    --name "${SCHED_NAME}" \
    --dispatch-table-header "$(basename "${SCHED_H}")" \
    --core-kinds "${CPU_P_KIND},${CPU_E_KIND}" \
    --backends "${BACKENDS}" \
    --registry "${REGISTRY}"

# 3) west build harness_xpurt with the generated sources + all backends.
echo "[xpurt] west build (BACKENDS=${BACKENDS}, pool=${AGENTS_POOL_THREADS})"
WEST_CMAKE_ARGS=(
    "-DMODEL_BACKENDS=${BACKENDS}"
    "-DMODEL_NAMES=${MODEL_NAMES}"
    "-DMODEL_DIRS_BASE=${MODEL_DIRS_BASE}"
    "-DXPURT_SCHEDULE_C=${SCHED_C}"
    "-DXPURT_MAIN_C=${MAIN_C}"
    "-DXPURT_INCLUDE_DIR=${GEN_DIR}"
    "-DAGENTS_POOL_THREADS=${AGENTS_POOL_THREADS}"
)
# Per-backend kernel cflags. Read each from agents.pipeline.backends and
# splice into a -DAGENTS_KERNEL_CFLAGS_<BS> variable; the harness CMake
# applies them to that backend's kernels.c source-property only.
for bs in "${BACKEND_LIST[@]}"; do
    BS_UPPER=$(echo "${bs}" | tr '[:lower:]' '[:upper:]')
    KERNEL_CFLAGS=$(python -c "
from agents.pipeline.backends import get
b = get('${bs}')
print(';'.join(b.kernel_cflags))
")
    if [[ -n "${KERNEL_CFLAGS}" ]]; then
        WEST_CMAKE_ARGS+=("-DAGENTS_KERNEL_CFLAGS_${BS_UPPER}=${KERNEL_CFLAGS}")
    fi
done

# Optional execution-trace capture (XPURT_TRACE={0,1}, default 0). When
# enabled, the binary emits an AGENTS_XPURT_TRACE_BEGIN..END CSV block
# that agents/scripts/plot_xpurt_trace.py renders into a Gantt chart.
if [[ "${XPURT_TRACE:-0}" == "1" ]]; then
    WEST_CMAKE_ARGS+=("-DAGENTS_XPURT_TRACE=ON")
fi
west build -p -b spike_riscv64 agents/harness_xpurt \
    --build-dir "${BUILD_DIR}" \
    -- "${WEST_CMAKE_ARGS[@]}"

# 4) Run on spike + verify. Use the union of all backends' spike-args
#    (deduped) so the simulator's --isa is broad enough for whichever
#    backend the schedule routes a given dispatch to.
echo "[xpurt] spike + verify"
SPIKE_ARGS=$(python -c "
from agents.pipeline.backends import get
seen = set()
out = []
for bs in '${BACKENDS}'.split(','):
    for a in get(bs).spike_args:
        if a not in seen:
            seen.add(a); out.append(a)
print(' '.join(out))
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
