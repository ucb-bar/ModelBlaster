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
#   REGISTRY        path to modelblaster/cores/*.json (default: chipyard_hetero_example.json)
#   BACKENDS        comma list of HW backends to BUILD into the binary
#                   (default: scalar,rvv). Must cover every core_kind in
#                   the schedule.
#   QUANT           fp32 (default)
#   CPU_P_KIND      registry kind for CPU_P slots (default: rvv)
#   CPU_E_KIND      registry kind for CPU_E slots (default: scalar)
#   MODELBLASTER_POOL_THREADS  modelblaster_pool worker count (default: 4)
#   FORCE_REGEN     {0,1}   re-run each model's run.sh first (default: 1)
#   XPURT_TRACE     {0,1}   enable execution-trace capture (default: 0)
#   RUNNER          {spike,firesim}  default spike. firesim picks the
#                   chipyard_riscv64 board + firesim_chipyard.conf overlay
#                   and dispatches to firesim_runner.py instead of spike.
#
# Pre-reqs (from repo root):
#   source tools/miniforge3/etc/profile.d/conda.sh && conda activate zephyr
#   source scripts/set_envvars_sdk.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PATH="/usr/bin:${PATH}"

# Per-stage timing markers. Mirrors the helper in examples/_run_lib.sh
# so hetero workloads (which dispatch through this runner) emit the
# same MODELBLASTER_STAGE_BEGIN/END events the arm driver's stdout
# parser already understands. Without this, hetero cells had
# extract_s / generate_kernels_s / build_s / run_s all null in the
# dashboard.
_MB_STAGE_T=0
_mb_stage_begin() {
    _MB_STAGE_T="$(date +%s.%N)"
    echo "MODELBLASTER_STAGE_BEGIN:$1"
}
_mb_stage_end() {
    local end="$(date +%s.%N)"
    local delta
    delta=$(awk -v a="${_MB_STAGE_T}" -v b="${end}" \
            'BEGIN{printf "%.3f", b-a}')
    echo "MODELBLASTER_STAGE_END:$1:${delta}"
}

# Default points at an XPU-RT-produced schedule in the sibling checkout
# (see XPU-RT README "Run XPU-RT Scheduler"). Override SCHEDULE_JSON to use
# a different one. XPURT_ROOT overrides the XPU-RT checkout location.
SCHEDULE_JSON="${SCHEDULE_JSON:-${XPURT_ROOT:-$(dirname "${REPO_ROOT}")/XPU-RT}/schedules/scheduled_networks_periodic_profiled.json}"
MODELS="${MODELS:-dronet,mlp_control}"
REGISTRY="${REGISTRY:-${REPO_ROOT}/cores/chipyard_hetero_example.json}"
BACKENDS="${BACKENDS:-scalar,rvv}"
QUANT="${QUANT:-fp32}"
# Optional per-model quant override (parallel to MODELS, comma list).
# Lets you build a binary that mixes e.g. fp32 mlp_control with int8
# dronet/yolov8 — each model picks its own
# modelblaster/examples/<m>/<quant>/generated/<backend> tree at stage time.
# Nothing in the harness CMake or generated C requires a shared quant:
# each model's kernels are model-suffixed (kernel_conv2d_s8_dronet vs
# kernel_linear_mlp_control), each has its own model_{input,output}_t,
# and each has its own weights.c — so int8 weight arrays and fp32
# weight arrays coexist in one binary fine. Default falls back to the
# single QUANT for every model so the dronet+mlp_control fp32 demo
# keeps working without changes.
QUANTS="${QUANTS:-}"
CPU_P_KIND="${CPU_P_KIND:-rvv}"
CPU_E_KIND="${CPU_E_KIND:-scalar}"
MODELBLASTER_POOL_THREADS="${MODELBLASTER_POOL_THREADS:-4}"
FORCE_REGEN="${FORCE_REGEN:-1}"
SCHED_NAME="${SCHED_NAME:-xpurt_demo_schedule}"
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

EXAMPLE_DIR="${REPO_ROOT}/examples/xpurt_demo"
GEN_DIR="${EXAMPLE_DIR}/${QUANT}/generated"
# Build dir tag carries the full backend set so cross-backend builds
# don't clobber each other; appended _firesim when targeting the chipyard
# board so the spike build doesn't get reused.
BUILD_TAG="$(echo "${BACKENDS}" | tr ',' '_')"
if [[ "${RUNNER}" == "firesim" ]]; then
    BUILD_TAG="${BUILD_TAG}_firesim"
fi
BUILD_DIR="${EXAMPLE_DIR}/${QUANT}/build/${BUILD_TAG}"
mkdir -p "${GEN_DIR}" "${BUILD_DIR%/*}"

IFS=',' read -ra MODEL_LIST <<< "${MODELS}"
IFS=',' read -ra BACKEND_LIST <<< "${BACKENDS}"

# Resolve per-model quant. If QUANTS is set, must have one entry per
# model in MODELS; otherwise every model uses the single QUANT default.
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

# Stage each (model, backend) pair's per-target artifacts. We need
# every backend's generated/<bs>/{model.c,kernels.c,weights.c} to exist
# so the harness can link them all.
MODEL_NAMES=""
MODEL_DIRS_BASE=""
IR_ARGS=()
for idx in "${!MODEL_LIST[@]}"; do
    m="${MODEL_LIST[$idx]}"
    m_quant="${QUANT_LIST[$idx]}"
    m_base="${REPO_ROOT}/examples/${m}/${m_quant}/generated"
    for bs in "${BACKEND_LIST[@]}"; do
        m_gen_dir="${m_base}/${bs}"
        if [[ "${FORCE_REGEN}" == "1" || ! -f "${m_gen_dir}/model.h" ]]; then
            echo "[stage] running modelblaster/examples/${m}/run.sh (TARGET=${bs}, QUANT=${m_quant})"
            # Staging just needs the generated/ artifacts (graph.json,
            # model.{h,c}, kernels.c, weights.c, buffers.c, test_io.h);
            # the per-model run.sh's [4/5] west build + [5/5] simulator
            # run are wasted work here. Force RUNNER=spike so we don't
            # accidentally fire up FireSim N times (one per model x
            # backend) while xpurt_demo itself is targeting firesim.
            # GLOBAL_CURATED_DIR must be forwarded here -- without it,
            # generate_kernels.py's reference-backend curated-swap block
            # (gated on `if global_curated_dir is not None`) never fires,
            # so every op silently falls back to spec.reference_impl
            # (generic scalar C) instead of the fast, verified-correct
            # curated kernel (e.g. conv2d_s8's rvv_vsmul_vnclip on RVV).
            # Confirmed root cause of a ~14-16x cycle-count blowup AND a
            # wrong-output regression (max_abs_err=51) on dronet in the
            # combined binary -- reference_impl is both far slower
            # (no RVV vectorization) and not bit-exact for this op,
            # while the curated kernel is verified bit-exact standalone.
            TARGET="${bs}" QUANT="${m_quant}" RUNNER=spike \
                GLOBAL_CURATED_DIR="${GLOBAL_CURATED_DIR:-${REPO_ROOT}/kernels}" \
                bash "${REPO_ROOT}/examples/${m}/run.sh" >/dev/null
        fi
    done
    MODEL_NAMES+="${MODEL_NAMES:+;}${m}"
    MODEL_DIRS_BASE+="${MODEL_DIRS_BASE:+;}${m_base}"
    IR_ARGS+=(--ir "${m}:${m_base}/graph.json")
done

# 1) Ingest the schedule into a C dispatch table.
# Hetero workloads have no distinct extract/skeleton stages here --
# the per-model run.sh invocations at the top of the file already
# emitted those for each (model, backend) pair. So the hetero
# pipeline only meaningfully measures ingest + generate_kernels (as
# one "generate_kernels" bucket for symmetry with the single-target
# stage taxonomy) + build + run. The arm driver tolerates missing
# stages (records null) so we don't need to backfill extract /
# generate_skeleton from here.
_mb_stage_begin generate_kernels
echo "[xpurt] ingest schedule"
SCHED_C="${GEN_DIR}/${SCHED_NAME}.c"
SCHED_H="${GEN_DIR}/${SCHED_NAME}.h"
python -m modelblaster.pipeline.ingest_xpurt_schedule \
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
python -m modelblaster.pipeline.generate_xpurt_main \
    --schedule "${SCHEDULE_JSON}" \
    --out "${MAIN_C}" \
    --name "${SCHED_NAME}" \
    --dispatch-table-header "$(basename "${SCHED_H}")" \
    --core-kinds "${CPU_P_KIND},${CPU_E_KIND}" \
    --backends "${BACKENDS}" \
    --registry "${REGISTRY}"

# 3) west build harness_xpurt with the generated sources + all backends.
_mb_stage_end generate_kernels
_mb_stage_begin build
echo "[xpurt] west build (BACKENDS=${BACKENDS}, pool=${MODELBLASTER_POOL_THREADS})"
WEST_CMAKE_ARGS=(
    "-DMODEL_BACKENDS=${BACKENDS}"
    "-DMODEL_NAMES=${MODEL_NAMES}"
    "-DMODEL_DIRS_BASE=${MODEL_DIRS_BASE}"
    "-DXPURT_SCHEDULE_C=${SCHED_C}"
    "-DXPURT_MAIN_C=${MAIN_C}"
    "-DXPURT_INCLUDE_DIR=${GEN_DIR}"
    "-DMODELBLASTER_POOL_THREADS=${MODELBLASTER_POOL_THREADS}"
)
# Per-backend kernel cflags. Read each from modelblaster.pipeline.backends and
# splice into a -DMODELBLASTER_KERNEL_CFLAGS_<BS> variable; the harness CMake
# applies them to that backend's kernels.c source-property only.  Use
# `resolved_kernel_cflags(repo_root)` so backends that bake the repo
# root into include paths (gemmini's -isystem<repo_root>/modelblaster/cores/gemmini)
# get those substitutions applied.
for bs in "${BACKEND_LIST[@]}"; do
    BS_UPPER=$(echo "${bs}" | tr '[:lower:]' '[:upper:]')
    KERNEL_CFLAGS=$(python -c "
from modelblaster.pipeline.backends import get
b = get('${bs}')
print(';'.join(b.resolved_kernel_cflags('${REPO_ROOT}')))
")
    if [[ -n "${KERNEL_CFLAGS}" ]]; then
        WEST_CMAKE_ARGS+=("-DMODELBLASTER_KERNEL_CFLAGS_${BS_UPPER}=${KERNEL_CFLAGS}")
    fi
done

# Optional execution-trace capture (XPURT_TRACE={0,1}, default 0). When
# enabled, the binary emits an MODELBLASTER_XPURT_TRACE_BEGIN..END CSV block
# that modelblaster/scripts/plot_xpurt_trace.py renders into a Gantt chart.
if [[ "${XPURT_TRACE:-0}" == "1" ]]; then
    WEST_CMAKE_ARGS+=("-DMODELBLASTER_XPURT_TRACE=ON")
fi

WEST_BUILD_EXTRA=()
# Per-target overlay sets CONFIG_MP_MAX_NUM_CPUS — not in prj.conf, so
# every target callsite has to declare its topology.
if [[ "${RUNNER}" == "firesim" ]]; then
    # Honor FIRESIM_CONF override (e.g. firesim_chipyard_dual_gemmini.conf
    # for the 2-core saturn-gemmini bitstream).  Otherwise auto-pick by
    # backend list: gemmini in the build → dual-gemmini overlay (2 harts),
    # else default 4-hart overlay.
    if [[ -n "${FIRESIM_CONF:-}" ]]; then
        EXTRA_CONF="${REPO_ROOT}/harness/backends/${FIRESIM_CONF}"
    elif [[ ",${BACKENDS}," == *,gemmini,* || ",${BACKENDS}," == *,gemmini_q31,* ]]; then
        # Both float-scale (gemmini) and Q0.31 (gemmini_q31) variants run
        # on the same dual-rocket-saturn-gemmini SoC topology — the
        # bitstream selection is driven by config_runtime.yaml's
        # default_hw_config, not by the Zephyr Kconfig overlay.
        EXTRA_CONF="${REPO_ROOT}/harness/backends/firesim_chipyard_dual_gemmini.conf"
    else
        EXTRA_CONF="${REPO_ROOT}/harness/backends/firesim_chipyard.conf"
    fi
elif [[ "${RUNNER}" == "spike" ]]; then
    # Spike runs default to -p4 (see SPIKE_FLAGS below); the overlay
    # matches with MP_MAX_NUM_CPUS=4. Override SPIKE_CONF if you point
    # spike_runner at a different -p value.
    EXTRA_CONF="${REPO_ROOT}/harness/backends/${SPIKE_CONF:-spike_quad.conf}"
fi
if [[ -z "${EXTRA_CONF:-}" || ! -f "${EXTRA_CONF}" ]]; then
    echo "ERROR: per-target overlay not found (RUNNER=${RUNNER}, EXTRA_CONF=${EXTRA_CONF:-<unset>})" >&2
    exit 1
fi
WEST_BUILD_EXTRA+=(
    -DEXTRA_CONF_FILE="${EXTRA_CONF}"
)

west build -p -b "${BOARD_TARGET}" harness_xpurt \
    --build-dir "${BUILD_DIR}" \
    -- "${WEST_CMAKE_ARGS[@]}" "${WEST_BUILD_EXTRA[@]}"

# 4) Run + verify. Spike: union of all backends' spike-args (deduped) so
#    the simulator's --isa covers whichever backend the schedule routes a
#    given dispatch to. FireSim: hardware is fixed (RVV-capable rocket),
#    so just hand the elf to firesim_runner.
_mb_stage_end build

# STOP_AFTER=build: emit the harness ELF and stop, skipping stage 4's
# simulator/FireSim run. Mirrors the identically-named knob in
# examples/_run_lib.sh: lets an external scheduler (deploy/fpga_queue ->
# AWS F2) own the hardware run instead of firesim_runner.py driving a
# LOCAL chipyard-fsim tree's `firesim infrasetup` + `runworkload`.
if [[ "${STOP_AFTER:-}" == "build" ]]; then
    echo "[xpurt] STOP_AFTER=build -> stopping before the run stage"
    echo "ELF: ${BUILD_DIR}/zephyr/zephyr.elf"
    ls -l "${BUILD_DIR}/zephyr/zephyr.elf" 2>/dev/null || echo "WARNING: elf not found"
    exit 0
fi

_mb_stage_begin run
echo "[xpurt] ${RUNNER} + verify"

# Per-backend verify tolerance. A model's dispatches may be routed to ANY of
# the linked BACKENDS by the schedule, so the binary's numeric envelope is the
# loosest of them (e.g. gemmini_q31 declares atol_override=128 for its Q0.31
# requantize drift, while rvv is bit-exact). Without this, the combined verify
# runs at atol=rtol=0 and a gemmini-routed model fails on legitimate drift.
# Mirrors the TOL_FLAGS logic in examples/_run_lib.sh.
TOL_FLAGS=$(python -c "
from modelblaster.pipeline.backends import get
atol = None
rtol = None
for bs in '${BACKENDS}'.split(','):
    b = get(bs.strip())
    if b.atol_override is not None:
        atol = b.atol_override if atol is None else max(atol, b.atol_override)
    if b.rtol_override is not None:
        rtol = b.rtol_override if rtol is None else max(rtol, b.rtol_override)
parts = []
if atol is not None:
    parts.append(f'--atol={atol}')
if rtol is not None:
    parts.append(f'--rtol={rtol}')
print(' '.join(parts))
")
if [[ -n "${TOL_FLAGS}" ]]; then
    echo "[xpurt] verify tolerance (max over BACKENDS=${BACKENDS}): ${TOL_FLAGS}"
fi

if [[ "${RUNNER}" == "spike" ]]; then
    SPIKE_ARGS=$(python -c "
from modelblaster.pipeline.backends import get
seen = set()
out = []
isa_candidates = []
for bs in '${BACKENDS}'.split(','):
    for a in get(bs).spike_args:
        if a.startswith('--isa='):
            # Spike accepts only ONE --isa flag — collect candidates and
            # keep the longest (each extension letter expands the
            # decoded ISA, so the longest string is a superset of any
            # shorter ones we'd merge with). gemmini emits 'rv64gc_zicntr'
            # and rvv emits 'rv64gcv_zicntr' — picking the longer one
            # (gcv) preserves both backends' opcodes.
            isa_candidates.append(a)
            continue
        if a not in seen:
            seen.add(a); out.append(a)
if isa_candidates:
    out.append(max(isa_candidates, key=len))
print(' '.join(out))
")
    SPIKE_FLAGS=("--spike-arg=-p${SPIKE_HARTS:-4}")
    for a in ${SPIKE_ARGS}; do
        SPIKE_FLAGS+=("--spike-arg=${a}")
    done
    # When the build mixes gemmini in (BACKENDS=gemmini,...), the spike
    # binary needs --extension=gemmini support, which only the
    # chipyard-built spike has. Default-pick that one if it exists;
    # SPIKE_BIN env var overrides explicitly.
    # spike-hetero is the merlin-side wrapper that loads BOTH the
    # Gemmini and Saturn-OPU extensions into one spike process; it is
    # the only spike build capable of running hetero workloads where
    # BACKENDS contains both gemmini and rvv_opu. The arm driver's
    # hetero_env_overlay points this env var at it when RUNNER=spike on
    # a hetero workload. Takes precedence over the chipyard-spike
    # autopick below so an explicitly-set wrapper is honored.
    if [[ -n "${MODELBLASTER_HETERO_SPIKE:-}" && -z "${SPIKE_BIN:-}" ]]; then
        if [[ -x "${MODELBLASTER_HETERO_SPIKE}" ]]; then
            SPIKE_BIN="${MODELBLASTER_HETERO_SPIKE}"
        else
            echo "ERROR: MODELBLASTER_HETERO_SPIKE=${MODELBLASTER_HETERO_SPIKE} is not executable" >&2
            exit 1
        fi
    fi
    SPIKE_BIN_DEFAULT="/scratch2/dima/chipyard-fsim/.conda-env/riscv-tools/bin/spike"
    if [[ ",${BACKENDS}," == *,gemmini,* && -z "${SPIKE_BIN:-}" ]]; then
        if [[ -x "${SPIKE_BIN_DEFAULT}" ]]; then
            SPIKE_BIN="${SPIKE_BIN_DEFAULT}"
        else
            echo "ERROR: BACKENDS includes gemmini but no chipyard spike found at ${SPIKE_BIN_DEFAULT}; set SPIKE_BIN explicitly" >&2
            exit 1
        fi
    fi
    if [[ -n "${SPIKE_BIN:-}" ]]; then
        SPIKE_FLAGS+=("--spike" "${SPIKE_BIN}")
    fi
    # When XPURT_TRACE=1 (auto-enabled by setting XPURT_SAVE_OUTPUT, or
    # explicit), capture full spike stdout so plot_xpurt_trace.py can
    # find the MODELBLASTER_XPURT_TRACE_BEGIN..END block.
    if [[ -n "${XPURT_SAVE_OUTPUT:-}" ]]; then
        SPIKE_FLAGS+=("--save-output" "${XPURT_SAVE_OUTPUT}")
    fi
    # Per-model golden lookup for mixed-quant builds (QUANTS=fp32,int8,int8
    # etc.) -- without this, every model's golden was looked up under the
    # single blanket ${QUANT} (defaulting to fp32), so any model actually
    # built at a different quant than the first one in MODELS failed
    # verification with "golden not found" even though it ran correctly.
    # spike_runner.py already supports --io-paths (net=path,...); the
    # firesim branch below builds the equivalent --quants flag, this
    # branch just never did. Falls back to the single --io/--quant path
    # when QUANTS isn't set (single-quant builds), matching prior behavior.
    _spike_io_flags=()
    if [[ -n "${QUANTS:-}" ]]; then
        _io_paths=""
        for idx in "${!MODEL_LIST[@]}"; do
            m="${MODEL_LIST[$idx]}"
            m_quant="${QUANT_LIST[$idx]}"
            _io_paths+="${_io_paths:+,}${m}=${REPO_ROOT}/examples/${m}/${m_quant}/generated/io.npz"
        done
        _spike_io_flags+=("--io-paths" "${_io_paths}")
    fi
    python -m modelblaster.validation.spike_runner \
        --elf "${BUILD_DIR}/zephyr/zephyr.elf" \
        --io  "${REPO_ROOT}/examples/${MODEL_LIST[0]}/${QUANT_LIST[0]}/generated/io.npz" \
        --models "${MODELS}" \
        --quant "${QUANT}" \
        "${_spike_io_flags[@]}" \
        --timeout "${SPIKE_TIMEOUT:-900}" \
        ${TOL_FLAGS} \
        "${SPIKE_FLAGS[@]}"
else
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
    # Pass per-model quants when QUANTS was set (mixed-quant builds).
    _quant_flag=()
    if [[ -n "${QUANTS:-}" ]]; then
        _quant_flag+=("--quants" "${QUANTS}")
    fi
    # Use the FIRST model's quant for the io.npz fallback path (single
    # --io arg). per-model golden lookup uses --quants when present.
    _first_quant="${QUANT}"
    if [[ -n "${QUANTS:-}" ]]; then
        _first_quant="${QUANTS%%,*}"
    fi
    python -m modelblaster.validation.firesim_runner \
        --elf "${BUILD_DIR}/zephyr/zephyr.elf" \
        --io  "${REPO_ROOT}/examples/${MODEL_LIST[0]}/${_first_quant}/generated/io.npz" \
        --models "${MODELS}" \
        --quant "${QUANT}" \
        ${TOL_FLAGS} \
        "${_quant_flag[@]}" \
        "${FIRESIM_FLAGS[@]}"
fi
_mb_stage_end run
