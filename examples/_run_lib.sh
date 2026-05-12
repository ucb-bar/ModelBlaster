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

# When quant=fp16, promote the target to its fp16-capable backend variant
# (e.g. rvv → rvv_f16) for stages that need zvfh/zfh compiler flags and
# spike ISA extensions. Directory layout still uses TARGET so fp32 and
# fp16 builds share the same paths under the quant-namespaced tree.
GEN_TARGET="${TARGET}"
if [[ "${QUANT}" == "fp16" ]]; then
    GEN_TARGET="${TARGET}_f16"
fi

EXAMPLE_DIR_REL="agents/examples/${MODEL_NAME}"
EXAMPLE_DIR="${REPO_ROOT}/${EXAMPLE_DIR_REL}"

cd "${REPO_ROOT}"
# Avoid the stale Vitis cmake on PATH (it's 3.3.2 and breaks west).
export PATH="/usr/bin:${PATH}"

IR_DIR="${EXAMPLE_DIR}/${QUANT}/generated"
GEN_DIR="${IR_DIR}/${TARGET}"
# Two build dirs because the LLM verify path (inside generate_kernels)
# always invokes spike with board=spike_riscv64, while RUNNER=firesim's
# runtime build uses chipyard_riscv64. west refuses to mix board
# targets in the same build dir, so:
#   VERIFY_BUILD_DIR — always spike, used by the BACKEND=llm verify loop
#   BUILD_DIR        — runtime, suffixed _firesim when applicable
# For RUNNER=spike the two are the same (no suffix). cache/ is shared
# (kernel source isn't board-dependent).
VERIFY_BUILD_DIR="${EXAMPLE_DIR}/${QUANT}/build/${TARGET}"
BUILD_SUFFIX=""
if [[ "${RUNNER:-spike}" == "firesim" ]]; then
    BUILD_SUFFIX="_firesim"
fi
BUILD_DIR="${EXAMPLE_DIR}/${QUANT}/build/${TARGET}${BUILD_SUFFIX}"
CACHE_DIR="${EXAMPLE_DIR}/${QUANT}/cache/${TARGET}"
mkdir -p "${GEN_DIR}" "${BUILD_DIR%/*}" "${CACHE_DIR}"

echo "[1/5] extract_graph (quant=${QUANT}) -> ${IR_DIR}"
# Skip the PyTorch extract pass when the IR is already on disk. Useful
# when (a) the active env lacks the model's PyTorch deps (set up the IR
# in a different env first), or (b) iterating on later stages without
# re-running tracing. Set FORCE_EXTRACT=1 to override.
if [[ -f "${IR_DIR}/graph.json" && -f "${IR_DIR}/weights.npz" && -f "${IR_DIR}/io.npz" && "${FORCE_EXTRACT:-0}" != "1" ]]; then
    echo "  (skipped — IR present at ${IR_DIR}; set FORCE_EXTRACT=1 to re-run)"
else
    python -m agents.pipeline.extract_graph \
        --model "${MODEL_NAME}" \
        --out-dir "${IR_DIR}" \
        --quant "${QUANT}"
fi

echo "[2/5] generate_skeleton (backend=${TARGET}) -> ${GEN_DIR}"
python -m agents.pipeline.generate_skeleton \
    --ir "${IR_DIR}/graph.json" \
    --weights "${IR_DIR}/weights.npz" \
    --io "${IR_DIR}/io.npz" \
    --out-dir "${GEN_DIR}" \
    --backend "${TARGET}"

echo "[3/5] generate_kernels (backend=${BACKEND} target=${GEN_TARGET} quant=${QUANT} optimize=${OPTIMIZE}) -> ${GEN_DIR}"
GEN_KERNELS_ARGS=(
    --ir "${IR_DIR}/graph.json"
    --out-dir "${GEN_DIR}"
    --backend "${BACKEND}"
    --target "${GEN_TARGET}"
    --quant "${QUANT}"
    --io "${IR_DIR}/io.npz"
    --repo-root "${REPO_ROOT}"
    --build-dir "${VERIFY_BUILD_DIR}"
    --harness-dir "agents/harness"
    --cache-dir "${CACHE_DIR}"
    --algorithms "${ALGORITHMS:-all}"
)
if [[ -n "${GLOBAL_CURATED_DIR:-}" ]]; then
    GEN_KERNELS_ARGS+=(--global-curated-dir "${GLOBAL_CURATED_DIR}")
fi
# MAX_ACCURACY_CLASS=bit_exact|numeric_drift|approximate restricts kernel
# selection to algorithms that meet at least the given accuracy class. Use
# bit_exact for golden-regression runs; default (unset) keeps the
# atol=8 envelope behavior.
if [[ -n "${MAX_ACCURACY_CLASS:-}" ]]; then
    GEN_KERNELS_ARGS+=(--max-accuracy-class "${MAX_ACCURACY_CLASS}")
fi
if [[ "${OPTIMIZE}" == "1" ]]; then
    GEN_KERNELS_ARGS+=(
        --optimize
        --beam "${BEAM:-2}"
        --expansions "${EXPANSIONS:-3}"
        --iterations "${ITERATIONS:-2}"
    )
    # Memory-aware optimize knobs. Both default off — set FIRESIM_EVAL=1
    # to re-rank top-K spike survivors on the FireSim FPGA and promote
    # the firesim-best to cache. Pair with CACHE_AWARE_PROMPT=1 to also
    # splice the target's memory-hierarchy stanza into the LLM optimize
    # prompt. FIRESIM_OPS is a comma-list to limit re-rank to specific
    # ops (e.g. "conv2d,linear") and skip cheap elementwise ops that
    # don't benefit.
    if [[ "${FIRESIM_EVAL:-0}" == "1" ]]; then
        GEN_KERNELS_ARGS+=(
            --firesim-eval
            --firesim-top-k "${FIRESIM_TOP_K:-3}"
        )
        if [[ -n "${FIRESIM_OPS:-}" ]]; then
            GEN_KERNELS_ARGS+=(--firesim-ops "${FIRESIM_OPS}")
        fi
    fi
    if [[ "${CACHE_AWARE_PROMPT:-0}" == "1" ]]; then
        GEN_KERNELS_ARGS+=(--cache-aware-prompt)
    fi
fi
python -m agents.pipeline.generate_kernels "${GEN_KERNELS_ARGS[@]}"

# RUNNER selects the simulator behind stages 4-5: spike (default; in-process
# spike subprocess) or firesim (build for chipyard_riscv64, copy elf into
# the FireSim sim slot, runworkload, tail uartlog). The build (4/5) and
# run (5/5) split is identical across runners — only the board name and
# the verifier differ.
RUNNER="${RUNNER:-spike}"
case "${RUNNER}" in
    spike)
        BOARD_TARGET="spike_riscv64"
        ;;
    firesim)
        # Chipyard's quad-rocket-saturn board target. Pulls in the
        # firesim_chipyard.conf overlay (shrunk stack + SMP knobs that
        # the working FireSim Zephyr samples use) so Zephyr boots on
        # the FPGA — the spike-only prj.conf hangs pre-banner there.
        BOARD_TARGET="chipyard_riscv64/rocketchip_virt_riscv64"
        ;;
    *)
        echo "ERROR: unsupported RUNNER=${RUNNER} (expected spike|firesim)" >&2
        exit 1
        ;;
esac

echo "[4/5] west build (board=${BOARD_TARGET}) -> ${BUILD_DIR}"
KERNEL_CFLAGS=$(python -c "
from agents.pipeline.backends import get
b = get('${GEN_TARGET}')
print(';'.join(b.resolved_kernel_cflags('${REPO_ROOT}')))
")
WEST_CMAKE_ARGS=(
    -DMODEL_DIR="${GEN_DIR}"
    -DAGENTS_BACKEND="${GEN_TARGET}"
)
if [[ -n "${KERNEL_CFLAGS}" ]]; then
    WEST_CMAKE_ARGS+=(-DAGENTS_KERNEL_CFLAGS="${KERNEL_CFLAGS}")
fi
WEST_BUILD_EXTRA=()
if [[ "${RUNNER}" == "firesim" ]]; then
    # Splice the firesim overlay through Zephyr's EXTRA_CONF_FILE knob.
    # `west build -- -DEXTRA_CONF_FILE=...` arrives as a CMake -D, which
    # Zephyr picks up before find_package(Zephyr) processes Kconfig.
    # Pick the overlay matching the active FireSim hwconfig — the
    # quad-rocket and dual-rocket-gemmini bitstreams have different
    # hart counts so MP_MAX_NUM_CPUS must match. Override via
    # FIRESIM_CONF env if running a different config.
    if [[ -n "${FIRESIM_CONF:-}" ]]; then
        FS_CONF="${REPO_ROOT}/agents/harness/backends/${FIRESIM_CONF}"
    elif [[ "${GEN_TARGET}" == "gemmini" || "${GEN_TARGET}" == "gemmini_q31" ]]; then
        # Both float-scale (gemmini) and Q0.31 (gemmini_q31) variants ride
        # the same dual-rocket-saturn-gemmini SoC topology, so the same
        # Zephyr SMP overlay applies. The runtime bitstream is selected
        # via config_runtime.yaml::default_hw_config.
        FS_CONF="${REPO_ROOT}/agents/harness/backends/firesim_chipyard_dual_gemmini.conf"
    else
        FS_CONF="${REPO_ROOT}/agents/harness/backends/firesim_chipyard.conf"
    fi
    WEST_BUILD_EXTRA+=(
        -DEXTRA_CONF_FILE="${FS_CONF}"
    )
fi
west build -p -b "${BOARD_TARGET}" agents/harness \
    --build-dir "${BUILD_DIR}" \
    -- "${WEST_CMAKE_ARGS[@]}" "${WEST_BUILD_EXTRA[@]}"

echo "[5/5] ${RUNNER} + compare"

# Optional IREE-shape per-dispatch profile (PROFILE_OUT_ROOT env).
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
        "--profile-cores=${PROFILE_CORES:-0}"
        "--profile-clock-mhz=${PROFILE_CLOCK_MHZ:-1000.0}"
    )
    if [[ -n "${PROFILE_CPU:-}" ]]; then
        PROFILE_FLAGS+=("--profile-cpu=${PROFILE_CPU}")
    fi
fi

# Per-backend verify tolerance applies to BOTH spike and firesim — gemmini's
# float-scale and Q0.31 requantize paths each drift ~1 int8 LSB per layer
# vs the PyTorch Q0.31 golden, well-covered by atol=8 on shallow nets.
# Backend.atol_override / rtol_override are the authoritative source.
TOL_FLAGS=$(python -c "
from agents.pipeline.backends import get
b = get('${GEN_TARGET}')
parts = []
if b.atol_override is not None:
    parts.append(f'--atol={b.atol_override}')
if b.rtol_override is not None:
    parts.append(f'--rtol={b.rtol_override}')
print(' '.join(parts))
")

if [[ "${RUNNER}" == "spike" ]]; then
    SPIKE_ARGS=$(python -c "
from agents.pipeline.backends import get
b = get('${GEN_TARGET}')
print(' '.join(b.spike_args))
")
    SPIKE_FLAGS=()
    for a in ${SPIKE_ARGS}; do
        SPIKE_FLAGS+=("--spike-arg=${a}")
    done
    # Gemmini backend needs the chipyard spike (has --extension=gemmini support
    # + libgemmini.so). Use AGENTS_GEMMINI_SPIKE env if set, else chipyard path.
    SPIKE_BIN_FLAGS=()
    if [[ "${GEN_TARGET}" == "gemmini" ]]; then
        _GEMMINI_SPIKE="${AGENTS_GEMMINI_SPIKE:-/scratch2/dima/chipyard-fsim/.conda-env/riscv-tools/bin/spike}"
        _GEMMINI_LIB_DIR="${AGENTS_GEMMINI_LIB_DIR:-/scratch2/dima/chipyard-fsim/.conda-env/riscv-tools/lib}"
        if [[ -f "${_GEMMINI_SPIKE}" ]]; then
            SPIKE_BIN_FLAGS+=(--spike "${_GEMMINI_SPIKE}")
            export LD_LIBRARY_PATH="${_GEMMINI_LIB_DIR}:${LD_LIBRARY_PATH:-}"
        fi
    fi
    python -m agents.validation.spike_runner \
        --elf "${BUILD_DIR}/zephyr/zephyr.elf" \
        --io "${IR_DIR}/io.npz" \
        --timeout "${SPIKE_TIMEOUT:-600}" \
        ${TOL_FLAGS} \
        "${SPIKE_BIN_FLAGS[@]}" \
        "${SPIKE_FLAGS[@]}" \
        "${PROFILE_FLAGS[@]}"
else
    # firesim: the runner copies the elf into the sim slot, runs
    # firesim runworkload, tails the uartlog until OUTPUT_END, then
    # firesim kill. FIRESIM_ROOT / FIRESIM_ENV / FIRESIM_SLOT env vars
    # override the install paths.
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
        --io "${IR_DIR}/io.npz" \
        ${TOL_FLAGS} \
        "${FIRESIM_FLAGS[@]}" \
        "${PROFILE_FLAGS[@]}"
fi
