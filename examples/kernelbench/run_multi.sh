#!/usr/bin/env bash
# Fuse several KernelBench level1 problems into ONE ELF and run them in a single
# spike/firesim invocation. On firesim this amortizes the (lengthy) per-problem
# infrasetup + runworkload over N problems — one FPGA setup, N cycle counts.
#
# Each bench is staged (extract/skeleton/kernels only, via STAGE_ONLY) into its
# own generated/<target>/ dir; harness_multi then links them all into a single
# binary with per-model mangled symbols + name-tagged output. The runner parses
# each model's block and compares it to that bench's io.npz golden (passed via
# --io-paths, since the flat kb_<name> tag doesn't map to examples/<tag>/).
#
# Env: BENCHES=comma-list of level1 basenames (required);
#      TARGET={scalar,rvv} (default rvv); QUANT={fp32,fp16} (default fp32);
#      RUNNER={spike,firesim} (default spike); SPIKE_HARTS (default 4);
#      SPIKE_RAM_SIZE / SPIKE_MEM_MB (stock dims); BENCH_MAX_ELEMENTS (default 65536).
set -euo pipefail
: "${BENCHES:?set BENCHES=comma-list of level1 basenames}"
: "${TARGET:=rvv}"; : "${QUANT:=fp32}"; : "${RUNNER:=spike}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# Parent repo on PYTHONPATH so this script's own `python -m/-c modelblaster.*`
# calls resolve (the staged run_one.sh sets this internally, but our direct
# backend/runner calls run outside that).
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
MB="${REPO_ROOT}/modelblaster"
KB_DIR="${MB}/examples/kernelbench"
BENCH_DIR="${MB}/bench/level1"
GEN_TARGET="${TARGET}"; [[ "${QUANT}" == "fp16" ]] && GEN_TARGET="${TARGET}_f16"
export PATH="/usr/bin:${PATH}"

IFS=',' read -ra LIST <<< "${BENCHES}"
NAMES=""; DIRS=""; IOPATHS=""
for b in "${LIST[@]}"; do
    f="${BENCH_DIR}/${b}.py"; [[ -f "$f" ]] || f="$(ls ${BENCH_DIR}/${b}*.py 2>/dev/null | head -1)"
    [[ -f "$f" ]] || { echo "SKIP $b (not found)" >&2; continue; }
    kb="kb_$(echo "$(basename "$f" .py)" | sed 's/[^A-Za-z0-9_]/_/g; s/__*/_/g; s/^_//; s/_$//')"
    echo "[stage] ${kb}"
    BENCH_FILE="$f" TARGET="${TARGET}" QUANT="${QUANT}" RUNNER=spike \
        BACKEND=reference FORCE_EXTRACT="${FORCE_EXTRACT:-1}" STAGE_ONLY=1 \
        bash "${KB_DIR}/run_one.sh" >/dev/null
    gd="${KB_DIR}/${kb}/${QUANT}/generated/${GEN_TARGET}"
    io="${KB_DIR}/${kb}/${QUANT}/generated/io.npz"
    NAMES+="${NAMES:+;}${kb}"; DIRS+="${DIRS:+;}${gd}"
    IOPATHS+="${IOPATHS:+,}${kb}=${io}"
done
[[ -n "${NAMES}" ]] || { echo "no benches staged" >&2; exit 1; }

BUILD_DIR="${KB_DIR}/_multi/${QUANT}/build/${GEN_TARGET}_${RUNNER}"
mkdir -p "${BUILD_DIR%/*}"
CF=$(python -c "from modelblaster.pipeline.backends import get; print(';'.join(get('${GEN_TARGET}').resolved_kernel_cflags('${MB}')))")

case "${RUNNER}" in
    spike)   BOARD="spike_riscv64" ;;
    firesim) BOARD="chipyard_riscv64/rocketchip_virt_riscv64" ;;
    *) echo "ERROR: RUNNER=${RUNNER} not supported (spike|firesim)" >&2; exit 1 ;;
esac

WEST_EXTRA=()
if [[ "${RUNNER}" == "spike" && -n "${SPIKE_RAM_SIZE:-}" ]]; then
    OVL="${BUILD_DIR%/build/*}/spike_ram.overlay"; mkdir -p "$(dirname "$OVL")"
    echo "&ram0 { reg = < 0x80000000 ${SPIKE_RAM_SIZE} >; };" > "$OVL"
    WEST_EXTRA+=(-DEXTRA_DTC_OVERLAY_FILE="${OVL}")
fi
if [[ "${RUNNER}" == "firesim" ]]; then
    WEST_EXTRA+=(-DEXTRA_CONF_FILE="${MB}/harness/backends/${FIRESIM_CONF:-firesim_chipyard.conf}")
fi

echo "[multi] west build (${#LIST[@]} benches, board=${BOARD}) -> ${BUILD_DIR}"
cd "${REPO_ROOT}"
west build -p -b "${BOARD}" "${MB}/harness_multi" --build-dir "${BUILD_DIR}" -- \
    -DMODELBLASTER_BACKEND="${GEN_TARGET}" \
    -DMODEL_NAMES="${NAMES}" -DMODEL_DIRS="${DIRS}" \
    ${CF:+-DMODELBLASTER_KERNEL_CFLAGS="${CF}"} "${WEST_EXTRA[@]}"

MODELS="${NAMES//;/,}"
TOL=$(python -c "
from modelblaster.pipeline.backends import get
b=get('${GEN_TARGET}'); p=[]
if b.atol_override is not None: p.append(f'--atol={b.atol_override}')
if b.rtol_override is not None: p.append(f'--rtol={b.rtol_override}')
print(' '.join(p))")

echo "[multi] ${RUNNER} run (${MODELS})"
if [[ "${RUNNER}" == "spike" ]]; then
    SPIKE_ARGS=$(python -c "from modelblaster.pipeline.backends import get; print(' '.join(get('${GEN_TARGET}').spike_args))")
    SF=(); for a in ${SPIKE_ARGS}; do SF+=("--spike-arg=${a}"); done
    SF+=("--spike-arg=-p${SPIKE_HARTS:-4}")
    [[ -n "${SPIKE_MEM_MB:-}" ]] && SF+=("--spike-arg=-m${SPIKE_MEM_MB}")
    python -m modelblaster.validation.spike_runner \
        --elf "${BUILD_DIR}/zephyr/zephyr.elf" \
        --models "${MODELS}" --io-paths "${IOPATHS}" --quant "${QUANT}" \
        --timeout "${SPIKE_TIMEOUT:-1200}" ${TOL} "${SF[@]}"
else
    FF=()
    [[ -n "${FIRESIM_ROOT:-}" ]] && FF+=("--firesim-root=${FIRESIM_ROOT}")
    [[ -n "${FIRESIM_ENV:-}" ]] && FF+=("--firesim-env=${FIRESIM_ENV}")
    [[ -n "${FIRESIM_SLOT:-}" ]] && FF+=("--firesim-slot=${FIRESIM_SLOT}")
    [[ -n "${FIRESIM_TIMEOUT:-}" ]] && FF+=("--timeout=${FIRESIM_TIMEOUT}")
    python -m modelblaster.validation.firesim_runner \
        --elf "${BUILD_DIR}/zephyr/zephyr.elf" \
        --models "${MODELS}" --io-paths "${IOPATHS}" --quant "${QUANT}" \
        ${TOL} "${FF[@]}"
fi
