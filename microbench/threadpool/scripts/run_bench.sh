#!/usr/bin/env bash
# Build + run a single threadpool microbench binary.
#
# Env:
#   BENCH=<pthreadpool|zephyr_threads|pthreads_raw>   (required)
#   VARIANT=<default|spin>                            (only meaningful for
#                                                       BENCH=pthreadpool)
#   RUNNER=<spike|firesim>                            (default: spike)
#   FIRESIM_TIMEOUT=600
#
# Output: writes the captured uartlog (or spike stdout) to
#   build/<bench>_<variant>_<runner>/raw.log
# Plus the parsed CSV block alone to
#   build/<bench>_<variant>_<runner>/bench.csv
set -euo pipefail

BENCH="${BENCH:?missing BENCH=...}"
VARIANT="${VARIANT:-default}"
RUNNER="${RUNNER:-spike}"

case "${BENCH}" in
    pthreadpool|zephyr_threads|pthreads_raw) ;;
    *) echo "BENCH=${BENCH} not supported" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

case "${RUNNER}" in
    spike)   BOARD="spike_riscv64" ;;
    firesim) BOARD="chipyard_riscv64/rocketchip_virt_riscv64" ;;
    *) echo "RUNNER=${RUNNER} not supported" >&2; exit 2 ;;
esac

TAG="${BENCH}_${VARIANT}_${RUNNER}"
BUILD_DIR="agents/microbench/threadpool/build/${TAG}"

WEST_ARGS=( "-DBENCH_TARGET=${BENCH}" )
if [[ "${BENCH}" == "pthreadpool" ]]; then
    WEST_ARGS+=( "-DBENCH_PTHREADPOOL_VARIANT=${VARIANT}" )
fi
if [[ "${RUNNER}" == "firesim" ]]; then
    WEST_ARGS+=( "-DEXTRA_CONF_FILE=${REPO_ROOT}/agents/harness/backends/firesim_chipyard.conf" )
fi

if [[ "${SKIP_BUILD:-0}" == "1" && -f "${BUILD_DIR}/zephyr/zephyr.elf" ]]; then
    echo "[microbench] SKIP_BUILD=1 — reusing ${BUILD_DIR}/zephyr/zephyr.elf"
else
    echo "[microbench] west build BENCH=${BENCH} VARIANT=${VARIANT} RUNNER=${RUNNER}"
    west build -p -b "${BOARD}" agents/microbench/threadpool \
        --build-dir "${BUILD_DIR}" -- "${WEST_ARGS[@]}"
fi

ELF="${BUILD_DIR}/zephyr/zephyr.elf"
RAW_LOG="${BUILD_DIR}/raw.log"
CSV_OUT="${BUILD_DIR}/bench.csv"

case "${RUNNER}" in
    spike)
        SPIKE="${SPIKE:-spike}"
        echo "[microbench] spike -p4 ${ELF}"
        # rv64gcv_zicntr — enable V extension AND the unprivileged
        # `cycle` CSR. Without _zicntr spike traps every rdcycle as
        # illegal-instruction, which kills the workers as soon as they
        # try to stamp a wake cycle. Matches what the production
        # backends.py uses for spike runs.
        ${SPIKE} -p4 --isa=rv64gcv_zicntr "${ELF}" 2>&1 | tee "${RAW_LOG}"
        ;;
    firesim)
        echo "[microbench] firesim runworkload"
        # Use the same conda + sourceme prologue as firesim_runner.py.
        FIRESIM_ROOT="${FIRESIM_ROOT:-/scratch2/dima/chipyard-fsim/sims/firesim}"
        FIRESIM_ENV="${FIRESIM_ENV:-/scratch2/dima/chipyard-fsim/env.sh}"
        FIRESIM_SLOT="${FIRESIM_SLOT:-firesim_rundir/sim_slot_0}"
        FIRESIM_TIMEOUT="${FIRESIM_TIMEOUT:-900}"
        # Coordination: politely wait if another FireSim run is active.
        while pgrep -f "FireSim-xilinx_alveo_u250" >/dev/null 2>&1; do
            echo "  ... another FireSim run active; sleeping 30s"
            sleep 30
        done
        python3 "${REPO_ROOT}/agents/microbench/threadpool/scripts/run_firesim_bench.py" \
            --elf "${ELF}" \
            --firesim-root "${FIRESIM_ROOT}" \
            --firesim-env "${FIRESIM_ENV}" \
            --firesim-slot "${FIRESIM_SLOT}" \
            --timeout "${FIRESIM_TIMEOUT}" \
            --raw-out "${RAW_LOG}" \
            --csv-out "${CSV_OUT}"
        ;;
esac

# For spike, parse out the THREADPOOL_BENCH_{BEGIN,END} block.
if [[ "${RUNNER}" == "spike" ]]; then
    python3 "${REPO_ROOT}/agents/microbench/threadpool/scripts/parse_log.py" \
        --raw "${RAW_LOG}" --out "${CSV_OUT}"
fi

echo "[microbench] CSV written to ${CSV_OUT}"
