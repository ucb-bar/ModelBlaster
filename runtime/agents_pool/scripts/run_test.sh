#!/usr/bin/env bash
# Build + run the agents_pool unit test on spike (always) and on
# firesim (when RUNNER includes "firesim"). Reports PASS/FAIL based on
# the AGENTS_POOL_TEST_{BEGIN,END} block, plus the PERF line so we can
# eyeball that per-call cycles is in the same ballpark as the
# bench_pthreads_raw row in agents/microbench/threadpool/results/firesim_overhead.csv.
#
# Env:
#   RUNNERS=spike,firesim     space- or comma-separated list (default: spike)
#   FIRESIM_TIMEOUT=900
#
# Outputs (under agents/runtime/agents_pool/results/):
#   spike_test.log
#   firesim_test.log     (only when firesim runner is requested)

set -euo pipefail

RUNNERS="${RUNNERS:-spike}"
RUNNERS="${RUNNERS//,/ }"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

RESULTS_DIR="${REPO_ROOT}/agents/runtime/agents_pool/results"
mkdir -p "${RESULTS_DIR}"

run_spike() {
    local build_dir="${REPO_ROOT}/agents/runtime/agents_pool/test_app/build/spike"
    echo "[agents_pool_test] west build (spike)"
    west build -p -b spike_riscv64 agents/runtime/agents_pool/test_app \
        --build-dir "${build_dir}"

    local elf="${build_dir}/zephyr/zephyr.elf"
    local log="${RESULTS_DIR}/spike_test.log"
    echo "[agents_pool_test] spike -p4 ${elf}"
    # rv64gcv_zicntr — same ISA string the bench harness uses; _zicntr
    # enables the unprivileged `cycle` CSR our PERF measurement reads.
    SPIKE="${SPIKE:-spike}"
    ${SPIKE} -p4 --isa=rv64gcv_zicntr "${elf}" 2>&1 | tee "${log}"

    if grep -q "^PASS:" "${log}"; then
        local perf
        perf="$(grep '^PERF:' "${log}" | head -1)"
        echo "[agents_pool_test] spike: PASS"
        echo "[agents_pool_test] spike: ${perf}"
        return 0
    fi
    echo "[agents_pool_test] spike: FAIL — see ${log}" >&2
    return 1
}

run_firesim() {
    local build_dir="${REPO_ROOT}/agents/runtime/agents_pool/test_app/build/firesim"
    echo "[agents_pool_test] west build (firesim)"
    west build -p -b chipyard_riscv64/rocketchip_virt_riscv64 \
        agents/runtime/agents_pool/test_app \
        --build-dir "${build_dir}" \
        -- -DEXTRA_CONF_FILE="${REPO_ROOT}/agents/harness/backends/firesim_chipyard.conf"

    local elf="${build_dir}/zephyr/zephyr.elf"
    local log="${RESULTS_DIR}/firesim_test.log"

    # Coordination — politely wait if another FireSim run is using the FPGA.
    while pgrep -f "FireSim-xilinx_alveo_u250" >/dev/null 2>&1; do
        echo "[agents_pool_test] another FireSim run active; waiting 30s"
        sleep 30
    done

    # Slim adaptation of the bench harness's run_firesim_bench.py — we
    # just need to copy the elf, runworkload, poll uartlog for our
    # END marker, and tear down. Reuses agents/microbench/threadpool
    # /scripts/run_firesim_bench.py's structure but inlined here so we
    # don't depend on the bench harness's module path.
    python3 "${REPO_ROOT}/agents/runtime/agents_pool/scripts/run_firesim.py" \
        --elf "${elf}" \
        --raw-out "${log}" \
        --timeout "${FIRESIM_TIMEOUT:-900}"

    if grep -q "^PASS:" "${log}"; then
        local perf
        perf="$(grep '^PERF:' "${log}" | head -1)"
        echo "[agents_pool_test] firesim: PASS"
        echo "[agents_pool_test] firesim: ${perf}"
        return 0
    fi
    echo "[agents_pool_test] firesim: FAIL — see ${log}" >&2
    return 1
}

ok=1
for r in ${RUNNERS}; do
    case "${r}" in
        spike)   run_spike   || ok=0 ;;
        firesim) run_firesim || ok=0 ;;
        *) echo "unknown runner '${r}'" >&2; ok=0 ;;
    esac
done

[[ ${ok} -eq 1 ]]
