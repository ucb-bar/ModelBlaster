#!/usr/bin/env bash
# microros_demo — sweep 4 fixed-HW placements as a baseline against the
# xpurt schedule. The dual-rocket-saturn-gemmini-q31 firesim bitstream
# has 2 harts:
#   hart 0 = gemmini_q31  (CPU_P)
#   hart 1 = rvv          (CPU_E)
#
# We sweep all 4 (dronet_hart, yolov8_hart) combinations to characterize
# how a "no cross-kind scheduler" baseline degrades vs the xpurt run:
#
#   1) dronet@h0(gemmini_q31), yolov8@h0(gemmini_q31)  — both same hart
#   2) dronet@h0(gemmini_q31), yolov8@h1(rvv)          — split, dronet on gemmini
#   3) dronet@h1(rvv),         yolov8@h0(gemmini_q31)  — split, yolov8 on gemmini
#   4) dronet@h1(rvv),         yolov8@h1(rvv)          — both same hart
#
# Each placement: rebuild harness_microros, run on FireSim, snapshot
# uartlog to /tmp/microros_baseline_<tag>.log + into the example's
# baselines/ dir. Plot afterwards via modelblaster/scripts/plot_ros_trace.py.
#
# dronet runs on a 50ms timer; yolov8 is one-shot; the harness terminates
# when yolov8 completes (matches the xpurt schedule's run window).
#
# Pre-reqs:
#   source scripts/activate_conda.sh && conda activate zephyr
#   source scripts/set_envvars_sdk.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
export PATH="/usr/bin:${PATH}"

QUANT="${QUANT:-int8}"
FORCE_REGEN="${FORCE_REGEN:-0}"
FIRESIM_TIMEOUT="${FIRESIM_TIMEOUT:-2400}"
SPIKE_OK="${SPIKE_OK:-0}"   # if 1, also run on spike (for smoke)
RUNNER="${RUNNER:-firesim}"

BASELINES_DIR="${REPO_ROOT}/modelblaster/examples/microros_demo/baselines"
mkdir -p "${BASELINES_DIR}"

# Each entry: tag : pin_backends_csv : pin_harts_csv
PLACEMENTS=(
    "dronet_h0_yolov8_h0:gemmini_q31,gemmini_q31:0,0"
    "dronet_h0_yolov8_h1:gemmini_q31,rvv:0,1"
    "dronet_h1_yolov8_h0:rvv,gemmini_q31:1,0"
    "dronet_h1_yolov8_h1:rvv,rvv:1,1"
)

OVERALL_RC=0
for entry in "${PLACEMENTS[@]}"; do
    IFS=':' read -ra parts <<< "${entry}"
    TAG="${parts[0]}"
    PIN_BACKENDS="${parts[1]}"
    PIN_HARTS="${parts[2]}"
    DRONET_HART="${PIN_HARTS%%,*}"
    YOLOV8_HART="${PIN_HARTS##*,}"

    # Broker hart: prefer the free hart (placements 1 & 4 where both
    # nets share a hart). For split placements (2, 3) put the broker on
    # dronet's hart since dronet has 50 ms gaps between iterations,
    # giving the broker thread plenty of headroom.
    if [[ "${DRONET_HART}" == "${YOLOV8_HART}" ]]; then
        # Both nets on the same hart — broker goes on the other one.
        if [[ "${DRONET_HART}" == "0" ]]; then BROKER_HART=1; else BROKER_HART=0; fi
    else
        BROKER_HART="${DRONET_HART}"
    fi

    echo
    echo "=========================================================="
    echo "[microros_4baselines] placement=${TAG}"
    echo "  dronet -> backend=$(echo "${PIN_BACKENDS}" | cut -d, -f1) hart=${DRONET_HART}"
    echo "  yolov8 -> backend=$(echo "${PIN_BACKENDS}" | cut -d, -f2) hart=${YOLOV8_HART}"
    echo "  broker -> hart=${BROKER_HART}"
    echo "=========================================================="

    rc=0
    MODELS=dronet,yolov8_nano \
    BACKENDS=gemmini_q31,rvv \
    PIN_BACKENDS="${PIN_BACKENDS}" \
    PIN_HARTS="${PIN_HARTS}" \
    PERIODS_MS=50,0 \
    MICROROS_BROKER_HART="${BROKER_HART}" \
    QUANT="${QUANT}" \
    MODELBLASTER_POOL_THREADS=1 \
    FORCE_REGEN="${FORCE_REGEN}" \
    RUNNER="${RUNNER}" \
    FIRESIM_TIMEOUT="${FIRESIM_TIMEOUT}" \
    bash modelblaster/examples/microros_demo/run.sh || rc=$?

    LOG_DST="${BASELINES_DIR}/${TAG}.uartlog.log"
    SRC_LOG="/scratch2/dima/chipyard-fsim/sims/firesim/firesim_rundir/sim_slot_0/uartlog"
    if [[ -f "${SRC_LOG}" ]]; then
        cp "${SRC_LOG}" "${LOG_DST}"
        cp "${SRC_LOG}" "/tmp/microros_baseline_${TAG}.log"
        echo "[microros_4baselines] uartlog -> ${LOG_DST}"
    else
        echo "[microros_4baselines] WARNING: uartlog not found at ${SRC_LOG}"
    fi

    if [[ ${rc} -ne 0 ]]; then
        echo "[microros_4baselines] placement=${TAG} run.sh rc=${rc}"
        OVERALL_RC=${rc}
    fi
done

echo
echo "=========================================================="
echo "[microros_4baselines] all placements done (overall rc=${OVERALL_RC})"
echo "captured uartlogs in: ${BASELINES_DIR}/*.uartlog.log"
echo "=========================================================="
exit ${OVERALL_RC}
