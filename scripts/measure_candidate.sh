#!/usr/bin/env bash
# Phase 2 — measurement primitive for the decision loop.
#
# Takes a Contract-2 fuse-or-split hint, applies it to the named
# network's IR, builds + verifies the harness on spike (cheap inner
# oracle), and emits a per-dispatch measured cycles report. Optionally
# escalates to FireSim measurement if --runner firesim is passed (off
# by default because the hetero 1+4+2 workload is unstable; see
# notes/firesim_measured_status.md).
#
# Usage:
#   scripts/measure_candidate.sh \
#       --hint   <hint.json> \
#       --model  <mlp_control|dronet|yolov8_nano> \
#       --target <rvv_opu|gemmini> \
#       --quant  <int8|fp32> \
#       --backend <reference|llm> \
#       --runner  <spike|firesim> \
#       --out-dir <artifacts/decision_loop/round_N_candI/>
#
# Output:
#   <out-dir>/measured_cycles.json   per-dispatch cycles + total
#   <out-dir>/measured_report.json   matches schema of emit_measured_report.py
#   <out-dir>/build.log              build output
#   <out-dir>/spike.log              runner output
#   <out-dir>/PASS or FAIL           verify gate

set -uo pipefail
# yolov8 build+run is ~30s and produces ~200 dispatch rows of profile
# table. `tee | grep ...` upstream produced SIGPIPE that propagated to
# the build/run, killing it with rc=141 mid-run. Disable pipefail on
# the tee leg by writing the log file directly and tailing afterward.
:

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- parse args -------------------------------------------------------------

HINT=""
MODEL=""
TARGET="rvv_opu"
QUANT="int8"
BACKEND="reference"
RUNNER="spike"
OUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hint)    HINT="$2"; shift 2;;
        --model)   MODEL="$2"; shift 2;;
        --target)  TARGET="$2"; shift 2;;
        --quant)   QUANT="$2"; shift 2;;
        --backend) BACKEND="$2"; shift 2;;
        --runner)  RUNNER="$2"; shift 2;;
        --out-dir) OUT_DIR="$2"; shift 2;;
        *) echo "unknown arg: $1" >&2; exit 2;;
    esac
done

for v in HINT MODEL OUT_DIR; do
    if [[ -z "${!v}" ]]; then echo "missing required --$(echo $v | tr A-Z a-z)" >&2; exit 2; fi
done

mkdir -p "${OUT_DIR}"

# ---- step 1: apply hint to IR ----------------------------------------------

IR_ORIG="${REPO_ROOT}/examples/${MODEL}/${QUANT}/generated/graph.json"
IR_AFTER="${OUT_DIR}/${MODEL}.afterhint.graph.json"

# Detect hint contract to choose the right rewriter.
HINT_CONTRACT="$(/scratch2/agustin/miniforge3/envs/merlin-dev/bin/python -c "
import json,sys
print(json.load(open('${HINT}'))['contract'])
")"

echo "[measure_candidate] hint=${HINT} (${HINT_CONTRACT})  model=${MODEL}" | tee "${OUT_DIR}/run.log"

case "${HINT_CONTRACT}" in
    modelblaster.fusion_hints/v1)
        uv run python -m modelblaster.pipeline.apply_fusion_hint \
            --hint "${HINT}" --model "${MODEL}" \
            --ir "${IR_ORIG}" --out "${IR_AFTER}" --pairwise \
            2>&1 | tee -a "${OUT_DIR}/run.log"
        ;;
    modelblaster.split_hints/v1)
        uv run python -m modelblaster.pipeline.apply_split_hint \
            --hint "${HINT}" --model "${MODEL}" \
            --ir "${IR_ORIG}" --out "${IR_AFTER}" \
            2>&1 | tee -a "${OUT_DIR}/run.log"
        ;;
    *)
        echo "FAIL: unknown hint contract ${HINT_CONTRACT}" > "${OUT_DIR}/FAIL"
        exit 3;;
esac

if [[ ! -s "${IR_AFTER}" ]]; then
    echo "FAIL: IR rewrite produced no output" > "${OUT_DIR}/FAIL"
    exit 3
fi

# ---- step 2: swap IR + build + verify --------------------------------------

BACKUP="${OUT_DIR}/${MODEL}.beforehint.graph.json"
cp "${IR_ORIG}" "${BACKUP}"
cp "${IR_AFTER}" "${IR_ORIG}"

# Source the benchmark env so `python`, `west`, etc. resolve.
set +u
source "${REPO_ROOT}/scripts/setup_benchmark_env.sh" 2>&1 >/dev/null || true
set -u

# Run the example pipeline: extract is skipped (IR present), then
# generate_skeleton -> generate_kernels -> build -> spike run.
RUN_STATUS=0
LLM_PROVIDER="${LLM_PROVIDER:-bedrock}" \
BACKEND="${BACKEND}" RUNNER="${RUNNER}" TARGET="${TARGET}" QUANT="${QUANT}" \
    uv run bash "${REPO_ROOT}/examples/${MODEL}/run.sh" \
    > "${OUT_DIR}/spike.log" 2>&1 || RUN_STATUS=$?

# Restore original IR FIRST, regardless of build outcome
cp "${BACKUP}" "${IR_ORIG}"

if [[ ${RUN_STATUS} -ne 0 ]]; then
    echo "FAIL: build/run rc=${RUN_STATUS}" > "${OUT_DIR}/FAIL"
    exit 4
fi

# Check spike PASS line
if ! grep -q "^PASS" "${OUT_DIR}/spike.log"; then
    echo "FAIL: spike verify did not say PASS" > "${OUT_DIR}/FAIL"
    exit 5
fi

# ---- step 3: extract per-dispatch cycles -----------------------------------

/scratch2/agustin/miniforge3/envs/merlin-dev/bin/python "${REPO_ROOT}/scripts/ingest_measured_cycles.py" \
    --spike-log "${OUT_DIR}/spike.log" \
    --out-cycles "${OUT_DIR}/measured_cycles.json" \
    --network "${MODEL}" \
    2>&1 | tee -a "${OUT_DIR}/run.log"

if [[ ! -s "${OUT_DIR}/measured_cycles.json" ]]; then
    echo "FAIL: no cycles ingested" > "${OUT_DIR}/FAIL"
    exit 6
fi

echo "PASS" > "${OUT_DIR}/PASS"
echo "[measure_candidate] OK -> ${OUT_DIR}/measured_cycles.json" | tee -a "${OUT_DIR}/run.log"
exit 0
