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
#       --runner  <k1|spike|firesim> \
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
RUNNER="k1"
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

# `uv run` was resolving a distribution at file://$REPO/XPU-RT -- a path that
# does not exist -- and failing before any measurement. Use the interpreter
# that already has this checkout importable.
MB_PY="${MB_PY:-/scratch2/agustin/miniforge3/envs/merlin-dev/bin/python}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# ---- step 1: apply hint to IR ----------------------------------------------

# The IR the runner will actually build. The spike/FireSim path reads the
# Zephyr example tree; the K1 path reads the board build tree, and they are
# different files. Pointing the K1 runner at the example tree would rewrite a
# graph nothing then builds -- the rewrite would "succeed" and the measurement
# would be of the unmodified model.
if [[ "${RUNNER}" == "k1" ]]; then
    IR_ORIG="${REPO_ROOT}/build/k1/${MODEL}/${QUANT}/graph.json"
else
    IR_ORIG="${REPO_ROOT}/examples/${MODEL}/${QUANT}/generated/graph.json"
fi
if [[ ! -s "${IR_ORIG}" ]]; then
    echo "FAIL: no IR at ${IR_ORIG} (runner=${RUNNER})" > "${OUT_DIR}/FAIL"
    exit 3
fi
IR_AFTER="${OUT_DIR}/${MODEL}.afterhint.graph.json"

# Detect hint contract to choose the right rewriter.
HINT_CONTRACT="$(/scratch2/agustin/miniforge3/envs/merlin-dev/bin/python -c "
import json,sys
print(json.load(open('${HINT}'))['contract'])
")"

echo "[measure_candidate] hint=${HINT} (${HINT_CONTRACT})  model=${MODEL}" | tee "${OUT_DIR}/run.log"

case "${HINT_CONTRACT}" in
    modelblaster.fusion_hints/v1)
        "${MB_PY}" -m modelblaster.pipeline.apply_fusion_hint \
            --hint "${HINT}" --model "${MODEL}" \
            --ir "${IR_ORIG}" --out "${IR_AFTER}" --pairwise \
            2>&1 | tee -a "${OUT_DIR}/run.log"
        ;;
    modelblaster.split_hints/v1)
        "${MB_PY}" -m modelblaster.pipeline.apply_split_hint \
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

if [[ "${RUNNER}" == "k1" ]]; then
    # THE BOARD PATH. `MB_IR` hands run_model_k1.sh a pre-staged graph and
    # skips extraction, which is the whole point: copying the rewrite over
    # build/k1/<model>/int8/graph.json and then letting step 1/5 re-extract
    # profiles the BASELINE and files the result under the rewrite's name.
    # That is the RVV_fused failure reached through a runbook step.
    #
    # The profile tree is written in place and gen_mb/profile is a symlink, so
    # the baseline results.csv for this model is backed up and restored around
    # the run -- otherwise measuring a candidate destroys the baseline every
    # other number was solved from.
    XPURT_ROOT="${XPURT_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)}"
    PROF_DIR="${XPURT_ROOT}/gen_mb/profile/${TARGET}/spacemit_x60/${MODEL}/${MODEL}.${QUANT}/${MODEL}_spacemit_x60_${TARGET}_${MODEL}.${QUANT}/topo_0"
    PROF_BAK="${OUT_DIR}/baseline_results.csv"
    [[ -f "${PROF_DIR}/results.csv" ]] && cp "${PROF_DIR}/results.csv" "${PROF_BAK}"

    RUN_STATUS=0
    MB_IR="${IR_AFTER}" \
    PROFILE_OUT_ROOT="${XPURT_ROOT}/gen_mb/profile" \
    CROSS="${CROSS:-}" \
        bash "${REPO_ROOT}/scripts/run_model_k1.sh" \
            "${MODEL}" "${QUANT}" "${TARGET}" 0 \
        > "${OUT_DIR}/k1.log" 2>&1 || RUN_STATUS=$?

    # The candidate's profile, before the baseline is put back.
    [[ -f "${PROF_DIR}/results.csv" ]] && cp "${PROF_DIR}/results.csv" "${OUT_DIR}/candidate_results.csv"
    [[ -f "${PROF_BAK}" ]] && cp "${PROF_BAK}" "${PROF_DIR}/results.csv"

    if [[ ${RUN_STATUS} -ne 0 ]]; then
        echo "FAIL: k1 build/run rc=${RUN_STATUS}" > "${OUT_DIR}/FAIL"
        exit 4
    fi
    # CORRECTNESS GATE. A rewrite that changes the answer is ineligible and its
    # timings mean nothing; this is the one gate the RVV_fused precedent
    # skipped.
    if ! grep -qE "max_abs_err=0([^0-9]|$)" "${OUT_DIR}/k1.log"; then
        echo "FAIL: board verify did not report max_abs_err=0" > "${OUT_DIR}/FAIL"
        grep -iE "max_abs_err|MODELBLASTER_VERIFY" "${OUT_DIR}/k1.log" | tail -3 >> "${OUT_DIR}/FAIL"
        exit 5
    fi
    "${MB_PY}" - "${OUT_DIR}/candidate_results.csv" "${OUT_DIR}/measured_cycles.json" "${MODEL}" <<'PY_CYCLES'
import csv, json, sys
src, dst, net = sys.argv[1], sys.argv[2], sys.argv[3]
rows = list(csv.DictReader(open(src)))
per = {int(r["dispatch_id"]): int(float(r.get("cycles") or 0)) for r in rows}
json.dump({"network": net, "runner": "k1",
           "per_dispatch_cycles": per,
           "total_cycles": sum(per.values()),
           "n_dispatches": len(per),
           "_note": "rdtime ticks at 24 MHz, measured on the board"},
          open(dst, "w"), indent=1)
print(f"ingested {len(per)} dispatches, {sum(per.values())} ticks")
PY_CYCLES
    echo "PASS" > "${OUT_DIR}/PASS"
    echo "[measure_candidate] k1: $(cat "${OUT_DIR}/measured_cycles.json" | "${MB_PY}" -c 'import json,sys; d=json.load(sys.stdin); print(d["n_dispatches"], "dispatches,", d["total_cycles"], "ticks")')"
    exit 0
fi

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
