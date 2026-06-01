#!/usr/bin/env bash
# Capture the gemmini+rvv (plain-RVV) pair schedule on FireSim.
# This uses BACKENDS=gemmini,rvv (rather than gemmini,rvv_opu) so
# generate_kernels picks from kernels/rvv/ on hart 1. The OPU bitstream
# runs plain RVV unchanged (OPU = RVV + extensions).
cd "$(dirname "$0")/.."
source scripts/setup_benchmark_env.sh >/dev/null 2>&1 || true
export FIRESIM_QUEUE=1
export RUNNER="firesim"
export FIRESIM_QUEUE_TIMEOUT="3600"
export XPURT_TRACE=1
export BACKENDS="gemmini,rvv_hetero"   # <-- rvv_hetero = plain RVV kernels + V-kernel-only Kconfig (avoids hart-0 trap)
export GLOBAL_CURATED_DIR="$PWD/kernels"
export REGISTRY="$PWD/cores/chipyard_gemmini_rvv_hetero.json"
export CPU_P_KIND="gemmini"
export CPU_E_KIND="rvv_hetero"
export MODELS="yolov8_nano_64,dronet,mlp_control"
export QUANTS="int8,int8,int8"

LOG=/tmp/mb-matrix/gemmini_rvv_capture.log
mkdir -p "$(dirname "$LOG")"
: > "$LOG"

cfg="heft_qrb_y64_gemmini_rvv"
FX="$PWD/schedule_fixtures/3way_${cfg}.json"
export SCHEDULE_JSON="$FX"
export SCHED_NAME="cpsat_${cfg//-/_}"
RESULTS_DIR="$PWD/benchmarks/results/A/3way_${cfg}/$(date +%Y%m%dT%H%M%SZ)"
mkdir -p "$RESULTS_DIR"

echo "=== gemmini+rvv capture [$cfg] start $(date +%T) -> $RESULTS_DIR ===" | tee -a "$LOG"
rm -rf "$PWD/examples/xpurt_demo/fp32/build/gemmini_rvv_opu_firesim" 2>/dev/null
rm -rf "$PWD/examples/xpurt_demo/fp32/build/gemmini_rvv_firesim" 2>/dev/null
uv run bash examples/xpurt_demo/run.sh 2>&1 | tee "$RESULTS_DIR/run_stdout.log" >> "$LOG"
rc=${PIPESTATUS[0]}
echo "  rc=$rc at $(date +%T)" | tee -a "$LOG"

JOB_ID=$(grep -oE 'job_id=([0-9]+)' "$RESULTS_DIR/run_stdout.log" | head -1 | cut -d= -f2)
Q_UART=""
if [[ -n "$JOB_ID" ]]; then
  Q_UART=$(find /scratch2/agustin/chipyard/sims/firesim/deploy/results-workload \
                -path "*-q${JOB_ID}/*" -name 'uartlog' 2>/dev/null | head -1)
fi
[[ -z "$Q_UART" ]] && Q_UART=$(find /scratch2/agustin/chipyard/sims/firesim/deploy/results-workload \
                -name 'uartlog' -newer "$RESULTS_DIR" 2>/dev/null | head -1)
if [[ -n "$Q_UART" ]]; then
  cp "$Q_UART" "$RESULTS_DIR/uartlog"
  python3 - "$RESULTS_DIR/uartlog" "$RESULTS_DIR/xpurt_trace.csv" <<'PY'
import sys
src, dst = sys.argv[1:3]
text = open(src).read()
B = "=== MODELBLASTER_XPURT_TRACE_BEGIN ==="
E = "=== MODELBLASTER_XPURT_TRACE_END ==="
if B in text and E in text:
    body = text[text.index(B)+len(B):text.index(E, text.index(B))].strip()
    with open(dst, "w") as f:
        f.write(body + "\n")
PY
fi
echo "=== done $(date +%T) ===" | tee -a "$LOG"
