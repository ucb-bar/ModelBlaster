#!/usr/bin/env bash
# Capture FireSim measurements for our CPSAT-scheduled 3-way fixtures.
# These run sequentially via the queue (the script call itself blocks
# until each run completes). XPURT_TRACE=1 records per-dispatch
# start/end in the in-binary array; the trace block is printed ONCE
# at end-of-run (no per-dispatch UART traffic during measurement).
#
# Configs:
#   cpsat_qrb        — 1 yolov8 + 2 dronet + 4 mlp_control (300 ops)
#   cpsat_dronet2_mlp4 — 2 dronet + 4 mlp_control (88 ops, no yolo)
cd "$(dirname "$0")/.."
source scripts/setup_benchmark_env.sh >/dev/null 2>&1 || true
export FIRESIM_QUEUE=1
export RUNNER="firesim"
export FIRESIM_QUEUE_TIMEOUT="3600"
export XPURT_TRACE=1
export BACKENDS="gemmini,rvv_opu"
export REGISTRY="$PWD/cores/chipyard_gemmini_opu_hetero.json"
export CPU_P_KIND="gemmini"
export CPU_E_KIND="rvv_opu"

LOG=/tmp/mb-matrix/cpsat_captures.log
mkdir -p "$(dirname "$LOG")"
: > "$LOG"

CONFIGS=(heft_qrb heft_dronet2_mlp4)
if [[ $# -gt 0 ]]; then CONFIGS=("$@"); fi

for cfg in "${CONFIGS[@]}"; do
  FX="$PWD/schedule_fixtures/3way_${cfg}.json"
  if [[ ! -f "$FX" ]]; then
    echo "MISSING fixture: $FX — skip $cfg" | tee -a "$LOG"
    continue
  fi
  RESULTS_DIR="$PWD/benchmarks/results/A/3way_${cfg}/$(date +%Y%m%dT%H%M%SZ)"
  mkdir -p "$RESULTS_DIR"

  case "$cfg" in
    *dronet2_mlp4)
      export MODELS="dronet,mlp_control"
      export QUANTS="int8,int8"
      ;;
    *)
      export MODELS="yolov8_nano,dronet,mlp_control"
      # All-int8: HEFT/MOSEK fixtures reference linear_s8 / elu_s8 etc
      # (the int8 graph.json), so mlp_control MUST be built int8 to
      # match. fp32 mlp would link the wrong kernels and produce garbage
      # outputs (the earlier heft_qrb FAILED for exactly this reason).
      export QUANTS="int8,int8,int8"
      ;;
  esac

  export SCHEDULE_JSON="$FX"
  safe_name="cpsat_${cfg//-/_}"
  safe_name="${safe_name//cpsat_cpsat_/cpsat_}"
  export SCHED_NAME="$safe_name"

  echo "=== cpsat sweep [$cfg / $safe_name] start $(date +%T) -> $RESULTS_DIR ===" | tee -a "$LOG"
  # FULL build-dir wipe between configs. A previous version only nuked
  # build/.../zephyr, which left the cmake cache + ninja graph in place
  # and let stale main.c targets leak from one config to the next.
  rm -rf "$PWD/examples/xpurt_demo/fp32/build/gemmini_rvv_opu_firesim" 2>/dev/null
  uv run bash examples/xpurt_demo/run.sh 2>&1 | tee "$RESULTS_DIR/run_stdout.log" >> "$LOG"
  rc=${PIPESTATUS[0]}
  echo "  rc=$rc at $(date +%T)" | tee -a "$LOG"

  JOB_ID=$(grep -oE 'job_id=([0-9]+)' "$RESULTS_DIR/run_stdout.log" | head -1 | cut -d= -f2)
  Q_UART=""
  if [[ -n "$JOB_ID" ]]; then
    Q_UART=$(find /scratch2/agustin/chipyard/sims/firesim/deploy/results-workload \
                  -path "*-q${JOB_ID}/*" -name 'uartlog' 2>/dev/null | head -1)
  fi
  if [[ -z "$Q_UART" ]]; then
    Q_UART=$(find /scratch2/agustin/chipyard/sims/firesim/deploy/results-workload \
                  -name 'uartlog' -newer "$RESULTS_DIR" 2>/dev/null | head -1)
  fi
  if [[ -n "$Q_UART" ]]; then
    cp "$Q_UART" "$RESULTS_DIR/uartlog"
    echo "  uartlog (job=${JOB_ID:-?}) -> $RESULTS_DIR/uartlog" | tee -a "$LOG"
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
    print(f"  trace -> {dst} ({body.count(chr(10))+1} rows)")
else:
    print("  WARN: trace block missing (XPURT_TRACE=0 in build?)")
PY
  else
    echo "  WARN: no uartlog found for job=${JOB_ID:-?}" | tee -a "$LOG"
  fi
done
echo "=== cpsat captures done $(date +%T) ===" | tee -a "$LOG"
