#!/usr/bin/env bash
#
# Sweep harness for KernelBench level1 -> Adreno fp16 (Qualcomm QRB5165 GPU).
#
# Usage:
#   ./run_level1.sh                  # full dataset (--all)
#   ./run_level1.sh --all
#   ./run_level1.sh --skip-ok        # only retry failed/missing IDs
#   ./run_level1.sh 1 19 23          # explicit IDs
#   ITERS=20 ./run_level1.sh 63
#
# Pipeline per ID (each step writes to <work>/level1_<id>/run.log):
#   1. level1_export.py  (PyTorch -> fp32 ONNX, fp16 .raw inputs, meta JSON)
#   2. onnx-simplifier
#   3. snpe-onnx-to-dlc --float_bitwidth 16   (fp16 DLC for Adreno GPU)
#   4. scp DLC + .raw inputs to board
#   5. qnn-net-run with libQnnGpu.so
#
# Failures are isolated per ID. The summary records the failing stage and a
# short stderr excerpt in the "note" field.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEVEL1_DIR="${LEVEL1_DIR:-$SCRIPT_DIR/../zephyr-chipyard-sw/agents/bench/level1}"
CONDA_ENV="${CONDA_ENV:-xpurt}"
BOARD_USER="${BOARD_USER:-root}"
BOARD_IP="${BOARD_IP:-10.44.120.201}"
DOCKER_IMAGE="${DOCKER_IMAGE:-qnn-convert}"
ITERS="${ITERS:-20}"
WORK_DIR="${WORK_DIR:-$SCRIPT_DIR/level1_work}"
RESULTS_FILE="${RESULTS_FILE:-$SCRIPT_DIR/level1_results.json}"
# 300s: GPU shader JIT compilation on Adreno can take 60-120s for first run.
RUN_TIMEOUT="${RUN_TIMEOUT:-300}"

# Parse args
IDS=()
ALL=false
SKIP_OK=false
for arg in "$@"; do
    case "$arg" in
        --all)     ALL=true ;;
        --skip-ok) SKIP_OK=true ;;
        [0-9]*)    IDS+=("$arg") ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done
if [ "${#IDS[@]}" -eq 0 ] && ! $SKIP_OK; then
    ALL=true
fi

if $ALL || $SKIP_OK; then
    mapfile -t IDS < <(
        ls "$LEVEL1_DIR" 2>/dev/null \
            | grep -E '^[0-9]+_.*\.py$' \
            | sed -E 's/^([0-9]+)_.*/\1/' \
            | sort -n
    )
fi

# --skip-ok: filter out IDs that already have status=ok in the results file.
if $SKIP_OK && [ -f "$RESULTS_FILE" ]; then
    mapfile -t OK_IDS < <(python3 -c "
import json, sys
data = json.load(open('$RESULTS_FILE'))
print('\n'.join(k for k,v in data.items() if v.get('status')=='ok'))
" 2>/dev/null)
    declare -A _OK
    for id in "${OK_IDS[@]}"; do _OK[$id]=1; done
    FILTERED=()
    for id in "${IDS[@]}"; do
        [ -z "${_OK[$id]:-}" ] && FILTERED+=("$id")
    done
    IDS=("${FILTERED[@]}")
    echo "Skipping ${#OK_IDS[@]} already-ok IDs; ${#IDS[@]} remaining."
fi

mkdir -p "$WORK_DIR"
RESULTS_TSV="${RESULTS_FILE%.json}.tsv"

# Append one result row immediately after each ID completes.
# Args: id base status stage avg_ms_wallclock avg_ms_gpu_kernel note
flush_result() {
    local id="$1" base="$2" status="$3" stage="$4" latency="$5" gpu_ms="$6" note="$7"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$id" "$base" "$status" "$stage" "$latency" "$gpu_ms" "$note" \
        >> "$RESULTS_TSV"
    # Rewrite JSON from all rows written so far.
    python3 - "$RESULTS_TSV" > "${RESULTS_FILE}.tmp" <<'PYEOF'
import json, sys
data = {}
with open(sys.argv[1]) as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 7:
            continue
        i, base, s, stg, l, g, n = parts
        data[i] = {"name": base, "status": s, "stage": stg,
                   "avg_ms_wallclock": float(l) if l else None,
                   "avg_ms_gpu_kernel": float(g) if g else None,
                   "note": n}
print(json.dumps(data, indent=2))
PYEOF
    mv "${RESULTS_FILE}.tmp" "$RESULTS_FILE"
}

# Clear TSV when sweeping all/skip-ok; for explicit IDs, keep existing rows so
# we can append without losing prior results. Dedup by ID happens at JSON time.
if $ALL || $SKIP_OK; then
    : > "$RESULTS_TSV"
fi

if ! sudo docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
    echo "Docker image $DOCKER_IMAGE not found. Build via deploy.sh first." >&2
    exit 1
fi

docker_run() {
    sudo docker run --rm -v "$WORK_DIR":/workspace "$DOCKER_IMAGE" "$@"
}

find_level1_source() {
    local id="$1"
    find "$LEVEL1_DIR" -maxdepth 1 -name "${id}_*.py" -print -quit
}

# Pick a useful one-line excerpt from a stderr log.
note_from_log() {
    local logfile="$1"
    [ -s "$logfile" ] || { echo ""; return; }
    grep -E -m1 \
        -e 'Error|error|Unsupported|unsupported|Traceback|raise|FAILED|not implemented|NotImplementedError|RuntimeError|ValueError' \
        "$logfile" 2>/dev/null \
        | tail -1 | head -c 220
}

declare -A STATUS LATENCY NOTE STAGE
TOTAL=${#IDS[@]}
DONE=0

echo "Sweeping ${TOTAL} level1 IDs ($ITERS iters/run, board=$BOARD_IP)"

for ID in "${IDS[@]}"; do
    DONE=$((DONE+1))
    NAME="level1_${ID}"
    STAGE_DIR="$WORK_DIR/$NAME"
    rm -rf "$STAGE_DIR"
    mkdir -p "$STAGE_DIR"
    LOG="$STAGE_DIR/run.log"
    : > "$LOG"

    SRC="$(find_level1_source "$ID")"
    if [ -z "$SRC" ]; then
        STATUS[$ID]="missing"; STAGE[$ID]="discover"; NOTE[$ID]="no level1 file"
        printf '[%3d/%3d] %4s  %-22s  ----\n' "$DONE" "$TOTAL" "$ID" "missing"
        flush_result "$ID" "missing" "missing" "discover" "" "" "no level1 file"
        continue
    fi
    BASE="$(basename "$SRC" .py)"

    # 1. Export ONNX
    if ! conda run -n "$CONDA_ENV" --no-capture-output \
            python3 "$SCRIPT_DIR/level1_export.py" \
                --source "$SRC" --out-dir "$STAGE_DIR" --name "$NAME" \
                >>"$LOG" 2>&1; then
        STATUS[$ID]="fail"; STAGE[$ID]="export"
        NOTE[$ID]="$(note_from_log "$LOG")"
        printf '[%3d/%3d] %4s  %-22s  export   %s\n' "$DONE" "$TOTAL" "$ID" "$BASE" "${NOTE[$ID]}"
        flush_result "$ID" "$BASE" "fail" "export" "" "" "${NOTE[$ID]}"
        continue
    fi

    META="$STAGE_DIR/${NAME}.json"
    LAYOUT_FLAGS="$(python3 -c "
import json
m = json.load(open('$META'))
print(' '.join(f'--input_layout {i[\"name\"]} NCHW' for i in m['inputs'] if i['rank'] == 4))
")"

    # 2 + 3. Simplify + convert in Docker
    CONVERT_LOG="$STAGE_DIR/convert.log"
    if ! docker_run bash -c "
        set -e
        python3.10 -c \"
import onnx
from onnxsim import simplify
m = onnx.load('/workspace/${NAME}/${NAME}.onnx')
ms, ok = simplify(m)
assert ok, 'simplify failed'
onnx.save(ms, '/workspace/${NAME}/${NAME}_simplified.onnx')
\" && \
        python3.10 /qnn/bin/x86_64-linux-clang/snpe-onnx-to-dlc \
            --input_network /workspace/${NAME}/${NAME}_simplified.onnx \
            ${LAYOUT_FLAGS} \
            --float_bitwidth 16 \
            --output_path /workspace/${NAME}/${NAME}.dlc
    " >"$CONVERT_LOG" 2>&1; then
        cat "$CONVERT_LOG" >> "$LOG"
        STATUS[$ID]="fail"; STAGE[$ID]="convert"
        NOTE[$ID]="$(note_from_log "$CONVERT_LOG")"
        printf '[%3d/%3d] %4s  %-22s  convert  %s\n' "$DONE" "$TOTAL" "$ID" "$BASE" "${NOTE[$ID]}"
        flush_result "$ID" "$BASE" "fail" "convert" "" "" "${NOTE[$ID]}"
        continue
    fi
    cat "$CONVERT_LOG" >> "$LOG"
    sudo chown -R "$(id -u):$(id -g)" "$STAGE_DIR" 2>/dev/null || true

    DLC="$STAGE_DIR/${NAME}.dlc"
    if [ ! -s "$DLC" ]; then
        STATUS[$ID]="fail"; STAGE[$ID]="convert"; NOTE[$ID]="DLC missing/empty"
        printf '[%3d/%3d] %4s  %-22s  convert  DLC empty\n' "$DONE" "$TOTAL" "$ID" "$BASE"
        flush_result "$ID" "$BASE" "fail" "convert" "" "" "DLC missing/empty"
        continue
    fi

    # 4. Deploy
    BOARD_DIR="/root/models/level1/$NAME"
    if ! ssh -o ConnectTimeout=10 "${BOARD_USER}@${BOARD_IP}" \
            "rm -rf $BOARD_DIR && mkdir -p $BOARD_DIR" >>"$LOG" 2>&1; then
        STATUS[$ID]="fail"; STAGE[$ID]="board_unreachable"; NOTE[$ID]="ssh failed"
        printf '[%3d/%3d] %4s  %-22s  ssh      board unreachable\n' "$DONE" "$TOTAL" "$ID" "$BASE"
        flush_result "$ID" "$BASE" "fail" "board_unreachable" "" "" "ssh failed"
        continue
    fi
    scp -q "$DLC" "${BOARD_USER}@${BOARD_IP}:$BOARD_DIR/${NAME}.dlc" 2>>"$LOG"
    scp -q "$STAGE_DIR/inputs/"*.raw "${BOARD_USER}@${BOARD_IP}:$BOARD_DIR/" 2>>"$LOG"

    INPUT_LINE="$(python3 -c "
import json
m = json.load(open('$META'))
print(' '.join(f'$BOARD_DIR/{i[\"name\"]}.raw' for i in m['inputs']))
")"
    LIST_LOCAL="$STAGE_DIR/input_list.txt"
    : > "$LIST_LOCAL"
    for _ in $(seq 1 "$ITERS"); do echo "$INPUT_LINE" >> "$LIST_LOCAL"; done
    scp -q "$LIST_LOCAL" "${BOARD_USER}@${BOARD_IP}:$BOARD_DIR/input_list.txt" 2>>"$LOG"

    # 5. Run on Adreno GPU (with timeout) + collect per-op profiling
    RUN_LOG="$STAGE_DIR/board_run.log"
    PROF_CSV="$STAGE_DIR/profile.csv"
    BENCH_OUT="$(timeout "$RUN_TIMEOUT" ssh "${BOARD_USER}@${BOARD_IP}" "
        QNN_SDK_ROOT=\${QNN_SDK_ROOT:-/root/qairt}
        export LD_LIBRARY_PATH=\$QNN_SDK_ROOT/lib/target:\${LD_LIBRARY_PATH:-}
        START=\$(date +%s%N)
        \$QNN_SDK_ROOT/bin/target/qnn-net-run \\
            --dlc_path $BOARD_DIR/${NAME}.dlc \\
            --backend \$QNN_SDK_ROOT/lib/target/libQnnGpu.so \\
            --input_list $BOARD_DIR/input_list.txt \\
            --output_dir $BOARD_DIR/output \\
            --profiling_level detailed >/tmp/${NAME}_run.log 2>&1
        RC=\$?
        END=\$(date +%s%N)
        if [ \$RC -ne 0 ]; then
            echo RUN_FAILED
            tail -8 /tmp/${NAME}_run.log
            exit \$RC
        fi
        python3 -c \"print((\$END - \$START) / 1e6 / $ITERS)\"
        echo CSV_BEGIN
        \$QNN_SDK_ROOT/bin/target/qnn-profile-viewer \\
            --input_log $BOARD_DIR/output/qnn-profiling-data.log \\
            --output /tmp/${NAME}_profile.csv 2>/dev/null \\
            && cat /tmp/${NAME}_profile.csv || true
        echo CSV_END
    " 2>&1)"
    RC=$?
    # Split board output into run log and profiling CSV
    echo "$BENCH_OUT" | sed '/CSV_BEGIN/,/CSV_END/d' > "$RUN_LOG"
    echo "$BENCH_OUT" | sed -n '/CSV_BEGIN/,/CSV_END/{/CSV_BEGIN/d;/CSV_END/d;p}' > "$PROF_CSV"

    if [ $RC -ne 0 ] || echo "$BENCH_OUT" | grep -q RUN_FAILED; then
        STATUS[$ID]="fail"; STAGE[$ID]="run"
        NOTE[$ID]="$(note_from_log "$RUN_LOG")"
        if [ $RC -eq 124 ]; then NOTE[$ID]="timeout >${RUN_TIMEOUT}s"; fi
        printf '[%3d/%3d] %4s  %-22s  run      %s\n' "$DONE" "$TOTAL" "$ID" "$BASE" "${NOTE[$ID]}"
        flush_result "$ID" "$BASE" "fail" "run" "" "" "${NOTE[$ID]}"
        continue
    fi

    AVG_MS="$(grep -v CSV_BEGIN "$RUN_LOG" | tail -1)"

    # Parse per-op GPU kernel time from profiling CSV (skip first iteration = GPU JIT warmup).
    GPU_MS="$(python3 - "$PROF_CSV" <<'PYEOF'
import sys, csv
try:
    text = open(sys.argv[1]).read()
except Exception:
    print(''); sys.exit(0)
lines = text.split('\n')
csv_start = next((i for i, l in enumerate(lines) if l.startswith('Msg Timestamp,')), -1)
if csv_start < 0:
    print(''); sys.exit(0)
reader = csv.reader(lines[csv_start:])
try:
    next(reader)
except StopIteration:
    print(''); sys.exit(0)
iterations, cur = [], None
for row in reader:
    if len(row) < 6:
        continue
    msg, source, level = row[1].strip(), row[4].strip(), row[5].strip()
    try:
        t = int(row[2].strip())
    except ValueError:
        continue
    if msg == 'EXECUTE' and level == 'ROOT' and source == 'BACKEND':
        cur = {'backend_us': t, 'ops_us': 0}
        iterations.append(cur)
    elif msg == 'EXECUTE' and level == 'SUB-EVENT' and cur is not None:
        cur['ops_us'] += t
steady = iterations[1:]  # skip first (GPU shader JIT)
if not steady:
    print('')
else:
    # Use min(backend_root, sub_event_sum) per iteration:
    # - small single-op models: sub_event_sum < backend_root (gives pure kernel time)
    # - large tiled models: backend_root < sub_event_sum (avoids double-counting
    #   overlapping tile dispatches; backend_root = first-tile-start to last-tile-end)
    def kern(it):
        s = it['ops_us']
        return min(it['backend_us'], s) if s > 0 else it['backend_us']
    print(f'{sum(kern(it) for it in steady) / len(steady) / 1000.0:.4f}')
PYEOF
)"

    STATUS[$ID]="ok"; STAGE[$ID]="run"; LATENCY[$ID]="$AVG_MS"
    printf '[%3d/%3d] %4s  %-22s  ok  wc=%.2f ms  gpu=%.3f ms\n' \
        "$DONE" "$TOTAL" "$ID" "$BASE" "$AVG_MS" "${GPU_MS:-0}"
    flush_result "$ID" "$BASE" "ok" "run" "$AVG_MS" "${GPU_MS}" ""
done

# --- Summary JSON (already written incrementally; just print stats) ---

echo ""
echo "=================================================="
echo "Summary -> $RESULTS_FILE"
echo "=================================================="
python3 - "$RESULTS_FILE" <<'EOF'
import json, sys, collections
data = json.load(open(sys.argv[1]))
ok   = [k for k,v in data.items() if v['status']=='ok']
fail = [k for k,v in data.items() if v['status']!='ok']
print(f"  OK   {len(ok)}/{len(data)}")
buckets = collections.Counter(v['stage'] for k,v in data.items() if v['status']!='ok')
for stage, n in buckets.most_common():
    print(f"  fail/{stage:<10} {n}")
EOF
