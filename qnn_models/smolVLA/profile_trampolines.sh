#!/usr/bin/env bash
#
# Push trampoline-phase DLCs to the board, build CPU ctx binaries, then
# profile each one via profile_seg (wallclock around QnnGraph_execute).
# Results are aggregated into trampolines_perf.json and merged back into
# the main segment_perf.json so the scheduler can use realistic HTA times
# (= conv time + trampoline time).
#
# Usage:
#   ./profile_trampolines.sh                  # full sweep (50 iters)
#   ./profile_trampolines.sh --iters 100
#   ./profile_trampolines.sh --skip-build     # skip ctx generation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAMP_DLC_DIR="$SCRIPT_DIR/vision_slices_v3/trampolines/dlc"
PROFILE_DIR="$REPO_ROOT/qnn_models/boards/qrb5165_v66/profiles/smolvlm_vision_v3"

BOARD="${QNN_BOARD_HOST:-root@<BOARD_IP>}"
REMOTE_BASE="/root/models/smolvlm_vision_v3"
REMOTE_CTX="$REMOTE_BASE/ctx"
ITERS="${ITERS:-50}"
SKIP_BUILD=false

while [ $# -gt 0 ]; do
    case "$1" in
        --iters) ITERS="$2"; shift 2;;
        --skip-build) SKIP_BUILD=true; shift;;
        *) echo "Unknown arg: $1" >&2; exit 1;;
    esac
done

echo "Trampoline-phase profiling sweep"
echo "  Board: $BOARD"
echo "  Iters: $ITERS"
echo "  DLC dir: $TRAMP_DLC_DIR"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Push trampoline DLCs to the board
# ---------------------------------------------------------------------------
echo "--- Step 1: Push trampoline DLCs ---"
ssh "$BOARD" "mkdir -p $REMOTE_BASE/trampoline_dlc"
n_pushed=0
for dlc in "$TRAMP_DLC_DIR"/*.dlc; do
    [ -f "$dlc" ] || continue
    scp -q "$dlc" "$BOARD:$REMOTE_BASE/trampoline_dlc/"
    n_pushed=$((n_pushed + 1))
done
echo "  Pushed $n_pushed trampoline DLCs"
echo ""

# ---------------------------------------------------------------------------
# Step 2: Build CPU ctx binaries on board (one per phase DLC)
# ---------------------------------------------------------------------------
if [ "$SKIP_BUILD" = "false" ]; then
    echo "--- Step 2: Build CPU ctx binaries on board ---"
    ssh "$BOARD" bash <<'EOF'
set +e
cd /root/models/smolvlm_vision_v3/ctx
QNN=/root/qairt
export LD_LIBRARY_PATH=$QNN/lib/target
N_OK=0; N_FAIL=0; N_SKIP=0
for dlc in /root/models/smolvlm_vision_v3/trampoline_dlc/*.dlc; do
    [ -f "$dlc" ] || continue
    name="$(basename "$dlc" .dlc)"
    bin="ctx_${name}__Cpu.bin"
    if [ -f "$bin" ]; then
        N_SKIP=$((N_SKIP + 1))
        continue
    fi
    $QNN/bin/target/qnn-context-binary-generator \
        --backend $QNN/lib/target/libQnnCpu.so \
        --model $QNN/lib/target/libQnnModelDlc.so \
        --dlc_path "$dlc" \
        --binary_file "${bin%.bin}" --output_dir . > /tmp/_ctxgen_tramp_${name}.log 2>&1
    rc=$?
    if [ "$rc" -eq 0 ] && [ -f "$bin" ]; then
        N_OK=$((N_OK + 1))
    else
        echo "FAIL: $name (rc=$rc)"
        tail -3 /tmp/_ctxgen_tramp_${name}.log
        N_FAIL=$((N_FAIL + 1))
    fi
done
echo "  Trampoline ctx binaries: OK=$N_OK  FAIL=$N_FAIL  SKIP=$N_SKIP"
EOF
fi
echo ""

# ---------------------------------------------------------------------------
# Step 3: Profile each trampoline phase
# ---------------------------------------------------------------------------
echo "--- Step 3: Profile each trampoline phase ---"
mkdir -p "$PROFILE_DIR"
TRAMP_JSON="$PROFILE_DIR/trampolines_perf.json"
echo "{" > "$TRAMP_JSON"
first_seg=true
TOTAL_OK=0; TOTAL_FAIL=0; TOTAL_SKIP=0

for i in $(seq 0 24); do
    seg=$(printf "dsp_seg_%02d" "$i")
    if [ "$first_seg" = "true" ]; then
        first_seg=false
    else
        echo "," >> "$TRAMP_JSON"
    fi
    echo "  \"$seg\": {" >> "$TRAMP_JSON"
    first_phase=true
    for p in 0 1 2; do
        phase_name="${seg}_tramp_p${p}"
        bin_name="ctx_${phase_name}__Cpu.bin"
        if [ "$first_phase" = "true" ]; then
            first_phase=false
        else
            echo "," >> "$TRAMP_JSON"
        fi
        line=$(ssh "$BOARD" bash <<EOF 2>/dev/null
export LD_LIBRARY_PATH=/root/qairt/lib/target
cd $REMOTE_CTX
if [ ! -f "$bin_name" ]; then
    echo '{"status":"no_ctx"}'
    exit 0
fi
$REMOTE_BASE/profile_seg "$bin_name" /root/qairt/lib/target/libQnnCpu.so $ITERS 2>/dev/null | grep '^{'
EOF
        )
        if [ -z "$line" ] || echo "$line" | grep -q '"status":"no_ctx"'; then
            printf '    "p%d": {"status":"no_ctx"}' "$p" >> "$TRAMP_JSON"
            TOTAL_SKIP=$((TOTAL_SKIP + 1))
        elif echo "$line" | grep -q '"status":"ok"'; then
            printf '    "p%d": %s' "$p" "$line" >> "$TRAMP_JSON"
            mean=$(echo "$line" | grep -o '"mean_us":[0-9.]*' | cut -d: -f2)
            echo "  $seg / p$p ... OK (mean=${mean} us)"
            TOTAL_OK=$((TOTAL_OK + 1))
        else
            printf '    "p%d": %s' "$p" "${line:-{\"status\":\"unknown_fail\"}}" >> "$TRAMP_JSON"
            echo "  $seg / p$p ... FAIL"
            TOTAL_FAIL=$((TOTAL_FAIL + 1))
        fi
    done
    echo "" >> "$TRAMP_JSON"
    echo -n "  }" >> "$TRAMP_JSON"
done
echo "" >> "$TRAMP_JSON"
echo "}" >> "$TRAMP_JSON"

echo ""
echo "=== Trampoline profiling complete ==="
echo "  OK=$TOTAL_OK  FAIL=$TOTAL_FAIL  SKIP=$TOTAL_SKIP"
echo "  Output: $TRAMP_JSON"
