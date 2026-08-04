#!/usr/bin/env bash
#
# SmolVLA Vision v3 — Correct Profiling via profile_segments.cpp
# ==============================================================
# Uses the same measurement method as the generated runtime: wallclock
# around QnnGraph_execute on pre-built context binaries.
#
# Flow:
#   1. Push quantized DLCs + CPU DLCs to board
#   2. Build context binaries per (segment, backend) on board
#   3. Build profile_seg on board (if not already built)
#   4. Run profile_seg per (context binary, backend lib)
#   5. Collect JSON results → segment_perf.json
#
# Prerequisites:
#   - Quantized DLCs at vision_slices_v3/dlc/ (from pipeline_vision_v3.sh build stage)
#   - Board reachable at $QNN_BOARD_HOST
#   - QAIRT 2.45 installed at /root/qairt on board
#
# Usage:
#   ./profile_vision_v3_correct.sh              # full sweep (physical board, default profile dir)
#   ./profile_vision_v3_correct.sh --iters 100  # more iterations
#   ./profile_vision_v3_correct.sh --skip-build # skip context binary gen
#
# Env overrides (for retargeting to a different board / output dir):
#   QNN_BOARD_HOST        SSH host (default: root@<BOARD_IP>)
#   PROFILE_DIR           output dir (default: <repo>/qnn_models/boards/qrb5165_v66/profiles/smolvlm_vision_v3)
#   REMOTE_BASE           on-board workspace (default: /root/models/smolvlm_vision_v3)
#   ADSP_EXTRA_PATHS      semi-colon-separated paths prepended to ADSP_LIBRARY_PATH
#                         (cloud QRB5165 needs "$QNN/lib/hexagon-v66/unsigned")
#
# Example — retarget to cloud QRB5165 with cloud-specific output dir:
#   QNN_BOARD_HOST=qrb_cloud \
#   PROFILE_DIR=qnn_models/boards/qrb5165_v66_cloud/profiles/smolvlm_vision_v3 \
#   REMOTE_BASE=/root/profile_v3 \
#   ADSP_EXTRA_PATHS=/root/qairt/lib/hexagon-v66/unsigned \
#       ./profile_vision_v3_correct.sh --iters 30

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNTIME_DIR="$REPO_ROOT/qnn_models/runtime"
DLC_DIR="$SCRIPT_DIR/vision_slices_v3/dlc"
PROFILE_DIR="${PROFILE_DIR:-$REPO_ROOT/qnn_models/boards/qrb5165_v66/profiles/smolvlm_vision_v3}"

BOARD="${QNN_BOARD_HOST:-root@<BOARD_IP>}"
REMOTE_BASE="${REMOTE_BASE:-/root/models/smolvlm_vision_v3}"
REMOTE_CTX="$REMOTE_BASE/ctx"
ITERS="${ITERS:-50}"
SKIP_BUILD=false

# Build the ADSP_LIBRARY_PATH used on the board for all DSP/HTA calls.
# Defaults match the physical board layout; cloud sets ADSP_EXTRA_PATHS.
# The value is fully expanded here (board SDK is at /root/qairt) so we
# can pass it through SSH heredocs without parent-shell escape games.
ADSP_PATH_BASE="/root/qairt/lib/hexagon-v66;/dsp/cdsp;/dsp"
if [ -n "${ADSP_EXTRA_PATHS:-}" ]; then
    ADSP_PATH_BASE="$ADSP_EXTRA_PATHS;$ADSP_PATH_BASE"
fi
export ADSP_PATH_BASE

while [ $# -gt 0 ]; do
    case "$1" in
        --iters) ITERS="$2"; shift 2;;
        --skip-build) SKIP_BUILD=true; shift;;
        *) echo "Unknown arg: $1" >&2; exit 1;;
    esac
done

echo "SmolVLA Vision v3 — Context Binary Profiling"
echo "  Board: $BOARD"
echo "  Iters: $ITERS"
echo "  DLC dir: $DLC_DIR"
echo ""

# ===========================================================================
# Step 1: Push profile_segments.cpp + build on board
# ===========================================================================
echo "--- Step 1: Build profile_seg on board ---"
ssh "$BOARD" "mkdir -p $REMOTE_BASE/{dlc,ctx}"
scp -q "$RUNTIME_DIR/profile_segments.cpp" "$BOARD:$REMOTE_BASE/"
ssh "$BOARD" bash <<EOF
set -e
cd $REMOTE_BASE
QNN=/root/qairt
if [ ! -f profile_seg ] || [ profile_segments.cpp -nt profile_seg ]; then
    g++ -std=c++2a -O2 -Wall -Wno-unused-variable -pthread \\
        -I"\$QNN/include" -I"\$QNN/include/QNN" \\
        profile_segments.cpp -o profile_seg -ldl
    echo "  Built profile_seg"
else
    echo "  profile_seg up to date"
fi
EOF

# ===========================================================================
# Step 2: Push DLCs to board
# ===========================================================================
echo ""
echo "--- Step 2: Push DLCs to board ---"
HTA_DLC_DIR="$SCRIPT_DIR/vision_slices_v3/hta_convs/dlc"
ssh "$BOARD" "mkdir -p $REMOTE_BASE/{dlc,hta_dlc,ctx}"
# DSP segments (quantized)
for dlc in "$DLC_DIR"/dsp_seg_*_quantized.dlc; do
    [ -f "$dlc" ] || continue
    scp -q "$dlc" "$BOARD:$REMOTE_BASE/dlc/"
done
# CPU segments (fp32)
for dlc in "$DLC_DIR"/cpu_seg_*.dlc; do
    [ -f "$dlc" ] || continue
    scp -q "$dlc" "$BOARD:$REMOTE_BASE/dlc/"
done
# HTA standalone conv DLCs (quantized)
if [ -d "$HTA_DLC_DIR" ]; then
    for dlc in "$HTA_DLC_DIR"/*_q.dlc; do
        [ -f "$dlc" ] || continue
        scp -q "$dlc" "$BOARD:$REMOTE_BASE/hta_dlc/"
    done
    echo "  HTA DLCs pushed ($(ls "$HTA_DLC_DIR"/*_q.dlc 2>/dev/null | wc -l) files)."
fi
echo "  DLCs pushed."

# ===========================================================================
# Step 3: Build context binaries on board
# ===========================================================================
if [ "$SKIP_BUILD" = "false" ]; then
    echo ""
    echo "--- Step 3: Build context binaries on board ---"

    # DSP segments: try DSP and CPU backends
    for i in $(seq 0 24); do
        seg=$(printf "dsp_seg_%02d" "$i")
        for lib in libQnnDsp.so libQnnCpu.so; do
            be_short=$(echo "$lib" | sed 's/libQnn//; s/\.so//')
            bin_name="ctx_${seg}__${be_short}.bin"

            ssh "$BOARD" bash <<EOF
set +e
cd $REMOTE_CTX
QNN=/root/qairt
[ -f "$bin_name" ] && { echo "  skip $seg/$be_short (exists)"; exit 0; }
export LD_LIBRARY_PATH=\$QNN/lib/target
export ADSP_LIBRARY_PATH="$ADSP_PATH_BASE"
SRC=$REMOTE_BASE/dlc/${seg}_quantized.dlc
[ -f "\$SRC" ] || { echo "  SKIP $seg/$be_short (no DLC)"; exit 0; }
echo -n "  $seg/$be_short ... "
\$QNN/bin/target/qnn-context-binary-generator \
    --backend \$QNN/lib/target/$lib \
    --model \$QNN/lib/target/libQnnModelDlc.so \
    --dlc_path "\$SRC" \
    --binary_file ${bin_name%.bin} --output_dir . > /tmp/_ctxgen_${seg}_${be_short}.log 2>&1
rc=\$?
if [ "\$rc" -eq 0 ] && [ -f "$bin_name" ]; then
    echo "OK ($(stat -c%s $bin_name) B)"
else
    echo "FAIL (rc=\$rc)"
fi
EOF
        done
    done

    # CPU segments: CPU backend only
    for i in $(seq 0 23); do
        seg=$(printf "cpu_seg_%02d" "$i")
        lib="libQnnCpu.so"
        be_short="Cpu"
        bin_name="ctx_${seg}__${be_short}.bin"

        ssh "$BOARD" bash <<EOF
set +e
cd $REMOTE_CTX
QNN=/root/qairt
[ -f "$bin_name" ] && { echo "  skip $seg/$be_short (exists)"; exit 0; }
export LD_LIBRARY_PATH=\$QNN/lib/target
SRC=$REMOTE_BASE/dlc/${seg}.dlc
[ -f "\$SRC" ] || { echo "  SKIP $seg/$be_short (no DLC)"; exit 0; }
echo -n "  $seg/$be_short ... "
\$QNN/bin/target/qnn-context-binary-generator \
    --backend \$QNN/lib/target/$lib \
    --model \$QNN/lib/target/libQnnModelDlc.so \
    --dlc_path "\$SRC" \
    --binary_file ${bin_name%.bin} --output_dir . > /tmp/_ctxgen_${seg}_${be_short}.log 2>&1
rc=\$?
if [ "\$rc" -eq 0 ] && [ -f "$bin_name" ]; then
    echo "OK ($(stat -c%s $bin_name) B)"
else
    echo "FAIL (rc=\$rc)"
fi
EOF
    done

    # HTA standalone conv1x1 DLCs: try HTA backend
    echo ""
    echo "  --- HTA context binaries (standalone conv ops) ---"
    HTA_OK=0; HTA_FAIL=0; HTA_SKIP=0
    for hta_dlc in "$HTA_DLC_DIR"/*_q.dlc; do
        [ -f "$hta_dlc" ] || continue
        conv_name="$(basename "$hta_dlc" _q.dlc)"
        bin_name="ctx_${conv_name}__Hta.bin"

        result=$(ssh "$BOARD" bash <<EOF 2>&1
set +e
cd $REMOTE_CTX
QNN=/root/qairt
[ -f "$bin_name" ] && { echo "EXISTS"; exit 0; }
export LD_LIBRARY_PATH=\$QNN/lib/target
export ADSP_LIBRARY_PATH="$ADSP_PATH_BASE"
SRC=$REMOTE_BASE/hta_dlc/$(basename "$hta_dlc")
[ -f "\$SRC" ] || { echo "NO_DLC"; exit 0; }
\$QNN/bin/target/qnn-context-binary-generator \
    --backend \$QNN/lib/target/libQnnHta.so \
    --model \$QNN/lib/target/libQnnModelDlc.so \
    --dlc_path "\$SRC" \
    --binary_file ${bin_name%.bin} --output_dir . > /tmp/_ctxgen_hta_${conv_name}.log 2>&1
rc=\$?
if [ "\$rc" -eq 0 ] && [ -f "$bin_name" ]; then
    echo "OK"
else
    echo "FAIL"
fi
EOF
        )
        case "$result" in
            *EXISTS*) HTA_SKIP=$((HTA_SKIP+1)) ;;
            *OK*) echo "  $conv_name/Hta: OK"; HTA_OK=$((HTA_OK+1)) ;;
            *NO_DLC*) HTA_SKIP=$((HTA_SKIP+1)) ;;
            *) echo "  $conv_name/Hta: FAIL (HTA validation)"; HTA_FAIL=$((HTA_FAIL+1)) ;;
        esac
    done
    echo "  HTA ctx binaries: OK=$HTA_OK  FAIL=$HTA_FAIL  SKIP=$HTA_SKIP"
fi

# ===========================================================================
# Step 4: Run profile_seg on each context binary
# ===========================================================================
echo ""
echo "--- Step 4: Profile each segment ---"
mkdir -p "$PROFILE_DIR"
PERF_JSON="$PROFILE_DIR/segment_perf.json"
echo "{" > "$PERF_JSON"
first_seg=true
OK=0; FAIL=0; SKIP=0

profile_one() {
    local seg="$1" be_short="$2" lib="$3"
    local bin_name="ctx_${seg}__${be_short}.bin"
    local out_key="${seg}__${be_short}"

    line=$(ssh "$BOARD" bash <<EOF 2>/dev/null
export LD_LIBRARY_PATH=/root/qairt/lib/target
export ADSP_LIBRARY_PATH="$ADSP_PATH_BASE"
cd $REMOTE_CTX
[ -f "$bin_name" ] || { echo '{"status":"no_ctx"}'; exit 0; }
$REMOTE_BASE/profile_seg "$bin_name" /root/qairt/lib/target/$lib $ITERS 2>/dev/null | grep '^{'
EOF
    )
    echo "$line"
}

for i in $(seq 0 24); do
    seg=$(printf "dsp_seg_%02d" "$i")

    if [ "$first_seg" = "true" ]; then
        first_seg=false
    else
        echo "," >> "$PERF_JSON"
    fi
    echo "  \"$seg\": {" >> "$PERF_JSON"
    first_be=true

    for lib in libQnnDsp.so libQnnCpu.so; do
        be_short=$(echo "$lib" | sed 's/libQnn//; s/\.so//')
        echo -n "  $seg / $be_short ... "

        line=$(profile_one "$seg" "$be_short" "$lib")

        if [ "$first_be" = "true" ]; then
            first_be=false
        else
            echo "," >> "$PERF_JSON"
        fi

        if [ -z "$line" ] || echo "$line" | grep -q '"status":"no_ctx"'; then
            printf '    "%s": {"status":"no_ctx"}' "$be_short" >> "$PERF_JSON"
            echo "SKIP (no ctx binary)"
            SKIP=$((SKIP+1))
        elif echo "$line" | grep -q '"status":"ok"'; then
            printf '    "%s": %s' "$be_short" "$line" >> "$PERF_JSON"
            mean=$(echo "$line" | grep -o '"mean_us":[0-9.]*' | cut -d: -f2)
            echo "OK (mean=${mean} us)"
            OK=$((OK+1))
        else
            printf '    "%s": %s' "$be_short" "${line:-{\"status\":\"unknown_fail\"}}" >> "$PERF_JSON"
            echo "FAIL"
            FAIL=$((FAIL+1))
        fi
    done

    # HTA: profile each standalone conv1x1 in this segment, sum their times
    # to estimate total HTA time for the segment (the non-conv ops would still
    # need CPU, but this gives the scheduler the HTA compute cost).
    hta_total_us=0
    hta_count=0
    hta_all_ok=true
    for hta_dlc in "$HTA_DLC_DIR"/${seg}_*_q.dlc; do
        [ -f "$hta_dlc" ] || continue
        conv_name="$(basename "$hta_dlc" _q.dlc)"
        hta_bin="ctx_${conv_name}__Hta.bin"
        hta_line=$(ssh "$BOARD" bash <<EOF 2>/dev/null
export LD_LIBRARY_PATH=/root/qairt/lib/target
export ADSP_LIBRARY_PATH="$ADSP_PATH_BASE"
cd $REMOTE_CTX
[ -f "$hta_bin" ] || { echo '{"status":"no_ctx"}'; exit 0; }
$REMOTE_BASE/profile_seg "$hta_bin" /root/qairt/lib/target/libQnnHta.so $ITERS 2>/dev/null | grep '^{'
EOF
        )
        if echo "$hta_line" | grep -q '"status":"ok"'; then
            conv_us=$(echo "$hta_line" | grep -o '"mean_us":[0-9.]*' | cut -d: -f2)
            hta_total_us=$(echo "$hta_total_us + $conv_us" | bc)
            hta_count=$((hta_count+1))
        else
            hta_all_ok=false
        fi
    done

    if [ "$hta_count" -gt 0 ]; then
        echo "," >> "$PERF_JSON"
        if [ "$hta_all_ok" = "true" ]; then
            printf '    "Hta": {"status":"ok","mean_us":%.2f,"note":"sum of %d conv ops"}' \
                "$hta_total_us" "$hta_count" >> "$PERF_JSON"
            echo "  $seg / Hta ... OK (sum=${hta_total_us} us, ${hta_count} convs)"
            OK=$((OK+1))
        else
            printf '    "Hta": {"status":"partial","mean_us":%.2f,"convs_ok":%d}' \
                "$hta_total_us" "$hta_count" >> "$PERF_JSON"
            echo "  $seg / Hta ... PARTIAL (${hta_count} convs OK, some failed)"
            OK=$((OK+1))
        fi
    fi

    echo "" >> "$PERF_JSON"
    echo -n "  }" >> "$PERF_JSON"
done

# CPU segments
for i in $(seq 0 23); do
    seg=$(printf "cpu_seg_%02d" "$i")
    lib="libQnnCpu.so"
    be_short="Cpu"

    echo "," >> "$PERF_JSON"
    echo "  \"$seg\": {" >> "$PERF_JSON"
    echo -n "  $seg / $be_short ... "

    line=$(profile_one "$seg" "$be_short" "$lib")

    if [ -z "$line" ] || echo "$line" | grep -q '"status":"no_ctx"'; then
        printf '    "%s": {"status":"no_ctx"}' "$be_short" >> "$PERF_JSON"
        echo "SKIP (no ctx binary)"
        SKIP=$((SKIP+1))
    elif echo "$line" | grep -q '"status":"ok"'; then
        printf '    "%s": %s' "$be_short" "$line" >> "$PERF_JSON"
        mean=$(echo "$line" | grep -o '"mean_us":[0-9.]*' | cut -d: -f2)
        echo "OK (mean=${mean} us)"
        OK=$((OK+1))
    else
        printf '    "%s": %s' "$be_short" "${line:-{\"status\":\"unknown_fail\"}}" >> "$PERF_JSON"
        echo "FAIL"
        FAIL=$((FAIL+1))
    fi
    echo "" >> "$PERF_JSON"
    echo -n "  }" >> "$PERF_JSON"
done

echo "" >> "$PERF_JSON"
echo "}" >> "$PERF_JSON"

echo ""
echo "=== Profiling complete ==="
echo "  OK=$OK  FAIL=$FAIL  SKIP=$SKIP"
echo "  Output: $PERF_JSON"
echo ""
echo "Next: run emit_vision_v3_profile.py --from-perf-json to convert to XPURT format"
