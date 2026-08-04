#!/usr/bin/env bash
#
# Sweep each per-segment sub-DLC across every backend that can run it,
# capture wallclock-around-graphExecute (launch + RPC + sync + compute)
# per call, and emit a JSON manifest the scheduler-graph emitter
# consumes.
#
# Strategy:
#   1. For each <base>_quantized.dlc on the host, ensure a per-backend
#      context binary exists on the board: ctx_<base>__<backend>.bin
#      (built via qnn-context-binary-generator on the appropriate lib).
#   2. Run profile_segments on each (.bin, lib) pair for N iters,
#      collect the JSON-status line per pair.
#   3. Aggregate into a single manifest:
#         { "<segment>": { "<backend>": {<stats>}, ... }, ... }
#
# The output manifest goes at <gen_dir>/segment_perf.json and is
# consumed by emit_segmented_graph_json.py to build per-network
# graph.jsons with realistic perf data.

set -euo pipefail
GEN_DIR="${1:?usage: profile_sweep.sh <runtime_gen_dir>}"
ITERS="${ITERS:-50}"
BOARD_USER="${BOARD_USER:-root}"
BOARD_IP="${BOARD_IP:-<BOARD_IP>}"
BOARD_DIR="${BOARD_DIR:-/root/qnn_runtime}"
BOARD_CTX="${BOARD_CTX:-/root/qnn_runtime_ctx}"

# Backends to sweep per (network, backend_label). For dronet HTA_split
# (sub-DLCs were sliced from dronet_bnfree.onnx so all ops are HTA-
# friendly), profile on HTA + DSP + CPU. For dronet CPU (just an Add),
# every backend takes it. For yolov8n's partition slices (sliced from
# yolov8n_nosplit.onnx with head ops requiring DSP), profile on DSP +
# CPU only.
declare -A BACKENDS=(
    [dronet_HTA_split]="libQnnHta.so libQnnDsp.so libQnnCpu.so"
    [dronet_CPU]="libQnnCpu.so libQnnDsp.so libQnnHta.so"
    [yolov8n_HTA_split]="libQnnDsp.so libQnnCpu.so"
)

# 1) Push profile_segments source + build on board.
ssh "$BOARD_USER@$BOARD_IP" "mkdir -p $BOARD_DIR"
scp -q "$(dirname "$0")/profile_segments.cpp" \
    "$BOARD_USER@$BOARD_IP:$BOARD_DIR/"
ssh "$BOARD_USER@$BOARD_IP" bash <<EOF
set -e
cd $BOARD_DIR
QNN=/root/qairt
g++ -std=c++2a -O2 -Wall -Wno-unused-variable -pthread \
    -I"\$QNN/include" -I"\$QNN/include/QNN" \
    profile_segments.cpp -o profile_seg -ldl
echo "==> built profile_seg"
EOF

# 2) Push the sub-DLCs we want to profile and build per-backend
#    context binaries on board.
sudo chown -R "$(id -u):$(id -g)" "$GEN_DIR/sub_dlc/" 2>/dev/null || true

PERF_OUT="$GEN_DIR/segment_perf.json"
echo "{" > "$PERF_OUT"
first_seg=true

for dlc in "$GEN_DIR/sub_dlc/"*_quantized.dlc; do
    [ -f "$dlc" ] || continue
    base=$(basename "$dlc" _quantized.dlc)        # e.g. dronet_HTA_split_seg0
    # Pick the BACKENDS key by stripping the trailing _seg<N>.
    key=$(echo "$base" | sed 's/_seg[0-9]*$//')
    libs="${BACKENDS[$key]:-libQnnCpu.so}"

    echo "==> profiling $base across [$libs]"
    scp -q "$dlc" "$BOARD_USER@$BOARD_IP:$BOARD_CTX/sub_dlc/" 2>/dev/null || true

    if [ "$first_seg" = "true" ]; then
        first_seg=false
    else
        echo "," >> "$PERF_OUT"
    fi
    echo "  \"$base\": {" >> "$PERF_OUT"

    first_be=true
    for lib in $libs; do
        be_short=$(echo "$lib" | sed 's/libQnn//; s/\.so//')
        bin_name="ctx_${base}__${be_short}.bin"

        # Build the per-backend context binary if it doesn't exist on board.
        if ! ssh "$BOARD_USER@$BOARD_IP" "test -f $BOARD_CTX/$bin_name" 2>/dev/null; then
            ssh "$BOARD_USER@$BOARD_IP" bash <<EOF
set +e
cd $BOARD_CTX
QNN=/root/qairt
LD_LIBRARY_PATH=\$QNN/lib/target \
ADSP_LIBRARY_PATH="\$QNN/lib/hexagon-v66;/dsp/cdsp;/dsp" \
\$QNN/bin/target/qnn-context-binary-generator \
    --backend \$QNN/lib/target/$lib \
    --model \$QNN/lib/target/libQnnModelDlc.so \
    --dlc_path sub_dlc/${base}_quantized.dlc \
    --binary_file ${bin_name%.bin} --output_dir . > _gen_${base}__${be_short}.log 2>&1
EOF
        fi

        # Run the profiler. profile_seg prints one JSON line per run.
        line=$(ssh "$BOARD_USER@$BOARD_IP" \
            "export LD_LIBRARY_PATH=/root/qairt/lib/target ADSP_LIBRARY_PATH='/root/qairt/lib/hexagon-v66;/dsp/cdsp;/dsp' && \
             $BOARD_DIR/profile_seg $BOARD_CTX/$bin_name /root/qairt/lib/target/$lib $ITERS 2>&1 | grep '^{'" \
            || true)

        if [ "$first_be" = "true" ]; then
            first_be=false
        else
            echo "," >> "$PERF_OUT"
        fi
        if [ -z "$line" ]; then
            line='{"status":"compose_fail"}'
        fi
        printf '    "%s": %s' "$be_short" "$line" >> "$PERF_OUT"
        echo "  $base / $be_short: $(echo "$line" | head -c 120)"
    done
    echo "" >> "$PERF_OUT"
    echo -n "  }" >> "$PERF_OUT"
done

echo "" >> "$PERF_OUT"
echo "}" >> "$PERF_OUT"
echo
echo "==> wrote $PERF_OUT"
