#!/usr/bin/env bash
#
# End-to-end multi-graph pipeline for the bundle-aware v3 vision schedule.
# Runs the steps in order; safe to interrupt and resume since each step
# is idempotent.
#
#   1. (one-time) build_multi_graph_ctx.py for each backend bundle
#   2. build_v3_bundles.py to refresh the placement plan + results.csv
#   3. run_xpurt_schedule.py
#   4. generate_runtime.py with --graph-index for each manifest
#   5. stage_v3_bundles_multigraph.py to symlink multi-graph .bins on board
#   6. push runtime + compile + run

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="<PATH_TO_CONDA_ENV>/bin/python"
BOARD="${QNN_BOARD_HOST:-root@<BOARD_IP>}"
BUDGET="${BUDGET:-}"   # empty = unconstrained (all 23 segs DSP-tramp)
OUT_DIR="$REPO_ROOT/qnn_models/runtime/gen/qrb5165_smolvla_v3_bundles_mg"
LOG_DIR="$REPO_ROOT/runs/v3_bundles_multigraph"
mkdir -p "$OUT_DIR" "$LOG_DIR"

V3=$REPO_ROOT/qnn_models/smolVLA/vision_slices_v3
TRAMPS=$V3/trampolines/multi_ctx
HTACONV=$V3/hta_convs/multi_ctx
DSPSEGS=$V3/multi_ctx

# Step 1: ensure multi-graph ctx binaries exist (skipped if already built)
echo "=== [1] build multi-graph ctx binaries ==="
for spec in \
    "v3_dsp_segs Dsp $V3/dlc dsp_seg_*_quantized.dlc" \
    "v3_dsp_segs Cpu $V3/dlc dsp_seg_*_quantized.dlc" \
    "v3_cpu_segs Cpu $V3/dlc cpu_seg_*.dlc" \
    "v3_hta_convs Hta $V3/hta_convs/dlc *_q.dlc" \
    "v3_tramps Dsp $V3/trampolines/dlc_dsp *_q.dlc" \
    "v3_tramps Cpu $V3/trampolines/dlc *.dlc"; do
    set -- $spec
    bundle=$1; be=$2; dir=$3; pat=$4
    $PYTHON "$REPO_ROOT/qnn_models/smolVLA/build_multi_graph_ctx.py" \
        --dlc-dir "$dir" --dlc-pattern "$pat" --backend "$be" \
        --chunk 10 --bundle-name "$bundle" --board "$BOARD" \
        2>&1 | tail -2
done

# Step 2 + 3: plan + schedule
echo ""
echo "=== [2,3] plan + schedule (budget=${BUDGET:-unconstrained}) ==="
if [ -n "$BUDGET" ]; then
    $PYTHON "$REPO_ROOT/qnn_models/smolVLA/build_v3_bundles.py" --dsp-tramp-budget "$BUDGET" 2>&1 | grep -E "Variant counts|Total dispatches"
else
    $PYTHON "$REPO_ROOT/qnn_models/smolVLA/build_v3_bundles.py" 2>&1 | grep -E "Variant counts|Total dispatches"
fi
$PYTHON "$REPO_ROOT/scripts/run_xpurt_schedule.py" \
    --networks-json "$REPO_ROOT/data/toplevel/networks_smolvla_vision_v3_bundles_qrb5165.json" \
    --solver greedy --profiled 2>&1 | grep -E "^  Makespan|Combination assignments" | head -2

# Step 4: generate runtime with graph-index entries
echo ""
echo "=== [4] generate runtime ==="
$PYTHON "$REPO_ROOT/qnn_models/runtime/generate_runtime.py" \
    --schedule "$REPO_ROOT/schedules/scheduled_networks_smolvla_vision_v3_bundles_qrb5165_greedy_profiled.json" \
    --out-dir "$OUT_DIR" \
    --backend-map CPU_P=HTA:libQnnHta.so,CPU_E=DSP:libQnnDsp.so,CPU_X=CPU:libQnnCpu.so \
    --from-segmented-schedule \
    --ctx-dir /root/qnn_runtime_ctx_v3_mg \
    --graph-index "$DSPSEGS/v3_dsp_segs_Dsp_graph_index.json" \
    --graph-index "$DSPSEGS/v3_dsp_segs_Cpu_graph_index.json" \
    --graph-index "$DSPSEGS/v3_cpu_segs_Cpu_graph_index.json" \
    --graph-index "$HTACONV/v3_hta_convs_Hta_graph_index.json" \
    --graph-index "$TRAMPS/v3_tramps_Dsp_graph_index.json" \
    --graph-index "$TRAMPS/v3_tramps_Cpu_graph_index.json" \
    --seg-perf "$REPO_ROOT/qnn_models/boards/qrb5165_v66/profiles/smolvlm_vision_v3/segment_perf.json" \
    2>&1 | tail -3

# Step 5: stage multi-graph ctx symlinks on board
echo ""
echo "=== [5] stage multi-graph symlinks on board ==="
$PYTHON "$REPO_ROOT/qnn_models/smolVLA/stage_v3_bundles_multigraph.py" \
    --ctx-src /root/multi_ctx --ctx-dst /root/qnn_runtime_ctx_v3_mg \
    --schedule "$REPO_ROOT/schedules/scheduled_networks_smolvla_vision_v3_bundles_qrb5165_greedy_profiled.json" \
    --graph-index "$DSPSEGS/v3_dsp_segs_Dsp_graph_index.json" \
    --graph-index "$DSPSEGS/v3_dsp_segs_Cpu_graph_index.json" \
    --graph-index "$DSPSEGS/v3_cpu_segs_Cpu_graph_index.json" \
    --graph-index "$HTACONV/v3_hta_convs_Hta_graph_index.json" \
    --graph-index "$TRAMPS/v3_tramps_Dsp_graph_index.json" \
    --graph-index "$TRAMPS/v3_tramps_Cpu_graph_index.json" \
    --seg-perf "$REPO_ROOT/qnn_models/boards/qrb5165_v66/profiles/smolvlm_vision_v3/segment_perf.json" \
    --out-script "$REPO_ROOT/qnn_models/smolVLA/stage_v3_bundles_multigraph.sh" \
    2>&1 | tail -5
ssh "$BOARD" "rm -f /root/qnn_runtime_ctx_v3_mg/*.bin"
scp -q "$REPO_ROOT/qnn_models/smolVLA/stage_v3_bundles_multigraph.sh" "$BOARD:/tmp/"
ssh "$BOARD" "bash /tmp/stage_v3_bundles_multigraph.sh 2>&1 | tail -1"

# Step 6: push runtime + build + run
echo ""
echo "=== [6] push + build + run ==="
ssh "$BOARD" "mkdir -p /root/qnn_runtime_v3_bundles_mg"
scp -q "$OUT_DIR/runtime_main.cpp" "$OUT_DIR/dispatch_table.h" \
    "$BOARD:/root/qnn_runtime_v3_bundles_mg/"
ssh "$BOARD" bash > "$LOG_DIR/run.log" 2>&1 <<'BEOF'
set -euo pipefail
cd /root/qnn_runtime_v3_bundles_mg
QNN_SDK_ROOT=/root/qairt
g++ -std=c++2a -O2 -Wall -Wno-unused-variable -pthread \
    -I$QNN_SDK_ROOT/include -I$QNN_SDK_ROOT/include/QNN \
    runtime_main.cpp -o qnn_runtime -ldl
echo "==> built $(stat -c%s qnn_runtime) B"
LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/target \
ADSP_LIBRARY_PATH="$QNN_SDK_ROOT/lib/hexagon-v66;/dsp/cdsp;/dsp" \
./qnn_runtime
BEOF
echo ""
echo "=== Run summary ==="
grep -E "summary|loaded eagerly|prefetched|reset.*backend" "$LOG_DIR/run.log" | tail -5
echo ""
echo "Full log: $LOG_DIR/run.log"
