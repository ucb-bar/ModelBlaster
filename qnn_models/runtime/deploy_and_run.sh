#!/usr/bin/env bash
#
# Push a generated QNN runtime to a board, compile it, run it, capture
# the trace. Works for both single-graph and multi-graph context layouts
# and for both the physical QRB5165 and the cloud QRB5165 (which needs
# /unsigned in ADSP_LIBRARY_PATH).
#
# Replaces three ad-hoc patterns I'd been hand-rolling each time:
#   - build_and_run.sh        (single-graph, hardcoded paths)
#   - stage_multi_graph_pipeline.sh (multi-graph, physical only)
#   - inline ssh+scp+g++ for cloud runs
#
# Usage:
#   bash deploy_and_run.sh <gen_dir> [board] [board-dir]
#
# Example (physical, single-graph):
#   bash deploy_and_run.sh qnn_models/runtime/gen/qrb5165_smolvla_v3_bundles
#
# Example (cloud, multi-graph, custom budget):
#   QNN_BOARD_HOST=qrb_cloud \
#   XPURT_DSP_CTX_BUDGET=2  XPURT_HTA_CTX_BUDGET=2 \
#   bash deploy_and_run.sh qnn_models/runtime/gen/qrb5165_smolvla_v3_bundles_mg
#
# Environment knobs:
#   QNN_BOARD_HOST     ssh host (default root@<BOARD_IP>)
#   BOARD_DIR          remote runtime dir (default /root/qnn_runtime)
#   CTX_DIR            remote ctx dir for symlinks (default /root/qnn_runtime_ctx)
#   QNN_SDK_ROOT       remote SDK root (default /root/qairt)
#   ADSP_EXTRA_PATHS   prepend to ADSP_LIBRARY_PATH (cloud: lib/hexagon-v66/unsigned)
#   XPURT_DSP_CTX_BUDGET    lazy DSP context budget (multi-graph runtimes)
#   XPURT_HTA_CTX_BUDGET    lazy HTA context budget
#   LOG_DIR            where to save the captured trace (default runs/<gen_dir basename>)

set -uo pipefail

GEN_DIR="${1:-}"
BOARD="${2:-${QNN_BOARD_HOST:-root@<BOARD_IP>}}"
BOARD_DIR_ARG="${3:-${BOARD_DIR:-/root/qnn_runtime}}"

if [ -z "$GEN_DIR" ] || [ ! -d "$GEN_DIR" ]; then
    echo "usage: $0 <generated_runtime_dir> [board] [board-dir]" >&2
    echo "  <generated_runtime_dir> must contain runtime_main.cpp + dispatch_table.h" >&2
    exit 1
fi
if [ ! -f "$GEN_DIR/runtime_main.cpp" ] || [ ! -f "$GEN_DIR/dispatch_table.h" ]; then
    echo "error: $GEN_DIR missing runtime_main.cpp or dispatch_table.h" >&2
    exit 1
fi

QNN_SDK_ROOT="${QNN_SDK_ROOT:-/root/qairt}"
CTX_DIR="${CTX_DIR:-/root/qnn_runtime_ctx}"
ADSP_EXTRA_PATHS="${ADSP_EXTRA_PATHS:-}"
LOG_DIR="${LOG_DIR:-runs/$(basename "$GEN_DIR")}"
mkdir -p "$LOG_DIR"

# Build the runtime env for the board-side run.
ADSP_BASE="$QNN_SDK_ROOT/lib/hexagon-v66;/dsp/cdsp;/dsp"
if [ -n "$ADSP_EXTRA_PATHS" ]; then
    ADSP_FULL="$ADSP_EXTRA_PATHS;$ADSP_BASE"
else
    # Auto-detect /unsigned for boards that need it (cloud QRB5165).
    ADSP_FULL="$QNN_SDK_ROOT/lib/hexagon-v66/unsigned;$ADSP_BASE"
fi

env_lines=""
[ -n "${XPURT_DSP_CTX_BUDGET:-}" ] && env_lines+="XPURT_DSP_CTX_BUDGET=$XPURT_DSP_CTX_BUDGET "
[ -n "${XPURT_HTA_CTX_BUDGET:-}" ] && env_lines+="XPURT_HTA_CTX_BUDGET=$XPURT_HTA_CTX_BUDGET "

echo "==> board     : $BOARD"
echo "==> gen dir   : $GEN_DIR"
echo "==> board dir : $BOARD_DIR_ARG"
echo "==> ctx dir   : $CTX_DIR"
echo "==> log dir   : $LOG_DIR"
[ -n "$env_lines" ] && echo "==> runtime env: $env_lines"
echo ""

# Multi-graph context binaries crash the cDSP user-PD on QRB5165 v66
# (see qnn_models/QRB5165_MULTIGRAPH_CDSP_CRASH_FORENSICS.md). The host
# runtime can't detect this — it reports QNN_SUCCESS on dead PDs.
# If the dispatch table references multi-graph bundles, warn loudly.
if grep -qE 'ctx_(sched|.*_chunk[0-9]+)_' "$GEN_DIR/dispatch_table.h" 2>/dev/null; then
    cat <<'WARN' >&2
==============================================================================
WARNING: dispatch_table.h references multi-graph context binaries
         (ctx_*_chunk*.bin). These trigger a silent cDSP user-PD crash
         on QRB5165 v66; the runtime will still report 97/97 success but
         the DSP is dead. See QRB5165_MULTIGRAPH_CDSP_CRASH_FORENSICS.md.

         If this is intentional (e.g. you're testing a firmware fix),
         verify outputs with diff_boundary_tensors.py after the run.
==============================================================================
WARN
fi

# --- 1) push sources + any locally-staged ctx bins ---
ssh "$BOARD" "mkdir -p $BOARD_DIR_ARG $CTX_DIR" || { echo "ssh failed — board reachable?"; exit 1; }
scp -q "$GEN_DIR/runtime_main.cpp" "$GEN_DIR/dispatch_table.h" "$BOARD:$BOARD_DIR_ARG/"
if [ -d "$GEN_DIR/ctx" ]; then
    n_ctx=$(ls "$GEN_DIR/ctx"/*.bin 2>/dev/null | wc -l)
    if [ "$n_ctx" -gt 0 ]; then
        echo "==> pushing $n_ctx locally-staged ctx bins..."
        scp -q "$GEN_DIR/ctx"/*.bin "$BOARD:$CTX_DIR/" || true
    fi
fi

# --- 2) compile on board ---
ssh "$BOARD" bash <<EOF
set -euo pipefail
cd "$BOARD_DIR_ARG"
g++ -std=c++2a -O2 -Wall -Wno-unused-variable -pthread \\
    -I"$QNN_SDK_ROOT/include" -I"$QNN_SDK_ROOT/include/QNN" \\
    runtime_main.cpp -o qnn_runtime -ldl
echo "==> built \$(stat -c%s qnn_runtime) B"
EOF

# --- 3) run + capture full log ---
RUN_LOG="$LOG_DIR/run.log"
echo "==> running..."
ssh "$BOARD" bash > "$RUN_LOG" 2>&1 <<EOF
set -uo pipefail
cd "$BOARD_DIR_ARG"
LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/target \\
ADSP_LIBRARY_PATH="$ADSP_FULL" \\
$env_lines \\
./qnn_runtime
EOF
RC=$?

echo "==> exit code: $RC"
echo "==> log: $RUN_LOG ($(wc -l < "$RUN_LOG") lines)"
echo ""
echo "--- summary ---"
grep -E "prefetched|loaded eagerly|wall=|MAKESPAN|summary|FAIL" "$RUN_LOG" | head -8 \
    || echo "(no summary lines — run may have crashed; check $RUN_LOG)"
echo ""
if [ $RC -ne 0 ]; then
    echo "  ! non-zero exit; consider running:"
    echo "    bash qnn_models/runtime/collect_board_diagnostics.sh $BOARD"
fi
exit $RC
