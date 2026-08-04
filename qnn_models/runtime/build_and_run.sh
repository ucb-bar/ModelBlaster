#!/usr/bin/env bash
#
# Push the generated runtime + every needed context binary to the QRB5165,
# build on-device, and run.
#
#   bash build_and_run.sh <out_dir>
# Example:
#   bash build_and_run.sh gen/qrb5165_dronet_yolov8
#
set -euo pipefail
OUT_DIR="${1:-gen/qrb5165_dronet_yolov8}"
BOARD_USER="${BOARD_USER:-root}"
BOARD_IP="${BOARD_IP:-<BOARD_IP>}"
BOARD_DIR="${BOARD_DIR:-/root/qnn_runtime}"
CTX_DIR="${CTX_DIR:-/root/qnn_runtime_ctx}"

if [ ! -d "$OUT_DIR" ]; then
    echo "error: $OUT_DIR not generated yet — run generate_runtime.py first" >&2
    exit 1
fi

# Push the generated sources.
ssh "$BOARD_USER@$BOARD_IP" "mkdir -p $BOARD_DIR $CTX_DIR"
scp -q "$OUT_DIR"/runtime_main.cpp "$OUT_DIR"/dispatch_table.h \
    "$BOARD_USER@$BOARD_IP:$BOARD_DIR/"

# Push every context binary the runtime expects. The names are
#   ctx_<network>_<label>.bin
# matching what generate_runtime.py emits in the runtime's ctx lookup
# path. We staged these locally as part of the runtime prep step.
if [ -d "$OUT_DIR/ctx" ]; then
    scp -q "$OUT_DIR"/ctx/*.bin "$BOARD_USER@$BOARD_IP:$CTX_DIR/" || true
fi

# Build + run on the board.
ssh "$BOARD_USER@$BOARD_IP" bash <<EOF
set -euo pipefail
cd "$BOARD_DIR"
QNN_SDK_ROOT=/root/qairt
g++ -std=c++2a -O2 -Wall -Wno-unused-variable -pthread \\
    -I"\$QNN_SDK_ROOT/include" -I"\$QNN_SDK_ROOT/include/QNN" \\
    runtime_main.cpp -o qnn_runtime -ldl
echo "==> built \$(stat -c%s qnn_runtime) bytes"
echo "==> ctx dir: \$(ls -lh $CTX_DIR | tail -n +2)"
LD_LIBRARY_PATH=\$QNN_SDK_ROOT/lib/target \\
ADSP_LIBRARY_PATH="\$QNN_SDK_ROOT/lib/hexagon-v66;/dsp/cdsp;/dsp" \\
./qnn_runtime
EOF
