#!/usr/bin/env bash
#
# End-to-end numeric-correctness check for the per-segment runtime.
# Steps:
#   1. Prepare an input blob in the layout/dtype the first segment's
#      sub-DLC expects (NHWC uint8 quant for dronet seg0). Calibration
#      data is fp32 NCHW, so we transpose + quantise here.
#   2. Run onnxruntime on the source ONNX with the original fp32 NCHW
#      input to get the golden output.
#   3. Push the prepared input blob to the board, run the runtime with
#      --input + --output-dir, scp outputs back.
#   4. Run validate_numeric.py to compute max_abs_err / cosine vs golden.
#
set -euo pipefail
GEN_DIR="${1:?usage: bash run_validation.sh <gen_dir>}"
NETWORK="${NETWORK:-dronet}"
SAMPLE="${SAMPLE:-qnn_models/boards/qrb5165_v66/calibration_data/calibration_data_dronet/input_0.raw}"
INPUT_NAME="${INPUT_NAME:-input}"
INPUT_SHAPE="${INPUT_SHAPE:-1,3,112,112}"
INPUT_QUANT_SCALE="${INPUT_QUANT_SCALE:-0.034882701933}"
INPUT_QUANT_OFFSET="${INPUT_QUANT_OFFSET:-129}"  # zero_point = -offset; sub-DLC stored offset=-129
ONNX="${ONNX:-qnn_models/dronet.onnx}"
GRAPH_JSON="${GRAPH_JSON:-qnn_models/boards/qrb5165_v66/graphs/dronet.int8.graph.json}"
BOARD_USER="${BOARD_USER:-root}"
BOARD_IP="${BOARD_IP:-<BOARD_IP>}"
BOARD_OUT="${BOARD_OUT:-/root/qnn_runtime_outputs}"

WORK="$GEN_DIR/validate"
mkdir -p "$WORK"

echo "==> 1. preparing QNN-layout quantized input"
PY=<PATH_TO_CONDA_ENV>/bin/python
"$PY" - "$SAMPLE" "$INPUT_SHAPE" "$INPUT_QUANT_SCALE" "$INPUT_QUANT_OFFSET" \
                  "$WORK/input_quant_nhwc.raw" <<'PY'
import sys, numpy as np
src, shape_csv, scale_s, off_s, dst = sys.argv[1:6]
shape = tuple(int(x) for x in shape_csv.split(","))
scale = float(scale_s); offset = int(off_s)
fp = np.fromfile(src, dtype=np.float32).reshape(shape)        # NCHW
nhwc = np.transpose(fp, (0, 2, 3, 1))                        # NCHW → NHWC
# QNN's "uFxp_8" stored as uint8 with offset already subtracted;
# at runtime QNN does (q - offset_signed) * scale where the .dlc
# has offset = -129 → qnn_q = round(fp/scale) + 129 (clamped).
q = np.clip(np.round(nhwc / scale) + offset, 0, 255).astype(np.uint8)
q.tofile(dst)
print(f"  wrote {dst}: {q.shape} {q.dtype} {q.nbytes} B  "
      f"q-range=[{q.min()}, {q.max()}]")
PY

echo "==> 2. running onnxruntime golden on $ONNX"
"$PY" - "$ONNX" "$INPUT_NAME" "$SAMPLE" "$INPUT_SHAPE" "$WORK/golden_outputs.npz" <<'PY'
import sys, numpy as np
import onnxruntime as ort
onnx_path, in_name, raw, shape_csv, dst = sys.argv[1:6]
shape = tuple(int(x) for x in shape_csv.split(","))
inp = np.fromfile(raw, dtype=np.float32).reshape(shape)
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
out_names = [o.name for o in sess.get_outputs()]
outs = sess.run(out_names, {in_name: inp})
np.savez(dst, **{n: a for n, a in zip(out_names, outs)})
print(f"  wrote {dst}: outputs={out_names}")
for n, a in zip(out_names, outs):
    print(f"    {n}: shape={a.shape} range=[{a.min():.4g}, {a.max():.4g}]")
PY

echo "==> 3. push input + run runtime on board"
scp -q "$WORK/input_quant_nhwc.raw" \
    "$BOARD_USER@$BOARD_IP:/root/qnn_runtime_input.raw"
ssh "$BOARD_USER@$BOARD_IP" "rm -rf $BOARD_OUT && mkdir -p $BOARD_OUT"

# build + run on board, with --input + --output-dir
ssh "$BOARD_USER@$BOARD_IP" mkdir -p /root/qnn_runtime
scp -q "$GEN_DIR"/runtime_main.cpp "$GEN_DIR"/dispatch_table.h \
    "$BOARD_USER@$BOARD_IP:/root/qnn_runtime/"
ssh "$BOARD_USER@$BOARD_IP" bash <<EOF
set -euo pipefail
cd /root/qnn_runtime
QNN_SDK_ROOT=/root/qairt
g++ -std=c++2a -O2 -Wall -Wno-unused-variable -pthread \
    -I"\$QNN_SDK_ROOT/include" -I"\$QNN_SDK_ROOT/include/QNN" \
    runtime_main.cpp -o qnn_runtime -ldl
LD_LIBRARY_PATH=\$QNN_SDK_ROOT/lib/target \
ADSP_LIBRARY_PATH="\$QNN_SDK_ROOT/lib/hexagon-v66;/dsp/cdsp;/dsp" \
QNN_RUNTIME_DUMP_ALL=1 \
./qnn_runtime --input "$INPUT_NAME=/root/qnn_runtime_input.raw" \
              --output-dir $BOARD_OUT 2>&1 | grep -E "^\[(main|run|summary|bringup)\]|wrote " | tail -30
EOF

echo "==> 4. pull outputs and compare"
mkdir -p "$WORK/runtime_outputs"
scp -q "$BOARD_USER@$BOARD_IP:$BOARD_OUT/*.raw" "$WORK/runtime_outputs/" 2>/dev/null || true
ls -lh "$WORK/runtime_outputs/" || true

"$PY" qnn_models/runtime/validate_numeric.py \
    --network "$NETWORK" \
    --onnx "$ONNX" \
    --input-name "$INPUT_NAME" \
    --input "$SAMPLE" \
    --input-shape "$INPUT_SHAPE" \
    --board-output-dir "$WORK/runtime_outputs" \
    --graph-json "$GRAPH_JSON"
