#!/usr/bin/env bash
#
# Fast feasibility check: build DLCs for ONE representative decode segment
# (dsp_seg_02 — a 34-op block with 7 Linear MatMuls rewritten to Conv1x1)
# on CPU and DSP, profile both on board, report wallclock ratio.
#
# If DSP is meaningfully faster than CPU (say ≥1.5×), it's worth proceeding
# with the full 65-segment pipeline. Otherwise, decode-on-DSP is the wrong
# move and we should look elsewhere.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="<PATH_TO_CONDA_ENV>/bin/python"
QNN_SDK="<PATH_TO_QAIRT_SDK>"
DOCKER_IMAGE="qnn-convert"
BOARD="${QNN_BOARD_HOST:-root@<BOARD_IP>}"
REMOTE_BASE="/root/decode_probe"

SEG="dsp_seg_02"
SRC_ONNX="$SCRIPT_DIR/decode_slices_v1/conv1x1/${SEG}.onnx"
WORK_DIR="$SCRIPT_DIR/decode_slices_v1/probe"
DLC_DIR="$WORK_DIR/dlc"
CAL_DIR="$WORK_DIR/cal"
mkdir -p "$DLC_DIR" "$CAL_DIR"

echo "Probing decode/$SEG: CPU vs DSP profile (single segment)"
echo "Source ONNX: $SRC_ONNX"
echo "---"

# Generate random fp32 calibration data for each input tensor
echo "Generating calibration data..."
$PYTHON << PY
import onnx, numpy as np
from pathlib import Path
m = onnx.load("$SRC_ONNX")
rng = np.random.default_rng(42)
cal_dir = Path("$CAL_DIR")
with open(cal_dir / "${SEG}_cal_list.txt", "w") as f:
    for sample_i in range(8):
        tokens = []
        for inp in m.graph.input:
            dims = [d.dim_value if d.dim_value > 0 else 1 for d in inp.type.tensor_type.shape.dim]
            elem_type = inp.type.tensor_type.elem_type
            if elem_type == onnx.TensorProto.FLOAT:
                data = rng.standard_normal(size=dims).astype(np.float32) * 0.3
            elif elem_type in (onnx.TensorProto.INT32, onnx.TensorProto.INT64):
                np_dt = np.int32 if elem_type == onnx.TensorProto.INT32 else np.int64
                data = rng.integers(0, 50, size=dims).astype(np_dt)
            else:
                data = rng.standard_normal(size=dims).astype(np.float32) * 0.3
            safe_name = inp.name.replace("/", "_").replace(".", "_").replace(":", "_")
            raw = cal_dir / f"sample{sample_i:02d}_{safe_name}.raw"
            data.tofile(str(raw))
            tokens.append(f"{inp.name}:={raw.absolute()}")
        f.write(" ".join(tokens) + "\n")
print(f"  cal_list: {cal_dir / f'${SEG}_cal_list.txt'} (8 samples)")
PY

# Build input flag string
INPUT_FLAGS=$($PYTHON -c "
import onnx
model = onnx.load('$SRC_ONNX')
flags = []
for inp in model.graph.input:
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    flags.append('-d {} {}'.format(inp.name, ','.join(str(d) for d in dims)))
    if len(dims) == 4:
        flags.append('--input_layout {} NCHW'.format(inp.name))
    else:
        flags.append('--input_layout {} NONTRIVIAL'.format(inp.name))
print(' '.join(flags))
")

# Rewrite cal list paths for Docker mount
sed "s|$CAL_DIR|/cal|g" "$CAL_DIR/${SEG}_cal_list.txt" > "$CAL_DIR/${SEG}_cal_list_docker.txt"

echo "Building DLCs (convert + quantize)..."
sudo docker run --rm \
    -v "$QNN_SDK":/qnn:ro \
    -v "$WORK_DIR":/workspace \
    -v "$CAL_DIR":/cal \
    "$DOCKER_IMAGE" bash -c "\
        pip install -q 'numpy<2' && \
        cp '$SRC_ONNX' /workspace/${SEG}.onnx 2>/dev/null || true; \
        python3.10 /qnn/bin/x86_64-linux-clang/snpe-onnx-to-dlc \
            --input_network /workspace/${SEG}.onnx \
            ${INPUT_FLAGS} \
            --output_path /workspace/dlc/${SEG}.dlc && \
        python3.10 /qnn/bin/x86_64-linux-clang/qairt-quantizer \
            --input_dlc /workspace/dlc/${SEG}.dlc \
            --output_dlc /workspace/dlc/${SEG}_q.dlc \
            --input_list /cal/${SEG}_cal_list_docker.txt \
            --act_bitwidth 8 --weights_bitwidth 8 --bias_bitwidth 8" 2>&1 | tail -5

# The conv1x1 dir doesn't have ${SEG}.onnx accessible inside Docker easily.
# Copy it into the workspace first:
cp -f "$SRC_ONNX" "$WORK_DIR/${SEG}.onnx" 2>/dev/null

# Retry conversion if the first attempt failed (we did it inside the container
# but the bind mount may not have propagated). Re-run from outside instead.
if [ ! -f "$DLC_DIR/${SEG}_q.dlc" ]; then
    sudo docker run --rm \
        -v "$QNN_SDK":/qnn:ro \
        -v "$WORK_DIR":/workspace \
        -v "$CAL_DIR":/cal \
        "$DOCKER_IMAGE" bash -c "\
            pip install -q 'numpy<2' && \
            python3.10 /qnn/bin/x86_64-linux-clang/snpe-onnx-to-dlc \
                --input_network /workspace/${SEG}.onnx \
                ${INPUT_FLAGS} \
                --output_path /workspace/dlc/${SEG}.dlc && \
            python3.10 /qnn/bin/x86_64-linux-clang/qairt-quantizer \
                --input_dlc /workspace/dlc/${SEG}.dlc \
                --output_dlc /workspace/dlc/${SEG}_q.dlc \
                --input_list /cal/${SEG}_cal_list_docker.txt \
                --act_bitwidth 8 --weights_bitwidth 8 --bias_bitwidth 8" 2>&1 | tail -5
fi
sudo chown -R "$(id -u):$(id -g)" "$DLC_DIR"

if [ ! -f "$DLC_DIR/${SEG}_q.dlc" ]; then
    echo "ERROR: DLC build failed"
    exit 1
fi
echo "OK: $DLC_DIR/${SEG}_q.dlc ($(stat -c%s $DLC_DIR/${SEG}_q.dlc) B)"

echo ""
echo "Pushing to board + building ctx binaries..."
ssh "$BOARD" "mkdir -p $REMOTE_BASE"
scp -q "$DLC_DIR/${SEG}_q.dlc" "$BOARD:$REMOTE_BASE/"

ssh "$BOARD" bash <<EOF
set +e
cd $REMOTE_BASE
QNN=/root/qairt
export LD_LIBRARY_PATH=\$QNN/lib/target
export ADSP_LIBRARY_PATH="\$QNN/lib/hexagon-v66;/dsp/cdsp;/dsp"
for BE in Cpu Dsp; do
    BIN=ctx_${SEG}__\${BE}.bin
    rm -f "\$BIN"
    \$QNN/bin/target/qnn-context-binary-generator \
        --backend \$QNN/lib/target/libQnn\${BE}.so \
        --model \$QNN/lib/target/libQnnModelDlc.so \
        --dlc_path "${SEG}_q.dlc" \
        --binary_file "ctx_${SEG}__\${BE}" --output_dir . > /tmp/_ctxgen_decode_\${BE}.log 2>&1
    if [ \$? -eq 0 ] && [ -f "\$BIN" ]; then
        echo "  ctx_${SEG}__\${BE}.bin: OK (\$(stat -c%s \$BIN) B)"
    else
        echo "  ctx_${SEG}__\${BE}.bin: FAIL"
        tail -3 /tmp/_ctxgen_decode_\${BE}.log
    fi
done

echo ""
echo "Profiling 50 iters per backend..."
for BE in Cpu Dsp; do
    BIN=ctx_${SEG}__\${BE}.bin
    if [ -f "\$BIN" ]; then
        echo "--- \${BE} ---"
        /root/models/smolvlm_vision_v3/profile_seg "\$BIN" \$QNN/lib/target/libQnn\${BE}.so 50 2>/dev/null | grep '^{'
    fi
done
EOF
