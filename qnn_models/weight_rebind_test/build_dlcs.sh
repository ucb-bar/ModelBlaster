#!/usr/bin/env bash
#
# Convert + quantize the weight-rebind test ONNX files, build ctx binaries
# for CPU / DSP / HTA. The variant-B (weight-as-graph-input) build is the
# crucial one: if HTA refuses to compile it, the rebind story doesn't
# apply on HTA regardless of runtime API behaviour.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="<PATH_TO_CONDA_ENV>/bin/python"
QNN_SDK="<PATH_TO_QAIRT_SDK>"
DOCKER_IMAGE="qnn-convert"
BOARD="${QNN_BOARD_HOST:-root@<BOARD_IP>}"
REMOTE_BASE="/root/weight_rebind_test"
DLC_DIR="$SCRIPT_DIR/dlcs"
CAL_DIR="$SCRIPT_DIR/calibration"
mkdir -p "$DLC_DIR"

build_variant() {
    local variant="$1"
    local cal_prefix="$2"   # e.g. "variant_a"
    local onnx="$SCRIPT_DIR/${variant}.onnx"
    local cal_list="$CAL_DIR/${cal_prefix}_cal_list.txt"
    local dlc_unq="$DLC_DIR/${variant}.dlc"
    local dlc_q="$DLC_DIR/${variant}_q.dlc"

    if [ ! -f "$cal_list" ]; then
        echo "ERROR: missing calibration list $cal_list"
        return 1
    fi
    # Rewrite cal list with /cal mount paths for the Docker container.
    local docker_cal_list="$CAL_DIR/${cal_prefix}_cal_list_docker.txt"
    sed "s|$CAL_DIR|/cal|g" "$cal_list" > "$docker_cal_list"

    # Determine input flags from the ONNX (variant-A has only x; variant-B has x and weight).
    local INPUT_FLAGS=$($PYTHON -c "
import onnx
m = onnx.load('$onnx')
flags = []
for inp in m.graph.input:
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    flags.append('-d {} {}'.format(inp.name, ','.join(str(d) for d in dims)))
    flags.append('--input_layout {} NCHW'.format(inp.name))
print(' '.join(flags))
")
    echo "  $variant: convert + quantize"
    sudo docker run --rm \
        -v "$QNN_SDK":/qnn:ro \
        -v "$SCRIPT_DIR":/workspace \
        -v "$CAL_DIR":/cal \
        "$DOCKER_IMAGE" bash -c "\
            pip install -q 'numpy<2' && \
            python3.10 /qnn/bin/x86_64-linux-clang/snpe-onnx-to-dlc \
                --input_network /workspace/${variant}.onnx \
                ${INPUT_FLAGS} \
                --output_path /workspace/dlcs/${variant}.dlc && \
            python3.10 /qnn/bin/x86_64-linux-clang/qairt-quantizer \
                --input_dlc /workspace/dlcs/${variant}.dlc \
                --output_dlc /workspace/dlcs/${variant}_q.dlc \
                --input_list /cal/${cal_prefix}_cal_list_docker.txt \
                --act_bitwidth 8 --weights_bitwidth 8 --bias_bitwidth 8" 2>&1 | tail -5
    if [ -f "$DLC_DIR/${variant}_q.dlc" ]; then
        sudo chown "$(id -u):$(id -g)" "$DLC_DIR/${variant}_q.dlc" "$DLC_DIR/${variant}.dlc"
        echo "  OK: $dlc_q ($(stat -c%s $dlc_q) B)"
    else
        echo "  FAIL: no quantized DLC produced"
    fi
}

echo "=== Building DLCs ==="
build_variant variant_a_const_weight variant_a
build_variant variant_b_input_weight variant_b
echo ""

# Push DLCs to board and try building ctx binaries for each backend.
echo "=== Pushing DLCs to board ==="
ssh "$BOARD" "mkdir -p $REMOTE_BASE"
scp -q "$DLC_DIR"/*_q.dlc "$BOARD:$REMOTE_BASE/"

echo ""
echo "=== Building ctx binaries on board (each variant × each backend) ==="
ssh "$BOARD" bash <<'EOF'
set +e
cd /root/weight_rebind_test
QNN=/root/qairt
export LD_LIBRARY_PATH=$QNN/lib/target
export ADSP_LIBRARY_PATH="$QNN/lib/hexagon-v66;/dsp/cdsp;/dsp"
for VARIANT in variant_a_const_weight_q variant_b_input_weight_q; do
    [ -f "$VARIANT.dlc" ] || { echo "  SKIP $VARIANT (no DLC)"; continue; }
    for BE in Cpu Dsp Hta; do
        BIN="${VARIANT}__${BE}.bin"
        [ -f "$BIN" ] && rm -f "$BIN"
        $QNN/bin/target/qnn-context-binary-generator \
            --backend $QNN/lib/target/libQnn${BE}.so \
            --model $QNN/lib/target/libQnnModelDlc.so \
            --dlc_path "$VARIANT.dlc" \
            --binary_file "${BIN%.bin}" --output_dir . > /tmp/_ctxgen_${VARIANT}_${BE}.log 2>&1
        rc=$?
        if [ "$rc" -eq 0 ] && [ -f "$BIN" ]; then
            echo "  $VARIANT × $BE : OK ($(stat -c%s $BIN) B)"
        else
            echo "  $VARIANT × $BE : FAIL (rc=$rc)"
            tail -3 /tmp/_ctxgen_${VARIANT}_${BE}.log | sed 's/^/    /'
        fi
    done
done
EOF
