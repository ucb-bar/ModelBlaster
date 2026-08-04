#!/usr/bin/env bash
#
# Convert each trampoline-phase ONNX into a DSP-targeted INT8-quantized DLC.
# Mirrors build_trampoline_dlcs.sh but:
#   - quantizes with --act_bitwidth 8 --weights_bitwidth 8 (DSP backend's
#     compose path requires int8 graphs)
#   - uses the per-phase calibration lists from gen_trampoline_calibration.py
#   - emits to a separate dlc_dsp/ dir so the CPU DLCs aren't overwritten
#
# Usage:
#   ./build_trampoline_dlcs_dsp.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="<PATH_TO_CONDA_ENV>/bin/python"
QNN_SDK="<PATH_TO_QAIRT_SDK>"
DOCKER_IMAGE="qnn-convert"

TRAMP_DIR="$SCRIPT_DIR/vision_slices_v3/trampolines"
DLC_DIR="$TRAMP_DIR/dlc_dsp"
CAL_DIR="$TRAMP_DIR/calibration"
mkdir -p "$DLC_DIR"

if [ ! -d "$CAL_DIR" ]; then
    echo "ERROR: calibration dir not found at $CAL_DIR — run gen_trampoline_calibration.py first"
    exit 1
fi

echo "Building DSP-targeted trampoline DLCs (int8 quantized)"
echo "Source ONNX: $TRAMP_DIR"
echo "Target DLC:  $DLC_DIR"
echo "Calibration: $CAL_DIR"
echo ""

n_total=0
n_done=0
n_skip=0
n_fail=0
for onnx in "$TRAMP_DIR"/dsp_seg_*_tramp_p*.onnx; do
    [ -f "$onnx" ] || continue
    n_total=$((n_total + 1))
    name="$(basename "$onnx" .onnx)"
    if [ -f "$DLC_DIR/${name}_q.dlc" ]; then
        echo "  skip $name (already int8)"
        n_skip=$((n_skip + 1))
        continue
    fi

    INPUT_FLAGS=$($PYTHON -c "
import onnx
model = onnx.load('$onnx')
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

    cal_list="$CAL_DIR/${name}_cal_list.txt"
    if [ ! -f "$cal_list" ]; then
        echo "  FAIL: missing $cal_list"
        n_fail=$((n_fail + 1))
        continue
    fi

    echo "  $name..."
    # Run conversion + quantization in one docker invocation. The .raw files
    # live under $CAL_DIR mounted at /cal so the cal list's absolute paths
    # resolve correctly inside the container; we rewrite them with a sed
    # so they reference /cal/<phase>/<file> instead of the host path.
    REWRITE_LIST="$CAL_DIR/${name}_cal_list_docker.txt"
    sed "s|$CAL_DIR|/cal|g" "$cal_list" > "$REWRITE_LIST"

    sudo docker run --rm \
        -v "$QNN_SDK":/qnn:ro \
        -v "$TRAMP_DIR":/workspace \
        -v "$CAL_DIR":/cal \
        "$DOCKER_IMAGE" bash -c "\
            pip install -q 'numpy<2' && \
            python3.10 /qnn/bin/x86_64-linux-clang/snpe-onnx-to-dlc \
                --input_network /workspace/${name}.onnx \
                ${INPUT_FLAGS} \
                --output_path /workspace/dlc_dsp/${name}.dlc && \
            python3.10 /qnn/bin/x86_64-linux-clang/qairt-quantizer \
                --input_dlc /workspace/dlc_dsp/${name}.dlc \
                --output_dlc /workspace/dlc_dsp/${name}_q.dlc \
                --input_list /cal/${name}_cal_list_docker.txt \
                --act_bitwidth 8 --weights_bitwidth 8 --bias_bitwidth 8" 2>&1 | tail -3

    if [ -f "$DLC_DIR/${name}_q.dlc" ]; then
        sudo chown "$(id -u):$(id -g)" "$DLC_DIR/${name}_q.dlc"
        [ -f "$DLC_DIR/${name}.dlc" ] && sudo chown "$(id -u):$(id -g)" "$DLC_DIR/${name}.dlc"
        n_done=$((n_done + 1))
    else
        n_fail=$((n_fail + 1))
        echo "    FAIL: $name produced no quantized DLC"
    fi
done

echo ""
echo "=== DSP trampoline DLC build summary ==="
echo "  total:     $n_total"
echo "  quantized: $n_done"
echo "  skipped:   $n_skip"
echo "  failed:    $n_fail"
echo "  Output:    $DLC_DIR"
ls -1 "$DLC_DIR"/*_q.dlc 2>/dev/null | wc -l | xargs -I{} echo "Quantized DLCs on disk: {}"
