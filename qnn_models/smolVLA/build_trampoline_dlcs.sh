#!/usr/bin/env bash
#
# Convert each trampoline-phase ONNX (from extract_trampoline_phases.py) into
# a CPU-targeted fp32 DLC via the qnn-convert Docker pipeline. These DLCs run
# the non-Conv ops (LayerNorm, Add, Mul, Pow, Transpose, Reshape, Split,
# MatMul) on QNN's CPU backend, bridging the HTA conv calls when the scheduler
# routes a DSP segment to HTA.
#
# Usage:
#   ./build_trampoline_dlcs.sh
#
# Reads:  vision_slices_v3/trampolines/dsp_seg_XX_tramp_pY.onnx
# Writes: vision_slices_v3/trampolines/dlc/dsp_seg_XX_tramp_pY.dlc

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="<PATH_TO_CONDA_ENV>/bin/python"
QNN_SDK="<PATH_TO_QAIRT_SDK>"
DOCKER_IMAGE="qnn-convert"

TRAMP_DIR="$SCRIPT_DIR/vision_slices_v3/trampolines"
DLC_DIR="$TRAMP_DIR/dlc"
mkdir -p "$DLC_DIR"

if [ ! -d "$TRAMP_DIR" ]; then
    echo "ERROR: $TRAMP_DIR not found — run extract_trampoline_phases.py first"
    exit 1
fi

echo "Building trampoline-phase DLCs (CPU, fp32)..."
echo "Source: $TRAMP_DIR"
echo "Target: $DLC_DIR"
echo ""

n_total=0
n_done=0
n_skip=0
n_fail=0
for onnx in "$TRAMP_DIR"/dsp_seg_*_tramp_p*.onnx; do
    [ -f "$onnx" ] || continue
    n_total=$((n_total + 1))
    name="$(basename "$onnx" .onnx)"
    if [ -f "$DLC_DIR/${name}.dlc" ]; then
        echo "  skip $name (exists)"
        n_skip=$((n_skip + 1))
        continue
    fi

    # Build per-input flags from the ONNX shape info.
    INPUT_FLAGS=$($PYTHON -c "
import onnx
model = onnx.load('$onnx')
flags = []
for inp in model.graph.input:
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    flags.append('-d {} {}'.format(inp.name, ','.join(str(d) for d in dims)))
    # 4D NCHW for conv outputs (which become trampoline inputs); other
    # ranks pass through as NONTRIVIAL.
    if len(dims) == 4:
        flags.append('--input_layout {} NCHW'.format(inp.name))
    else:
        flags.append('--input_layout {} NONTRIVIAL'.format(inp.name))
print(' '.join(flags))
")

    echo "  $name..."
    sudo docker run --rm \
        -v "$QNN_SDK":/qnn:ro \
        -v "$TRAMP_DIR":/workspace \
        "$DOCKER_IMAGE" bash -c "\
            pip install -q 'numpy<2' && \
            python3.10 /qnn/bin/x86_64-linux-clang/snpe-onnx-to-dlc \
                --input_network /workspace/${name}.onnx \
                ${INPUT_FLAGS} \
                --output_path /workspace/dlc/${name}.dlc" 2>&1 | tail -3

    if [ -f "$DLC_DIR/${name}.dlc" ]; then
        sudo chown "$(id -u):$(id -g)" "$DLC_DIR/${name}.dlc"
        n_done=$((n_done + 1))
    else
        n_fail=$((n_fail + 1))
        echo "    FAIL: $name produced no DLC"
    fi
done

echo ""
echo "=== Trampoline DLC build summary ==="
echo "  total:    $n_total"
echo "  built:    $n_done"
echo "  skipped:  $n_skip (already existed)"
echo "  failed:   $n_fail"
echo ""
echo "Output: $DLC_DIR"
ls -1 "$DLC_DIR" | wc -l | xargs -I{} echo "DLCs on disk: {}"
