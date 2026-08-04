#!/usr/bin/env bash
#
# SmolVLA Vision Encoder v3 Pipeline
# ====================================
# Reproducible end-to-end flow: slice → convert → quantize → profile → emit
#
# Stages:
#   1. slice      — Run slice_vision_v3.py to produce 49 ONNX segments
#   2. rewrite    — Run rewrite_matmul_to_conv1x1.py on DSP segments (for DSP Conv path)
#   3. build      — Convert ONNX → DLC (fp32 + int8 quantized) for CPU/DSP
#   4. build-hta  — Convert standalone Conv1x1 models → DLC for HTA
#   5. profile    — Build context binaries + run profile_seg (wallclock graphExecute)
#   6. emit       — Parse segment_perf.json into gen/profile/ results.csv for XPURT
#   7. schedule   — Run XPURT greedy scheduler and produce Gantt plot
#
# Prerequisites:
#   - smolvlm_vision.onnx in this directory
#   - Docker image "qnn-convert" with QNN SDK tools
#   - Board reachable at $BOARD (default: root@<BOARD_IP>)
#   - xpurt conda env with onnx, onnxruntime, numpy
#
# Usage:
#   ./pipeline_vision_v3.sh                 # full pipeline
#   ./pipeline_vision_v3.sh slice           # only slice
#   ./pipeline_vision_v3.sh build           # only build DLCs
#   ./pipeline_vision_v3.sh profile         # only profile on board
#   ./pipeline_vision_v3.sh emit            # only emit XPURT profiles
#   ./pipeline_vision_v3.sh schedule        # only run scheduler
#   ./pipeline_vision_v3.sh profile emit schedule  # skip build, run tail
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="<PATH_TO_CONDA_ENV>/bin/python"
QNN_SDK="<PATH_TO_QAIRT_SDK>"
DOCKER_IMAGE="qnn-convert"
BOARD="${QNN_BOARD_HOST:-root@<BOARD_IP>}"
NUM_CALIB=10
N_PROFILE_ITERS=10

# Directories
VISION_ONNX="$SCRIPT_DIR/smolvlm_vision.onnx"
SLICES_DIR="$SCRIPT_DIR/vision_slices_v3"
CONV1X1_DIR="$SLICES_DIR/conv1x1"
HTA_CONVS_DIR="$SLICES_DIR/hta_convs"
DLC_DIR="$SLICES_DIR/dlc"
CALIB_DIR="$SLICES_DIR/calibration"
PROFILE_DIR="$REPO_ROOT/qnn_models/boards/qrb5165_v66/profiles/smolvlm_vision_v3"

# Remote paths on the board
REMOTE_BASE="/root/models/smolvlm_vision_v3"

# Parse stages to run (default: all)
STAGES=()
if [ $# -eq 0 ]; then
    STAGES=(slice rewrite build build-hta profile emit schedule)
else
    STAGES=("$@")
fi

should_run() {
    for s in "${STAGES[@]}"; do
        [ "$s" = "$1" ] && return 0
    done
    return 1
}

docker_run() {
    sudo docker run --rm \
        -v "$QNN_SDK":/qnn:ro \
        -v "$SLICES_DIR":/workspace \
        "$DOCKER_IMAGE" "$@"
}

# ===========================================================================
# Stage 1: SLICE — Cut smolvlm_vision.onnx into 49 segments
# ===========================================================================
if should_run "slice"; then
    echo "============================================================"
    echo "Stage 1: SLICE (smolvlm_vision.onnx → 49 segments)"
    echo "============================================================"
    if [ ! -f "$VISION_ONNX" ]; then
        echo "ERROR: $VISION_ONNX not found"
        exit 1
    fi
    $PYTHON "$SCRIPT_DIR/slice_vision_v3.py"
    echo ""
fi

# ===========================================================================
# Stage 2: REWRITE — MatMul → Conv1x1 on DSP segments
# ===========================================================================
if should_run "rewrite"; then
    echo "============================================================"
    echo "Stage 2: REWRITE (MatMul → Conv1x1 for DSP segments)"
    echo "============================================================"
    $PYTHON "$SCRIPT_DIR/rewrite_matmul_to_conv1x1.py" "$SLICES_DIR" --batch --validate
    echo ""
    # Also extract standalone HTA convs
    echo "--- Extracting standalone HTA Conv1x1 ops ---"
    $PYTHON "$SCRIPT_DIR/extract_hta_convs.py" \
        --slices-dir "$CONV1X1_DIR" \
        --out-dir "$HTA_CONVS_DIR"
    echo ""
fi

# ===========================================================================
# Stage 3: BUILD — Convert DSP/CPU segments to DLC + quantize
# ===========================================================================
if should_run "build"; then
    echo "============================================================"
    echo "Stage 3: BUILD (ONNX → DLC, quantize int8)"
    echo "============================================================"
    mkdir -p "$DLC_DIR" "$CALIB_DIR"

    # 3a. Generate calibration data
    echo "--- Generating calibration data ($NUM_CALIB samples) ---"
    if [ ! -f "$CALIB_DIR/dsp_seg_00_cal_list.txt" ]; then
        $PYTHON "$SCRIPT_DIR/gen_vision_slice_calibration.py" \
            --src "$VISION_ONNX" \
            --slices-dir "$SLICES_DIR" \
            --out-dir "$CALIB_DIR" \
            --num-samples $NUM_CALIB
    else
        echo "  (calibration data already exists, skipping)"
    fi
    echo ""

    # 3b. Convert and quantize DSP segments (use conv1x1 variants)
    echo "--- Converting DSP segments (conv1x1 variant) ---"
    for seg_onnx in "$CONV1X1_DIR"/dsp_seg_*.onnx; do
        seg_name="$(basename "$seg_onnx" .onnx)"
        [ -f "$DLC_DIR/${seg_name}_quantized.dlc" ] && { echo "  skip $seg_name (exists)"; continue; }

        echo "  $seg_name..."
        INPUT_FLAGS=$($PYTHON -c "
import onnx
model = onnx.load('$seg_onnx')
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

        # Mount conv1x1 dir as workspace for this conversion
        sudo docker run --rm \
            -v "$QNN_SDK":/qnn:ro \
            -v "$CONV1X1_DIR":/workspace \
            -v "$CALIB_DIR":/calib \
            "$DOCKER_IMAGE" bash -c "\
                pip install -q 'numpy<2' && \
                python3.10 /qnn/bin/x86_64-linux-clang/snpe-onnx-to-dlc \
                    --input_network /workspace/${seg_name}.onnx \
                    ${INPUT_FLAGS} \
                    --output_path /workspace/dlc/${seg_name}.dlc" 2>&1 | tail -3

        if [ -f "$CONV1X1_DIR/dlc/${seg_name}.dlc" ]; then
            sudo chown "$(id -u):$(id -g)" "$CONV1X1_DIR/dlc/${seg_name}.dlc"
            # Copy to main DLC dir
            cp "$CONV1X1_DIR/dlc/${seg_name}.dlc" "$DLC_DIR/${seg_name}.dlc"

            # Quantize
            CAL_LIST="$CALIB_DIR/${seg_name}_cal_list.txt"
            if [ -f "$CAL_LIST" ]; then
                sudo docker run --rm \
                    -v "$QNN_SDK":/qnn:ro \
                    -v "$DLC_DIR":/dlcs \
                    -v "$CALIB_DIR":/calib \
                    "$DOCKER_IMAGE" python3.10 /qnn/bin/x86_64-linux-clang/qairt-quantizer \
                        --input_dlc "/dlcs/${seg_name}.dlc" \
                        --output_dlc "/dlcs/${seg_name}_quantized.dlc" \
                        --input_list "/calib/${seg_name}_cal_list.txt" \
                        --act_bitwidth 8 \
                        --weights_bitwidth 8 \
                        --bias_bitwidth 8 2>&1 | tail -3
                [ -f "$DLC_DIR/${seg_name}_quantized.dlc" ] && \
                    sudo chown "$(id -u):$(id -g)" "$DLC_DIR/${seg_name}_quantized.dlc"
            fi
        fi
    done
    echo ""

    # 3c. Convert CPU segments (fp32, no quantization needed)
    echo "--- Converting CPU segments (fp32) ---"
    for seg_onnx in "$SLICES_DIR"/cpu_seg_*.onnx; do
        seg_name="$(basename "$seg_onnx" .onnx)"
        [ -f "$DLC_DIR/${seg_name}.dlc" ] && { echo "  skip $seg_name (exists)"; continue; }

        INPUT_FLAGS=$($PYTHON -c "
import onnx
model = onnx.load('$seg_onnx')
flags = []
for inp in model.graph.input:
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    flags.append('-d {} {}'.format(inp.name, ','.join(str(d) for d in dims)))
    flags.append('--input_layout {} NONTRIVIAL'.format(inp.name))
print(' '.join(flags))
")

        echo "  $seg_name..."
        docker_run bash -c "\
            pip install -q 'numpy<2' && \
            python3.10 /qnn/bin/x86_64-linux-clang/snpe-onnx-to-dlc \
                --input_network /workspace/${seg_name}.onnx \
                ${INPUT_FLAGS} \
                --output_path /workspace/dlc/${seg_name}.dlc" 2>&1 | tail -3
        [ -f "$DLC_DIR/${seg_name}.dlc" ] && \
            sudo chown "$(id -u):$(id -g)" "$DLC_DIR/${seg_name}.dlc"
    done
    echo ""
fi

# ===========================================================================
# Stage 4: BUILD-HTA — Convert standalone Conv1x1 models for HTA
# ===========================================================================
if should_run "build-hta"; then
    echo "============================================================"
    echo "Stage 4: BUILD-HTA (standalone Conv ops → quantized DLC)"
    echo "============================================================"
    mkdir -p "$HTA_CONVS_DIR/dlc"

    # Match all extracted HTA conv models (both conv1x1 and patch embed)
    for conv_onnx in "$HTA_CONVS_DIR"/dsp_seg_*.onnx; do
        [ -f "$conv_onnx" ] || continue
        conv_name="$(basename "$conv_onnx" .onnx)"
        [ -f "$HTA_CONVS_DIR/dlc/${conv_name}_q.dlc" ] && { echo "  skip $conv_name (exists)"; continue; }

        INPUT_FLAGS=$($PYTHON -c "
import onnx
model = onnx.load('$conv_onnx')
flags = []
for inp in model.graph.input:
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    flags.append('-d {} {}'.format(inp.name, ','.join(str(d) for d in dims)))
    flags.append('--input_layout {} NCHW'.format(inp.name))
print(' '.join(flags))
")

        echo "  $conv_name..."
        sudo docker run --rm \
            -v "$QNN_SDK":/qnn:ro \
            -v "$HTA_CONVS_DIR":/workspace \
            "$DOCKER_IMAGE" bash -c "\
                pip install -q 'numpy<2' && \
                python3.10 /qnn/bin/x86_64-linux-clang/snpe-onnx-to-dlc \
                    --input_network /workspace/${conv_name}.onnx \
                    ${INPUT_FLAGS} \
                    --output_path /workspace/dlc/${conv_name}.dlc && \
                python3.10 /qnn/bin/x86_64-linux-clang/qairt-quantizer \
                    --input_dlc /workspace/dlc/${conv_name}.dlc \
                    --output_dlc /workspace/dlc/${conv_name}_q.dlc \
                    --act_bitwidth 8 --weights_bitwidth 8 --bias_bitwidth 8" 2>&1 | tail -5
        [ -f "$HTA_CONVS_DIR/dlc/${conv_name}_q.dlc" ] && \
            sudo chown "$(id -u):$(id -g)" "$HTA_CONVS_DIR/dlc/${conv_name}_q.dlc"
        [ -f "$HTA_CONVS_DIR/dlc/${conv_name}.dlc" ] && \
            sudo chown "$(id -u):$(id -g)" "$HTA_CONVS_DIR/dlc/${conv_name}.dlc"
    done
    echo ""
fi

# ===========================================================================
# Stage 5: PROFILE — Build context binaries + profile via graphExecute wallclock
# ===========================================================================
if should_run "profile"; then
    echo "============================================================"
    echo "Stage 5: PROFILE (context binaries + profile_seg wallclock)"
    echo "============================================================"
    # Delegate to the correct profiling script that uses profile_segments.cpp
    # (wallclock around QnnGraph_execute, matching the generated runtime's measurement)
    bash "$SCRIPT_DIR/profile_vision_v3_correct.sh" --iters "$N_PROFILE_ITERS"
    echo ""
fi

# ===========================================================================
# Stage 6: EMIT — Parse segment_perf.json → XPURT gen/profile/ results.csv
# ===========================================================================
if should_run "emit"; then
    echo "============================================================"
    echo "Stage 6: EMIT (segment_perf.json → XPURT results.csv)"
    echo "============================================================"
    PERF_JSON="$PROFILE_DIR/segment_perf.json"
    if [ -f "$PERF_JSON" ]; then
        $PYTHON "$SCRIPT_DIR/emit_vision_v3_profile.py" --target qrb5165_v66 \
            --from-perf-json "$PERF_JSON"
    else
        echo "  WARNING: $PERF_JSON not found, falling back to legacy CSV parsing"
        $PYTHON "$SCRIPT_DIR/emit_vision_v3_profile.py" --target qrb5165_v66 \
            --profile-dir "$PROFILE_DIR"
    fi
    echo ""
fi

# ===========================================================================
# Stage 7: SCHEDULE — Run XPURT and plot
# ===========================================================================
if should_run "schedule"; then
    echo "============================================================"
    echo "Stage 7: SCHEDULE (XPURT greedy + plot)"
    echo "============================================================"
    # Ensure dispatch graph exists
    $PYTHON "$SCRIPT_DIR/gen_vision_v3_dispatch_graph.py"
    echo ""
    # Run scheduler
    $PYTHON "$REPO_ROOT/scripts/run_xpurt_schedule.py" \
        --networks-json "$REPO_ROOT/data/toplevel/networks_smolvla_vision_v3_qrb5165.json" \
        --solver greedy \
        --profiled
    echo ""
    echo "=== Pipeline complete ==="
    echo "  Plot: plots/networks_smolvla_vision_v3_qrb5165_greedy_profiled.png"
    echo "  Schedule: schedules/scheduled_networks_smolvla_vision_v3_qrb5165_greedy_profiled.json"
fi
