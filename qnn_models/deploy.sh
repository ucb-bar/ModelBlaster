#!/usr/bin/env bash
#
# Full pipeline: export ONNX → convert to QNN DLC → deploy to QRB5165 → benchmark.
#
# Usage:
#   ./deploy.sh                              # all 3 baseline models
#   ./deploy.sh --model=dronet               # single baseline model
#   ./deploy.sh --model=smolvla              # all 9 SmolVLA submodels
#   ./deploy.sh --model=smolvlm_expert_decode  # single SmolVLA submodel
#   ./deploy.sh --input=model.onnx           # arbitrary ONNX file
#   ./deploy.sh --input=export_custom.py     # custom export script
#   ./deploy.sh --skip-export                # skip ONNX export (use existing .onnx)
#   ./deploy.sh --export-only                # just export ONNX(es)
#   ./deploy.sh --convert-only               # just convert ONNX → DLC
#   ./deploy.sh --run-only                   # just benchmark (models must be on board)
#   ./deploy.sh --reconvert                  # skip onnxsim, convert directly from ONNX
#   ./deploy.sh --iters=100                  # set iteration count
#   ./deploy.sh --profile-csv=out.csv        # export benchmark results to CSV
#
# Prerequisites:
#   - conda env with pytorch, torchvision, ultralytics, onnx2tf, matplotlib
#   - Docker with sudo access (for QNN conversion)
#   - SSH key access to board (root@<BOARD_IP>)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLES_DIR="$SCRIPT_DIR/samples"
SMOLVLA_DIR="$SCRIPT_DIR/smolVLA"
CONDA_ENV="${CONDA_ENV:-merlin-dev}"
BOARD_USER="${BOARD_USER:-root}"
BOARD_IP="${BOARD_IP:-<BOARD_IP>}"
QNN_SDK="<PATH_TO_QAIRT_SDK>"
DOCKER_IMAGE="${DOCKER_IMAGE:-qnn-convert}"
RESULTS_FILE="$SCRIPT_DIR/benchmark_results.json"

# --- Parse args ---
EXPORT=true
CONVERT=true
DEPLOY=true
RUN=true
RECONVERT=false
ITERS=50
SELECTED_MODEL=""
INPUT_FILE=""
PROFILE_CSV=""
WORK_DIR="$SCRIPT_DIR"

for arg in "$@"; do
    case $arg in
        --export-only)  CONVERT=false; DEPLOY=false; RUN=false ;;
        --convert-only) EXPORT=false; DEPLOY=false; RUN=false ;;
        --run-only)     EXPORT=false; CONVERT=false; DEPLOY=false ;;
        --skip-export)  EXPORT=false ;;
        --reconvert)    RECONVERT=true ;;
        --model=*)      SELECTED_MODEL="${arg#*=}" ;;
        --input=*)      INPUT_FILE="${arg#*=}" ;;
        --profile-csv=*) PROFILE_CSV="${arg#*=}" ;;
        --iters=*)      ITERS="${arg#*=}" ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# ===========================================================================
# SmolVLA model definitions
# ===========================================================================
SMOLVLA_MODELS=(
    action_in_projector
    action_out_projector
    state_projector
    time_in_projector
    time_out_projector
    smolvlm_text
    smolvlm_vision
    smolvlm_expert_decode
    smolvlm_expert_prefill
)

is_smolvla_model() {
    local m="$1"
    for sm in "${SMOLVLA_MODELS[@]}"; do
        [[ "$sm" == "$m" ]] && return 0
    done
    return 1
}

# ===========================================================================
# Baseline model definitions (export scripts live in samples/)
# ===========================================================================
declare -A MODEL_INPUT_NAME MODEL_HW MODEL_EXPORT MODEL_CONVERTER MODEL_ONNX_INPUTS MODEL_INPUT_RANK
MODEL_INPUT_NAME[dronet]="input"
MODEL_HW[dronet]="112 112 3"
MODEL_EXPORT[dronet]="export_onnx.py"
MODEL_CONVERTER[dronet]="onnx"
MODEL_INPUT_RANK[dronet]=4

MODEL_INPUT_NAME[mobilenet_v2]="input"
MODEL_HW[mobilenet_v2]="224 224 3"
MODEL_EXPORT[mobilenet_v2]="export_mobilenet.py"
MODEL_CONVERTER[mobilenet_v2]="onnx"
MODEL_INPUT_RANK[mobilenet_v2]=4

MODEL_INPUT_NAME[yolov8s]="images"
MODEL_HW[yolov8s]="640 640 3"
MODEL_EXPORT[yolov8s]="export_yolo.py"
MODEL_CONVERTER[yolov8s]="tflite"
MODEL_INPUT_RANK[yolov8s]=4

# ===========================================================================
# Model selection: --model=smolvla, --model=<name>, --input=<file>, or default
# ===========================================================================
if [ "$SELECTED_MODEL" = "smolvla" ]; then
    ALL_MODELS=("${SMOLVLA_MODELS[@]}")
    EXPORT=false
elif [ -n "$INPUT_FILE" ]; then
    if [ ! -f "$INPUT_FILE" ]; then
        echo "ERROR: Input file not found: $INPUT_FILE"
        exit 1
    fi
    INPUT_FILE="$(cd "$(dirname "$INPUT_FILE")" && pwd)/$(basename "$INPUT_FILE")"
    WORK_DIR="$(dirname "$INPUT_FILE")"
    EXT="${INPUT_FILE##*.}"
    case "$EXT" in
        onnx)
            DERIVED_MODEL="$(basename "$INPUT_FILE" .onnx)"
            EXPORT=false
            ;;
        py)
            DERIVED_MODEL="$(basename "$INPUT_FILE" .py)"
            ;;
        *)
            echo "ERROR: --input file must be .py or .onnx (got .$EXT)"
            exit 1
            ;;
    esac
    if [ -z "$SELECTED_MODEL" ]; then
        SELECTED_MODEL="$DERIVED_MODEL"
    fi
    if [ -z "${MODEL_EXPORT[$SELECTED_MODEL]:-}" ]; then
        MODEL_CONVERTER[$SELECTED_MODEL]="onnx"
        if [ "$EXT" = "py" ]; then
            MODEL_INPUT_NAME[$SELECTED_MODEL]="input"
            MODEL_HW[$SELECTED_MODEL]="224 224 3"
            MODEL_EXPORT[$SELECTED_MODEL]="$(basename "$INPUT_FILE")"
        else
            ONNX_META=$(python3 -c "
import onnx, json
model = onnx.load('$INPUT_FILE')
inputs = []
for inp in model.graph.input:
    name = inp.name
    dims = [d.dim_value if d.dim_value > 0 else 1 for d in inp.type.tensor_type.shape.dim]
    inputs.append((name, dims))
print(inputs[0][0])
print(len(inputs[0][1]))
print(json.dumps(inputs))
" 2>&1)
            ONNX_INPUT_NAME=$(echo "$ONNX_META" | sed -n '1p')
            ONNX_INPUT_RANK=$(echo "$ONNX_META" | sed -n '2p')
            ONNX_INPUTS_JSON=$(echo "$ONNX_META" | sed -n '3p')
            MODEL_INPUT_NAME[$SELECTED_MODEL]="$ONNX_INPUT_NAME"
            MODEL_INPUT_RANK[$SELECTED_MODEL]="$ONNX_INPUT_RANK"
            MODEL_HW[$SELECTED_MODEL]="_auto_"
            MODEL_EXPORT[$SELECTED_MODEL]="_onnx_provided_"
            MODEL_ONNX_INPUTS[$SELECTED_MODEL]="$ONNX_INPUTS_JSON"
        fi
    fi
    ALL_MODELS=("$SELECTED_MODEL")
elif [ -n "$SELECTED_MODEL" ]; then
    ALL_MODELS=("$SELECTED_MODEL")
    if is_smolvla_model "$SELECTED_MODEL"; then
        EXPORT=false
    fi
else
    ALL_MODELS=(dronet mobilenet_v2 yolov8s)
fi

# Auto-register SmolVLA models from their ONNX files
for m in "${ALL_MODELS[@]}"; do
    if is_smolvla_model "$m" && [ -z "${MODEL_EXPORT[$m]:-}" ]; then
        ONNX_FILE="$SMOLVLA_DIR/${m}.onnx"
        if [ ! -f "$ONNX_FILE" ]; then
            echo "ERROR: Missing $ONNX_FILE"
            echo "  Download SmolVLA models: hf download ainekko/smolvla_base_onnx --local-dir $SMOLVLA_DIR/"
            exit 1
        fi
        ONNX_META=$(python3 -c "
import onnx, json
model = onnx.load('$ONNX_FILE')
inputs = []
for inp in model.graph.input:
    name = inp.name
    dims = [d.dim_value if d.dim_value > 0 else 1 for d in inp.type.tensor_type.shape.dim]
    inputs.append((name, dims))
print(inputs[0][0])
print(len(inputs[0][1]))
print(json.dumps(inputs))
" 2>&1)
        ONNX_INPUT_NAME=$(echo "$ONNX_META" | sed -n '1p')
        ONNX_INPUT_RANK=$(echo "$ONNX_META" | sed -n '2p')
        ONNX_INPUTS_JSON=$(echo "$ONNX_META" | sed -n '3p')
        MODEL_INPUT_NAME[$m]="$ONNX_INPUT_NAME"
        MODEL_INPUT_RANK[$m]="$ONNX_INPUT_RANK"
        MODEL_HW[$m]="_auto_"
        MODEL_EXPORT[$m]="_onnx_provided_"
        MODEL_CONVERTER[$m]="onnx"
        MODEL_ONNX_INPUTS[$m]="$ONNX_INPUTS_JSON"
    fi
done

# Validate model names
for m in "${ALL_MODELS[@]}"; do
    if [ -z "${MODEL_EXPORT[$m]:-}" ]; then
        echo "Unknown model: $m"
        echo "  Built-in:  dronet, mobilenet_v2, yolov8s"
        echo "  SmolVLA:   smolvla (all 9), or individual names"
        echo "  Custom:    --input=<file.onnx|file.py>"
        exit 1
    fi
done

echo "Models to process: ${ALL_MODELS[*]}"
echo ""

# --- Ensure Docker image exists ---
if $CONVERT; then
    if ! sudo docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
        echo "Building Docker image $DOCKER_IMAGE..."
        sudo docker build -f "$SCRIPT_DIR/Dockerfile.qnn-convert" -t "$DOCKER_IMAGE" "$SCRIPT_DIR"
    fi
fi

docker_run() {
    local mount_dir="$1"
    shift
    sudo docker run --rm \
        -v "$QNN_SDK":/qnn:ro \
        -v "$mount_dir":/workspace \
        "$DOCKER_IMAGE" "$@"
}

# ===========================================================================
# Process each model
# ===========================================================================
for MODEL in "${ALL_MODELS[@]}"; do
    echo ""
    echo "###############################################"
    echo "# Model: $MODEL"
    echo "###############################################"
    echo ""

    # Resolve per-model working directory
    if is_smolvla_model "$MODEL"; then
        MODEL_DIR="$SMOLVLA_DIR"
    else
        MODEL_DIR="$WORK_DIR"
    fi

    ONNX_FILE="$MODEL_DIR/${MODEL}.onnx"
    DLC_FILE="$MODEL_DIR/${MODEL}.dlc"
    QUANTIZED_DLC="$MODEL_DIR/${MODEL}_quantized.dlc"
    CAL_DIR="$MODEL_DIR/calibration_data_${MODEL}"
    CAL_LIST="$MODEL_DIR/calibration_list_${MODEL}.txt"
    INPUT_NAME="${MODEL_INPUT_NAME[$MODEL]}"
    HW_SPEC="${MODEL_HW[$MODEL]}"
    if [ "$HW_SPEC" != "_auto_" ]; then
        read -r H W C <<< "$HW_SPEC"
    fi
    BOARD_MODEL_DIR="/root/models/${MODEL}"
    CONVERTER="${MODEL_CONVERTER[$MODEL]}"

    # --- Step 1: Export ONNX ---
    if $EXPORT; then
        echo "=== [$MODEL] Step 1: Exporting ONNX ==="
        EXPORT_SCRIPT="${MODEL_EXPORT[$MODEL]}"
        if [ -n "$INPUT_FILE" ] && [[ "$INPUT_FILE" == *.py ]]; then
            conda run -n "$CONDA_ENV" --no-capture-output \
                python3 "$INPUT_FILE" --output "$ONNX_FILE"
        else
            conda run -n "$CONDA_ENV" --no-capture-output \
                python3 "$SAMPLES_DIR/$EXPORT_SCRIPT" --output "$ONNX_FILE"
        fi
        echo ""
    elif [ -n "$INPUT_FILE" ] && [[ "$INPUT_FILE" == *.onnx ]] && [ "$ONNX_FILE" != "$INPUT_FILE" ]; then
        echo "=== [$MODEL] Using provided ONNX: $INPUT_FILE ==="
        cp "$INPUT_FILE" "$ONNX_FILE"
    fi

    # --- Step 2: Convert ONNX → DLC ---
    if $CONVERT; then
        if [ ! -f "$ONNX_FILE" ]; then
            echo "ERROR: Missing $ONNX_FILE — skipping conversion"
            continue
        fi

        echo "=== [$MODEL] Step 2: Converting to DLC ==="

        if [ "$CONVERTER" = "onnx" ]; then
            # Build per-input --input_dim and --input_layout flags
            INPUT_DIM_FLAGS=""
            LAYOUT_FLAG=""
            if [ "$HW_SPEC" = "_auto_" ]; then
                INPUTS_JSON="${MODEL_ONNX_INPUTS[$MODEL]:-}"
                INPUT_DIM_FLAGS=$(python3 -c "
import json
inputs = json.loads('${INPUTS_JSON}')
flags = []
for name, dims in inputs:
    flags.append('-d {} {}'.format(name, ','.join(str(d) for d in dims)))
    if len(dims) == 4:
        flags.append('--input_layout {} NCHW'.format(name))
    else:
        flags.append('--input_layout {} NONTRIVIAL'.format(name))
print(' '.join(flags))
")
            else
                INPUT_RANK="${MODEL_INPUT_RANK[$MODEL]:-4}"
                if [ "$INPUT_RANK" -eq 4 ]; then
                    LAYOUT_FLAG="--input_layout ${INPUT_NAME} NCHW"
                fi
            fi

            if $RECONVERT; then
                # --reconvert: skip onnxsim, convert directly from original ONNX
                echo "  Reconverting (skipping onnxsim)..."
                docker_run "$MODEL_DIR" bash -c "\
                    pip install -q 'numpy<2' && \
                    python3.10 /qnn/bin/x86_64-linux-clang/snpe-onnx-to-dlc \
                        --input_network /workspace/${MODEL}.onnx \
                        ${LAYOUT_FLAG} \
                        ${INPUT_DIM_FLAGS} \
                        --output_path /workspace/${MODEL}.dlc"
                sudo chown "$(id -u):$(id -g)" "$DLC_FILE"
            else
                # Standard path: onnxsim → snpe-onnx-to-dlc
                docker_run "$MODEL_DIR" bash -c "\
                    pip install -q onnxruntime onnx-simplifier 'numpy<2' && \
                    python3.10 -c \"
import onnx
from onnxsim import simplify
model = onnx.load('/workspace/${MODEL}.onnx')
model_simp, check = simplify(model)
assert check, 'Simplification failed'
onnx.save(model_simp, '/workspace/${MODEL}_simplified.onnx')
print('Simplified ONNX saved')
\" && \
                    python3.10 /qnn/bin/x86_64-linux-clang/snpe-onnx-to-dlc \
                        --input_network /workspace/${MODEL}_simplified.onnx \
                        ${LAYOUT_FLAG} \
                        ${INPUT_DIM_FLAGS} \
                        --output_path /workspace/${MODEL}.dlc"
                sudo chown "$(id -u):$(id -g)" \
                    "$MODEL_DIR/${MODEL}_simplified.onnx" "$DLC_FILE"
            fi

        elif [ "$CONVERTER" = "tflite" ]; then
            # YOLOv8s path: ONNX → TFLite (via onnx2tf) → DLC
            echo "  Converting ONNX → TFLite (onnx2tf)..."
            conda run -n "$CONDA_ENV" --no-capture-output \
                python3 "$SAMPLES_DIR/onnx2tf_convert.py" \
                    --input "$ONNX_FILE" \
                    --output-dir "$MODEL_DIR/${MODEL}_saved_model"

            echo "  Converting TFLite → DLC..."
            docker_run "$MODEL_DIR" bash -c "\
                apt-get update -qq && \
                apt-get install -y -qq libllvm14 >/dev/null 2>&1 && \
                pip install -q 'numpy<2' decorator attrs scipy psutil pytest tflite >/dev/null 2>&1 && \
                python3.10 /qnn/bin/x86_64-linux-clang/snpe-tflite-to-dlc \
                    --input_network /workspace/${MODEL}_saved_model/${MODEL}_float32.tflite \
                    --output_path /workspace/${MODEL}.dlc"

            sudo chown "$(id -u):$(id -g)" "$DLC_FILE"
        fi

        echo "DLC saved: $DLC_FILE ($(du -h "$DLC_FILE" | cut -f1))"
        echo ""

        # (Re)generate calibration data
        rm -rf "$CAL_DIR"
        echo "  Generating calibration data..."
        mkdir -p "$CAL_DIR"
        if [ "$HW_SPEC" = "_auto_" ]; then
            INPUTS_JSON="${MODEL_ONNX_INPUTS[$MODEL]:-}"
            python3 -c "
import numpy as np, json, os

inputs = json.loads('$INPUTS_JSON')
cal_list = []
for i in range(10):
    paths = []
    for name, dims in inputs:
        data = np.random.randn(*dims).astype(np.float32)
        path = os.path.join('$CAL_DIR', '{}_{}.raw'.format(name, i))
        data.tofile(path)
        paths.append('/workspace/calibration_data_${MODEL}/{}_{}.raw'.format(name, i))
    cal_list.append(' '.join(paths))

with open('$CAL_LIST', 'w') as f:
    for line in cal_list:
        f.write(line + '\n')
"
        else
            python3 -c "
import numpy as np
for i in range(10):
    data = np.random.randn(1, $H, $W, $C).astype(np.float32)
    data.tofile('$CAL_DIR/input_{}.raw'.format(i))
"
            : > "$CAL_LIST"
            for i in $(seq 0 9); do
                echo "/workspace/calibration_data_${MODEL}/input_${i}.raw" >> "$CAL_LIST"
            done
        fi

        # Quantize DLC → INT8
        echo "=== [$MODEL] Step 2b: Quantizing DLC → INT8 ==="
        if docker_run "$MODEL_DIR" python3.10 /qnn/bin/x86_64-linux-clang/qairt-quantizer \
            --input_dlc "/workspace/${MODEL}.dlc" \
            --output_dlc "/workspace/${MODEL}_quantized.dlc" \
            --input_list "/workspace/calibration_list_${MODEL}.txt" \
            --act_bitwidth 8 \
            --weights_bitwidth 8 \
            --bias_bitwidth 8 2>&1; then
            sudo chown "$(id -u):$(id -g)" "$QUANTIZED_DLC"
            echo "Quantized DLC: $QUANTIZED_DLC ($(du -h "$QUANTIZED_DLC" | cut -f1))"
        else
            echo "  Quantization failed — DSP backend will be skipped for $MODEL"
        fi
        echo ""
    fi

    # --- Step 3: Deploy to board ---
    if $DEPLOY; then
        echo "=== [$MODEL] Step 3: Deploying to $BOARD_IP ==="
        if [ ! -f "$DLC_FILE" ]; then
            echo "  ERROR: Missing $DLC_FILE — skipping deploy"
            continue
        fi
        ssh "${BOARD_USER}@${BOARD_IP}" "mkdir -p $BOARD_MODEL_DIR"
        scp "$DLC_FILE" "${BOARD_USER}@${BOARD_IP}:${BOARD_MODEL_DIR}/${MODEL}.dlc"
        if [ -f "$QUANTIZED_DLC" ]; then
            scp "$QUANTIZED_DLC" "${BOARD_USER}@${BOARD_IP}:${BOARD_MODEL_DIR}/${MODEL}_quantized.dlc"
        fi
        scp "$SCRIPT_DIR/benchmark_qnn.sh" "${BOARD_USER}@${BOARD_IP}:${BOARD_MODEL_DIR}/benchmark_qnn.sh"

        # Generate dummy input on board
        if [ "$HW_SPEC" = "_auto_" ]; then
            INPUTS_JSON="${MODEL_ONNX_INPUTS[$MODEL]:-}"
            B64_JSON=$(echo "$INPUTS_JSON" | base64 -w0)
            ssh "${BOARD_USER}@${BOARD_IP}" "python3 -c '
import numpy as np, json, base64
inputs = json.loads(base64.b64decode(\"${B64_JSON}\").decode())
entries = []
for name, dims in inputs:
    data = np.random.randn(*dims).astype(np.float32)
    path = \"${BOARD_MODEL_DIR}/{}.raw\".format(name)
    data.tofile(path)
    entries.append(\"{}:={}\".format(name, path))
with open(\"${BOARD_MODEL_DIR}/input_list.txt\", \"w\") as f:
    f.write(\" \".join(entries) + \"\\n\")
'"
        else
            ssh "${BOARD_USER}@${BOARD_IP}" "python3 -c '
import numpy as np
data = np.random.randn(1, $H, $W, $C).astype(np.float32)
data.tofile(\"${BOARD_MODEL_DIR}/input.raw\")
with open(\"${BOARD_MODEL_DIR}/input_list.txt\", \"w\") as f:
    f.write(\"${INPUT_NAME}:=${BOARD_MODEL_DIR}/input.raw\\n\")
'"
        fi

        echo "Deployed to ${BOARD_USER}@${BOARD_IP}:${BOARD_MODEL_DIR}/"
        echo ""
    fi

    # --- Step 4: Benchmark on board ---
    if $RUN; then
        echo "=== [$MODEL] Step 4: Benchmarking ==="
        ssh "${BOARD_USER}@${BOARD_IP}" "mkdir -p $BOARD_MODEL_DIR"
        scp "$SCRIPT_DIR/benchmark_qnn.sh" \
            "${BOARD_USER}@${BOARD_IP}:${BOARD_MODEL_DIR}/benchmark_qnn.sh"
        ssh "${BOARD_USER}@${BOARD_IP}" \
            "bash ${BOARD_MODEL_DIR}/benchmark_qnn.sh ${BOARD_MODEL_DIR} ${MODEL} ${ITERS}" \
            2>&1 | tee "/tmp/benchmark_${MODEL}.log"
    fi
done

# ===========================================================================
# Step 5: Collect results and generate plot
# ===========================================================================
if $RUN; then
    echo ""
    echo "###############################################"
    echo "# Collecting results"
    echo "###############################################"

    # Parse RESULT: lines from benchmark logs and merge into existing results
    python3 -c "
import json, glob, re, os

results_path = '$RESULTS_FILE'
if os.path.exists(results_path):
    with open(results_path) as f:
        results = json.load(f)
else:
    results = {}

for logfile in sorted(glob.glob('/tmp/benchmark_*.log')):
    with open(logfile) as f:
        for line in f:
            m = re.match(r'RESULT:(\S+):(\w+):([\d.]+|FAILED)', line.strip())
            if m:
                model, backend, val = m.groups()
                if model not in results:
                    results[model] = {}
                results[model][backend] = float(val) if val != 'FAILED' else None

with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print('Results saved to $RESULTS_FILE')
print(json.dumps(results, indent=2))
"

    echo ""
    echo "Generating benchmark plot..."
    conda run -n "$CONDA_ENV" --no-capture-output \
        python3 "$SCRIPT_DIR/plot_benchmarks.py" --results "$RESULTS_FILE"
    echo "Plot saved to $SCRIPT_DIR/plots/qnn_benchmark.png"

    # Export to CSV if requested
    if [ -n "$PROFILE_CSV" ]; then
        python3 -c "
import json, csv

with open('$RESULTS_FILE') as f:
    results = json.load(f)

csv_path = '$PROFILE_CSV'
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    backends = sorted({b for bs in results.values() for b in bs})
    w.writerow(['model'] + backends)
    for model in sorted(results):
        row = [model] + [results[model].get(b, '') for b in backends]
        w.writerow(row)

print(f'Profile CSV saved to {csv_path}')
"
    fi
fi
