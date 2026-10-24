#!/usr/bin/env bash
# Runs every GPU-feasibility harness on the GPU host. Results land in results/.
#   [PY=python] [IR_ROOT=ROOT/ir] [KB=KernelBlaster checkout] [ARCH=89] [SPECS=specs.json] bash run_all.sh
# nvcc must be on PATH (or set CUDA_HOME); MB_PATH_EXCLUDE drops PATH entries.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
H="$ROOT/harness"; PY="${PY:-python}"; IR="${IR_ROOT:-$ROOT/ir}"; R="${RESULTS:-$ROOT/results}"; B="${BUILD:-$ROOT/build}"
KB="${KB:-$ROOT/../third_party/KernelBlaster}"; ARCH="${ARCH:-89}"
mkdir -p "$R" "$B"
SPECS="${SPECS:-$B/specs.json}"; [ -f "$SPECS" ] || $PY "$H/../tools/dump_specs.py" "$SPECS" > /dev/null || exit 1
[ -z "${MB_PATH_EXCLUDE:-}" ] || export PATH=$(echo "$PATH" | tr : '\n' | grep -Ev "$MB_PATH_EXCLUDE" | paste -sd:)
[ -z "${CUDA_HOME:-}" ] || export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# KernelBlaster's compile/parse modules' extra deps, if installed privately
# (pip --target ROOT/pydeps) to leave a shared torch env untouched.
export PYTHONPATH="$ROOT/pydeps${PYTHONPATH:+:$PYTHONPATH}"
cd "$H"

echo "== env"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES" | tee "$R/env.txt"
nvcc --version | tail -1 | tee -a "$R/env.txt"
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)" | tee -a "$R/env.txt"

echo "== H1 walk_torch"
$PY walk_torch.py "$IR"/*/ --json "$R/h1_walk_torch.json" > "$R/h1_walk_torch.log" 2>&1; echo "exit $?"

echo "== build hand kernels"
nvcc -O3 -arch=sm_$ARCH -Xcompiler -fPIC -shared -o "$B/lib_hand.so" cuda/mb_cuda_kernels.cu 2>&1 | tee "$R/hand_build.log"

echo "== H2 ref2cuda"
$PY ref2cuda.py "$SPECS" "$B/seeds" --arch sm_$ARCH --jobs 24 2>&1 | tee "$R/h2_ref2cuda.log"
cp "$B/seeds/ref2cuda_results.json" "$R/h2_ref2cuda.json"

echo "== H3 walk_cuda (hand)"
$PY walk_cuda.py "$IR"/*/ --specs "$SPECS" --hand "$B/lib_hand.so" --impl hand \
    --json "$R/h3_walk_cuda_hand.json" > "$R/h3_walk_cuda_hand.log" 2>&1; echo "exit $?"

echo "== H3 walk_cuda (seed: auto-ported single-thread reference kernels)"
# Seeds are 1 GPU thread each, so only the small IRs are practical here.
timeout 3600 $PY walk_cuda.py "$IR"/{kb_19_ReLU,kb_23_Softmax,kb_40_LayerNorm,kb_97_ScaledDotProductAttention,kb_1_Square_matrix_multiplication_,mlp_control,lenet,relu6net}/ \
    --specs "$SPECS" --seeds "$B/seeds" --impl seed --reps 2 \
    --json "$R/h3_walk_cuda_seed.json" > "$R/h3_walk_cuda_seed.log" 2>&1; echo "exit $?"

echo "== H4 kb_bridge"
T=("$IR/dronet:conv_modules.0" "$IR/dronet:bn_modules.0" "$IR/dronet:linear1"
   "$IR/yolov8_nano:l0.act" "$IR/yolov8_nano:l10" "$IR/mobilenet_v2:features.1.conv.0.0"
   "$IR/kb_1_Square_matrix_multiplication_:matmul" "$IR/yolov8_nano:cat")
$PY kb_bridge.py --kb "$KB" --specs "$SPECS" --hand-cu cuda/mb_cuda_kernels.cu \
    --out "$B/kb_problems" --arch "$ARCH" --seed hand --ncu "${T[@]}" > "$R/h4_kb_bridge_hand.log" 2>&1; echo "exit $?"
$PY kb_bridge.py --kb "$KB" --specs "$SPECS" --hand-cu cuda/mb_cuda_kernels.cu \
    --out "$B/kb_problems" --arch "$ARCH" --seed ref --ncu "$IR/dronet:conv_modules.0" "$IR/yolov8_nano:l0.act" "$IR/kb_1_Square_matrix_multiplication_:matmul" \
    > "$R/h4_kb_bridge_ref.log" 2>&1; echo "exit $?"
cp "$B"/kb_problems/kb_bridge_*.json "$R/" 2>/dev/null
echo done
