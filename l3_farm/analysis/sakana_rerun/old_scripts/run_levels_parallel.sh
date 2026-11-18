#!/usr/bin/env bash
set -u  # keep "undefined variable" safety, but drop -e so loop continues

# Parallel settings
JOBS="${JOBS:-$(nproc)}"   # max parallel compiles/timing
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Optional single level argument (default level_1); pass "level_2" to target level 2
LEVEL_ARG="${1:-level_1}"

# Use local temp for torch extension cache to avoid network FS contention
BASE_EXT_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions}"
export TORCH_EXTENSIONS_DIR="$BASE_EXT_DIR"
mkdir -p "$BASE_EXT_DIR" || true

run_level() {
    local LEVEL="$1"
    local TASKS_DIR="tasks/kernelbench/$LEVEL"
    local HIGHLIGHTED_DIR="highlighted/kernelbench/$LEVEL"
    if [[ ! -d "$HIGHLIGHTED_DIR" && -d "highlighted_old/kernelbench/$LEVEL" ]]; then
        HIGHLIGHTED_DIR="highlighted_old/kernelbench/$LEVEL"
    fi
    local OUT_DIR="out/kernelbench/$LEVEL"

    mkdir -p "$OUT_DIR"
    echo "==> Running $LEVEL with concurrency=$JOBS, TORCH_EXTENSIONS_DIR=$BASE_EXT_DIR (no NCU)"

    for task_path in "$HIGHLIGHTED_DIR"/task_*; do
        task_name=$(basename "$task_path")
        task_dir="$TASKS_DIR/$task_name"
        cuda_file="$task_path/kernel.cu"
        out_task_dir="$OUT_DIR/$task_name"

        if [[ -d "$task_dir" && -f "$cuda_file" ]]; then
            mkdir -p "$out_task_dir"
            # Limit concurrent jobs
            while [ "$(jobs -pr | wc -l)" -ge "$JOBS" ]; do sleep 0.1; done
            (
                # Isolate extension cache per task to avoid lock/contention
                export TORCH_EXTENSIONS_DIR="$BASE_EXT_DIR/$LEVEL/$task_name"
                mkdir -p "$TORCH_EXTENSIONS_DIR" || true
                echo "Running $task_name..."
                # Compile, test correctness, and timing ONLY (no NCU)
                "$PYTHON_BIN" - "$task_dir" "$cuda_file" > "$out_task_dir/run.log" 2>&1 <<'PY'
import os, sys, json
from robust_kbench.primitives.evaluate import eval_torch_runtime, correct_cuda_kernel, eval_cuda_kernel

task_dir = sys.argv[1]
cuda_code_path = sys.argv[2]

torch_results, torch_compile_results = eval_torch_runtime(
    task_dir=task_dir,
    multi_init_settings=False,
    multi_input_settings=False,
    warmup_time=1,
    repetition_time=10,
    eval_type="kernelbench",
    timeout=600,
    gpu_id=0,
    ext_dir=os.environ.get("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions"),
    forward=True,
    debug=False,
)

correct_results = correct_cuda_kernel(
    task_dir=task_dir,
    cuda_code_path=cuda_code_path,
    op_atol=1e-3,
    op_rtol=1e-2,
    multi_init_settings=False,
    multi_input_settings=False,
    ext_dir=os.environ.get("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions"),
    forward=True,
    timeout=600,
)
print("==> Correctness results:")
print(json.dumps(correct_results, indent=2))

if correct_results.get("summary", {}).get("correct", False):
    cuda_results = eval_cuda_kernel(
        task_dir=task_dir,
        cuda_code_path=cuda_code_path,
        warmup_time=1,
        repetition_time=10,
        eval_type="kernelbench",
        multi_init_settings=False,
        multi_input_settings=False,
        timeout=600,
        forward=True,
    )
    print("==> Evaluation results:")
    print(json.dumps(cuda_results, indent=2))
    speedup_native = torch_results["summary"]["avg_mean_time"] / cuda_results["summary"]["avg_mean_time"]
    speedup_compile = torch_compile_results["summary"]["avg_mean_time"] / cuda_results["summary"]["avg_mean_time"]
    print(f"==> Speedup CUDA over Torch native: {speedup_native}x")
    print(f"==> Speedup CUDA over Torch compile: {speedup_compile}x")
PY
            ) &
        else
            echo "Skipping $task_name (missing task_dir or kernel.cu)"
        fi
    done
    wait
}

run_level "$LEVEL_ARG"
