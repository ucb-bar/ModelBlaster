#!/usr/bin/env bash
set -u  # keep "undefined variable" safety, but drop -e so loop continues

PYTHON_BIN="${PYTHON_BIN:-python3}"

LEVEL=level_1
TASKS_DIR="tasks/kernelbench/$LEVEL"
HIGHLIGHTED_DIR="highlighted/kernelbench/$LEVEL"
if [[ ! -d "$HIGHLIGHTED_DIR" && -d "highlighted_old/kernelbench/$LEVEL" ]]; then
    HIGHLIGHTED_DIR="highlighted_old/kernelbench/$LEVEL"
fi
OUT_DIR="out/kernelbench/$LEVEL"

mkdir -p "$OUT_DIR"

for task_path in "$HIGHLIGHTED_DIR"/task_*; do
    task_name=$(basename "$task_path")
    task_dir="$TASKS_DIR/$task_name"
    cuda_file="$task_path/kernel.cu"
    out_task_dir="$OUT_DIR/$task_name"

    if [[ -d "$task_dir" && -f "$cuda_file" ]]; then
        mkdir -p "$out_task_dir"
        echo "Running $task_name..."
        if ! "$PYTHON_BIN" run_kernel.py \
            --task_dir "$task_dir" \
            --cuda_code_path "$cuda_file" \
            > "$out_task_dir/run.log" 2>&1; then
            echo "Task $task_name failed, see $out_task_dir/run.log"
        fi
    else
        echo "Skipping $task_name (missing task_dir or kernel.cu)"
    fi
done

LEVEL=level_2
TASKS_DIR="tasks/kernelbench/$LEVEL"
HIGHLIGHTED_DIR="highlighted/kernelbench/$LEVEL"
if [[ ! -d "$HIGHLIGHTED_DIR" && -d "highlighted_old/kernelbench/$LEVEL" ]]; then
    HIGHLIGHTED_DIR="highlighted_old/kernelbench/$LEVEL"
fi
OUT_DIR="out/kernelbench/$LEVEL"

mkdir -p "$OUT_DIR"

for task_path in "$HIGHLIGHTED_DIR"/task_*; do
    task_name=$(basename "$task_path")
    task_dir="$TASKS_DIR/$task_name"
    cuda_file="$task_path/kernel.cu"
    out_task_dir="$OUT_DIR/$task_name"

    if [[ -d "$task_dir" && -f "$cuda_file" ]]; then
        mkdir -p "$out_task_dir"
        echo "Running $task_name..."
        if ! "$PYTHON_BIN" run_kernel.py \
            --task_dir "$task_dir" \
            --cuda_code_path "$cuda_file" \
            > "$out_task_dir/run.log" 2>&1; then
            echo "Task $task_name failed, see $out_task_dir/run.log"
        fi
    else
        echo "Skipping $task_name (missing task_dir or kernel.cu)"
    fi
done
