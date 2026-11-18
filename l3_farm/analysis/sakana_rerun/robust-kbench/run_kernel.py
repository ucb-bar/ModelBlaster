import os
import json
import argparse
from robust_kbench.primitives.evaluate import (
    eval_torch_runtime,
    correct_cuda_kernel,
    eval_cuda_kernel,
    prof_cuda_kernel,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", type=str, required=True)
    parser.add_argument("--cuda_code_path", type=str, required=True)
    parser.add_argument("--op_atol", type=float, default=1e-3)
    parser.add_argument("--op_rtol", type=float, default=1e-3)
    parser.add_argument("--rep_time", type=int, default=1000)
    parser.add_argument("--warmup_time", type=int, default=25)
    parser.add_argument("--eval_type", type=str, default="kernelbench")
    parser.add_argument("--multi_init_settings", type=bool, default=False)
    parser.add_argument("--multi_input_settings", type=bool, default=False)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    return args


args = parse_args()
torch_results, torch_compile_results = eval_torch_runtime(
    task_dir=args.task_dir,
    multi_init_settings=args.multi_init_settings,
    multi_input_settings=args.multi_input_settings,
    warmup_time=args.warmup_time,
    repetition_time=args.rep_time,
    eval_type=args.eval_type,
    timeout=args.timeout,
    gpu_id=0,
    ext_dir=os.path.expanduser("~/.cache/torch_extensions/py311_cu124"),
    forward=not args.backward,
    debug=False,
)

correct_results = correct_cuda_kernel(
    task_dir=args.task_dir,
    cuda_code_path=args.cuda_code_path,
    op_atol=args.op_atol,
    op_rtol=args.op_rtol,
    multi_init_settings=args.multi_init_settings,
    multi_input_settings=args.multi_input_settings,
    ext_dir=os.path.expanduser("~/.cache/torch_extensions/py311_cu124"),
    forward=not args.backward,
    timeout=args.timeout,
)
print(f"==> Correctness results: {args.task_dir}")
print(f"    --- CUDA file: {args.cuda_code_path}")
print("    --- CUDA correctness test data")
print(json.dumps(correct_results, indent=4))


if correct_results["summary"]["correct"]:
    cuda_results = eval_cuda_kernel(
        task_dir=args.task_dir,
        cuda_code_path=args.cuda_code_path,
        warmup_time=args.warmup_time,
        repetition_time=args.rep_time,
        eval_type=args.eval_type,
        multi_init_settings=args.multi_init_settings,
        multi_input_settings=args.multi_input_settings,
        timeout=args.timeout,
        forward=not args.backward,
    )
    print(f"==> Evaluation results: {args.task_dir}")
    print(f"    --- CUDA file: {args.cuda_code_path}")
    print("    --- CUDA correctness test data")
    print(json.dumps(cuda_results, indent=4))

if correct_results["summary"]["correct"]:
    # Calculate the speedup
    speedup = (
        torch_results["summary"]["avg_mean_time"]
        / cuda_results["summary"]["avg_mean_time"]
    )
    print(f"==> Speedup CUDA over Torch native: {speedup}x")

    # Calculate the speedup
    speedup = (
        torch_compile_results["summary"]["avg_mean_time"]
        / cuda_results["summary"]["avg_mean_time"]
    )
    print(f"==> Speedup CUDA over Torch compile: {speedup}x")

if correct_results["summary"]["correct"]:
    prof_results = prof_cuda_kernel(
        cuda_code_path=args.cuda_code_path,
        task_dir=args.task_dir,
        torch_prof=True,
        ncu_prof=True,
        clang_tidy=True,
    )
    print(f"==> Profiling results: {args.task_dir}")
    print(f"    --- CUDA file: {args.cuda_code_path}")
    print("    --- CUDA profiling test data")
    print(json.dumps(prof_results, indent=4))
