import os
import json
import argparse
import sys
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
    parser.add_argument("--rep_time", type=int, default=10)
    parser.add_argument("--warmup_time", type=int, default=1)
    parser.add_argument("--eval_type", type=str, default="kernelbench")
    parser.add_argument("--multi_init_settings", type=bool, default=False)
    parser.add_argument("--multi_input_settings", type=bool, default=False)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--ncu_only", action="store_true")
    args = parser.parse_args()
    return args


args = parse_args()

# Fast path: run only NCU profiling and print cycles
if args.ncu_only:
    # Preflight diagnostics for common NCU lock issues
    try:
        tmp_path = "/tmp"
        lock_path = "/tmp/nsight-compute-lock"
        print(f"==> Debug: checking {tmp_path} permissions")
        try:
            st = os.stat(tmp_path)
            print(
                f"    mode: {oct(st.st_mode & 0o7777)}, uid: {st.st_uid}, gid: {st.st_gid}"
            )
        except Exception as e:
            print(f"    failed to stat {tmp_path}: {e}")
        exists = os.path.exists(lock_path)
        print(f"==> Debug: lock file exists: {exists}")
        if exists:
            try:
                stl = os.stat(lock_path)
                can_write = os.access(lock_path, os.W_OK)
                print(
                    f"    lock mode: {oct(stl.st_mode & 0o777)}, uid: {stl.st_uid}, gid: {stl.st_gid}, writable_by_user: {can_write}"
                )
            except Exception as e:
                print(f"    failed to stat {lock_path}: {e}")
    except Exception as e:
        print(f"==> Debug: error during preflight tmp/lock diagnostics: {e}")

    prof_results = prof_cuda_kernel(
        cuda_code_path=args.cuda_code_path,
        task_dir=args.task_dir,
        torch_prof=False,
        ncu_prof=True,
        clang_tidy=False,
        forward=not args.backward,
        debug=True,
    )
    print(f"==> NCU profiling results: {args.task_dir}")
    print(f"    --- CUDA file: {args.cuda_code_path}")
    print("    --- NCU metrics summary")
    print(json.dumps(prof_results.get("ncu"), indent=4))

    # Attempt to extract raw cycles metric if available
    ncu_data = prof_results.get("ncu") or {}
    metrics = ncu_data.get("metrics") or {}
    cycles_value = None
    cycles_unit = None
    preferred_cycle_keys = [
        "Elapsed Cycles",
        "sm__cycles_elapsed.avg",
        "sm__cycles_elapsed.sum",
        "sm__cycles_elapsed",
    ]
    for key in preferred_cycle_keys:
        if key in metrics and isinstance(metrics[key], dict):
            cycles_value = metrics[key].get("avg_value")
            cycles_unit = metrics[key].get("unit")
            break
    if cycles_value is None:
        for key, val in metrics.items():
            if isinstance(val, dict) and "avg_value" in val and "cycle" in key.lower():
                cycles_value = val.get("avg_value")
                cycles_unit = val.get("unit")
                break
    if cycles_value is not None:
        unit_str = f" {cycles_unit}" if cycles_unit else ""
        print(f"==> Raw performance (cycles, avg): {cycles_value}{unit_str}")
    else:
        print("==> Raw cycles metric not found in NCU output.")
        if isinstance(metrics, dict) and metrics:
            print("    Available metric keys:")
            print(json.dumps(list(metrics.keys()), indent=4))
        # Extra debugging info: list eval_results and show raw NCU output if present
        try:
            profile_dir = os.path.join(os.path.dirname(args.cuda_code_path), "eval_results")
            print(f"==> Debug: listing profile dir: {profile_dir}")
            if os.path.isdir(profile_dir):
                print(json.dumps(sorted(os.listdir(profile_dir)), indent=4))
                raw_ncu_path = os.path.join(profile_dir, "raw_profile_ncu.txt")
                if os.path.exists(raw_ncu_path):
                    print(f"==> Debug: raw NCU output from {raw_ncu_path}")
                    try:
                        with open(raw_ncu_path, "r") as f:
                            raw_text = f.read()
                        # Print last ~2000 chars to keep logs manageable
                        tail = raw_text[-2000:]
                        print(tail)
                        if "InterprocessLockFailed" in raw_text or "nsight-compute-lock" in raw_text:
                            print(
                                "==> Hint: NCU lock error detected. Try one or more of the following (requires sudo):" \
                                "\n    - Remove stale lock: sudo rm -f /tmp/nsight-compute-lock" \
                                "\n    - Ensure /tmp perms: sudo chmod 1777 /tmp" \
                                "\n    - Temporarily relax kernel setting: sudo sysctl fs.protected_regular=0"
                            )
                    except Exception as e:
                        print(f"==> Debug: failed to read raw NCU output: {e}")
            else:
                print("==> Debug: profile dir does not exist.")
        except Exception as e:
            print(f"==> Debug: error while gathering debug info: {e}")
    sys.exit(0)

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
