import os
import numpy as np
from torch.utils.cpp_extension import load
import json
from typing import Dict, Callable
from robust_kbench.kernel_task import KernelTask
from robust_kbench.sandbox.eval_backward_fn import time_function_kernel_bench
from robust_kbench.sandbox.results import TimeEvalResult
from robust_kbench.sandbox.cuda_info_fn import get_cuda_gpu_info
from robust_kbench.utils import graceful_eval_cleanup, easy_to_device
from robust_kbench.sandbox.capture_fn import capture_output

COMPILE_FLAGS = [
    "-O3",  # Enable aggressive optimizations (level 3)
    "--use_fast_math",  # Enable fast math operations (may reduce precision)
]
CUTLASS_PATH = os.path.join(
    os.path.dirname(__file__), "../../../third_party/cutlass/include"
)
CUTLASS_PATH = os.path.expanduser(CUTLASS_PATH)


def cuda_eval(
    task: KernelTask,
    cuda_fname: str,
    eval_runtime_fn: Callable = time_function_kernel_bench,
    warmup_time: int = 25,
    repetition_time: int = 10000,
) -> Dict[str, Dict[str, TimeEvalResult]]:
    """
    Evaluate the CUDA runtime for all input and init settings.
    """
    # Get device info for eval
    device_info = get_cuda_gpu_info()

    # Get all considered input & module initialization settings
    input_settings = task.get_input_settings()
    init_settings = task.get_init_settings()
    configs_str = task.get_configs_str()
    num_configs = len(configs_str)
    all_results = {}
    setting_results = {}
    avg_mean_time, avg_median_time, avg_iqr_time = [], [], []

    # Loop over all init settings
    for i in range(num_configs):
        # create a string from the init setting dict
        config_str = configs_str[i]
        input_setting = input_settings[i]
        init_setting = init_settings[i]
        # Loop over all input settings
        result_runtime = cuda_eval_single(
            task,
            eval_runtime_fn,
            cuda_fname,
            input_setting,
            init_setting,
            warmup_time,
            repetition_time,
        )
        graceful_eval_cleanup()
        # create a string from the input setting dict
        setting_results[config_str] = result_runtime.to_dict()
        avg_mean_time.append(result_runtime.mean_time)
        avg_median_time.append(result_runtime.median_time)
        avg_iqr_time.append(result_runtime.iqr_time)

    all_results["summary"] = {
        # Add average times
        "avg_mean_time": np.mean(avg_mean_time),
        "avg_median_time": np.mean(avg_median_time),
        "avg_iqr_time": np.mean(avg_iqr_time),
        # Add all unique init and input settings to index
        "all_configs_str": configs_str,
    }
    all_results["eval_settings"] = setting_results
    all_results["device_info"] = device_info
    return all_results


def cuda_eval_single(
    task: KernelTask,
    eval_runtime_fn: Callable,
    cuda_fname: str,
    input_setting,
    init_setting,
    warmup_time: int = 25,
    repetition_time: int = 10000,
) -> TimeEvalResult:
    """
    Evaluate the CUDA runtime for a single input and init setting.
    """
    # Check if CUDA file exists
    if not os.path.exists(cuda_fname):
        raise FileNotFoundError(f"File {cuda_fname} not found")

    # Get CUDA function
    cuda_fn_ = load(
        name=task.task_name,
        sources=[cuda_fname],
        extra_cuda_cflags=COMPILE_FLAGS,
        extra_include_paths=[CUTLASS_PATH],
        with_cuda=True,
        verbose=True,
    ).backward
    autograd_fn = task.autograd_fn
    autograd_fn.backward_fn = cuda_fn_

    # Get inputs, model class instance, torch fn
    inputs = task.get_inputs(**input_setting)
    inputs = easy_to_device(inputs, "cuda")
    model = task.model(**init_setting).to("cuda")

    time = eval_runtime_fn(
        model,
        autograd_fn.apply,
        *inputs,
        warmup_time=warmup_time,
        n_iter=repetition_time,
    )
    del model, autograd_fn
    return time


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", default="tasks/diagonal_matmul", type=str)
    parser.add_argument(
        "--filename", default="tasks/diagonal_matmul/kernel.cu", type=str
    )
    parser.add_argument("--repetition_time", default=10000, type=int)
    parser.add_argument("--warmup_time", default=25, type=int)
    parser.add_argument("--eval_type", default="kernelbench", type=str)
    parser.add_argument("--multi_init_settings", action="store_true")
    parser.add_argument("--multi_input_settings", action="store_true")
    parser.add_argument("--config_fname", default=None, type=str)
    parser.add_argument("--store_results", action="store_true")
    args = parser.parse_args()

    # Get directory of filename
    filename_dir = os.path.dirname(args.filename)
    filename_base = os.path.basename(args.filename)
    task = KernelTask(
        args.task_dir,
        multi_input_settings=args.multi_input_settings,
        multi_init_settings=args.multi_init_settings,
        forward=False,
        config_fname=args.config_fname,
    )

    if args.eval_type == "kernelbench":
        eval_runtime_fn = time_function_kernel_bench
    else:
        raise ValueError(f"Invalid eval type: {args.eval_type}")

    with capture_output() as (stdout, stderr, subprocess_output):
        try:
            cuda_results = cuda_eval(
                task,
                args.filename,
                eval_runtime_fn,
                repetition_time=args.repetition_time,
                warmup_time=args.warmup_time,
            )
            print(f"Evaluated CUDA kernel: {args.filename}")
            print(json.dumps(cuda_results, indent=4))
            error = False

        except Exception as e:
            error = True
            error_msg = str(e)
            print(f"Error evaluating CUDA kernel: {e}")
            cuda_results = {
                "error": error_msg,
            }

    # Add captured output to results
    captured_out = stdout.getvalue()
    captured_err = stderr.getvalue()
    nvcc_output = "\n".join(subprocess_output)

    if error:
        cuda_results = {
            "error": error_msg,
        }
    cuda_results["stdout"] = captured_out
    cuda_results["stderr"] = captured_err
    cuda_results["nvcc_output"] = nvcc_output
    cuda_results["cuda_fname"] = args.filename
    print(cuda_results)
    if args.store_results:
        # Store results in filename_dir/filename_base.json
        eval_dir = os.path.join(filename_dir, "eval_results")
        # make eval_dir if it doesn't exist
        os.makedirs(eval_dir, exist_ok=True)
        with open(os.path.join(eval_dir, "time_results.json"), "w") as f:
            json.dump(cuda_results, f, indent=4)
        print(f"Stored results in {eval_dir}/time_results.json")
    # Exit with 1
    exit(1)
