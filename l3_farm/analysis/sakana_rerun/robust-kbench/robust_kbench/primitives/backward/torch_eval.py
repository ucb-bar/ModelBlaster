from typing import Dict, Callable
import os
import torch
import numpy as np
import torch
import json
from robust_kbench.kernel_task import KernelTask
from robust_kbench.sandbox.eval_backward_fn import time_function_kernel_bench
from robust_kbench.sandbox.results import TimeEvalResult
from robust_kbench.sandbox.cuda_info_fn import get_cuda_gpu_info
from robust_kbench.utils import graceful_eval_cleanup, easy_to_device


def torch_eval_single(
    task: KernelTask,
    eval_runtime_fn: Callable,
    input_setting,
    init_setting,
    compile=False,
    warmup_time=25,
    repetition_time=10000,
):
    """
    Evaluate the torch runtime for a single input and init setting.
    """
    # Get inputs, model class instance, torch function (compiled or not)
    inputs = task.get_inputs(**input_setting)
    inputs = easy_to_device(inputs, "cuda")
    model = task.model(**init_setting).to("cuda")
    torch_fn = task.forward_fn
    torch_fn_ = torch.compile(torch_fn, mode="max-autotune") if compile else torch_fn

    time = eval_runtime_fn(
        model,
        torch_fn_,
        *inputs,
        warmup_time=warmup_time,
        n_iter=repetition_time,
    )
    del model, torch_fn, torch_fn_
    return time


def torch_eval(
    task: KernelTask,
    eval_runtime_fn: Callable,
    warmup_time: int = 25,
    repetition_time: int = 10000,
    compile: bool = False,
) -> Dict[str, Dict[str, TimeEvalResult]]:
    """
    Evaluate the torch runtime for all input and init settings.
    """
    # Get device info for torch eval
    device_info = get_cuda_gpu_info()

    # Get all considered input & module initialization settings
    input_settings = task.get_input_settings()
    init_settings = task.get_init_settings()
    configs_str = task.get_configs_str()
    assert len(input_settings) == len(init_settings) == len(configs_str)
    num_configs = len(input_settings)
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
        result_runtime = torch_eval_single(
            task,
            eval_runtime_fn,
            input_setting,
            init_setting,
            compile,
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
        "avg_mean_time": np.mean(avg_mean_time),
        "avg_median_time": np.mean(avg_median_time),
        "avg_iqr_time": np.mean(avg_iqr_time),
        "all_configs_str": configs_str,
    }
    all_results["eval_settings"] = setting_results
    all_results["device_info"] = device_info
    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", default="tasks/diagonal_matmul", type=str)
    parser.add_argument(
        "--filename", default="tasks/diagonal_matmul/kernel.cu", type=str
    )
    parser.add_argument("--torch_native", action="store_true")
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--repetition_time", default=10000, type=int)
    parser.add_argument("--warmup_time", default=25, type=int)
    parser.add_argument("--eval_type", default="kernelbench", type=str)
    parser.add_argument("--multi_init_settings", action="store_true")
    parser.add_argument("--multi_input_settings", action="store_true")
    parser.add_argument("--config_fname", default=None, type=str)
    parser.add_argument("--store_results", action="store_true")
    args = parser.parse_args()

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

    # Get directory of filename
    filename_dir = os.path.dirname(args.filename)
    filename_base = os.path.basename(args.filename)

    if args.torch_native:
        try:
            torch_results = torch_eval(
                task,
                eval_runtime_fn,
                compile=False,
                repetition_time=args.repetition_time,
                warmup_time=args.warmup_time,
            )
        except Exception as e:
            print(f"Error evaluating torch: {e}")

        if args.store_results:
            # Store results in filename_dir/filename_base.json
            eval_dir = os.path.join(args.task_dir, "eval_results", "backward")
            # make eval_dir if it doesn't exist
            os.makedirs(eval_dir, exist_ok=True)
            with open(os.path.join(eval_dir, "torch_native_results.json"), "w") as f:
                json.dump(torch_results, f, indent=4)

    if args.torch_compile:
        try:
            torch_compile_results = torch_eval(
                task,
                eval_runtime_fn,
                compile=True,
                repetition_time=args.repetition_time,
                warmup_time=args.warmup_time,
            )
        except Exception as e:
            print(f"Error evaluating torch compile: {e}")

        if args.store_results:
            # Store results in filename_dir/filename_base.json
            eval_dir = os.path.join(args.task_dir, "eval_results", "backward")
            # make eval_dir if it doesn't exist
            os.makedirs(eval_dir, exist_ok=True)
            with open(os.path.join(eval_dir, "torch_compile_results.json"), "w") as f:
                json.dump(torch_compile_results, f, indent=4)

    # Exit with 1
    exit(1)
