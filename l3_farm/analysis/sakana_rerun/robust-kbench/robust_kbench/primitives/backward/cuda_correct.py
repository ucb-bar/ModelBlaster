from typing import Dict
import os
from torch.utils.cpp_extension import load
import json
from typing import Dict
from robust_kbench.kernel_task import KernelTask
from robust_kbench.sandbox.capture_fn import capture_output
from robust_kbench.sandbox.correct_backward_fn import check_backward_correctness
from robust_kbench.sandbox.results import TestEvalResult
from robust_kbench.sandbox.cuda_info_fn import get_cuda_gpu_info
from robust_kbench.utils import graceful_eval_cleanup

COMPILE_FLAGS = [
    "-O3",  # Enable aggressive optimizations (level 3)
    "--use_fast_math",  # Enable fast math operations (may reduce precision)
]
CUTLASS_PATH = os.path.join(
    os.path.dirname(__file__), "../../../third_party/cutlass/include"
)
CUTLASS_PATH = os.path.expanduser(CUTLASS_PATH)


def cuda_correct_single(
    task: KernelTask,
    cuda_fname: str,
    input_setting: Dict,
    init_setting: Dict,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    num_correct_trials: int = 5,
) -> TestEvalResult:
    """
    Test correctness of the CUDA kernel for a single input and init setting.
    """
    # Get CUDA function
    cuda_fn = load(
        name=task.task_name,
        sources=[cuda_fname],
        extra_cuda_cflags=COMPILE_FLAGS,
        extra_include_paths=[CUTLASS_PATH],
        with_cuda=True,
        verbose=True,
    ).backward

    # Check correctness
    result_correctness = check_backward_correctness(
        task.model,
        task.forward_fn,
        task.autograd_fn,
        cuda_fn,
        task.get_inputs,
        input_setting,
        init_setting,
        num_correct_trials=num_correct_trials,
        rtol=rtol,
        atol=atol,
    )
    return result_correctness


def cuda_correct(
    task: KernelTask,
    cuda_fname: str,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    num_correct_trials: int = 5,
) -> Dict[str, Dict[str, TestEvalResult]]:
    """
    Test the correctness of the CUDA kernel for all input and init settings.
    """
    # Get device info for test
    device_info = get_cuda_gpu_info()

    # Check if CUDA file exists
    if not os.path.exists(cuda_fname):
        raise FileNotFoundError(f"File {cuda_fname} not found")

    # Get all considered input & module initialization settings
    input_settings = task.get_input_settings()
    init_settings = task.get_init_settings()
    configs_str = task.get_configs_str()
    num_configs = len(configs_str)

    all_results, setting_results = {}, {}
    summary_correct = True
    summary_max_diff = 0.0
    summary_total_correct_trials = 0

    # Loop over all init settings
    for i in range(num_configs):
        # create a string from the init setting dict
        init_setting = init_settings[i]
        input_setting = input_settings[i]
        config_str = configs_str[i]
        # Loop over all input settings
        result_correctness = cuda_correct_single(
            task,
            cuda_fname,
            input_setting,
            init_setting,
            rtol,
            atol,
            num_correct_trials,
        )
        graceful_eval_cleanup()
        # create a string from the input setting dict
        setting_results[config_str] = result_correctness.to_dict()

        # Update summary correctness results
        summary_correct = summary_correct and result_correctness.correct
        summary_max_diff = max(summary_max_diff, result_correctness.max_diff)
        summary_total_correct_trials += result_correctness.num_correct

    # Create summary results
    all_results["summary"] = {
        "correct": summary_correct,
        "max_diff": summary_max_diff,
        "total_trials": summary_total_correct_trials,
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
    parser.add_argument("--op_atol", default=1e-5, type=float)
    parser.add_argument("--op_rtol", default=1e-5, type=float)
    parser.add_argument("--multi_init_settings", action="store_true")
    parser.add_argument("--multi_input_settings", action="store_true")
    parser.add_argument("--config_fname", default=None, type=str)
    parser.add_argument("--num_correct_trials", default=5, type=int)
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

    with capture_output() as (stdout, stderr, subprocess_output):
        try:
            error = False
            correct_results = cuda_correct(
                task,
                args.filename,
                atol=args.op_atol,
                rtol=args.op_rtol,
                num_correct_trials=args.num_correct_trials,
            )
            # print(f"Tested CUDA kernel: {args.filename}")
            # print(json.dumps(correct_results, indent=4))
        except Exception as e:
            error = True
            error_msg = str(e)
            print(f"Error testing CUDA kernel: {e}")
            correct_results = {
                "summary": {
                    "correct": False,
                    "error": error_msg,
                },
            }

    # Add captured output to results
    captured_out = stdout.getvalue()
    captured_err = stderr.getvalue()
    nvcc_output = "\n".join(subprocess_output)

    correct_results["stdout"] = captured_out
    correct_results["stderr"] = captured_err
    correct_results["nvcc_output"] = nvcc_output
    correct_results["cuda_fname"] = args.filename

    if args.store_results:
        # Store results in filename_dir/filename_base.json
        eval_dir = os.path.join(filename_dir, "eval_results")
        # make eval_dir if it doesn't exist
        os.makedirs(eval_dir, exist_ok=True)
        with open(os.path.join(eval_dir, "test_results.json"), "w") as f:
            json.dump(correct_results, f, indent=4)

    # Exit with 1
    exit(1)
