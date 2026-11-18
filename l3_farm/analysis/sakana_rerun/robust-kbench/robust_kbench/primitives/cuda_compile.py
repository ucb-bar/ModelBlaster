import os
from torch.utils.cpp_extension import load
import json
from robust_kbench.kernel_task import KernelTask
from robust_kbench.sandbox.cuda_info_fn import get_cuda_gpu_info
from robust_kbench.sandbox.capture_fn import capture_output

COMPILE_FLAGS = [
    "-O3",  # Enable aggressive optimizations (level 3)
    "--use_fast_math",  # Enable fast math operations (may reduce precision)
    "-lineinfo",  # Include line number info in generated code
    "-v",  # Verbose output during compilation
    "-Xcompiler",  # Pass following options to host compiler
    "-Wall",  # Enable all compiler warnings
    "-g",  # Generate debug information
    "-G",  # Generate debug information for device code
]

CUTLASS_PATH = os.path.join(
    os.path.dirname(__file__), "../../third_party/cutlass/include"
)
CUTLASS_PATH = os.path.expanduser(CUTLASS_PATH)


def cuda_compile(
    task: KernelTask,
    cuda_fname: str,
):
    """
    Compile the CUDA kernel for the given task.
    """
    # Get device info for torch eval
    device_info = get_cuda_gpu_info()

    with capture_output() as (stdout, stderr, subprocess_output):
        try:
            load(
                name=task.task_name,
                sources=[cuda_fname],
                extra_cuda_cflags=COMPILE_FLAGS,
                extra_include_paths=[CUTLASS_PATH],
                with_cuda=True,
                verbose=True,
            )
            error = False
            error_msg = ""
        except Exception as e:
            error = True
            error_msg = str(e)
            print(f"Error compiling CUDA kernel: {e}")

    # Get captured output
    captured_out = stdout.getvalue()
    captured_err = stderr.getvalue()
    nvcc_output = "\n".join(subprocess_output)

    cuda_compile_results = {
        "error": error,
        "error_msg": error_msg,
        "stdout": captured_out,
        "stderr": captured_err,
        "compile_output": nvcc_output,
        "device_info": device_info,
    }
    return cuda_compile_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", default="tasks/diagonal_matmul", type=str)
    parser.add_argument(
        "--filename", default="tasks/diagonal_matmul/forward.cu", type=str
    )
    parser.add_argument("--store_results", action="store_true")
    args = parser.parse_args()

    task = KernelTask(args.task_dir)
    # Get directory of filename
    filename_dir = os.path.dirname(args.filename)
    filename_base = os.path.basename(args.filename)

    try:
        cuda_compile_results = cuda_compile(task, args.filename)
        cuda_compile_results["cuda_fname"] = args.filename
    except Exception as e:
        print(f"Error compiling CUDA kernel: {e}")
        exit(1)

    if args.store_results:
        # Store results in filename_dir/filename_base.json
        eval_dir = os.path.join(filename_dir, "eval_results")
        # make eval_dir if it doesn't exist
        os.makedirs(eval_dir, exist_ok=True)
        with open(os.path.join(eval_dir, "compile_results.json"), "w") as f:
            json.dump(cuda_compile_results, f, indent=4)

    # Exit with 1
    exit(1)
