import subprocess
import os
from typing import Optional, List
import torch
from subprocess import CalledProcessError
import shutil
import site
import sysconfig
from pathlib import Path
import argparse
import tempfile
import json
from robust_kbench.kernel_task import KernelTask
from robust_kbench.primitives.forward.cuda_eval import cuda_eval as cuda_forward_eval
from robust_kbench.primitives.backward.cuda_eval import cuda_eval as cuda_backward_eval

TORCH_PROF_KEYS = [
    "cpu_time_total",
    "device_time_total",
    "cpu_memory_usage",
    "device_memory_usage",
    "self_cpu_time_total",
    "self_device_time_total",
    "self_cpu_memory_usage",
    "self_device_memory_usage",
]

base_path = os.path.join(os.path.dirname(__file__))
cuda_forward_ops_path = os.path.join(
    base_path,
    "forward",
    "cuda_eval.py",
)
cuda_backward_ops_path = os.path.join(
    base_path,
    "backward",
    "cuda_eval.py",
)
config_path = os.path.join(base_path, "config.ncu-cfg")


def run_torch_profiling(
    task_dir: str,
    cuda_file: str,
    rep_time: int = 10000,
    filter_keys: Optional[List[str]] = ["cpu_time_total", "device_time_total"],
    filter_num: int = 5,
    forward: bool = True,
    config_fname: Optional[str] = None,
) -> dict:
    """
    Run Torch Profiler on a CUDA kernel. Filters to the union of filter_num events for each filter_key,
    so the default is the events with the 5 longest cpu_time_total and 5 longest device_time_total.

    Args:
        op_name: Name of the operation to profile (matmul, rms_norm, etc.)
        cuda_file: CUDA source file path from the parent directory (cuda/matmul.cu, etc.)
        num_evals: Number of kernel evaluations
        shape: Shape of the input tensor (e.g. 1024 for a 1024x1024 matrix for matmul)
        filter_keys: List of keys to filter the profiler results by
        filter_num: Number of top events to keep for each filter key
    """
    if forward:
        cuda_eval = cuda_forward_eval
    else:
        cuda_eval = cuda_backward_eval

    try:
        task = KernelTask(task_dir, forward=forward, config_fname=config_fname)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            cuda_eval(
                task,
                cuda_file,
                repetition_time=rep_time,
                warmup_time=5,
            )

        # Write raw profiler output to text file
        out_dir = os.path.join(os.path.dirname(cuda_file), "eval_results")
        raw_output_path = os.path.join(out_dir, "raw_profile_torch.txt")
        # make out_dir if it doesn't exist
        os.makedirs(out_dir, exist_ok=True)
        with open(raw_output_path, "w") as f:
            f.write(str(prof.key_averages()))

        prof_dict = {}
        for event in prof.key_averages():
            prof_dict[event.key] = {
                k: v
                for k, v in event.__dict__.items()
                if k != "key" and k in TORCH_PROF_KEYS
            }

        # Filter events if filter keys are provided
        if filter_keys:
            # For each filter key, get the top filter_num events
            top_events = set()
            for key in filter_keys:
                # Sort events by the filter key value and take top filter_num
                sorted_events = sorted(
                    prof_dict.items(), key=lambda x: x[1].get(key, 0), reverse=True
                )[: min(filter_num, len(prof_dict))]
                # Add event names to set
                top_events.update(event[0] for event in sorted_events)

            # Keep only the events that were in the top filter_num for at least one key
            prof_dict = {k: v for k, v in prof_dict.items() if k in top_events}
    except Exception as e:
        print(f"Error running torch profiling: {e}")
        prof_dict = None
    return prof_dict


def run_ncu_profiling(
    task_dir: str,
    cuda_file: str,
    rep_time: int = 10000,
    kernel_name: str | None = None,
    forward: bool = True,
    config_fname: Optional[str] = None,
) -> dict | None:
    """Run NSight Compute profiling on a CUDA kernel.

    Args:
        task_level: Level of the task to profile
        task_id: ID of the task to profile
        cuda_file: CUDA source file path from the parent directory (cuda/matmul.cu, etc.)
        rep_time: Number of repetitions to evaluate the kernel
        kernel_name: Name of the CUDA function to profile. This is passed to nvidia compute, e.g. "matrixMultiplyKernel" for matmul. Inferred if not provided.
        path_to_test_and_eval_ops_file: Path to the test_and_eval_ops.py file
    """
    # Create temporary file for profiling output
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    temp_file.close()

    # If kernel_name is not provided, we have to infer it from the cuda_file
    # We do this by taking the function name next to __global__ in the cuda_file
    if kernel_name is None:
        with open(cuda_file, "r") as f:
            for line in f:
                if "__global__" in line:
                    kernel_name = line.split("void")[1].split("(")[0].strip()

    if kernel_name is None:
        raise ValueError("Kernel name not found in CUDA file")

    # check that config_path exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    if forward:
        cuda_eval_ops_path = cuda_forward_ops_path
    else:
        cuda_eval_ops_path = cuda_backward_ops_path
    # Build ncu command
    cmd = [
        "ncu",
        "--config-file-path",
        config_path,
        "-k",
        kernel_name,
        "python",
        cuda_eval_ops_path,
        "--task_dir",
        task_dir,
        "--filename",
        cuda_file,
        "--repetition_time",
        str(rep_time),
        "--warmup_time",
        "5",
    ]
    if config_fname:
        cmd.extend(["--config_fname", config_fname])
    # print(f"Running ncu command: {' '.join(cmd)}")
    try:
        # Run the command and wait for completion
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy(),
        )
        stdout, stderr = process.communicate()  # Wait for completion

        # Write raw output to text file
        out_dir = os.path.join(os.path.dirname(cuda_file), "eval_results")
        raw_output_path = os.path.join(out_dir, "raw_profile_ncu.txt")
        # make out_dir if it doesn't exist
        os.makedirs(out_dir, exist_ok=True)
        with open(raw_output_path, "w") as f:
            f.write(stdout)
            f.write("\nSTDERR:\n")
            f.write(stderr)

        # Write the stdout to our temp file for processing
        with open(temp_file.name, "w") as f:
            f.write(stdout)
        ncu_dict = process_ncu_profile(temp_file.name)
        if not ncu_dict["metrics"]:  # Only raise if we didn't get any metrics
            raise CalledProcessError(process.returncode, cmd, stdout, stderr)
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

    return ncu_dict


def process_ncu_profile(prof_file: str) -> dict:
    """Process NSight Compute profiling results.

    Args:
        prof_file: Path to the profiling CSV file

    Returns:
        Tuple containing:
        - Dictionary of metric results with their statistics
        - Dictionary of rules with their descriptions
    """
    # Read CSV file, skipping until after disconnection message
    with open(prof_file, "r") as f:
        lines = f.readlines()
    start_idx = 0
    for i, line in enumerate(lines):
        if "==PROF== Disconnected from process" in line:
            start_idx = i + 2  # Skip the disconnection line and header line
            break

    # Process the CSV data line by line
    data = []
    rules = {}

    for line in lines[start_idx:]:
        # Split on commas that are not within quotes
        parts = []
        current = []
        in_quotes = False

        for char in line:
            if char == '"':
                in_quotes = not in_quotes
            elif char == "," and not in_quotes:
                parts.append("".join(current).strip('"'))
                current = []
            else:
                current.append(char)
        parts.append("".join(current).strip('"\n'))

        if len(parts) >= 15:  # Ensure we have at least the required fields
            if parts[14].strip():  # If has Metric Value
                data.append(
                    {
                        "ID": parts[0],
                        "Metric Name": parts[12],
                        "Metric Unit": parts[13],
                        "Metric Value": parts[14],
                        "Rule Name": parts[15] if len(parts) > 15 else "",
                        "Rule Type": parts[16] if len(parts) > 16 else "",
                        "Rule Description": parts[17] if len(parts) > 17 else "",
                    }
                )
            elif len(parts) > 17 and parts[15].strip():  # If has Rule info
                rules[parts[15]] = {  # Rule Name as key
                    "type": parts[16],
                    "description": parts[17],
                }

    # Convert to results dictionary
    results = {}
    metric_groups = {}

    # Group by metric name
    for row in data:
        metric_name = row["Metric Name"]
        if metric_name not in metric_groups:
            metric_groups[metric_name] = []
        metric_groups[metric_name].append(row)

    # Calculate statistics for each metric
    n = len(set(row["ID"] for row in data))

    for metric_name, group in metric_groups.items():
        try:
            values = [float(row["Metric Value"]) for row in group]
            avg_value = sum(values) / len(values)
            variance = sum((x - avg_value) ** 2 for x in values) / len(values)

            if not group[0]["Rule Name"]:
                # Basic metrics
                results[metric_name] = {
                    "unit": group[0]["Metric Unit"],
                    "avg_value": avg_value,
                    "variance": variance,
                    "n": n,
                }
            else:
                # Metrics with rules
                results[metric_name] = {
                    "unit": group[0]["Metric Unit"],
                    "rule_name": group[0]["Rule Name"],
                    "rule_type": group[0]["Rule Type"],
                    "rule_description": group[0]["Rule Description"],
                    "avg_value": avg_value,
                    "variance": variance,
                    "n": n,
                }
        except ValueError:
            # Skip metrics that can't be converted to float
            continue

    return {"metrics": results, "rules": rules}


def run_clang_tidy(cuda_file: str) -> dict:
    """
    Run clang-tidy on a CUDA source file with all necessary include paths.

    Args:
        cuda_file: Path to the CUDA source file

    Returns:
        Output from clang-tidy as a string

    Raises:
        FileNotFoundError: If clang-tidy or CUDA is not found
        subprocess.CalledProcessError: If clang-tidy fails to run
    """
    # Find clang-tidy
    clang_tidy = shutil.which("clang-tidy")
    if not clang_tidy:
        raise FileNotFoundError("clang-tidy not found in PATH")

    # Find CUDA path
    possible_cuda_paths = [
        "/usr/local/cuda",
        "/opt/cuda",
        os.getenv("CUDA_PATH"),
        os.getenv("CUDA_HOME"),
    ]
    cuda_path = None
    for path in possible_cuda_paths:
        if path and os.path.exists(path):
            cuda_path = path
            break
    if not cuda_path:
        # Try to find it from nvcc location
        nvcc_path = shutil.which("nvcc")
        if nvcc_path:
            cuda_path = str(Path(nvcc_path).parent.parent)
    if not cuda_path:
        raise FileNotFoundError("CUDA installation not found")

    # Get Python paths
    site_packages = site.getsitepackages()[0]
    python_include = sysconfig.get_path("include")
    conda_prefix = os.getenv("CONDA_PREFIX")

    # Construct the command
    cmd = [
        clang_tidy,
        cuda_file,
        "--",
        "-x",
        "cuda",
        f"--cuda-path={cuda_path}",
        f"-I{os.path.join(cuda_path, 'include')}",
        f"-I{os.path.join(site_packages, 'torch', 'include')}",
        f"-I{os.path.join(site_packages, 'torch', 'include', 'torch', 'csrc', 'api', 'include')}",
        f"-I{python_include}",
    ]

    # Add conda include if in conda environment
    if conda_prefix:
        cmd.append(f"-I{os.path.join(conda_prefix, 'include')}")

    try:
        # Run clang-tidy and ensure completion
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,  # Added check=True to ensure completion
        )
        # Write raw output to text file
        out_dir = os.path.join(os.path.dirname(cuda_file), "eval_results")
        raw_output_path = os.path.join(out_dir, "raw_profile_clang.txt")
        # make out_dir if it doesn't exist
        os.makedirs(out_dir, exist_ok=True)
        with open(raw_output_path, "w") as f:
            f.write(result.stdout)
            f.write("\nSTDERR:\n")
            f.write(result.stderr)
        return {"stdout": result.stdout, "stderr": result.stderr, "errored": False}
    except subprocess.CalledProcessError as e:
        # Return error output if clang-tidy fails
        return {"stdout": e.stdout, "stderr": e.stderr, "errored": True}


def profile(
    task_dir: str,
    cuda_file: str,
    config_fname: Optional[str] = None,
    rep_time: int = 10000,
    filter_keys: Optional[List[str]] = ["cpu_time_total", "device_time_total"],
    filter_num: int = 5,
    torch_prof: bool = False,
    ncu_prof: bool = False,
    clang_tidy: bool = False,
    forward: bool = True,
) -> dict:
    profile_dir = os.path.join(os.path.dirname(cuda_file), "eval_results")
    os.makedirs(profile_dir, exist_ok=True)
    results_paths = {}
    if torch_prof:
        print(f"==> Profiling {task_dir} --> Torch Profiling {cuda_file}")
        torch_dict = run_torch_profiling(
            task_dir=task_dir,
            cuda_file=cuda_file,
            rep_time=rep_time,
            filter_keys=filter_keys,
            filter_num=filter_num,
            forward=forward,
            config_fname=config_fname,
        )
        torch_profile_path = os.path.join(profile_dir, "profile_torch.json")
        with open(torch_profile_path, "w") as f:
            json.dump(torch_dict, f)
        print(f"==> Saved Torch profile to {torch_profile_path}")
        results_paths["torch_prof"] = torch_profile_path

    if ncu_prof:
        print(f"==> Profiling {task_dir} --> NCU {cuda_file}")
        ncu_dict = run_ncu_profiling(
            task_dir=task_dir,
            cuda_file=cuda_file,
            rep_time=rep_time,
            forward=forward,
            config_fname=config_fname,
        )
        ncu_profile_path = os.path.join(profile_dir, "profile_ncu.json")
        with open(ncu_profile_path, "w") as f:
            json.dump(ncu_dict, f)
        print(f"==> Saved NCU profile to {ncu_profile_path}")
        results_paths["ncu_prof"] = ncu_profile_path

    if clang_tidy:
        print(f"==> Profiling {task_dir} --> Clang Tidy {cuda_file}")
        clang_dict = run_clang_tidy(cuda_file)
        clang_profile_path = os.path.join(profile_dir, "profile_clang.json")
        with open(clang_profile_path, "w") as f:
            json.dump(clang_dict, f)
        print(f"==> Saved Clang Tidy profile to {clang_profile_path}")
        results_paths["clang_tidy"] = clang_profile_path
    print(f"Profiled CUDA kernel: {results_paths}")
    return results_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile CUDA kernels")
    parser.add_argument(
        "--task_dir",
        default="tasks/linear",
        help="Path to the task directory",
    )
    parser.add_argument(
        "--config_fname",
        default=None,
        help="Path to the config file",
    )
    parser.add_argument(
        "--cuda_file",
        default="tasks/linear/forward.cu",
        help="Path to CUDA source file (e.g. kernels/cuda/matmul.cu)",
    )
    parser.add_argument(
        "--torch_prof",
        action="store_true",
        help="Run Torch Profiling",
    )
    parser.add_argument(
        "--ncu_prof",
        action="store_true",
        help="Run NCU Profiling",
    )
    parser.add_argument(
        "--clang_tidy",
        action="store_true",
        help="Run Clang Tidy",
    )
    parser.add_argument(
        "--backward",
        action="store_true",
        help="Run backward profiling",
    )
    parser.add_argument(
        "--repetition_time",
        type=int,
        default=10000,
        help="Number of repetitions to evaluate the kernel",
    )

    args = parser.parse_args()
    # Save results to profile_dir
    results_paths = profile(
        task_dir=args.task_dir,
        cuda_file=args.cuda_file,
        config_fname=args.config_fname,
        torch_prof=args.torch_prof,
        ncu_prof=args.ncu_prof,
        clang_tidy=args.clang_tidy,
        forward=not args.backward,
        rep_time=args.repetition_time,
    )
