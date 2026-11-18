import os
import json
import signal
import subprocess
from typing import Union, Optional

# Get default test_and_eval_ops_path
base_dir = os.path.dirname(__file__)
cuda_compile_ops_path = os.path.join(base_dir, "cuda_compile.py")
cuda_profile_ops_path = os.path.join(base_dir, "cuda_profile.py")

forward_path = os.path.join(base_dir, "forward")
torch_eval_forward_ops_path = os.path.join(forward_path, "torch_eval.py")
cuda_correct_forward_ops_path = os.path.join(forward_path, "cuda_correct.py")
cuda_eval_forward_ops_path = os.path.join(forward_path, "cuda_eval.py")

backward_path = os.path.join(base_dir, "backward")
torch_eval_backward_ops_path = os.path.join(backward_path, "torch_eval.py")
cuda_correct_backward_ops_path = os.path.join(backward_path, "cuda_correct.py")
cuda_eval_backward_ops_path = os.path.join(backward_path, "cuda_eval.py")


def get_os_env(
    gpu_id: int,
    ext_dir: str,
    cuda_correct: bool = False,
    cuda_compile: bool = False,
) -> dict:
    """Get the environment variables for the subprocess execution."""
    os_env = os.environ.copy()
    os_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os_env["TORCH_EXTENSIONS_DIR"] = ext_dir
    # Needed for MKL threading - weird numpy error otherwise
    os_env["MKL_THREADING_LAYER"] = "GNU"

    if cuda_correct:
        # CUDA_LAUNCH_BLOCKING -- no async API calls = better error messages
        os_env["CUDA_LAUNCH_BLOCKING"] = "1"
        # TORCH_SHOW_CPP_STACKTRACE -- show full stack trace on error
        os_env["TORCH_SHOW_CPP_STACKTRACE"] = "1"
        # TORCH_USE_CUDA_DSA -- enable CUDA Debug Support Assistant
        os_env["TORCH_USE_CUDA_DSA"] = "1"

    if cuda_compile:
        os_env["CMAKE_VERBOSE_MAKEFILE"] = "1"
    return os_env


def exec_command(
    cmd: list,
    os_env: dict,
    timeout: int = 300,
    debug: bool = False,
) -> None:
    """Executes a single CUDA kernel file either as test or eval."""
    if debug:
        print(f"Running command: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=os_env,
            universal_newlines=True,
            bufsize=1,
            start_new_session=True,
        )
        # Print output in real-time while waiting for process to complete
        if debug:
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                print(line, end="")  # Print each line as it comes in

        # Add timeout - terminate the process if taking too long
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Kill the process and its children if timeout occurs
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        raise RuntimeError(f"The kernel execution timed out after {timeout} seconds.")


def torch_eval(
    task_dir: str,
    compile: bool,
    multi_init_settings: bool = False,
    multi_input_settings: bool = False,
    warmup_time: int = 25,
    repetition_time: int = 10000,
    eval_type: str = "kernelbench",
    gpu_id: int = 0,
    ext_dir: str = os.path.expanduser("~/.cache/torch_extensions/py311_cu124"),
    timeout: int = 300,
    debug: bool = False,
    forward: bool = True,
    config_fname: Optional[str] = None,
) -> dict:
    """Evaluates a single CUDA kernel file on a specific GPU."""
    os_env = get_os_env(gpu_id, ext_dir)
    if forward:
        torch_eval_ops_path = torch_eval_forward_ops_path
    else:
        torch_eval_ops_path = torch_eval_backward_ops_path
    cmd = [
        "python",  # Start a fresh Python process
        torch_eval_ops_path,
        "--task_dir",
        task_dir,
        "--warmup_time",
        str(warmup_time),
        "--repetition_time",
        str(repetition_time),
        "--eval_type",
        eval_type,
        "--store_results",
    ]
    # Add whether to run with multiple init and eval settings
    if multi_init_settings:
        cmd.extend(["--multi_init_settings"])
    if multi_input_settings:
        cmd.extend(["--multi_input_settings"])

    # Add config file if provided
    if config_fname:
        cmd.extend(["--config_fname", config_fname])

    # Add whether to run torch, compile, test, or eval
    if compile:
        cmd.extend(["--torch_compile"])
    else:
        cmd.extend(["--torch_native"])

    try:
        exec_command(
            cmd=cmd,
            os_env=os_env,
            timeout=timeout,
            debug=debug,
        )
    except Exception as e:
        print(f"Error evaluating torch: {e}")
        return None

    eval_dir = os.path.join(task_dir, "eval_results")
    if forward:
        eval_dir = os.path.join(eval_dir, "forward")
    else:
        eval_dir = os.path.join(eval_dir, "backward")
    if compile:
        results_path = os.path.join(eval_dir, "torch_compile_results.json")
    else:
        results_path = os.path.join(eval_dir, "torch_native_results.json")

    # Check if file exists
    if not os.path.exists(results_path):
        return None

    with open(results_path, "r") as f:
        results = json.load(f)
    return results


def cuda_compile(
    task_dir: str,
    cuda_fname: str,
    gpu_id: int = 0,
    ext_dir: str = os.path.expanduser("~/.cache/torch_extensions/py311_cu124"),
    timeout: int = 300,
    debug: bool = False,
) -> dict:
    """Compiles a single CUDA kernel file."""
    os_env = get_os_env(gpu_id, ext_dir, cuda_compile=True)
    cmd = [
        "python",  # Start a fresh Python process
        cuda_compile_ops_path,
        "--task_dir",
        task_dir,
        "--filename",
        cuda_fname,
        "--store_results",
    ]
    try:
        exec_command(
            cmd=cmd,
            os_env=os_env,
            timeout=timeout,
            debug=debug,
        )
        # Load results from json file
        eval_dir = os.path.join(os.path.dirname(cuda_fname), "eval_results")
        results_path = os.path.join(eval_dir, "compile_results.json")
        # Check if file exists
        if not os.path.exists(results_path):
            return None

        with open(results_path, "r") as f:
            results = json.load(f)

        return results
    except Exception as e:
        print(f"Error compiling kernel: {e}")
        return None


def cuda_correct(
    task_dir: str,
    cuda_fname: str,
    multi_init_settings: bool = False,
    multi_input_settings: bool = False,
    op_atol: float = 1e-5,
    op_rtol: float = 1e-5,
    num_correct_trials: int = 5,
    gpu_id: int = 0,
    ext_dir: str = os.path.expanduser("~/.cache/torch_extensions/py311_cu124"),
    timeout: int = 300,
    debug: bool = False,
    forward: bool = True,
    config_fname: Optional[str] = None,
) -> Union[dict, None]:
    """Tests a single CUDA kernel file."""
    os_env = get_os_env(gpu_id, ext_dir, cuda_correct=True)
    if forward:
        cuda_correct_ops_path = cuda_correct_forward_ops_path
    else:
        cuda_correct_ops_path = cuda_correct_backward_ops_path
    cmd = [
        "python",  # Start a fresh Python process
        cuda_correct_ops_path,
        "--task_dir",
        task_dir,
        "--filename",
        cuda_fname,
        "--op_atol",
        str(op_atol),
        "--op_rtol",
        str(op_rtol),
        "--num_correct_trials",
        str(num_correct_trials),
        "--store_results",
    ]
    # Add whether to run with multiple init and eval settings
    if multi_init_settings:
        cmd.extend(["--multi_init_settings"])
    if multi_input_settings:
        cmd.extend(["--multi_input_settings"])

    # Add config file if provided
    if config_fname:
        cmd.extend(["--config_fname", config_fname])

    try:
        exec_command(
            cmd=cmd,
            os_env=os_env,
            timeout=timeout,
            debug=debug,
        )
        # Load results from json file
        eval_dir = os.path.join(os.path.dirname(cuda_fname), "eval_results")
        results_path = os.path.join(eval_dir, "test_results.json")

        # Check if file exists
        if not os.path.exists(results_path):
            results = {
                "summary": {
                    "correct": False,
                    "error": "CUDA kernel terminated abnormally. Potentially due to illegal memory access.",
                },
                "cuda_fname": cuda_fname,
                "stdout": "",
                "stderr": "CUDA kernel terminated abnormally. Potentially due to illegal memory access.",
            }
            # Store results in file
            os.makedirs(eval_dir, exist_ok=True)
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            return results

        with open(results_path, "r") as f:
            results = json.load(f)

        return results
    except Exception as e:
        print(f"Error testing kernel: {e}")
        return None


def cuda_eval(
    task_dir: str,
    cuda_fname: str,
    multi_init_settings: bool = False,
    multi_input_settings: bool = False,
    warmup_time: int = 25,
    repetition_time: int = 10000,
    eval_type: str = "kernelbench",
    gpu_id: int = 0,
    ext_dir: str = os.path.expanduser("~/.cache/torch_extensions/py311_cu124"),
    timeout: int = 300,
    debug: bool = False,
    forward: bool = True,
    config_fname: Optional[str] = None,
) -> Union[dict, None]:
    """Evaluates a single CUDA kernel file on a specific GPU."""
    os_env = get_os_env(gpu_id, ext_dir)
    if forward:
        cuda_eval_ops_path = cuda_eval_forward_ops_path
    else:
        cuda_eval_ops_path = cuda_eval_backward_ops_path
    cmd = [
        "python",  # Start a fresh Python process
        cuda_eval_ops_path,
        "--task_dir",
        task_dir,
        "--filename",
        cuda_fname,
        "--repetition_time",
        str(repetition_time),
        "--warmup_time",
        str(warmup_time),
        "--eval_type",
        eval_type,
        "--store_results",
    ]
    # Add whether to run with multiple init and eval settings
    if multi_init_settings:
        cmd.extend(["--multi_init_settings"])
    if multi_input_settings:
        cmd.extend(["--multi_input_settings"])

    # Add config file if provided
    if config_fname:
        cmd.extend(["--config_fname", config_fname])

    try:
        exec_command(
            cmd=cmd,
            os_env=os_env,
            timeout=timeout,
            debug=debug,
        )
        # Load results from json file
        eval_dir = os.path.join(os.path.dirname(cuda_fname), "eval_results")
        results_path = os.path.join(eval_dir, "time_results.json")

        # Check if file exists
        if not os.path.exists(results_path):
            return None

        with open(results_path, "r") as f:
            results = json.load(f)

        return results
    except Exception as e:
        print(f"Error evaluating kernel: {e}")
        return None


def cuda_profile(
    task_dir: str,
    cuda_fname: str,
    torch_prof: bool = True,
    ncu_prof: bool = True,
    clang_tidy: bool = True,
    repetition_time: int = 10000,
    gpu_id: int = 0,
    ext_dir: str = os.path.expanduser("~/.cache/torch_extensions/py311_cu1214"),
    debug: bool = False,
    forward: bool = True,
    config_fname: Optional[str] = None,
) -> dict:
    """Profiles a single CUDA kernel file."""
    # Set GPU using environment variable BEFORE any PyTorch operations
    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    my_env["TORCH_EXTENSIONS_DIR"] = ext_dir
    my_env["MKL_THREADING_LAYER"] = "GNU"

    # NOTE: Runs on a single task configuration
    cmd = [
        "python",  # Start a fresh Python process
        cuda_profile_ops_path,
        "--task_dir",
        task_dir,
        "--cuda_file",
        cuda_fname,
        "--repetition_time",
        str(repetition_time),
    ]
    if torch_prof:
        cmd.extend(["--torch_prof"])
    if ncu_prof:
        cmd.extend(["--ncu_prof"])
    if clang_tidy:
        cmd.extend(["--clang_tidy"])
    if not forward:
        cmd.extend(["--backward"])

    # Add config file if provided
    if config_fname:
        cmd.extend(["--config_fname", config_fname])

    if debug:
        print(f"Running command: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=my_env,
        universal_newlines=True,
        bufsize=1,
        start_new_session=True,
    )

    # check if process is still running - required for full execution
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if debug:
            print(line, end="")  # Print in real-time

    # Load all profile files
    profile_dir = os.path.join(os.path.dirname(cuda_fname), "eval_results")
    profiles = {"ncu": None, "torch": None, "clang": None}
    for file in os.listdir(profile_dir):
        if file.startswith("profile_ncu"):
            with open(os.path.join(profile_dir, file), "r") as f:
                profiles["ncu"] = json.load(f)
        elif file.startswith("profile_torch"):
            with open(os.path.join(profile_dir, file), "r") as f:
                profiles["torch"] = json.load(f)
        elif file.startswith("profile_clang"):
            with open(os.path.join(profile_dir, file), "r") as f:
                profiles["clang"] = json.load(f)
    return profiles
