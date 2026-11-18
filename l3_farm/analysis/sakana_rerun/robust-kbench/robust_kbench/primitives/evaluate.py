import tempfile
import uuid
import time
import os
from typing import Optional
from robust_kbench.utils import graceful_eval_cleanup
from robust_kbench.primitives.execute import (
    torch_eval,
    cuda_compile,
    cuda_correct,
    cuda_eval,
    cuda_profile,
)


def eval_torch_runtime(
    task_dir: str,
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
):
    start_time = time.time()
    OP_TYPE = "FORWARD" if forward else "BACKWARD"
    print(f"EVALUATE - START - {OP_TYPE} => Torch native - {eval_type}...")
    # Run eval for torch and compile once
    torch_native_results = torch_eval(
        task_dir=task_dir,
        compile=False,
        multi_init_settings=multi_init_settings,
        multi_input_settings=multi_input_settings,
        warmup_time=warmup_time,
        repetition_time=repetition_time,
        gpu_id=gpu_id,
        ext_dir=ext_dir,
        timeout=timeout,
        eval_type=eval_type,
        debug=debug,
        forward=forward,
        config_fname=config_fname,
    )
    graceful_eval_cleanup()
    end_time = time.time()
    print(
        f"EVALUATE -  DONE - {OP_TYPE} => Torch native - Avg. Runtime: {torch_native_results['summary']['avg_mean_time']:.2f}s - Time: {end_time - start_time:.2f}s"
    )

    print(f"EVALUATE - START - {OP_TYPE} => Torch compile - {eval_type}...")
    start_time = time.time()
    torch_compile_results = torch_eval(
        task_dir=task_dir,
        compile=True,
        multi_init_settings=multi_init_settings,
        multi_input_settings=multi_input_settings,
        warmup_time=warmup_time,
        repetition_time=repetition_time,
        eval_type=eval_type,
        gpu_id=gpu_id,
        ext_dir=ext_dir,
        timeout=timeout,
        debug=debug,
        forward=forward,
        config_fname=config_fname,
    )
    graceful_eval_cleanup()
    end_time = time.time()
    print(
        f"EVALUATE -  DONE - {OP_TYPE} => Torch compile - Avg. Runtime: {torch_compile_results['summary']['avg_mean_time']:.2f}s - Time: {end_time - start_time:.2f}s"
    )
    return torch_native_results, torch_compile_results


def compile_cuda_kernel(
    task_dir: str,
    cuda_code_path: str,
    gpu_id: int = 0,
    ext_dir: str = os.path.expanduser("~/.cache/torch_extensions/py311_cu124"),
    timeout: int = 300,
    debug: bool = False,
):
    start_time = time.time()
    cuda_code_print = "/".join(cuda_code_path.split("/")[-3:])
    print(f"COMPILE  - START => CUDA code {cuda_code_print}...")
    compile_results = cuda_compile(
        task_dir=task_dir,
        cuda_fname=cuda_code_path,
        gpu_id=gpu_id,
        ext_dir=ext_dir,
        timeout=timeout,
        debug=debug,
    )
    graceful_eval_cleanup()
    end_time = time.time()
    print(
        f"COMPILE  -  DONE => CUDA code {cuda_code_print} - Time: {end_time - start_time:.2f}s"
    )
    return compile_results


def correct_cuda_kernel(
    task_dir: str,
    cuda_code_path: str,
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
):
    start_time = time.time()
    unique_id = str(uuid.uuid4())[:8]
    ext_dir = os.path.join(tempfile.gettempdir(), f"torch_extensions_{unique_id}")
    os.makedirs(ext_dir, exist_ok=True)

    cuda_code_print = "/".join(cuda_code_path.split("/")[-3:])
    OP_TYPE = "FORWARD" if forward else "BACKWARD"
    print(f"TESTING  - START - {OP_TYPE} => CUDA code {cuda_code_print}...")
    correct_results = cuda_correct(
        task_dir=task_dir,
        cuda_fname=cuda_code_path,
        multi_init_settings=multi_init_settings,
        multi_input_settings=multi_input_settings,
        op_atol=op_atol,
        op_rtol=op_rtol,
        num_correct_trials=num_correct_trials,
        gpu_id=gpu_id,
        ext_dir=ext_dir,
        timeout=timeout,
        debug=debug,
        forward=forward,
        config_fname=config_fname,
    )
    end_time = time.time()
    print(
        f"TESTING  -  DONE - {OP_TYPE} => CUDA code {cuda_code_print} - Correct: {correct_results['summary']['correct']} - Time: {end_time - start_time:.2f}s"
    )
    graceful_eval_cleanup()
    return correct_results


def eval_cuda_kernel(
    task_dir: str,
    cuda_code_path: str,
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
):
    cuda_code_print = "/".join(cuda_code_path.split("/")[-3:])
    OP_TYPE = "FORWARD" if forward else "BACKWARD"
    print(
        f"EVALUATE - START - {OP_TYPE} => CUDA code {cuda_code_print} - {eval_type}..."
    )
    start_time = time.time()
    cuda_results = cuda_eval(
        task_dir=task_dir,
        cuda_fname=cuda_code_path,
        multi_init_settings=multi_init_settings,
        multi_input_settings=multi_input_settings,
        warmup_time=warmup_time,
        repetition_time=repetition_time,
        eval_type=eval_type,
        gpu_id=gpu_id,
        ext_dir=ext_dir,
        timeout=timeout,
        debug=debug,
        forward=forward,
        config_fname=config_fname,
    )
    end_time = time.time()
    print(
        f"EVALUATE -  DONE - {OP_TYPE} => CUDA code {cuda_code_print} - Avg. Runtime: {cuda_results['summary']['avg_mean_time']:.2f}s - Time: {end_time - start_time:.2f}s"
    )
    graceful_eval_cleanup()
    return cuda_results


def prof_cuda_kernel(
    cuda_code_path: str,
    task_dir: str,
    torch_prof: bool = False,
    ncu_prof: bool = False,
    clang_tidy: bool = False,
    gpu_id: int = 0,
    ext_dir: str = os.path.expanduser("~/.cache/torch_extensions/py311_cu124"),
    debug: bool = False,
    forward: bool = True,
    config_fname: Optional[str] = None,
):
    start_time = time.time()
    # only print last 3 subfolders of cuda_code_path
    cuda_code_print = "/".join(cuda_code_path.split("/")[-3:])
    OP_TYPE = "FORWARD" if forward else "BACKWARD"
    print(f"PROFILE  - START - {OP_TYPE} => CUDA code {cuda_code_print}...")
    prof_results = cuda_profile(
        task_dir=task_dir,
        cuda_fname=cuda_code_path,
        torch_prof=torch_prof,
        ncu_prof=ncu_prof,
        clang_tidy=clang_tidy,
        gpu_id=gpu_id,
        ext_dir=ext_dir,
        debug=debug,
        forward=forward,
        config_fname=config_fname,
    )
    graceful_eval_cleanup()
    end_time = time.time()
    # Get keys that have not None values
    prof_obtained = [k for k, v in prof_results.items() if v is not None]
    print(
        f"PROFILE  -  DONE - {OP_TYPE} => CUDA code {cuda_code_print} => {prof_obtained} - Time: {end_time - start_time:.2f}s"
    )
    return prof_results
