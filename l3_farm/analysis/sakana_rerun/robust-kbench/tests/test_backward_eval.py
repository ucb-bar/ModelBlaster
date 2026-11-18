import pytest
import os
import shutil
from robust_kbench.primitives.evaluate import (
    eval_torch_runtime,
    compile_cuda_kernel,
    eval_cuda_kernel,
    prof_cuda_kernel,
    correct_cuda_kernel,
)


@pytest.fixture(params=["tasks/linear"])
def task_dir(request):
    return request.param


@pytest.fixture
def cuda_fname():
    return "tasks/linear/backward.cu"


DEBUG = False


@pytest.fixture(autouse=True)
def cleanup(task_dir):
    yield  # This runs the test
    # Cleanup after the test
    eval_results_path = os.path.join(task_dir, "eval_results")
    if os.path.exists(eval_results_path):
        shutil.rmtree(eval_results_path)


def test_eval_torch_runtime(task_dir, cuda_fname):
    eval_results = eval_torch_runtime(
        task_dir=task_dir,
        multi_init_settings=False,
        multi_input_settings=False,
        warmup_time=5,
        repetition_time=10,
        eval_type="kernelbench",
        gpu_id=0,
        ext_dir=os.path.expanduser("~/.cache/torch_extensions/py311_cu124"),
        timeout=300,
        debug=DEBUG,
        forward=False,
    )
    assert eval_results is not None


def test_compile_cuda_kernel(task_dir, cuda_fname):
    compile_results = compile_cuda_kernel(
        cuda_code_path=cuda_fname,
        task_dir=task_dir,
        gpu_id=0,
        ext_dir=os.path.expanduser("~/.cache/torch_extensions/py311_cu124"),
        timeout=300,
        debug=DEBUG,
    )
    assert compile_results is not None


def test_correct_cuda_kernel(task_dir, cuda_fname):
    test_results = correct_cuda_kernel(
        cuda_code_path=cuda_fname,
        task_dir=task_dir,
        multi_init_settings=False,
        multi_input_settings=False,
        op_atol=1e-5,
        op_rtol=1e-5,
        num_correct_trials=5,
        gpu_id=0,
        ext_dir=os.path.expanduser("~/.cache/torch_extensions/py311_cu124"),
        timeout=300,
        debug=DEBUG,
        forward=False,
    )
    assert test_results is not None


def test_eval_cuda_kernel(task_dir, cuda_fname):
    eval_results = eval_cuda_kernel(
        cuda_code_path=cuda_fname,
        task_dir=task_dir,
        multi_init_settings=False,
        multi_input_settings=False,
        warmup_time=5,
        repetition_time=10,
        eval_type="kernelbench",
        gpu_id=0,
        ext_dir=os.path.expanduser("~/.cache/torch_extensions/py311_cu124"),
        timeout=300,
        debug=DEBUG,
        forward=False,
    )
    assert eval_results is not None


def test_prof_cuda_kernel(task_dir, cuda_fname):
    prof_results = prof_cuda_kernel(
        cuda_code_path=cuda_fname,
        task_dir=task_dir,
        torch_prof=True,
        ncu_prof=True,
        clang_tidy=True,
        gpu_id=0,
        ext_dir=os.path.expanduser("~/.cache/torch_extensions/py311_cu124"),
        debug=DEBUG,
        forward=False,
    )
    assert prof_results is not None
