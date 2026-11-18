import os
import pytest
import shutil
from robust_kbench.primitives.execute import (
    torch_eval,
    cuda_compile,
    cuda_correct,
    cuda_eval,
    cuda_profile,
)


@pytest.fixture(params=["tasks/linear"])
def task_dir(request):
    return request.param


@pytest.fixture
def cuda_fname(task_dir):
    return os.path.join(task_dir, "forward.cu")


@pytest.fixture(autouse=True)
def cleanup(task_dir):
    yield  # This runs the test
    # Cleanup after the test
    eval_results_path = os.path.join(task_dir, "eval_results")
    if os.path.exists(eval_results_path):
        shutil.rmtree(eval_results_path)


DEBUG = False


def test_torch_eval(task_dir):
    torch_native_results = torch_eval(
        compile=False,
        task_dir=task_dir,
        warmup_time=5,
        repetition_time=10,
        eval_type="kernelbench",
        gpu_id=0,
        ext_dir=os.path.expanduser("~/.cache/torch_extensions/py311_cu124"),
        timeout=300,
        debug=DEBUG,
        forward=True,
    )
    assert isinstance(torch_native_results, dict)
    for k in ["summary", "device_info", "eval_settings"]:
        assert k in torch_native_results
    assert "avg_mean_time" in torch_native_results["summary"]
    assert torch_native_results["summary"]["avg_mean_time"] > 0
    assert os.path.exists(
        os.path.join(task_dir, "eval_results", "forward", "torch_native_results.json")
    )


def test_torch_compile(task_dir):
    torch_compiled_results = torch_eval(
        compile=True,
        task_dir=task_dir,
        warmup_time=5,
        repetition_time=10,
        eval_type="kernelbench",
        gpu_id=0,
        ext_dir=os.path.expanduser("~/.cache/torch_extensions/py311_cu124"),
        timeout=300,
        debug=DEBUG,
        forward=True,
    )
    assert isinstance(torch_compiled_results, dict)
    for k in ["summary", "device_info", "eval_settings"]:
        assert k in torch_compiled_results
    assert "avg_mean_time" in torch_compiled_results["summary"]
    assert torch_compiled_results["summary"]["avg_mean_time"] > 0
    # check if eval_results/torch_compile_results.json exists
    assert os.path.exists(
        os.path.join(task_dir, "eval_results", "forward", "torch_compile_results.json")
    )


def test_cuda_compile(task_dir, cuda_fname):
    cuda_compiled_results = cuda_compile(
        task_dir=task_dir,
        cuda_fname=cuda_fname,
        gpu_id=0,
        ext_dir=os.path.expanduser("~/.cache/torch_extensions/py311_cu124"),
        timeout=300,
        debug=DEBUG,
    )
    assert isinstance(cuda_compiled_results, dict)
    for k in ["error", "error_msg", "stdout", "stderr"]:
        assert k in cuda_compiled_results
    # check if eval_results/compile_results.json exists
    assert os.path.exists(
        os.path.join(task_dir, "eval_results", "compile_results.json")
    )


def test_cuda_correct(task_dir, cuda_fname):
    correct_results = cuda_correct(
        task_dir=task_dir,
        cuda_fname=cuda_fname,
        op_atol=1e-5,
        op_rtol=1e-5,
        multi_init_settings=False,
        multi_input_settings=False,
        gpu_id=0,
        ext_dir=os.path.expanduser("~/.cache/torch_extensions/py311_cu124"),
        timeout=300,
        debug=DEBUG,
        forward=True,
    )
    assert isinstance(correct_results, dict)
    for k in ["summary", "device_info", "eval_settings"]:
        assert k in correct_results
    assert correct_results["summary"]["correct"] is True
    # check if eval_results/test_results.json exists
    assert os.path.exists(os.path.join(task_dir, "eval_results", "test_results.json"))


def test_cuda_eval(task_dir, cuda_fname):
    cuda_results = cuda_eval(
        task_dir=task_dir,
        cuda_fname=cuda_fname,
        multi_init_settings=False,
        multi_input_settings=False,
        repetition_time=10,
        warmup_time=5,
        eval_type="kernelbench",
        gpu_id=0,
        ext_dir=os.path.expanduser("~/.cache/torch_extensions/py311_cu124"),
        timeout=300,
        debug=DEBUG,
        forward=True,
    )
    assert isinstance(cuda_results, dict)
    for k in ["summary", "device_info", "eval_settings"]:
        assert k in cuda_results
    assert "avg_mean_time" in cuda_results["summary"]
    assert cuda_results["summary"]["avg_mean_time"] > 0
    # check if eval_results/eval_results.json exists
    assert os.path.exists(os.path.join(task_dir, "eval_results", "time_results.json"))


def test_cuda_profile(task_dir, cuda_fname):
    profile_results = cuda_profile(
        task_dir=task_dir,
        cuda_fname=cuda_fname,
        torch_prof=True,
        ncu_prof=True,
        clang_tidy=True,
        gpu_id=0,
        ext_dir=os.path.expanduser("~/.cache/torch_extensions/py311_cu124"),
        debug=DEBUG,
        forward=True,
    )
    assert isinstance(profile_results, dict)
    for k in ["torch", "ncu", "clang"]:
        assert k in profile_results
    # check if eval_results/torch_profile.json exists
    assert os.path.exists(os.path.join(task_dir, "eval_results", "profile_torch.json"))
    assert os.path.exists(os.path.join(task_dir, "eval_results", "profile_ncu.json"))
    assert os.path.exists(os.path.join(task_dir, "eval_results", "profile_clang.json"))
