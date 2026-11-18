import pytest
from robust_kbench.kernel_task import KernelTask
from robust_kbench.primitives.backward.cuda_correct import cuda_correct
from robust_kbench.primitives.cuda_compile import cuda_compile
from robust_kbench.primitives.backward.cuda_eval import cuda_eval
from robust_kbench.primitives.backward.torch_eval import torch_eval

from robust_kbench.sandbox.eval_backward_fn import (
    time_function_kernel_bench as time_function,
)

# Define the task directories you want to test
TASK_DIRS = ["tasks/linear"]
CONFIG_FNAME = "config_backward.json"


@pytest.fixture(params=TASK_DIRS)
def get_ops(request):
    """Fixture that provides ops for each task directory"""
    task_dir = request.param
    return (
        KernelTask(
            task_dir,
            multi_init_settings=True,
            multi_input_settings=True,
            forward=False,
            config_fname=CONFIG_FNAME,
        ),
        task_dir,
    )


def test_torch_eval(get_ops):
    task, task_dir = get_ops
    torch_native_results = torch_eval(
        task,
        time_function,
        warmup_time=5,
        repetition_time=10,
        compile=False,
    )
    assert isinstance(torch_native_results, dict)
    for k in ["summary", "device_info", "eval_settings"]:
        assert k in torch_native_results
    assert "avg_mean_time" in torch_native_results["summary"]
    assert torch_native_results["summary"]["avg_mean_time"] > 0


def test_torch_compile(get_ops):
    task, task_dir = get_ops
    torch_compiled_results = torch_eval(
        task,
        time_function,
        repetition_time=10,
        warmup_time=5,
        compile=True,
    )
    assert isinstance(torch_compiled_results, dict)
    for k in ["summary", "device_info", "eval_settings"]:
        assert k in torch_compiled_results
    assert "avg_mean_time" in torch_compiled_results["summary"]
    assert torch_compiled_results["summary"]["avg_mean_time"] > 0


def test_cuda_compile(get_ops):
    task, task_dir = get_ops
    cuda_compiled_results = cuda_compile(
        task,
        f"{task_dir}/backward.cu",
    )
    assert isinstance(cuda_compiled_results, dict)
    for k in ["error", "error_msg", "stdout", "stderr"]:
        assert k in cuda_compiled_results


def test_cuda_test(get_ops):
    task, task_dir = get_ops
    correct_results = cuda_correct(
        task,
        f"{task_dir}/backward.cu",
        atol=1e-5,
        rtol=1e-5,
        num_correct_trials=5,
    )
    assert isinstance(correct_results, dict)
    for k in ["summary", "device_info", "eval_settings"]:
        assert k in correct_results
    assert correct_results["summary"]["correct"] is True


def test_cuda_eval(get_ops):
    task, task_dir = get_ops
    cuda_results = cuda_eval(
        task,
        f"{task_dir}/backward.cu",
        time_function,
        repetition_time=10,
        warmup_time=5,
    )
    assert isinstance(cuda_results, dict)
    for k in ["summary", "device_info", "eval_settings"]:
        assert k in cuda_results
    assert "avg_mean_time" in cuda_results["summary"]
    assert cuda_results["summary"]["avg_mean_time"] > 0
