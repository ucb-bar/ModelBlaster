import pytest
import os
import shutil
from robust_kbench.parallel import ParallelKernelExecutor


@pytest.fixture(
    params=[
        "tests/linear",
    ]
)
def task_dir(request):
    return request.param


CONFIG_FNAME = "config_forward.json"


@pytest.fixture
def cuda_files():
    return [
        "tests/linear/k1/forward.cu",
        "tests/linear/k2/forward.cu",
        "tests/linear/k3/forward.cu",
        "tests/linear/k4/forward.cu",
    ]


@pytest.fixture
def executor(task_dir):
    return ParallelKernelExecutor(
        task_dir=task_dir,
        op_atol=1e-5,
        op_rtol=1e-5,
        warmup_time=5,
        repetition_time=10,
        multi_init_settings=False,
        multi_input_settings=False,
        timeout=300,
        torch_prof=True,
        ncu_prof=False,
        clang_tidy=False,
        forward=True,
        config_fname=CONFIG_FNAME,
    )


@pytest.fixture(autouse=True)
def cleanup(executor):
    yield  # This runs the test
    # Cleanup after the test
    for k_d in ["k1", "k2", "k3", "k4"]:
        eval_results_path = os.path.join(executor.task_dir, k_d, "eval_results")
        if os.path.exists(eval_results_path):
            shutil.rmtree(eval_results_path)


DEBUG = True


def test_compile(executor, cuda_files):
    compile_results = executor.compile(cuda_files, debug=DEBUG)
    assert len(compile_results) == 4
    assert isinstance(compile_results[0], dict)
    assert isinstance(compile_results[1], dict)
    assert isinstance(compile_results[2], dict)
    assert isinstance(compile_results[3], dict)


def test_correctness(executor, cuda_files):
    test_results = executor.test(cuda_files, debug=DEBUG)
    assert len(test_results) == 4
    assert isinstance(test_results[0], dict)
    assert isinstance(test_results[1], dict)
    assert isinstance(test_results[2], dict)
    assert isinstance(test_results[3], dict)


def test_evaluate(executor, cuda_files):
    eval_results = executor.evaluate(cuda_files, debug=DEBUG)
    assert len(eval_results) == 4
    assert isinstance(eval_results[0], dict)
    assert isinstance(eval_results[1], dict)
    assert isinstance(eval_results[2], dict)
    assert isinstance(eval_results[3], dict)


def test_profile(executor, cuda_files):
    profile_results = executor.profile(cuda_files, debug=DEBUG)
    assert len(profile_results) == 4
    assert isinstance(profile_results[0], dict)
    assert isinstance(profile_results[1], dict)
    assert isinstance(profile_results[2], dict)
    assert isinstance(profile_results[3], dict)
