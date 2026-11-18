import pytest
import os
import shutil
from robust_kbench.primitives.evaluate import correct_cuda_kernel


@pytest.fixture(
    params=[
        "tasks/linear",
    ]
)
def task_dir(request):
    return request.param


@pytest.fixture
def cuda_fname(task_dir):
    return os.path.join(task_dir, "backward.cu")


DEBUG = False


@pytest.fixture(autouse=True)
def cleanup(task_dir):
    yield  # This runs the test
    # Cleanup after the test
    eval_results_path = os.path.join(task_dir, "eval_results")
    if os.path.exists(eval_results_path):
        shutil.rmtree(eval_results_path)


def test_correct_cuda_kernel(task_dir, cuda_fname):
    test_results = correct_cuda_kernel(
        cuda_code_path=cuda_fname,
        task_dir=task_dir,
        multi_init_settings=True,
        multi_input_settings=True,
        op_atol=1e-5,
        op_rtol=1e-5,
        num_correct_trials=5,
        gpu_id=0,
        ext_dir=os.path.expanduser("~/.cache/torch_extensions/py311_cu124"),
        timeout=300,
        debug=DEBUG,
        forward=False,
    )
    assert test_results["summary"]["correct"] == True
