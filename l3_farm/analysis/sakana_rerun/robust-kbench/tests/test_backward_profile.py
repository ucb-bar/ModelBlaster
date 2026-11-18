import pytest
import os
import shutil
from robust_kbench.primitives.cuda_profile import (
    run_torch_profiling,
    run_ncu_profiling,
    run_clang_tidy,
)


def test_torch_profile():
    results_torch = run_torch_profiling(
        "tasks/linear",
        "tasks/linear/backward.cu",
        forward=False,
    )
    assert results_torch is not None


def test_ncu_profile():
    results_ncu = run_ncu_profiling(
        "tasks/linear",
        "tasks/linear/backward.cu",
        forward=False,
    )
    assert results_ncu is not None
    KEYS = ["metrics", "rules"]
    for key in KEYS:
        assert key in results_ncu.keys()


def test_clang_tidy():
    results_clang_tidy = run_clang_tidy(
        "tasks/linear/backward.cu",
    )
    assert results_clang_tidy is not None
    KEYS = ["stdout", "stderr", "errored"]
    for key in KEYS:
        assert key in results_clang_tidy.keys()


@pytest.fixture(autouse=True)
def cleanup():
    yield  # This runs the test
    # Cleanup after the test
    eval_results_path = os.path.join("tasks/linear", "eval_results")
    if os.path.exists(eval_results_path):
        shutil.rmtree(eval_results_path)
