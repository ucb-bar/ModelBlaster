import torch
from typing import Dict
from robust_kbench.utils import set_seed, easy_to_device
from robust_kbench.sandbox.results import TestEvalResult
from robust_kbench.utils import graceful_eval_cleanup


def check_forward_correctness(
    model_cls,
    module_fn,
    cuda_fn,
    get_inputs_fn,
    input_settings: Dict,
    init_settings: Dict,
    num_correct_trials: int = 5,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> TestEvalResult:
    """
    Check the correctness of the model by comparing the output of the model with the output of the CUDA function.
    """
    num_correct = 0
    correct = True
    max_diff = 0

    # Generate num_correct_trials seeds deterministically from the initial seed
    torch.manual_seed(42)
    correctness_trial_seeds = [
        torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_correct_trials)
    ]

    with torch.no_grad():
        for trial in range(num_correct_trials):
            trial_seed = int(correctness_trial_seeds[trial])

            set_seed(trial_seed)
            inputs = get_inputs_fn(**input_settings)
            inputs = easy_to_device(inputs, "cuda")

            # Make sure that model instances have same weights
            set_seed(trial_seed)
            model_ref = model_cls(**init_settings).to("cuda")
            set_seed(trial_seed)
            model_cuda = model_cls(**init_settings).to("cuda")

            # first eval with cuda model
            output_cuda = model_cuda(*inputs, fn=cuda_fn)
            graceful_eval_cleanup(device="cuda")

            # first eval with original model
            output_ref = model_ref(*inputs, fn=module_fn)
            graceful_eval_cleanup(device="cuda")

            # second eval with cuda model
            output_cuda2 = model_cuda(*inputs, fn=cuda_fn)
            graceful_eval_cleanup(device="cuda")

            # check output difference between original and cuda
            correct1 = torch.allclose(output_ref, output_cuda, atol=atol, rtol=rtol)

            # check output value difference between original and new model
            correct2 = torch.allclose(output_ref, output_cuda2, atol=atol, rtol=rtol)

            if correct1 and correct2:
                num_correct += 1
            else:
                correct = False
                max_diff1 = float(torch.max(torch.abs(output_ref - output_cuda)))
                max_diff2 = float(torch.max(torch.abs(output_ref - output_cuda2)))
                max_diff = max(max_diff, max(max_diff1, max_diff2))
            del model_ref, model_cuda, inputs, output_ref, output_cuda, output_cuda2

    return TestEvalResult(
        correct=correct,
        num_correct=num_correct,
        max_diff=float(max_diff),
        atol=atol,
        rtol=rtol,
        total_correct_trials=num_correct_trials,
    )
