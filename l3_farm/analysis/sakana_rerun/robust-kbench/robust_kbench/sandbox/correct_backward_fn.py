import torch
from typing import Dict, Tuple
from robust_kbench.utils import set_seed, easy_to_device
from robust_kbench.sandbox.results import TestEvalResult
from robust_kbench.utils import graceful_eval_cleanup


def collect_gradients(model, inputs):
    """Collect all gradients from model parameters and inputs

    Args:
        model: PyTorch model
        inputs: List/tuple of input tensors that have requires_grad=True

    Returns:
        Dictionary mapping names to gradients
    """
    gradients = {}

    # Collect parameter gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[f"param_{name}"] = param.grad.clone()

    # Collect input gradients
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    for i, input_tensor in enumerate(inputs):
        if input_tensor.grad is not None:
            gradients[f"input_{i}"] = input_tensor.grad.clone()

    return gradients


def check_gradients_close(
    grad_cuda: Dict[str, torch.Tensor],
    grad_ref: Dict[str, torch.Tensor],
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> Tuple[bool, float]:
    """Check if all gradients are close between CUDA and reference implementation

    Args:
        grad_cuda: Dictionary of gradients from CUDA implementation
        grad_ref: Dictionary of gradients from reference implementation
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison

    Returns:
        tuple: (is_correct, max_diff_dict, mismatched_grads)
    """
    all_correct = True
    max_diff_dict = {}
    mismatched_grads = {}
    max_diff_all = 0

    # Check all gradients exist in both dictionaries
    if set(grad_cuda.keys()) != set(grad_ref.keys()):
        missing_cuda = set(grad_ref.keys()) - set(grad_cuda.keys())
        missing_ref = set(grad_cuda.keys()) - set(grad_ref.keys())
        raise ValueError(
            f"Gradient keys don't match. Missing in CUDA: {missing_cuda}, Missing in ref: {missing_ref}"
        )

    # Compare each gradient
    for name in grad_cuda:
        cuda_grad = grad_cuda[name]
        ref_grad = grad_ref[name]

        # Check shapes match
        if cuda_grad.shape != ref_grad.shape:
            all_correct = False
            mismatched_grads[name] = {
                "error": f"Shape mismatch: CUDA {cuda_grad.shape} vs ref {ref_grad.shape}"
            }
            continue

        # Check values are close
        is_close = torch.allclose(cuda_grad, ref_grad, atol=atol, rtol=rtol)
        if not is_close:
            all_correct = False
            max_diff = float(torch.max(torch.abs(cuda_grad - ref_grad)))
            max_diff_all = max(max_diff_all, max_diff)
            max_diff_dict[name] = max_diff
            mismatched_grads[name] = {
                "max_diff": max_diff,
                "max_diff_relative": max_diff / (torch.max(torch.abs(ref_grad)) + 1e-8),
            }

    return all_correct, max_diff_all


def check_backward_correctness(
    model_cls,
    module_fn,
    autograd_fn,
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

    # Set the backward function to the CUDA function
    autograd_fn.backward_fn = cuda_fn
    for trial in range(num_correct_trials):
        trial_seed = int(correctness_trial_seeds[trial])

        set_seed(trial_seed)
        inputs_ref = easy_to_device(get_inputs_fn(**input_settings), "cuda")
        set_seed(trial_seed)
        inputs_cuda = easy_to_device(get_inputs_fn(**input_settings), "cuda")

        # Ensure gradients are retained for intermediate tensors
        # Seems required for input gradients to be available
        for x in inputs_cuda:
            if x.requires_grad:
                x.retain_grad()
        for y in inputs_ref:
            if y.requires_grad:
                y.retain_grad()

        # Make sure that model instances have same weights
        set_seed(trial_seed)
        model_ref = model_cls(**init_settings).to("cuda")
        set_seed(trial_seed)
        model_cuda = model_cls(**init_settings).to("cuda")

        # first eval with cuda model
        model_cuda.zero_grad()
        output_cuda = model_cuda(*inputs_cuda, fn=autograd_fn.apply)
        graceful_eval_cleanup(device="cuda")
        set_seed(trial_seed)
        grad_output_cuda = torch.randn_like(output_cuda)
        output_cuda.backward(grad_output_cuda)
        grad_cuda = collect_gradients(model_cuda, inputs_cuda)

        # first eval with original model
        model_ref.zero_grad()
        output_ref = model_ref(*inputs_ref, fn=module_fn)
        graceful_eval_cleanup(device="cuda")
        set_seed(trial_seed)
        grad_output_ref = torch.randn_like(output_ref)
        output_ref.backward(grad_output_ref)
        grad_ref = collect_gradients(model_ref, inputs_ref)

        # check output difference between original and cuda
        trial_correct, trial_max_diff = check_gradients_close(
            grad_cuda, grad_ref, atol=atol, rtol=rtol
        )

        if trial_correct:
            num_correct += 1
        else:
            correct = False
            max_diff = max(max_diff, trial_max_diff)
        del (
            model_ref,
            model_cuda,
            inputs_ref,
            inputs_cuda,
            output_ref,
            output_cuda,
            grad_output_cuda,
        )

    return TestEvalResult(
        correct=correct,
        num_correct=num_correct,
        max_diff=float(max_diff),
        atol=atol,
        rtol=rtol,
        total_correct_trials=num_correct_trials,
    )
