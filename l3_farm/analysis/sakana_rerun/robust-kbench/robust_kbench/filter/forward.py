import torch
from robust_kbench.filter.llm import query
from robust_kbench.kernel_task import KernelTask
from robust_kbench.utils import set_seed, easy_to_device
import re


def filter_output_range(task: KernelTask, num_seeds: int = 10):
    """Filter out tasks whose outputs are within the range (-0.01, 0.01)."""
    outputs = []
    input_config = task.get_input_settings()[0]
    init_config = task.get_init_settings()[0]
    for seed in range(num_seeds):
        set_seed(seed)
        inputs = task.get_inputs(**input_config)
        model = task.model(**init_config).to("cuda")
        inputs = easy_to_device(inputs, "cuda")
        with torch.no_grad():
            out = model.forward(*inputs, fn=task.forward_fn).float()
            # Move output to CPU before appending
            outputs.append(out.cpu())

    # Stack outputs and check if ALL values are within (-0.01, 0.01)
    all_outputs = torch.stack(outputs)
    filter_catch = ((all_outputs > -0.01) & (all_outputs < 0.01)).all()

    return bool(filter_catch)


def filter_output_std(task: KernelTask, num_seeds: int = 10):
    """Filter out tasks whose outputs don't vary enough across different seeds."""
    outputs = []
    input_config = task.get_input_settings()[0]
    init_config = task.get_init_settings()[0]
    for seed in range(num_seeds):
        set_seed(seed)
        inputs = task.get_inputs(**input_config)
        model = task.model(**init_config).to("cuda")
        inputs = easy_to_device(inputs, "cuda")
        with torch.no_grad():
            out = model.forward(*inputs, fn=task.forward_fn).float()
            # Move output to CPU before appending
            outputs.append(out.cpu())

    # Check if per dimension std is less than 0.01
    all_outputs = torch.stack(outputs)
    stds = torch.std(all_outputs, dim=0)
    filter_catch = (stds < 0.01).all()

    return bool(filter_catch)


def filter_output_axes(task: KernelTask, num_seeds: int = 10):
    """Filter out tasks whose outputs don't vary enough across different axes."""
    outputs = []
    input_config = task.get_input_settings()[0]
    init_config = task.get_init_settings()[0]
    for seed in range(num_seeds):
        set_seed(seed)
        inputs = task.get_inputs(**input_config)
        model = task.model(**init_config).to("cuda")
        inputs = easy_to_device(inputs, "cuda")
        with torch.no_grad():
            out = model.forward(*inputs, fn=task.forward_fn).float()
            # Move output to CPU before appending
            outputs.append(out.cpu())

    # Stack outputs and check if std across axes is too small
    all_outputs = torch.stack(outputs)

    # Calculate std across each axis and check if any axis has low variance
    axis_stds = []
    for axis in range(all_outputs.ndim):
        axis_stds.append(torch.std(all_outputs, dim=axis))

    # Filter if ANY axis has all stds < 0.01 (indicating low variance along that axis)
    filter_catch = any((std < 0.01).all() for std in axis_stds)

    return bool(filter_catch)


def filter_init_impact(task: KernelTask, num_seeds: int = 10):
    """Filter out tasks whose model weights don't affect the output."""
    outputs = []
    set_seed(42)
    input_config = task.get_input_settings()[0]
    init_config = task.get_init_settings()[0]
    for seed in range(num_seeds):
        set_seed(seed)
        inputs = task.get_inputs(**input_config)
        model = task.model(**init_config).to("cuda")
        inputs = easy_to_device(inputs, "cuda")
        with torch.no_grad():
            out = model.forward(*inputs, fn=task.forward_fn).float()
            # Move output to CPU before appending
            outputs.append(out.cpu())

    # Check if the output is the same for all seeds
    all_outputs = torch.stack(outputs)
    stds = torch.std(all_outputs, dim=0)
    filter_catch = (stds < 0.01).all()
    return bool(filter_catch)


def filter_input_impact(task: KernelTask, num_seeds: int = 10):
    """Filter out tasks whose inputs don't affect the output."""
    outputs = []
    set_seed(42)
    input_config = task.get_input_settings()[0]
    init_config = task.get_init_settings()[0]
    model = task.model(**init_config).to("cuda")

    for seed in range(num_seeds):
        set_seed(seed)
        inputs = task.get_inputs(**input_config)
        inputs = easy_to_device(inputs, "cuda")
        with torch.no_grad():
            out = model.forward(*inputs, fn=task.forward_fn).float()
            # Move output to CPU before appending
            outputs.append(out.cpu())

    # Stack outputs and check if std across axes is too small
    all_outputs = torch.stack(outputs)
    stds = torch.std(all_outputs, dim=0)
    filter_catch = (stds < 0.01).all()
    return bool(filter_catch)


def analyze_pytorch_code(kernel_code: str) -> str:
    return f"""Here is the PyTorch Functional code:

{kernel_code}

Please provide a very careful analysis to determine if the given module_fn function contains redundant or inefficient operations.

Please structure your response as:
REDUNDANT_ANSWER: ###True### or ###False###
INEFFICIENT_ANSWER: ###True### or ###False###
"""


def extract_analysis_answers(llm_output: str) -> tuple[bool, bool]:
    """
    Extract redundancy and inefficiency answers from LLM output.

    Args:
        llm_output: String output from LLM containing ###True### or ###False### markers

    Returns:
        tuple[bool, bool]: (is_redundant, is_inefficient)
    """

    # Define patterns to match
    redundant_pattern = r"REDUNDANT_ANSWER:\s*###(True|False)###"
    inefficient_pattern = r"INEFFICIENT_ANSWER:\s*###(True|False)###"

    # Extract values
    redundant_match = re.search(redundant_pattern, llm_output)
    inefficient_match = re.search(inefficient_pattern, llm_output)

    if not redundant_match or not inefficient_match:
        raise ValueError("Could not find expected answer format in LLM output")

    # Convert string matches to boolean
    is_redundant = redundant_match.group(1) == "True"
    is_inefficient = inefficient_match.group(1) == "True"

    return is_redundant, is_inefficient


def filter_llm_sanity(
    module_text: str,
    model_name: str = "claude-3-7-sonnet-20250219",
    temperature: float = 0.0,
    max_tokens: int = 8192,
):
    """Filter out tasks whose model weights don't affect the output."""
    SYSTEM_PROMPT = """You are an expert PyTorch code engineer specializing in identifying redundant operations of `module_fn` and inefficient operations of the `module_fn` in the provided code."""

    outputs = query(
        model_names=model_name,
        msg=analyze_pytorch_code(module_text),
        system_message=SYSTEM_PROMPT,
        temperatures=temperature,
        max_tokens=max_tokens,
    )

    # Extract True/False response
    is_redundant, is_inefficient = extract_analysis_answers(outputs[0])
    return is_redundant, is_inefficient, outputs[0]
