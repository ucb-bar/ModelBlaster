import torch
from torch.utils._pytree import tree_map
from typing import Any


def graceful_eval_cleanup(device: str = "cuda"):
    """
    Clean up env, gpu cache, and compiled CUDA extensions after evaluation
    """
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.synchronize(device=device)


def easy_to_device(pytree: Any, device: torch.device):
    """Move all tensors in the pytree to the given device."""
    return tree_map(
        lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, pytree
    )


def set_seed(seed: int):
    """Set the seed for the random number generator."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
