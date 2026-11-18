import torch
import numpy as np
from robust_kbench.sandbox.results import TimeEvalResult


def time_function_kernel_bench(
    model,
    autograd_fn,
    *args,
    warmup_time: int = 25,
    n_iter: int = 10000,
) -> TimeEvalResult:
    """Time a function using Cuda events."""
    # Warmup trials
    for _ in range(warmup_time):
        model.zero_grad()
        out = model.forward(*args, fn=autograd_fn)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        torch.cuda.synchronize(device="cuda")

    elapsed_times = []

    # Runtime evaluation trials
    for _ in range(n_iter):
        # create event marker default is not interprocess
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        model.zero_grad()
        out = model.forward(*args, fn=autograd_fn)
        grad_out = torch.randn_like(out)
        torch.cuda.synchronize(device="cuda")

        start_event.record()
        out.backward(grad_out)
        end_event.record()

        # Synchronize to ensure the events have completed
        torch.cuda.synchronize(device="cuda")
        # Calculate the elapsed time in milliseconds
        elapsed_time_ms = start_event.elapsed_time(end_event)
        elapsed_times.append(elapsed_time_ms)

    return TimeEvalResult(
        mean_time=np.mean(elapsed_times),
        median_time=np.median(elapsed_times),
        iqr_time=np.percentile(elapsed_times, 75) - np.percentile(elapsed_times, 25),
    )
