"""Minimal shim for the one use the harness makes of pycuda (robust_kbench/sandbox/cuda_info_fn.py: device-info metadata only,
never timing or kernel launch). pycuda does not build on the L40S nodes; this answers the same calls from torch.cuda."""
