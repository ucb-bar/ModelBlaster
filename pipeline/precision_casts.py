"""fp16 <-> fp32 cast ops for the precision policy (A2 item 3; ``precision_policy``).

Inserted only by ``precision_policy.apply`` at the boundaries of an fp32 island in an fp16 IR: an fp32 op reading an
fp16 tensor gets ``cast_f16_to_f32`` (exact), an fp16 op reading an fp32 tensor gets ``cast_f32_to_f16`` (rounds
once, nearest-even, exactly what fp16 storage of that tensor would have done). No IR without a policy contains
them. Hand CUDA kernels: ``gpu_eval/harness/cuda/mb_cuda_kernels_f16_hand.cu``; walker rules:
``gpu_eval/harness/walk_torch.py``.
"""
from __future__ import annotations

import ctypes

CAST_KINDS = ("cast_f16_to_f32", "cast_f32_to_f16")

_SPECS = [
    ("cast_f16_to_f32",
     "void kernel_cast_f16_to_f32(const _Float16 *input, float *output, int n)",
     "Widen an fp16 tensor to fp32 storage (exact): output[i] = (float)input[i], i < n. Inserted by the precision "
     "policy where an fp32 island reads an fp16 tensor.",
     "void kernel_cast_f16_to_f32(const _Float16 *input, float *output, int n) {\n"
     "    for (int i = 0; i < n; i++) output[i] = (float)input[i];\n}\n"),
    ("cast_f32_to_f16",
     "void kernel_cast_f32_to_f16(const float *input, _Float16 *output, int n)",
     "Round an fp32 tensor to fp16 storage once (round-to-nearest-even): output[i] = (_Float16)input[i], i < n. "
     "Inserted by the precision policy where an fp16 op reads a tensor of an fp32 island.",
     "void kernel_cast_f32_to_f16(const float *input, _Float16 *output, int n) {\n"
     "    for (int i = 0; i < n; i++) output[i] = (_Float16)input[i];\n}\n"),
]


def cast_specs(kernel_spec_cls) -> dict:
    """KernelSpecs of the two casts (``reference_kernels`` registers them; taking the class avoids a cycle)."""
    _fp, _hp = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint16)
    argt = {"cast_f16_to_f32": [_hp, _fp, ctypes.c_int], "cast_f32_to_f16": [_fp, _hp, ctypes.c_int]}
    return {op: kernel_spec_cls(op=op, signature=sig, semantics=sem, reference_impl=impl,
                                argtypes_factory=(lambda a=argt[op]: list(a)))
            for op, sig, sem, impl in _SPECS}
