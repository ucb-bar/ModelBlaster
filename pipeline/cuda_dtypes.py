"""Storage dtypes of ModelBlaster's CUDA path: names, spelling, tolerances.

The fp16 contract (docs/fp16_cuda.md): an `_f16` op stores its float tensors
(activations, weights, constants) as IEEE binary16 and computes in fp32 --
every element is widened on load, accumulated in fp32 or better and rounded
once (round-to-nearest-even) on store. Integer tensors (gather indices) stay
int32. In C the type is `_Float16`; in CUDA it is `__half` (nvcc cannot
generate code for `_Float16`; both are the same 2-byte binary16), so the
CUDA entry point of an `_f16` KernelSpec is its signature with that spelling.

No imports beyond the standard library: the GPU harness loads this file by
path (gpu_eval/harness/mbir.py) without importing ModelBlaster.
"""
from __future__ import annotations

FP32, FP16 = "fp32", "fp16"

# buffer dtype (mb-kb-problem/1 `buffers[].dtype`) -> element bytes
BUFFER_BYTES = {"f32": 4, "f16": 2, "i32": 4, "i64": 8}

# Tolerances per storage dtype. fp32: the Level2/3 gates' bounds, unchanged.
# fp16: from the fp16 rounding measured on KernelBench Level3 (all 50 IRs,
# docs/fp16_cuda.md "Tolerances"); u = 2^-11 is the binary16 unit roundoff,
# one ulp of a value in [1, 2) is 2u = 2^-10:
#   slot_rel/abs  each op alone on the reference's fp16 inputs: err <= slot_rel *
#                 max(|inputs|, |ref|) + slot_abs. Kernel and reference round
#                 (nearly) the same fp32 value once, so they differ by <= 1 ulp:
#                 measured max 7.8e-4 of the scale in the normal range, and
#                 exactly 1 subnormal ulp (2^-24) where activations vanish
#                 (EfficientNetB0/B1, MobileNetV2). Bound: 2 ulps, 2e-3 / 2^-23.
#   output_*      a partition's outputs vs the fp16-storage reference walk
#                 (the driver): 1-ulp differences re-rounded through every
#                 downstream op; measured max 2.0e-3 of max|ref| (NetVLAD).
#   golden_*      a model's output vs the extract golden (fp32 model on the
#                 fp16-rounded inputs): the error of fp16 storage itself;
#                 measured max 1.5e-3 (IR walk, host C, hand CUDA).
#                 Both output bounds: 1e-2, 5x the largest measured error.
TOLERANCE = {
    FP32: {"output_rel": 1e-3, "output_abs": 1e-5, "slot_rel": 1e-3, "slot_abs": 0.0,
           "golden_rel": 1e-3, "golden_abs": 1e-5},
    FP16: {"output_rel": 1e-2, "output_abs": 1e-5, "slot_rel": 2e-3, "slot_abs": 2.0 ** -23,
           "golden_rel": 1e-2, "golden_abs": 1e-5},
}


def op_dtype(kind: str) -> str:
    """Storage dtype of an op kind (``*_f16`` ops are fp16; fused ops follow
    their name, ``fused__a_f16__b_f16``)."""
    return FP16 if kind.endswith("_f16") else FP32


def cuda_signature(signature: str) -> str:
    """The CUDA spelling of a KernelSpec signature (`_Float16` -> `__half`)."""
    return signature.replace("_Float16", "__half")


def buffer_dtype(np_dtype) -> str:
    """mb-kb-problem buffer dtype of a numpy dtype (ints are carried as int32)."""
    kind, size = np_dtype.kind, np_dtype.itemsize
    if kind in "iub":
        return "i32"
    return "f16" if size == 2 else "f32"


def tolerance(dtype: str) -> dict:
    return dict(TOLERANCE[dtype])
