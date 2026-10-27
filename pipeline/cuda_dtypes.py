"""Storage dtypes of ModelBlaster's CUDA path: names, spelling, tolerances.

The fp16 contract (docs/fp16_cuda.md): an `_f16` op stores its float tensors
(activations, weights, constants) as IEEE binary16 and computes in fp32 --
every element is widened on load, accumulated in fp32 or better and rounded
once (round-to-nearest-even) on store. Integer tensors (gather indices) stay
int32. In C the type is `_Float16`; in CUDA it is `__half` (nvcc cannot
generate code for `_Float16`; both are the same 2-byte binary16), so the
CUDA entry point of an `_f16` KernelSpec is its signature with that spelling.

The bf16 contract (docs/bf16_cuda.md) is the same with bfloat16 (8-bit
exponent, 8-bit significand): `_bf16` ops, `__bf16` in C (GCC/Clang's
storage-and-conversion type), `__nv_bfloat16` in CUDA (<cuda_bf16.h>).
numpy has no bfloat16, so a bf16 IR's weights.npz / io.npz hold float32
arrays whose values are exactly bf16 (`round_bf16`); the IR's `quant: bf16`
and tensor dtype `bf16` say so, and every byte-level consumer (CUDA upload,
kb_export's data.bin) narrows them exactly (`bf16_bits`).

No module-level imports beyond the standard library (numpy is imported where
used): the GPU harness loads this file by
path (gpu_eval/harness/mbir.py) without importing ModelBlaster.
"""
from __future__ import annotations

FP32, FP16, BF16 = "fp32", "fp16", "bf16"
# op-kind suffix and IR tensor dtype per reduced storage dtype
SUFFIX = {FP16: "_f16", BF16: "_bf16"}
TENSOR_DTYPE = {FP32: "f32", FP16: "f16", BF16: "bf16"}

# buffer dtype (mb-kb-problem/1 `buffers[].dtype`) -> element bytes
BUFFER_BYTES = {"f32": 4, "f16": 2, "bf16": 2, "i32": 4, "i64": 8}

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
# bf16: u = 2^-8 (one ulp of [1, 2) is 2^-7 = 7.8e-3), measured the same way
# on Level3 + Level2 bf16 IRs (docs/bf16_cuda.md "Tolerances"):
#   slot_rel/abs  2 ulps, 1.6e-2; bf16 has fp32's exponent range, so
#                 activations that vanish stay normal: no subnormal term.
#   output_*, golden_*  see the doc's table.
TOLERANCE = {
    FP32: {"output_rel": 1e-3, "output_abs": 1e-5, "slot_rel": 1e-3, "slot_abs": 0.0,
           "golden_rel": 1e-3, "golden_abs": 1e-5},
    FP16: {"output_rel": 1e-2, "output_abs": 1e-5, "slot_rel": 2e-3, "slot_abs": 2.0 ** -23,
           "golden_rel": 1e-2, "golden_abs": 1e-5},
    BF16: {"output_rel": 8e-2, "output_abs": 1e-5, "slot_rel": 1.6e-2, "slot_abs": 0.0,
           "golden_rel": 8e-2, "golden_abs": 1e-5},
}


def _suffix_dtype(name: str) -> str:
    if name.endswith("_bf16"):
        return BF16
    return FP16 if name.endswith("_f16") else FP32


def op_dtype(kind: str) -> str:
    """Storage dtype of an op kind or C entry (``*_f16`` ops are fp16, ``*_bf16``
    bf16). A fused op follows its stages: ``fused__<kind>[__<param>]__<kind>...``,
    whose name may end in the param that reads the intermediate
    (``fused__ewise2_f16__ewise2_f16__a``: still fp16)."""
    base = kind.removeprefix("kernel_")
    if base.startswith("fused__"):
        stages = [p for p in base.split("__")[1:] if p]
        return _suffix_dtype(stages[0]) if stages else FP32
    return _suffix_dtype(base)


def ir_dtype(graph: dict) -> str:
    """Storage dtype of an extracted IR (its top-level ``quant``; int8 and
    anything else read as fp32 here)."""
    q = graph.get("quant")
    return q if q in (FP16, BF16) else FP32


def cuda_signature(signature: str) -> str:
    """The CUDA spelling of a KernelSpec signature (`_Float16` -> `__half`,
    `__bf16` -> `__nv_bfloat16`)."""
    return signature.replace("_Float16", "__half").replace("__bf16", "__nv_bfloat16")


def buffer_dtype(np_dtype, storage: str = FP32) -> str:
    """mb-kb-problem buffer dtype of a numpy dtype (ints are carried as int32).
    ``storage``: the IR's dtype -- a float array of a bf16 IR is bf16 (numpy has
    no bfloat16; its float32 values are bf16-exact)."""
    kind, size = np_dtype.kind, np_dtype.itemsize
    if kind in "iub":
        return "i32"
    if storage == BF16:
        return "bf16"
    return "f16" if size == 2 else "f32"


def round_bf16(a):
    """float32 array of the bfloat16 values nearest to ``a`` (round-to-nearest-
    even; inf / nan kept, nan quiet). numpy only; ``a`` is any float array."""
    import numpy as np
    x = np.ascontiguousarray(a, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    nan = np.isnan(x)
    out = np.where(nan, (u | 0x400000) & 0xFFFF0000, r).astype(np.uint32)
    return out.view(np.float32).reshape(np.shape(a))


def bf16_bits(a):
    """uint16 bit patterns of the bf16 values nearest to ``a`` (the 2-byte
    storage a `__nv_bfloat16` buffer holds)."""
    import numpy as np
    return (round_bf16(a).reshape(-1).view(np.uint32) >> 16).astype(np.uint16)


def bf16_exact(a) -> bool:
    """Every value of float array ``a`` is a bfloat16 (low 16 bits zero)."""
    import numpy as np
    x = np.ascontiguousarray(a, dtype=np.float32)
    return bool((((x.view(np.uint32) & 0xFFFF) == 0) | np.isnan(x)).all())


def tolerance(dtype: str) -> dict:
    return dict(TOLERANCE[dtype])
