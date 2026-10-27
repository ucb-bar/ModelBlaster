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

Per-tensor dtypes (docs/tensor_dtypes.md, `--quant native`): every IR tensor
carries its own storage dtype (graph.json `dtype`: f32 / f16 / bf16 for
floats, i64 / i32 / b8 / u8 for the integer and bool graph inputs and
outputs), taken from the source model. An op still has one float storage
dtype (its kind's suffix); where a tensor's dtype differs from what its op
reads or writes, extraction inserts an explicit `cast_<src>_to_<dst>` op.
Weights carry theirs in graph.json `weight_dtypes`. `ir_dtype` of such an IR
is its narrowest float storage (the bound its golden check needs);
`tensor_dtype` / `weight_dtype` give each tensor's own.

No module-level imports beyond the standard library (numpy is imported where
used): the GPU harness loads this file by
path (gpu_eval/harness/mbir.py) without importing ModelBlaster.
"""
from __future__ import annotations

FP32, FP16, BF16 = "fp32", "fp16", "bf16"
# `--quant native`: per-tensor dtypes from the source model (a mixed IR)
NATIVE = "native"
# op-kind suffix and IR tensor dtype per reduced storage dtype
SUFFIX = {FP16: "_f16", BF16: "_bf16"}
TENSOR_DTYPE = {FP32: "f32", FP16: "f16", BF16: "bf16"}
# IR tensor dtype of a float tensor -> its storage dtype (the tolerance key)
FLOAT_TENSOR = {"f32": FP32, "f16": FP16, "bf16": BF16}
# integer / bool tensor dtypes a mixed IR carries as real integers at its
# graph inputs and outputs (b8: bool, one byte 0/1)
INT_TENSOR = ("i64", "i32", "b8", "u8")

# buffer dtype (mb-kb-problem/1 `buffers[].dtype`) -> element bytes
BUFFER_BYTES = {"f32": 4, "f16": 2, "bf16": 2, "i32": 4, "i64": 8, "b8": 1, "u8": 1}
# tensor dtype -> (C type, CUDA type) of its pointer in a cast kernel
CTYPE = {"f32": ("float", "float"), "f16": ("_Float16", "__half"), "bf16": ("__bf16", "__nv_bfloat16"),
         "i64": ("int64_t", "int64_t"), "i32": ("int32_t", "int32_t"), "b8": ("uint8_t", "uint8_t"),
         "u8": ("uint8_t", "uint8_t")}
# the casts a mixed IR uses: every dtype to and from f32 (ints are carried as
# exact f32 inside the graph), and between the two 16-bit floats
CASTS = sorted({(a, "f32") for a in CTYPE if a != "f32"} | {("f32", b) for b in CTYPE if b != "f32"}
               | {("f16", "bf16"), ("bf16", "f16")})

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
    anything else read as fp32 here). A mixed IR (``quant: native``) reads as
    its narrowest float storage: bf16 if any tensor is bf16, else fp16 if any
    is f16, else fp32 -- the bound a check through all of its tensors needs."""
    q = graph.get("quant")
    if q == NATIVE:
        dts = {m.get("dtype") for m in graph.get("tensors", {}).values()}
        dts |= set((graph.get("weight_dtypes") or {}).values())
        return BF16 if "bf16" in dts else FP16 if "f16" in dts else FP32
    return q if q in (FP16, BF16) else FP32


def is_mixed(graph: dict) -> bool:
    """An IR with per-tensor dtypes (``--quant native``)."""
    return graph.get("quant") == NATIVE


def storage_of(tensor_dtype: str | None) -> str:
    """Storage dtype (fp32 / fp16 / bf16: the tolerance key) of an IR tensor
    dtype; integer / bool tensors hold exact values and use fp32's."""
    return FLOAT_TENSOR.get(tensor_dtype or "f32", FP32)


def cast_kind(src: str, dst: str) -> str:
    """Op kind of the cast between two IR tensor dtypes (``cast_bf16_to_f32``).
    Its kind ends in the destination, so ``op_dtype`` is the dtype it stores."""
    return f"cast_{src}_to_{dst}"


def parse_cast(kind: str) -> tuple[str, str] | None:
    """(src, dst) of a per-tensor cast kind, else None (also for the int8
    path's ``cast_i8_to_f16`` / ``cast_f16_to_i8``)."""
    import re
    m = re.fullmatch(r"cast_([a-z0-9]+)_to_([a-z0-9]+)", kind.removeprefix("kernel_"))
    if m and (m.group(1), m.group(2)) in CASTS:
        return m.group(1), m.group(2)
    return None


def tensor_dtype(graph: dict, t: str) -> str:
    """IR dtype of tensor t (graph.json ``dtype``; f32 when absent)."""
    return (graph.get("tensors", {}).get(t) or {}).get("dtype") or "f32"


def weight_dtype(graph: dict, key: str, np_dtype) -> str:
    """Buffer dtype of a weights.npz array: a mixed IR names it in
    ``weight_dtypes``; otherwise as ``buffer_dtype`` with the IR's dtype."""
    wd = (graph.get("weight_dtypes") or {}).get(key)
    if wd:
        return wd
    return buffer_dtype(np_dtype, ir_dtype(graph) if not is_mixed(graph) else FP32)


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


_BF16_CHUNK = 1 << 24          # elements per step: bounded temporaries for multi-GB arrays


def _bf16_hi(u):
    """High halves (uint32, still shifted) of the RNE bf16 roundings of float32 bit patterns
    ``u`` (uint32; nan quiet). uint32 arithmetic: only nan patterns can wrap, and those are
    replaced."""
    import numpy as np
    r = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    nan = (u & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    return np.where(nan, (u | np.uint32(0x400000)) & np.uint32(0xFFFF0000), r)


def round_bf16(a):
    """float32 array of the bfloat16 values nearest to ``a`` (round-to-nearest-
    even; inf / nan kept, nan quiet). numpy only; ``a`` is any float array."""
    import numpy as np
    x = np.ascontiguousarray(a, dtype=np.float32).reshape(-1)
    out = np.empty(x.size, np.uint32)
    u = x.view(np.uint32)
    for i in range(0, x.size, _BF16_CHUNK):
        out[i:i + _BF16_CHUNK] = _bf16_hi(u[i:i + _BF16_CHUNK])
    return out.view(np.float32).reshape(np.shape(a))


def bf16_bits(a):
    """uint16 bit patterns of the bf16 values nearest to ``a`` (the 2-byte
    storage a `__nv_bfloat16` buffer holds), in chunks: no full-size temporary."""
    import numpy as np
    x = np.ascontiguousarray(a, dtype=np.float32).reshape(-1)
    out = np.empty(x.size, np.uint16)
    u = x.view(np.uint32)
    for i in range(0, x.size, _BF16_CHUNK):
        out[i:i + _BF16_CHUNK] = _bf16_hi(u[i:i + _BF16_CHUNK]) >> np.uint32(16)
    return out


def bf16_exact(a) -> bool:
    """Every value of float array ``a`` is a bfloat16 (low 16 bits zero)."""
    import numpy as np
    x = np.ascontiguousarray(a, dtype=np.float32)
    return bool((((x.view(np.uint32) & 0xFFFF) == 0) | np.isnan(x)).all())


def tolerance(dtype: str) -> dict:
    return dict(TOLERANCE[dtype])


def narrowest(dtypes) -> str:
    """The loosest-bounded storage dtype among ``dtypes`` (IR tensor or storage
    dtypes): bf16, else fp16, else fp32."""
    s = {storage_of(d) if d in FLOAT_TENSOR or d in INT_TENSOR else d for d in dtypes}
    return BF16 if BF16 in s else FP16 if FP16 in s else FP32


def cone_dtypes(graph: dict, tensors) -> dict:
    """Per tensor of ``tensors``: the narrowest float storage among it and every
    tensor it depends on (ops' inputs and weights, transitively). An fp32 output
    computed from bf16 intermediates carries their rounding, so its output bound
    is bf16's; an fp32 output of fp32 math keeps fp32's."""
    prod, wts = {}, {}
    for o in graph.get("ops", []):
        for t in o.get("outputs", []) + o.get("internal", []):
            prod[t] = o
    memo: dict = {}

    def cone(t):
        if t in memo:
            return memo[t]
        memo[t] = FP32
        seen, stack, dts = set(), [t], set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            dts.add(tensor_dtype(graph, x))
            o = prod.get(x)
            if o is None:
                continue
            stack.extend(o.get("inputs", []))
            for sub in o.get("sub_ops", [o]):
                for role in ("weight", "bias", "gamma_key", "beta_key"):
                    k = sub.get(role)
                    if isinstance(k, str):
                        dts.add((graph.get("weight_dtypes") or {}).get(k, "f32"))
        memo[t] = narrowest(dts)
        return memo[t]
    return {t: cone(t) for t in tensors}


def np_storage(arr, dtype: str):
    """``arr`` as a mixed IR's npz holds a tensor of IR dtype ``dtype``: bf16 as
    float32 with bf16-exact values (numpy has no bfloat16), f16 as float16,
    f32 as float32, integers / bool at their own numpy type."""
    import numpy as np
    a = np.asarray(arr)
    if dtype == "bf16":
        return round_bf16(a.astype(np.float32))
    return a.astype({"f32": np.float32, "f16": np.float16, "i64": np.int64, "i32": np.int32,
                     "b8": np.bool_, "u8": np.uint8}[dtype])
