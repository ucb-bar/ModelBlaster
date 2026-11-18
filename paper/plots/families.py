"""Map technique names to optimization families (keyword rules, first match wins)."""
RULES = [
    ("Tensor cores / implicit GEMM", ("tensor_core", "mma", "implicit_gemm", "wgmma", "fp8")),
    ("Async copy / pipelining", ("cp_async", "pipeline", "double_buffer", "mbarrier", "prefetch", "multistage", "warp_specialized", "producer")),
    ("Fusion", ("fusion", "fused", "fold", "epilogue", "single_pass", "concat", "chain")),
    ("Shared memory staging / tiling", ("shared_memory", "smem", "tiling", "tile", "swizzle", "bank_conflict", "halo", "stage")),
    ("Register blocking / ILP", ("register", "unroll", "ilp", "rolling_window", "blocked")),
    ("Memory access / coalescing / vectorization", ("coalesc", "vector", "alignment", "gather", "stream", "contiguous", "transpose", "layout", "permute", "stride", "index_math", "broadcast")),
    ("Caching", ("cache", "l2_", "l1_", "rasteriz", "texture", "read_only", "restrict")),
    ("Parallelism / launch config", ("split_k", "stream_k", "persistent", "coarsening", "launch", "occupancy", "sm_fill", "grid", "chunk", "wave", "block_per", "one_block", "thread", "parallel")),
    ("Reductions / warp primitives", ("reduction", "shuffle", "warp", "reduce", "scan")),
    ("Math / precision", ("half", "fp16", "fma", "mad_", "packed", "tanh", "minmax", "intrinsic", "math", "fast")),
    ("Algorithmic", ("algorithm", "winograd", "flash", "analytic", "recurrent", "im2col", "gram")),
]
def family(name: str) -> str:
    n = (name or "").lower()
    for fam, keys in RULES:
        if any(k in n for k in keys): return fam
    return "Other"
