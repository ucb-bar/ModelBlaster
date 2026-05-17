# TCM-aware codegen — design note

How to extend the modelblaster flow so kernels and codegen can take advantage
of **tightly-coupled SW-managed memories** (TCMs / scratchpads) when
the HW config exposes them.

Today's flow assumes a single flat (cached) memory hierarchy. The
existing `MemoryModel` dataclass in
`modelblaster/optimize/firesim_eval/cache_aware_prompt.py` describes cached
L1/L2/DRAM and feeds the optimize-phase LLM prompt. That model is fine
for FireSim quad-rocket-saturn, but breaks down for configs like:

- **Shuttle + TCM** (`OPUV128D64DualShuttleConfig` in saturn's
  `chipyard/OPUConfigs.scala` — wires `WithTCM(size = 128 << 10)`).
- **KU040 scratchpad-only SoC** (no DDR; entire working set in BRAM-
  backed SRAM — see `notes/ku040_bitstream_plan.md`).
- **Gemmini-style accelerator** (already has its own scratchpad + acc,
  but the modelblaster flow currently routes it via the vendored `gemmini.h`
  tiled-conv API, not as a generic TCM).

The point of this note: design the *generic* TCM-awareness layer that
sits beside (not inside) the gemmini path, and that future configs can
plug into without breaking existing flows.

## Why TCMs change the codegen model

A cached hierarchy is opaque to the kernel author: you write the
clearest loop and the cache hides the cost. A TCM is *visible*:
addresses below TCM_BASE+TCM_SIZE are 1-cycle SRAM, addresses outside
are DRAM (or main memory). The kernel must:

1. Decide what to put where.
2. Issue explicit DMA (or load/store) to move data in/out.
3. Synchronize so the data is in TCM by the time it's used.

The win is large: a 128 KB TCM at 1 cycle is ~50× faster per access
than a 30-cycle DRAM round trip. For a kernel where 100 MB of weight
streams in once and gets reused over many output pixels, staging
through TCM is the difference between a memory-bound kernel and a
compute-bound one.

The modelblaster flow has two kinds of memory consumers that need TCM
visibility independently:

- **Curated / LLM-generated kernels** (compute-side reuse): the kernel
  body issues DMAs for its working set and reads from TCM.
- **The skeleton** (cross-kernel reuse): for a sequence of kernels
  that touch the same weight tensor, the skeleton can stage that
  tensor into TCM once and have every kernel use it from there.

Both need the backend to *declare* TCM presence + size; only the first
needs DMA macros in kernel source; only the second needs a scratchpad
allocator. Treat as two independent slices.

## Layer 1: backend declaration

Extend `modelblaster/pipeline/backends.py::Backend` with an optional `tcm`
field describing the scratchpad:

```python
@dataclass(frozen=True)
class TCMSpec:
    name: str            # e.g. "shuttle_tcm", "ku040_sram"
    base_addr: int       # physical address
    size_bytes: int
    line_bytes: int      # natural DMA / fetch granularity
    access_cycles: int   # core-cycle cost per access (typically 1-3)
    # DMA engine cost model (for dynamic prefetch decisions):
    #   dma_setup_cycles + ceil(bytes / dma_bytes_per_cycle)
    dma_setup_cycles: int = 0
    dma_bytes_per_cycle: int = 8

@dataclass(frozen=True)
class Backend:
    ...
    tcm: Optional[TCMSpec] = None
```

`tcm=None` means "no TCM, fall back to flat memory" — existing backends
keep working unchanged. Backends like `shuttle_tcm` set `tcm=TCMSpec(...)`.

Concrete examples we'd add:

```python
SHUTTLE_TCM = TCMSpec(
    name="shuttle_tcm",
    base_addr=0x60000000,
    size_bytes=128 << 10,    # WithTCM(size=128<<10) in OPUConfigs.scala
    line_bytes=64,
    access_cycles=1,
    dma_setup_cycles=4,
    dma_bytes_per_cycle=16,
)

KU040_SRAM = TCMSpec(
    name="ku040_sram",
    base_addr=0x70020000,
    size_bytes=256 << 10,
    line_bytes=64,
    access_cycles=1,
    dma_setup_cycles=0,         # CPU-issued lw/sw — no DMA
    dma_bytes_per_cycle=8,
)
```

Plumbing: at codegen time the backend's `tcm` is propagated through
`generate_skeleton.py` and `generate_kernels.py` as:

1. A C macro `-DMODELBLASTER_TCM_BASE=0x...`, `-DMODELBLASTER_TCM_SIZE=...`, etc.
   exported via `Backend.kernel_cflags`. Curated kernels can `#ifdef
   MODELBLASTER_TCM_BASE` to gate their TCM-aware paths.
2. A Python-visible field that `generate_skeleton.py` reads when
   deciding whether to issue prefetch directives.

## Layer 2: linker-side TCM region

Zephyr already supports custom memory regions through the device tree
and linker script. Adding a TCM region is mechanical for a board with
a `chosen { zephyr,sram-tcm = &tcm; }` style entry.

For each backend with `tcm` set, the harness needs:

1. **A devicetree fragment** at `modelblaster/harness/backends/<name>.overlay`
   describing the TCM region with the correct base+size.
2. **A linker symbol** `__tcm_start`, `__tcm_end` carved out by the
   linker so C code can refer to the region.
3. **A simple allocator** in the harness — `tcm_alloc(n)` /
   `tcm_free(p)` — backing the `__tcm_start..__tcm_end` arena. Could
   just be a bump pointer reset between dispatches.

Existing precedent: gemmini's tiled scratchpad allocator is handled
inside `gemmini.h` (private). The TCM allocator should mirror that
pattern but live in `modelblaster/harness/src/tcm_allocator.c`, shared
across kernels.

## Layer 3: TCM-aware kernel macros

Curated kernels gate on the `MODELBLASTER_TCM_BASE` define. The convention
for kernel sources:

```c
#include "saturn_opu.h"
#ifdef MODELBLASTER_TCM_BASE
  #include "tcm_allocator.h"
#endif

void kernel_linear_s8(...) {
#ifdef MODELBLASTER_TCM_BASE
    /* Stage the per-call weight slice into TCM. */
    int8_t *w_tcm = (int8_t *)tcm_alloc(N * K * sizeof(int8_t));
    if (w_tcm) {
        memcpy_dma(w_tcm, weight, N * K);
        weight = w_tcm;
    }
    /* (Fall through to the regular MAC loop — it reads from `weight`
     * which is now in TCM.) */
#endif
    ...
    /* MAC body unchanged */
    ...
#ifdef MODELBLASTER_TCM_BASE
    if (w_tcm) tcm_free(w_tcm);
#endif
}
```

`memcpy_dma()` is a generic wrapper that compiles to:

- Plain `memcpy` on backends without DMA engines (CPU-issued copy).
- A DMA descriptor + wait on configs with one (e.g. saturn vmu, or
  Hwacha's vmh).
- `tiled_*` calls on gemmini-style configs (but the gemmini backend
  already does this through its own API; TCM-aware path is for *new*
  generic accelerators).

For curated kernel authors the rule is: if your working set is bigger
than L1 *and* TCM is large enough to fit it, prefer staging through
TCM. The compile-time `#ifdef MODELBLASTER_TCM_BASE` keeps the same source
file working on TCM-less backends.

## Layer 4: skeleton-side scratchpad allocator (cross-kernel)

The bigger lever is the cross-kernel allocator. For a kernel sequence:

```
dispatch_0: conv2d(input, weight_0, output_0)
dispatch_1: conv2d(output_0, weight_1, output_1)
dispatch_2: conv2d(output_1, weight_2, output_2)
...
```

`weight_0..N` are static; if they fit in TCM together, staging them
once at startup saves DMA per dispatch. The skeleton can decide this
at codegen time.

### Allocator interface

```python
class TCMAllocator:
    """Decide which buffers live in TCM at codegen time.

    Inputs:
      - The IR's `tensors` block (which tensors exist, sizes, dtype).
      - The dispatch list (which tensors each dispatch reads/writes).
      - The backend's TCM size.

    Output: a placement decision per tensor —
      {"tensor_name": ("tcm", offset_bytes) or ("dram", None)}
    """
    def __init__(self, ir, backend): ...
    def place_static(self) -> dict[str, tuple[str, int | None]]:
        """Compute the static placement at codegen time."""
        ...
```

The decision shape is "weight live in TCM" vs "weight live in DRAM".
A greedy heuristic ordered by `bytes_accessed * reuse_count` covers
the high-value cases without an ILP — start there; promote to ILP if
profiling reveals a useful gap.

### Generated C consequences

Once `TCMAllocator.place_static()` returns, `generate_skeleton.py`
emits:

1. A TCM region in `buffers.c` with the placed weights:
   ```c
   __attribute__((section(".tcm_data")))
   static const int8_t TCM_weight_0[...] = { ... };
   ```
2. Per-dispatch pointer fixups: in `model.c`, kernels that read a
   TCM-placed weight receive the TCM address instead of the DRAM one.

Existing code paths that already exist as plumbing surfaces:

- `modelblaster/pipeline/generate_skeleton.py::_backend_pack_weight` is the
  one place that decides weight tensor layout per-backend — extend
  alongside it to decide weight tensor *placement* per-backend.
- `modelblaster/harness/src/main.c` already understands extern-shared
  buffers across the multi-net flow; extending to TCM-section
  externs is a one-line addition.

## Layer 5: profile-driven TCM decisions

The greedy "place largest-and-most-reused tensors in TCM first"
heuristic is fine for a v1, but better placement comes from looking
at the same profile.csv we already write:

```
dispatch_id, name, op, shape, cycles
0, conv2d, conv2d_s8, ..., 12000
1, batch_norm, batchnorm2d_s8, ..., 300
...
```

A second pass after the first profile run knows actual cycle costs.
Combining with per-tensor reuse counts (extracted from the dispatch
list at codegen time), the allocator can solve a knapsack:

> Subject to total TCM bytes <= TCM_SIZE, maximize total cycles saved.

Where "cycles saved" per tensor is `bytes * reuse_count * (latency_dram - latency_tcm) / line_bytes`.

This profile-driven loop reuses the same `--firesim-eval` re-rank
infrastructure: profile once on flat memory, decide TCM placement,
regenerate skeleton, profile again, compare.

## Multi-core / XPURT integration

Today's XPURT scheduling assigns each entry to a core_kind (CPU_P,
CPU_E, GEMMINI). Some core_kinds may have their own TCM. Extension:

1. `modelblaster/cores/<config>.json` core-registry entries gain an
   optional `"tcm": {...}` block matching the `TCMSpec` Python shape:
   ```json
   {
     "core_kinds": {
       "SHUTTLE_TCM": {
         "backend": "shuttle_opu",
         "harts": [2, 3],
         "tcm": {"base": "0x60000000", "size": 131072, ...}
       },
       "CPU_E": {"backend": "scalar", "harts": [0, 1]}
     }
   }
   ```
2. `modelblaster/pipeline/ingest_xpurt_schedule.py` plumbs the per-core
   TCM spec into the per-entry codegen context. Kernel calls
   targeting a SHUTTLE_TCM-kind entry get `weight_ptr = tcm_ptr_X`,
   while kernel calls on CPU_E get the DRAM pointer.

3. The harness's per-core worker thread does an initial-stage DMA of
   the TCM-placed weights into its bank before entering the dispatch
   loop. Synchronized via `k_sem` chain like other startup ordering.

This way the XPURT scheduler can take TCM presence into account
*indirectly* — by costing dispatches differently on TCM-equipped vs
TCM-less cores in the profile data (already provided by per-core
results.csv files), the existing MILP picks the lower-cost core
automatically. No scheduler-side change needed.

## Worked example: dronet on Shuttle+TCM

Dronet has 10 conv2d_s8 + 2 linear_s8. Weight tensors (in bytes,
i8 weights):

```
conv1: 3*32*5*5 = 2400
conv2-9: ~200K total
linear1, linear2: 2*2048 = 4 KB
```

Total weight footprint ≈ 200 KB. A 128 KB Shuttle TCM can hold ~half.
Greedy by reuse count:

- linear1, linear2 weights (called once each) → low priority for TCM.
- conv2..conv9 weights, each touched once → equal priority.
- conv1 weight is small (2.4 KB) and high-reuse (sweeps the whole
  112×112 input) → high priority for TCM.

The allocator would place conv1's weights + the largest 1-2 mid-layer
conv weights into TCM, leaving the rest in DRAM. Profile-driven
refinement then picks the actual cycle-saving set after one pass.

## Implementation phases

Phase 1 (~1 day): backend-side declaration only.
- `TCMSpec` dataclass + `Backend.tcm` field.
- C macros plumbed to `Backend.kernel_cflags`.
- One worked-example backend (e.g. `shuttle_tcm`) declared but no
  kernels using it yet.

Phase 2 (~2 days): kernel-side macros.
- `modelblaster/cores/tcm/include/tcm_allocator.h` with `tcm_alloc /
  tcm_free / memcpy_dma` (initially `memcpy` fallback).
- One curated kernel (e.g. `rvv_linear_s8` with TCM staging) gated
  on `MODELBLASTER_TCM_BASE`.
- Verify correctness on spike (TCM region is just normal SRAM there).

Phase 3 (~3 days): skeleton-side allocator.
- `TCMAllocator.place_static()` greedy heuristic.
- `generate_skeleton.py` emits `.tcm_data` section + pointer fixups.
- Linker script + Zephyr overlay for TCM region.

Phase 4 (~2 days): profile-driven allocator.
- Read profile.csv from a flat-memory run.
- Knapsack solver over (bytes, reuse, savings).
- Loop: flat profile → place → repacked profile → compare.

Phase 5 (later): XPURT core-registry integration.
- `modelblaster/cores/<config>.json` entries carry TCM specs.
- Per-core TCM staging at harness startup.

The first two phases are independently useful (a curated kernel can
write its own TCM staging, no skeleton work needed). Defer 3+ until
we have a real config to validate against.

## Open questions

1. **TCM eviction on dispatch boundaries?** If a kernel allocates
   scratch in TCM mid-call, when does it get freed? Per-call bump
   reset (Phase 2) is simplest; cross-call retention (for hot
   intermediate tensors) needs more careful allocator state.

2. **DMA engine availability?** Saturn's VMU can do strided i32 / fp32
   loads; using it as a generic memcpy-DMA needs a wrapper. KU040 has
   no DMA — fall back to `memcpy` (CPU-issued lw/sw). Cleanest API
   is a `memcpy_dma(dst, src, n)` that selects internally.

3. **TCM vs cached on the SAME tensor?** For multi-core schedules,
   a tensor might be TCM-resident on one core and DRAM-resident on
   another (e.g. a shared weight that's hot on CPU_P but cold on
   CPU_E). Skeleton needs to emit two pointer initializations and
   the kernel call site needs to know which to pass — same shape
   as the existing per-core dispatch table, just one more field per
   tensor.

4. **Profile schema extension?** The IREE-shape `results.csv` doesn't
   carry "which buffers were in TCM" — adding it would let plot
   tooling visualize TCM hit/miss alongside cycles.

## References

- Existing cache-only model: `modelblaster/optimize/firesim_eval/cache_aware_prompt.py`
- KU040 scratchpad-only SoC plan: `modelblaster/notes/ku040_bitstream_plan.md`
- Shuttle TCM declaration: `hw/chipyard/generators/saturn/chipyard/OPUConfigs.scala` (`OPUV128D64DualShuttleConfig`)
- Gemmini's scratchpad (existing, accelerator-private model): `modelblaster/notes/gemmini_extension_plan.md`
- Multi-core core-registry format: `modelblaster/notes/dispatch_and_cores.md`
- IREE-shape profile schema: `modelblaster/notes/profile_emission.md`
