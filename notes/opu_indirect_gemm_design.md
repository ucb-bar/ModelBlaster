# OPU indirect-GEMM conv2d — design note

How to map conv2d onto the Saturn OPU's outer-product MAC engine, using
XNNPACK's indirection-buffer trick to avoid materialising an im2col
matrix.

## The mismatch

The OPU's natural operation is `m1[r, c] += vs1[r] * vs2[c]` over a
square mlmax × mlmax tile. Matmul fits this directly. Conv2d doesn't —
its inputs are spread over a 4D tensor `[N, IC, IH, IW]` and the
"reduction dim" is `KH*KW*IC` taps drawn from different spatial
positions per output pixel.

The naïve fix is **im2col**: lower conv → matmul by physically
gathering all the `KH*KW*IC` elements per output pixel into a flat
matrix `[OH*OW, KH*KW*IC]`. The matrix is then handed to the OPU MAC.
Problem: for a 3×3 conv this duplicates each input element ~9× in the
im2col matrix, and the matrix is huge: `OH*OW*KH*KW*IC` bytes. For
dronet's first layer at IH=112, that's ~3 MB of im2col per call.

**Indirection** (XNNPACK's contribution) sidesteps the duplication by
keeping the input tensor in its native layout and using a small table
of pointers to address it: one pointer per `(output_pixel, kernel_tap)`
gives the start of an `IC`-byte slice. Total table size is
`OH*OW*KH*KW` pointers × 8 bytes = the indirection-table cost, with
zero data duplication.

For dronet's first layer: 56*56*9 = 28224 pointers × 8 bytes = 226 KB.
For a 3×3 conv with smaller spatial dim (say 16×16): 16*16*9 = 2304
pointers × 8 bytes = 18 KB. Much more manageable than 3 MB of im2col.

## Compile-time vs runtime indirection — answer to the obvious question

XNNPACK builds the indirection at runtime because it's a runtime
library that doesn't know conv params until inference. **In our codegen
flow we already know everything** — conv params from the IR, buffer
addresses from the buffer-allocation pass. The indirection table is
fully deterministic before the binary even runs.

So yes — building it at compile time IS the right call when feasible.
Two encodings work:

### Option A — pointer table (8 bytes per entry)

```c
static const int8_t *const __indir_conv2d_3[/*OH*OW*KH*KW*/] = {
    &buffer_5[((0 * IH + 0) * IW + 0) * IC],   /* (oh=0,ow=0,kh=0,kw=0) */
    &buffer_5[((0 * IH + 0) * IW + 1) * IC],   /* (oh=0,ow=0,kh=0,kw=1) */
    ...
    __opu_zero_buf,                             /* padding entry */
    ...
};
```

GCC accepts `&array[N]` as a constant-address static initializer (the
linker resolves `buffer_5`'s base, the offsets are constants). Padding
entries point at a small static all-zero `int8_t[IC_max]` buffer.

### Option B — offset table (4 bytes per entry)

```c
#define INDIR_PAD  0xFFFFFFFF
static const uint32_t __indir_conv2d_3[] = {
    ((0 * IH + 0) * IW + 0) * IC,
    ((0 * IH + 0) * IW + 1) * IC,
    ...
    INDIR_PAD,
    ...
};

/* kernel: const int8_t *p = (off == INDIR_PAD) ? __opu_zero_buf : input + off; */
```

Half the flash footprint but adds a branch per entry in the hot loop —
on the OPU outer-product pattern with vectorized loads, that branch is
hard to vectorize and likely costs more than the flash savings buy.

### Decision

Option A (pointer table) is cleaner and faster. Flash cost for dronet
across its 10 conv layers: ~2 MB total of indirection tables, fitting
in the DDR-backed `.rodata` region. For embedded SoCs with limited
flash (the KU040 scratchpad-only build is 256 KB SRAM), generation
could be gated by a `CONFIG_OPU_INDIR_RUNTIME` Kconfig fallback — or
the per-tile runtime build (Option C below) used instead.

### Option C — per-tile runtime build (the v1 in this PR)

The compile-time table is the right answer architecturally, but it
requires invasive skeleton changes: every conv2d_s8 dispatch needs
its own `static const int8_t *const __indir_conv2d_N[]` and the kernel
signature must gain an extra arg to receive it. That's a real change
to the modelblaster-flow `KernelSpec.signature` convention (all conv2d_s8
implementations must accept the same signature).

**Sidestep**: build the indirection **per-tile at runtime, never
whole-layer**. The kernel processes `OW_BLK = mlmax` output pixels at
a time. Per-tile indirection is just `KH*KW * OW_BLK` pointers — for a
5×5 conv with mlmax=64 that's 25*64 = 1600 pointers = 12.8 KB max,
easily stack-allocated. Build cost is `O(KH*KW * OW_BLK)` per tile;
across the whole conv it's the same total work as one whole-layer
build, just amortized into the per-tile loop.

This is the v1 we'll ship. The compile-time path (Option A) is the
clear follow-up once we add per-algorithm signature support to the
skeleton — track in `notes/saturn_opu_backend.md` under "pending".

## Kernel structure (Option C — v1)

```
input  [N, IC, IH, IW] int8   ─┐
weight [OC, IC, KH, KW] int8  ─┤
bias   [OC] int32             ─┤
output [N, OC, OH, OW] int8   ─┘
```

Outer loop: tile over `(n, oh, ow_tile)`, where `ow_tile` advances by
`OW_BLK = mlmax`. Output channel dimension is the OPU MAC's c-axis
(`OC_BLK = mlmax`); input pixels are the r-axis.

```c
for n in N:
  for oh in OH:
    for ow_tile in OW step OW_BLK:
        ow_blk = min(OW - ow_tile, OW_BLK)

        /* Build indir [KH*KW][ow_blk] — pointers to IC slices */
        for kh in KH:
            ih = oh * SH - PH + kh
            if ih not in [0, IH): mark padding
            for kw in KW:
                iw_base = ow_tile * SW - PW + kw
                for p in 0..ow_blk:
                    iw = iw_base + p * SW
                    if (ih, iw) in bounds:
                        indir[(kh*KW+kw)*OW_BLK + p] =
                            &input[((n*IC)*IH + ih) * IW + iw]
                    else:
                        indir[(kh*KW+kw)*OW_BLK + p] = zero_buf

        /* OPU MAC for this output tile */
        for oc_tile in OC step OC_BLK:
            oc_blk = min(OC - oc_tile, OC_BLK)

            OPMVINBCAST m1 <- bias[oc_tile..+oc_blk]
                                 (or 0 if NULL, padded to mlmax)

            for k_idx in [0, KH*KW):
                for ic in [0, IC):
                    /* vs1: ow_blk i8 elements, one per output pixel.
                     * Each lane reads `indir[k_idx*OW_BLK + p][ic]` */
                    vs1 = vluxei8 (indir_table_for_k_idx, +ic)

                    /* vs2: oc_blk i8 weights at (oc, ic, kh=k_idx/KW, kw=k_idx%KW) */
                    vs2 = vlse8 (weight + ..., stride = IC*KH*KW)

                    VOPACC m1, vs2, vs1

            /* drain rows of m1 (one per output pixel),
             * Q0.31 requantize, i8 store with OC stride */
```

### Why `vluxei` works for vs1

XNNPACK's table is a list of pointers, but for OPU we want `OW_BLK`
bytes (one per output pixel) gathered from `OW_BLK` different
locations + offset `ic`. RVV's indexed load handles this in one
instruction:

```
vluxei64.v vs1, (base), vindex
  for lane p in 0..vl:
    vs1[p] = mem[base + vindex[p]]
```

Set `base = 0`, `vindex = indir[k_idx*OW_BLK..]`-as-i64-offsets +
`ic`. One instruction, gathers `OW_BLK` bytes from `OW_BLK` pointers.

For OPU's `mlmax=16..64`, the index vector is just `mlmax × 8 bytes`
= 128..512 bytes. Fits in vector registers (LMUL=8 at e64 holds 64
elements on V512).

### Padding cost

Padded entries point at `__opu_zero_buf` — a `static int8_t
zero_buf[IC_max]` allocated in `.bss`. The vluxei8 gathers `0`s
from these addresses, and the VOPACC adds zero contribution to m1.
No branching in the hot loop.

`IC_max` is sized at codegen time as the largest IC across all
conv2d_s8 layers in the model (typically 32..1280). One-time
allocation per model.

### Numerics

The OPU MAC produces i32 accumulator equal to the scalar reference
sum (modulo lane summation order which doesn't matter for integer
ops). Q0.31 requantize tail matches the existing scalar reference
formula exactly. Bit-exact result vs. the reference impl.

## Skeleton-side requirements (v1)

Minimal — the curated kernel handles everything internally:

1. `__opu_zero_buf` in the existing `buffers.c` infrastructure, sized
   `IC_max`. The skeleton emits this when it sees `rvv_opu` as the
   target and any conv2d_s8 op in the IR.
2. No signature change. The kernel signature stays the standard
   `kernel_conv2d_s8(...)`.

That's it. The kernel's `AlgorithmCandidate(name="indir_gemm",
target_affinity=("rvv_opu",))` registers it in `CONV2D_S8.algorithms`.

## Future work (compile-time path)

Once we want to eliminate the per-tile runtime indirection-build cost:

1. **Extend `KernelSpec`** to support per-algorithm signatures
   (currently one signature per op). The `indir_gemm` algorithm
   declares a different C signature with an `int8_t *const *indir`
   argument.

2. **Skeleton emits per-layer indirection tables** at codegen time,
   keyed by `<model>_<op_name>_indir`. Each is a static const array
   of pointers / offsets initialized from the conv params + input
   buffer's known address (from the buffer-allocation pass).

3. **Walker dispatches** the indir-arg variant of the kernel when the
   `indir_gemm` algorithm is chosen for that conv op.

Estimated cost: ~3 days for the per-algorithm-signature plumbing, +
~1 day for the skeleton emission. Defer until we see actual perf
data showing the per-tile runtime build matters (current expectation:
the build cost is ~50 cycles per tile of 16 output pixels, vs. the
OPU MAC at ~K*16 cycles per tile — so ~3% overhead for 3×3 IC=32,
not worth optimizing yet).

## Edge cases / limitations

- **OW_BLK > mlmax not supported in v1.** Keeps the indirection table
  bounded. Limits the perf ceiling on wide OW; can extend later by
  running multiple OPU tiles per (oh, ow_tile) iteration.
- **Strided convs with stride > KW**: the indirection step pattern
  is uniform `p * SW`, no special handling needed.
- **Dilated convs**: same as strided; just `kh*dilation_h` in the
  indirection-table build. Modelblaster-flow IR doesn't emit dilation > 1
  for conv2d_s8 today; v1 punts dilation > 1 to scalar fallback.
- **Asymmetric quant**: same constraint as the matmul/linear OPU
  kernels — `input_offset != 0` or `filter_offset != 0` falls back
  to scalar reference (the OPU MAC computes raw sum, no offset
  correction).
- **Per-channel weight scale** (`conv2d_s8_pc`): separate op-kind
  with its own signature. The same indirect-GEMM pattern applies;
  port a `rvv_opu_conv2d_s8_pc_indir_gemm.c` variant once the
  `conv2d_s8` one is validated.

## Worked numbers

For yolov8_nano's 3×3 conv2d_s8 with IC=64, OC=128, IH=IW=32, stride=1:
- output pixels per tile = mlmax = 16 (on V128 spike)
- pre-indirection: 1 spike call per output tile builds 9*16=144 pointers
  (~50 cycles)
- OPU MAC: K = IC*KH*KW = 64*9 = 576 VOPACCs per tile
- Per tile: ~600 cycles total (build dominated by MAC)
- Total tiles: OH*OW/16 = 64. Layer cycles: ~38K
- Compare to scalar reference (estimate from earlier RVV widening
  measurements): ~250K cycles for the same shape → ~6.5× speedup
  before any cache effects.

On FireSim with cache modeled, the win should be larger because the
weight tensor (64*128*9 = 72 KB) fits in L2 and is reused across all
output pixels — that's the whole point of indirect GEMM.

## References

- XNNPACK indirection design:
  https://github.com/google/XNNPACK/blob/master/src/conv-hwc-indirection.c
- OPU programming model: `modelblaster/cores/saturn_opu/include/saturn_opu.h`
- This kernel: `modelblaster/kernels/rvv_opu/rvv_opu_conv2d_s8_indir_gemm.c`
- RVV indexed load semantics: V spec §7.5 (vluxei*)
