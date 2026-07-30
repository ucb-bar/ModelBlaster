# Why XNNPACK's int8 conv is ~3.5× faster than ModelBlaster's on RVV

Root-cause analysis of the DroNet int8 convolution gap measured on FireSim
(real RTL, dual-Rocket Saturn V256D128 RVV, 1 core). Companion to
[`flow_comparison.md`](flow_comparison.md) and the plot
`plots/dronet_firesim_compare.png`.

## TL;DR

Same hardware, same ISA, same core instruction — the gap is **kernel
structure**, not silicon.

| conv kernel (DroNet, 13.0M MACs) | cycles | **MAC/cycle** |
|---|---:|---:|
| ModelBlaster `rvv_conv2d_s8_rvv_vsmul_vnclip` (curated, generic) | 14,079,285 | **0.92** |
| XNNPACK `qs8-qc8w-igemm-7x4v` (int8 IGEMM) | 4,039,774 | **3.22** |
| ratio | | **3.49×** |

Both kernels issue the *identical* core op — `vwmacc.vx` (vector-scalar
widening int8→int32 MAC) — so per-instruction MAC width is equal. The gap is
purely **how many useful MAC-instructions issue per cycle** = data reuse +
instruction mix. XNNPACK is **not** using a wider/different ALU, a better
requantize, or a special ISA feature.

### Update: weight-buffer layout (HWIO ↔ IHWO) is perf-neutral

After merging upstream (which repacks conv weights `ihwoc`=IHWO, commit
`15a3b17`), the curated kernels were re-aligned to read IHWO
(`((ic*KH+kh)*KW+kw)*OC+oc`) instead of HWIO. Re-measured on FireSim (bit-exact):

| MB conv (DroNet) | total cycles | MAC/cycle | gap vs ET |
|---|---:|---:|---:|
| HWIO (old) | 14,079,285 | 0.92 | 3.49× |
| **IHWO (new)** | **14,472,123** | **0.90** | **3.58×** |

The weight-buffer layout barely moves the needle (IHWO is ~3% *slower* here, per
conv) — **it is not a major factor.** In both layouts OC is the contiguous
vector axis (identical `vle8` loads) and the kernel still streams the *entire*
weight buffer once per output pixel with no reuse; only the block order differs,
which the Saturn memory system handles about the same. The ~3.5× gap is
unchanged, confirming it comes from the structural causes below (register
tiling/reuse, NHWC-vs-NCHW *activation* layout, pre-packing) — **not** the weight
layout. (Note: the "layout" that does matter is the NHWC **activation** layout —
cause #2 — which forces MB's scalar strided output store; that is orthogonal to
the weight-buffer packing changed here.)

## The three structural causes (ranked)

### 1. Register tiling / weight reuse (dominant)

**ModelBlaster** (`modelblaster/kernels/rvv/rvv_conv2d_s8_rvv_vsmul_vnclip.c`)
is a *direct* conv: `for n → oh → ow → oc_base`, and for **each single output
pixel** it re-runs the whole `IC×KH×KW` reduction. Per reduction step it emits
three vector ops for one MAC group:

```c
vint8m1_t  vw8  = vle8(wp, vl);                 // reload weights every pixel
vint16m2_t vw16 = vwadd_vx(vw8, filter_offset); // re-fold offset (redundant)
vacc            = vwmacc_vx(vacc, in_v, vw16);  // 1 useful MAC-instr
```

⇒ **~1 MAC-instruction per 3 vector instructions**, and with no output-pixel
tiling the *same weights are streamed from memory and re-offset `OH×OW` times*.

**XNNPACK** (`.../qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4v-minmax-fp32-rvv.c`,
MR=7 variant selected at runtime) is implicit GEMM with an **MR×NR register
tile** (MR=7 rows, NR=32 channels). One weight-vector load feeds all MR rows
from registers:

```c
const vint8m1_t  vb  = vle8(w, vl);         // ONE weight load ...
const vint16m2_t vb0 = vsext_vf2(vb);
vacc0 = vwmacc_vx(vacc0, va0, vb0);         // ... reused across
vacc1 = vwmacc_vx(vacc1, va1, vb0);         //     MR=7 output rows
...                                         //     → 7 MACs / load
```

⇒ **4–7 MAC-instructions per weight load**, weights loaded once per reduction
tap. This ~1:3 vs ~7:1 MAC-to-overhead ratio, plus killing the `OH×OW`
redundant weight re-streaming, is most of the 3.5×.

### 2. NCHW (MB) vs NHWC (XNNPACK) — layout forces a scalar store

MB vectorizes over output channels, but in **NCHW** channels are strided by
`OH×OW`, so it cannot vector-store — it stages to a temp and does a **scalar
scatter** (`rvv_conv2d_s8_rvv_vsmul_vnclip.c`):

```c
int8_t _obuf[256]; vse8(_obuf, vout8, vl);
for (_vi=0..vl) op[_vi * (OH*OW)] = _obuf[_vi];   // vl SCALAR strided stores / pixel
```

XNNPACK's **NHWC** keeps channels contiguous → single contiguous vector store,
and the reduction streams contiguous channel vectors. Per-layer MB efficiency
confirms the penalty scales with spatial size and inversely with reduction
depth (the fixed per-pixel overhead isn't amortized when the reduction is
shallow):

| conv | MB MAC/cyc | why |
|---|---:|---|
| conv0 (112², IC=3, 3×3) | 0.71 | large spatial → ~100k scalar stores; shallow IC=3 |
| 1×1 convs (conv3/6/9) | 0.49–0.54 | tiny `IC×1×1` reduction → per-pixel overhead dominates |
| deep 3×3 (IC≥32) | 1.0–1.1 | deep reduction amortizes the fixed overhead best |

XNNPACK's GEMM tiling amortizes this uniformly, so it has no such dip.

### 3. Weight/offset/bias pre-packing

XNNPACK folds bias (the leading `vle32` accumulator seed), filter zero-point,
and per-channel scale into a **packed weight blob once at setup**. MB carries
raw weights and re-derives the offset (`vwadd`) every inner iteration of every
pixel. (Requantize tails are comparable — MB integer `vsmul`+`vnclip` vs
XNNPACK `fp32` convert/scale, both one vector pass per output tile — so
requant is **not** a cause.)

## What would close the gap

ModelBlaster's conv would need the XNNPACK structure:
1. **im2col / indirection-buffer IGEMM** formulation.
2. **MR×NR register tile** so each loaded weight feeds MR output rows.
3. **Weight pre-packing** (fold zero-point + bias once, not per pixel).
4. **Contiguous-store output** — NHWC, or an OC-blocked NCHW tile that stores
   vectors instead of a scalar scatter.

This is exactly what the `BACKEND=llm` per-op path is meant to synthesize — a
targeted LLM run on `conv2d_s8` (with firesim-eval selection) is the way to see
how close a tuned ModelBlaster conv can get. The kernel measured here is the
**generic curated** `rvv_vsmul_vnclip` (`source: curated`, not `llm`), a single
shape-generic kernel used for all 10 conv layers.

## Method note

MB per-op cycles: clean in-binary rdcycle from
`modelblaster/examples/dronet/int8/generated/profile.csv` (bit-exact PASS).
ET per-op cycles: XNNPACK-profiling FireSim run (`MB_XNN_PROFILE=ON`); the
`>>` lines are clean per-op rdcycle deltas (that build's *total* is
HTIF-logging-polluted — full-model totals come from the profiling-off run).
MAC counts = `OC·OH·OW·IC·KH·KW` per layer. Regenerate the plot/table via
`modelblaster/scripts/plot_dronet_firesim_compare.py`.
