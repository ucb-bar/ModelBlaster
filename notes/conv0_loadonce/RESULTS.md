# Load-once conv0 kernel — measured results (riskybird-35)

Bit-exact, SW-only (no bitstream), on the flashed At35 core
(`RocketArty200TDroneGemminiSaturnFp16At35Config`, mesh16, Q0.31 acc-scale).

## What it does
`kernels/gemmini_q31_rvv/conv2d_pool_loadonce.h` — `mb_conv2d_pool_loadonce_s8()`.
The IC=1 grayscale conv0 (112²→56², OC=32, 3×3 s2, fused 3×3-s2 maxpool→27²) was
**LOAD-DMA-bound**: `loop_conv_ws`/`tiled_conv` re-mvins each output tile's input
band, so the 12.5 KB input is re-fetched ~13× (overlapping 3×3-s2 halos). The whole
padded input (114×114×1 ≈ 13K scratchpad rows) + the 3×3×1×32 weights fit the 256 KB
scratchpad, so the kernel mvins the input + weights **once**, leaves them resident,
and loops accumulator-bounded output bands doing bias-mvin → preload/compute
(reading A straight out of the resident global input, never re-mvin'd) → HW pooled
mvout (config_st pool tail, exactly as `LoopConvSt`). Pure DMA reorder, no math change.

First-layer pixel packing (`max_pixels_per_row = DIM/IC clamped to K = 3`) folds the
3 kernel columns into one compute; the input mvin is configured with the matching
`pixel_repeats` so the packed read lines up (required for correctness — with
`pixel_repeats=1` on the load the packed compute mis-reads, `max_abs_err=6`).

## Measured (gemmini_ubench_lo, real conv0 shape + params, hard Gemmini counters)

conv0 params: IC=1 IH=IW=112 OC=32 K=3 S=2 P=1, pool 3×3 s2 p0 → 27×27×32,
mult=1118622017 shift=7 clamp[-128,127] (scale_q31=8739235) — verbatim from
`generated_gray_fusedpool/hetero_tiled/model.c`.

| path | wall (cyc) | EX | LD | ST | notes |
|---|---:|---:|---:|---:|---|
| A = `tiled_conv_auto`+pool (current fused conv0) | 998,000 | 64 | 909,731 | 554 | baseline |
| **E = load-once, fence-free WDMA poll, NULL yield** | **451,843** | 34,319 | 390,498 | 21,109 | timeouts=0 |
| Y = load-once, poll + yield hook (preemptible) | 455,243 | — | — | — | yield_calls=524, timeouts=0 |

Bit-exact gate: **errAE = 0** (E vs A, identical), errE = 0, errA = 0, **errY = 0,
errYA = 0** (yield path identical too), vs a scalar conv+requant+maxpool golden.
pooled = 23,328 int8 outputs. (Earlier fenced version measured 456,145; the
targeted WDMA poll is a hair faster than a full-array fence.)

## Fence-free + preemptible (FC co-residency reuse)
Between output tiles the kernel does NOT `gemmini_fence()`; it gates accumulator
reuse on a `WDMA_BYTES_SENT` poll (counter slot 4, a k_COUNTER ROCC read that
commits immediately and does not drain), with an **optional `yield_fn` hook**
called between reads (NULL ⇒ spin = max-throughput DroNet path; `k_yield` ⇒
preemptible FC path). Per-tile store bytes are calibrated from ONE measured fence
per tile size (full band + short final band — `OC*OH*OW` logical bytes don't map
1:1 to the HW counter), then subsequent same-size tiles poll the running expected
total; a ~50 ms budget guards against a stuck poll (`timeouts` counts breaches).
This is the same fence-vs-poll mechanism as a41c8c06's `gwork.c` `GOP_TILES_POLL`,
so the load-once conv0 doubles as the preemptible poll-conv the FC co-residency
(accd7b08) needs — killing the load-DMA bottleneck AND giving preemptibility in
one kernel. Measured: **yield hook fires (yield_calls=524), 0 timeouts, bit-exact.**

## Reading it
- conv0 **1,000K → 456K cyc, bit-exact** (2.19×), inside the 300–500K target.
- **LD collapses 908K → 390K** — the ~518K removed is exactly the eliminated input
  reload. The residual 390K is the single input load (~60K) + the bias acc-mvin
  (~330K, replicated to every output cell), which is inherent to the WS
  bias-accumulate dataflow and is present identically in path A (there it hides
  under the reload). It does not overlap here (ALL3=0), so it sets the floor; a
  double-buffered acc / async dispatch could hide it further (future work).
- EX 113K→35K came from the pixel packing; ST is the fused pooled mvout.

## E2E projection
Per the throughput model (`notes/conv0_fit_and_throughput_model.md`,
`conv0_reducibility_assessment.md`): conv0 500K → 16.4 fps, 300K → 18.1 fps; at
456K the DroNet serial makespan ≈ 2,628K − (997K − 456K) ≈ 2,087K → **≈ 16.8 fps**
(from 13.31), latency-only, no pipeline, no accuracy change.

## Reproduce
`notes/conv0_loadonce/build_and_measure.sh` (off-bench build + guarded on-bench
JTAG-load of the isolated `gemmini_ubench_lo`; SRAM load, no reflash).
Validation source: `notes/conv0_loadonce/ubench_bench.c`.
Note: the flashed binary FTDI currently enumerates **0403:6011 at USB 3-6**, so use
a 6011 openocd cfg (`ftdi vid_pid 0x0403 0x6011`), `FTDI_LOCATION=3-6`, console
`/dev/ttyUSB1 @115200`.
