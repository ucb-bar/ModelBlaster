# Phase G — Final Comparison (v8 → v9 → v10 → v11g → v14 → v17 → v18 → v19)

Hybrid policy `hybrid_periodic_mosek_yolo`, canonical workload
4 MLP@10ms + 2 Dronet@20ms + 1 Yolo@100ms, hetero bitstream
(Gemmini RoCC + Saturn-OPU on Shuttle tiles), `FIRESIM_QUEUE=1`.

## Run table

| Run | Notes | gemmini wall | rvv_opu wall | makespan | mlp err | dronet err | yolo err |
|:---|:---|---:|---:|---:|---:|---:|---:|
| v8 | baseline (no instrumentation) | n/a | n/a | **638 ms** | 0 | 72 | 22 |
| v9 | G1 instrumentation, same code as v8 | 709 ms | **786 ms** | 786 ms | 0 | 72 | 22 |
| v10 | + RVV silu_s8 LUT kernel (G5) | 494 ms | **571 ms** | **571 ms** | 0 | 72 | 22 |
| v11g | + im2col_rvv_reduce conv2d_s8 + per-dispatch VS re-arm + producer-fanout | 454 ms | **531 ms** | **531 ms** | 0 | 72 | 22 |
| v13 | + per-channel LUT batchnorm2d_s8 (no guard) — REJECTED | 518 ms | 601 ms | 601 ms | 0 | 72 | 22 |
| v14 | + per-channel LUT batchnorm2d_s8 (spatial ≥ 256 guard) | 435 ms | **515 ms** | **515 ms** | 0 | 72 | 22 |
| v15 | + im2col_outerprod conv2d_s8 (Saturn VOPACC) — REJECTED | n/a | n/a | n/a | n/a | n/a | n/a |
| v17 | + im2col_rvv_reduce conv2d_s8 cached for **dronet** | 216 ms | **223 ms** | **223 ms** | 0 | 72 | 22 |
| v18 | + per-input-LUT cat2_c1_s8 (cat3/4 LUT verify failed) | 205 ms | **211 ms** | **211 ms** | 0 | 72 | 22 |
| v19 | + per-input-LUT cat3_c1_s8 + cat4_c1_s8 (name collision fix) | 191 ms | **198 ms** | **198 ms** | 0 | 72 | 22 |

Improvement v9 → v10: **−215 ms on makespan (−27%)**, **−223 ms
on rvv_opu kernel time (−30%)**, exactly the silu_s8 contribution
the PDB predicted (~12.1 M spike cycles vs ~120 M reference).

## Per-hart attribution (mtime µs)

```
v9   gemmini  708,799   kernel=109,080  dep_wait=599,085  sync=116  gate=448  idle=47
v9   rvv_opu  786,534   kernel=732,620  dep_wait= 53,586  sync=234  gate=  0  idle=54

v10  gemmini  493,768   kernel=109,403  dep_wait=383,728  sync=119  gate=449  idle=56
v10  rvv_opu  571,117   kernel=509,152  dep_wait= 61,646  sync=203  gate=  0  idle=72

v11g gemmini  453,983   kernel=109,080  dep_wait=344,213  sync=159  gate=456  idle=47
v11g rvv_opu  531,250   kernel=470,095  dep_wait= 60,762  sync=285  gate=  0  idle=54

v13  gemmini  517,519   kernel=108,728  dep_wait=408,120  sync=141  gate=456  idle=54
v13  rvv_opu  601,276   kernel=541,511  dep_wait= 59,368  sync=303  gate=  0  idle=45

v14  gemmini  434,994   kernel=108,852  dep_wait=325,460  sync=148  gate=456  idle=52
v14  rvv_opu  514,622   kernel=452,560  dep_wait= 61,667  sync=289  gate=  0  idle=55

v17  gemmini  216,112   kernel=109,355  dep_wait=106,074  sync=157  gate=455  idle=49
v17  rvv_opu  222,738   kernel=157,862  dep_wait= 64,478  sync=281  gate=  0  idle=64

v18  gemmini  204,704   kernel=109,448  dep_wait= 94,579  sync=144  gate=456  idle=58
v18  rvv_opu  211,348   kernel=143,962  dep_wait= 66,990  sync=285  gate=  0  idle=58

v19  gemmini  191,352   kernel=110,006  dep_wait= 80,663  sync=155  gate=457  idle=51
v19  rvv_opu  197,984   kernel=129,416  dep_wait= 68,177  sync=285  gate=  0  idle=58
```

v18 → v19: per-input-LUT cat3 + cat4 landed after fixing a static-
helper name collision (`_build_cat_lut` defined in all three LUT
kernel files would multiply-define when picker took all three —
spike-harness build failed, picker fell back to scalar reference
for cat3/cat4). Renamed each to `_build_cat{2,3,4}_lut`.

Cat totals on rvv_opu hart:
```
v17 (no cat LUTs):   34.3 M rdcycle
v18 (cat2 LUT):      20.8 M rdcycle  (-39%)
v19 (cat2+3+4 LUT):   5.7 M rdcycle  (-85% total, -73% v18→v19)
```

Per-shape v18→v19 cat3 ratios (5x5..40x40): **0.16x..0.22x**
(5-6x speedup). cat4 same range. The biggest absolute saving:
H=40 W=40 C_inputs=16|16|16 went 4.6 M → 833 k rdcycle.

makespan 211 → 198 ms (-13 ms, -6%). Cumulative v10 → v19 =
**571 → 198 ms (-65%)**.

v17 → v18: cat2_c1_s8 picker landed `per_input_lut`. Per-call
cat_13 (yolov8 H=W=20 C_inputs=64|80) dropped 4.22 M → 638 k
rdcycle (-85%). Across the 7 cat2 calls in yolov8, total cat2
contribution dropped roughly 16 M → 6 M rdcycle. Cascade saved
~14 ms on rvv_opu kernel and ~12 ms on gemmini dep_wait. cat3
and cat4 LUT verify FAILED in spike-harness even though host
bit-exact tests across 14 seeds × 4 shapes passed; #276 tracks
diagnosis.

Cumulative v10 → v18: **571 → 211 ms (-61%)**.

**v17 is the biggest single step on the optimization curve.** Root
cause of the v8-v14 plateau: `examples/dronet/int8/cache/rvv_opu/`
had **no** `rvv_opu_conv2d_s8_im2col_rvv_reduce.c` since cache
inception, so dronet's 5 rvv_opu conv2d_s8 calls (one of them
49 M rdcycle each — `conv_modules.8` IC=128 KH=3) were running the
scalar reference. The v11g work cached rvv_reduce for yolov8 only
(its 4 small detect-head convs) and never landed it for dronet.
v15 attempted to add im2col_outerprod ahead of rvv_reduce; outerprod
spike-harness verify failed for dronet (uncached root cause —
crashed spike before the rvv_reduce candidate could run, so the
picker fell straight back to scalar reference for the conv2d_s8 op).
**v15 was reverted before measurement** — the v15 row in the table
is documentational.

v17 just copied rvv_reduce + silu_lut_gather into dronet's per-net
cache. Picker accepted both. Measured deltas vs v14:

```
dronet conv_modules sum   131 M → 8.8 M  rdcycle  (14.9× speedup)
all rvv_opu conv2d_s8     135 M →  12 M  rdcycle  (11.0×)
gemmini wall              435 → 216 ms   (−219 ms, −50%)
gemmini dep_wait          325 → 106 ms   (−219 ms — cascade)
rvv_opu wall              515 → 223 ms   (−292 ms, −57%)
rvv_opu kernel            453 → 158 ms   (−295 ms)
makespan                  515 → 223 ms   (−292 ms, −57%)
yolov8 instance           363 → 195 ms   (−168 ms, −46%)
dronet instance           149 →  39 ms   (−110 ms, −74%)
mlp_control instance       80 →  11 ms   (−69 ms, −87%)
```

The dronet/mlp instance walls drop disproportionately because the
walker's periodic gate stops being slack-eaten by upstream rvv_opu
conv work — when dronet conv finishes 14× faster, every downstream
periodic instance launches on schedule and finishes promptly.

Correctness unchanged from baseline: mlp bit-exact, dronet 72,
yolov8 22 (pre-existing #247 / im2col tail drift).

v11g → v13 (LUT no guard): batchnorm2d_s8 sum 73 M → **121 M** rdcycle
(+48 M regression). The 256-entry LUT build (256× cast/mul/FMA/div/
roundf/clamps) only amortizes when H*W ≥ ~256 per channel — yolov8's
deeper layers (5×5=25 px, 10×10=100 px) lose 3-12× per call.
Documented in `v13_bn_lut_cached/REJECTED.md`.

v11g → v14 (LUT with `spatial >= 256` guard): batchnorm2d_s8 sum
73 M → **46 M** rdcycle (−27 M, −37%). Per-shape ratios (v14/v11g):

```
spatial   v14/v11g    notes
   25     0.93x       reference branch (guard skips LUT)
  100     0.95x       reference branch
  400     0.30-0.76x  LUT wins consistently
  729     0.46x       LUT
 1600     0.14-0.57x  LUT — biggest wins on dronet bn_modules
 6400     0.17x       LUT — single 80×80 BN, 6× speedup
```

makespan 531 ms → 515 ms (−16 ms, −3%). yolov8 instance wall
383,653 → 363,344 µs (−20 ms). Correctness unchanged (mlp bit-exact,
dronet 72, yolov8 22).

v10 → v11g delta: rvv_opu kernel down 39 ms (509 → 470), gemmini
dep_wait down 40 ms (384 → 344), makespan down 40 ms (571 → 531).
The conv2d_s8 reduce kernel landed cleanly with the per-dispatch
VS re-arm fix (commit f98a5dd) — yolov8 instance wall dropped
422,892 → 383,653 µs (-39 ms), aligning with the kernel-time delta.

The 39 ms saving is smaller than the spike microbench's 20× ratio
predicted because the reduce kernel only beats reference scalar on
the *inner k-reduction*, not on im2col packing or accumulator
write-out. conv2d_s8 remains the single largest rvv_opu kernel
category on the joint run:

```
v11g per-op rvv_opu kernel rdcycle attribution (sum across 283 ops):

  conv2d_s8        131.3 M (9 ops, avg 14.6 M)   ← largest single op
  batchnorm2d_s8    73.3 M (60 ops, avg 1.22 M)
  cat2_c1_s8        16.0 M (7 ops, avg 2.29 M)
  silu_s8           12.4 M (54 ops, avg 0.23 M)  ← LUT kernel from v10
  cat3_c1_s8        12.0 M (6 ops, avg 2.00 M)
  cat4_c1_s8         6.6 M (3 ops, avg 2.19 M)
  add_s8             6.4 M (10 ops)
  ── remainder (maxpool/upsample/linear/relu/elu/sigmoid) < 7 M ──
```

Next candidates by impact:
- `batchnorm2d_s8` (73 M rdcycle, 60 calls) — a vectorized
  `vmadd` pass over `(c · scale + bias)` would likely cut 50-60%.
- Outer-product `conv2d_s8_im2col_outerprod` kernel — the
  AlgorithmCandidate is registered (#264) and FPGA-verified
  opcodes confirmed (#266) but not yet on the realized path for
  v11g. Picking it up should approach the spike 50× ceiling on
  the larger yolov8 conv shapes.
- `cat*_c1_s8` (34 M total) — currently scalar memcpy-style; an
  RVV vle/vse loop would cut 40-50%.

`kernel` time on rvv_opu dropped by 223 ms (732→509). gemmini's
kernel time stayed flat (silu is rvv_opu only). gemmini's
`dep_wait` dropped 215 ms (599→384) because rvv_opu finishes
producing yolov8 intermediate buffers sooner. The runtime-overhead
categories (`sync`, `gate`, `idle`) remain sub-1 ms in both runs —
confirming once more that the runtime itself is not the bottleneck.

## Per-network worst-instance wall (µs)

| | v8 | v9 | v10 | v11g | v14 | v17 | v18 | v19 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp_control | 77,583 | 77,783 | 77,397 | 77,316 | 79,677 | 10,511 | 6,693 | **6,681** |
| dronet | 178,237 | 178,280 | 153,483 | 145,168 | 148,849 | 38,505 | 37,246 | **36,216** |
| yolov8_nano | 638,448 | 638,187 | 422,892 | 383,653 | 363,344 | 195,025 | 182,174 | **168,576** |

yolov8 saw the biggest drop (−215 ms = −34%) because all 57 silu
ops live in yolov8.

## What's still in the gap

Calibrated solver prediction (G3) against the v9 PDB: 782 ms
(matched v9 measurement within 0.5%). With the new silu kernel,
the per-op cycle estimates in the PDB are stale again — the
predicted makespan should drop to roughly v10 measured (571 ms)
once the v10 cycles are re-ingested. Tracked as a follow-up.

Remaining slow rvv_opu kernels:
- conv2d_s8 (3 shapes, ~140 ms total) — proper OPU outer-product
  kernel needed; the spike→FPGA divergence on RVV intrinsics
  (vluxei8 unsupported by Saturn-OPU bitstream) means a hand-rolled
  conv must stick to the bitstream's actual ISA subset.
- batchnorm2d_s8 (8 ms) — small.

These are tracked under #255 as future work.

## v11 — initial mis-diagnosis, then unblocked

A v11 sequence tried to vectorize the rvv_opu-routed conv2d_s8
kernel using RVV. Every variant crashed at the first vsetvli with
illegal-instruction. A first ISA probe binary
(`examples/opu_probe/`) ran on hart 0 only (Rocket + Gemmini) and
reported `misa.V_bit=0`, leading to a wrong conclusion that the
bitstream lacked V everywhere. A **second probe** (same source,
extended to spawn a thread pinned to hart 1) found the real story:

```
HART_0 mhartid=0 misa.V_bit=0   (Rocket+Gemmini — no V, expected)
HART_1 mhartid=1 misa.V_bit=1   ← V IS on the Shuttle tile
HART_1 CSR_INIT mstatus.VS=0    ← but Zephyr left it Off
HART_1 CSR_AFTER mstatus.VS=1   ← csrs mstatus, 0x200 latches!
OPU_PROBE_01..10                ← ALL PASS:
  vsetvli e8/m1, e16/m2, e32/m4       ✓
  vle8 / vse8                         ✓
  vwmul.vv, vwadd.wv, vredsum.vs      ✓
  OPMVINBCAST, VOPACC, VMV_VR         ✓
```

The bitstream is exactly what the registry claims (Shuttle + Saturn
OPU vector unit with the four custom OP-V instructions); `misa` is
per-hart and Zephyr's `HAS_V()` reads it on the primary hart (hart
0 here) so it leaves `mstatus.VS=Off` on every hart, including the
V-capable hart 1. First vsetvli on hart 1 then traps.

**Fix (commit 1a12db9):** at the top of every `xpurt_worker()`,
read THIS hart's misa, and if V is set, raise `mstatus.VS` to
Initial via `csrs mstatus, 0x200`. Cleanly no-op on non-V harts.
Once VS is Initial, every standard RVV instruction the conv2d_s8
kernels emit executes normally on hart 1 — confirmed by all 10
probe tests passing.

With that fix applied, the v11 kernels are viable:
- `im2col_rvv_reduce`: ~20× cycle reduction over reference scalar
  (verified bit-exact on spike earlier; FPGA measurement pending).
- `im2col_outerprod`: ~50× ceiling via Saturn OPU outer-product
  engine (VOPACC + OPMVINBCAST + VMV_VR). FPGA-verified opcodes
  now confirmed.
- `im2col_vlA_scalarMAC`: kept as a safe e8/m1-only fallback.

## Correctness gates

- mlp_control: bit-exact through v8/v9/v10. **PASS** ✓
- dronet: max_abs_err=72 since v8 (was bit-exact in d000753 with
  fewer instances). Diagnosed under #247 — likely shared
  `out_dronet[]` buffer overwrite across the 4 dronet instances.
  Not a v10-introduced regression.
- yolov8: max_abs_err=22 since v8 (known gemmini_im2col tail
  drift on yolov8 deeper conv shapes). Not a v10-introduced
  regression.

## Reproducing v10

```bash
cd /scratch2/agustin/ModelBlaster
source scripts/setup_benchmark_env.sh
export PYTHONPATH=/scratch2/agustin/ModelBlaster/src
export SCHEDULE_JSON=/scratch2/agustin/XPU-RT/schedules/scheduled_networks_1yolo_4mlp_2dronet_firesim_hybrid.json
export MODELS=mlp_control,dronet,yolov8_nano
export REGISTRY=/scratch2/agustin/ModelBlaster/cores/chipyard_gemmini_opu_hetero.json
export BACKENDS=gemmini_q31,rvv_opu CPU_P_KIND=gemmini CPU_E_KIND=rvv_opu
export RUNNER=firesim FIRESIM_QUEUE=1 MAX_ACCURACY_CLASS=bit_exact
export QUANT=int8 FORCE_REGEN=0 GENERATED_PREFIX=hybrid_v10
bash examples/xpurt_demo/run.sh

python3 scripts/parse_runtime_breakdown.py \
    artifacts/runtime_optimization/v10_silu_kernel/run.log
```

The cached silu kernel is at
`examples/yolov8_nano/int8/cache/rvv_opu/rvv_opu_silu_s8_rvv_lut_gather.c`
and the AlgorithmCandidate is registered in
`pipeline/reference_kernels.py` (search for `rvv_lut_gather`).
