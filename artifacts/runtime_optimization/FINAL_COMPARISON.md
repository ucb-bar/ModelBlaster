# Phase G — Final Comparison (v8 → v9 → v10 → v11g)

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
```

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

| | v8 | v9 | v10 | v11g |
|:---|---:|---:|---:|---:|
| mlp_control | 77,583 | 77,783 | 77,397 | **77,316** |
| dronet | 178,237 | 178,280 | 153,483 | **145,168** |
| yolov8_nano | 638,448 | 638,187 | 422,892 | **383,653** |

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
