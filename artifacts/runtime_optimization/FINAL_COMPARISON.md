# Phase G — Final Comparison (v8 → v9 → v10)

Hybrid policy `hybrid_periodic_mosek_yolo`, canonical workload
4 MLP@10ms + 2 Dronet@20ms + 1 Yolo@100ms, hetero bitstream
(Gemmini RoCC + Saturn-OPU on Shuttle tiles), `FIRESIM_QUEUE=1`.

## Run table

| Run | Notes | gemmini wall | rvv_opu wall | makespan | mlp err | dronet err | yolo err |
|:---|:---|---:|---:|---:|---:|---:|---:|
| v8 | baseline (no instrumentation) | n/a | n/a | **638 ms** | 0 | 72 | 22 |
| v9 | G1 instrumentation, same code as v8 | 709 ms | **786 ms** | 786 ms | 0 | 72 | 22 |
| v10 | + RVV silu_s8 LUT kernel (G5) | 494 ms | **571 ms** | **571 ms** | 0 | 72 | 22 |

Improvement v9 → v10: **−215 ms on makespan (−27%)**, **−223 ms
on rvv_opu kernel time (−30%)**, exactly the silu_s8 contribution
the PDB predicted (~12.1 M spike cycles vs ~120 M reference).

## Per-hart attribution (mtime µs)

```
v9   gemmini  708,799   kernel=109,080  dep_wait=599,085  sync=116  gate=448  idle=47
v9   rvv_opu  786,534   kernel=732,620  dep_wait= 53,586  sync=234  gate=  0  idle=54

v10  gemmini  493,768   kernel=109,403  dep_wait=383,728  sync=119  gate=449  idle=56
v10  rvv_opu  571,117   kernel=509,152  dep_wait= 61,646  sync=203  gate=  0  idle=72
```

`kernel` time on rvv_opu dropped by 223 ms (732→509). gemmini's
kernel time stayed flat (silu is rvv_opu only). gemmini's
`dep_wait` dropped 215 ms (599→384) because rvv_opu finishes
producing yolov8 intermediate buffers sooner. The runtime-overhead
categories (`sync`, `gate`, `idle`) remain sub-1 ms in both runs —
confirming once more that the runtime itself is not the bottleneck.

## Per-network worst-instance wall (µs)

| | v8 | v9 | v10 |
|:---|---:|---:|---:|
| mlp_control | 77,583 | 77,783 | **77,397** |
| dronet | 178,237 | 178,280 | **153,483** |
| yolov8_nano | 638,448 | 638,187 | **422,892** |

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

## v11 attempt — blocked by bitstream limitation

A v11 sequence tried to vectorize the rvv_opu-routed conv2d_s8
kernel using RVV. Every variant crashed at the first vsetvli with
illegal-instruction. Root cause was confirmed by a dedicated ISA
probe binary (`examples/opu_probe/`, FireSim job 220):

```
CSR_INIT misa=0x800000000094112d (V_bit=0)
         mstatus=0x8000000a00018088 (VS=0)
CSR_AFTER mstatus=0x8000000a00018088 (VS=0)   # manual csrs mstatus, 0x200 IGNORED
OPU_PROBE_01: vsetvli e8/m1 SET rs1=16 START
  mcause: 2, Illegal instruction
  mtval: c0975d7                              # vsetvli s11, s2, e8, m1, ta, ma
```

`misa.V = 0` on this FireSim `FireSimGemminiAndOPUShuttleConfig`
bitstream. mstatus.VS is hardwired to Off when misa.V is 0, so any
attempt to enable VS is silently dropped, and every standard RVV
opcode (vsetvli, vle8, vse8, vwmul, vredsum, …) traps. The Saturn
OPU custom OP-V instructions (VOPACC / OPMVINBCAST / VMV_VR /
VMV_RV) use the same OP-V opcode space and almost certainly also
require RVV's vsetvli setup to be usable — so they're effectively
unreachable too on this specific bitstream.

**Conclusion: the rvv_opu-side scalar-reference conv2d_s8 path that
v10 already ships IS the production result for this FPGA image.**
Until a bitstream is synthesized with real V extension support, all
rvv_opu kernels must be scalar (no inline RVV asm, no `__riscv_v*`
intrinsics, no Saturn OPU custom). The four kernels added in the
v11 attempt (`im2col_rvv_reduce`, `im2col_outerprod`,
`im2col_vlA_scalarMAC`, the OPU probe) are kept in the codebase
under `kernels/rvv_opu/` for the future bitstream but are removed
from the yolov8 per-model cache so the picker falls through to the
scalar reference.

Path forward to actually beat v10's 571 ms on this same bitstream:
1. **Build a new bitstream** with `FireSimSaturnGENV256D128ShuttleConfig`
   (already in `config_hwdb.yaml`) or a hetero variant whose Saturn
   stage advertises misa.V=1. Then v11's `im2col_rvv_reduce` and
   eventually `im2col_outerprod` become unblocked and the predicted
   20x cycle reduction on conv2d_s8 lands.
2. **Runtime-side wins on the existing bitstream** (orthogonal to
   the kernel ISA question): producer-side fanout signaling (G2d),
   async dispatch overlap, walker hot-loop scrubbing — each saves
   sub-50 ms but they're additive.

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
