# K3 Phase A probes (steps 1, 3b, 5, 4-vwmacc)
Raw: toolchain_march.txt, spill/ (objdump + summary.txt), microcost_cpu0.txt (+_run2 repeat), microcost_cpu8.txt.
Probe: scripts/k3_rvv_microcost_probe.c, runner scripts/run_k3_probe.sh. rdtime 24 MHz (41.7 ns tick), median of 41 reps; CPU0 built zvl256b, CPU8 zvl1024b.

## 1. -march acceptance (compile of a tiny vle8/vse8 file, -O2 -c)
| -march | cross gcc 14.3 (SpaceMiT) | board gcc 15.2 |
|---|---|---|
| rv64gcv_zvl1024b | ok | ok |
| rv64gcv_zvl1024b_xsmtvdotii | REJECT: "extension 'xsmtvdotii' starts with 'x' but is unsupported non-standard extension" | ok |
| rv64gcv_xsmtvdot | REJECT (same message) | ok |
| rv64gcv_xsmti8i32mm | REJECT (same) | REJECT (same) |
| rv64gcv_zvl1024b_zfh_zvfh | ok | ok |
(Board acceptance means only the -march parses; IME mnemonics untested. Cross gcc would need .insn.)

## 3b. Spill hazard (-O2, rvv_k3 flags, -include riscv_vector.h mb_rvv_vxrm_compat.h)
All 4 kernels at both zvl256b and zvl1024b: 0 vs1r/vs2r/vs4r/vs8r, 0 vl*re*.v, 0 vlenb reads. Stack frames identical (conv_silu 1200 vs 1184 B, conv 608, maxpool 592, linear 160). No whole-register spills -> no hazard for these 4 at -O2. (Only vector stack traffic is the kernels' own explicit vse8 to local arrays, e.g. maxpool tmp[256].) Hazard remains possible for other kernels/opt levels; not proven absent globally.

## 5/4. Micro-costs (ns; see raw files for all columns)
| measurement | CPU0 (X100, vlenb 32) | CPU8 (A100, vlenb 128) |
|---|---|---|
| vle8 e8m1, 2048 elems | 125 ns (16.4 el/ns) | 292 ns (7.0 el/ns) |
| vle8 e8m4, 2048 elems | 42 ns (49 el/ns) | 83 ns (24.6 el/ns) |
| vlse8 stride2 e8m1, 2048 elems | 708 ns (2.9 el/ns) | 31167 ns (0.066 el/ns) |
| vlse8 stride2 e8m4, 2048 elems | 833 ns (2.5 el/ns) | 27792 ns (0.074 el/ns) |
| vredsum e32 latency m1 / m4 | 2.28 / 5.05 | 10.7 / 37.3 |
| vredsum e32 throughput m1 / m4 | 0.65 / 1.30 | 2.77 / 9.44 |
| vredsum at vl=16 vs vlmax | same | same (cost set by LMUL, not vl) |
| 32B scatter stride 64: vse8+scalar loop / vsse8 | 62.5 / 57.9 | 57.3 / 125 |
| vse8 + 32 scalar byte reloads | 18.2 | 69.0 |
| vse8 only (32B) | 1.3 | 2.6 |
| vsetvli const / alternating e8m1<->e32m4 | 0.90 / 0.90 | 2.20 / 2.28 |
| alt with vadd between / const with vadd | 1.14 / 0.90 | 4.48 / 2.20 |
| vwmacc.vx i8->i16 ns/instr (indep) | 0.936 | 2.238 (any vl) |
| vwmacc i8->i16 MAC/ns, vl=vlmax / 32 / 16 | 34.2 (vl32) / 34.2 / 17.1 | 57.2 (vl128) / 14.3 / 7.1 |
| vwmacc i16->i32 MAC/ns vlmax / vl32 / vl16 | 17.1 / 17.1 / 17.1 | 28.6 (vl64) / 14.3 / 7.1 |
| vwmacc chained latency i8 | 1.38 | 2.81 |
| scalar int8 MAC loop | 0.854 ns/MAC (1.17/ns) | 2.238 ns/MAC (0.45/ns) |

## Interpretation
- Instruction cost on A100 is ~constant per instruction, independent of vl: vwmacc at vl=16/32 gets 1/8..1/4 of vlmax throughput. Short-vl kernels (OC=16/32) waste the A100. At vlmax the A100 beats X100 only 1.7x (57 vs 34 MAC/ns), well below 4x width; the per-instr latency/issue is ~2.4x worse (2 GHz + slower pipeline).
- Strided load vlse8 is catastrophic on A100: ~15 ns per element (~1950 ns per e8m1 instr, 6950 per m4), 400x slower than unit stride and 40x slower than X100. Strongly supports the maxpool vlse8 hypothesis (6.26 vs 0.44 ms).
- vredsum m4 costs 37 ns latency (9.4 thr) on A100 vs 5 on X100 (7x), m1 is 3.5x cheaper than m4 on A100. Supports linear_s8 vredsum hypothesis; use m1 or avoid per-(m,n) reductions.
- Scalar core is 2.6x slower on A100 (2.24 vs 0.85 ns/MAC): the scalar `roundf`/divide fallback (LUT gate) and scalar _obuf scatter loops are hit hardest. vse8+scalar reload 69 vs 18 ns (3.8x worse). Supports LUT gate + _obuf hypotheses.
- vsse8 (stride 64) on A100 is 2.2x slower than the scalar scatter (125 vs 57 ns per 32B) -> the plan's "replace scatter with vsse8" is REFUTED as a win on A100 (slightly better than scalar on X100 only 8%). Prefer pixel-major tile / unit-stride stores with a layout change.
- vsetvli churn: no extra cost on X100; on A100 alternating vsetvli with ops between doubles cost (4.48 vs 2.20 ns for 2 vsetvl+2 vadd) but vsetvli itself is cheap (2.2 ns). Minor.
- Caveats: 41.7 ns timer tick (long inner loops used); vle8 e8m1 loop figures include loop overhead; vlse8 measured with stride 2 only.
