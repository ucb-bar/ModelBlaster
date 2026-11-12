# K3 IME (vmadot) probes - Phase A steps 2, 3, 4, 6
Probes: scripts/k3_ime_presence_probe.c, k3_ime_semantics_probe.c, k3_ime_throughput_probe.c (run via scripts/run_k3_probe.sh).
Raw: presence_cpuN.txt (all 16 CPUs), semantics_cpu{0,8}.txt, throughput_cpu{0,4,8,12}.txt. All via .insn (cross gcc 14.3).

## Presence (SIGILL handler; vl=32/VLMAX/64 all same)
| CPUs | s/u/us/su int8 (f7=0x71) | int4 (f7=0x61) | LMUL=2 | odd vd |
|---|---|---|---|---|
| 0-7 (X100, vlenb 32) | EXEC | SIGILL | SIGILL | SIGILL |
| 8-15 (A100, vlenb 128) | EXEC | EXEC | SIGILL | SIGILL |

## Verified semantics (vmadot vd, vs1=A, vs2=B; `.insn r 0x2b, f3, 0x71, vd, vs1, vs2`)
- Tile int8 M x K x N: A100 8x16x8, X100 4x8x4 (so M = N = vlenb/16.. i.e. K = vlenb/8... A uses vlenb bytes, B vlenb bytes).
- A (vs1): bytes row-major `A[i*K+k]`. B (vs2): `B[j*K+k]` (B transposed / "column-major"). C: int32 `C[i*N+j]` row-major over the concatenation (vd elements first, then vd+1); vd must be even. `C += A . B^T` (accumulates; verified by issuing twice and with random nonzero C0).
- funct3: 3 = A s8 * B s8; 0 = u*u; 1 = A unsigned * B signed; 2 = A signed * B unsigned. Each matched the model in 20000/20000 random trials (incl. extreme values), CPU 0 and CPU 8.
- vl is IGNORED on K3 (vl = 1..VLMAX all give the full-tile result; K1 needed vl=32). Inputs must be e8 m1; LMUL=2 traps. Output read with vsetvli e32,m1 + vse32 of vd and vd+1.
- int4 (A100 only, f7=0x61): 8x32x8, signed nibbles, low nibble = lower k; 5000/5000 random match.
- Scan method: all 128x128 single-hot (A byte, B byte) pairs hit exactly the predicted C element (0 mismatches, 0 multi-hit).

Reference model (in the probe, `ref_vmadot`):
```c
for i<M, j<N: s=0; for k<K: s += a(A[i*K+k]) * b(B[j*K+k]); C[i*N+j] += s;   // a/b sign- or zero-extended per funct3
```
K-loop for GEMM: advance k by K=16 (A100) per vmadot, keep C accumulators in vd/vd+1 across the whole K; pack A as [m-tile][k-slab][8 rows x 16 B] and B as [n-tile][k-slab][8 cols x 16 B].

## Throughput (rdtime 24 MHz, median of 41, 1024 vmadots/run)
| measurement | A100 (CPU 8, 12) | X100 (CPU 0, 4) |
|---|---|---|
| dependent chain latency | 4.48 ns | 2.28 ns |
| 2 / 4 / 8 independent accs | 449 / 868 / 1480 MAC/ns | 112 / 225 / 262 MAC/ns |
| peak (8 accs), ns/instr | 0.69 (1024 MAC) | 0.49 (128 MAC) |
| int4 8 accs | 0.65 ns, 3146 MAC/ns | n/a |
| vwmacc i8->i16 (earlier probe) | 57 MAC/ns | 34 MAC/ns |
| peak ratio vs vwmacc | 26x | 7.7x |
| 2x2 tiles, 4 vle8 per 4 vmadot, L1 | 127-141 MAC/ns | 112 |
| 2x4 tiles, 6 vle8 per 8 vmadot, L1 | 213-265 MAC/ns | 150 |
| 2x4 tiles, whole-reg vl2re8+vl4re8 | 213-227 | 131 |
| vl4re8 L1 load rate | ~25 B/ns (20 ns per 512 B) | 47 B/ns |
vl is irrelevant to vmadot cost. A100 is LOAD-BOUND: a 128 B vle8 costs ~5 ns (whole-register loads no faster), so useful MAC/ns ~= 8*m*n/(m+n) MAC/B x 25 B/ns (2x4 -> ~265; 3x4 would need ~31 vregs, ~340). Needs >=4 (ideally 8) independent accumulator pairs to hide the 4.5 ns latency.

## Recommendation
Yes for both clusters; IME is the first real way to make the A100 competitive.
- A100: even load-bound, a 2x4-tiled int8 GEMM sustains ~210-265 MAC/ns vs 57 for the best RVV widening MAC (4-5x), and peak is 26x. The conv problem (short OC filling 1/8 of a register, scalar pixel loops, scalar requant) disappears if conv is lowered to im2col+vmadot GEMM (M=pixels in multiples of 8, N=OC multiples of 8, K multiple of 16; pad). Requant epilogue stays scalar/RVV, so expect Amdahl limits on small layers; small M (mlp_control) wastes tile rows.
- X100: IME1 (4x4x8-style, vl-independent) executes on all 8 X100 cores, 262 MAC/ns peak (7.7x vwmacc), ~110-150 realistic; worth an `ime` alternative there too (K1 kernel ime_*_4x4x8 has to be retargeted: vl no longer needs to be 32, tile K=8 on X100, K=16 on A100, so a parameterised kernel).
- Caveat: operand loading dominates on A100; pack operands once, keep tiles in registers, use 2x4 or larger blocking. Latencies/rates measured on the L1-resident synthetic; real kernels need an end-to-end check.
