# linear_s8 / matmul_s8 on the K3 A100 (IME vmadot 8x16x8)

Kernels: `kernels/ime_k3_a100/ime_k3_a100_{linear,matmul}_s8_ime_vmadot_8x16x8.c`. Measured with `scripts/verify_kernel_on_board.py --board k3 --cpu 8 ... --baseline-kernel <rvv kernel>` (median of 21-51 reps, 24 MHz timer, tick 0.042 us). Baseline = current shared `kernels/rvv` kernel built with the same A100 flags (`rvv_k3_a100`, zvl1024b). All numbers microseconds. `max_abs_err == 0` in every case (full and small-value regimes). Files: `*/after_ime_cpu8.json`, `*/before_rvv_k3_a100_cpu8.json`, `*/before_rvv_k3_cpu0.json` (X100 reference).

## linear_s8 (real shapes from build/k3/*/int8/graph.json)

| shape | A100 rvv (before) | A100 new | speedup | X100 rvv (ref) | path |
|---|---|---|---|---|---|
| dronet:linear1 N1 M1 K2048 | 1.54 | 1.25 | 1.23x | 0.29 | diag (pipelined dot) |
| dronet:linear2 N1 M1 K2048 | 2.00 | 1.58 | 1.26x | 0.29 | diag (pipelined dot) |
| mlp_control:mlp.0 N256 M1 K16 | 29.21 | 3.92 | 7.46x | 6.46 | direct (K==16) |
| mlp_control:mlp.2 N128 M1 K256 | 35.67 | 12.96 | 2.75x | 6.58 | diag pairs (T=2) |
| mlp_control:mlp.4 N64 M1 K128 | 11.71 | 4.38 | 2.68x | 2.21 | diag quads (T=1) |
| mlp_control:mlp.6 N4 M1 K64 | 0.58 | 0.38 | 1.56x | 0.12 | k64 |
| **total** | 80.7 | 24.5 | 3.30x | 16.0 | |

Synthetic linear shapes (22 shapes incl. odd N/K, M=1..128, K%16 tails, K=7..2048) were also all bit-exact; M=128 K=256 N=256: 337 us vs 5889 us (17x), M=64 K=512 N=128: 175 vs 2438 us. Tiny shapes with N<=5 are at parity (+-5%).

## matmul_s8 (no model uses it; synthetic shapes, attention/FFN-like)

| shape | A100 rvv (before) | A100 new | speedup | X100 rvv (ref) | path |
|---|---|---|---|---|---|
| M=7 K=64 N=7 tb=1 | 11.38 | 2.96 | 3.85x | 2.88 | IME packed |
| M=7 K=7 N=64 tb=0 | 22.12 | 2.08 | 10.62x | 13.92 | rvv tb0 small-M |
| M=7 K=512 N=512 tb=0 | 628.83 | 260.08 | 2.42x | 515.71 | rvv tb0 small-M |
| M=7 K=512 N=512 tb=1 | 1512.71 | 319.67 | 4.73x | 516.62 | IME packed |
| M=64 K=512 N=512 tb=0 | 5722.58 | 1372.42 | 4.17x | 4750.38 | IME packed |
| M=64 K=512 N=512 tb=1 | 13246.83 | 509.96 | 25.98x | 4734.54 | IME packed |
| M=33 K=130 N=45 tb=0 | 207.17 | 61.75 | 3.35x | 95.54 | IME packed |
| M=33 K=130 N=45 tb=1 | 422.00 | 30.33 | 13.91x | 117.25 | IME packed |
| M=1 K=64 N=100 tb=1 | 21.79 | 6.75 | 3.23x | 6.08 | rvv tb1 small-M |
| M=128 K=256 N=256 tb=0 | 3621.96 | 472.46 | 7.67x | 2602.42 | IME packed |
| M=1 K=512 N=512 tb=0 | 91.33 | 69.75 | 1.31x | 72.46 | rvv tb0 small-M |
| M=31 K=512 N=512 tb=0 | 2774.38 | 1094.00 | 2.54x | 2298.92 | rvv tb0 small-M |
| M=32 K=512 N=512 tb=0 | 2863.62 | 1258.62 | 2.28x | 2375.04 | IME packed |
| M=3 K=33 N=150 tb=0 | 29.42 | 5.75 | 5.12x | 16.54 | rvv tb0 small-M |
| M=1 K=2048 N=64 tb=1 | 95.12 | 49.42 | 1.92x | 27.46 | rvv tb1 small-M |
| M=2 K=512 N=512 tb=1 | 442.92 | 201.33 | 2.20x | 147.71 | rvv tb1 small-M |
| M=8 K=16 N=8 tb=1 | 13.38 | 1.08 | 12.35x | 3.38 | IME packed |
| M=9 K=17 N=9 tb=0 | 8.46 | 3.50 | 2.42x | 3.33 | rvv tb0 small-M |
| **total** | 31736 | 5722 | 5.55x | | |

## Design and threshold logic
Measured facts (A100, cpu 8) that drive it: the old rvv linear is reduction bound (vredsum m4 37 ns, vwadd.wv m4, `vwredsum` ~80 ns); `vluxei64` gathers ~300 ns per 128 B tile (rejected); strided/segment memory ops are ~4 ns/element; vmadot 0.7 ns throughput / 4.5 ns latency; weights arrive at only ~3 B/ns per call (the per-call load floor: 32 KB = 8-10 us), so **load count, not MACs, is what is optimized**.

linear_s8 paths, chosen per call (see header of the .c):
- `K == 16` (mlp.0): 16-byte weight rows are B-tile rows, so 8 rows = 128 contiguous bytes; one e8m4 load = 4 tiles, A = input rows. 7.5x.
- `M < 8 && K % 128 == 0` (mlp.2, mlp.4, dronet): "diag" path. A tile = 8 contiguous 16-byte chunks of x, B tile = the same 8 chunks of one weight row, so C[i][i] = x_i . w_i and the row dot product is the diagonal sum accumulated over all tiles. Only the diagonal is useful (1/8 of the tile) but everything is contiguous e8m1/m2/m4 loads with no repack, x stays in registers, rows are batched (4 rows per e8m4 load for K=128, 2 for K=256, software-pipelined x/w loads for K>=1024), and the diagonal is extracted with two `vand` masks + one `vredsum` e32m1 per row (no m4 reduction, no per-(m,n) chain). 
- `K == 64 && M < 8` (mlp.6): same idea, 2 rows per tile, masks pick lanes {0,9,18,27}/{4,13,22,31}.
- `M < 8`, other K: pure RVV, vwmul + vwredsum, 4 rows interleaved, vector requant.
- `M >= 8` (IME_LIN_SMALL_M = 8): weights packed once per 32 output columns, A per 96 KB M-block, 2(m) x 4(n) vmadot blocking with 8 accumulator pairs. Packed at M<8 is 2-5x slower than diag/rvv (pack cost not amortized), so the threshold is exactly the tile height.
- requant is vectorized and bit-exact: vsmul(RNU) = Q0.31 round, vssra(RNU) = rounding shift, vmax/vmin, truncating vncvt. Nonzero offsets or |shift| > 30 fall back to a scalar loop (never occurs in this tree).

matmul_s8: packed IME for all M except (a) `transpose_b != 0 && M < 3` and (b) `transpose_b == 0 && M < 32`, which use an in-file RVV path because B packing/transposing (~1.3 ns/B for tb=1, ~4 ns/B for tb=0, which uses 16 vle8 + 2 `vssseg8e8` per 16x32 block) is not amortized (measured crossovers: tb=1 at M~3, tb=0 at M~34). Requant is bit-exact vs the `roundf((float)acc*total)` oracle using `vfcvt.x.f` with static rm=RMM (ties away from zero).

## Verification
- `max_abs_err == 0`: 6 real linear shapes, 22 synthetic linear, 18 matmul shapes (both transpose_b, odd sizes, K tails, M 1..128), on cpu 8, both value regimes.
- No whole-register spills: `objdump -d` of both objects built with the rvv_k3_a100 flags (`-march=rv64gcv_zvl1024b_zfh_zvfh -O2`) contains no `vs[1248]r.v` / `vl[1248]re*.v`. (GCC initially hoisted a zero vector out of loops and spilled it around the inline asm; fixed with an opaque zero source and `noinline` requant helpers.)
- `check_rvv_avl.py`: 0 chained AVLs; `check_rvv_vtype.py` (--mattr rv64gcv_zvl1024b): OK for both.
- `--backend ime_k3_a100` (no --kernel-file) picks `kernels/ime_k3_a100/ime_k3_a100_linear_s8_ime_vmadot_8x16x8.c` (confirmed on cpu 8, VERIFY PASS).

## Open issues
- dronet linear (N=1, K=2048) and mlp.6 are bound by per-call cold-load latency (a plain load-and-xor of the same 4 KB costs ~1.25 us); the kernel is within ~15% of that floor, speedup there is only 1.2-1.6x.
- The X100 `rvv` linear is still faster than the A100 new kernel at the same shapes (mlp.2 6.6 vs 13.0 us, dronet 0.29 vs 1.25 us); the A100's weight fetch (~3 B/ns) is the limit, not compute.
- Large-M epilogue (requant + 8-byte row scatter per tile) and, for matmul tb=0, the B transpose are the remaining costs (M=128,K=256,N=256 linear is ~25 MAC/ns vs ~270 peak). A register-level transpose could cut tb=0 pack cost 5x+.
- Only nonzero-offset linear and |shift|>30 use scalar code; matmul has no scalar path (K<=0 handled).
- `scripts/verify_kernel_on_board.py` formats whole-number float shape params (e.g. default scale_div=1.0) as `1f`, which does not compile; synthetic matmul shapes need an explicit non-integer scale_div (I used 1.25). Not edited (not my file).
- harness JSON/tag for `--shape` cases repeats the first shape's tag/dims; shapes are in the order passed.
