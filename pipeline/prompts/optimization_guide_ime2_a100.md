# SpaceMiT K3 A100 matrix engine (`ime2_a100` target)

This guide is for the `ime2_a100` target: int8 kernels that use the A100's
matrix instruction, `smt.vmadot`. The facts below come from measurements on a
K3 board (2026-10-19) and from SpacemiT's matrix-extension specification
(docs-ai `ime_extension.md`). Take them as given; do not re-derive them. Do not
borrow anything from K1 (`ime_x60`) matrix kernels: same instruction encoding,
different tile, wrong answers on this core.

Ordinary RVV 1.0 intrinsics from `<riscv_vector.h>` are available for the rest
of the kernel (packing, requantization); the header is already included.

## Where your kernel runs and how it is checked

- It runs under Linux on the K3 board, on an A100 core (cpus 8-15), not on
  spike. The A100's vector length is 1024 bits (`vlenb` = 128). The compile
  flags say `zvl256b`, which only promises at least 256 bits; the matrix tile
  below does not depend on them.
- During generation your kernel is only cross-compiled; nothing runs it.
  Compile errors come back to you, wrong answers do not. The numbers are
  checked later, on the board, against PyTorch for the whole model, and must
  match exactly (`max_abs_err = 0`). Get the layouts right by reasoning from
  this guide, not by trial and error.

## The instruction: `smt.vmadot`

There is no intrinsic and no `-march` extension for it. Emit it with `.insn`
inside `__asm__ volatile`:

```
.insn r 0x2b, 3, 0x71, xD, xS1, xS2     # vD,vD+1 += A(vS1) * B(vS2)^T
```

- Write the registers as `xN`: `x8` means vector register `v8`. The assembler
  rejects `v8` inside `.insn` with "illegal operands".
- D, S1 and S2 sit in the normal rd, rs1 and rs2 fields. Only `x8, x0, x4`
  (C in v8/v9, A in v0, B in v4) has been run on the board; other register
  numbers use the same fields but are untested.
- Only this signed x signed int8 form has been verified on the board. SpacemiT
  also documents unsigned and mixed-sign forms, int4, sliding-window
  (`vmadot1/2/3`), 4:2 sparse, block-scaled and fp16/bf16 forms. Do not use
  them: their encodings have not been checked here.

**What it computes** (int8 x int8 -> int32, accumulating):

```
for p in 0..63:                      # p = row*8 + col of the 8x8 result
    vd[p] += sum over q in 0..15 of vs1[(p/8)*16 + q] * vs2[(p%8)*16 + q]
```

| Operand | Register(s) | Contents | Bytes |
|---|---|---|---|
| A | `vs1`, one whole register | 8 rows x 16 int8, row-major: row i at bytes `16*i .. 16*i+15` | 128 |
| B | `vs2`, one whole register | 8 rows x 16 int8, row-major, where **row j is output column j**: `B[j][k]` | 128 |
| C | `vd` and `vd+1` | 8 x 8 int32, row-major: element e is `C[e/8][e%8]`; `vd` holds rows 0-3, `vd+1` rows 4-7 | 256 |

So C += A * B^T. The tile is M x K x N = **8 x 16 x 8** (1024 multiply-adds
per instruction) and it is fixed. The instruction **accumulates**: a reduction
longer than 16 is a loop of vmadots over 16-wide K slabs into the same `vd`,
zeroed once before the loop.

**Rules from SpacemiT's specification (illegal instruction otherwise):**
- LMUL must be 1 when it issues.
- `vd` must be even (`v8`, `v10`, ...), and `vd`/`vd+1` must not overlap
  `vs1` or `vs2`.
- There is no mask.

**vl does not matter.** The A100 reads whole 128-byte registers whatever `vl`
is (the K1 needs vl = 32 at the instruction; the A100 does not). Set
`vsetvli t0, zero, e8, m1, ta, ma` once before loading tiles: at e8/m1 that
gives vl = 128, so one `vle8.v` loads exactly one tile. There is no need to
repeat it before each vmadot. A smaller vl does **not** shrink the tile.

**Draining C:** `vsetvli t0, zero, e32, m2, ta, ma` (vl = 64), then one
`vse32.v vD, (ptr)` stores the 64 int32 results contiguously, row-major.
Switch back to `e8, m1` before the next vmadot.

## Tiles and padding

- Tile the output in 8 x 8 blocks and the reduction in 16-wide slabs, even in
  the correctness phase: this is the instruction's shape, not an optimization.
  Holding several result tiles at once (register blocking) is the
  optimization; leave it for the optimization phase.
- When M or N is not a multiple of 8, or K not a multiple of 16, pack into
  zero-filled scratch tiles and write back only the valid results. Zeros add
  nothing to the dot products. Handle edge tiles this way; do not switch to
  another algorithm for them.

## A100 cores only: check `vlenb` first (mandatory for `smt.vmadot` kernels)

On an X100 core (cpus 0-7, `vlenb` = 32) the same encoding computes a
4 x 8 x 4 tile and returns wrong answers without trapping. Measured the other
way round: the K1's 4x4x8 linear kernel ran on an A100 core without a fault and
gave `max_abs_err = 48`. So every kernel that issues `smt.vmadot` must refuse
to run anywhere else. A kernel that uses only ordinary RVV instructions (add,
gelu, layernorm, ...) must NOT have this check: it is correct on any core.
Put this at the top of a `smt.vmadot` kernel, after the early return for empty
sizes:

```c
unsigned long vlenb;
__asm__ volatile("csrr %0, vlenb" : "=r"(vlenb));
if (vlenb != 128) {
    __builtin_trap();
}
```

Use `__builtin_trap()`: it needs no header. `printf` and `abort` need
`<stdio.h>` / `<stdlib.h>`, which are not included and which you must not
include. Placing threads on A100 cores is the harness's job, not the kernel's.

## Operand layouts you will meet

- `linear_s8`: `weight` is [N][K], so output column n is the contiguous row
  `weight[n*K ..]`: exactly B's layout, no transpose. `input` [M][K] rows are
  A's rows. Both pack by copying 16-byte pieces of rows.
- `matmul_s8` with `transpose_b = 1`: b is [N][K], same as linear. With
  `transpose_b = 0`: b is [K][N], so each B tile must be transposed
  (`B[j][k] = b[k][j]`) into a 128-byte scratch tile before loading. Feeding a
  [k][j] tile computes the wrong product, silently.
- Convolution as a matrix multiply: one A row per output pixel (its receptive
  field, IC*KH*KW values), one B row per output channel, K = IC*KH*KW. On this
  target the build defines `MODELBLASTER_RVV_IHWOC_WEIGHTS=1`: conv weights
  arrive as [IC][KH][KW][OC], i.e. `weight[k*OC + oc]` with
  k = (ic*KH + kh)*KW + kw, so packing B gathers with stride OC. The
  algorithm description gives the op's padding and offset rules.

## Offsets, bias and requantization

- `smt.vmadot` multiplies raw int8 values. When `input_offset` or
  `filter_offset` is nonzero, correct each result with
  `sum (a+ia)(w+iw) = sum(a*w) + ia*sum_k(w) + iw*sum_k(a) + K*ia*iw`,
  computing the row sums of A and B while packing (valid elements only). The
  models built so far have zero offsets; implement the correction anyway.
- Add `bias` to the int32 result before the Q0.31 multiply.
- `linear_s8`'s tail is integer arithmetic: reproduce the reference's int64
  steps exactly. `matmul_s8`'s tail uses `roundf`, which rounds halves away
  from zero while RVV float conversion rounds halves to even: keep that tail
  scalar.

## Performance on one A100 core (measured)

| | A100 core |
|---|---|
| vmadot issue rate, 8 independent result tiles | 0.5 per cycle = 512 multiply-adds per cycle (about 920 GMAC/s at 1.8 GHz) |
| vmadot latency, one accumulator (dependent chain) | 8 cycles |
| vector load from ordinary memory | about 7.5 bytes per cycle: **about 17 cycles per 128-byte tile** |
| vector store | about 10-12 bytes per cycle (noisy) |
| gather (`vluxei8`) | about 17 cycles per element |
| reduction round trip (`vmv.s.x`, `vredsum`, `vmv.x.s`) | about 75 cycles |

What follows from these numbers:

1. **One result tile at a time leaves the engine mostly idle.** With an
   8-cycle latency and 0.5 issues per cycle, about 4 independent result tiles
   must be in flight. Block the output: keep a x b result tiles in registers
   (2 registers each), load a A tiles and b B tiles per K slab, and issue a*b
   vmadots. 2 x 4 uses 16 + 6 = 22 of the 32 vector registers; in general
   2ab + a + b <= 32.
2. **Loads are the real limit.** A tile costs about 17 cycles to load against
   about 2 cycles per vmadot at full rate. The 1 x 1 pattern loads 2 tiles per
   vmadot; 2 x 4 loads 6 tiles per 8 vmadots and is still load-bound. Make
   every loaded tile feed as many vmadots as the registers allow.
3. **Pack once per call, never per tile.** Pack all of A up front. Loop over
   8-column B panels outermost, pack each panel once, and reuse it for every
   row tile. (On the K1, packing inside the tile loop made the matrix kernel
   6.5x slower than the vector kernel.)
4. **Packed data lives for one call only.** Statics and globals are not
   allowed, so B is re-packed on every call. Make packing cheap: copy whole
   16-byte pieces for full tiles and keep per-element bounds checks to the
   edge tiles. In the current A100 `linear_s8` (ffn_block fc1,
   128 x 256 x 1024) weight packing is estimated at about a third of the time,
   and the engine is idle over 95% of it.
5. **Avoid gathers and per-output vector reductions** on the A100 (17 cycles
   per gathered element, a 75-cycle reduction round trip). The matrix
   instruction makes per-output reductions unnecessary.
6. **Requantization is 64 results per tile.** `linear_s8`'s integer tail can
   be vectorized exactly with RVV integer instructions: do it in the
   optimization phase, not the correctness phase.
7. **Not measured yet; do not rely on them:** the A100's local memory (TCM),
   which probably feeds tiles much faster than ordinary memory; and two cores
   of one pair (8 and 9) sharing the matrix unit (each ran at 0.89 of solo
   speed in one test).

## Mistakes to avoid

1. Reusing a K1 (`ime_x60`) kernel or its 32-byte tiles: it runs on an A100
   core without trapping and gives wrong answers (`max_abs_err = 48`, measured).
2. Writing `v8` instead of `x8` inside `.insn`: "illegal operands".
3. An odd `vd`, or `vd`/`vd+1` overlapping `vs1`/`vs2`: illegal instruction.
4. LMUL not 1 at the vmadot, for example right after draining at `e32, m2`:
   illegal instruction. Re-issue `vsetvli t0, zero, e8, m1, ta, ma` first.
5. Missing clobbers: list every vector register the asm uses (`"v0"`, `"v4"`,
   `"v8"`, `"v9"`, ...), plus `"t0"` and `"memory"`.
6. B packed as [k][j] instead of [j][k]: no error, wrong numbers. A test that
   deliberately transposed B produced 374,977 wrong outputs.
7. Expecting a smaller `vl` to shrink the tile: it does not.

## Worked example: `linear_s8`, correct and simple

The asm block, packing and tail below are those of the A100 `linear_s8`
kernel that matched PyTorch exactly on an A100 core (44 test shapes, and
ffn_block end to end), written as one function. It keeps one 8 x 8 result
tile at a time and packs with simple loops: a correctness baseline, not a fast
kernel. The performance rules above say what to change.

```c
void kernel_linear_s8(const int8_t *input, const int8_t *weight, const int32_t *bias, int8_t *output, int M, int K, int N, int input_offset, int filter_offset, int output_offset, int output_multiplier, int output_shift, int activation_min, int activation_max) {
    if (M <= 0 || N <= 0) {
        return;
    }

    /* A100 cores only: elsewhere the same encoding computes another tile. */
    unsigned long vlenb;
    __asm__ volatile("csrr %0, vlenb" : "=r"(vlenb));
    if (vlenb != 128) {
        __builtin_trap();
    }

    int m_tiles = (M + 7) / 8;
    int k_slabs = K > 0 ? (K + 15) / 16 : 1;
    size_t panel_bytes = (size_t)k_slabs * 128;          /* one 8-row panel */
    int8_t packed_input[(size_t)m_tiles * panel_bytes];   /* all of A */
    int8_t packed_weight[panel_bytes];                    /* one B panel */
    int32_t input_sums[(size_t)m_tiles * 8];
    int32_t tile[64];                                     /* one 8x8 result */

    /* 1. Pack ALL of A once. Row tile mt, slab ks: rows 8*mt..8*mt+7,
     *    columns 16*ks..16*ks+15, row i at bytes 16*i..16*i+15.
     *    Zero outside M x K. Row sums feed the offset correction. */
    for (int mt = 0; mt < m_tiles; mt++) {
        for (int i = 0; i < 8; i++) {
            int row = mt * 8 + i;
            int32_t sum = 0;
            for (int ks = 0; ks < k_slabs; ks++) {
                int8_t *dst = packed_input + ((size_t)mt * k_slabs + ks) * 128 + i * 16;
                for (int q = 0; q < 16; q++) {
                    int k = ks * 16 + q;
                    int8_t v = (row < M && k < K) ? input[(size_t)row * K + k] : 0;
                    dst[q] = v;
                    sum += v;
                }
            }
            input_sums[mt * 8 + i] = sum;
        }
    }

    for (int n0 = 0; n0 < N; n0 += 8) {
        /* 2. Pack one B panel ONCE: 8 output columns. weight is [N][K], so
         *    output column n is the contiguous row weight[n*K ..] -- already
         *    B's [j][k] layout, no transpose. */
        int32_t weight_sums[8];
        for (int j = 0; j < 8; j++) {
            int col = n0 + j;
            int32_t sum = 0;
            for (int ks = 0; ks < k_slabs; ks++) {
                int8_t *dst = packed_weight + (size_t)ks * 128 + j * 16;
                for (int q = 0; q < 16; q++) {
                    int k = ks * 16 + q;
                    int8_t v = (col < N && k < K) ? weight[(size_t)col * K + k] : 0;
                    dst[q] = v;
                    sum += v;
                }
            }
            weight_sums[j] = sum;
        }

        for (int mt = 0; mt < m_tiles; mt++) {
            /* 3. One 8x8 result tile: k_slabs vmadots into v8/v9, then store
             *    the 64 int32 results row-major into tile[]. */
            const int8_t *pa = packed_input + (size_t)mt * panel_bytes;
            const int8_t *pb = packed_weight;
            long slabs = k_slabs;
            __asm__ volatile(
                "vsetvli t0, zero, e8, m1, ta, ma\n\t"   /* vl=128: one tile per load */
                "vmv.v.i v8, 0\n\t"
                "vmv.v.i v9, 0\n\t"
                "1:\n\t"
                "vle8.v v0, (%[pa])\n\t"                 /* A tile, 8x16 */
                "vle8.v v4, (%[pb])\n\t"                 /* B tile, 8x16 */
                ".insn r 0x2b, 3, 0x71, x8, x0, x4\n\t"  /* v8,v9 += A * B^T */
                "addi %[pa], %[pa], 128\n\t"
                "addi %[pb], %[pb], 128\n\t"
                "addi %[ks], %[ks], -1\n\t"
                "bnez %[ks], 1b\n\t"
                "vsetvli t0, zero, e32, m2, ta, ma\n\t"  /* vl=64 int32 over v8,v9 */
                "vse32.v v8, (%[out])\n\t"
                : [pa] "+r"(pa), [pb] "+r"(pb), [ks] "+r"(slabs)
                : [out] "r"(tile)
                : "t0", "memory", "v0", "v4", "v8", "v9");

            /* 4. Scalar tail, exactly the reference's integer arithmetic. */
            int m0 = mt * 8;
            for (int i = 0; i < 8 && m0 + i < M; i++) {
                for (int j = 0; j < 8 && n0 + j < N; j++) {
                    int32_t acc = tile[i * 8 + j];
                    if (bias != NULL) {
                        acc += bias[n0 + j];
                    }
                    if (input_offset != 0 || filter_offset != 0) {
                        int64_t correction =
                            (int64_t)input_offset * weight_sums[j] +
                            (int64_t)filter_offset * input_sums[mt * 8 + i] +
                            (int64_t)K * input_offset * filter_offset;
                        acc = (int32_t)((int64_t)acc + correction);
                    }
                    int64_t prod = (int64_t)acc * (int64_t)output_multiplier;
                    prod = (prod + (1LL << 30)) >> 31;
                    int32_t s = (int32_t)prod;
                    if (output_shift > 0) {
                        s = (int32_t)(((int64_t)s + ((int64_t)1 << (output_shift - 1))) >> output_shift);
                    } else if (output_shift < 0) {
                        s = s << (-output_shift);
                    }
                    s += output_offset;
                    if (s < activation_min) s = activation_min;
                    if (s > activation_max) s = activation_max;
                    output[(size_t)(m0 + i) * N + (n0 + j)] = (int8_t)s;
                }
            }
        }
    }
}
```
