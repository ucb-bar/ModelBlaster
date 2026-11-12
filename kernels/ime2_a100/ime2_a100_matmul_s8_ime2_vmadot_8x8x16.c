/* source: curated */
/* algorithm: ime2_vmadot_8x8x16 */
/* accuracy_class: bit_exact */
/* origin: written by Kimi K2.5 (Bedrock moonshotai.kimi-k2.5) through ModelBlaster's
 *   BACKEND=llm flow, attn_block run 20261028-003243 (1 call, first attempt), from the ime2_a100 guide and the
 *   matmul_s8 ime2_vmadot_8x8x16 candidate. Unchanged from the LLM's output.
 *
 * TESTED: instruction emulated in C vs the matmul_s8 reference, 20 cases bit-exact (attn_block
 * shapes, both transpose_b, scale_div 1 and 8, odd sizes);
 * docker/ime2_test/ime2_llm_kernels_emu_test.py matmul <this file>. attn_block end to end on A100
 * cpu 8: max_abs_err=0. Known gap: returns without writing output when K == 0 (the reference
 * writes zeros); no model has K == 0.
 * A100 cores only: aborts unless vlenb == 128. Correctness baseline, not tuned.
 */
/* A100 only: GCC's own short vector code (auto-vectorized loops, vl ~8) is slow
 * on this core (a reloaded-line vle8 costs ~17 cycles vs 2 on the X100), so it is
 * turned off for the A100 kernels alone; a global flag slows the X100 side 11%
 * (TASK-012). Popped at the end of the file. */
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-slp-vectorize")
void kernel_matmul_s8(const int8_t *a, const int8_t *b, int8_t *output, int M, int K, int N, float scale_a, float scale_b, float scale_out, int transpose_b, float scale_div, int activation_min, int activation_max) {
    if (M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    /* A100 cores only: elsewhere the same encoding computes another tile. */
    unsigned long vlenb;
    __asm__ volatile("csrr %0, vlenb" : "=r"(vlenb));
    if (vlenb != 128) {
        __builtin_trap();
    }

    int m_tiles = (M + 7) / 8;
    int k_slabs = (K + 15) / 16;
    size_t panel_bytes = (size_t)k_slabs * 128;          /* one 8-row panel */
    int8_t packed_a[(size_t)m_tiles * panel_bytes];       /* all of A */
    int8_t packed_b[panel_bytes];                         /* one B panel */
    int32_t tile[64];                                     /* one 8x8 result */

    /* 1. Pack ALL of A once. Row tile mt, slab ks: rows 8*mt..8*mt+7,
     *    columns 16*ks..16*ks+15, row i at bytes 16*i..16*i+15.
     *    Zero outside M x K. */
    for (int mt = 0; mt < m_tiles; mt++) {
        for (int i = 0; i < 8; i++) {
            int row = mt * 8 + i;
            for (int ks = 0; ks < k_slabs; ks++) {
                int8_t *dst = packed_a + ((size_t)mt * k_slabs + ks) * 128 + i * 16;
                for (int q = 0; q < 16; q++) {
                    int k = ks * 16 + q;
                    int8_t v = (row < M && k < K) ? a[(size_t)row * K + k] : 0;
                    dst[q] = v;
                }
            }
        }
    }

    /* Precompute float scaling factor once. */
    float total = (scale_a * scale_b) / (scale_out * scale_div);

    for (int n0 = 0; n0 < N; n0 += 8) {
        /* 2. Pack one B panel ONCE: 8 output columns. */
        for (int j = 0; j < 8; j++) {
            int col = n0 + j;
            for (int ks = 0; ks < k_slabs; ks++) {
                int8_t *dst = packed_b + (size_t)ks * 128 + j * 16;
                for (int q = 0; q < 16; q++) {
                    int k = ks * 16 + q;
                    int8_t v = 0;
                    if (col < N && k < K) {
                        if (transpose_b) {
                            /* b is [N][K]: output column j is contiguous row b[j*K ..] */
                            v = b[(size_t)col * K + k];
                        } else {
                            /* b is [K][N]: gather/transpose */
                            v = b[(size_t)k * N + col];
                        }
                    }
                    dst[q] = v;
                }
            }
        }

        for (int mt = 0; mt < m_tiles; mt++) {
            /* 3. One 8x8 result tile: k_slabs vmadots into v8/v9, then store
             *    the 64 int32 results row-major into tile[]. */
            const int8_t *pa = packed_a + (size_t)mt * panel_bytes;
            const int8_t *pb = packed_b;
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
                : "t0", "memory", "vl", "vtype", "v0", "v4", "v8", "v9");

            /* 4. Scalar tail: float requantization, textually the reference's. */
            int m0 = mt * 8;
            for (int i = 0; i < 8 && m0 + i < M; i++) {
                for (int j = 0; j < 8 && n0 + j < N; j++) {
                    int32_t acc = tile[i * 8 + j];
                    int32_t v = (int32_t)roundf((float)acc * total);
                    if (v < activation_min) v = activation_min;
                    if (v > activation_max) v = activation_max;
                    output[(size_t)(m0 + i) * N + (n0 + j)] = (int8_t)v;
                }
            }
        }
    }
}

#pragma GCC pop_options
