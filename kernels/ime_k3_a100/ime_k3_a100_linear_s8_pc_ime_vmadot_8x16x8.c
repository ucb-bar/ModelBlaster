/* source: curated */
/* algorithm: ime_vmadot_8x16x8 */
/* accuracy_class: bit_exact */
/* linear_s8_pc: the per-channel ("per-filter") twin of ime_k3_a100_linear_s8_ime_vmadot_8x16x8.c
 * (k3-work TASK-019, 2026-11-17), which stays unchanged. The same kernel, its static names
 * prefixed ime_lpc_ (so both can share a kernels.c), with output_multiplier[n] /
 * output_shift[n] per output channel n in the requantize step (ime_lpc_requant, RQPC; the
 * packed path's 8x8 C tiles get per-lane lists like their bias). A shift outside [-30, 30]
 * in any channel, or nonzero input/filter offsets, take the scalar path. Written without
 * the board; the board check is a byte compare against rvv_linear_s8_pc_direct.c (k3-work
 * experiments/vint-pc-kernels-20261117). The per-layer kernel's notes follow. */
/* origin: hand-written for the SpaceMiT K3 A100 (VLEN 1024). IME int8 tile
 *   8(M) x 16(K) x 8(N): `.insn r 0x2b,3,0x71, vd, vs1=A, vs2=B` =>
 *   C(i32 8x8) += A . B^T,  A[i*16+k] (128 B, e8 m1), B[j*16+k] (128 B),
 *   C row-major over (vd,vd+1), vd even, LMUL 1.  artifacts/k3_probes/ime/summary.md.
 *
 * MEASURED FACTS THAT SHAPE THE KERNEL (A100, CPU 8)
 *   - weights arrive from L2: merely streaming a 32 KB weight matrix is ~8 us
 *     (~4 B/ns); the old rvv kernel is ~30 us, compute/reduction bound
 *     (vredsum m4 37 ns, vwadd.wv m4 chains, ~70 ns per (n, 128-K chunk)).
 *   - vluxei64 gather of a B tile costs ~300 ns: useless; vwredsum ~80 ns.
 *   - vmadot throughput 0.7 ns, latency 4.5 ns (4 independent accumulators).
 *
 * FOUR PATHS, chosen per call (see kernel_linear_s8):
 *   K == 16        "direct":  16-byte weight rows ARE B-tile rows, so 8 rows are
 *                  128 contiguous bytes: one e8m4 load = 4 B tiles, no repack,
 *                  A = the (<=8) input rows. Any M.
 *   M < IME_LIN_SMALL_M, K%128==0 "diag":   a weight row is K/16 chunks of 16 B. A tile = 8
 *                  contiguous 16-B chunks of x, B tile = the same 8 chunks of one
 *                  weight row; only the diagonal C[i][i] = x_i . w_i is useful,
 *                  and sum_i C[i][i] over all tiles of the row is the dot product.
 *                  Everything is contiguous e8m4/m2/m1 loads (no repack), two
 *                  weight rows share each x load, and the 8 diagonal entries are
 *                  extracted with two vand masks + one vredsum e32m1 per row (no
 *                  m4 reduction, no per-(m,n) chain). IME computes 1/8 of the tile
 *                  usefully but that is irrelevant: the kernel is load-bound.
 *   M < IME_LIN_SMALL_M otherwise "rvv":    vectorized over K with 4 weight rows interleaved,
 *                  vwmul + vwredsum, vector requant (no vredsum m4, no scalar tail).
 *   M >= 8         "packed":  weights packed once per N-block of 32, A packed per
 *                  M-block, 2(m) x 4(n) vmadot register blocking (8 accumulators).
 *
 * REQUANT is vectorized and bit-exact: Q0.31 rounding multiply = vsmul vxrm=RNU
 * ((a*b + 2^30) >> 31), shift = vssra RNU ((x + 2^(s-1)) >> s, no int32 overflow),
 * clamp vmax/vmin, truncating vncvt = the C int8 cast. Only deviation from the
 * oracle: acc == mult == INT32_MIN saturates in vsmul (C wraps); unreachable for a
 * positive multiplier. Nonzero input/filter offsets or |output_shift| > 30 take the
 * scalar fallback (never happens in this tree: quantization is symmetric).
 */

#ifndef IME_LIN_SMALL_M
#define IME_LIN_SMALL_M 8   /* M below this: diag/rvv/k64 small-M paths; else packed IME */
#endif

static void ime_lpc_scalar(const int8_t *input, const int8_t *weight, const int32_t *bias,
                           int8_t *output, int M, int K, int N, int input_offset,
                           int filter_offset, int output_offset, const int32_t *output_multiplier,
                           const int32_t *output_shift, int activation_min, int activation_max) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t acc = bias ? bias[n] : 0;
            for (int k = 0; k < K; k++) {
                int32_t in_v = (int32_t)input[m * K + k] + input_offset;
                int32_t w_v = (int32_t)weight[n * K + k] + filter_offset;
                acc += in_v * w_v;
            }
            const int32_t sh = output_shift[n];
            int64_t prod = (int64_t)acc * (int64_t)output_multiplier[n];
            prod = (prod + (1LL << 30)) >> 31;
            int32_t scaled = (int32_t)prod;
            if (sh > 62) {
                scaled = 0;
            } else if (sh > 0) {
                scaled = (int32_t)(((int64_t)scaled + ((int64_t)1 << (sh - 1))) >> sh);
            } else if (sh < 0) {
                scaled = scaled << (-sh);
            }
            scaled += output_offset;
            if (scaled < activation_min) scaled = activation_min;
            if (scaled > activation_max) scaled = activation_max;
            output[m * N + n] = (int8_t)scaled;
        }
    }
}

/* n <= 128 byte copy / zero with one vector load+store (libc memcpy is a call + byte loop). */
/* Opaque zero: stops GCC hoisting a zero vector out of a loop and spilling it (vs1r.v) around inline asm. */
static volatile int ime_pc_zero_src;

static inline void ime_pc_cpv(int8_t *d, const int8_t *s, size_t n) {
    size_t vl = __riscv_vsetvl_e8m1(n);
    __riscv_vse8_v_i8m1(d, __riscv_vle8_v_i8m1(s, vl), vl);
}
static inline void ime_pc_zv(int8_t *d, size_t n) {
    size_t vl = __riscv_vsetvl_e8m1(n);
    __riscv_vse8_v_i8m1(d, __riscv_vmv_v_x_i8m1((int8_t)ime_pc_zero_src, vl), vl);
}

/* Vector requantize of n contiguous int32 accumulators (+bias) to int8, with one
 * multiplier and shift per element (mult[i], shift[i]: acc[i]'s output channel). Per lane
 * the shift is applied as vssra by max(shift, 0) and vsll by max(-shift, 0): one of the
 * two is the per-layer step's shift, the other 0 (a no-op). Shifts in [-30, 30]. */
static __attribute__((noinline)) void ime_lpc_requant(const int32_t *acc, const int32_t *bias, int8_t *out, int n,
                            int output_offset, const int32_t *mult, const int32_t *shift,
                            int activation_min, int activation_max) {
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)n);
        vint32m1_t v = __riscv_vle32_v_i32m1(acc, vl);
        if (bias) v = __riscv_vadd_vv_i32m1(v, __riscv_vle32_v_i32m1(bias, vl), vl);
        v = __riscv_vsmul_vv_i32m1(v, __riscv_vle32_v_i32m1(mult, vl), __RISCV_VXRM_RNU, vl);
        {
            vint32m1_t sh = __riscv_vle32_v_i32m1(shift, vl);
            vuint32m1_t rs = __riscv_vreinterpret_v_i32m1_u32m1(__riscv_vmax_vx_i32m1(sh, 0, vl));
            vuint32m1_t ls = __riscv_vreinterpret_v_i32m1_u32m1(
                __riscv_vmax_vx_i32m1(__riscv_vneg_v_i32m1(sh, vl), 0, vl));
            v = __riscv_vssra_vv_i32m1(v, rs, __RISCV_VXRM_RNU, vl);
            v = __riscv_vsll_vv_i32m1(v, ls, vl);
        }
        v = __riscv_vadd_vx_i32m1(v, output_offset, vl);
        v = __riscv_vmax_vx_i32m1(v, activation_min, vl);
        v = __riscv_vmin_vx_i32m1(v, activation_max, vl);
        vint16mf2_t h = __riscv_vncvt_x_x_w_i16mf2(v, vl);
        vint8mf4_t b = __riscv_vncvt_x_x_w_i8mf4(h, vl);
        __riscv_vse8_v_i8mf4(out, b, vl);
        acc += vl; if (bias) bias += vl; mult += vl; shift += vl; out += vl; n -= (int)vl;
    }
}

/* PM, PS: the multipliers and shifts of the n accumulators' channels. */
#define RQPC(acc, bias, out, n, PM, PS) \
    ime_lpc_requant((acc), (bias), (out), (n), output_offset, (PM), (PS), activation_min, activation_max)

static const int32_t ime_lpc_masks[64] __attribute__((aligned(16))) = {
    -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
    0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0,
    0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0,
    0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1};

/* ---- path "diag" ------------------------------------------------------- */
/* Diagonal extraction masks: lanes {0,9,18,27} of vd and {4,13,22,31} of vd+1. */
#define D_PRO                                                                         \
    "vsetvli t0, %[n32], e32, m1, ta, ma\n\t"                                        \
    "vle32.v v12, (%[mk])\n\t"                                                       \
    "addi t1, %[mk], 128\n\t"                                                        \
    "vle32.v v13, (t1)\n\t"                                                          \
    "vmv.s.x v14, zero\n\t"                                                          \
    "vsetvli t0, %[n128], e8, m1, ta, ma\n\t"
#define D_ZERO                                                                        \
    "vmv.v.i v16, 0\n\t" "vmv.v.i v17, 0\n\t" "vmv.v.i v24, 0\n\t" "vmv.v.i v25, 0\n\t"
#define D_RED                                                                         \
    "vsetvli t0, %[n32], e32, m1, ta, ma\n\t"                                        \
    "vand.vv v16, v16, v12\n\t" "vand.vv v17, v17, v13\n\t" "vadd.vv v16, v16, v17\n\t" \
    "vand.vv v24, v24, v12\n\t" "vand.vv v25, v25, v13\n\t" "vadd.vv v24, v24, v25\n\t" \
    "vredsum.vs v15, v16, v14\n\t"                                                   \
    "vredsum.vs v11, v24, v14\n\t"                                                   \
    "vmv.x.s t1, v15\n\t" "vmv.x.s t2, v11\n\t"                                      \
    "sw t1, 0(%[sm])\n\t" "sw t2, 4(%[sm])\n\t"                                      \
    "addi %[sm], %[sm], 8\n\t"                                                       \
    "vsetvli t0, %[n128], e8, m1, ta, ma\n\t"
#define D_STEP                                                                        \
    "add %[wa], %[wa], %[st]\n\t" "add %[wb], %[wb], %[st]\n\t"                      \
    "addi %[np], %[np], -1\n\t" "bnez %[np], 1b\n\t"
#define D_CLOB                                                                        \
    : "t0", "t1", "t2", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", \
      "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v24", "v25"
#define D_IN                                                                          \
    : [n128] "r"(n128), [n256] "r"(n256), [n512] "r"(n512), [n32] "r"(n32), [mk] "r"(masks), \
      [x] "r"(x), [st] "r"(st)
#define D_OUT : [wa] "+r"(wa), [wb] "+r"(wb), [np] "+r"(np), [sm] "+r"(sums)

/* P row pairs (rows adjacent, stride K); T = K/128 in {1,2,4}; writes 2P sums. */
static void ime_lpc_diag_pairs(const int8_t *x, const int8_t *wa, int K, int T, int P,
                               const int32_t *masks, int32_t *sums) {
    size_t n128 = 128, n256 = 256, n512 = 512, n32 = 32;
    const int8_t *wb = wa + K;
    size_t st = (size_t)2 * K;
    size_t np = (size_t)P;
    if (T == 1) {
        __asm__ volatile(D_PRO "vle8.v v0, (%[x])\n\t" "1:\n\t"
            "vle8.v v4, (%[wa])\n\t" "vle8.v v8, (%[wb])\n\t" D_ZERO
            ".insn r 0x2b, 3, 0x71, x16, x0, x4\n\t"
            ".insn r 0x2b, 3, 0x71, x24, x0, x8\n\t"
            D_RED D_STEP D_OUT D_IN D_CLOB);
    } else if (T == 2) {
        /* rows a,b are adjacent (stride 256 B): one e8m4 load = v4,v5 (row a), v6,v7 (row b) */
        __asm__ volatile(D_PRO "vsetvli t0, %[n256], e8, m2, ta, ma\n\t" "vle8.v v0, (%[x])\n\t" "1:\n\t"
            "vsetvli t0, %[n512], e8, m4, ta, ma\n\t"
            "vle8.v v4, (%[wa])\n\t"
            "vsetvli t0, %[n128], e8, m1, ta, ma\n\t" D_ZERO
            ".insn r 0x2b, 3, 0x71, x16, x0, x4\n\t"
            ".insn r 0x2b, 3, 0x71, x24, x0, x6\n\t"
            ".insn r 0x2b, 3, 0x71, x16, x1, x5\n\t"
            ".insn r 0x2b, 3, 0x71, x24, x1, x7\n\t"
            D_RED D_STEP D_OUT D_IN D_CLOB);
    } else {
        __asm__ volatile(D_PRO "vsetvli t0, %[n512], e8, m4, ta, ma\n\t" "vle8.v v0, (%[x])\n\t" "1:\n\t"
            "vsetvli t0, %[n512], e8, m4, ta, ma\n\t"
            "vle8.v v4, (%[wa])\n\t" "vle8.v v8, (%[wb])\n\t"
            "vsetvli t0, %[n128], e8, m1, ta, ma\n\t" D_ZERO
            ".insn r 0x2b, 3, 0x71, x16, x0, x4\n\t"
            ".insn r 0x2b, 3, 0x71, x24, x0, x8\n\t"
            ".insn r 0x2b, 3, 0x71, x16, x1, x5\n\t"
            ".insn r 0x2b, 3, 0x71, x24, x1, x9\n\t"
            ".insn r 0x2b, 3, 0x71, x16, x2, x6\n\t"
            ".insn r 0x2b, 3, 0x71, x24, x2, x10\n\t"
            ".insn r 0x2b, 3, 0x71, x16, x3, x7\n\t"
            ".insn r 0x2b, 3, 0x71, x24, x3, x11\n\t"
            D_RED D_STEP D_OUT D_IN D_CLOB);
    }
}

/* K == 128 (T=1): four adjacent rows = 512 contiguous bytes = one e8m4 load. */
static void ime_lpc_diag_quads(const int8_t *x, const int8_t *w, int Q, const int32_t *masks,
                               int32_t *sums) {
    size_t n128 = 128, n512 = 512, n32 = 32;
    size_t q = (size_t)Q;
#define QEX(lo, hi, dst) \
    "vand.vv " lo ", " lo ", v12\n\t" "vand.vv " hi ", " hi ", v13\n\t" \
    "vadd.vv " lo ", " lo ", " hi "\n\t" "vredsum.vs " dst ", " lo ", v14\n\t"
    __asm__ volatile(
        D_PRO
        "vle8.v v0, (%[x])\n\t"
        "1:\n\t"
        "vsetvli t0, %[n512], e8, m4, ta, ma\n\t"
        "vle8.v v4, (%[w])\n\t"
        "vsetvli t0, %[n128], e8, m1, ta, ma\n\t"
        "vmv.v.i v16, 0\n\t" "vmv.v.i v17, 0\n\t" "vmv.v.i v18, 0\n\t" "vmv.v.i v19, 0\n\t"
        "vmv.v.i v24, 0\n\t" "vmv.v.i v25, 0\n\t" "vmv.v.i v26, 0\n\t" "vmv.v.i v27, 0\n\t"
        ".insn r 0x2b, 3, 0x71, x16, x0, x4\n\t"
        ".insn r 0x2b, 3, 0x71, x18, x0, x5\n\t"
        ".insn r 0x2b, 3, 0x71, x24, x0, x6\n\t"
        ".insn r 0x2b, 3, 0x71, x26, x0, x7\n\t"
        "vsetvli t0, %[n32], e32, m1, ta, ma\n\t"
        QEX("v16", "v17", "v15")
        QEX("v18", "v19", "v20")
        QEX("v24", "v25", "v21")
        QEX("v26", "v27", "v22")
        "vmv.x.s t1, v15\n\t" "vmv.x.s t2, v20\n\t"
        "sw t1, 0(%[sm])\n\t" "sw t2, 4(%[sm])\n\t"
        "vmv.x.s t1, v21\n\t" "vmv.x.s t2, v22\n\t"
        "sw t1, 8(%[sm])\n\t" "sw t2, 12(%[sm])\n\t"
        "addi %[sm], %[sm], 16\n\t" "addi %[w], %[w], 512\n\t"
        "addi %[np], %[np], -1\n\t"
        "vsetvli t0, %[n128], e8, m1, ta, ma\n\t"
        "bnez %[np], 1b\n\t"
        : [w] "+r"(w), [np] "+r"(q), [sm] "+r"(sums)
        : [n128] "r"(n128), [n256] "r"(n128), [n512] "r"(n512), [n32] "r"(n32), [mk] "r"(masks),
          [x] "r"(x)
        : "t0", "t1", "t2", "memory", "v0", "v4", "v5", "v6", "v7", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v24", "v25", "v26", "v27");
#undef QEX
}

/* One weight row, any T (= K/128): groups of 4 tiles via e8m4 loads, 4 independent
 * accumulator pairs, then the same diagonal extraction. */
static int32_t ime_lpc_dot1(const int8_t *x, const int8_t *w, int T, const int32_t *masks) {
    size_t n128 = 128, n512 = 512, n32 = 32;
    size_t n4 = (size_t)(T >> 2), n1 = (size_t)(T & 3);
    int32_t out;
    __asm__ volatile(
        "vsetvli t0, %[n32], e32, m1, ta, ma\n\t"
        "vle32.v v12, (%[mk])\n\t"
        "addi t1, %[mk], 128\n\t"
        "vle32.v v13, (t1)\n\t"
        "vmv.s.x v14, zero\n\t"
        "vsetvli t0, %[n128], e8, m1, ta, ma\n\t"
        "vmv.v.i v16, 0\n\t" "vmv.v.i v17, 0\n\t" "vmv.v.i v18, 0\n\t" "vmv.v.i v19, 0\n\t"
        "vmv.v.i v20, 0\n\t" "vmv.v.i v21, 0\n\t" "vmv.v.i v22, 0\n\t" "vmv.v.i v23, 0\n\t"
        "beqz %[n4], 2f\n\t"
        "1:\n\t"
        "vsetvli t0, %[n512], e8, m4, ta, ma\n\t"
        "vle8.v v0, (%[x])\n\t"
        "vle8.v v4, (%[w])\n\t"
        "vsetvli t0, %[n128], e8, m1, ta, ma\n\t"
        ".insn r 0x2b, 3, 0x71, x16, x0, x4\n\t"
        ".insn r 0x2b, 3, 0x71, x18, x1, x5\n\t"
        ".insn r 0x2b, 3, 0x71, x20, x2, x6\n\t"
        ".insn r 0x2b, 3, 0x71, x22, x3, x7\n\t"
        "addi %[x], %[x], 512\n\t" "addi %[w], %[w], 512\n\t"
        "addi %[n4], %[n4], -1\n\t" "bnez %[n4], 1b\n\t"
        "2:\n\t"
        "beqz %[n1], 4f\n\t"
        "3:\n\t"
        "vle8.v v0, (%[x])\n\t"
        "vle8.v v4, (%[w])\n\t"
        ".insn r 0x2b, 3, 0x71, x16, x0, x4\n\t"
        "addi %[x], %[x], 128\n\t" "addi %[w], %[w], 128\n\t"
        "addi %[n1], %[n1], -1\n\t" "bnez %[n1], 3b\n\t"
        "4:\n\t"
        "vsetvli t0, %[n32], e32, m1, ta, ma\n\t"
        "vadd.vv v16, v16, v18\n\t" "vadd.vv v20, v20, v22\n\t" "vadd.vv v16, v16, v20\n\t"
        "vadd.vv v17, v17, v19\n\t" "vadd.vv v21, v21, v23\n\t" "vadd.vv v17, v17, v21\n\t"
        "vand.vv v16, v16, v12\n\t" "vand.vv v17, v17, v13\n\t" "vadd.vv v16, v16, v17\n\t"
        "vredsum.vs v15, v16, v14\n\t"
        "vmv.x.s %[o], v15\n\t"
        : [x] "+r"(x), [w] "+r"(w), [n4] "+r"(n4), [n1] "+r"(n1), [o] "=r"(out)
        : [n128] "r"(n128), [n512] "r"(n512), [n32] "r"(n32), [mk] "r"(masks)
        : "t0", "t1", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v12", "v13",
          "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
    return out;
}

/* Software-pipelined single-row dot for T multiple of 8: while one 4-tile group is
 * multiplied the next group's x/w e8m4 loads are already in flight (the A100's
 * L2->core latency, not instruction count, is what a cold row costs).
 * `iters` = T/8; accumulates into v16/v17 .. v22/v23 and extracts the diagonal. */
static int32_t ime_lpc_dot_pipe(const int8_t *x, const int8_t *w, int iters, const int32_t *masks) {
    size_t n128 = 128, n512 = 512, n32 = 32;
    size_t it = (size_t)iters;
    int32_t out;
    __asm__ volatile(
        "vsetvli t0, %[n32], e32, m1, ta, ma\n\t"
        "vle32.v v12, (%[mk])\n\t"
        "addi t1, %[mk], 128\n\t"
        "vle32.v v13, (t1)\n\t"
        "vmv.s.x v14, zero\n\t"
        "vsetvli t0, %[n128], e8, m1, ta, ma\n\t"
        "vmv.v.i v16, 0\n\t" "vmv.v.i v17, 0\n\t" "vmv.v.i v18, 0\n\t" "vmv.v.i v19, 0\n\t"
        "vmv.v.i v20, 0\n\t" "vmv.v.i v21, 0\n\t" "vmv.v.i v22, 0\n\t" "vmv.v.i v23, 0\n\t"
        "vsetvli t0, %[n512], e8, m4, ta, ma\n\t"
        "vle8.v v24, (%[x])\n\t"
        "vle8.v v28, (%[w])\n\t"
        "1:\n\t"
        "addi %[x], %[x], 512\n\t" "addi %[w], %[w], 512\n\t"
        "vle8.v v0, (%[x])\n\t"
        "vle8.v v4, (%[w])\n\t"
        "vsetvli t0, %[n128], e8, m1, ta, ma\n\t"
        ".insn r 0x2b, 3, 0x71, x16, x24, x28\n\t"
        ".insn r 0x2b, 3, 0x71, x18, x25, x29\n\t"
        ".insn r 0x2b, 3, 0x71, x20, x26, x30\n\t"
        ".insn r 0x2b, 3, 0x71, x22, x27, x31\n\t"
        "addi %[it], %[it], -1\n\t"
        "addi %[x], %[x], 512\n\t" "addi %[w], %[w], 512\n\t"
        "beqz %[it], 2f\n\t"
        "vsetvli t0, %[n512], e8, m4, ta, ma\n\t"
        "vle8.v v24, (%[x])\n\t"
        "vle8.v v28, (%[w])\n\t"
        "vsetvli t0, %[n128], e8, m1, ta, ma\n\t"
        ".insn r 0x2b, 3, 0x71, x16, x0, x4\n\t"
        ".insn r 0x2b, 3, 0x71, x18, x1, x5\n\t"
        ".insn r 0x2b, 3, 0x71, x20, x2, x6\n\t"
        ".insn r 0x2b, 3, 0x71, x22, x3, x7\n\t"
        "vsetvli t0, %[n512], e8, m4, ta, ma\n\t"
        "j 1b\n\t"
        "2:\n\t"
        ".insn r 0x2b, 3, 0x71, x16, x0, x4\n\t"
        ".insn r 0x2b, 3, 0x71, x18, x1, x5\n\t"
        ".insn r 0x2b, 3, 0x71, x20, x2, x6\n\t"
        ".insn r 0x2b, 3, 0x71, x22, x3, x7\n\t"
        "vsetvli t0, %[n32], e32, m1, ta, ma\n\t"
        "vadd.vv v16, v16, v18\n\t" "vadd.vv v20, v20, v22\n\t" "vadd.vv v16, v16, v20\n\t"
        "vadd.vv v17, v17, v19\n\t" "vadd.vv v21, v21, v23\n\t" "vadd.vv v17, v17, v21\n\t"
        "vand.vv v16, v16, v12\n\t" "vand.vv v17, v17, v13\n\t" "vadd.vv v16, v16, v17\n\t"
        "vredsum.vs v15, v16, v14\n\t"
        "vmv.x.s %[o], v15\n\t"
        : [x] "+r"(x), [w] "+r"(w), [it] "+r"(it), [o] "=r"(out)
        : [n128] "r"(n128), [n512] "r"(n512), [n32] "r"(n32), [mk] "r"(masks)
        : "t0", "t1", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v12", "v13",
          "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
          "v26", "v27", "v28", "v29", "v30", "v31");
    return out;
}

/* ---- path "k64": K == 64 (S=4 chunks per row) ------------------------------ */
/* A tile = [x(64 B) | 0], B tile = 2 contiguous rows. C[s][4q+s] = x_s . w_{row q, chunk s}
 * so the two row sums are the lane sets {0,9,18,27} (masks v12) and {4,13,22,31} (v13)
 * of vd. Writes 2 sums per tile. */
static void ime_lpc_k64(const int8_t *atile, const int8_t *w, int tiles, const int32_t *masks,
                        int32_t *sums) {
    size_t n128 = 128, n32 = 32;
    size_t nt = (size_t)tiles;
    __asm__ volatile(
        "vsetvli t0, %[n32], e32, m1, ta, ma\n\t"
        "vle32.v v12, (%[mk])\n\t"
        "addi t1, %[mk], 128\n\t"
        "vle32.v v13, (t1)\n\t"
        "vmv.s.x v14, zero\n\t"
        "vsetvli t0, %[n128], e8, m1, ta, ma\n\t"
        "vle8.v v1, (%[a])\n\t"
        "1:\n\t"
        "vle8.v v4, (%[w])\n\t"
        "vmv.v.i v16, 0\n\t" "vmv.v.i v17, 0\n\t"
        ".insn r 0x2b, 3, 0x71, x16, x1, x4\n\t"
        "vsetvli t0, %[n32], e32, m1, ta, ma\n\t"
        "vand.vv v8, v16, v12\n\t" "vand.vv v10, v16, v13\n\t"
        "vredsum.vs v9, v8, v14\n\t" "vredsum.vs v11, v10, v14\n\t"
        "vmv.x.s t1, v9\n\t" "vmv.x.s t2, v11\n\t"
        "sw t1, 0(%[sm])\n\t" "sw t2, 4(%[sm])\n\t"
        "vsetvli t0, %[n128], e8, m1, ta, ma\n\t"
        "addi %[w], %[w], 128\n\t" "addi %[sm], %[sm], 8\n\t"
        "addi %[nt], %[nt], -1\n\t" "bnez %[nt], 1b\n\t"
        : [w] "+r"(w), [nt] "+r"(nt), [sm] "+r"(sums)
        : [n128] "r"(n128), [n32] "r"(n32), [mk] "r"(masks), [a] "r"(atile)
        : "t0", "t1", "t2", "memory", "v1", "v4", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
          "v16", "v17");
}

/* ---- path "direct" (K == 16): 4 contiguous B tiles per call -------------- */
static void ime_lpc_direct4(const int8_t *a_tile, const int8_t *w512, int32_t *c0, int32_t *c1,
                            int32_t *c2, int32_t *c3, size_t vl1, size_t vl2) {
    size_t n128 = 128, n512 = 512;
    __asm__ volatile(
        "vsetvli t0, %[n128], e8, m1, ta, ma\n\t"
        "vle8.v v0, (%[a])\n\t"
        "vmv.v.i v8, 0\n\t" "vmv.v.i v9, 0\n\t" "vmv.v.i v10, 0\n\t" "vmv.v.i v11, 0\n\t"
        "vmv.v.i v12, 0\n\t" "vmv.v.i v13, 0\n\t" "vmv.v.i v14, 0\n\t" "vmv.v.i v15, 0\n\t"
        "vsetvli t0, %[n512], e8, m4, ta, ma\n\t"
        "vle8.v v16, (%[w])\n\t"
        "vsetvli t0, %[n128], e8, m1, ta, ma\n\t"
        ".insn r 0x2b, 3, 0x71, x8, x0, x16\n\t"
        ".insn r 0x2b, 3, 0x71, x10, x0, x17\n\t"
        ".insn r 0x2b, 3, 0x71, x12, x0, x18\n\t"
        ".insn r 0x2b, 3, 0x71, x14, x0, x19\n\t"
        "vsetvli t0, %[vl1], e32, m1, ta, ma\n\t"
        "vse32.v v8, (%[c0])\n\t" "vse32.v v10, (%[c1])\n\t"
        "vse32.v v12, (%[c2])\n\t" "vse32.v v14, (%[c3])\n\t"
        "vsetvli t0, %[vl2], e32, m1, ta, ma\n\t"
        "addi t1, %[c0], 128\n\t" "vse32.v v9, (t1)\n\t"
        "addi t1, %[c1], 128\n\t" "vse32.v v11, (t1)\n\t"
        "addi t1, %[c2], 128\n\t" "vse32.v v13, (t1)\n\t"
        "addi t1, %[c3], 128\n\t" "vse32.v v15, (t1)\n\t"
        :
        : [a] "r"(a_tile), [w] "r"(w512), [c0] "r"(c0), [c1] "r"(c1), [c2] "r"(c2), [c3] "r"(c3),
          [n128] "r"(n128), [n512] "r"(n512), [vl1] "r"(vl1), [vl2] "r"(vl2)
        : "t0", "t1", "memory", "v0", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19");
}

/* ---- path "packed": 2(m) x 4(n) tiles over `slabs` k-slabs --------------- */
/* a0,a1: packed A tiles [slab][128]; bp: packed B [slab][4 tiles][128].
 * c: out, 8 pairs stored as [mi][t][64] ints. */
static void ime_lpc_packed(const int8_t *a0, const int8_t *a1, const int8_t *bp, int slabs,
                           int32_t *c) {
    size_t n128 = 128, n512 = 512, n32 = 32;
    __asm__ volatile(
        "vsetvli t0, %[n128], e8, m1, ta, ma\n\t"
        "vmv.v.i v8, 0\n\t" "vmv.v.i v9, 0\n\t" "vmv.v.i v10, 0\n\t" "vmv.v.i v11, 0\n\t"
        "vmv.v.i v12, 0\n\t" "vmv.v.i v13, 0\n\t" "vmv.v.i v14, 0\n\t" "vmv.v.i v15, 0\n\t"
        "vmv.v.i v24, 0\n\t" "vmv.v.i v25, 0\n\t" "vmv.v.i v26, 0\n\t" "vmv.v.i v27, 0\n\t"
        "vmv.v.i v28, 0\n\t" "vmv.v.i v29, 0\n\t" "vmv.v.i v30, 0\n\t" "vmv.v.i v31, 0\n\t"
        "1:\n\t"
        "vsetvli t0, %[n512], e8, m4, ta, ma\n\t"
        "vle8.v v16, (%[bp])\n\t"
        "vsetvli t0, %[n128], e8, m1, ta, ma\n\t"
        "vle8.v v0, (%[a0])\n\t"
        "vle8.v v1, (%[a1])\n\t"
        ".insn r 0x2b, 3, 0x71, x8, x0, x16\n\t"
        ".insn r 0x2b, 3, 0x71, x10, x0, x17\n\t"
        ".insn r 0x2b, 3, 0x71, x12, x0, x18\n\t"
        ".insn r 0x2b, 3, 0x71, x14, x0, x19\n\t"
        ".insn r 0x2b, 3, 0x71, x24, x1, x16\n\t"
        ".insn r 0x2b, 3, 0x71, x26, x1, x17\n\t"
        ".insn r 0x2b, 3, 0x71, x28, x1, x18\n\t"
        ".insn r 0x2b, 3, 0x71, x30, x1, x19\n\t"
        "addi %[a0], %[a0], 128\n\t" "addi %[a1], %[a1], 128\n\t" "addi %[bp], %[bp], 512\n\t"
        "addi %[ks], %[ks], -1\n\t"
        "bnez %[ks], 1b\n\t"
        "vsetvli t0, %[n32], e32, m1, ta, ma\n\t"
        "addi t1, %[c], 128\n\t"
        "vse32.v v8, (%[c])\n\t"      "vse32.v v9, (t1)\n\t"
        "addi t2, %[c], 256\n\t" "addi t1, %[c], 384\n\t"
        "vse32.v v10, (t2)\n\t"       "vse32.v v11, (t1)\n\t"
        "addi t2, %[c], 512\n\t" "addi t1, %[c], 640\n\t"
        "vse32.v v12, (t2)\n\t"       "vse32.v v13, (t1)\n\t"
        "addi t2, %[c], 768\n\t" "addi t1, %[c], 896\n\t"
        "vse32.v v14, (t2)\n\t"       "vse32.v v15, (t1)\n\t"
        "addi t2, %[c], 1024\n\t" "addi t1, %[c], 1152\n\t"
        "vse32.v v24, (t2)\n\t"       "vse32.v v25, (t1)\n\t"
        "addi t2, %[c], 1280\n\t" "addi t1, %[c], 1408\n\t"
        "vse32.v v26, (t2)\n\t"       "vse32.v v27, (t1)\n\t"
        "addi t2, %[c], 1536\n\t" "addi t1, %[c], 1664\n\t"
        "vse32.v v28, (t2)\n\t"       "vse32.v v29, (t1)\n\t"
        "addi t2, %[c], 1792\n\t" "addi t1, %[c], 1920\n\t"
        "vse32.v v30, (t2)\n\t"       "vse32.v v31, (t1)\n\t"
        : [a0] "+r"(a0), [a1] "+r"(a1), [bp] "+r"(bp), [ks] "+r"(slabs)
        : [c] "r"(c), [n128] "r"(n128), [n512] "r"(n512), [n32] "r"(n32)
        : "t0", "t1", "t2", "memory", "v0", "v1", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
          "v15", "v16", "v17", "v18", "v19", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
          "v31");
}

/* ---- path "rvv": small M, any K -------------------------------------------- */
static void ime_lpc_rvv_small(const int8_t *input, const int8_t *weight, const int32_t *bias,
                              int8_t *output, int M, int K, int N, int output_offset,
                              const int32_t *output_multiplier, const int32_t *output_shift, int activation_min,
                              int activation_max) {
    int32_t sums[32];
    for (int m = 0; m < M; m++) {
        const int8_t *x = input + (size_t)m * K;
        for (int n0 = 0; n0 < N; n0 += 32) {
            int nb = N - n0 < 32 ? N - n0 : 32;
            int j = 0;
            for (; j + 4 <= nb; j += 4) {
                const int8_t *w0 = weight + (size_t)(n0 + j) * K;
                const int8_t *w1 = w0 + K, *w2 = w1 + K, *w3 = w2 + K;
                vint32m1_t a0 = __riscv_vmv_s_x_i32m1(ime_pc_zero_src, 1), a1 = a0, a2 = a0, a3 = a0;
                for (int k = 0; k < K;) {
                    size_t vl = __riscv_vsetvl_e8m1((size_t)(K - k));
                    vint8m1_t vx = __riscv_vle8_v_i8m1(x + k, vl);
                    a0 = __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vwmul_vv_i16m2(vx, __riscv_vle8_v_i8m1(w0 + k, vl), vl), a0, vl);
                    a1 = __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vwmul_vv_i16m2(vx, __riscv_vle8_v_i8m1(w1 + k, vl), vl), a1, vl);
                    a2 = __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vwmul_vv_i16m2(vx, __riscv_vle8_v_i8m1(w2 + k, vl), vl), a2, vl);
                    a3 = __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vwmul_vv_i16m2(vx, __riscv_vle8_v_i8m1(w3 + k, vl), vl), a3, vl);
                    k += (int)vl;
                }
                sums[j] = __riscv_vmv_x_s_i32m1_i32(a0); sums[j + 1] = __riscv_vmv_x_s_i32m1_i32(a1);
                sums[j + 2] = __riscv_vmv_x_s_i32m1_i32(a2); sums[j + 3] = __riscv_vmv_x_s_i32m1_i32(a3);
            }
            for (; j < nb; j++) {
                const int8_t *w0 = weight + (size_t)(n0 + j) * K;
                vint32m1_t a0 = __riscv_vmv_s_x_i32m1(ime_pc_zero_src, 1);
                for (int k = 0; k < K;) {
                    size_t vl = __riscv_vsetvl_e8m1((size_t)(K - k));
                    a0 = __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vwmul_vv_i16m2(__riscv_vle8_v_i8m1(x + k, vl), __riscv_vle8_v_i8m1(w0 + k, vl), vl), a0, vl);
                    k += (int)vl;
                }
                sums[j] = __riscv_vmv_x_s_i32m1_i32(a0);
            }
            RQPC(sums, bias ? bias + n0 : (const int32_t *)0, output + (size_t)m * N + n0, nb, output_multiplier + n0, output_shift + n0);
        }
    }
}

void kernel_linear_s8_pc(const int8_t *input, const int8_t *weight, const int32_t *bias, int8_t *output, int M, int K, int N, int input_offset, int filter_offset, int output_offset, const int32_t *output_multiplier, const int32_t *output_shift, int activation_min, int activation_max) {
    if (M <= 0 || N <= 0) return;
    int shifts_ok = 1;      /* every channel's shift in [-30, 30], as the per-layer kernel asks */
    for (int n = 0; n < N; n++)
        if (output_shift[n] > 30 || output_shift[n] < -30) { shifts_ok = 0; break; }
    if (input_offset != 0 || filter_offset != 0 || !shifts_ok || K <= 0) {
        ime_lpc_scalar(input, weight, bias, output, M, K, N, input_offset, filter_offset,
                       output_offset, output_multiplier, output_shift, activation_min,
                       activation_max);
        return;
    }

    /* ---------------- path "direct": K == 16 ------------------------------- */
    if (K == 16) {
        int8_t atile[128] __attribute__((aligned(16)));
        int8_t wtmp[512] __attribute__((aligned(16)));
        int32_t ct[4][64] __attribute__((aligned(16)));
        int32_t sums[32] __attribute__((aligned(16)));
        int8_t obuf[64];
        for (int m0 = 0; m0 < M; m0 += 8) {
            int mr = M - m0 < 8 ? M - m0 : 8;
            __builtin_memcpy(atile, input + (size_t)m0 * 16, (size_t)mr * 16);
            if (mr < 8) __builtin_memset(atile + mr * 16, 0, (size_t)(8 - mr) * 16);
            size_t cnt = (size_t)mr * 8;
            size_t vl1 = cnt < 32 ? cnt : 32u, vl2 = cnt > 32 ? cnt - 32 : 0u;
            for (int n0 = 0; n0 < N; n0 += 32) {
                int nb = N - n0 < 32 ? N - n0 : 32;
                const int8_t *w = weight + (size_t)n0 * 16;
                if (nb < 32) {                      /* tail: never read past `weight` */
                    __builtin_memset(wtmp, 0, sizeof(wtmp));
                    __builtin_memcpy(wtmp, w, (size_t)nb * 16);
                    w = wtmp;
                }
                ime_lpc_direct4(atile, w, ct[0], ct[1], ct[2], ct[3], vl1, vl2);
                int nt = (nb + 7) >> 3;
                if (mr == 1) {
                    for (int t = 0; t < nt; t++) __builtin_memcpy(sums + t * 8, ct[t], 32);
                    RQPC(sums, bias ? bias + n0 : (const int32_t *)0, output + (size_t)m0 * N + n0, nb, output_multiplier + n0, output_shift + n0);
                } else {
                    for (int t = 0; t < nt; t++) {
                        int nr = nb - t * 8 < 8 ? nb - t * 8 : 8;
                        for (int i = 0; i < mr; i++)
                            RQPC(ct[t] + i * 8, bias ? bias + n0 + t * 8 : (const int32_t *)0,
                                 output + (size_t)(m0 + i) * N + n0 + t * 8, nr,
                                 output_multiplier + n0 + t * 8, output_shift + n0 + t * 8);
                    }
                }
            }
            (void)obuf;
        }
        return;
    }

    /* ---------------- path "k64": K == 64, small M ------------------------- */
    if (K == 64 && M < IME_LIN_SMALL_M) {
        const int32_t *masks = ime_lpc_masks;
        int8_t atile[128] __attribute__((aligned(16)));
        int8_t wtmp[128] __attribute__((aligned(16)));
        int32_t sums[32 + 2] __attribute__((aligned(16)));
        for (int m = 0; m < M; m++) {
            __builtin_memcpy(atile, input + (size_t)m * 64, 64);
            __builtin_memset(atile + 64, 0, 64);
            for (int n0 = 0; n0 < N; n0 += 32) {
                int nb = N - n0 < 32 ? N - n0 : 32;
                int full = nb >> 1;
                if (full) ime_lpc_k64(atile, weight + (size_t)n0 * 64, full, masks, sums);
                if (nb & 1) {
                    __builtin_memset(wtmp, 0, sizeof(wtmp));
                    __builtin_memcpy(wtmp, weight + (size_t)(n0 + nb - 1) * 64, 64);
                    ime_lpc_k64(atile, wtmp, 1, masks, sums + 2 * full);
                }
                RQPC(sums, bias ? bias + n0 : (const int32_t *)0, output + (size_t)m * N + n0, nb, output_multiplier + n0, output_shift + n0);
            }
        }
        return;
    }

    /* ---------------- path "diag": small M, K % 128 == 0 ------------------- */
    if (M < IME_LIN_SMALL_M && (K & 127) == 0) {
        const int32_t *masks = ime_lpc_masks;
        int T = K >> 7;
        int32_t sums[34] __attribute__((aligned(16)));
        int8_t pair_tmp[2 * 512];
        for (int m = 0; m < M; m++) {
            const int8_t *x = input + (size_t)m * K;
            for (int n0 = 0; n0 < N; n0 += 32) {
                int nb = N - n0 < 32 ? N - n0 : 32;
                const int8_t *w = weight + (size_t)n0 * K;
                if (T == 1) {
                    int Q = nb >> 2;
                    if (Q) ime_lpc_diag_quads(x, w, Q, masks, sums);
                    if (nb & 3) {
                        int r = nb & 3;
                        int32_t qs[4];
                        __builtin_memset(pair_tmp, 0, 512);
                        __builtin_memcpy(pair_tmp, w + (size_t)(Q * 4) * K, (size_t)r * K);
                        ime_lpc_diag_quads(x, pair_tmp, 1, masks, qs);
                        for (int i = 0; i < r; i++) sums[Q * 4 + i] = qs[i];
                    }
                } else if (T == 2 || T == 4) {
                    int P = nb >> 1;
                    if (P) ime_lpc_diag_pairs(x, w, K, T, P, masks, sums);
                    if (nb & 1) {
                        __builtin_memcpy(pair_tmp, w + (size_t)(nb - 1) * K, (size_t)K);
                        __builtin_memcpy(pair_tmp + K, pair_tmp, (size_t)K);
                        ime_lpc_diag_pairs(x, pair_tmp, K, T, 1, masks, sums + 2 * P);
                    }
                } else {
                    int Tp = T >> 3, Tr = T & 7;
                    for (int j = 0; j < nb; j++) {
                        const int8_t *wj = w + (size_t)j * K;
                        int32_t acc = 0;
                        if (Tp) acc += ime_lpc_dot_pipe(x, wj, Tp, masks);
                        if (Tr) acc += ime_lpc_dot1(x + Tp * 1024, wj + Tp * 1024, Tr, masks);
                        sums[j] = acc;
                    }
                }
                if (nb == 1) {      /* scalar tail is cheaper than 20 vector ops for one value */
                    const int32_t q_mult = output_multiplier[n0], q_shift = output_shift[n0];
                    int32_t acc = sums[0] + (bias ? bias[n0] : 0);
                    int64_t prod = ((int64_t)acc * (int64_t)q_mult + (1LL << 30)) >> 31;
                    int32_t sc = (int32_t)prod;
                    if (q_shift > 0) sc = (int32_t)(((int64_t)sc + ((int64_t)1 << (q_shift - 1))) >> q_shift);
                    else if (q_shift < 0) sc = sc << (-q_shift);
                    sc += output_offset;
                    if (sc < activation_min) sc = activation_min;
                    if (sc > activation_max) sc = activation_max;
                    output[(size_t)m * N + n0] = (int8_t)sc;
                } else {
                    RQPC(sums, bias ? bias + n0 : (const int32_t *)0, output + (size_t)m * N + n0, nb, output_multiplier + n0, output_shift + n0);
                }
            }
        }
        return;
    }

    /* ---------------- path "rvv": small M, other K -------------------------- */
    if (M < IME_LIN_SMALL_M) {
        ime_lpc_rvv_small(input, weight, bias, output, M, K, N, output_offset, output_multiplier,
                          output_shift, activation_min, activation_max);
        return;
    }

    /* ---------------- path "packed": M >= 8 --------------------------------- */
    {
        int slabs = (K + 15) >> 4;
        size_t slab_stride = 128;
        /* A block: rows per block limited to ~96 KB of packed A */
        int mt_blk = (int)((96u * 1024u) / ((size_t)slabs * slab_stride));
        if (mt_blk < 2) mt_blk = 2;
        mt_blk &= ~1;
        int mtiles = (M + 7) >> 3;
        if (mt_blk > mtiles + (mtiles & 1)) mt_blk = mtiles + (mtiles & 1);
        int8_t *apk = (int8_t *)__builtin_alloca((size_t)mt_blk * slabs * slab_stride + 16);
        int8_t *bp = (int8_t *)__builtin_alloca((size_t)slabs * 512 + 16);
        int32_t *ct = (int32_t *)__builtin_alloca(2 * 4 * 64 * sizeof(int32_t) + 16);
        int32_t *brep = (int32_t *)__builtin_alloca(4 * 64 * sizeof(int32_t) + 16);
        /* the multiplier and shift of each lane of the 4 C tiles, like brep */
        int32_t prep_m[4 * 64] __attribute__((aligned(16))), prep_s[4 * 64] __attribute__((aligned(16)));
        int8_t *obuf = (int8_t *)__builtin_alloca(64 + 16);
        apk = (int8_t *)(((uintptr_t)apk + 15) & ~(uintptr_t)15);
        bp = (int8_t *)(((uintptr_t)bp + 15) & ~(uintptr_t)15);
        ct = (int32_t *)(((uintptr_t)ct + 15) & ~(uintptr_t)15);
        brep = (int32_t *)(((uintptr_t)brep + 15) & ~(uintptr_t)15);
        int8_t zero128[128] __attribute__((aligned(16)));
        __builtin_memset(zero128, 0, sizeof(zero128));

        for (int mt0 = 0; mt0 < mtiles; mt0 += mt_blk) {
            int nmt = mtiles - mt0 < mt_blk ? mtiles - mt0 : mt_blk;
            /* pack A: apk[mt][slab][i][16] */
            for (int mt = 0; mt < nmt; mt++) {
                for (int s = 0; s < slabs; s++) {
                    int8_t *dst = apk + ((size_t)mt * slabs + s) * 128;
                    for (int i = 0; i < 8; i++) {
                        int row = (mt0 + mt) * 8 + i;
                        int kk = s * 16;
                        int take = K - kk < 16 ? K - kk : 16;
                        if (row < M) {
                            ime_pc_cpv(dst + i * 16, input + (size_t)row * K + kk, (size_t)take);
                            if (take < 16) ime_pc_zv(dst + i * 16 + take, (size_t)(16 - take));
                        } else {
                            ime_pc_zv(dst + i * 16, 16);
                        }
                    }
                }
            }
            for (int n0 = 0; n0 < N; n0 += 32) {
                /* pack B: bp[slab][t][j][16] for 4 tiles; rows >= N are zero */
                for (int s = 0; s < slabs; s++) {
                    int kk = s * 16;
                    int take = K - kk < 16 ? K - kk : 16;
                    for (int t = 0; t < 4; t++) {
                        int8_t *dst = bp + ((size_t)s * 4 + t) * 128;
                        for (int j = 0; j < 8; j++) {
                            int col = n0 + t * 8 + j;
                            if (col < N) {
                                ime_pc_cpv(dst + j * 16, weight + (size_t)col * K + kk, (size_t)take);
                                if (take < 16) ime_pc_zv(dst + j * 16 + take, (size_t)(16 - take));
                            } else {
                                ime_pc_zv(dst + j * 16, 16);
                            }
                        }
                    }
                }
                if (bias) {
                    for (int t = 0; t < 4; t++)
                        for (int j = 0; j < 8; j++) {
                            int col = n0 + t * 8 + j;
                            int32_t b = col < N ? bias[col] : 0;
                            for (int i = 0; i < 8; i++) brep[t * 64 + i * 8 + j] = b;
                        }
                }
                for (int t = 0; t < 4; t++)
                    for (int j = 0; j < 8; j++) {
                        int col = n0 + t * 8 + j;
                        int32_t pm = col < N ? output_multiplier[col] : 0;
                        int32_t ps = col < N ? output_shift[col] : 0;
                        for (int i = 0; i < 8; i++) {
                            prep_m[t * 64 + i * 8 + j] = pm;
                            prep_s[t * 64 + i * 8 + j] = ps;
                        }
                    }
                int nt = (N - n0 + 7) >> 3; if (nt > 4) nt = 4;
                for (int mt = 0; mt < nmt; mt += 2) {
                    const int8_t *a0 = apk + (size_t)mt * slabs * 128;
                    int two = mt + 1 < nmt;
                    const int8_t *a1 = two ? a0 + (size_t)slabs * 128 : a0;
                    ime_lpc_packed(a0, a1, bp, slabs, ct);
                    for (int u = 0; u < (two ? 2 : 1); u++) {
                        int m0 = (mt0 + mt + u) * 8;
                        int mr = M - m0 < 8 ? M - m0 : 8;
                        for (int t = 0; t < nt; t++) {
                            int nr = N - n0 - t * 8 < 8 ? N - n0 - t * 8 : 8;
                            RQPC(ct + (u * 4 + t) * 64, bias ? brep + t * 64 : (const int32_t *)0, obuf, 64,
                                 prep_m + t * 64, prep_s + t * 64);
                            for (int i = 0; i < mr; i++)
                                ime_pc_cpv(output + (size_t)(m0 + i) * N + n0 + t * 8, obuf + i * 8, (size_t)nr);
                        }
                    }
                }
            }
        }
        (void)zero128;
    }
}
