/* source: curated */
/* algorithm: zvt_tile_f32 */
/* accuracy_class: numeric_drift */
/* origin: hand-written Zvt (draft RISC-V vector-matrix tile extension)
 *   fp32 linear. Semantics: riscv-isa-sim 72958551 riscv/insns/vtfmm_tvv.h.
 *
 * ENGINE. With mtype widen=1 and vtype e32, vtfmm.tvv is a rank-1 update
 * (tk must be 1):
 *     acc[m][n] = f32_add(acc[m][n], f32_mul(A[m], B[n]))
 * -- a separately rounded multiply and add, in k order. That is exactly
 * the reference loop `acc = bias; acc += in*w;` *without* FMA contraction,
 * so the kernel is bit-exact against a reference compiled with
 * -ffp-contract=off (what -std=c17, i.e. CONFIG_STD_C17, implies). A
 * reference built with contraction (GNU C default) uses fmadd and will
 * differ by rounding.
 *
 * LAYOUT. A = input column k over m (stride 4*K bytes), B = weight
 * column k over n (stride 4*K bytes): strided vlse32 straight into v0 / v8.
 * The accumulator (tile 0) is seeded with the bias row (vtle32 of
 * bias[n0..], once per tile row) so the add order matches the reference.
 * Results are vtse32'd straight into the output rows -- no scalar tail.
 *
 * SIZE-AGNOSTIC: T = granted msettm/msettn (min(TE, LMUL*VLEN/32)),
 * capped at MB_ZVT_TMAX. Whole tile = one asm statement.
 */

#ifndef MB_ZVT_TMAX
#define MB_ZVT_TMAX 128
#endif

void kernel_linear(const float *input, const float *weight, const float *bias,
                   float *output, int M, int K, int N) {
    if (M <= 0 || N <= 0) {
        return;
    }

    const unsigned long vtype = ZVT_VTYPE_E32;
    const unsigned long mtype = ZVT_MTYPE(1, 1, 0); /* widen=1, tk=1 */
    const long stride = (long)K * 4;
    const long out_stride = (long)N * 4;

    unsigned long T;
    {
        unsigned long want = MB_ZVT_TMAX, gm, gn, tmp;
        __asm__ volatile(
            "vsetvl %[tmp], x0, %[vt]\n\t"
            ZVT_S_MSETMTYPE("%[mt]", "%[vt]")
            ZVT_S_MSETTM("%[gm]", "%[w]")
            ZVT_S_MSETTN("%[gn]", "%[w]")
            : [gm] "=&r"(gm), [gn] "=&r"(gn), [tmp] "=&r"(tmp)
            : [mt] "r"(mtype), [vt] "r"(vtype), [w] "r"(want)
            : "memory");
        T = gm < gn ? gm : gn;
    }

    for (int m0 = 0; m0 < M; m0 += (int)T) {
        unsigned long mr = (unsigned long)((M - m0) < (int)T ? (M - m0) : (int)T);

        for (int n0 = 0; n0 < N; n0 += (int)T) {
            unsigned long nc = (unsigned long)((N - n0) < (int)T ? (N - n0) : (int)T);
            const float *pa = input + (size_t)m0 * (size_t)K;
            const float *pb = weight + (size_t)n0 * (size_t)K;
            const float *pbias = bias != NULL ? bias + n0 : NULL;
            float *po = output + (size_t)m0 * (size_t)N + (size_t)n0;
            long ks = K;
            unsigned long tmp;

            __asm__ volatile(
                "vsetvl %[tmp], x0, %[vt]\n\t"
                ZVT_S_MSETMTYPE("%[mt]", "%[vt]")
                ZVT_S_MSETTM("%[tmp]", "%[mr]")
                ZVT_S_MSETTN("%[tmp]", "%[nc]")
                /* seed: acc rows = bias[n0..n0+nc) (or 0) */
                ZVT_S_VTZERO(0)
                "beqz %[pbias], 3f\n\t"
                "li t1, 0\n"
                "2:\n\t"
                ZVT_S_VTLE32("%[pbias]", "t1")
                "addi t1, t1, 1\n\t"
                "bltu t1, %[mr], 2b\n"
                "3:\n\t"
                "beqz %[ks], 5f\n"
                "4:\n\t"
                ZVT_S_MSETTN("%[tmp]", "%[mr]")
                "vlse32.v v0, (%[pa]), %[st]\n\t"
                ZVT_S_MSETTN("%[tmp]", "%[nc]")
                "vlse32.v v8, (%[pb]), %[st]\n\t"
                ZVT_S_VTFMM(0, 0, 8)
                "addi %[pa], %[pa], 4\n\t"
                "addi %[pb], %[pb], 4\n\t"
                "addi %[ks], %[ks], -1\n\t"
                "bnez %[ks], 4b\n"
                "5:\n\t"
                /* store mr rows of nc floats (vl == nc) */
                "li t1, 0\n"
                "6:\n\t"
                ZVT_S_VTSE32("%[po]", "t1")
                "add %[po], %[po], %[ost]\n\t"
                "addi t1, t1, 1\n\t"
                "bltu t1, %[mr], 6b\n\t"
                : [pa] "+r"(pa), [pb] "+r"(pb), [ks] "+r"(ks), [po] "+r"(po),
                  [tmp] "=&r"(tmp)
                : [mt] "r"(mtype), [vt] "r"(vtype), [mr] "r"(mr), [nc] "r"(nc),
                  [st] "r"(stride), [ost] "r"(out_stride), [pbias] "r"(pbias)
                : "t1", "memory",
                  "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                  "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
        }
    }
}
