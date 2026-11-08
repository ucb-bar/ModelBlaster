/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: hand-written. Folded integer batchnorm (pipeline/fold_quant_consts.py):
 *         per channel a = x*M[c] + BP[c] (BP = B + 2^(S-1)), v = round_half_away
 *         ((x*M + B) / 2^S), clamp. Element (o, c, i) is at ((o*C + c)*inner + i),
 *         so NHWC (inner == 1) and NCHW (inner == H*W) share this file.
 *
 *   NHWC. Channel strips outer, pixels inner: one strip's M, BP and S (320 bytes
 *   for 32 channels) are loaded into vector registers ONCE per strip, not once
 *   per pixel. On a vector unit whose loads and stores go straight to DRAM
 *   (no L1, no L2) the bytes a kernel moves are its cost; reloading the
 *   constants per pixel measured ~12 cycles/element on such a core, holding
 *   them ~2 cycles/element.
 *
 *   ROUNDING. a = acc + rnd with acc = x*M + B. round_half_away(acc / 2^S) is
 *   (a + (acc < 0 ? -1 : 0)) >> S (derivation in rvv_add_q_s8_direct.c), and
 *   acc < 0 <=> a < rnd. vnclip with vxrm = rdn is the arithmetic shift,
 *   saturating to int16, and a second vnclip by 0 saturates to int8; the clamp
 *   to [lo, hi] (inside int8) then equals the reference. When lo >= 0 the -1
 *   term is not needed: for acc < 0 the reference gives v <= 0, which clamps to
 *   lo, and floor(a / 2^S) <= 0 clamps to lo as well.
 *   fold_quant_consts guarantees 128*|M| + |B| + rnd < 2^31. */
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

void kernel_batchnorm2d_q_s8(const int8_t *input, const int32_t *M,
                             const int32_t *BP, const int16_t *S, int8_t *output,
                             int outer, int C, int inner,
                             int activation_min, int activation_max) {
    const int32_t lo = activation_min < -128 ? -128 : (activation_min > 127 ? 127 : activation_min);
    const int32_t hi = activation_max < -128 ? -128 : (activation_max > 127 ? 127 : activation_max);
    if (inner == 1 && lo >= 0) {
        const size_t vlmax = __riscv_vsetvlmax_e32m8();
        for (int c0 = 0; c0 < C; c0 += (int)vlmax) {
            const size_t vl = __riscv_vsetvl_e32m8((size_t)(C - c0));
            const vint32m8_t vm = __riscv_vle32_v_i32m8(M + c0, vl);
            const vint32m8_t vb = __riscv_vle32_v_i32m8(BP + c0, vl);
            const vuint16m4_t sh = __riscv_vreinterpret_v_i16m4_u16m4(__riscv_vle16_v_i16m4(S + c0, vl));
            const int8_t *ip = input + c0;
            int8_t *op = output + c0;
            for (int p = 0; p < outer; p++, ip += C, op += C) {
                vint32m8_t a = __riscv_vsext_vf4_i32m8(__riscv_vle8_v_i8m2(ip, vl), vl);
                a = __riscv_vmadd_vv_i32m8(a, vm, vb, vl);           /* x*M + (B + rnd) */
                vint16m4_t v16 = __riscv_vnclip_wv_i16m4(a, sh, __RISCV_VXRM_RDN, vl);
                vint8m2_t v8 = __riscv_vnclip_wx_i8m2(v16, 0, __RISCV_VXRM_RDN, vl);
                v8 = __riscv_vmax_vx_i8m2(v8, (int8_t)lo, vl);
                v8 = __riscv_vmin_vx_i8m2(v8, (int8_t)hi, vl);
                __riscv_vse8_v_i8m2(op, v8, vl);
            }
        }
        return;
    }
    if (inner == 1) {
        {
            /* Clamp below zero needs the -1 term, so the per-channel rnd as well: at
             * LMUL = 8 that is five register groups and the constants spill, so this
             * (general, unused by DroNet) case runs 16-channel strips at LMUL = 4. */
            const size_t vlmax4 = __riscv_vsetvlmax_e32m4();
            for (int c0 = 0; c0 < C; c0 += (int)vlmax4) {
                const size_t vl = __riscv_vsetvl_e32m4((size_t)(C - c0));
                const vint32m4_t vm = __riscv_vle32_v_i32m4(M + c0, vl);
                const vint32m4_t vb = __riscv_vle32_v_i32m4(BP + c0, vl);
                const vuint16m2_t sh = __riscv_vreinterpret_v_i16m2_u16m2(__riscv_vle16_v_i16m2(S + c0, vl));
                const vint32m4_t vr = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vsll_vv_u32m4(
                    __riscv_vmv_v_x_u32m4(1, vl),
                    __riscv_vsub_vx_u32m4(__riscv_vzext_vf2_u32m4(sh, vl), 1, vl), vl));
                const int8_t *ip = input + c0;
                int8_t *op = output + c0;
                for (int p = 0; p < outer; p++, ip += C, op += C) {
                    vint32m4_t a = __riscv_vsext_vf4_i32m4(__riscv_vle8_v_i8m1(ip, vl), vl);
                    a = __riscv_vmadd_vv_i32m4(a, vm, vb, vl);
                    vbool8_t neg = __riscv_vmslt_vv_i32m4_b8(a, vr, vl);
                    a = __riscv_vsub_vx_i32m4_mu(neg, a, a, 1, vl);
                    vint16m2_t v16 = __riscv_vnclip_wv_i16m2(a, sh, __RISCV_VXRM_RDN, vl);
                    vint8m1_t v8 = __riscv_vnclip_wx_i8m1(v16, 0, __RISCV_VXRM_RDN, vl);
                    v8 = __riscv_vmax_vx_i8m1(v8, (int8_t)lo, vl);
                    v8 = __riscv_vmin_vx_i8m1(v8, (int8_t)hi, vl);
                    __riscv_vse8_v_i8m1(op, v8, vl);
                }
            }
        }
        return;
    }
    /* NCHW: one channel plane at a time, scalar constants. */
    for (int o = 0; o < outer; o++) {
        for (int c = 0; c < C; c++) {
            const int32_t m = M[c], bp = BP[c];
            const int s = S[c];
            const int32_t rnd = (int32_t)1 << (s - 1);
            const size_t base = ((size_t)o * C + c) * (size_t)inner;
            size_t vl;
            for (int i = 0; i < inner; i += (int)vl) {
                vl = __riscv_vsetvl_e8m2((size_t)(inner - i));
                vint32m8_t a = __riscv_vsext_vf4_i32m8(__riscv_vle8_v_i8m2(input + base + i, vl), vl);
                a = __riscv_vadd_vx_i32m8(__riscv_vmul_vx_i32m8(a, m, vl), bp, vl);
                if (lo < 0) {
                    vbool4_t neg = __riscv_vmslt_vx_i32m8_b4(a, rnd, vl);
                    a = __riscv_vsub_vx_i32m8_mu(neg, a, a, 1, vl);
                }
                vint16m4_t v16 = __riscv_vnclip_wx_i16m4(a, (size_t)s, __RISCV_VXRM_RDN, vl);
                vint8m2_t v8 = __riscv_vnclip_wx_i8m2(v16, 0, __RISCV_VXRM_RDN, vl);
                v8 = __riscv_vmax_vx_i8m2(v8, (int8_t)lo, vl);
                v8 = __riscv_vmin_vx_i8m2(v8, (int8_t)hi, vl);
                __riscv_vse8_v_i8m2(output + base + i, v8, vl);
            }
        }
    }
}
