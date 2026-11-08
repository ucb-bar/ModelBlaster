/* RISC-V Zvt (vector-matrix tile extension, draft) encodings for kernels.
 *
 * Mirrors zephyr include/zephyr/arch/riscv/zvt.h (ucb-bar/zephyr branch
 * zvt-matrix-context) but is standalone so kernels also build outside
 * Zephyr. Encodings: riscv-isa-sim 72958551 riscv/encoding.h, verified
 * with that spike's spike-dasm. Zvt is a draft -- re-verify if the spike
 * pin moves.
 *
 * No assembler knows Zvt yet (binutils 2.43, LLVM 19), so:
 *   - GPR-operand ops are emitted with `.insn r` (works in GNU as and
 *     clang; the compiler may allocate the registers);
 *   - vector/tile-operand ops are emitted as `.word` with constant
 *     register numbers (GNU as rejects vector regs inside `.insn`).
 *
 * Use these *inside one asm block per tile computation*: any compiler-
 * emitted vsetvl{i} between tile ops clears mtype and makes the next
 * tile instruction illegal.
 */
#ifndef MODELBLASTER_ZVT_INSN_H
#define MODELBLASTER_ZVT_INSN_H

/* asm-string fragments; operands named per GNU inline asm. */
#define ZVT_S_MSETMTYPE(mt, vt) ".insn r 0x57, 7, 0x42, x0, " mt ", " vt "\n\t"
#define ZVT_S_MSETTN(rd, rs)    ".insn r 0x57, 7, 0x44, " rd ", " rs ", x0\n\t"
#define ZVT_S_MSETTM(rd, rs)    ".insn r 0x57, 7, 0x44, " rd ", " rs ", x1\n\t"
#define ZVT_S_MSETTK(rd, rs)    ".insn r 0x57, 7, 0x44, " rd ", " rs ", x2\n\t"
#define ZVT_S_VTLE32(base, tss) ".insn r 0x07, 7, 0x29, x0, " base ", " tss "\n\t"
#define ZVT_S_VTSE32(base, tss) ".insn r 0x27, 7, 0x29, x0, " base ", " tss "\n\t"

#define ZVT_STR_(x) #x
#define ZVT_STR(x) ZVT_STR_(x)
#define ZVT_S_WORD3(match, rd, rs1, rs2) \
    ".word (" ZVT_STR(match) " | ((" ZVT_STR(rd) ") << 7) | ((" \
    ZVT_STR(rs1) ") << 15) | ((" ZVT_STR(rs2) ") << 20))\n\t"

/* acc tile td (td encoded as rd = 2*td); vs2 = A columns, vs1 = B columns */
#define ZVT_S_VTMMS(td, vs2, vs1) ZVT_S_WORD3(0xf20000f7, (td) * 2, vs1, vs2)
#define ZVT_S_VTMMU(td, vs2, vs1) ZVT_S_WORD3(0xf2000077, (td) * 2, vs1, vs2)
#define ZVT_S_VTFMM(td, vs2, vs1) ZVT_S_WORD3(0xf2001077, (td) * 2, vs1, vs2)
#define ZVT_S_VTZERO(td)          ZVT_S_WORD3(0x43e06057, (td) * 2, 0, 0)

/* mtype: [1:0] log2(widen)+1, [7:5] tk, [23:10] tm */
#define ZVT_MTYPE(widen_code, tk, tm) \
    ((unsigned long)(widen_code) | ((unsigned long)(tk) << 5) | \
     ((unsigned long)(tm) << 10))
#define ZVT_WIDEN4 3 /* i8 -> i32 */
#define ZVT_VTYPE_E8      (0UL << 3)
#define ZVT_VTYPE_E32     (2UL << 3)
#define ZVT_VTYPE_ALTFMT  (1UL << 8) /* vtmms: signed x signed */

/* Tile selector for vtle/vtse: tile [30:27], column-major [24], index */
#define ZVT_TSS(tile, colmajor, idx) \
    (((unsigned long)(tile) << 27) | ((unsigned long)(colmajor) << 24) | \
     (unsigned long)(idx))

/* ---- RVV helpers shared by the zvt kernels -------------------------------
 * Plain RVV between tile asm blocks is fine: each tile block re-establishes
 * its own config, and vsetvl never touches tile contents. */
#if defined(__riscv_vector)
#include <riscv_vector.h>
#include <stdint.h>

/* d[i] = s[i*stride], n elements. */
static inline void zvt_vcopy8(int8_t *d, const int8_t *s, long stride, int n)
{
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e8m8((size_t)n);
        vint8m8_t v = stride == 1 ? __riscv_vle8_v_i8m8(s, vl)
                                  : __riscv_vlse8_v_i8m8(s, stride, vl);
        __riscv_vse8_v_i8m8(d, v, vl);
        d += vl; s += (long)vl * stride; n -= (int)vl;
    }
}

static inline void zvt_vzero8(int8_t *d, int n)
{
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e8m8((size_t)n);
        __riscv_vse8_v_i8m8(d, __riscv_vmv_v_x_i8m8(0, vl), vl);
        d += vl; n -= (int)vl;
    }
}

/* Requantize n int32 accumulators to int8, bit-exact with the pipeline's
 * linear_s8 / conv2d_s8 reference tails:
 *   acc' = acc[i] + s1*v1[i] + s2*v2[i] + add      (int32, wrapping)
 *   prod = (int32)(((int64)acc' * mult + 2^30) >> 31)
 *   shift>0: (int32)(((int64)prod + 2^(shift-1)) >> shift); shift<0: prod << -shift
 *   + out_off, clamp [amin, amax], -> int8
 * v1/v2 may be NULL. */
static inline void zvt_requant_s8(int8_t *dst, const int32_t *acc,
                                  const int32_t *v1, int32_t s1,
                                  const int32_t *v2, int32_t s2, int32_t add,
                                  int n, int32_t mult, int shift, int32_t out_off,
                                  int32_t amin, int32_t amax)
{
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e32m4((size_t)n);
        vint32m4_t a = __riscv_vle32_v_i32m4(acc, vl);
        if (v1) a = __riscv_vmacc_vx_i32m4(a, s1, __riscv_vle32_v_i32m4(v1, vl), vl);
        if (v2) a = __riscv_vmacc_vx_i32m4(a, s2, __riscv_vle32_v_i32m4(v2, vl), vl);
        a = __riscv_vadd_vx_i32m4(a, add, vl);
        vint64m8_t p = __riscv_vwmul_vx_i64m8(a, mult, vl);
        p = __riscv_vadd_vx_i64m8(p, (int64_t)1 << 30, vl);
        p = __riscv_vsra_vx_i64m8(p, 31, vl);
        vint32m4_t q = __riscv_vncvt_x_x_w_i32m4(p, vl);           /* (int32)prod */
        if (shift > 0) {
            vint64m8_t w = __riscv_vsext_vf2_i64m8(q, vl);
            w = __riscv_vadd_vx_i64m8(w, (int64_t)1 << (shift - 1), vl);
            w = __riscv_vsra_vx_i64m8(w, (size_t)shift, vl);
            q = __riscv_vncvt_x_x_w_i32m4(w, vl);
        } else if (shift < 0) {
            q = __riscv_vsll_vx_i32m4(q, (size_t)(-shift), vl);
        }
        q = __riscv_vadd_vx_i32m4(q, out_off, vl);
        q = __riscv_vmax_vx_i32m4(q, amin, vl);
        q = __riscv_vmin_vx_i32m4(q, amax, vl);
        vint16m2_t h = __riscv_vncvt_x_x_w_i16m2(q, vl);
        __riscv_vse8_v_i8m1(dst, __riscv_vncvt_x_x_w_i8m1(h, vl), vl);
        dst += vl; acc += vl; n -= (int)vl;
        if (v1) v1 += vl;
        if (v2) v2 += vl;
    }
}

/* int8 dot product over n elements -> int32 (exact: |a*b| <= 2^14). */
static inline int32_t zvt_vdot8(const int8_t *a, const int8_t *b, int n)
{
    vint32m1_t sum = __riscv_vmv_s_x_i32m1(0, 1);
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e8m4((size_t)n);
        vint16m8_t pr = __riscv_vwmul_vv_i16m8(__riscv_vle8_v_i8m4(a, vl),
                                               __riscv_vle8_v_i8m4(b, vl), vl);
        sum = __riscv_vwredsum_vs_i16m8_i32m1(pr, sum, vl);
        a += vl; b += vl; n -= (int)vl;
    }
    return __riscv_vmv_x_s_i32m1_i32(sum);
}
#endif /* __riscv_vector */

#endif
