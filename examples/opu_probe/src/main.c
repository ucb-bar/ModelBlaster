/* Saturn-OPU FireSim bitstream ISA probe.
 *
 * Tests which RVV / Saturn-OPU custom OP-V opcodes the actual
 * FireSimGemminiAndOPUShuttleConfig FPGA implementation accepts vs.
 * traps as illegal instruction. Necessary because:
 *
 *  - The Saturn-OPU spike extension (libsaturn_opu.so) decodes all
 *    of these functionally, so spike verify passes for kernels that
 *    crash on FPGA.
 *  - In v10 v1 the silu kernel got past vsetvli e8/m1 SET form +
 *    vle8 + vadd before trapping on vluxei8 — so we have direct
 *    evidence vsetvli SET works for e8/m1.
 *  - In v11 attempts #1 and #2 the conv2d kernel trapped at vsetvli
 *    e32/m4 (probe form with rs1=zero, then SET form with rs1=
 *    SIZE_MAX). Both were illegal-instruction on the bitstream.
 *
 * This probe runs candidate opcodes one at a time, printing a
 * marker before and after each. If an opcode traps, the post-marker
 * never lands and the offending test is the last "START" without a
 * matching "OK" in the log.
 *
 * Tests roughly in order from already-proven to most speculative.
 */
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* Saturn OPU custom opcodes via inline asm — same encodings as
 * cores/saturn_opu/include/saturn_opu.h. We don't include the header
 * to keep this binary self-contained. */
#define VMV_VR(vd, rs1, ms2) \
    asm volatile(".insn r 0x57, 0x6, 0x5d, " vd ", %0, " ms2 \
                 : : "r"(rs1));
#define OPMVINBCAST(md, vs2) \
    asm volatile(".insn r 0x57, 0x6, 0x59, " md ", x0, " vs2);
#define VOPACC(md, vs2, vs1) \
    asm volatile(".insn r 0x57, 0x2, 0x51, " md ", " vs1 ", " vs2);

#define MARK_START(n, desc) \
    printf("OPU_PROBE_%02d: %s START\n", n, desc)
#define MARK_OK(n) \
    printf("OPU_PROBE_%02d: OK\n", n)

/* Inputs sized to keep things tiny — one vector of i8 lanes. */
static int8_t  in_a[64] __attribute__((aligned(64)));
static int8_t  in_b[64] __attribute__((aligned(64)));
static int8_t  out_8[64] __attribute__((aligned(64)));
static int32_t out_32[64] __attribute__((aligned(64)));

int main(void)
{
    printf("=== Saturn-OPU FireSim ISA probe START ===\n");

    for (int i = 0; i < 64; i++) {
        in_a[i] = (int8_t)i;
        in_b[i] = (int8_t)(i + 1);
    }

    /* TEST 01: vsetvli SET, e8/m1, rs1 = small (16).
     *   Already proven by v10 v1's silu kernel (vsetvli with rs1=avl
     *   in K loop ran successfully before vluxei8 trapped). Used here
     *   as a sanity check — if THIS traps we know the FPGA's vector
     *   unit is essentially absent. */
    MARK_START(1, "vsetvli e8/m1 SET rs1=16");
    {
        size_t vl;
        asm volatile("vsetvli %0, %1, e8, m1, ta, ma"
                     : "=r"(vl) : "r"((size_t)16));
        printf("        vl=%lu\n", (unsigned long)vl);
    }
    MARK_OK(1);

    /* TEST 02: vle8.v after the e8/m1 vsetvli.
     *   v10 v1 ran this. Sanity check. */
    MARK_START(2, "vle8.v e8/m1");
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"((size_t)16));
    asm volatile("vle8.v v16, (%0)" : : "r"(in_a));
    asm volatile("vse8.v v16, (%0)" : : "r"(out_8));
    MARK_OK(2);

    /* TEST 03: vsetvli SET, e8/m1, rs1 = SIZE_MAX (-1).
     *   v11 attempt #2 trapped on this for e32/m4. Test whether
     *   e8/m1 with SIZE_MAX is also rejected — implementation may
     *   have a generic range check on rs1 rather than per-vtype. */
    MARK_START(3, "vsetvli e8/m1 SET rs1=SIZE_MAX");
    {
        size_t vl;
        asm volatile("vsetvli %0, %1, e8, m1, ta, ma"
                     : "=r"(vl) : "r"((size_t)-1));
        printf("        vl=%lu\n", (unsigned long)vl);
    }
    MARK_OK(3);

    /* TEST 04: vsetvli probe — rs1=zero, rd != zero, e8/m1.
     *   v11 attempt #1 trapped on this. Test for confirmation. */
    MARK_START(4, "vsetvli e8/m1 PROBE rs1=zero");
    {
        size_t vlmax;
        asm volatile("vsetvli %0, zero, e8, m1, ta, ma" : "=r"(vlmax));
        printf("        vlmax=%lu\n", (unsigned long)vlmax);
    }
    MARK_OK(4);

    /* TEST 05: vsetvli SET, e16/m2, rs1 = 16.
     *   Needed by vwmul_vv_i16m2 intrinsic. Unknown. */
    MARK_START(5, "vsetvli e16/m2 SET rs1=16");
    {
        size_t vl;
        asm volatile("vsetvli %0, %1, e16, m2, ta, ma"
                     : "=r"(vl) : "r"((size_t)16));
        printf("        vl=%lu\n", (unsigned long)vl);
    }
    MARK_OK(5);

    /* TEST 06: vsetvli SET, e32/m4, rs1 = 16.
     *   v11 attempt #2 trapped here with rs1=SIZE_MAX. Test with a
     *   small rs1 to see if the issue is the vtype or rs1 range. */
    MARK_START(6, "vsetvli e32/m4 SET rs1=16");
    {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m4, ta, ma"
                     : "=r"(vl) : "r"((size_t)16));
        printf("        vl=%lu\n", (unsigned long)vl);
    }
    MARK_OK(6);

    /* TEST 07: vsetvli SET, e32/m4 with rd=zero (no return).
     *   May be more conservative — implementation may skip rd write
     *   path. Tests whether e32/m4 vtype itself is rejected. */
    MARK_START(7, "vsetvli e32/m4 SET rd=zero");
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"((size_t)16));
    MARK_OK(7);

    /* TEST 08: change-vtype-only form (rd=zero, rs1=zero).
     *   Standard "switch vtype without changing vl" emitted by gcc
     *   between widening ops. */
    MARK_START(8, "vsetvli e32/m4 CHANGE-VTYPE-ONLY");
    asm volatile("vsetvli zero, zero, e32, m4, ta, ma");
    MARK_OK(8);

    /* TEST 09: widening multiply — vwmul.vv. e8/m1 inputs, e16/m2
     *   output. Required by im2col_rvv_reduce kernel. */
    MARK_START(9, "vwmul_vv e8/m1 -> e16/m2");
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"((size_t)16));
    asm volatile("vle8.v v16, (%0)" : : "r"(in_a));
    asm volatile("vle8.v v17, (%0)" : : "r"(in_b));
    asm volatile("vwmul.vv v18, v16, v17");  /* v18-v19 hold i16 result */
    MARK_OK(9);

    /* TEST 10: widening accumulate vwadd.wv. e16/m2 + e16/m2 -> e32/m4. */
    MARK_START(10, "vwadd_wv e16/m2 -> e32/m4");
    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"((size_t)16));
    asm volatile("vmv.v.i v8, 0");  /* i16/m2 zero accumulator */
    asm volatile("vwadd.wv v20, v8, v18");  /* widen-add into i32/m4 */
    MARK_OK(10);

    /* TEST 11: reduction vredsum.vs. */
    MARK_START(11, "vredsum_vs e32/m4");
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"((size_t)16));
    asm volatile("vmv.s.x v24, x0");  /* init reduction scalar to 0 */
    asm volatile("vredsum.vs v25, v20, v24");
    MARK_OK(11);

    /* TEST 12: OPMVINBCAST (Saturn OPU custom).
     *   m1 += seed broadcast across rows. Used by linear/conv OPU
     *   outerprod kernel. */
    MARK_START(12, "OPMVINBCAST m1, v0");
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"((size_t)16));
    asm volatile("vmv.v.i v0, 0");
    OPMVINBCAST("x1", "x0");  /* m1 = "x1", v0 = "x0" per saturn_opu.h hack */
    MARK_OK(12);

    /* TEST 13: VOPACC (Saturn OPU custom).
     *   m1 += vs2 outer-product vs1. */
    MARK_START(13, "VOPACC m1, v18, v16");
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"((size_t)16));
    asm volatile("vle8.v v16, (%0)" : : "r"(in_a));
    asm volatile("vle8.v v18, (%0)" : : "r"(in_b));
    VOPACC("x1", "x18", "x16");  /* m1 += v18 ⊗ v16 */
    MARK_OK(13);

    /* TEST 14: VMV_VR (Saturn OPU custom).
     *   Drain matrix row into a vector register. */
    MARK_START(14, "VMV_VR v0, x0, m1");
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"((size_t)16));
    VMV_VR("x0", (uintptr_t)0, "x1");  /* vd=v0, row=0, ms2=m1 */
    asm volatile("vse32.v v0, (%0)" : : "r"(out_32));
    MARK_OK(14);

    printf("=== Saturn-OPU FireSim ISA probe END (all tests passed) ===\n");
    return 0;
}
