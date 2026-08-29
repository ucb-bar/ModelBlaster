/* Raw MAC-unit throughput: smt.vmadot vs RVV vwmacc, no memory traffic.
 *
 * This settles the CEILING question that no kernel measurement can. Both loops
 * issue only their MAC instruction on registers already loaded -- no loads, no
 * stores, no packing, no requantize. Whatever comes out is the best either unit
 * can ever do, and the ratio is the most IME could ever win by.
 *
 * MACs per instruction:
 *   smt.vmadot   4x4x8 = 128   (one 4x4 tile accumulating 8 k-steps)
 *   vwmacc.vv    vl=32  =  32  (e8m1 -> i16m2 widening multiply-accumulate)
 *
 * So if both issue at one per cycle, IME wins 4x. If vmadot is internally
 * sequenced over its 8 k-steps it is 16 MACs/cycle against RVV's 32 and LOSES
 * by 2x no matter how well the kernel is written. That is the whole question.
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define ITERS 2000000

int main(void) {
    int8_t A[32], B[32];
    for (int i = 0; i < 32; i++) { A[i] = (int8_t)(i + 1); B[i] = (int8_t)(32 - i); }
    size_t n32 = 32;
    volatile int32_t sink[8];

    /* ---- IME: vmadot, 128 MACs per instruction, 8-way unrolled ---- */
    long it = ITERS;
    double t0 = now_s();
    __asm__ volatile(
        "vsetvli t0, %[n32], e8, m1, ta, ma\n\t"
        "vle8.v v0, (%[pa])\n\t"
        "vle8.v v4, (%[pb])\n\t"
        "vmv.v.i v8, 0\n\t" "vmv.v.i v9, 0\n\t"
        "1:\n\t"
        ".insn r 0x2b, 3, 0x71, x8, x0, x4\n\t"
        ".insn r 0x2b, 3, 0x71, x8, x0, x4\n\t"
        ".insn r 0x2b, 3, 0x71, x8, x0, x4\n\t"
        ".insn r 0x2b, 3, 0x71, x8, x0, x4\n\t"
        ".insn r 0x2b, 3, 0x71, x8, x0, x4\n\t"
        ".insn r 0x2b, 3, 0x71, x8, x0, x4\n\t"
        ".insn r 0x2b, 3, 0x71, x8, x0, x4\n\t"
        ".insn r 0x2b, 3, 0x71, x8, x0, x4\n\t"
        "addi %[it], %[it], -1\n\t"
        "bnez %[it], 1b\n\t"
        "vsetvli t0, %[n8], e32, m1, ta, ma\n\t"
        "vse32.v v8, (%[sk])\n\t"
        : [it] "+r"(it)
        : [pa] "r"(A), [pb] "r"(B), [n32] "r"(n32), [n8] "r"((size_t)8),
          [sk] "r"(sink)
        : "t0", "memory", "v0", "v4", "v8", "v9");
    double t_ime = now_s() - t0;

    /* ---- RVV: vwmacc.vv, 32 MACs per instruction, 8-way unrolled ---- */
    it = ITERS;
    t0 = now_s();
    __asm__ volatile(
        "vsetvli t0, %[n32], e8, m1, ta, ma\n\t"
        "vle8.v v0, (%[pa])\n\t"
        "vle8.v v1, (%[pb])\n\t"
        "vmv.v.i v16, 0\n\t"
        "1:\n\t"
        "vwmacc.vv v16, v0, v1\n\t" "vwmacc.vv v18, v0, v1\n\t"
        "vwmacc.vv v20, v0, v1\n\t" "vwmacc.vv v22, v0, v1\n\t"
        "vwmacc.vv v24, v0, v1\n\t" "vwmacc.vv v26, v0, v1\n\t"
        "vwmacc.vv v28, v0, v1\n\t" "vwmacc.vv v30, v0, v1\n\t"
        "addi %[it], %[it], -1\n\t"
        "bnez %[it], 1b\n\t"
        : [it] "+r"(it)
        : [pa] "r"(A), [pb] "r"(B), [n32] "r"(n32)
        : "t0", "memory", "v0", "v1", "v16", "v18", "v20", "v22",
          "v24", "v26", "v28", "v30");
    double t_rvv = now_s() - t0;

    /* ---- RVV, the sequence a BIT-EXACT int8 matmul actually needs ----
     * vwmacc.vv i8->i16 is not enough: at K=512 with |v|<=127 the sum reaches
     * ~8.2M and i16 saturates at 32767. Real accumulation is i32, which costs
     * vwmul.vv (i8->i16) plus vwadd.wv (i16 into i32): 2 instructions per 32
     * MACs. This is what kernels/rvv/rvv_matmul_s8_rvv_k_reduce_n_lanes.c does
     * and therefore the honest baseline for the ceiling. */
    it = ITERS;
    t0 = now_s();
    __asm__ volatile(
        "vsetvli t0, %[n32], e8, m1, ta, ma\n\t"
        "vle8.v v0, (%[pa])\n\t"
        "vle8.v v1, (%[pb])\n\t"
        "vmv.v.i v16, 0\n\t" "vmv.v.i v20, 0\n\t"
        "vmv.v.i v24, 0\n\t" "vmv.v.i v28, 0\n\t"
        "1:\n\t"
        "vwmul.vv v4, v0, v1\n\t"  "vwadd.wv v16, v16, v4\n\t"
        "vwmul.vv v6, v0, v1\n\t"  "vwadd.wv v20, v20, v6\n\t"
        "vwmul.vv v8, v0, v1\n\t"  "vwadd.wv v24, v24, v8\n\t"
        "vwmul.vv v10, v0, v1\n\t" "vwadd.wv v28, v28, v10\n\t"
        "addi %[it], %[it], -1\n\t"
        "bnez %[it], 1b\n\t"
        : [it] "+r"(it)
        : [pa] "r"(A), [pb] "r"(B), [n32] "r"(n32)
        : "t0", "memory", "v0", "v1", "v4", "v6", "v8", "v10",
          "v16", "v20", "v24", "v28");
    double t_rvv32 = now_s() - t0;
    double rvv32_macs = (double)ITERS * 4.0 * 32.0;

    double ime_macs = (double)ITERS * 8.0 * 128.0;
    double rvv_macs = (double)ITERS * 8.0 * 32.0;
    printf("smt.vmadot : %8.4f s for %.3g MACs  -> %7.2f GMAC/s  (%.1f MACs/instr)\n",
           t_ime, ime_macs, ime_macs / t_ime / 1e9, 128.0);
    printf("vwmacc.vv  : %8.4f s for %.3g MACs  -> %7.2f GMAC/s  (%.1f MACs/instr)\n",
           t_rvv, rvv_macs, rvv_macs / t_rvv / 1e9, 32.0);
    printf("\ninstruction issue rate:\n");
    printf("  vmadot : %8.3f ns/instr\n", t_ime / (ITERS * 8.0) * 1e9);
    printf("  vwmacc : %8.3f ns/instr\n", t_rvv / (ITERS * 8.0) * 1e9);
    printf("vwmul+vwadd: %8.4f s for %.3g MACs  -> %7.2f GMAC/s  (32 MACs / 2 instr)\n",
           t_rvv32, rvv32_macs, rvv32_macs / t_rvv32 / 1e9);
    printf("\nCEILING vs i16-accumulating vwmacc (NOT bit-exact for K=512): %.2fx\n",
           (ime_macs / t_ime) / (rvv_macs / t_rvv));
    printf("CEILING vs the i32 sequence a real int8 matmul needs:        %.2fx\n",
           (ime_macs / t_ime) / (rvv32_macs / t_rvv32));
    return 0;
}
