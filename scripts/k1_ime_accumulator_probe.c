/* What order does smt.vmadot leave the 4x4 int32 accumulator in?
 *
 * The micro-tile is M=4, N=4, K=8, hardware-forced: at VLEN=256 with
 * vl=32,e8,m1 the MAC table gives 4x4x8. The 16 int32 results land in a
 * REGISTER PAIR (vd, vd+1) -- 2 x 32 bytes = 16 int32 -- and nothing we have
 * says which element of that pair is C[i][j]. Guessing it wrong produces a
 * kernel that computes a permuted answer, which the golden compare rejects
 * without telling you the permutation. So: make all 16 results distinct and
 * decodable, run one vmadot, and print the pair in memory order.
 *
 * A[i][k] = (k==0) ? i+1 : (k==1) ? 1 : 0
 * B[j][k] = (k==0) ? 16   : (k==1) ? j+1 : 0
 *   => C[i][j] = sum_k A[i][k]*B[j][k] = 16*(i+1) + (j+1)
 * so every value is unique in 17..68 and decodes as
 *   i = (v - 17) / 16,  j = (v - 17) % 16.
 *
 * Run on cluster 0 (harts 0-3). smt.vmadot SIGILLs on harts 4-7.
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

int main(void) {
    int8_t  A[32], B[32];
    int32_t C[16];

    memset(A, 0, sizeof A);
    memset(B, 0, sizeof B);
    for (int i = 0; i < 4; i++) { A[i*8 + 0] = (int8_t)(i + 1); A[i*8 + 1] = 1; }
    for (int j = 0; j < 4; j++) { B[j*8 + 0] = 16;              B[j*8 + 1] = (int8_t)(j + 1); }
    memset(C, 0, sizeof C);

    size_t n32 = 32, n8 = 8;
    __asm__ volatile(
        "vsetvli t0, %[n32], e8, m1, ta, ma\n\t"
        "vle8.v  v0, (%[pa])\n\t"
        "vle8.v  v4, (%[pb])\n\t"
        "vmv.v.i v8, 0\n\t"
        "vmv.v.i v9, 0\n\t"
        /* vl=32,e8,m1 must be live at the instruction itself. */
        "vsetvli t0, %[n32], e8, m1, ta, ma\n\t"
        ".insn r 0x2b, 3, 0x71, x8, x0, x4\n\t"     /* smt.vmadot v8, v0, v4 */
        "vsetvli t0, %[n8], e32, m1, ta, ma\n\t"
        "vse32.v v8, (%[pc0])\n\t"
        "vse32.v v9, (%[pc1])\n\t"
        :
        : [pa] "r"(A), [pb] "r"(B), [pc0] "r"(C), [pc1] "r"(C + 8),
          [n32] "r"(n32), [n8] "r"(n8)
        : "t0", "memory", "v0", "v4", "v8", "v9");

    printf("raw pair, memory order (v8 lanes 0-7 then v9 lanes 0-7):\n ");
    for (int e = 0; e < 16; e++) printf(" %3d", C[e]);
    printf("\n\ndecoded as C[i][j] = 16*(i+1) + (j+1):\n");
    for (int e = 0; e < 16; e++) {
        int v = C[e];
        if (v < 17 || v > 68 || ((v - 17) / 16) > 3 || ((v - 17) % 16) > 3) {
            printf("  elem %2d (%s lane %d) = %d  <- NOT a valid C[i][j]\n",
                   e, e < 8 ? "v8" : "v9", e % 8, v);
        } else {
            printf("  elem %2d (%s lane %d) = %d  ->  C[%d][%d]\n",
                   e, e < 8 ? "v8" : "v9", e % 8, v, (v - 17) / 16, (v - 17) % 16);
        }
    }
    return 0;
}

/* RESULT, measured on the K1 2026-08-28 (hart 3, GCC 14.3):
 *
 *   raw pair, memory order (v8 lanes 0-7 then v9 lanes 0-7):
 *     17  18  19  20  33  34  35  36  49  50  51  52  65  66  67  68
 *
 * THE ACCUMULATOR IS PLAIN ROW-MAJOR ACROSS THE PAIR. Element e of the
 * concatenated (vd, vd+1) is C[e/4][e%4]: vd holds rows 0-1, vd+1 rows 2-3.
 * No swizzle, no interleave, no lane permutation. A drain loop can therefore
 * `vse32.v` both registers into 16 contiguous int32 and index them directly.
 *
 * CONTROL, and it is what makes the above believable: the same binary on hart
 * 5 exits 132 (SIGILL). Cluster 1 does not implement smt.vmadot, so if the
 * instruction had been elided by the compiler or silently ignored by the core,
 * hart 5 would have printed the same numbers instead of dying. The values
 * above were produced by the MAC unit.
 *
 *   riscv64-...-gcc -O2 -march=rv64gcv_zvl256b_zfh_zvfh -mabi=lp64d -static
 *   scp; ssh k1 'taskset -c 3 ./k1_ime_accumulator_probe'
 */
