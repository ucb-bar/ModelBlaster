#define _GNU_SOURCE
/* Which IME instruction families does this K1 actually implement?
 *
 * The extension spec (spacemit-com/riscv-ime-extension-spec) defines three families:
 *   OPMMA   func7 111000  vmadot        -- the base integer dot, what our kernels emit
 *   OPMMA-k func7 111001  vmadot1/2/3/n -- sliding-window integer, A re-indexed by a slide offset
 *   OPFMMA  func7 111010  vfmadot...    -- fp16/bf16 matrix dot
 * func3 selects signedness for OPMMA (0 uu, 1 us, 2 su, 3 ss) and the slide for OPFMMA.
 * .insn's funct7 field is 7 bits: the spec's 6-bit func7 plus the vm/mode bit, so
 * 111000|1 = 0x71 (what our kernels use), 111001|1 = 0x73, 111010|1 = 0x75.
 *
 * Each candidate runs under the SAME vset as our working kernel, so a trap means the
 * family is absent rather than the MAC configuration being unsupported -- the spec says
 * an unsupported MAC config also raises illegal-instruction, which is why the control
 * (plain vmadot, known good) is probed identically alongside.
 */
#include <stdio.h>
#include <signal.h>
#include <setjmp.h>
#include <string.h>
#include <stdint.h>
#include <sched.h>
#include <stdlib.h>

static sigjmp_buf jb;
static void ill(int s){ (void)s; siglongjmp(jb, 1); }

static int8_t A[64] __attribute__((aligned(64)));
static int8_t B[64] __attribute__((aligned(64)));

#define PROBE(name, body) do {                                              \
    struct sigaction sa, old; memset(&sa,0,sizeof sa);                      \
    sa.sa_handler = ill; sigaction(SIGILL, &sa, &old);                      \
    const int8_t *pa = A, *pb = B;                                          \
    size_t n32 = 32, n8 = 8; (void)n8;                                      \
    if (sigsetjmp(jb, 1) == 0) { body; printf("  %-28s EXECUTES\n", name); } \
    else                        printf("  %-28s SIGILL\n", name);           \
    sigaction(SIGILL, &old, NULL);                                          \
} while (0)

#define SETUP \
    "vsetvli t0, %[n32], e8, m1, ta, ma\n\t" \
    "vmv.v.i v8, 0\n\t" "vmv.v.i v9, 0\n\t" \
    "vle8.v v0, (%[pa])\n\t" "vle8.v v4, (%[pb])\n\t"

#define ASMOUT : [pa]"+r"(pa), [pb]"+r"(pb) : [n32]"r"(n32) \
               : "t0","v0","v1","v4","v5","v8","v9","memory"

int main(int argc, char **argv) {
    int hart = argc > 1 ? atoi(argv[1]) : 0;
    cpu_set_t m; CPU_ZERO(&m); CPU_SET(hart, &m);
    if (sched_setaffinity(0, sizeof m, &m) != 0) { perror("affinity"); return 2; }
    for (int i = 0; i < 64; i++) { A[i] = (int8_t)(i%7-3); B[i] = (int8_t)(i%5-2); }
    printf("hart %d\n", hart);

    /* control: the exact instruction our kernels already run */
    PROBE("vmadot ss  (f7=0x71,f3=3)", __asm__ volatile(SETUP ".insn r 0x2b, 3, 0x71, x8, x0, x4\n\t" ASMOUT));
    /* the other signedness variants of the same family */
    PROBE("vmadot uu  (f7=0x71,f3=0)", __asm__ volatile(SETUP ".insn r 0x2b, 0, 0x71, x8, x0, x4\n\t" ASMOUT));
    PROBE("vmadot us  (f7=0x71,f3=1)", __asm__ volatile(SETUP ".insn r 0x2b, 1, 0x71, x8, x0, x4\n\t" ASMOUT));
    PROBE("vmadot su  (f7=0x71,f3=2)", __asm__ volatile(SETUP ".insn r 0x2b, 2, 0x71, x8, x0, x4\n\t" ASMOUT));
    /* sliding-window integer family -- the one built for convolution */
    PROBE("vmadot1 ss (f7=0x73,f3=3)", __asm__ volatile(SETUP ".insn r 0x2b, 3, 0x73, x8, x0, x4\n\t" ASMOUT));
    PROBE("vmadot1 ss (f7=0x72,f3=3)", __asm__ volatile(SETUP ".insn r 0x2b, 3, 0x72, x8, x0, x4\n\t" ASMOUT));
    /* fp16/bf16 matrix family */
    PROBE("vfmadot    (f7=0x75,f3=0)", __asm__ volatile(SETUP ".insn r 0x2b, 0, 0x75, x8, x0, x4\n\t" ASMOUT));
    PROBE("vfmadot    (f7=0x74,f3=0)", __asm__ volatile(SETUP ".insn r 0x2b, 0, 0x74, x8, x0, x4\n\t" ASMOUT));
    return 0;
}
