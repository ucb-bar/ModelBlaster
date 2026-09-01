/* Saturn-OPU FireSim bitstream ISA probe — per-hart edition.
 *
 * Previous probe (commit 63b1308) read misa from main() and reported
 * misa.V=0. That was hart 0, the Rocket + Gemmini RoCC tile, which
 * was never expected to have V. The "rvv_opu" tile is hart 1
 * (Shuttle), and `misa` is per-hart, so the right test is to spawn
 * a thread pinned to hart 1 and run the V probe THERE.
 *
 * Setup: two threads.
 *   - hart 0 thread: prints its own misa/mstatus + mhartid for
 *     reference. No V tests (Rocket has no V).
 *   - hart 1 thread: prints its misa/mstatus + mhartid, attempts
 *     manual mstatus.VS=Initial write, then runs the staged V
 *     opcode tests. If V isn't supported on hart 1 either, we know
 *     the bitstream lacks RVV everywhere. If V IS supported on
 *     hart 1, we know exactly what the conv2d kernel needs to do
 *     (force a thread onto hart 1 + enable VS).
 */
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <zephyr/kernel.h>

/* See first probe's comment. Zephyr's global build clobbers -march
 * to scalar-only; enable V locally for the assembler. */
asm(".option arch, +v");

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

static int8_t  in_a[64] __attribute__((aligned(64)));
static int8_t  in_b[64] __attribute__((aligned(64)));
static int8_t  out_8[64] __attribute__((aligned(64)));
static int32_t out_32[64] __attribute__((aligned(64)));

static inline unsigned long read_mhartid(void)
{
    unsigned long h;
    asm volatile("csrr %0, mhartid" : "=r"(h));
    return h;
}

static void print_csrs(const char *label)
{
    unsigned long misa, mstatus;
    asm volatile("csrr %0, misa"    : "=r"(misa));
    asm volatile("csrr %0, mstatus" : "=r"(mstatus));
    printf("%s mhartid=%lu misa=0x%lx (V_bit=%lu) mstatus=0x%lx (VS=%lu)\n",
           label,
           read_mhartid(),
           misa, (misa >> 21) & 1UL,
           mstatus, (mstatus >> 9) & 0x3UL);
}

/* Per-hart probe body. Both threads call this; hart 1 expects V to
 * be present, hart 0 logs but skips the V tests. */
static void run_probe_tests(int run_v_tests)
{
    print_csrs("CSR_INIT");

    /* Manual mstatus.VS = Initial. Has no effect if misa.V=0
     * (mstatus.VS hardwired to Off). Should latch VS=01 on a
     * V-capable hart that the OS left unmapped. */
    unsigned long bit = (1UL << 9);
    asm volatile("csrs mstatus, %0" : : "r"(bit));
    print_csrs("CSR_AFTER");

    if (!run_v_tests) {
        printf("hart=%lu skipping V tests (not the V-tile)\n",
               read_mhartid());
        return;
    }

    for (int i = 0; i < 64; i++) {
        in_a[i] = (int8_t)i;
        in_b[i] = (int8_t)(i + 1);
    }

    MARK_START(1, "vsetvli e8/m1 SET rs1=16");
    {
        size_t vl;
        asm volatile("vsetvli %0, %1, e8, m1, ta, ma"
                     : "=r"(vl) : "r"((size_t)16));
        printf("        vl=%lu\n", (unsigned long)vl);
    }
    MARK_OK(1);

    MARK_START(2, "vle8.v + vse8.v e8/m1");
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"((size_t)16));
    asm volatile("vle8.v v16, (%0)" : : "r"(in_a));
    asm volatile("vse8.v v16, (%0)" : : "r"(out_8));
    MARK_OK(2);

    MARK_START(3, "vsetvli e16/m2 SET rs1=16");
    {
        size_t vl;
        asm volatile("vsetvli %0, %1, e16, m2, ta, ma"
                     : "=r"(vl) : "r"((size_t)16));
        printf("        vl=%lu\n", (unsigned long)vl);
    }
    MARK_OK(3);

    MARK_START(4, "vsetvli e32/m4 SET rs1=16");
    {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m4, ta, ma"
                     : "=r"(vl) : "r"((size_t)16));
        printf("        vl=%lu\n", (unsigned long)vl);
    }
    MARK_OK(4);

    MARK_START(5, "vwmul_vv e8/m1 -> e16/m2");
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"((size_t)16));
    asm volatile("vle8.v v16, (%0)" : : "r"(in_a));
    asm volatile("vle8.v v17, (%0)" : : "r"(in_b));
    asm volatile("vwmul.vv v18, v16, v17");
    MARK_OK(5);

    MARK_START(6, "vwadd_wv e16/m2 -> e32/m4");
    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"((size_t)16));
    asm volatile("vmv.v.i v8, 0");
    asm volatile("vwadd.wv v20, v8, v18");
    MARK_OK(6);

    MARK_START(7, "vredsum_vs e32/m4");
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"((size_t)16));
    asm volatile("vmv.s.x v24, x0");
    asm volatile("vredsum.vs v25, v20, v24");
    MARK_OK(7);

    MARK_START(8, "OPMVINBCAST m1, v0 (Saturn custom)");
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"((size_t)16));
    asm volatile("vmv.v.i v0, 0");
    OPMVINBCAST("x1", "x0");
    MARK_OK(8);

    MARK_START(9, "VOPACC m1, v18, v16 (Saturn custom)");
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"((size_t)16));
    asm volatile("vle8.v v16, (%0)" : : "r"(in_a));
    asm volatile("vle8.v v18, (%0)" : : "r"(in_b));
    VOPACC("x1", "x18", "x16");
    MARK_OK(9);

    MARK_START(10, "VMV_VR v0, x0, m1 (Saturn custom)");
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"((size_t)16));
    VMV_VR("x0", (uintptr_t)0, "x1");
    asm volatile("vse32.v v0, (%0)" : : "r"(out_32));
    MARK_OK(10);
}

/* Hart 1 thread entry. Pinned via affinity in main(). */
static K_THREAD_STACK_DEFINE(hart1_stack, 16384);
static struct k_thread hart1_thread;
static struct k_sem hart1_done;

static void hart1_entry(void *a, void *b, void *c)
{
    (void)a; (void)b; (void)c;
    printf("=== HART_1_PROBE START (mhartid=%lu) ===\n", read_mhartid());
    run_probe_tests(/*run_v_tests=*/1);
    printf("=== HART_1_PROBE END ===\n");
    k_sem_give(&hart1_done);
}

int main(void)
{
    printf("=== Saturn-OPU FireSim ISA probe START ===\n");
    printf("=== HART_0_PROBE START (mhartid=%lu) ===\n", read_mhartid());
    run_probe_tests(/*run_v_tests=*/0);
    printf("=== HART_0_PROBE END ===\n");

    /* Spawn hart 1 thread and wait. CPU mask 0x2 = pin to hart 1
     * (CONFIG_SCHED_CPU_MASK_PIN_ONLY enforces single-hart). */
    k_sem_init(&hart1_done, 0, 1);
    /* K_FOREVER delays auto-start so we can set cpu affinity before
     * the thread runs (k_thread_cpu_mask_* require the thread to be
     * not-started). */
    k_thread_create(&hart1_thread, hart1_stack,
                    K_THREAD_STACK_SIZEOF(hart1_stack),
                    hart1_entry, NULL, NULL, NULL,
                    /*prio=*/5, /*options=*/0, K_FOREVER);
    k_thread_name_set(&hart1_thread, "hart1_probe");
    k_thread_cpu_mask_clear(&hart1_thread);
    k_thread_cpu_mask_enable(&hart1_thread, 1);  /* hart 1 only */
    k_thread_start(&hart1_thread);

    if (k_sem_take(&hart1_done, K_MSEC(60000)) != 0) {
        printf("ERROR: hart1 probe timed out — possibly stuck in trap loop\n");
    }

    printf("=== Saturn-OPU FireSim ISA probe END ===\n");
    return 0;
}
