/* Standalone FPGA reproducer for the two curated fp16 conv kernels.
 * Runs every ViNT conv2d_f16 / depthwise_conv2d_f16 shape in isolation. */
#include <stdio.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/reboot.h>

void kernel_conv2d_f16(const _Float16 *input, const _Float16 *weight,
                       const _Float16 *bias, _Float16 *output,
                       int N, int IC, int IH, int IW, int OC,
                       int KH, int KW, int SH, int SW, int PH, int PW);
void kernel_depthwise_conv2d_f16(const _Float16 *input, const _Float16 *weight,
                                 const _Float16 *bias, _Float16 *output,
                                 int N, int IC, int IH, int IW, int OC,
                                 int KH, int KW, int SH, int SW, int PH, int PW);

/* Push the working buffers up to roughly ViNT's .bss addresses so any
 * address-range-dependent effect is preserved. */
#ifndef KUT_ARENA_BYTES
#define KUT_ARENA_BYTES (0x2E00000)
#endif
/* One big .bss arena; the working buffers are carved out of the TOP of it so
 * they land at roughly ViNT's addresses (~0x82exxxxx) rather than just above
 * .text. Keeps any address-range-dependent effect in play. */
static unsigned char g_arena[KUT_ARENA_BYTES] __attribute__((aligned(64)));

#define IN_MAX   (600 * 1024)
#define W_MAX    (512 * 1024)
#define OUT_MAX  (300 * 1024)
static _Float16 *g_in;
static _Float16 *g_w;
static _Float16 *g_bias;
static _Float16 *g_out;

static void kut_carve(void)
{
    unsigned char *top = g_arena + KUT_ARENA_BYTES;
    top -= OUT_MAX * sizeof(_Float16);  g_out  = (_Float16 *)top;
    top -= IN_MAX  * sizeof(_Float16);  g_in   = (_Float16 *)top;
    top -= W_MAX   * sizeof(_Float16);  g_w    = (_Float16 *)top;
    top -= 2048    * sizeof(_Float16);  g_bias = (_Float16 *)top;
}

typedef struct {
    int did; int dw; int N, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW; int has_bias;
} shape_t;

static const shape_t SHAPES[] = {
#include "shapes.inc"
};
#define NSHAPES ((int)(sizeof(SHAPES) / sizeof(SHAPES[0])))

static unsigned long lcg = 12345u;
static _Float16 rnd(void)
{
    lcg = lcg * 1103515245u + 12345u;
    /* small magnitudes so fp16 accumulation cannot overflow */
    return (_Float16)((float)((int)((lcg >> 16) & 0xff) - 128) * 0.00390625f);
}

int main(void)
{
#if defined(CONFIG_SMP) && defined(CONFIG_RISCV_ISA_EXT_V) && (CONFIG_MP_MAX_NUM_CPUS > 1)
    k_thread_cpu_pin(k_current_get(), 1);
#endif
#ifdef KUT_MASK_IRQ
    /* Mitigation under test: mask machine interrupts for the whole vector
     * workload. Traps taken while Saturn has vector work in flight corrupt
     * one scalar register; masking removes the trigger entirely. */
    __asm__ volatile("csrci mstatus, 8" ::: "memory");
#endif
    kut_carve();
    printf("kut harness: nshapes=%d in=%p w=%p bias=%p out=%p arena=%p\n",
           NSHAPES, (void *)g_in, (void *)g_w, (void *)g_bias,
           (void *)g_out, (void *)g_arena);

    for (int s = 0; s < NSHAPES; s++) {
        const shape_t *p = &SHAPES[s];
        long in_elems, w_elems, out_elems;
        int OH = (p->IH + 2 * p->PH - p->KH) / p->SH + 1;
        int OW = (p->IW + 2 * p->PW - p->KW) / p->SW + 1;
        if (p->dw) {
            in_elems  = (long)p->N * p->OC * p->IH * p->IW;
            w_elems   = (long)p->OC * p->KH * p->KW;
        } else {
            in_elems  = (long)p->N * p->IC * p->IH * p->IW;
            w_elems   = (long)p->OC * p->IC * p->KH * p->KW;
        }
        out_elems = (long)p->N * p->OC * OH * OW;
        if (in_elems > IN_MAX || w_elems > W_MAX || out_elems > OUT_MAX
            || p->OC > 2048) {
            printf("SKIP %d too big in=%ld w=%ld out=%ld\n",
                   p->did, in_elems, w_elems, out_elems);
            continue;
        }
        for (long i = 0; i < in_elems; i++) g_in[i] = rnd();
        for (long i = 0; i < w_elems; i++) g_w[i] = rnd();
        for (int i = 0; i < p->OC; i++) g_bias[i] = rnd();
        for (long i = 0; i < out_elems; i++) g_out[i] = (_Float16)0.0f;

        printf("RUN %d %s N=%d IC=%d IH=%d IW=%d OC=%d K=%dx%d S=%dx%d P=%dx%d "
               "OH=%d OW=%d bias=%d\n",
               p->did, p->dw ? "dw" : "conv", p->N, p->IC, p->IH, p->IW, p->OC,
               p->KH, p->KW, p->SH, p->SW, p->PH, p->PW, OH, OW, p->has_bias);

        const _Float16 *b = p->has_bias ? g_bias : NULL;
        if (p->dw) {
            kernel_depthwise_conv2d_f16(g_in, g_w, b, g_out, p->N, p->IC,
                                        p->IH, p->IW, p->OC, p->KH, p->KW,
                                        p->SH, p->SW, p->PH, p->PW);
        } else {
            kernel_conv2d_f16(g_in, g_w, b, g_out, p->N, p->IC, p->IH, p->IW,
                              p->OC, p->KH, p->KW, p->SH, p->SW, p->PH, p->PW);
        }
        double acc = 0.0;
        for (long i = 0; i < out_elems; i++) acc += (double)(float)g_out[i];
        printf("OK %d sum=%.6g\n", p->did, acc);
    }
    printf("=== KUT DONE ===\n");
    sys_reboot(SYS_REBOOT_COLD);
    return 0;
}
