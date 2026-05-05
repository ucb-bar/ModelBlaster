/* Single-op gemmini correctness test.
 *
 * Runs ONE conv2d_s8 with both scalar reference and gemmini paths,
 * prints both outputs side-by-side and a max-abs-diff. Small enough
 * dims (1x4x8x8 NCHW input, 4x4x3x3 weight) that the printout fits
 * on a few lines and rounding drift is obvious.
 *
 * No model harness, no dispatch table — just two function calls and
 * a printf.
 */
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "gemmini.h"

/* Test shape — small enough that one DIM=16 tile covers IC and OC. */
#define N    1
#define IC   4
#define IH   8
#define IW   8
#define OC   4
#define KH   3
#define KW   3
#define SH   1
#define SW   1
#define PH   1
#define PW   1
#define OH   8   /* (8 + 2*1 - 3)/1 + 1 */
#define OW   8

/* Q0.31 requantize params — pick small enough multiplier+shift that
 * outputs land in the int8 range without saturating everything. */
#define MULT  1073741824   /* 0.5 in Q0.31 */
#define SHIFT 4            /* effective scale = 0.5 / 16 = 1/32 */
#define ACT_MIN -128
#define ACT_MAX  127

/* Pseudo-random int8 input/weight/bias. Same content for both runs. */
static int8_t  input  [N*IC*IH*IW];
static int8_t  weight [OC*IC*KH*KW];
static int32_t bias   [OC];
static int8_t  out_scalar[N*OC*OH*OW];
static int8_t  out_gemmini[N*OC*OH*OW];

/* NHWC + OHWI scratch for gemmini call. */
static elem_t ws_input  [N*IH*IW*IC]  __attribute__((aligned(64)));
static elem_t ws_weight [OC*KH*KW*IC] __attribute__((aligned(64)));
static elem_t ws_output [N*OH*OW*OC]  __attribute__((aligned(64)));

static void scalar_conv(void)
{
    for (int n = 0; n < N; n++)
    for (int oc = 0; oc < OC; oc++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        int32_t acc = bias[oc];
        for (int ic = 0; ic < IC; ic++)
        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            int ih = oh*SH - PH + kh;
            int iw = ow*SW - PW + kw;
            if (ih < 0 || ih >= IH || iw < 0 || iw >= IW) continue;
            acc += (int32_t)input[((n*IC+ic)*IH+ih)*IW+iw]
                 * (int32_t)weight[((oc*IC+ic)*KH+kh)*KW+kw];
        }
        /* SCALE=1.0 path: just clamp + saturate to match gemmini's
         * ACC_SCALE_IDENTITY behavior. */
        int32_t scaled = acc;
        if (scaled < -128) scaled = -128;
        if (scaled > 127)  scaled = 127;
        out_scalar[((n*OC+oc)*OH+oh)*OW+ow] = (int8_t)scaled;
    }
}

static void gemmini_conv(void)
{
    /* One-time gemmini state flush — required by the library to put
     * the accelerator in a known state. The bareMetalC examples do
     * this at the top of main(). */
    gemmini_flush(0);


    /* NCHW -> NHWC. */
    for (int n = 0; n < N; n++)
    for (int h = 0; h < IH; h++)
    for (int w = 0; w < IW; w++)
    for (int c = 0; c < IC; c++)
        ws_input[((n*IH + h)*IW + w)*IC + c] =
            input[((n*IC + c)*IH + h)*IW + w];

    /* OIHW -> OHWI. */
    for (int oc = 0; oc < OC; oc++)
    for (int kh = 0; kh < KH; kh++)
    for (int kw = 0; kw < KW; kw++)
    for (int ic = 0; ic < IC; ic++)
        ws_weight[((oc*KH + kh)*KW + kw)*IC + ic] =
            weight[((oc*IC + ic)*KH + kh)*KW + kw];

    float scale = ACC_SCALE_IDENTITY;   /* 1.0 — debug: conv math only */
    printf("scale = %g (DEBUG: ACC_SCALE_IDENTITY)\n", (double)scale);

    tiled_conv_auto(
        N, IH, IW, IC,
        OC, OH, OW,
        SH, 1, 1, PH, KH,
        false, false, false, false, false,
        ws_input, ws_weight, bias, ws_output,
        NO_ACTIVATION, scale,
        0, 0, 0,
        WS
    );

    /* NHWC -> NCHW. */
    for (int n = 0; n < N; n++)
    for (int c = 0; c < OC; c++)
    for (int h = 0; h < OH; h++)
    for (int w = 0; w < OW; w++)
        out_gemmini[((n*OC + c)*OH + h)*OW + w] =
            ws_output[((n*OH + h)*OW + w)*OC + c];
}

int main(void)
{
    /* Deterministic LCG fill. */
    uint32_t s = 12345;
    for (int i = 0; i < (int)sizeof(input); i++) {
        s = s * 1103515245u + 12345u;
        input[i] = (int8_t)(s >> 16);
    }
    for (int i = 0; i < (int)sizeof(weight); i++) {
        s = s * 1103515245u + 12345u;
        weight[i] = (int8_t)((s >> 16) & 0x07) - 4;  /* in [-4, 3] */
    }
    for (int i = 0; i < OC; i++) bias[i] = 0;   /* zero bias for isolation */

    scalar_conv();
    gemmini_conv();

    int max_diff = 0;
    int diffs = 0;
    for (int i = 0; i < N*OC*OH*OW; i++) {
        int d = (int)out_gemmini[i] - (int)out_scalar[i];
        if (d < 0) d = -d;
        if (d > max_diff) max_diff = d;
        if (d > 0) diffs++;
    }
    printf("max_abs_diff = %d, diffs = %d / %d\n",
           max_diff, diffs, N*OC*OH*OW);

    /* Print first row of output (8 elems) of channel 0 from both. */
    printf("scalar  c0,r0:");
    for (int i = 0; i < OW; i++) printf(" %4d", out_scalar[i]);
    printf("\ngemmini c0,r0:");
    for (int i = 0; i < OW; i++) printf(" %4d", out_gemmini[i]);
    printf("\n");
    printf("scalar  c1,r0:");
    for (int i = 0; i < OW; i++) printf(" %4d", out_scalar[OH*OW + i]);
    printf("\ngemmini c1,r0:");
    for (int i = 0; i < OW; i++) printf(" %4d", out_gemmini[OH*OW + i]);
    printf("\n");

    return max_diff <= 3 ? 0 : 1;
}
