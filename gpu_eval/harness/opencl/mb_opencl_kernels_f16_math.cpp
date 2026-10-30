// fp16-MATH OpenCL seeds for devices with cl_khr_fp16 (Adreno 650 and newer). Default for the adreno650 target
// (MB_OPENCL_F16_MATH=1 / auto, see pipeline/opencl_seeds.py:f16_math_enabled); MB_OPENCL_F16_MATH=0 restores the
// storage-only seeds (vload_half / vstore_half, fp32 math) that mb_opencl_kernels_f16_hand.cpp and
// opencl_seeds.to_f16 produce.
//
// What is different from the storage-only seeds:
//   * the programs enable `cl_khr_fp16` (the runtime prelude does it too when the device lists it) and read and
//     write `half` data through DIRECT pointers: vload8 / vload4 / vstore8 on `__global half *`, 128-bit
//     transactions, no vload_half;
//   * every work-item owns a register tile (4 output pixels x 8 output channels, 4 rows x 8 columns, 8 elements);
//   * the arithmetic is half8 where that is exact (relu / relu6 / add / mul) and fp32 where the
//     fp16 contract needs it (dot-product accumulation, transcendental maps): fp16 operands, fp32 accumulators,
//     one RNE rounding per stored value -- the same numerics contract as the CUDA fp16 seeds.
//   * conv2d has a switch for FP16 ACCUMULATION: build option -DMB_ACC_HALF=1 (MB_CL_OPTS) accumulates in half8
//     (fp16 ALU rate is 2x fp32 on Adreno) and flushes into the fp32 accumulators every MB_ACC_FLUSH (default 16)
//     (ic, kh, kw) steps; MB_ACC_FLUSH=0 is pure fp16 accumulation. It is NOT the default: a long unflushed fp16
//     sum can leave the flow's per-slot bound (see docs/opencl_adreno.md for what was measured).
// The generic (non-hand) families -- relu, add, ... and the opcode ops -- are generated in opencl_seeds.py.
#include "mb_opencl_runtime.h"

// ---------------------------------------------------------------- conv2d (groups = 1, any stride / pad / dilation)
// weights re-laid out once per call as wt[(ic*KH*KW + kh*KW + kw) * OCp + oc] (OC padded to a multiple of MB_OT), so a
// work-item reads its MB_OT output channels of one filter tap as one half8 / half4.
// Tile: 4 output pixels x MB_OT output channels per work-item; MB_OT = 8 (default) or 4, set in kConvOpts below together
// with the other knobs: "-DMB_OT=4", "-DMB_ACC_HALF=1 -DMB_ACC_FLUSH=16" (fp16 accumulation, see the file header).
// Measured on Adreno 650 (cc_runs REPORTs): 4 channels were 1.8x faster at IC = 8 (cc_conv_div_leaky), 8 channels
// 1.6x faster at IC = 64 (cc_conv_min_add_mul): register pressure vs weight-load amortisation; tune per shape.
static const char kConvOpts[] = "-DMB_OT=8";
static mb_program cl_conv2d = MB_CL_OPTS(kConvOpts, R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifndef MB_OT
#define MB_OT 8
#endif
#ifndef MB_ACC_HALF
#define MB_ACC_HALF 0
#endif
#ifndef MB_ACC_FLUSH
#define MB_ACC_FLUSH 16
#endif
#if MB_OT == 8
typedef float8 fvec; typedef half8 hvec;
#define MB_LDW(p) vload8(0, (p))
#define MB_F(h) convert_float8(h)
#define MB_BV(t) (float8)(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7])
#define MB_STORE_ALL MB_ST(0) MB_ST(1) MB_ST(2) MB_ST(3) MB_ST(4) MB_ST(5) MB_ST(6) MB_ST(7)
#else
typedef float4 fvec; typedef half4 hvec;
#define MB_LDW(p) vload4(0, (p))
#define MB_F(h) convert_float4(h)
#define MB_BV(t) (float4)(t[0], t[1], t[2], t[3])
#define MB_STORE_ALL MB_ST(0) MB_ST(1) MB_ST(2) MB_ST(3)
#endif
// re-layout: one work-item per tap k (coalesced reads of w[oc][k] across the work-group, no integer division), the
// OCp channels written as hvec stores; w[N][K] -> wt[K][Np]
__kernel void k_wpack(__global const half *w, __global half *wt, int OC, int OCp, int K) {
    int k = (int)get_global_id(0);
    if (k >= K) return;
    for (int o = 0; o < OCp; o += MB_OT) {
        half t[MB_OT];
        for (int j = 0; j < MB_OT; j++) t[j] = (o + j < OC) ? w[(o + j) * K + k] : (half)0;
        for (int j = 0; j < MB_OT; j++) wt[k * OCp + o + j] = t[j];
    }
}

#if MB_ACC_HALF
// fp16 accumulation: half partial sums, flushed into the fp32 accumulators every MB_ACC_FLUSH taps
#define MB_MAC(W, X0, X1, X2, X3) { hvec w8 = (W); \
    h0 = mad((hvec)(X0), w8, h0); h1 = mad((hvec)(X1), w8, h1); h2 = mad((hvec)(X2), w8, h2); h3 = mad((hvec)(X3), w8, h3); \
    if (MB_ACC_FLUSH > 0 && ++cnt == MB_ACC_FLUSH) { cnt = 0; \
        a0 += MB_F(h0); a1 += MB_F(h1); a2 += MB_F(h2); a3 += MB_F(h3); \
        h0 = (hvec)(0.0h); h1 = h0; h2 = h0; h3 = h0; } }
#else
// fp32 accumulation of fp16 operands (the product of two halves is exact in fp32)
#define MB_MAC(W, X0, X1, X2, X3) { fvec wf = MB_F(W); \
    a0 = mad((fvec)((float)(X0)), wf, a0); a1 = mad((fvec)((float)(X1)), wf, a1); \
    a2 = mad((fvec)((float)(X2)), wf, a2); a3 = mad((fvec)((float)(X3)), wf, a3); }
#endif
#define MB_LD(J) (((J) >= 0 && (J) < IW) ? xr[(J)] : (half)0)
#define MB_ST(J) if (oc0 + J < OC) { \
    half4 o = convert_half4((float4)(a0.s##J, a1.s##J, a2.s##J, a3.s##J)); \
    __global half *yo = yb + J * pl; \
    if (nv >= 4) vstore4(o, 0, yo); else { yo[0] = o.s0; if (nv > 1) yo[1] = o.s1; if (nv > 2) yo[2] = o.s2; } }

// one work-item: 4 consecutive output pixels x MB_OT output channels of image n, row oh; any stride / dilation / padding
// (a stride-1 sliding window kept in registers and a bounds-check-free interior path were measured SLOWER, see docs)
__kernel void k_conv(__global const half *x, __global const half *wt, __global const half *b, __global half *y,
                     int IC, int IH, int IW, int OC, int OH, int OW, int KH, int KW,
                     int SH, int SW, int PH, int PW, int DH, int DW, int OCp) {
    int ow0 = (int)get_global_id(0) * 4, oh = (int)get_global_id(1), z = (int)get_global_id(2);
    if (ow0 >= OW || oh >= OH) return;
    int OCB = OCp / MB_OT, ocb = z % OCB, n = z / OCB;
    int oc0 = ocb * MB_OT;
    float bt[MB_OT];
    for (int j = 0; j < MB_OT; j++) bt[j] = (b && oc0 + j < OC) ? (float)b[oc0 + j] : 0.0f;
    fvec bv = MB_BV(bt);
    fvec a0 = bv, a1 = bv, a2 = bv, a3 = bv;                   // fp32 accumulators (bias folded in)
#if MB_ACC_HALF
    hvec h0 = (hvec)(0.0h), h1 = h0, h2 = h0, h3 = h0;
    int cnt = 0;
#endif
    __global const half *xn = x + (long)n * IC * IH * IW;
    for (int ic = 0; ic < IC; ic++) {
        for (int kh = 0; kh < KH; kh++) {
            int ih = oh * SH - PH + kh * DH;
            if (ih < 0 || ih >= IH) continue;
            __global const half *xr = xn + (long)(ic * IH + ih) * IW;
            __global const half *wp = wt + (long)((ic * KH + kh) * KW) * OCp + oc0;
            for (int kw = 0; kw < KW; kw++) {
                int iw = ow0 * SW - PW + kw * DW;
                half x0 = MB_LD(iw), x1 = MB_LD(iw + SW), x2 = MB_LD(iw + 2 * SW), x3 = MB_LD(iw + 3 * SW);
                MB_MAC(MB_LDW(wp + kw * OCp), x0, x1, x2, x3)
            }
        }
    }
#if MB_ACC_HALF
    a0 += MB_F(h0); a1 += MB_F(h1); a2 += MB_F(h2); a3 += MB_F(h3);
#endif
    long pl = (long)OH * OW;
    __global half *yb = y + ((long)n * OC + oc0) * pl + (long)oh * OW + ow0;
    int nv = OW - ow0;
    MB_STORE_ALL
}
)CL");

void kernel_conv2d_f16(const __half *input, const __half *weight, const __half *bias, __half *output,
                       int N, int IC, int IH, int IW, int OC, int KH, int KW,
                       int SH, int SW, int PH, int PW, int DH, int DW) {
    const char *ot = strstr(kConvOpts, "MB_OT=");
    const int OT = ot ? atoi(ot + 6) : 8;                       // the program's MB_OT (kConvOpts)
    int OH = (IH + 2 * PH - DH * (KH - 1) - 1) / SH + 1, OW = (IW + 2 * PW - DW * (KW - 1) - 1) / SW + 1;
    int OCp = (OC + OT - 1) / OT * OT;
    int K = IC * KH * KW;
    __half *wt = (__half *)MB_SCRATCH(cl_conv2d, (size_t)OCp * K * sizeof(__half));
    mb_launch(MB_KERNEL(cl_conv2d, k_wpack), mb_grid(K), weight, wt, OC, OCp, K);
    mb_launch(MB_KERNEL(cl_conv2d, k_conv), mb_range3((OW + 3) / 4, OH, (long)N * (OCp / OT), 16, 4, 1),
              input, (const __half *)wt, bias, output, IC, IH, IW, OC, OH, OW, KH, KW, SH, SW, PH, PW, DH, DW, OCp);
}

// ---------------------------------------------------------------- GEMM family
// C[M,N] = A[M,K] * B[K, ldb] (+ bias[N]); one work-item: 4 rows x 8 columns, fp32 accumulators, half4 reads of A,
// half8 reads of B; B's row stride ldb must keep the 8-wide read in the row (N % 8 == 0 or a padded copy).
static mb_program cl_gemm = MB_CL(R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// w[N][K] -> wt[K][Np]: one work-item per k (coalesced reads, no integer division), 8 columns per hvec store
__kernel void k_wpack_t(__global const half *w, __global half *wt, int N, int Np, int K) {
    int k = (int)get_global_id(0);
    if (k >= K) return;
    for (int o = 0; o < Np; o += 8) {
        half t[8];
        for (int j = 0; j < 8; j++) t[j] = (o + j < N) ? w[(o + j) * K + k] : (half)0;
        for (int j = 0; j < 8; j++) wt[k * Np + o + j] = t[j];
    }
}
#define MB_ROW(C, X) C = mad((float8)((float)X.s0), f0, C); C = mad((float8)((float)X.s1), f1, C); \
                     C = mad((float8)((float)X.s2), f2, C); C = mad((float8)((float)X.s3), f3, C);
#define MB_STR(R, C) if (m0 + R < M) { half8 o8 = convert_half8(C + bv); \
    __global half *yo = Cm + (long)(m0 + R) * N + n0; \
    if (n0 + 8 <= N) vstore8(o8, 0, yo); \
    else { half ot[8]; vstore8(o8, 0, ot); for (int j = 0; j < N - n0; j++) yo[j] = ot[j]; } }
__kernel void k_gemm(__global const half *A, __global const half *B, __global half *Cm, __global const half *bias,
                     int M, int K, int N, int ldb) {
    int n0 = (int)get_global_id(0) * 8, m0 = (int)get_global_id(1) * 4;
    if (n0 >= N || m0 >= M) return;
    float8 c0 = (float8)(0.0f), c1 = c0, c2 = c0, c3 = c0;
    __global const half *a0 = A + (long)m0 * K;
    __global const half *a1 = A + (long)min(m0 + 1, M - 1) * K;      // clamped rows: loads stay in bounds, stores are masked
    __global const half *a2 = A + (long)min(m0 + 2, M - 1) * K;
    __global const half *a3 = A + (long)min(m0 + 3, M - 1) * K;
    int k = 0;
    for (; k + 4 <= K; k += 4) {
        __global const half *bp = B + (long)k * ldb + n0;
        float8 f0 = convert_float8(vload8(0, bp)), f1 = convert_float8(vload8(0, bp + ldb));
        float8 f2 = convert_float8(vload8(0, bp + 2 * (long)ldb)), f3 = convert_float8(vload8(0, bp + 3 * (long)ldb));
        half4 x0 = vload4(0, a0 + k), x1 = vload4(0, a1 + k), x2 = vload4(0, a2 + k), x3 = vload4(0, a3 + k);
        MB_ROW(c0, x0) MB_ROW(c1, x1) MB_ROW(c2, x2) MB_ROW(c3, x3)
    }
    for (; k < K; k++) {
        float8 f = convert_float8(vload8(0, B + (long)k * ldb + n0));
        c0 = mad((float8)((float)a0[k]), f, c0); c1 = mad((float8)((float)a1[k]), f, c1);
        c2 = mad((float8)((float)a2[k]), f, c2); c3 = mad((float8)((float)a3[k]), f, c3);
    }
    float bt[8];
    for (int j = 0; j < 8; j++) bt[j] = (bias && n0 + j < N) ? (float)bias[n0 + j] : 0.0f;
    float8 bv = (float8)(bt[0], bt[1], bt[2], bt[3], bt[4], bt[5], bt[6], bt[7]);
    MB_STR(0, c0) MB_STR(1, c1) MB_STR(2, c2) MB_STR(3, c3)
}
// any shape: one output per work-item, fp32 accumulation
__kernel void k_gemm_s(__global const half *A, __global const half *B, __global half *Cm, __global const half *bias,
                       int M, int K, int N, int ldb) {
    long t = get_global_id(0);
    if (t >= (long)M * N) return;
    int n = (int)(t % N);
    long m = t / N;
    float s = bias ? (float)bias[n] : 0.0f;
    for (int k = 0; k < K; k++) s = mad((float)A[m * K + k], (float)B[(long)k * ldb + n], s);
    Cm[t] = convert_half(s);
}
)CL");

void kernel_matmul_f16(const __half *A, const __half *B, __half *C, int M, int K, int N) {
    if ((N & 7) == 0)
        mb_launch(MB_KERNEL(cl_gemm, k_gemm), mb_range2((N + 7) / 8, (M + 3) / 4, 16, 4), A, B, C,
                  (const __half *)nullptr, M, K, N, N);
    else
        mb_launch(MB_KERNEL(cl_gemm, k_gemm_s), mb_grid((long)M * N), A, B, C, (const __half *)nullptr, M, K, N, N);
}

// nn.Linear: out[M,N] = in[M,K] * W[N,K]^T + bias[N]; W is re-laid out once per call to [K][Np] (Np = N up to a multiple of 8)
void kernel_linear_f16(const __half *input, const __half *weight, const __half *bias, __half *output, int M, int K, int N) {
    int Np = (N + 7) & ~7;
    __half *wt = (__half *)MB_SCRATCH(cl_gemm, (size_t)K * Np * sizeof(__half));
    mb_launch(MB_KERNEL(cl_gemm, k_wpack_t), mb_grid(K), weight, wt, N, Np, K);
    mb_launch(MB_KERNEL(cl_gemm, k_gemm), mb_range2((N + 7) / 8, (M + 3) / 4, 16, 4), input, (const __half *)wt, output,
              bias, M, K, N, Np);
}

// ---------------------------------------------------------------- batchnorm (inference, folded scale / bias)
// y = x * scale[c] + bias[c]. HW % 8 == 0: 8 consecutive elements share a channel (half8 path).
static mb_program cl_batchnorm = MB_CL(R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// y = x * scale[c] + bias[c]: half8 loads / stores, the multiply-add in fp32 (the product of two halves is exact in
// fp32, so this equals the fp32-math reference up to one extra fp32 rounding). A half `fma` is NOT used: on Adreno it
// is emulated (a correctly rounded half fma needs wider arithmetic) and made this kernel 100x slower than the fp32 mad.
// 32-bit index math: a 64-bit divide / modulo is emulated on Adreno too.
__kernel void k_bn8(__global const half *x, __global const half *s, __global const half *b, __global half *y,
                    int C, int HW, int tot) {
    int i = (int)get_global_id(0) * 8;
    if (i >= tot) return;
    int c = (i / HW) % C;
    float8 v = convert_float8(vload8(0, x + i));
    vstore8(convert_half8(mad(v, (float8)((float)s[c]), (float8)((float)b[c]))), 0, y + i);
}
__kernel void k_bn1(__global const half *x, __global const half *s, __global const half *b, __global half *y,
                    int C, int HW, int tot) {
    int i = (int)get_global_id(0);
    if (i >= tot) return;
    int c = (i / HW) % C;
    y[i] = convert_half(mad((float)x[i], (float)s[c], (float)b[c]));
}
__kernel void k_bnl(__global const half *x, __global const half *s, __global const half *b, __global half *y,
                    int C, int HW, long tot) {      // > 2^31 elements
    long i = get_global_id(0);
    if (i >= tot) return;
    int c = (int)((i / HW) % C);
    y[i] = convert_half(mad((float)x[i], (float)s[c], (float)b[c]));
}
)CL");

void kernel_batchnorm2d_f16(const __half *input, const __half *scale, const __half *bias, __half *output,
                            int N, int C, int H, int W) {
    long tot = (long)N * C * H * W;
    int HW = H * W;
    if (tot > 0x7fffffffL)
        mb_launch(MB_KERNEL(cl_batchnorm, k_bnl), mb_grid(tot), input, scale, bias, output, C, HW, tot);
    else if ((HW & 7) == 0)
        mb_launch(MB_KERNEL(cl_batchnorm, k_bn8), mb_grid(tot / 8), input, scale, bias, output, C, HW, (int)tot);
    else
        mb_launch(MB_KERNEL(cl_batchnorm, k_bn1), mb_grid(tot), input, scale, bias, output, C, HW, (int)tot);
}

// ---------------------------------------------------------------- broadcasting binary op (opcode ops: add / sub / mul / div / min / max / ...)
// Three things made the fp32 seed's kernel take ~1 s for a 2 M-element bias add on Adreno, all measured (cc_runs REPORTs of
// cc_conv_min_add_mul / cc_conv_relu_bias / cc_resnetblock, and the first version of this file): 64-bit div / mod in
// mb_idx6; the opcode `switch` whose cases include pow() (27 ms -> 0.5 ms once add / sub / mul / min / max got their own
// kernels); and one division per element. Here: 32-bit index math, one kernel per common opcode (EW2 macro below; the
// switch kernels remain for the rare opcodes), and two vector paths -- both operands contiguous in the output's shape, and
// b broadcast over the trailing `inner` elements (a per-channel bias over [N, C, H*W]) -- 8 elements per work-item, half8
// loads / stores, fp32 math, one division per 8 elements.
static mb_program cl_ewise2 = MB_CL(R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
float mb_e2(float x, float y, int opc, float alpha) {
    switch (opc) {
    case 0: return x + alpha * y;  case 1: return x - alpha * y;  case 2: return x * y;
    case 3: return x / y;  case 4: return (x > y || isnan(x)) ? x : y;
    case 5: return (x < y || isnan(x)) ? x : y;  case 6: return pow(x, y);
    case 7: return y != 0.0f ? alpha : x;  case 8: return x == y;  case 9: return x != y;
    case 10: return x < y;  case 11: return x <= y;  case 12: return x > y;  case 13: return x >= y;
    default: return x;
    }
}
__kernel void k_ewise2(__global const half *a, __global const half *b, __global half *y, int n, int opc, float alpha,
                       int d1, int d2, int d3, int d4, int d5,
                       int sa0, int sa1, int sa2, int sa3, int sa4, int sa5,
                       int sb0, int sb1, int sb2, int sb3, int sb4, int sb5) {
    int i = (int)get_global_id(0);
    if (i >= n) return;
    int r = i, c5 = r % d5; r /= d5;
    int c4 = r % d4; r /= d4;
    int c3 = r % d3; r /= d3;
    int c2 = r % d2; r /= d2;
    int c1 = r % d1, c0 = r / d1;
    int ia = c0 * sa0 + c1 * sa1 + c2 * sa2 + c3 * sa3 + c4 * sa4 + c5 * sa5;
    int ib = c0 * sb0 + c1 * sb1 + c2 * sb2 + c3 * sb3 + c4 * sb4 + c5 * sb5;
    y[i] = convert_half(mb_e2((float)a[ia], (float)b[ib], opc, alpha));
}
// EXPR is written for float8 and for float (the tail): fa, fb, alpha
#define EW2(NAME, EXPR) \
__kernel void k_ew2_v8_##NAME(__global const half *a, __global const half *b, __global half *y, int n, float alpha) { \
    int i = (int)get_global_id(0) * 8; \
    if (i + 8 <= n) { float8 fa = convert_float8(vload8(0, a + i)), fb = convert_float8(vload8(0, b + i)); \
        vstore8(convert_half8(EXPR), 0, y + i); } \
    else for (int j = i; j < n; j++) { float fa = (float)a[j], fb = (float)b[j]; y[j] = convert_half(EXPR); } } \
__kernel void k_ew2_bc_##NAME(__global const half *a, __global const half *b, __global half *y, int n, int inner, float alpha) { \
    int i = (int)get_global_id(0) * 8; \
    if (i + 8 <= n && (inner & 7) == 0) { float8 fa = convert_float8(vload8(0, a + i)), fb = (float8)((float)b[i / inner]); \
        vstore8(convert_half8(EXPR), 0, y + i); } \
    else for (int j = i; j < min(i + 8, n); j++) { float fa = (float)a[j], fb = (float)b[j / inner]; y[j] = convert_half(EXPR); } }
EW2(add, fa + alpha * fb)
EW2(sub, fa - alpha * fb)
EW2(mul, fa * fb)
EW2(div, fa / fb)
EW2(max, select(fb, fa, (fa > fb) | isnan(fa)))
EW2(min, select(fb, fa, (fa < fb) | isnan(fa)))
)CL");

static bool mb_contig6(const int *d, const int *s) {       // strides s are those of a row-major tensor of shape d (size-1 dims free)
    long e = 1;
    for (int k = 5; k >= 0; --k) {
        if (d[k] > 1 && s[k] != e) return false;
        e *= d[k];
    }
    return true;
}

void kernel_ewise2_f16(const __half *a, const __half *b, __half *output, int opc, float alpha,
                       int d0, int d1, int d2, int d3, int d4, int d5,
                       int sa0, int sa1, int sa2, int sa3, int sa4, int sa5,
                       int sb0, int sb1, int sb2, int sb3, int sb4, int sb5) {
    int d[6] = {d0, d1, d2, d3, d4, d5}, sa[6] = {sa0, sa1, sa2, sa3, sa4, sa5}, sb[6] = {sb0, sb1, sb2, sb3, sb4, sb5};
    long nl = (long)d0 * d1 * d2 * d3 * d4 * d5;
    if (nl > 0x7fffffffL) { mb_error("ewise2_f16: %ld elements need 64-bit indices", nl); return; }
    int n = (int)nl;
#define MB_EW2_CASE(OPC, NAME, KIND, ...) case OPC: mb_launch(MB_KERNEL(cl_ewise2, k_ew2_##KIND##_##NAME), mb_grid(((long)n + 7) / 8), __VA_ARGS__); return;
    if (opc >= 0 && opc <= 5 && mb_contig6(d, sa) && mb_contig6(d, sb)) {
        switch (opc) {
            MB_EW2_CASE(0, add, v8, a, b, output, n, alpha) MB_EW2_CASE(1, sub, v8, a, b, output, n, alpha)
            MB_EW2_CASE(2, mul, v8, a, b, output, n, alpha) MB_EW2_CASE(3, div, v8, a, b, output, n, alpha)
            MB_EW2_CASE(4, max, v8, a, b, output, n, alpha) MB_EW2_CASE(5, min, v8, a, b, output, n, alpha)
        }
    }
    if (opc >= 0 && opc <= 5 && mb_contig6(d, sa)) {           // b: contiguous over the leading dims, broadcast over the trailing `inner`
        int t = 6;
        long inner = 1;
        while (t > 0 && (d[t - 1] == 1 || sb[t - 1] == 0)) { inner *= d[t - 1]; --t; }
        long e = 1;
        bool ok = true;
        for (int k = t - 1; k >= 0; --k) { if (d[k] > 1 && sb[k] != e) ok = false; e *= d[k]; }
        if (ok && inner > 0) {
            int in = (int)inner;
            switch (opc) {
                MB_EW2_CASE(0, add, bc, a, b, output, n, in, alpha) MB_EW2_CASE(1, sub, bc, a, b, output, n, in, alpha)
                MB_EW2_CASE(2, mul, bc, a, b, output, n, in, alpha) MB_EW2_CASE(3, div, bc, a, b, output, n, in, alpha)
                MB_EW2_CASE(4, max, bc, a, b, output, n, in, alpha) MB_EW2_CASE(5, min, bc, a, b, output, n, in, alpha)
            }
        }
    }
    mb_launch(MB_KERNEL(cl_ewise2, k_ewise2), mb_grid(n), a, b, output, n, opc, alpha, d1, d2, d3, d4, d5,
              sa0, sa1, sa2, sa3, sa4, sa5, sb0, sb1, sb2, sb3, sb4, sb5);
#undef MB_EW2_CASE
}
