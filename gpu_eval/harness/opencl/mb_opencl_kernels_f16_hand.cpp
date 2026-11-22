// fp16 OpenCL seeds whose `_f16` KernelSpec has a signature that differs from
// the fp32 op's (the other `_f16` seeds are generated from the fp32 ones by
// pipeline/opencl_seeds.py:to_f16). Same contract: binary16 storage
// (vload_half / vstore_half), fp32 math, one rounding per stored value.
#include "mb_opencl_runtime.h"

// softmax over the last axis with a fused input scale (SDPA's 1/sqrt(d)); one
// work-group per row, the exponentials recomputed in the normalising pass
// rather than stored in fp16 and rescaled (which would round twice)
static mb_program cl_softmax_scaled = MB_CL(R"CL(
float wg_max(float v, __local float *sh) {
    int t = get_local_id(0);
    sh[t] = v; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] = fmax(sh[t], sh[t + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float r = sh[0]; barrier(CLK_LOCAL_MEM_FENCE);
    return r;
}
float wg_sum(mb_ksum v, __local float *sh) {
    int t = get_local_id(0);
    sh[t] = v.s; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float r = sh[0]; barrier(CLK_LOCAL_MEM_FENCE);
    return r;
}
__kernel void k_softmax_scaled(__global const half *x, __global half *y, int K, float scale) {
    __local float sh[MB_THREADS];
    long r = get_group_id(0); int t = get_local_id(0), nt = get_local_size(0);
    __global const half *xr = x + r * K; __global half *yr = y + r * K;
    float mx = -INFINITY;
    for (int k = t; k < K; k += nt) mx = fmax(mx, vload_half(k, xr) * scale);
    mx = wg_max(mx, sh);
    mb_ksum s = mb_ksum0();
    for (int k = t; k < K; k += nt) mb_kadd(&s, exp(vload_half(k, xr) * scale - mx));
    float inv = 1.0f / wg_sum(s, sh);
    for (int k = t; k < K; k += nt) vstore_half(exp(vload_half(k, xr) * scale - mx) * inv, k, yr);
}
)CL");
void kernel_softmax_f16(const __half *input, __half *output, int M, int K, float input_scale) {
    mb_launch(MB_KERNEL(cl_softmax_scaled, k_softmax_scaled), mb_groups(M), input, output, K, input_scale);
}

// torch AdaptiveAvgPool2d windows: [floor(o*I/O), ceil((o+1)*I/O))
static mb_program cl_adaptive_avg = MB_CL(R"CL(
__kernel void k_adaptive_avg(__global const half *x, __global half *y, int N, int C, int IH, int IW, int OH, int OW) {
    long i = MB_TID; long tot = (long)N * C * OH * OW;
    if (i >= tot) return;
    int ow = i % OW, oh = (i / OW) % OH; long nc = i / ((long)OW * OH);
    int h0 = (oh * IH) / OH, h1 = ((oh + 1) * IH + OH - 1) / OH;
    int w0 = (ow * IW) / OW, w1 = ((ow + 1) * IW + OW - 1) / OW;
    mb_ksum s = mb_ksum0();
    for (int h = h0; h < h1; h++)
        for (int w = w0; w < w1; w++) mb_kadd(&s, vload_half((nc * IH + h) * IW + w, x));
    vstore_half(s.s / (float)((h1 - h0) * (w1 - w0)), i, y);
}
)CL");
void kernel_adaptive_avg_pool2d_f16(const __half *input, __half *output, int N, int C, int IH, int IW,
                                    int OH, int OW) {
    long tot = (long)N * C * OH * OW;
    mb_launch(MB_KERNEL(cl_adaptive_avg, k_adaptive_avg), mb_grid(tot), input, output, N, C, IH, IW, OH, OW);
}

// channel concat [N, Ca, H, W] ++ [N, Cb, H, W] (an exact copy: ushort moves)
static mb_program cl_cat2_h = MB_CL(R"CL(
__kernel void k_copy_c_h(__global const ushort *src, __global ushort *dst, int N, int c, int C, int c_off, int HW) {
    long i = MB_TID; long tot = (long)N * c * HW; if (i >= tot) return;
    long n = i / ((long)c * HW), r = i % ((long)c * HW);
    dst[(n * C + c_off) * HW + r] = src[i];
}
)CL");
void kernel_cat2_c1_f16(const __half *a, const __half *b, __half *output, int N, int H, int W, int Ca, int Cb) {
    int C = Ca + Cb, HW = H * W;
    mb_launch(MB_KERNEL(cl_cat2_h, k_copy_c_h), mb_grid((long)N * Ca * HW), a, output, N, Ca, C, 0, HW);
    mb_launch(MB_KERNEL(cl_cat2_h, k_copy_c_h), mb_grid((long)N * Cb * HW), b, output, N, Cb, C, Ca, HW);
}

// The fp32 k_sdpa accumulates P.V in its output row, which in fp16 would be an
// fp16 accumulation: one work-item per output element, all sums in fp32.
static mb_program cl_sdpa_h = MB_CL(R"CL(
__kernel void k_sdpa_h(__global const half *Q, __global const half *Kt, __global const half *V, __global half *O,
                       int BH, int S, int D) {
    long t = MB_TID; if (t >= (long)BH * S * D) return;
    int d0 = t % D; long r = t / D; long bh = r / S; int i = r % S;
    __global const half *q = Q + (bh * S + i) * D;
    __global const half *Kb = Kt + bh * S * D;
    __global const half *Vb = V + bh * S * D;
    float scale = rsqrt((float)D), mx = -INFINITY;
    for (int j = 0; j < S; j++) {
        mb_ksum s = mb_ksum0();
        for (int d = 0; d < D; d++) mb_kadd(&s, vload_half(d, q) * vload_half((long)j * D + d, Kb));
        mx = fmax(mx, s.s * scale);
    }
    float sum = 0.0f, acc = 0.0f;
    for (int j = 0; j < S; j++) {
        mb_ksum s = mb_ksum0();
        for (int d = 0; d < D; d++) mb_kadd(&s, vload_half(d, q) * vload_half((long)j * D + d, Kb));
        float p = exp(s.s * scale - mx); sum += p;
        acc += p * vload_half((long)j * D + d0, Vb);
    }
    vstore_half(acc / sum, t, O);
}
)CL");
void kernel_sdpa_f16(const __half *Q, const __half *K, const __half *V, __half *output, int BH, int S, int D) {
    mb_launch(MB_KERNEL(cl_sdpa_h, k_sdpa_h), mb_grid((long)BH * S * D), Q, K, V, output, BH, S, D);
}
