// bf16 launchers that can't be generated from mb_cuda_kernels.cu: their fp32
// code rounds an intermediate in the output buffer, which in bf16 would be a
// second rounding (or a bf16 accumulation). f16_kernels.py appends these to
// the generated mb_cuda_kernels_bf16.cu in place of the generated entries of
// the same name. Same contract: __nv_bfloat16 storage, fp32 math, one rounding
// per stored value. Signatures are the fp32 ops' (every `_bf16` KernelSpec is
// derived from its fp32 spec).
#include <cuda_bf16.h>

// softmax over the last axis: the fp32 kernel stores the exponentials in the
// output and rescales them in place; here they are recomputed in the
// normalising pass, so each output is rounded once
__global__ void k_softmax_bf(const __nv_bfloat16 *x, __nv_bfloat16 *y, int M, int K) {
    long r = MB_TID; if (r >= M) return;
    const __nv_bfloat16 *xr = x + r * K; __nv_bfloat16 *yr = y + r * K;
    float mx = -INFINITY; for (int k = 0; k < K; k++) mx = fmaxf(mx, mb_ld(xr[k]));
    double s = 0.0; for (int k = 0; k < K; k++) s += expf(mb_ld(xr[k]) - mx);
    float inv = (float)(1.0 / s);
    for (int k = 0; k < K; k++) yr[k] = expf(mb_ld(xr[k]) - mx) * inv;
}
extern "C" void kernel_softmax_bf16(const __nv_bfloat16 *input, __nv_bfloat16 *output, int M, int K) {
    k_softmax_bf<<<mb_blocks(M), mb_threads(M)>>>(input, output, M, K);
}

// The fp32 k_sdpa accumulates P.V in its output row, which in bf16 would be
// a bf16 accumulation: one thread per output element instead, all sums in
// fp32 (scores recomputed per d; a correctness seed, like the fp32 one).
__global__ void k_sdpa_bf(const __nv_bfloat16 *Q, const __nv_bfloat16 *Kt, const __nv_bfloat16 *V,
                          __nv_bfloat16 *O, int BH, int S, int D) {
    long t = MB_TID; if (t >= (long)BH * S * D) return;
    int d0 = t % D; long r = t / D; long bh = r / S; int i = r % S;
    const __nv_bfloat16 *q = Q + (bh * S + i) * D, *Kb = Kt + bh * S * D, *Vb = V + bh * S * D;
    float scale = rsqrtf((float)D), mx = -INFINITY;
    for (int j = 0; j < S; j++) {
        float s = 0; for (int d = 0; d < D; d++) s += mb_ld(q[d]) * mb_ld(Kb[(long)j * D + d]);
        mx = fmaxf(mx, s * scale);
    }
    float sum = 0, acc = 0;
    for (int j = 0; j < S; j++) {
        float s = 0; for (int d = 0; d < D; d++) s += mb_ld(q[d]) * mb_ld(Kb[(long)j * D + d]);
        float p = expf(s * scale - mx); sum += p;
        acc += p * mb_ld(Vb[(long)j * D + d0]);
    }
    O[t] = acc / sum;
}
extern "C" void kernel_sdpa_bf16(const __nv_bfloat16 *Q, const __nv_bfloat16 *K, const __nv_bfloat16 *V,
                                 __nv_bfloat16 *output, int BH, int S, int D) {
    k_sdpa_bf<<<mb_blocks((long)BH * S * D), mb_threads((long)BH * S * D)>>>(Q, K, V, output, BH, S, D);
}
