// fp16 <-> fp32 storage casts at precision-policy island boundaries (pipeline/precision_casts.py,
// pipeline/precision_policy.py). Only partitions of a policy IR use them; kb_export._MixedLibrary adds this file to
// the fp16 and fp32 hand libraries. Same helpers as the hand libraries (MB_TID, mb_blocks, mb_threads).
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define MB_THREADS 256
#define MB_TID ((long)blockIdx.x * blockDim.x + threadIdx.x)
static inline int mb_sm_count() {
    static int sms = 0;
    if (!sms) {
        int d = 0;
        cudaGetDevice(&d);
        if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, d) != cudaSuccess || sms <= 0) sms = 1;
    }
    return sms;
}
static inline int mb_threads(long n) {
    int b = MB_THREADS;
    while (b > 32 && (long)b * mb_sm_count() > n) b /= 2;
    return b;
}
static inline int mb_blocks(long n) { int b = mb_threads(n); return (int)((n > 0 ? n + b - 1 : b) / b); }

__global__ void k_cast_h2f(const __half *x, float *y, long n) { long i = MB_TID; if (i < n) y[i] = __half2float(x[i]); }
extern "C" void kernel_cast_f16_to_f32(const __half *input, float *output, int n) {
    k_cast_h2f<<<mb_blocks(n), mb_threads(n)>>>(input, output, n);
}

__global__ void k_cast_f2h(const float *x, __half *y, long n) { long i = MB_TID; if (i < n) y[i] = __float2half_rn(x[i]); }
extern "C" void kernel_cast_f32_to_f16(const float *input, __half *output, int n) {
    k_cast_f2h<<<mb_blocks(n), mb_threads(n)>>>(input, output, n);
}
