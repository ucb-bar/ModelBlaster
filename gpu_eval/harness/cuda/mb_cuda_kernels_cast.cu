// Hand-written CUDA casts between the IR tensor dtypes of a mixed IR
// (`extract_graph --quant native`, docs/tensor_dtypes.md): one extern "C"
// entry per `cast_<src>_to_<dst>` KernelSpec (pipeline/reference_kernels.py
// `_cast_spec`), same signature in the CUDA spelling. Float -> float rounds
// once (round-to-nearest-even); integers / bool convert exactly through float
// (they are integer-valued f32 inside the graph); f32 -> integer truncates
// toward zero, -> bool (b8, one byte) is `!= 0`. Elementwise, one thread per
// element; the conversions compile for sm_75 too.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Each .so links its own (static) cudart, so error state must be read from
// inside the library that launched.
extern "C" int mb_last_error(void) { return (int)cudaGetLastError(); }

#define MB_THREADS 256
// at least one block: a zero-size launch is an invalid configuration (every
// kernel bounds-checks its thread id)
#define MB_GRID(n) (int)(((long)(n) > 0 ? (long)(n) + MB_THREADS - 1 : MB_THREADS) / MB_THREADS)
#define MB_TID ((long)blockIdx.x * blockDim.x + threadIdx.x)

__global__ void k_cast_b8_to_f32(const uint8_t *in, float *out, int n) { long i = MB_TID; if (i < n) out[i] = (float)in[i]; }
__global__ void k_cast_bf16_to_f16(const __nv_bfloat16 *in, __half *out, int n) { long i = MB_TID; if (i < n) out[i] = __float2half_rn(__bfloat162float(in[i])); }
__global__ void k_cast_bf16_to_f32(const __nv_bfloat16 *in, float *out, int n) { long i = MB_TID; if (i < n) out[i] = __bfloat162float(in[i]); }
__global__ void k_cast_f16_to_bf16(const __half *in, __nv_bfloat16 *out, int n) { long i = MB_TID; if (i < n) out[i] = __float2bfloat16_rn(__half2float(in[i])); }
__global__ void k_cast_f16_to_f32(const __half *in, float *out, int n) { long i = MB_TID; if (i < n) out[i] = __half2float(in[i]); }
__global__ void k_cast_f32_to_b8(const float *in, uint8_t *out, int n) { long i = MB_TID; if (i < n) out[i] = (uint8_t)(in[i] != 0.0f); }
__global__ void k_cast_f32_to_bf16(const float *in, __nv_bfloat16 *out, int n) { long i = MB_TID; if (i < n) out[i] = __float2bfloat16_rn(in[i]); }
__global__ void k_cast_f32_to_f16(const float *in, __half *out, int n) { long i = MB_TID; if (i < n) out[i] = __float2half_rn(in[i]); }
__global__ void k_cast_f32_to_i32(const float *in, int32_t *out, int n) { long i = MB_TID; if (i < n) out[i] = (int32_t)in[i]; }
__global__ void k_cast_f32_to_i64(const float *in, int64_t *out, int n) { long i = MB_TID; if (i < n) out[i] = (int64_t)in[i]; }
__global__ void k_cast_f32_to_u8(const float *in, uint8_t *out, int n) { long i = MB_TID; if (i < n) out[i] = (uint8_t)in[i]; }
__global__ void k_cast_i32_to_f32(const int32_t *in, float *out, int n) { long i = MB_TID; if (i < n) out[i] = (float)in[i]; }
__global__ void k_cast_i64_to_f32(const int64_t *in, float *out, int n) { long i = MB_TID; if (i < n) out[i] = (float)in[i]; }
__global__ void k_cast_u8_to_f32(const uint8_t *in, float *out, int n) { long i = MB_TID; if (i < n) out[i] = (float)in[i]; }

extern "C" void kernel_cast_b8_to_f32(const uint8_t *in, float *out, int n) { k_cast_b8_to_f32<<<MB_GRID(n), MB_THREADS>>>(in, out, n); }
extern "C" void kernel_cast_bf16_to_f16(const __nv_bfloat16 *in, __half *out, int n) { k_cast_bf16_to_f16<<<MB_GRID(n), MB_THREADS>>>(in, out, n); }
extern "C" void kernel_cast_bf16_to_f32(const __nv_bfloat16 *in, float *out, int n) { k_cast_bf16_to_f32<<<MB_GRID(n), MB_THREADS>>>(in, out, n); }
extern "C" void kernel_cast_f16_to_bf16(const __half *in, __nv_bfloat16 *out, int n) { k_cast_f16_to_bf16<<<MB_GRID(n), MB_THREADS>>>(in, out, n); }
extern "C" void kernel_cast_f16_to_f32(const __half *in, float *out, int n) { k_cast_f16_to_f32<<<MB_GRID(n), MB_THREADS>>>(in, out, n); }
extern "C" void kernel_cast_f32_to_b8(const float *in, uint8_t *out, int n) { k_cast_f32_to_b8<<<MB_GRID(n), MB_THREADS>>>(in, out, n); }
extern "C" void kernel_cast_f32_to_bf16(const float *in, __nv_bfloat16 *out, int n) { k_cast_f32_to_bf16<<<MB_GRID(n), MB_THREADS>>>(in, out, n); }
extern "C" void kernel_cast_f32_to_f16(const float *in, __half *out, int n) { k_cast_f32_to_f16<<<MB_GRID(n), MB_THREADS>>>(in, out, n); }
extern "C" void kernel_cast_f32_to_i32(const float *in, int32_t *out, int n) { k_cast_f32_to_i32<<<MB_GRID(n), MB_THREADS>>>(in, out, n); }
extern "C" void kernel_cast_f32_to_i64(const float *in, int64_t *out, int n) { k_cast_f32_to_i64<<<MB_GRID(n), MB_THREADS>>>(in, out, n); }
extern "C" void kernel_cast_f32_to_u8(const float *in, uint8_t *out, int n) { k_cast_f32_to_u8<<<MB_GRID(n), MB_THREADS>>>(in, out, n); }
extern "C" void kernel_cast_i32_to_f32(const int32_t *in, float *out, int n) { k_cast_i32_to_f32<<<MB_GRID(n), MB_THREADS>>>(in, out, n); }
extern "C" void kernel_cast_i64_to_f32(const int64_t *in, float *out, int n) { k_cast_i64_to_f32<<<MB_GRID(n), MB_THREADS>>>(in, out, n); }
extern "C" void kernel_cast_u8_to_f32(const uint8_t *in, float *out, int n) { k_cast_u8_to_f32<<<MB_GRID(n), MB_THREADS>>>(in, out, n); }
