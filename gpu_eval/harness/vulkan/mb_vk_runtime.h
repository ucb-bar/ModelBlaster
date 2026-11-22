// ModelBlaster Vulkan compute runtime (AMD RDNA via RADV, or any Vulkan 1.2 GPU).
//
// The Vulkan analogue of the CUDA harness (gpu_eval/harness/cuda): a shared
// library whose `kernel_<op>` entry points have EXACTLY the op's
// KernelSpec.signature (pipeline/reference_kernels.py), so walk_vulkan.py binds
// IR ops to them the way walk_cuda.py binds CUDA launchers.
//
// Pointer model. Every pointer in a kernel signature is a HOST address inside a
// buffer from mb_vk_alloc(): device memory that is host-visible and persistently
// mapped (on an APU such as the Steam Deck's Van Gogh this is the same DRAM).
// A launcher translates it to the GPU address of that byte
// (VK_KHR_buffer_device_address) and passes it in push constants; shaders
// dereference it with GL_EXT_buffer_reference. So views and offsets are plain
// pointer arithmetic, as in MB's C ABI, the CUDA device pointers and the OpenCL
// harness's SVM pointers.
//
// Ordering. Launches are recorded into one command buffer with a full
// compute->compute memory barrier between them (CUDA default-stream order) and
// submitted by mb_vk_sync() (or automatically every MB_VK_FLUSH dispatches).
// The host may read results only after mb_vk_sync().
//
// Errors never abort: the first one is kept and reported by mb_last_error() /
// mb_vk_error().
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int mb_vk_init(void);                       // 0 ok; idempotent
const char *mb_vk_device_name(void);
const char *mb_vk_info(void);               // one-line device / driver / memory summary
int mb_last_error(void);                    // 0 = no error (sticky until mb_vk_clear_error)
const char *mb_vk_error(void);
void mb_vk_clear_error(void);

void *mb_vk_alloc(size_t bytes);            // host-mapped, GPU-visible; 256-byte aligned
size_t mb_vk_mark(void);                    // arena: allocations after a mark are freed by release(mark)
void mb_vk_release(size_t mark);
void *mb_vk_scratch(int slot, size_t bytes);  // launcher scratch (grow-only, not in the arena)
uint64_t mb_vk_devaddr(const void *p);      // GPU address of a host pointer (0 for NULL / foreign)
size_t mb_vk_bytes_in_use(void);

void mb_vk_sync(void);                      // submit what is recorded, wait, make it host-visible
void mb_vk_timestamps(int on);              // record a GPU timestamp at every mb_vk_ts() call
int mb_vk_ts(void);                         // record one; returns its index (-1 when off)
int mb_vk_ts_count(void);
double mb_vk_ts_ns(int i);                  // valid after mb_vk_sync()
void mb_vk_ts_reset(void);
long mb_vk_dispatch_count(void);
int mb_vk_capture_begin(void);              // record (not run) launches until capture_end -> graph id
int mb_vk_capture_end(void);
void mb_vk_replay(int graph);               // submit a captured graph again and wait
int mb_vk_graph_cbs(int graph);             // command buffers in a graph (-1: none)
uint32_t mb_vk_max_group_x(void);           // per-dimension workgroup-count limit
// Packed weights: register p (an mb_vk_alloc buffer) as the [N, K] weight w laid
// out for the skinny GEMM (per 64-column block and k-quad, the 64 columns' 4
// values contiguous: shaders/skinny_body.glsl, walk_vulkan.pack_weight), fp32,
// or fp16 when f16 != 0. p == NULL unregisters. A registration is dropped when
// mb_vk_release frees either buffer.
void mb_vk_set_packed(const void *w, const void *p, int f16);
int mb_vk_packed_count(void);
int mb_vk_has_int_dot(void);                // packed int8 dot products enabled (VK_KHR_shader_integer_dot_product)

#ifdef __cplusplus
}

// ---- launcher side (C++) -----------------------------------------------------
// Push constants: all pointers first (8 bytes each, as uint64 GPU addresses),
// then 4-byte scalars, in the order the shader's push_constant block declares.
struct mb_pc {
    uint8_t b[256];
    uint32_t n = 0;
    mb_pc &p(const void *ptr);              // pointer -> GPU address (NULL -> 0)
    mb_pc &i(int v);
    mb_pc &u(uint32_t v);
    mb_pc &f(float v);
};

// Look up (and on first use create) the compute pipeline for an embedded SPIR-V module.
int mb_vk_pipeline(const char *name);
// The same with specialization constants: constant_id i = consts[i] (4-byte ints);
// one pipeline per (name, values).
int mb_vk_pipeline_spec(const char *name, const int *consts, int nconsts);
void mb_vk_dispatch(int pipe, uint32_t gx, uint32_t gy, uint32_t gz, const mb_pc &pc);
// 1-D launch of n invocations with local size 256: the grid is folded into 2-D
// when it exceeds the per-dimension limit (shaders use MB_GID from mb_common.glsl).
void mb_vk_launch1d(int pipe, long n, const mb_pc &pc);
struct mb_packed { const void *p; int f16; };
const mb_packed *mb_vk_find_packed(const void *w);   // NULL: none registered
#endif
