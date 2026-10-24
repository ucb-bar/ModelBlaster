// ModelBlaster OpenCL runtime for multi-slot problems (the walker head).
//
// Every slot is C++11 that holds its OpenCL C programs as raw strings plus a
// host launcher with EXACTLY the op's KernelSpec signature. Pointers in that
// signature are coarse-grained SVM pointers (clSVMAlloc), so views and offsets
// are plain pointer arithmetic, as in MB's C ABI and the CUDA slots.
//
//   static mb_program cl_relu = MB_CL(R"CL(
//   __kernel void k_relu(__global const float *x, __global float *y, int n) {
//       long i = get_global_id(0); if (i < n) y[i] = fmax(x[i], 0.0f);
//   }
//   )CL");
//   void kernel_relu(const float *input, float *output, int n) {
//       mb_launch(MB_KERNEL(cl_relu, k_relu), mb_grid(n), input, output, n);
//   }
//
// A program is built on first use (OpenCL C 1.2, plus its own options and the
// device prelude below). Every enqueue goes through mb_launch / mb_copy /
// mb_zero, which record a profiling event per command; the driver turns those
// into per-op, per-slot and per-kernel times (cl_event START..END, the Adreno
// flow's methodology). Errors never abort: the first one is kept in
// mb_rt.err and reported by the driver as MB_ERROR.
#pragma once
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 200
#endif
#include <CL/cl.h>
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <string>
#include <vector>

#define MB_THREADS 256

// Device prelude: prepended to every program.
static const char *mb_cl_prelude = R"CL(
#define MB_THREADS 256
#define MB_TID ((long)get_global_id(0))
// Compensated (Kahan) float sum: Adreno has no fp64, and a plain fp32 sum over
// 1e3..1e6 terms misses the 1e-3 checks on ill-conditioned graphs.
typedef struct { float s, c; } mb_ksum;
mb_ksum mb_ksum0(void) { mb_ksum a; a.s = 0.0f; a.c = 0.0f; return a; }
void mb_kadd(mb_ksum *a, float v) { float y = v - a->c; float t = a->s + y; a->c = (t - a->s) - y; a->s = t; }
// flat index -> 6-d coordinates (d0 implied)
void mb_idx6(long i, int d1, int d2, int d3, int d4, int d5, long *o) {
    long r = i;
    o[5] = r % d5; r /= d5; o[4] = r % d4; r /= d4; o[3] = r % d3; r /= d3;
    o[2] = r % d2; r /= d2; o[1] = r % d1; r /= d1; o[0] = r;
}
// float atomic add (OpenCL 1.2 has only integer atomics)
void mb_atomic_addf(volatile __global float *p, float v) {
    union { unsigned int u; float f; } o, n;
    do { o.f = *p; n.f = o.f + v; } while (atomic_cmpxchg((volatile __global unsigned int *)p, o.u, n.u) != o.u);
}
)CL";

struct mb_program {
    const char *src;
    const char *opts;            // extra build options, e.g. "-cl-mad-enable -DTILE=16"
    cl_program prog;
    int failed;
    void *kernels;               // std::vector<std::pair<std::string, cl_kernel>> *
    void *scratch;
    size_t scratch_bytes;
};
#define MB_CL(src) {src, "", NULL, 0, NULL, NULL, 0}
#define MB_CL_OPTS(opts, src) {src, opts, NULL, 0, NULL, NULL, 0}

struct mb_event { cl_event ev; int op; const char *kernel; mb_program *prog; };
struct mb_kinfo { mb_program *prog; const char *name; cl_kernel k; int op; };   // name: interned

struct mb_runtime {
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    cl_context ctx = NULL;
    cl_command_queue queue = NULL;
    int failed = 0;
    std::string err;
    int cur_op = -1;             // op being run (set by the driver), for attribution
    int record = 1;              // record profiling events
    std::vector<mb_event> events;
    std::vector<mb_kinfo> kinfo; // every kernel created, with the op that first used it
    std::string build_opts;      // global extra options (env MB_CL_OPTS)
    std::deque<std::string> names;   // kernel names (stable addresses for events)
    std::map<std::pair<cl_kernel, unsigned>, cl_kernel> variants;   // (kernel, NULL-arg mask) -> kernel object
};
mb_runtime mb_rt;

void mb_error(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void mb_error(const char *fmt, ...) {
    if (mb_rt.failed) return;
    char buf[8192];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    mb_rt.failed = 1;
    mb_rt.err = buf;
}

cl_kernel mb_kernel(mb_program *p, const char *name) {
    if (mb_rt.failed || p->failed) return NULL;
    cl_int e;
    if (!p->prog) {
        const char *srcs[2] = {mb_cl_prelude, p->src};
        p->prog = clCreateProgramWithSource(mb_rt.ctx, 2, srcs, NULL, &e);
        std::string opts = std::string("-cl-std=CL1.2 ") + (p->opts ? p->opts : "") + " " + mb_rt.build_opts;
        e = clBuildProgram(p->prog, 1, &mb_rt.device, opts.c_str(), NULL, NULL);
        if (e != CL_SUCCESS) {
            size_t n = 0;
            clGetProgramBuildInfo(p->prog, mb_rt.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &n);
            std::string log(n, '\0');
            clGetProgramBuildInfo(p->prog, mb_rt.device, CL_PROGRAM_BUILD_LOG, n, &log[0], NULL);
            p->failed = 1;
            mb_error("cl_build (op %d, kernel %s): error %d\n%s", mb_rt.cur_op, name, e, log.c_str());
            return NULL;
        }
        p->kernels = new std::vector<std::pair<std::string, cl_kernel> >();
    }
    auto *ks = (std::vector<std::pair<std::string, cl_kernel> > *)p->kernels;
    for (auto &kv : *ks)
        if (kv.first == name) return kv.second;
    cl_kernel k = clCreateKernel(p->prog, name, &e);
    if (e != CL_SUCCESS) { mb_error("clCreateKernel %s: error %d", name, e); return NULL; }
    ks->push_back(std::make_pair(std::string(name), k));
    mb_rt.names.push_back(name);
    mb_rt.kinfo.push_back({p, mb_rt.names.back().c_str(), k, mb_rt.cur_op});
    return k;
}
#define MB_KERNEL(prog, name) mb_kernel(&(prog), #name)

// Grow-only scratch buffer owned by a program (one per slot program).
void *mb_scratch(mb_program *p, size_t bytes) {
    if (bytes > p->scratch_bytes) {
        if (p->scratch) { clFinish(mb_rt.queue); clSVMFree(mb_rt.ctx, p->scratch); }
        p->scratch = clSVMAlloc(mb_rt.ctx, CL_MEM_READ_WRITE, bytes, 0);
        p->scratch_bytes = p->scratch ? bytes : 0;
        if (!p->scratch) mb_error("scratch alloc of %zu bytes failed", bytes);
    }
    return p->scratch;
}
#define MB_SCRATCH(prog, bytes) mb_scratch(&(prog), (bytes))

// ---- NDRange
struct mb_range { cl_uint dims; size_t g[3]; size_t l[3]; };
static size_t mb_round_up(long n, long m) { if (n < 1) n = 1; return (size_t)((n + m - 1) / m * m); }
// one work-item per element, rounded up to whole work-groups (at least one)
mb_range mb_grid(long n, int local = MB_THREADS) { return {1, {mb_round_up(n, local), 1, 1}, {(size_t)local, 1, 1}}; }
// CUDA-style <<<groups, local>>>: `groups` work-groups of `local` items
mb_range mb_groups(long groups, int local = MB_THREADS) { return {1, {(size_t)(groups > 0 ? groups : 1) * local, 1, 1}, {(size_t)local, 1, 1}}; }
mb_range mb_range2(long gx, long gy, int lx, int ly) {
    return {2, {mb_round_up(gx, lx), mb_round_up(gy, ly), 1}, {(size_t)lx, (size_t)ly, 1}};
}
mb_range mb_range3(long gx, long gy, long gz, int lx, int ly, int lz) {
    return {3, {mb_round_up(gx, lx), mb_round_up(gy, ly), mb_round_up(gz, lz)}, {(size_t)lx, (size_t)ly, (size_t)lz}};
}

// ---- kernel arguments: SVM pointers, scalars, __local sizes, cl_mem
struct mb_local { size_t bytes; };
static cl_int mb_arg(cl_kernel k, cl_uint i, mb_local v) { return clSetKernelArg(k, i, v.bytes, NULL); }
static cl_int mb_arg(cl_kernel k, cl_uint i, cl_mem v) { return clSetKernelArg(k, i, sizeof(cl_mem), &v); }
static cl_int mb_arg(cl_kernel k, cl_uint i, std::nullptr_t) { return clSetKernelArg(k, i, sizeof(cl_mem), NULL); }
template <class T> static cl_int mb_arg(cl_kernel k, cl_uint i, T *p) {
    return p ? clSetKernelArgSVMPointer(k, i, (const void *)p) : clSetKernelArg(k, i, sizeof(cl_mem), NULL);
}
static cl_int mb_arg(cl_kernel k, cl_uint i, int v) { return clSetKernelArg(k, i, sizeof v, &v); }
static cl_int mb_arg(cl_kernel k, cl_uint i, unsigned v) { return clSetKernelArg(k, i, sizeof v, &v); }
static cl_int mb_arg(cl_kernel k, cl_uint i, long v) { return clSetKernelArg(k, i, sizeof v, &v); }
static cl_int mb_arg(cl_kernel k, cl_uint i, unsigned long v) { return clSetKernelArg(k, i, sizeof v, &v); }
static cl_int mb_arg(cl_kernel k, cl_uint i, long long v) { return clSetKernelArg(k, i, sizeof v, &v); }
static cl_int mb_arg(cl_kernel k, cl_uint i, float v) { return clSetKernelArg(k, i, sizeof v, &v); }
static cl_int mb_arg(cl_kernel k, cl_uint i, cl_float2 v) { return clSetKernelArg(k, i, sizeof v, &v); }
static cl_int mb_arg(cl_kernel k, cl_uint i, cl_float4 v) { return clSetKernelArg(k, i, sizeof v, &v); }
// no fp64 on Adreno: pass float (write 0.5f, not 0.5)
static cl_int mb_arg(cl_kernel k, cl_uint i, double v) = delete;

static int mb_set_args(cl_kernel, cl_uint) { return 0; }
template <class A, class... R> static int mb_set_args(cl_kernel k, cl_uint i, A a, R... r) {
    cl_int e = mb_arg(k, i, a);
    if (e != CL_SUCCESS) { mb_error("clSetKernelArg #%u: error %d (type/size mismatch?)", i, e); return 1; }
    return mb_set_args(k, i + 1, r...);
}

static void mb_record(cl_event ev, const char *what, mb_program *p) {
    if (mb_rt.record) mb_rt.events.push_back({ev, mb_rt.cur_op, what, p});
    else clReleaseEvent(ev);
}

static const char *mb_kernel_name(cl_kernel k) {
    for (auto &ki : mb_rt.kinfo) if (ki.k == k) return ki.name;
    return "?";
}

// NULL pointer arguments: some ICDs (NVIDIA) reject a NULL argument once that
// argument held an SVM pointer, so each pattern of NULL arguments gets its own
// kernel object (bias-less and biased calls of one conv/linear slot)
template <class T> static void mb_null(unsigned &m, cl_uint i, T *p) { if (!p) m |= 1u << (i & 31); }
static void mb_null(unsigned &m, cl_uint i, std::nullptr_t) { m |= 1u << (i & 31); }
template <class T> static void mb_null(unsigned &, cl_uint, const T &) {}
static void mb_null_mask(unsigned &, cl_uint) {}
template <class A, class... R> static void mb_null_mask(unsigned &m, cl_uint i, A a, R... r) {
    mb_null(m, i, a); mb_null_mask(m, i + 1, r...);
}
static cl_kernel mb_variant(cl_kernel k, unsigned mask) {
    auto key = std::make_pair(k, mask);
    auto it = mb_rt.variants.find(key);
    if (it != mb_rt.variants.end()) return it->second;
    for (size_t j = 0; j < mb_rt.kinfo.size(); ++j) {
        mb_kinfo ki = mb_rt.kinfo[j];
        if (ki.k != k) continue;
        cl_int e;
        cl_kernel v = clCreateKernel(ki.prog->prog, ki.name, &e);
        if (e != CL_SUCCESS) { mb_error("clCreateKernel %s: error %d", ki.name, e); return NULL; }
        mb_rt.kinfo.push_back({ki.prog, ki.name, v, ki.op});
        return mb_rt.variants[key] = v;
    }
    return k;
}

template <class... A> void mb_launch(cl_kernel k, const mb_range &r, A... args) {
    if (mb_rt.failed || !k) return;
    unsigned nulls = 0;
    mb_null_mask(nulls, 0, args...);
    if (nulls && !(k = mb_variant(k, nulls))) return;
    if (mb_set_args(k, 0, args...)) return;
    cl_event ev = NULL;
    cl_int e = clEnqueueNDRangeKernel(mb_rt.queue, k, r.dims, NULL, r.g, r.l[0] ? r.l : NULL, 0, NULL, &ev);
    if (e != CL_SUCCESS) {
        mb_error("enqueue %s (op %d): error %d (global %zu x %zu x %zu, local %zu x %zu x %zu)", mb_kernel_name(k),
                 mb_rt.cur_op, e, r.g[0], r.g[1], r.g[2], r.l[0], r.l[1], r.l[2]);
        return;
    }
    mb_program *p = NULL;
    for (auto &ki : mb_rt.kinfo) if (ki.k == k) { p = ki.prog; break; }
    mb_record(ev, mb_kernel_name(k), p);
}

// device-to-device copy / zero fill, on the queue and timed like a kernel
void mb_copy(void *dst, const void *src, size_t bytes) {
    if (mb_rt.failed || !bytes) return;
    cl_event ev = NULL;
    cl_int e = clEnqueueSVMMemcpy(mb_rt.queue, CL_FALSE, dst, src, bytes, 0, NULL, &ev);
    if (e != CL_SUCCESS) { mb_error("mb_copy %zu bytes: error %d", bytes, e); return; }
    mb_record(ev, "mb_copy", NULL);
}
void mb_zero(void *dst, size_t bytes) {
    if (mb_rt.failed || !bytes) return;
    cl_event ev = NULL;
    cl_uchar z = 0;
    cl_int e = clEnqueueSVMMemFill(mb_rt.queue, dst, &z, 1, bytes, 0, NULL, &ev);
    if (e != CL_SUCCESS) { mb_error("mb_zero %zu bytes: error %d", bytes, e); return; }
    mb_record(ev, "mb_zero", NULL);
}

// ---- driver-facing setup (the driver owns buffers and checks)
int mb_cl_init() {
    const char *pe = getenv("MB_CL_PLATFORM"), *de = getenv("MB_CL_DEVICE"), *oe = getenv("MB_CL_OPTS");
    if (oe) mb_rt.build_opts = oe;
    cl_platform_id plats[16];
    cl_uint np = 0;
    if (clGetPlatformIDs(16, plats, &np) != CL_SUCCESS || np == 0) { mb_error("no OpenCL platform"); return 1; }
    int pi = pe ? atoi(pe) : -1, di = de ? atoi(de) : 0;
    for (cl_uint p = 0; p < np && !mb_rt.device; ++p) {
        if (pi >= 0 && (int)p != pi) continue;
        cl_device_id devs[16];
        cl_uint nd = 0;
        if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 16, devs, &nd) != CL_SUCCESS || nd == 0) continue;
        if (di < (int)nd) { mb_rt.platform = plats[p]; mb_rt.device = devs[di]; }
    }
    if (!mb_rt.device) { mb_error("no OpenCL GPU device (MB_CL_PLATFORM/MB_CL_DEVICE)"); return 1; }
    cl_int e;
    mb_rt.ctx = clCreateContext(NULL, 1, &mb_rt.device, NULL, NULL, &e);
    if (e != CL_SUCCESS) { mb_error("clCreateContext: %d", e); return 1; }
    cl_queue_properties qp[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    mb_rt.queue = clCreateCommandQueueWithProperties(mb_rt.ctx, mb_rt.device, qp, &e);
    if (e != CL_SUCCESS) { mb_error("clCreateCommandQueueWithProperties: %d", e); return 1; }
    cl_device_svm_capabilities svm = 0;
    clGetDeviceInfo(mb_rt.device, CL_DEVICE_SVM_CAPABILITIES, sizeof svm, &svm, NULL);
    if (!(svm & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)) { mb_error("device has no coarse-grain SVM"); return 1; }
    return 0;
}

// ---- accessors for the driver (a separate TU)
cl_device_id mb_device() { return mb_rt.device; }
cl_context mb_context() { return mb_rt.ctx; }
cl_command_queue mb_queue() { return mb_rt.queue; }
const char *mb_take_error() {     // the first error since the last call, or NULL
    static std::string last;
    if (!mb_rt.failed) return NULL;
    last = mb_rt.err;
    mb_rt.failed = 0;
    mb_rt.err.clear();
    return last.c_str();
}
void mb_set_op(int i) { mb_rt.cur_op = i; }
void mb_set_record(int on) { mb_rt.record = on; }
int mb_num_events() { return (int)mb_rt.events.size(); }
// op, kernel name, START/END ns of recorded command i
void mb_event_at(int i, int *op, const char **kernel, cl_ulong *t0, cl_ulong *t1) {
    mb_event &e = mb_rt.events[i];
    *op = e.op; *kernel = e.kernel;
    clGetEventProfilingInfo(e.ev, CL_PROFILING_COMMAND_START, sizeof *t0, t0, NULL);
    clGetEventProfilingInfo(e.ev, CL_PROFILING_COMMAND_END, sizeof *t1, t1, NULL);
}
void mb_clear_events() { for (auto &e : mb_rt.events) clReleaseEvent(e.ev); mb_rt.events.clear(); }
int mb_num_kernels() { return (int)mb_rt.kinfo.size(); }
void mb_kernel_at(int i, cl_kernel *k, const char **name, int *op) {
    *k = mb_rt.kinfo[i].k; *name = mb_rt.kinfo[i].name; *op = mb_rt.kinfo[i].op;
}
