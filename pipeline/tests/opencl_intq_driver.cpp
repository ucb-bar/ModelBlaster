// Bit-exactness of the naive OpenCL integer-closed data-movement seeds vs their spec reference_impl, and the
// MB_ERROR behaviour of the arithmetic placeholders. Built and run by test_kb_export_opencl.py (IntegerKindsTest),
// which writes slots.cpp: the nine slots as kb_export wraps them plus the references renamed ref_<op>.
#include "mb_opencl_runtime.h"
#include "slots.cpp"
#include <random>
static std::mt19937_64 rng(12345);
static size_t esz(int bits) { return bits == 8 ? 1 : bits <= 16 ? 2 : bits == 32 ? 4 : 8; }
struct Buf { void *p; size_t n; };
static Buf svm(size_t bytes) { Buf b{clSVMAlloc(mb_rt.ctx, CL_MEM_READ_WRITE, bytes ? bytes : 8, 0), bytes}; return b; }
static void map(Buf b, cl_map_flags f) { clEnqueueSVMMap(mb_rt.queue, CL_TRUE, f, b.p, b.n ? b.n : 8, 0, NULL, NULL); }
static void unmap(Buf b) { clEnqueueSVMUnmap(mb_rt.queue, b.p, 0, NULL, NULL); clFinish(mb_rt.queue); }
static Buf rnd(size_t bytes) { Buf b = svm(bytes); map(b, CL_MAP_WRITE); for (size_t i = 0; i < bytes; i++) ((unsigned char *)b.p)[i] = rng(); unmap(b); return b; }
static int fails = 0, cases = 0;
static void check(const char *what, Buf dev, const void *ref, size_t bytes) {
    clFinish(mb_rt.queue); map(dev, CL_MAP_READ);
    int bad = memcmp(dev.p, ref, bytes) != 0 || mb_rt.failed;
    unmap(dev); cases++;
    if (bad) { fails++; printf("FAIL %s%s%s\n", what, mb_rt.failed ? " err=" : "", mb_rt.failed ? mb_rt.err.c_str() : ""); }
}
int main() {
    if (mb_cl_init()) { printf("init failed: %s\n", mb_rt.err.c_str()); return 2; }
    char name[256] = {0}; clGetDeviceInfo(mb_rt.device, CL_DEVICE_NAME, sizeof name, name, NULL);
    printf("device: %s\n", name);
    const int widths[] = {8, 15, 16, 32, 64};
    char w[160];
    for (int bits : widths) {
        size_t e = esz(bits);
        { int n = 1000 + bits; Buf s = rnd(n * e), d = svm(n * e); std::vector<char> r(n * e);
          mbslot_copy_q::kernel_copy_q(s.p, d.p, n, bits); map(s, CL_MAP_READ); ref_copy_q(s.p, r.data(), n, bits); unmap(s);
          snprintf(w, sizeof w, "copy_q bits=%d", bits); check(w, d, r.data(), n * e); }
        { int n = 77, t = 5; Buf s = rnd(n * e), d = svm((size_t)n * t * e); std::vector<char> r((size_t)n * t * e);
          mbslot_repeat_q::kernel_repeat_q(s.p, d.p, n, t, bits); map(s, CL_MAP_READ); ref_repeat_q(s.p, r.data(), n, t, bits); unmap(s);
          snprintf(w, sizeof w, "repeat_q bits=%d", bits); check(w, d, r.data(), r.size()); }
        { int N = 2, IC = 9, c0 = 3, c1 = 7, HW = 13; Buf s = rnd((size_t)N * IC * HW * e), d = svm((size_t)N * (c1 - c0) * HW * e);
          std::vector<char> r((size_t)N * (c1 - c0) * HW * e);
          mbslot_slice_c_q::kernel_slice_c_q(s.p, d.p, bits, N, IC, c0, c1, HW); map(s, CL_MAP_READ); ref_slice_c_q(s.p, r.data(), bits, N, IC, c0, c1, HW); unmap(s);
          snprintf(w, sizeof w, "slice_c_q bits=%d", bits); check(w, d, r.data(), r.size()); }
        int perms[][4] = {{0, 2, 3, 1}, {0, 3, 1, 2}, {3, 2, 1, 0}, {1, 0, 3, 2}, {0, 1, 2, 3}};
        for (auto &p : perms) { int d0 = 2, d1 = 3, d2 = 5, d3 = 7; size_t tot = (size_t)d0 * d1 * d2 * d3;
          Buf s = rnd(tot * e), d = svm(tot * e); std::vector<char> r(tot * e);
          mbslot_permute4_q::kernel_permute4_q(s.p, d.p, bits, d0, d1, d2, d3, p[0], p[1], p[2], p[3]);
          map(s, CL_MAP_READ); ref_permute4_q(s.p, r.data(), bits, d0, d1, d2, d3, p[0], p[1], p[2], p[3]); unmap(s);
          snprintf(w, sizeof w, "permute4_q bits=%d perm=%d%d%d%d", bits, p[0], p[1], p[2], p[3]); check(w, d, r.data(), r.size()); }
        for (int ob : widths) for (int bb : {8, 16}) { int N = 2, Ca = 3, Cb = 5, HW = 11;
          Buf a = rnd((size_t)N * Ca * HW * e), b = rnd((size_t)N * Cb * HW * esz(bb)), d = svm((size_t)N * (Ca + Cb) * HW * esz(ob));
          std::vector<char> r((size_t)N * (Ca + Cb) * HW * esz(ob));
          mbslot_cat2_c1_q::kernel_cat2_c1_q(a.p, bits, Ca, b.p, bb, Cb, d.p, ob, N, HW);
          map(a, CL_MAP_READ); map(b, CL_MAP_READ); ref_cat2_c1_q(a.p, bits, Ca, b.p, bb, Cb, r.data(), ob, N, HW); unmap(a); unmap(b);
          snprintf(w, sizeof w, "cat2_c1_q a=%d b=%d out=%d", bits, bb, ob); check(w, d, r.data(), r.size()); }
    }
    printf("data-movement: %d/%d cases bit-exact\n", cases - fails, cases);
    // placeholders: each must set MB_ERROR naming itself
    int ph = 0;
    struct { const char *name; void (*f)(); } P[] = {
        {"kernel_gap_q", [] { mbslot_gap_q::kernel_gap_q(NULL, 8, NULL, 8, 1, 1, 1, 0, 0); }},
        {"kernel_mul_c1_q", [] { mbslot_mul_c1_q::kernel_mul_c1_q(NULL, 8, NULL, 8, NULL, 8, 1, 1, 1, 0, 0); }},
        {"kernel_add_q", [] { mbslot_add_q::kernel_add_q(NULL, 8, NULL, 8, NULL, 8, 1, 1, NULL, 0, 0, 1, 0); }},
        {"kernel_vint_action_post_acc", [] { mbslot_vint_action_post_acc::kernel_vint_action_post_acc(NULL, NULL, NULL, NULL, NULL, 1); }}};
    for (auto &p : P) { mb_rt.failed = 0; mb_rt.err.clear(); p.f();
        int ok = mb_rt.failed && strstr(mb_rt.err.c_str(), p.name) && strstr(mb_rt.err.c_str(), "placeholder");
        printf("%s %s -> MB_ERROR \"%s\"\n", ok ? "OK  " : "FAIL", p.name, mb_rt.err.c_str()); ph += ok; }
    printf("placeholders: %d/4 report MB_ERROR\n", ph);
    return (fails || ph != 4) ? 1 : 0;
}
