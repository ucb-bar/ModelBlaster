"""K3 simple-layer pool paths (pipeline/k3_pool_ops.py, k3-work TASK-017).

* Only the K3 backends get them; K1 and every other backend keep the plain call.
* Each `mbp_<op>` wrapper, compiled on the host with the real modelblaster_pool and
  scalar kernels, gives exactly the serial kernel's output at widths 1/2/3/4/8, for
  network shapes and odd ones (lengths that are not multiples of 64 or of the width,
  1 element, 1 channel, misaligned output pointers), 100 back-to-back calls each,
  with 64 guard bytes before and after every output untouched; once plain and once
  under AddressSanitizer + UndefinedBehaviorSanitizer.
* With a recording pool that runs each slice alone on canary-filled memory: every
  output byte is written by exactly one slice, nothing outside the output is touched,
  and no two slices write the same 64-byte line.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))
from modelblaster.pipeline import generate_skeleton as gs  # noqa: E402
from modelblaster.pipeline import k3_pool_ops  # noqa: E402

CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


def test_enabled_backends():
    for be in ("rvv_k3", "rvv_k3_a100", "ime_k3_x100", "ime_k3_a100"):
        assert k3_pool_ops.enabled(be)
    for be in ("rvv_x60", "ime_x60", "rvv", "scalar", "gemmini", None, ""):
        assert not k3_pool_ops.enabled(be)


def _mini_ir():
    t = lambda shape, scale=0.05: {"shape": shape, "dtype": "i8",  # noqa: E731
                                   "quant": {"scale": scale, "zero_point": 0}}
    q1 = {"scale_in": 0.05, "scale_out": 0.04, "activation_min": -128, "activation_max": 127}
    ops = [
        ("silu_s8", ["x"], ["a"], {"n": 256}, q1),
        ("add_s8", ["a", "x"], ["b"], {"n": 256},
         {"scale_a": 0.04, "scale_b": 0.05, "scale_out": 0.06, "activation_min": -128, "activation_max": 127}),
        ("maxpool2d_s8", ["b"], ["c"], {"N": 1, "C": 4, "IH": 8, "IW": 8, "OH": 4, "OW": 4, "KH": 2, "KW": 2,
                                        "SH": 2, "SW": 2, "PH": 0, "PW": 0, "DH": 1, "DW": 1}, None),
        ("upsample_nearest_s8", ["c"], ["d"], {"N": 1, "C": 4, "IH": 4, "IW": 4, "scale": 2}, None),
        ("cat2_c1_s8", ["d", "b"], ["e"], {"N": 1, "H": 8, "W": 8, "C_inputs": [4, 4], "C_total": 8},
         {"scales_in": [0.06, 0.06], "scale_out": 0.06, "activation_min": -128, "activation_max": 127}),
        ("batchnorm2d_s8", ["e"], ["f"], {"N": 1, "C": 8, "H": 8, "W": 8}, q1),
        ("elu_s8", ["f"], ["g"], {"n": 512}, dict(q1, alpha=1.0)),
        ("relu_s8", ["g"], ["h"], {"n": 512}, None),
        ("sigmoid_s8", ["h"], ["y"], {"n": 512}, q1),
    ]
    shapes = {"x": [1, 4, 8, 8], "a": [1, 4, 8, 8], "b": [1, 4, 8, 8], "c": [1, 4, 4, 4], "d": [1, 4, 8, 8],
              "e": [1, 8, 8, 8], "f": [1, 8, 8, 8], "g": [1, 8, 8, 8], "h": [1, 8, 8, 8], "y": [1, 8, 8, 8]}
    ir_ops = []
    for i, (op, ins, outs, sh, q) in enumerate(ops):
        o = {"name": f"op{i}", "op": op, "inputs": ins, "outputs": outs, "shape": sh, "dispatch_id": i,
             "hardware_target": "any", "depends_on": [i - 1] if i else []}
        if q:
            o["quant"] = q
        if op == "batchnorm2d_s8":
            o["weight"], o["bias"] = "bn.scale", "bn.bias"
        ir_ops.append(o)
    return {"name": "m", "version": 1, "quant": "int8", "input": {"tensor": "x", "tensors": ["x"]},
            "output": {"tensor": "y", "tensors": ["y"]}, "tensors": {k: t(v) for k, v in shapes.items()},
            "ops": ir_ops, "dispatches": list(range(len(ir_ops)))}


@pytest.mark.parametrize("backend", ["ime_k3_x100", "rvv_k3", "rvv_k3_a100", "ime_k3_a100",
                                     "rvv_x60", "ime_x60"])
def test_codegen_routes_k3_only(tmp_path, backend):
    gs.emit_model(_mini_ir(), str(tmp_path), platform="linux", backend=backend)
    src = (tmp_path / "model.c").read_text()
    ops = ["silu_s8", "add_s8", "maxpool2d_s8", "upsample_nearest_s8", "cat2_c1_s8", "batchnorm2d_s8",
           "elu_s8", "relu_s8", "sigmoid_s8"]
    if k3_pool_ops.enabled(backend):
        for op in ops:
            assert f"mbp_{op}_m(s->pool, " in src, op
            assert f"static inline void mbp_{op}_m(void *pool_" in src, op
    else:
        assert "mbp_" not in src
        for op in ops:
            assert f"kernel_{op}_m(" in src, op


REF = r"""
#include <math.h>
#include <stdint.h>
#include <stddef.h>
static int8_t q8(float y, int lo, int hi) { int32_t v = (int32_t)roundf(y); return (int8_t)(v < lo ? lo : v > hi ? hi : v); }
void kernel_silu_s8_t(const int8_t *in, int8_t *out, int n, float si, float so, int lo, int hi) {
    for (int i = 0; i < n; i++) { float f = in[i] * si; out[i] = q8(f / (1.0f + expf(-f)) / so, lo, hi); } }
void kernel_sigmoid_s8_t(const int8_t *in, int8_t *out, int n, float si, float so, int lo, int hi) {
    for (int i = 0; i < n; i++) { float f = in[i] * si; out[i] = q8(1.0f / (1.0f + expf(-f)) / so, lo, hi); } }
void kernel_elu_s8_t(const int8_t *in, int8_t *out, int n, float si, float so, int lo, int hi, float al) {
    for (int i = 0; i < n; i++) { float f = in[i] * si; out[i] = q8((f > 0 ? f : al * (expf(f) - 1.0f)) / so, lo, hi); } }
void kernel_relu_s8_t(const int8_t *in, int8_t *out, int n) { for (int i = 0; i < n; i++) out[i] = in[i] > 0 ? in[i] : 0; }
void kernel_add_s8_t(const int8_t *a, const int8_t *b, int8_t *out, int n, float sa, float sb, float so, int lo, int hi) {
    for (int i = 0; i < n; i++) out[i] = q8((a[i] * sa + b[i] * sb) / so, lo, hi); }
void kernel_mul_s8_t(const int8_t *a, const int8_t *b, int8_t *out, int n, float sa, float sb, float so, int lo, int hi) {
    for (int i = 0; i < n; i++) out[i] = q8((a[i] * sa) * (b[i] * sb) / so, lo, hi); }
void kernel_batchnorm2d_s8_t(const int8_t *in, const float *s, const float *b, int8_t *out, int N, int C, int H, int W,
                             float si, float so, int lo, int hi) {
    for (int n = 0; n < N; n++) for (int c = 0; c < C; c++) for (int i = 0; i < H * W; i++) {
        size_t x = ((size_t)n * C + c) * H * W + i; out[x] = q8((s[c] * (in[x] * si) + b[c]) / so, lo, hi); } }
void kernel_maxpool2d_s8_t(const int8_t *in, int8_t *out, int N, int C, int IH, int IW, int KH, int KW, int SH, int SW,
                           int PH, int PW, int DH, int DW) {
    int OH = (IH + 2*PH - DH*(KH-1) - 1) / SH + 1, OW = (IW + 2*PW - DW*(KW-1) - 1) / SW + 1;
    for (int nc = 0; nc < N * C; nc++) for (int oh = 0; oh < OH; oh++) for (int ow = 0; ow < OW; ow++) {
        int8_t m = INT8_MIN;
        for (int kh = 0; kh < KH; kh++) for (int kw = 0; kw < KW; kw++) {
            int ih = oh*SH - PH + kh*DH, iw = ow*SW - PW + kw*DW;
            if (ih >= 0 && ih < IH && iw >= 0 && iw < IW && in[((size_t)nc*IH + ih)*IW + iw] > m) m = in[((size_t)nc*IH + ih)*IW + iw];
        }
        out[((size_t)nc*OH + oh)*OW + ow] = m;
    } }
void kernel_upsample_nearest_s8_t(const int8_t *in, int8_t *out, int N, int C, int IH, int IW, int s) {
    int OH = IH * s, OW = IW * s;
    for (int nc = 0; nc < N * C; nc++) for (int oh = 0; oh < OH; oh++) for (int ow = 0; ow < OW; ow++)
        out[((size_t)nc*OH + oh)*OW + ow] = in[((size_t)nc*IH + oh/s)*IW + ow/s]; }
#define CATBODY(K) \
    const int8_t *ins[K]; int cs[K]; float sc[K]; CATARGS \
    int ct = 0; for (int j = 0; j < K; j++) ct += cs[j]; size_t hw = (size_t)H * W; \
    for (int n = 0; n < N; n++) { size_t off = 0; for (int j = 0; j < K; j++) { float r = sc[j] / so; \
        for (size_t x = 0; x < (size_t)cs[j] * hw; x++) out[((size_t)n * ct) * hw + off + x] = q8(ins[j][(size_t)n * cs[j] * hw + x] * r, lo, hi); \
        off += (size_t)cs[j] * hw; } }
#define CATARGS ins[0]=i0; cs[0]=c0; sc[0]=s0; ins[1]=i1; cs[1]=c1; sc[1]=s1;
void kernel_cat2_c1_s8_t(const int8_t *i0, int c0, float s0, const int8_t *i1, int c1, float s1, int8_t *out,
                         int N, int H, int W, float so, int lo, int hi) { CATBODY(2) }
#undef CATARGS
#define CATARGS ins[0]=i0; cs[0]=c0; sc[0]=s0; ins[1]=i1; cs[1]=c1; sc[1]=s1; ins[2]=i2; cs[2]=c2; sc[2]=s2;
void kernel_cat3_c1_s8_t(const int8_t *i0, int c0, float s0, const int8_t *i1, int c1, float s1, const int8_t *i2, int c2,
                         float s2, int8_t *out, int N, int H, int W, float so, int lo, int hi) { CATBODY(3) }
#undef CATARGS
#define CATARGS ins[0]=i0; cs[0]=c0; sc[0]=s0; ins[1]=i1; cs[1]=c1; sc[1]=s1; ins[2]=i2; cs[2]=c2; sc[2]=s2; ins[3]=i3; cs[3]=c3; sc[3]=s3;
void kernel_cat4_c1_s8_t(const int8_t *i0, int c0, float s0, const int8_t *i1, int c1, float s1, const int8_t *i2, int c2,
                         float s2, const int8_t *i3, int c3, float s3, int8_t *out, int N, int H, int W, float so,
                         int lo, int hi) { CATBODY(4) }
"""

# Each case: op, ints. Network shapes of mlp_control / dronet / yolov8_nano plus odd ones.
CASES = r"""
static const struct { int op; int i[14]; } CASES[] = {
  /* 0 silu 1 sigmoid 2 elu 3 relu 4 add 5 mul: n */
  {0,{102400}},{0,{25600}},{0,{1600}},{0,{1}},{0,{63}},{0,{65}},{0,{257}},{0,{1023}},{0,{4097}},
  {1,{1}},{1,{777}},{2,{256}},{2,{64}},{2,{5}},{3,{2048}},{3,{3}},{4,{25600}},{4,{6272}},{4,{129}},{4,{1}},{5,{1000}},{5,{2}},
  /* 6 batchnorm: N C H W */
  {6,{1,32,27,27}},{6,{1,64,7,7}},{6,{1,1,1,1}},{6,{1,3,5,7}},{6,{2,5,3,3}},
  /* 7 maxpool: N C IH IW KH KW SH SW PH PW DH DW */
  {7,{1,32,56,56,3,3,2,2,0,0,1,1}},{7,{1,128,5,5,5,5,1,1,2,2,1,1}},{7,{1,1,5,5,5,5,1,1,2,2,1,1}},{7,{1,3,7,9,3,3,2,2,1,1,1,1}},
  {7,{2,5,6,6,3,3,1,1,1,1,1,1}},{7,{1,7,9,9,3,3,3,3,1,1,1,1}},{7,{1,2,8,8,3,3,1,1,1,1,2,2}},
  /* 8 upsample: N C IH IW scale */
  {8,{1,128,10,10,2}},{8,{1,256,5,5,2}},{8,{1,1,1,1,2}},{8,{1,3,5,33,2}},{8,{2,3,4,5,3}},{8,{1,9,3,3,2}},
  /* 9 cat2 / 10 cat3 / 11 cat4: N H W c0 c1 c2 c3 */
  {9,{1,10,10,256,128}},{9,{1,20,20,64,80}},{9,{1,3,3,1,3}},{9,{2,5,5,3,2}},
  {10,{1,40,40,16,16,16}},{10,{1,5,5,128,128,128}},{10,{1,1,7,5,1,2}},
  {11,{1,20,20,32,32,32,32}},{11,{1,5,5,128,128,128,128}},{11,{1,5,5,1,2,3,1}},
};
#define NCASES ((int)(sizeof CASES / sizeof CASES[0]))
"""

NCASES = sum(1 for ln in CASES.splitlines() for _ in __import__("re").finditer(r"\{\d+,\{", ln))

DRIVER = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "modelblaster_pool.h"
#include "decls.h"
#include "wrappers.inc"
#include "cases.inc"
#define G 64
static unsigned rs = 7;
static int8_t rnd(void) { rs = rs * 1103515245u + 12345u; return (int8_t)(rs >> 16); }
static size_t outsize(int k, const int *i) {
    switch (CASES[k].op) {
    case 0: case 1: case 2: case 3: case 4: case 5: return (size_t)i[0];
    case 6: return (size_t)i[0] * i[1] * i[2] * i[3];
    case 7: { int OH = (i[2] + 2*i[8] - i[10]*(i[4]-1) - 1) / i[6] + 1, OW = (i[3] + 2*i[9] - i[11]*(i[5]-1) - 1) / i[7] + 1;
              return (size_t)i[0] * i[1] * OH * OW; }
    case 8: return (size_t)i[0] * i[1] * i[2] * i[3] * i[4] * i[4];
    default: { size_t c = 0; for (int j = 0; j < CASES[k].op - 7; j++) c += (size_t)i[3 + j]; return (size_t)i[0] * c * i[1] * i[2]; }
    }
}
static int8_t *in8[4]; static float bs[512], bb[512];
static void call(void *pool, int k, int8_t *out) {
    const int *i = CASES[k].i;
    switch (CASES[k].op) {
    case 0: if (pool) mbp_silu_s8_t(pool, in8[0], out, i[0], 0.05f, 0.04f, -128, 127); else kernel_silu_s8_t(in8[0], out, i[0], 0.05f, 0.04f, -128, 127); break;
    case 1: if (pool) mbp_sigmoid_s8_t(pool, in8[0], out, i[0], 0.03f, 0.004f, -100, 120); else kernel_sigmoid_s8_t(in8[0], out, i[0], 0.03f, 0.004f, -100, 120); break;
    case 2: if (pool) mbp_elu_s8_t(pool, in8[0], out, i[0], 0.05f, 0.048f, -128, 127, 1.0f); else kernel_elu_s8_t(in8[0], out, i[0], 0.05f, 0.048f, -128, 127, 1.0f); break;
    case 3: if (pool) mbp_relu_s8_t(pool, in8[0], out, i[0]); else kernel_relu_s8_t(in8[0], out, i[0]); break;
    case 4: if (pool) mbp_add_s8_t(pool, in8[0], in8[1], out, i[0], 0.23f, 0.47f, 0.53f, -128, 127); else kernel_add_s8_t(in8[0], in8[1], out, i[0], 0.23f, 0.47f, 0.53f, -128, 127); break;
    case 5: if (pool) mbp_mul_s8_t(pool, in8[0], in8[1], out, i[0], 0.2f, 0.3f, 0.01f, -128, 127); else kernel_mul_s8_t(in8[0], in8[1], out, i[0], 0.2f, 0.3f, 0.01f, -128, 127); break;
    case 6: if (pool) mbp_batchnorm2d_s8_t(pool, in8[0], bs, bb, out, i[0], i[1], i[2], i[3], 0.03f, 0.5f, 0, 127);
            else kernel_batchnorm2d_s8_t(in8[0], bs, bb, out, i[0], i[1], i[2], i[3], 0.03f, 0.5f, 0, 127); break;
    case 7: if (pool) mbp_maxpool2d_s8_t(pool, in8[0], out, i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8], i[9], i[10], i[11]);
            else kernel_maxpool2d_s8_t(in8[0], out, i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8], i[9], i[10], i[11]); break;
    case 8: if (pool) mbp_upsample_nearest_s8_t(pool, in8[0], out, i[0], i[1], i[2], i[3], i[4]);
            else kernel_upsample_nearest_s8_t(in8[0], out, i[0], i[1], i[2], i[3], i[4]); break;
    case 9: if (pool) mbp_cat2_c1_s8_t(pool, in8[0], i[3], 0.06f, in8[1], i[4], 0.02f, out, i[0], i[1], i[2], 0.06f, -128, 127);
            else kernel_cat2_c1_s8_t(in8[0], i[3], 0.06f, in8[1], i[4], 0.02f, out, i[0], i[1], i[2], 0.06f, -128, 127); break;
    case 10: if (pool) mbp_cat3_c1_s8_t(pool, in8[0], i[3], 0.5f, in8[1], i[4], 0.23f, in8[2], i[5], 0.53f, out, i[0], i[1], i[2], 0.5f, -128, 127);
             else kernel_cat3_c1_s8_t(in8[0], i[3], 0.5f, in8[1], i[4], 0.23f, in8[2], i[5], 0.53f, out, i[0], i[1], i[2], 0.5f, -128, 127); break;
    case 11: if (pool) mbp_cat4_c1_s8_t(pool, in8[0], i[3], 0.05f, in8[1], i[4], 0.03f, in8[2], i[5], 0.035f, in8[3], i[6], 0.05f, out, i[0], i[1], i[2], 0.05f, -100, 100);
             else kernel_cat4_c1_s8_t(in8[0], i[3], 0.05f, in8[1], i[4], 0.03f, in8[2], i[5], 0.035f, in8[3], i[6], 0.05f, out, i[0], i[1], i[2], 0.05f, -100, 100); break;
    }
}
"""

MAIN_REAL = r"""
int main(void) {
    for (int j = 0; j < 4; j++) { in8[j] = malloc(1 << 20); for (int x = 0; x < (1 << 20); x++) in8[j][x] = rnd(); }
    for (int c = 0; c < 512; c++) { bs[c] = 0.5f + 0.01f * (c % 37); bb[c] = -3.0f + 0.25f * (c % 23); }
    static const int W[] = {1, 2, 3, 4, 8};
    static const int MIS[] = {0, 1, 13, 63};
    int bad = 0, runs = 0;
    for (unsigned wi = 0; wi < sizeof W / sizeof W[0]; wi++) {
        int harts[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < W[wi]; i++) harts[i] = i % 2;
        modelblaster_pool_t pool = modelblaster_pool_create_on_harts(W[wi], harts);
        if (!pool) { printf("pool failed\n"); return 2; }
        for (int k = 0; k < NCASES; k++) {
            size_t n = outsize(k, CASES[k].i);
            int8_t *ref = malloc(n + 1);
            call(NULL, k, ref);
            for (unsigned m = 0; m < sizeof MIS / sizeof MIS[0]; m++) {
                uint8_t *raw = aligned_alloc(64, ((n + 3 * G + 64) + 63) / 64 * 64);
                int8_t *out = (int8_t *)raw + G + MIS[m];
                memset(raw, 0xA5, n + 3 * G + 64);
                for (int rep = 0; rep < 100; rep++) {   /* 100 back-to-back calls */
                    call(pool, k, out);
                    if (memcmp(out, ref, n)) { bad++; printf("MISMATCH w=%d case %d mis %d rep %d\n", W[wi], k, MIS[m], rep); break; }
                }
                for (int g = 0; g < G + MIS[m]; g++) if (raw[g] != 0xA5) { bad++; printf("GUARD-BEFORE w=%d case %d\n", W[wi], k); break; }
                for (int g = 0; g < G; g++) if ((uint8_t)out[n + g] != 0xA5) { bad++; printf("GUARD-AFTER w=%d case %d\n", W[wi], k); break; }
                runs++;
                free(raw);
            }
            free(ref);
        }
        modelblaster_pool_destroy(pool);
    }
    for (int j = 0; j < 4; j++) free(in8[j]);
    printf("POOL_RESULT cases=%d runs=%d bad=%d\n", NCASES, runs, bad);
    return bad != 0;
}
"""

# A pool that runs each slice alone on canary-filled memory and records what it wrote.
FAKE_POOL = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "modelblaster_pool.h"
struct modelblaster_pool_state { unsigned T; };
static struct modelblaster_pool_state fake[9];
modelblaster_pool_t mk_pool(unsigned T) { fake[T].T = T; return &fake[T]; }
unsigned modelblaster_pool_get_threads_count(modelblaster_pool_t p) { return p ? p->T : 1; }
int8_t *g_out; size_t g_n; int g_bad; long g_slices;
#define G 64
void modelblaster_pool_parallelize_1d(modelblaster_pool_t p, void (*fn)(void *, size_t), void *ctx, size_t range, unsigned f) {
    (void)p; (void)f;
    signed char *who = malloc(g_n + 1);        /* slice that wrote each byte, -1 none */
    memset(who, -1, g_n + 1);
    uint8_t *gb = (uint8_t *)g_out - G, *ga = (uint8_t *)g_out + g_n;
    for (size_t t = 0; t < range; t++) {
        for (int pass = 0; pass < 2; pass++) {
            uint8_t c = pass ? 0xAA : 0x55;
            memset(g_out, c, g_n);
            fn(ctx, t);
            for (size_t x = 0; x < g_n; x++) {
                if ((uint8_t)g_out[x] == c) continue;
                if (who[x] >= 0 && who[x] != (signed char)t) { g_bad++; printf("BYTE %zu written by slices %d and %zu\n", x, who[x], t); }
                who[x] = (signed char)t;
            }
            for (int g = 0; g < G; g++) if (gb[g] != 0xA5 || ga[g] != 0xA5) { g_bad++; printf("GUARD hit by slice %zu\n", t); break; }
        }
        g_slices++;
    }
    for (size_t x = 0; x < g_n; x++) if (who[x] < 0) { g_bad++; printf("BYTE %zu written by no slice\n", x); break; }
    /* no two slices in one 64-byte line (absolute addresses) */
    for (size_t x = 0; x < g_n; x++) {
        size_t y = x + 1;
        if (y < g_n && who[y] != who[x] && ((uintptr_t)(g_out + x) >> 6) == ((uintptr_t)(g_out + y) >> 6)) {
            g_bad++; printf("LINE shared by slices %d and %d at byte %zu\n", who[x], who[y], x); break; }
    }
    free(who);
    for (size_t t = 0; t < range; t++) fn(ctx, t);   /* leave the real result */
}
"""

MAIN_FAKE = r"""
extern int8_t *g_out; extern size_t g_n; extern int g_bad; extern long g_slices;
modelblaster_pool_t mk_pool(unsigned T);
int main(void) {
    for (int j = 0; j < 4; j++) { in8[j] = malloc(1 << 20); for (int x = 0; x < (1 << 20); x++) in8[j][x] = rnd(); }
    for (int c = 0; c < 512; c++) { bs[c] = 0.5f + 0.01f * (c % 37); bb[c] = -3.0f + 0.25f * (c % 23); }
    static const unsigned W[] = {2, 3, 4, 8};
    static const int MIS[] = {0, 1, 13, 63};
    int bad = 0, runs = 0;
    for (unsigned wi = 0; wi < 4; wi++)
        for (int k = 0; k < NCASES; k++) {
            size_t n = outsize(k, CASES[k].i);
            int8_t *ref = malloc(n + 1);
            call(NULL, k, ref);
            for (unsigned m = 0; m < 4; m++) {
                uint8_t *raw = aligned_alloc(64, ((n + 3 * G + 64) + 63) / 64 * 64);
                memset(raw, 0xA5, n + 3 * G + 64);
                g_out = (int8_t *)raw + G + MIS[m]; g_n = n;
                call(mk_pool(W[wi]), k, g_out);
                if (memcmp(g_out, ref, n)) { bad++; printf("MISMATCH w=%u case %d\n", W[wi], k); }
                runs++;
                free(raw);
            }
            free(ref);
        }
    for (int j = 0; j < 4; j++) free(in8[j]);
    printf("FAKE_RESULT cases=%d runs=%d bad=%d ownership_bad=%d slices=%ld\n", NCASES, runs, bad, g_bad, g_slices);
    return (bad || g_bad) != 0;
}
"""

DECLS = r"""
#include <stdint.h>
void kernel_silu_s8_t(const int8_t *, int8_t *, int, float, float, int, int);
void kernel_sigmoid_s8_t(const int8_t *, int8_t *, int, float, float, int, int);
void kernel_elu_s8_t(const int8_t *, int8_t *, int, float, float, int, int, float);
void kernel_relu_s8_t(const int8_t *, int8_t *, int);
void kernel_add_s8_t(const int8_t *, const int8_t *, int8_t *, int, float, float, float, int, int);
void kernel_mul_s8_t(const int8_t *, const int8_t *, int8_t *, int, float, float, float, int, int);
void kernel_batchnorm2d_s8_t(const int8_t *, const float *, const float *, int8_t *, int, int, int, int, float, float, int, int);
void kernel_maxpool2d_s8_t(const int8_t *, int8_t *, int, int, int, int, int, int, int, int, int, int, int, int);
void kernel_upsample_nearest_s8_t(const int8_t *, int8_t *, int, int, int, int, int);
void kernel_cat2_c1_s8_t(const int8_t *, int, float, const int8_t *, int, float, int8_t *, int, int, int, float, int, int);
void kernel_cat3_c1_s8_t(const int8_t *, int, float, const int8_t *, int, float, const int8_t *, int, float, int8_t *, int, int, int, float, int, int);
void kernel_cat4_c1_s8_t(const int8_t *, int, float, const int8_t *, int, float, const int8_t *, int, float, const int8_t *, int, float, int8_t *, int, int, int, float, int, int);
"""


def _write(tmp_path: Path) -> None:
    (tmp_path / "wrappers.inc").write_text(k3_pool_ops.wrappers_c("t", k3_pool_ops.OPS))
    (tmp_path / "decls.h").write_text(DECLS)
    (tmp_path / "cases.inc").write_text(CASES)
    (tmp_path / "ref.c").write_text(REF)
    (tmp_path / "real.c").write_text(DRIVER + MAIN_REAL)
    (tmp_path / "fake_main.c").write_text(DRIVER + MAIN_FAKE)
    (tmp_path / "fake_pool.c").write_text(FAKE_POOL)


def _build(tmp_path: Path, name: str, srcs: list[str], san: bool) -> Path:
    rt = ROOT / "runtime"
    exe = tmp_path / (name + ("_san" if san else ""))
    flags = ["-fsanitize=address,undefined", "-fno-sanitize-recover=all", "-fno-omit-frame-pointer", "-g"] if san else []
    cmd = [CC, "-O1", "-Wall", "-Wno-unused-function", "-pthread", "-DMB_PLATFORM_POSIX", "-D_GNU_SOURCE",
           "-DMODELBLASTER_USE_POOL", *flags, f"-I{rt}", f"-I{rt / 'modelblaster_pool'}", f"-I{tmp_path}",
           "-o", str(exe), *srcs, str(tmp_path / "ref.c"), "-lm"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-4000:]
    return exe


@pytest.mark.skipif(CC is None, reason="no C compiler on PATH")
@pytest.mark.parametrize("san", [False, True])
def test_wrappers_bit_exact_real_pool(tmp_path, san):
    _write(tmp_path)
    rt = ROOT / "runtime"
    exe = _build(tmp_path, "real", [str(tmp_path / "real.c"), str(rt / "modelblaster_pool" / "modelblaster_pool.c"),
                                    str(rt / "mb_placement.c")], san)
    out = subprocess.run([str(exe)], capture_output=True, text=True, timeout=900)
    assert f"POOL_RESULT cases={NCASES} runs={5 * NCASES * 4} bad=0" in out.stdout, out.stdout[-3000:] + out.stderr[-3000:]
    assert out.returncode == 0 and "runtime error" not in out.stderr and "Sanitizer" not in out.stderr


@pytest.mark.skipif(CC is None, reason="no C compiler on PATH")
@pytest.mark.parametrize("san", [False, True])
def test_slices_own_their_lines(tmp_path, san):
    _write(tmp_path)
    exe = _build(tmp_path, "fake", [str(tmp_path / "fake_main.c"), str(tmp_path / "fake_pool.c")], san)
    out = subprocess.run([str(exe)], capture_output=True, text=True, timeout=900)
    assert f"FAKE_RESULT cases={NCASES} runs={4 * NCASES * 4} bad=0 ownership_bad=0" in out.stdout, out.stdout[-3000:] + out.stderr[-3000:]
    assert out.returncode == 0 and "runtime error" not in out.stderr and "Sanitizer" not in out.stderr
