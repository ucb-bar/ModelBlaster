"""Multi-core (pool) versions of the simple layers on the K3 backends (k3-work TASK-017).

TASK-016 gave conv2d_s8 a row split over a modelblaster_pool; every simple layer
between the convs (SiLU, add, maxpool, upsample, concat, ELU, batchnorm, ...) still
ran on the first core of its group while the others waited. This module emits a
`mbp_<op>_<mid>(pool, <the kernel's own arguments>)` wrapper per op for model.c.
generate_skeleton routes a K3 backend's call through it; the pool's thread count
at run time is the width (as for parallel_linear_s8), and a NULL or one-thread pool,
or a build without MODELBLASTER_USE_POOL, makes the plain kernel call.

How each op is cut, always on the OUTPUT bytes and always at absolute 64-byte line
boundaries (`mbp_cut`), so no two cores ever write the same cache line:

  * per-element ops (silu, elu, add, mul, relu, sigmoid): core t runs the kernel on
    its own run of elements;
  * batchnorm2d_s8: the same run, called once per channel segment in it (the
    kernel's per-channel constants then come from that channel);
  * maxpool2d_s8, upsample_nearest_s8: core t calls the kernel on the whole channel
    planes inside its run; a plane cut by a line boundary is computed whole into a
    scratch plane (stack; heap above 4 KB) by each core that shares it, and each
    copies its own bytes;
  * cat{2,3,4}_c1_s8: core t requantizes and copies the part of each input's run
    that falls in its bytes (the kernel itself, called with one input of one
    channel and the rest empty).

Bit-exact with the serial call by construction: every output byte is produced by the
same kernel from the same inputs and parameters; only which core calls it changes.
K3 backends only (`enabled`): the K1 path and every other backend keep the plain
call, and their generated code does not change.
"""
from __future__ import annotations

from typing import Optional

#: Backend name prefixes with these pool paths (the K3 backends: rvv_k3, rvv_k3_a100,
#: ime_k3_x100, ime_k3_a100). The same prefixes as rules.conv_split_axis.
BACKEND_PREFIXES = ("rvv_k3", "ime_k3")

#: Per-element ops: (pointer args in kernel order, the last is the output; scalar
#: args after the element count, as (C type, name)).
ELEMENTWISE: dict[str, tuple[tuple[str, ...], tuple[tuple[str, str], ...]]] = {
    "silu_s8": (("in", "out"), (("float", "si"), ("float", "so"), ("int", "lo"), ("int", "hi"))),
    "sigmoid_s8": (("in", "out"), (("float", "si"), ("float", "so"), ("int", "lo"), ("int", "hi"))),
    "elu_s8": (("in", "out"), (("float", "si"), ("float", "so"), ("int", "lo"), ("int", "hi"),
                                ("float", "alpha"))),
    "relu_s8": (("in", "out"), ()),
    "add_s8": (("a", "b", "out"), (("float", "sa"), ("float", "sb"), ("float", "so"),
                                    ("int", "lo"), ("int", "hi"))),
    "mul_s8": (("a", "b", "out"), (("float", "sa"), ("float", "sb"), ("float", "so"),
                                    ("int", "lo"), ("int", "hi"))),
}
PLANE_OPS = ("maxpool2d_s8", "upsample_nearest_s8")
CHANNEL_OPS = ("batchnorm2d_s8",)
CAT_OPS = {"cat2_c1_s8": 2, "cat3_c1_s8": 3, "cat4_c1_s8": 4}

OPS = frozenset(ELEMENTWISE) | frozenset(PLANE_OPS) | frozenset(CHANNEL_OPS) | frozenset(CAT_OPS)


def enabled(backend: Optional[str]) -> bool:
    """Does `backend` get the simple-layer pool paths (a K3 backend)?"""
    return bool(backend) and any(str(backend).startswith(p) for p in BACKEND_PREFIXES)


_GUARD = "defined(MODELBLASTER_USE_POOL) && (defined(MB_PLATFORM_POSIX) || defined(MODELBLASTER_PLATFORM_LINUX))"

_COMMON = """
/* ---- K3 simple-layer pool paths (k3-work TASK-017, pipeline/k3_pool_ops.py) ----
 * The width is the pool's thread count. Every cut between two cores' outputs lies on
 * an absolute 64-byte line boundary, so no two cores write the same cache line. */
#if {guard}
#define MBP_POOL_{mid} 1
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
/* Bytes [*lo, *hi) of an n-byte output at `out` for core t of T. */
static inline void mbp_cut_{mid}(const void *out, size_t n, size_t T, size_t t,
                                 size_t *lo, size_t *hi) {{
    uintptr_t a = (uintptr_t)out;
    size_t b[2];
    for (int k = 0; k < 2; k++) {{
        size_t i = t + (size_t)k;
        if (i == 0) {{ b[k] = 0; continue; }}
        if (i >= T) {{ b[k] = n; continue; }}
        uintptr_t x = (a + (uintptr_t)(n * i / T) + 63u) & ~(uintptr_t)63u;
        b[k] = (size_t)(x - a) < n ? (size_t)(x - a) : n;
    }}
    *lo = b[0];
    *hi = b[1];
}}
/* The pool's width when at least two cores get output bytes, else 1 (serial). */
static inline size_t mbp_width_{mid}(void *pool_, const void *out, size_t n) {{
    modelblaster_pool_t pool = (modelblaster_pool_t)pool_;
    size_t T = pool ? (size_t)modelblaster_pool_get_threads_count(pool) : 1, busy = 0;
    if (T < 2 || n == 0) return 1;
    for (size_t t = 0; t < T; t++) {{
        size_t lo, hi;
        mbp_cut_{mid}(out, n, T, t, &lo, &hi);
        busy += lo < hi;
    }}
    return busy > 1 ? T : 1;
}}
#endif
"""

def _elementwise(op: str, mid: str) -> str:
    ptrs, scal = ELEMENTWISE[op]
    out = ptrs[-1]
    pdecl = ", ".join(("int8_t *" if p == out else "const int8_t *") + p for p in ptrs)
    sdecl = "".join(f", {t} {n}" for t, n in scal)
    fields = "".join(f" {('int8_t *' if p == out else 'const int8_t *')}{p};" for p in ptrs)
    fields += "".join(f" {t} {n};" for t, n in scal)
    call_ctx = ", ".join(f"c->{p} + a" for p in ptrs) + ", (int)(b - a)" + "".join(f", c->{n}" for _, n in scal)
    call_dir = ", ".join(ptrs) + ", n" + "".join(f", {n}" for _, n in scal)
    init = ", ".join(f".{p} = {p}" for p in ptrs) + "".join(f", .{n} = {n}" for _, n in scal)
    return f"""
#if {_GUARD}
typedef struct {{{fields} size_t n, T; }} mbp_{op}_ctx_{mid};
static void mbp_{op}_fn_{mid}(void *c_, size_t t) {{
    mbp_{op}_ctx_{mid} *c = (mbp_{op}_ctx_{mid} *)c_;
    size_t a, b;
    mbp_cut_{mid}(c->{out}, c->n, c->T, t, &a, &b);
    if (a < b) kernel_{op}_{mid}({call_ctx});
}}
#endif
static inline void mbp_{op}_{mid}(void *pool_, {pdecl}, int n{sdecl}) {{
#ifdef MBP_POOL_{mid}
    size_t T = n > 0 ? mbp_width_{mid}(pool_, {out}, (size_t)n) : 1;
    if (T > 1) {{
        mbp_{op}_ctx_{mid} ctx = {{{init}, .n = (size_t)n, .T = T}};
        modelblaster_pool_parallelize_1d((modelblaster_pool_t)pool_, mbp_{op}_fn_{mid}, &ctx, T, 0);
        return;
    }}
#else
    (void)pool_;
#endif
    kernel_{op}_{mid}({call_dir});
}}
"""


def _batchnorm(mid: str) -> str:
    return f"""
#if {_GUARD}
typedef struct {{ const int8_t *in; const float *s, *b; int8_t *out; size_t n, hw, C, T;
                  float si, so; int lo, hi; }} mbp_batchnorm2d_s8_ctx_{mid};
static void mbp_batchnorm2d_s8_fn_{mid}(void *c_, size_t t) {{
    mbp_batchnorm2d_s8_ctx_{mid} *c = (mbp_batchnorm2d_s8_ctx_{mid} *)c_;
    size_t a, b;
    mbp_cut_{mid}(c->out, c->n, c->T, t, &a, &b);
    while (a < b) {{   /* one call per channel segment of [a, b) */
        size_t plane = a / c->hw, end = (plane + 1) * c->hw;
        if (end > b) end = b;
        size_t ch = plane % c->C;
        kernel_batchnorm2d_s8_{mid}(c->in + a, c->s + ch, c->b + ch, c->out + a, 1, 1, 1,
                                    (int)(end - a), c->si, c->so, c->lo, c->hi);
        a = end;
    }}
}}
#endif
static inline void mbp_batchnorm2d_s8_{mid}(void *pool_, const int8_t *in, const float *s,
        const float *bias, int8_t *out, int N, int C, int H, int W,
        float si, float so, int lo, int hi) {{
#ifdef MBP_POOL_{mid}
    size_t n = (N > 0 && C > 0 && H > 0 && W > 0) ? (size_t)N * C * H * W : 0;
    size_t T = mbp_width_{mid}(pool_, out, n);
    if (T > 1) {{
        mbp_batchnorm2d_s8_ctx_{mid} ctx = {{.in = in, .s = s, .b = bias, .out = out, .n = n,
            .hw = (size_t)H * W, .C = (size_t)C, .T = T, .si = si, .so = so, .lo = lo, .hi = hi}};
        modelblaster_pool_parallelize_1d((modelblaster_pool_t)pool_, mbp_batchnorm2d_s8_fn_{mid}, &ctx, T, 0);
        return;
    }}
#else
    (void)pool_;
#endif
    kernel_batchnorm2d_s8_{mid}(in, s, bias, out, N, C, H, W, si, so, lo, hi);
}}
"""


def _planes(op: str, mid: str) -> str:
    """maxpool2d_s8 / upsample_nearest_s8: whole planes direct, cut planes via scratch."""
    if op == "maxpool2d_s8":
        sig = ("const int8_t *in, int8_t *out, int N, int C, int IH, int IW, int KH, int KW, "
               "int SH, int SW, int PH, int PW, int DH, int DW")
        fields = "int IH, IW, KH, KW, SH, SW, PH, PW, DH, DW;"
        init = (".IH = IH, .IW = IW, .KH = KH, .KW = KW, .SH = SH, .SW = SW, "
                ".PH = PH, .PW = PW, .DH = DH, .DW = DW")
        geom = ("int OH = (IH + 2*PH - DH*(KH-1) - 1) / SH + 1, "
                "OW = (IW + 2*PW - DW*(KW-1) - 1) / SW + 1;\n"
                "    size_t ip = (size_t)IH * IW, op = (OH > 0 && OW > 0) ? (size_t)OH * OW : 0;")
        args = "c->IH, c->IW, c->KH, c->KW, c->SH, c->SW, c->PH, c->PW, c->DH, c->DW"
        dargs = "N, C, IH, IW, KH, KW, SH, SW, PH, PW, DH, DW"
    else:
        sig = "const int8_t *in, int8_t *out, int N, int C, int IH, int IW, int scale"
        fields = "int IH, IW, scale;"
        init = ".IH = IH, .IW = IW, .scale = scale"
        geom = ("size_t ip = (size_t)IH * IW, "
                "op = (IH > 0 && IW > 0 && scale > 0) ? ip * scale * scale : 0;")
        args = "c->IH, c->IW, c->scale"
        dargs = "N, C, IH, IW, scale"
    return f"""
#if {_GUARD}
typedef struct {{ const int8_t *in; int8_t *out; size_t n, ip, op, T; {fields} volatile int failed; }} mbp_{op}_ctx_{mid};
static void mbp_{op}_fn_{mid}(void *c_, size_t t) {{
    mbp_{op}_ctx_{mid} *c = (mbp_{op}_ctx_{mid} *)c_;
    size_t a, b;
    mbp_cut_{mid}(c->out, c->n, c->T, t, &a, &b);
    size_t p = a / c->op;                 /* plane holding byte a */
    while (a < b) {{
        size_t ps = p * c->op, pe = ps + c->op;
        if (a == ps && pe <= b) {{          /* a run of whole planes, written in place */
            size_t np = (b - a) / c->op;
            kernel_{op}_{mid}(c->in + p * c->ip, c->out + a, 1, (int)np, {args});
            a += np * c->op; p += np;
        }} else {{                           /* a plane the cut crosses: whole into scratch */
            int8_t stk[4096], *s = c->op <= sizeof stk ? stk : (int8_t *)malloc(c->op);
            if (s == NULL) {{ c->failed = 1; return; }}
            kernel_{op}_{mid}(c->in + p * c->ip, s, 1, 1, {args});
            size_t e = pe < b ? pe : b;
            memcpy(c->out + a, s + (a - ps), e - a);
            if (s != stk) free(s);
            a = e; p++;
        }}
    }}
}}
#endif
static inline void mbp_{op}_{mid}(void *pool_, {sig}) {{
#ifdef MBP_POOL_{mid}
    {geom}
    size_t n = (N > 0 && C > 0) ? (size_t)N * C * op : 0;
    size_t T = mbp_width_{mid}(pool_, out, n);
    if (T > 1) {{
        mbp_{op}_ctx_{mid} ctx = {{.in = in, .out = out, .n = n, .ip = ip, .op = op, .T = T, {init}, .failed = 0}};
        modelblaster_pool_parallelize_1d((modelblaster_pool_t)pool_, mbp_{op}_fn_{mid}, &ctx, T, 0);
        if (!ctx.failed) return;
        /* a core had no memory for its scratch plane: redo the whole layer serially */
    }}
#else
    (void)pool_;
#endif
    kernel_{op}_{mid}(in, out, {dargs});
}}
"""


def _cat(op: str, mid: str) -> str:
    k = CAT_OPS[op]
    sig = ", ".join(f"const int8_t *in{j}, int c{j}, float s{j}" for j in range(k))
    # one input of one channel (h = W = the run), the other k-1 inputs empty
    one = lambda src: ", ".join(  # noqa: E731
        (f"{src} + x, 1, c->s[j]" if j == 0 else f"{src}, 0, c->s[j]") for j in range(k))
    dcall = ", ".join(f"in{j}, c{j}, s{j}" for j in range(k))
    return f"""
#if {_GUARD}
typedef struct {{ const int8_t *in[{k}]; int c[{k}]; float s[{k}]; int8_t *out; size_t n, hw, T;
                  int N, ct; float so; int lo, hi; }} mbp_{op}_ctx_{mid};
static void mbp_{op}_fn_{mid}(void *c_, size_t t) {{
    mbp_{op}_ctx_{mid} *c = (mbp_{op}_ctx_{mid} *)c_;
    size_t a, b;
    mbp_cut_{mid}(c->out, c->n, c->T, t, &a, &b);
    if (a >= b) return;
    for (int nb = 0; nb < c->N; nb++) {{
        size_t off = (size_t)nb * c->ct * c->hw;
        for (int j = 0; j < {k}; j++) {{
            size_t len = (size_t)c->c[j] * c->hw, lo = off > a ? off : a, hi = off + len < b ? off + len : b;
            if (lo < hi) {{
                const int8_t *src = c->in[j] + (size_t)nb * len;
                size_t x = lo - off;
                kernel_{op}_{mid}({one("src")}, c->out + lo, 1, 1, (int)(hi - lo), c->so, c->lo, c->hi);
            }}
            off += len;
        }}
    }}
}}
#endif
static inline void mbp_{op}_{mid}(void *pool_, {sig}, int8_t *out, int N, int H, int W,
                                  float so, int lo, int hi) {{
#ifdef MBP_POOL_{mid}
    size_t hw = (H > 0 && W > 0) ? (size_t)H * W : 0;
    int ct = {" + ".join(f"c{j}" for j in range(k))};
    size_t n = (N > 0 && ct > 0) ? (size_t)N * ct * hw : 0;
    size_t T = mbp_width_{mid}(pool_, out, n);
    if (T > 1) {{
        mbp_{op}_ctx_{mid} ctx = {{.in = {{{", ".join(f"in{j}" for j in range(k))}}},
            .c = {{{", ".join(f"c{j}" for j in range(k))}}}, .s = {{{", ".join(f"s{j}" for j in range(k))}}},
            .out = out, .n = n, .hw = hw, .T = T, .N = N, .ct = ct, .so = so, .lo = lo, .hi = hi}};
        modelblaster_pool_parallelize_1d((modelblaster_pool_t)pool_, mbp_{op}_fn_{mid}, &ctx, T, 0);
        return;
    }}
#else
    (void)pool_;
#endif
    kernel_{op}_{mid}({dcall}, out, N, H, W, so, lo, hi);
}}
"""


def wrappers_c(mid: str, ops) -> str:
    """C for model.c: the pool wrappers of `ops` (those in OPS), or "" when none."""
    ops = sorted(set(ops) & OPS)
    if not ops:
        return ""
    parts = [_COMMON.format(guard=_GUARD, mid=mid)]
    for op in ops:
        if op in ELEMENTWISE:
            parts.append(_elementwise(op, mid))
        elif op in CHANNEL_OPS:
            parts.append(_batchnorm(mid))
        elif op in PLANE_OPS:
            parts.append(_planes(op, mid))
        else:
            parts.append(_cat(op, mid))
    return "".join(parts)
