# RVV (RISC-V Vector) kernel optimization guide

Reusable instruction set for the LLM kernel-generation/optimization stage
when the target is RVV (rv64gcv). Kernels are compiled with `-march=rv64gcv
-mabi=lp64d` by `riscv64-zephyr-elf-gcc` at `-O2` and linked into a Zephyr
application that runs on `spike --isa=rv64gcv_zicntr`. Vector extension is
enabled at runtime via `CONFIG_RISCV_ISA_EXT_V=y`.

You may use the standard RVV intrinsics from `<riscv_vector.h>` — that
header is automatically `#include`d at the top of `kernels.c`, so do **not**
add it yourself.

The compiler at `-O2` will scalar-pipeline well, but it will NOT
auto-vectorize meaningfully on its own. Vectorization is your job.

---

## Core RVV pattern (length-agnostic)

RVV is **vector-length-agnostic**: the same source compiles for any VLEN.
You query the runtime vector length via `__riscv_vsetvl_e32m1(remaining)`
and stride your loop by the returned `vl`.

The canonical fp32 dot-product over a length-K vector:

```c
size_t vl;
vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
for (int k = 0; k < K; k += vl) {
    vl = __riscv_vsetvl_e32m1(K - k);
    vfloat32m1_t va = __riscv_vle32_v_f32m1(a + k, vl);
    vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + k, vl);
    acc = __riscv_vfmacc_vv_f32m1(acc, va, vb, vl);
}
/* horizontal reduction to scalar */
vfloat32m1_t vsum = __riscv_vfredusum_vs_f32m1_f32m1(
    acc, __riscv_vfmv_s_f_f32m1(0.0f, 1), __riscv_vsetvlmax_e32m1());
float dot = __riscv_vfmv_f_s_f32m1_f32(vsum);
```

Key points:
- Use `vl = vsetvl_e32m1(remaining)` at the top of each iteration; do NOT
  hardcode VLEN.
- `vfmacc_vv` is fused-multiply-accumulate: `acc += va * vb`.
- `vfredusum_vs` (or `vfredsum_vs`) horizontally reduces a vector to a
  scalar. Use it once after the loop, not inside.

## Reduction-style kernels (matmul row, dot product)

`matmul row × col` reduces to a series of dot products. For each output
element you accumulate into a vector register, then horizontally reduce.

For a `[M, K] @ [N, K]^T` matmul:

```c
for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
        size_t vl;
        vfloat32m1_t vacc = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
        const float *in_row  = input  + m * K;
        const float *w_row   = weight + n * K;
        for (int k = 0; k < K; k += vl) {
            vl = __riscv_vsetvl_e32m1(K - k);
            vfloat32m1_t va = __riscv_vle32_v_f32m1(in_row + k, vl);
            vfloat32m1_t vb = __riscv_vle32_v_f32m1(w_row + k, vl);
            vacc = __riscv_vfmacc_vv_f32m1(vacc, va, vb, vl);
        }
        vfloat32m1_t vsum = __riscv_vfredusum_vs_f32m1_f32m1(
            vacc, __riscv_vfmv_s_f_f32m1(0.0f, 1),
            __riscv_vsetvlmax_e32m1());
        float acc = __riscv_vfmv_f_s_f32m1_f32(vsum);
        if (bias) acc += bias[n];
        output[m * N + n] = acc;
    }
}
```

## Higher LMUL (multiply-LEN) for more throughput

Each register group LMUL setting gives more elements per op at the cost of
fewer architectural register groups. `m1` ≡ 1 register, `m2` ≡ 2, `m4` ≡ 4,
`m8` ≡ 8. Replace the suffix in the type and intrinsic names:

```c
vfloat32m4_t vacc = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
size_t vl = __riscv_vsetvl_e32m4(K - k);
vfloat32m4_t va = __riscv_vle32_v_f32m4(in_row + k, vl);
/* etc. */
```

`m4` typically wins on small scalar pipelines because it amortizes the
overhead of `vsetvl` and the reduction. `m8` saturates the register file
and can hurt; usually `m1` or `m4` is the right ceiling on a scalar in-order
core.

### CRITICAL: reductions always output `m1`

This is the single most common bug in vectorized matmul. The horizontal-sum
intrinsic is `vfredusum_vs_<input_lmul>_<output_lmul>`, and the output is
**always `m1`**, regardless of input LMUL. The init/result type is `m1` too.

The correct shape of an `m4` reduction is:

```c
/* accumulate into m4 */
vfloat32m4_t vacc = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
/* ... loop: vfmacc_vv_f32m4(vacc, ...) ... */

/* reduce m4 -> m1 (init is m1, output is m1, vl is m4's max) */
vfloat32m1_t vinit = __riscv_vfmv_s_f_f32m1(0.0f, 1);
vfloat32m1_t vsum  = __riscv_vfredusum_vs_f32m4_f32m1(
    vacc, vinit, __riscv_vsetvlmax_e32m4());
float acc = __riscv_vfmv_f_s_f32m1_f32(vsum);
```

Common mistakes the verify will catch (don't do these):

| ❌ wrong | ✅ correct |
|---|---|
| `__riscv_vfredusum_vs_f32m4_f32m4(...)` | `__riscv_vfredusum_vs_f32m4_f32m1(...)` |
| `__riscv_vfmv_s_f_f32m4(0.0f, 1)` for the init | `__riscv_vfmv_s_f_f32m1(0.0f, 1)` |
| `__riscv_vfmv_f_s_f32m4_f32(vsum)` | `__riscv_vfmv_f_s_f32m1_f32(vsum)` |

If you can't keep these straight, **stay on `m1` for reductions**. The
correctness gain far exceeds the small throughput loss.

## Multiple accumulators within a single vector

For long-K dot products, the in-pipeline FMA dependency on `vacc` is still
serial across iterations. Two independent accumulators break the chain:

```c
vfloat32m1_t a0 = __riscv_vfmv_v_f_f32m1(0.0f, vmax);
vfloat32m1_t a1 = __riscv_vfmv_v_f_f32m1(0.0f, vmax);
size_t vl;
int k = 0;
for (; k + 2*vmax <= K; k += 2*vmax) {
    a0 = __riscv_vfmacc_vv_f32m1(a0,
        __riscv_vle32_v_f32m1(in + k, vmax),
        __riscv_vle32_v_f32m1(w  + k, vmax), vmax);
    a1 = __riscv_vfmacc_vv_f32m1(a1,
        __riscv_vle32_v_f32m1(in + k + vmax, vmax),
        __riscv_vle32_v_f32m1(w  + k + vmax, vmax), vmax);
}
/* tail */
for (; k < K; k += vl) {
    vl = __riscv_vsetvl_e32m1(K - k);
    a0 = __riscv_vfmacc_vv_f32m1(a0,
        __riscv_vle32_v_f32m1(in + k, vl),
        __riscv_vle32_v_f32m1(w  + k, vl), vl);
}
vfloat32m1_t merged = __riscv_vfadd_vv_f32m1(a0, a1, vmax);
/* reduce merged */
```

## Vectorizing over output columns (conv2d, matmul row-output dimension)

A second valid strategy for matmul-like kernels is to vectorize over an
**output** dimension instead of the reduction dim. For 2-D convolution this
means each lane computes a different output column for the same (n, oc, oh):

```c
size_t vl;
for (int oh = 0; oh < OH; oh++) {
    for (int ow = 0; ow < OW; ow += vl) {
        vl = __riscv_vsetvl_e32m1(OW - ow);
        vfloat32m1_t vacc = __riscv_vfmv_v_f_f32m1(
            bias ? bias[oc] : 0.0f, vl);
        for (int ic = 0; ic < IC; ic++) {
            for (int kh = 0; kh < KH; kh++) {
                int ih = oh * SH - PH + kh;
                if (ih < 0 || ih >= IH) continue;
                for (int kw = 0; kw < KW; kw++) {
                    /* Each lane reads input[ih, ow*SW + lane*SW - PW + kw].
                     * For SW == 1 a contig load is enough; for SW > 1 use a
                     * strided load with byte stride = SW * sizeof(float). */
                    /* (Edge handling for ih and the iw range omitted here
                     *  for clarity — handle PW/IW bounds as in the scalar
                     *  reference.) */
                    int iw0 = ow * SW - PW + kw;
                    vfloat32m1_t vinput = __riscv_vlse32_v_f32m1(
                        input + ((n*IC + ic)*IH + ih)*IW + iw0,
                        (ptrdiff_t)SW * (ptrdiff_t)sizeof(float), vl);

                    /* CRITICAL: weight is the SAME SCALAR for every lane —
                     * all lanes here share (oc, ic, kh, kw). Broadcast it,
                     * do not vector-load it. */
                    float w = weight[((oc*IC + ic)*KH + kh)*KW + kw];
                    vfloat32m1_t vweight = __riscv_vfmv_v_f_f32m1(w, vl);
                    vacc = __riscv_vfmacc_vv_f32m1(vacc, vinput, vweight, vl);
                }
            }
        }
        __riscv_vse32_v_f32m1(
            output + ((n*OC + oc)*OH + oh)*OW + ow, vacc, vl);
    }
}
```

### When OW is small, vectorize over OC instead

The OW-vectorized pattern wastes lanes when `OW < vlmax/2` (e.g.
DroNet's `conv_modules.7` has OW=4 — half the typical vlmax=8 with
LMUL=1). For these, swap the inner shape: vectorize over **OC** so each
lane computes a different output channel for the same (n, oh, ow), and
broadcast the input pixel:

```c
for (int oc0 = 0; oc0 < OC; oc0 += vl) {
    vl = __riscv_vsetvl_e32m1(OC - oc0);
    vfloat32m1_t vacc = bias
        ? __riscv_vle32_v_f32m1(bias + oc0, vl)
        : __riscv_vfmv_v_f_f32m1(0.0f, vl);
    for (int ic = 0; ic < IC; ic++) {
        for (int kh = 0; kh < KH; kh++) {
            int ih = oh * SH - PH + kh;
            if (ih < 0 || ih >= IH) continue;
            for (int kw = 0; kw < KW; kw++) {
                int iw = ow * SW - PW + kw;
                if (iw < 0 || iw >= IW) continue;
                /* Input is the SAME SCALAR for all lanes (same n, ic,
                 * ih, iw). Broadcast it. */
                float v = input[((n*IC + ic)*IH + ih)*IW + iw];
                /* Weight differs per lane — `oc + lane`, same (ic, kh,
                 * kw). Adjacent in OIHW memory: stride =
                 * IC*KH*KW * sizeof(float). */
                vfloat32m1_t vweight = __riscv_vlse32_v_f32m1(
                    weight + ((oc0*IC + ic)*KH + kh)*KW + kw,
                    (ptrdiff_t)IC*KH*KW * (ptrdiff_t)sizeof(float), vl);
                vacc = __riscv_vfmacc_vf_f32m1(vacc, v, vweight, vl);
            }
        }
    }
    /* Strided store across the output_channel dimension. */
    __riscv_vsse32_v_f32m1(output + ((n*OC + oc0)*OH + oh)*OW + ow,
                           (ptrdiff_t)OH*OW * (ptrdiff_t)sizeof(float),
                           vacc, vl);
}
```

Pick this pattern whenever `OC ≥ 2*OW` and the OC stride is not so
big that strided loads dominate. It's an especially good fit for the
deeper layers of dense CNNs — DroNet's last block is OC=128, OW=4.

### When KH=1 and KW=1, the inner dims collapse — use im2col_gemm

For 1×1 convolutions there's no gather work; im2col_gemm degenerates
into a `[OC, IC] @ [IC, OH*OW]` matmul which has a clean RVV reduction.
Even if you keep the direct algorithm for 3×3, switching to
`im2col_gemm` for 1×1 strides through the inner-product dimension
contiguously and lets you use higher LMUL.

### LMUL=4 for the OW-vectorized conv2d body

Replace `_f32m1` with `_f32m4` and `vsetvl_e32m1` with `vsetvl_e32m4`
in the inner accumulator + load. Keep all reduction-style intrinsics
on `_f32m1` (per the LMUL section above). This typically lifts conv2d
throughput 1.3-1.6x on Rocket-class cores because the FMA dependency
chain is amortized over 4× more lanes per `vsetvl`.

### Maxpool / avgpool: stride matters for lane mapping

The OW-vectorized pattern for maxpool2d looks superficially like conv2d's
inner load, but you can't reuse a contiguous load when `SW > 1`. Adjacent
OUTPUT columns `ow, ow+1, ow+2, ...` map to input columns
`ow*SW, (ow+1)*SW, (ow+2)*SW, ...` — i.e. strided by `SW` floats.

**Wrong** (treats input columns as adjacent in memory):
```c
int iw = ow * SW + kw;
vfloat32m1_t v = __riscv_vle32_v_f32m1(  // contiguous load
    input + ((n*C + c)*IH + ih)*IW + iw, vl);
```

**Correct** for `SW=1`:
```c
int iw = ow + kw;  /* SW=1, no scaling needed */
vfloat32m1_t v = __riscv_vle32_v_f32m1(
    input + ((n*C + c)*IH + ih)*IW + iw, vl);
```

**Correct** for `SW>1`:
```c
int iw0 = ow * SW + kw;
vfloat32m1_t v = __riscv_vlse32_v_f32m1(  /* note: vlse, not vle */
    input + ((n*C + c)*IH + ih)*IW + iw0,
    (ptrdiff_t)SW * (ptrdiff_t)sizeof(float),  /* byte stride */
    vl);
```

DroNet's first maxpool has `SW=2`, so the contiguous-load form will
produce subtly wrong outputs (max over the wrong elements per lane —
typically larger values because adjacent elements often correlate).
The verify catches this with a small absolute error (~0.04) that
compounds across the network.

### `vfmacc_vf` vs `vfmacc_vv`: pick by what's the scalar

Both intrinsics fuse-multiply-accumulate, but they take different
shapes:

- `vfmacc_vv(acc_vec, a_vec, b_vec, vl)` — both operands vectors
- `vfmacc_vf(acc_vec, a_scalar_float, b_vec, vl)` — first multiplicand scalar

**The order matters.** `vfmacc_vf` is `acc += a_scalar * b_vec`. If you
have `acc += b_vec * a_scalar` you must call it with `(acc, a_scalar,
b_vec, vl)`, not `(acc, b_vec, a_scalar, vl)`. Don't pass two vectors
to `vfmacc_vf` — the compiler error is
"incompatible type for argument 2 of '__riscv_vfmacc_vf_f32m1'
expected 'float' but argument is of type 'vfloat32m1_t'".

If both operands are vectors, use `vfmacc_vv`. If one is scalar and
you can identify which, broadcast the scalar (`vfmv_v_f`) once outside
the loop and use `vfmacc_vv` — easier than getting the `_vf` argument
order right.

### Mask types are `vbool*_t`, not `vfloat*_t`

Predicate intrinsics like `vfgt_vf`, `vmflt_vv`, `vmsne_vv` return a
**bool vector** — `vbool32_t` for `e32m1`, `vbool16_t` for `e32m2`, etc.
You can't assign that to a `vfloat32m1_t`. The two patterns that
compile:

```c
/* Right: take a mask, then merge with it */
vbool32_t mask = __riscv_vmfgt_vf_f32m1_b32(v, 0.0f, vl);  /* note: vmfgt, not vfgt */
v = __riscv_vmerge_vvm_f32m1(v_neg_branch, v_pos_branch, mask, vl);
```

For elu specifically, the merge pattern is:

```c
size_t vl = __riscv_vsetvl_e32m1(remaining);
vfloat32m1_t v = __riscv_vle32_v_f32m1(input + i, vl);
vbool32_t neg_mask = __riscv_vmflt_vf_f32m1_b32(v, 0.0f, vl);
/* neg branch: alpha * (expf(x) - 1).  expf is scalar — fall back to
 * a scalar tail loop for the negative lanes; vector path computes
 * positives in-register. */
```

**But for elu in practice, the simplest working pattern is plain
scalar** (see the elu code block in the transcendentals section).

### CRITICAL antipatterns in OW-vectorized conv

The single most common bug is **vector-loading the weight** when it should
be broadcast. Adjacent lanes correspond to adjacent **output** columns —
they all share the same `(oc, ic, kh, kw)` filter element. Don't do this:

```c
/* WRONG: loads weight[..., kw], weight[..., kw+1], ... into adjacent lanes,
 * which contaminates the result with whatever follows in the weight buffer
 * (e.g. the next output channel's filter for kw=KW-1). */
vfloat32m1_t vweight = __riscv_vle32_v_f32m1(
    weight + ((oc*IC + ic)*KH + kh)*KW + kw, vl);
```

The right intrinsic is `vfmv_v_f_f32m1(scalar, vl)` (broadcast a scalar).

A second pitfall: input bounds checking. For SW>1 with padding, the iw of
each lane is `ow*SW + lane*SW - PW + kw`. If any lane falls outside
`[0, IW)`, you can't just zero a single lane — either fall back to the
scalar path for that (ow, kw) combo, or use a masked strided load. The
simplest correct strategy is to keep the existing PW=0 / SW=1 fast path
vectorized and dispatch to scalar for general padded/strided cases until
we have masked-load support in the guide.

## Elementwise / point ops (relu, add, mul, etc.)

These are one-pass and have no reduction. Loop the vsetvl-stride pattern
once over the data:

```c
size_t vl;
for (int i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m1(n - i);
    vfloat32m1_t v = __riscv_vle32_v_f32m1(input + i, vl);
    v = __riscv_vfmax_vf_f32m1(v, 0.0f, vl);   /* relu */
    __riscv_vse32_v_f32m1(output + i, v, vl);
}
```

`__riscv_vfmax_vf_f32m1(v, 0.0f, vl)` is the cleanest ReLU — element-wise
max against a broadcast scalar. Higher LMUL (`m4`, `m8`) usually wins here
because there's no reduction overhead.

---

## RVV does NOT have transcendental intrinsics

There is **no** `__riscv_vfexp`, `__riscv_vexpf`, `__riscv_vflog`,
`__riscv_vfsin`, `__riscv_vftanh`, `__riscv_vsigmoid`, `__riscv_vferf`,
`__riscv_vfpow` etc. **None of these exist.** Do not call them.
The base RVV ISA only has arithmetic, comparison, mask, reduction,
slide, and bit ops on vectors.

This applies to **every** op whose math involves a transcendental:

| op | math | strategy |
|---|---|---|
| sigmoid | `1 / (1 + expf(-x))` | scalar `expf` per lane (`n=1` at model output head — don't bother vectorizing at all) |
| **elu** | `x >= 0 ? x : alpha * (expf(x) - 1)` | scalar `expf` per lane; vectorize the `x >= 0` short-circuit if you must |
| tanh | `(expf(x)-expf(-x))/(expf(x)+expf(-x))` | scalar `tanhf` per lane |
| softmax | `expf(x) / sum(expf(x))` | scalar pass for `expf`, vector pass for the divide |

**The correct shape for an elu kernel on RVV** is plain scalar inside
a length-`n` loop. There is no win from `vsetvl` here because every
lane needs a separate `expf` call:

```c
#include <math.h>
void kernel_elu(const float *input, float *output, int n, float alpha) {
    for (int i = 0; i < n; i++) {
        float x = input[i];
        output[i] = x >= 0.0f ? x : alpha * (expf(x) - 1.0f);
    }
}
```

If you really want vectorization for elu, the **only** correct pattern
is: copy positives through with vector ops (using `vfmax_vf` against 0
to clear negatives, etc.) but compute the negative branch's `expf`
elementwise in scalar. There is no speed gain from the half-vectorized
approach at typical n; **stick with the plain scalar version above**.

Or implement a polynomial / minimax approximation manually using only
vfmul/vfadd/vfmadd. (Not necessary for first-pass correctness — leave
the optimization phase to add this if it's worth it.)

For ops like sigmoid that may run on a tiny tensor (n=1 at a model
output head), the scalar version is strictly faster — vector overhead
exceeds the math.

## Constraints — same as scalar with two additions

- **DO** include intrinsics from `<riscv_vector.h>` (`__riscv_*` form).
  The header is already in scope; do NOT add `#include` yourself.
- **DO NOT** mix RVV intrinsics with manual inline asm. Pick one. Intrinsics
  are strongly preferred for portability across VLEN/SEW.
- **DO NOT** assume a specific VLEN. Always use `vsetvl(remaining)` and
  let the loop adapt. Hardcoding `vl = 4` will work in spike's default
  config but is fragile.
- **DO NOT** call `csrr` or write `mstatus` from C. Vector-unit enable is
  handled by Zephyr at thread context switch time when
  `CONFIG_RISCV_ISA_EXT_V=y`.

## Numerical-equivalence reminders

The reference is the **scalar** implementation. Verify tolerances are
`atol=1e-5, rtol=1e-4` for both per-op and end-to-end model-output
checks. The two MUST match — a kernel that PASSes per-op verify must
also be safe in composition with the rest of the model (otherwise
multi-op drift accumulates past the run-level gate).

Reordering FP ops via vector reduction is allowed and expected within
the tolerance — what matters is that the reordering is *bounded*:

- **Single horizontal reduction is fine.** `vfredusum_vs` produces
  results that differ by a few ULPs from the scalar accumulator.
- **Multiple parallel accumulators (e.g. for unrolling) are fine** as
  long as you reduce them at the end. The total drift scales with the
  number of accumulators, not the loop length.
- **AVOID Kahan-style or pairwise reordering on top of vector
  reduction.** The drift compounds.
- **AVOID reading partial sums into FP32 from int8 accumulator paths.**
  Stay in int32 / float32 throughout the reduction.

If a candidate "wins" by 5–10x and the per-op error is borderline
(near the 1e-5 atol), assume it will fail at run level and reject it.
A 2x kernel that passes cleanly is worth more than a 6x kernel that
breaks composition.

---

## When intrinsic names get fuzzy

Conventions you can rely on:

| operation | intrinsic |
|---|---|
| set vl up to remaining | `__riscv_vsetvl_e32m1(remaining)` |
| set vl to max | `__riscv_vsetvlmax_e32m1()` |
| broadcast scalar to vec | `__riscv_vfmv_v_f_f32m1(scalar, vl)` |
| broadcast scalar to scalar-vec | `__riscv_vfmv_s_f_f32m1(scalar, 1)` |
| extract first vec element | `__riscv_vfmv_f_s_f32m1_f32(vec)` |
| load contig fp32 | `__riscv_vle32_v_f32m1(ptr, vl)` |
| load strided fp32 | `__riscv_vlse32_v_f32m1(ptr, byte_stride, vl)` |
| store contig fp32 | `__riscv_vse32_v_f32m1(ptr, vec, vl)` |
| FMA: acc = acc + a*b | `__riscv_vfmacc_vv_f32m1(acc, a, b, vl)` |
| add: a + b | `__riscv_vfadd_vv_f32m1(a, b, vl)` |
| mul: a * b | `__riscv_vfmul_vv_f32m1(a, b, vl)` |
| max(a, scalar) | `__riscv_vfmax_vf_f32m1(a, scalar, vl)` |
| horizontal sum reduce | `__riscv_vfredusum_vs_f32m1_f32m1(vec, init, vl)` |

Replace `m1` with `m2`/`m4`/`m8` to widen, and adjust `vsetvl_e32m<N>` to
match. Mixing widths in one expression won't compile.

---

## Quantized (int8) kernels

For ops with the `_s8` suffix (e.g. `kernel_linear_s8`, `kernel_conv2d_s8`),
the inputs and outputs are int8, the bias and accumulators are int32, and
the kernel ends with a Q0.31 fixed-point requantize to compress the int32
back into int8. The same vsetvl-strided-loop pattern applies, but you'll
use **integer** intrinsics with **widening** ops to grow int8 lanes into
int16/int32 accumulators.

### Critical rule: widening intrinsics widen ONE level

`vwmul`/`vwmacc` widen by 2× (e.g. i8×i8 → i16, i16×i16 → i32). They do
**not** go i8 → i32 in one step. To get from int8 inputs to an int32
accumulator (the natural shape for `linear_s8` / `conv2d_s8`), use a
**two-stage** widening: first multiply i8×i8 into i16, then add the i16
products into the i32 accumulator.

### Canonical int8 dot-product on RVV (two-stage widening)

```c
size_t vl;
size_t vlmax = __riscv_vsetvlmax_e32m4();
/* Initialize accumulator to ZERO. Bias is added as a scalar AFTER the
 * horizontal reduction — DO NOT broadcast bias[n] into vacc here. */
vint32m4_t vacc = __riscv_vmv_v_x_i32m4(0, vlmax);
for (int k = 0; k < K; k += vl) {
    vl = __riscv_vsetvl_e8m1(K - k);
    vint8m1_t va = __riscv_vle8_v_i8m1(input  + k, vl);
    vint8m1_t vb = __riscv_vle8_v_i8m1(weight + k, vl);
    /* Stage 1: widening multiply i8 * i8 -> i16. */
    vint16m2_t prod = __riscv_vwmul_vv_i16m2(va, vb, vl);
    /* Stage 2: widening add accumulator (i32m4) += i16m2. The `_wv` form
     * keeps the wider operand in place and widens the narrower one on
     * the fly. vl is still in i8/i16 element count. */
    vacc = __riscv_vwadd_wv_i32m4(vacc, prod, vl);
}
/* horizontal reduce int32 m4 -> int32 m1 (single scalar) */
vint32m1_t vinit = __riscv_vmv_s_x_i32m1(0, 1);
vint32m1_t vsum  = __riscv_vredsum_vs_i32m4_i32m1(vacc, vinit, vlmax);
int32_t acc = __riscv_vmv_x_s_i32m1_i32(vsum);
/* Add bias as a scalar AFTER the reduction. */
if (bias) acc += bias[n];
/* now do the scalar Q0.31 requantize tail (multiplier+shift+offset+clamp) */
```

### Alternative (simpler but lower throughput): sign-extend first

If the two-stage widening is too tricky to get right, you can sign-extend
i8 directly to i32 once, then use the non-widening MAC:

```c
vint8m1_t va = __riscv_vle8_v_i8m1(input + k, vl);
vint8m1_t vb = __riscv_vle8_v_i8m1(weight + k, vl);
vint32m4_t va32 = __riscv_vsext_vf4_i32m4(va, vl);  /* i8 → i32, 4x widen */
vint32m4_t vb32 = __riscv_vsext_vf4_i32m4(vb, vl);
vacc = __riscv_vmacc_vv_i32m4(vacc, va32, vb32, vl);
```

This processes 1 i8 element per i32 lane (so 4× fewer elements per cycle
than the two-stage version), but the structure is more obvious.

### Key int8 intrinsic names

| operation | intrinsic |
|---|---|
| set vl for int8, lmul=1     | `__riscv_vsetvl_e8m1(remaining)` |
| set vl for int32 acc lmul=4 | `__riscv_vsetvlmax_e32m4()` |
| broadcast int32 scalar      | `__riscv_vmv_v_x_i32m4(0, vl)` |
| broadcast to scalar-vec     | `__riscv_vmv_s_x_i32m1(0, 1)` |
| extract first int32 lane    | `__riscv_vmv_x_s_i32m1_i32(vec)` |
| load contig int8            | `__riscv_vle8_v_i8m1(ptr, vl)` |
| store contig int8           | `__riscv_vse8_v_i8m1(ptr, vec, vl)` |
| widening multiply i8*i8 → i16 | `__riscv_vwmul_vv_i16m2(a_i8m1, b_i8m1, vl)` |
| widening add  i32m4 += i16m2  | `__riscv_vwadd_wv_i32m4(acc_i32m4, b_i16m2, vl)` |
| sign-extend i8 → i32 (4×)   | `__riscv_vsext_vf4_i32m4(a_i8m1, vl)` |
| non-widening MAC i32 += i32*i32 | `__riscv_vmacc_vv_i32m4(acc, a_i32m4, b_i32m4, vl)` |
| reduce i32m4 → i32m1 scalar | `__riscv_vredsum_vs_i32m4_i32m1(vec_m4, init_m1, vl)` |
| int8 max with scalar        | `__riscv_vmax_vx_i8m1(a, scalar, vl)` |

### Requantize tail (post-reduction, scalar)

The Q0.31 multiply + shift + clamp at the end of `kernel_linear_s8` is
**scalar** — there's no value in vectorizing it because it runs once per
output element (not once per K-reduction step). Implement it exactly as
the reference does:

```c
int64_t prod = (int64_t)acc * (int64_t)output_multiplier;
prod = (prod + (1LL << 30)) >> 31;
int32_t scaled = (int32_t)prod;
if (output_shift > 0) scaled = (scaled + (1 << (output_shift - 1))) >> output_shift;
else                  scaled = scaled << (-output_shift);
scaled += output_offset;
if (scaled < activation_min) scaled = activation_min;
if (scaled > activation_max) scaled = activation_max;
output[m * N + n] = (int8_t)scaled;
```

### Common int8 antipatterns

- **CRITICAL — bias goes after the reduction, not into the vector init.**
  Initialize the int32 accumulator to **zero** with `vmv_v_x_i32m4(0, vlmax)`.
  After accumulating and horizontally reducing to a scalar, add `bias[n]`
  to that scalar. If you instead initialize the vector with
  `vmv_v_x_i32m4(bias[n], vlmax)`, the bias gets broadcast to every lane
  (typically 16 lanes at VLEN=128), and the subsequent `vredsum` adds it
  16 times — your output will be off by `15 * bias[n]`. This is the
  single most common bug in vectorized matmul-with-bias.
- **Don't** mix vfmacc (float) with vwmacc (integer widening) — these are
  different families; the prompt for an `_s8` kernel needs integer
  intrinsics throughout.
- **Don't** vectorize the requantize tail with `vwmul_vv_i64m8` etc. —
  there's no win, and 64-bit RVV ops are awkward to use.
- **Don't** truncate the int32 accumulator to int16 mid-reduction. K can
  be up to 2048 (DroNet) and the worst-case acc magnitude is `K * 127 *
  127 = ~33 million` which exceeds int16 range. Stay in int32.
- **Don't** apply `input_offset` / `filter_offset` to the int8 vector with
  `vadd_vx_i8m1` (it would overflow if the offset is large). The reference
  applies them to the int32 widened values. For our symmetric quant case
  (all offsets = 0) the addition is a no-op so it's invisible, but get the
  shape right anyway: skip the offset add entirely if the offset is
  statically known to be 0.
