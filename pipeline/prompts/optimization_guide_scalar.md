# Kernel optimization guide

Reusable instruction set for the LLM kernel-optimization stage. The targets
are scalar fp32 kernels for embedded RISC-V (rv64imafdc) compiled with `-O2`
by `riscv64-zephyr-elf-gcc` and run on `spike`. No vector extension, no
intrinsics, no inline asm — straight C99.

The compiler at `-O2` already handles:

- common subexpression elimination
- pointer-from-index strength reduction
- aggressive register allocation
- inlining of leaf functions in the same TU
- pointer hoisting across simple loops

So source-level rewrites that just rephrase the same dataflow (renaming
indices, hoisting `input + m*K` into a pointer, etc.) usually do **not**
change cycles. The list below is biased toward changes the compiler
**cannot** infer on its own.

---

## Techniques the compiler will not do for you

### 1. Multiple accumulators (break the FP dependency chain)

A scalar `for (k) acc += a[k] * b[k];` has a serial dependency on `acc`.
On rv64fd one fp-add takes multiple cycles to retire, so you stall waiting
on the previous accumulator. Split the work across N independent
accumulators (typical N = 2 or 4) and sum at the end:

```c
float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
int k = 0;
for (; k + 4 <= K; k += 4) {
    acc0 += a[k+0] * b[k+0];
    acc1 += a[k+1] * b[k+1];
    acc2 += a[k+2] * b[k+2];
    acc3 += a[k+3] * b[k+3];
}
float acc = (acc0 + acc1) + (acc2 + acc3);
for (; k < K; k++) acc += a[k] * b[k];
```

This is the single highest-leverage transform for fp32 dot-products. Be
aware: it changes the summation order, which can shift the result by a few
ULPs — still well within the verify tolerance (atol=1e-4, rtol=1e-3).

### 2. Outer-product / register tiling for matmul

In a `[M, K] @ [N, K]^T` matmul, accumulate into a tile of outputs at once
to reuse loaded inputs. Even a 1×4 unroll over `n` reuses each `input[m,k]`
load four times:

```c
for (int n = 0; n + 4 <= N; n += 4) {
    float a0 = bias ? bias[n+0] : 0;
    float a1 = bias ? bias[n+1] : 0;
    float a2 = bias ? bias[n+2] : 0;
    float a3 = bias ? bias[n+3] : 0;
    for (int k = 0; k < K; k++) {
        float x = input[m*K + k];
        a0 += x * weight[(n+0)*K + k];
        a1 += x * weight[(n+1)*K + k];
        a2 += x * weight[(n+2)*K + k];
        a3 += x * weight[(n+3)*K + k];
    }
    output[m*N + n+0] = a0;
    output[m*N + n+1] = a1;
    output[m*N + n+2] = a2;
    output[m*N + n+3] = a3;
}
/* tail loop for leftover n */
```

For `M*K` small (the typical embedded case), even modest tile factors win
because the inner loop becomes load-bound on `weight` and the input load
becomes free.

### 3. Hoist invariant work out of inner loops

The compiler hoists pure expressions, but it will *not* hoist:

- the `bias ? bias[n] : 0.0f` ternary if `bias` could legally be `NULL`
  (pointer comparison is observable). Decide once, outside the loops.
- branches with side effects.

```c
/* before */
for (n) for (k) { ... acc = (bias ? bias[n] : 0.0f) + ...; }

/* after */
for (n) {
    float b = bias ? bias[n] : 0.0f;
    for (k) { ... acc = b + ...; }
}
```

### 4. Loop fusion across unfused kernel calls

When two kernels run back-to-back over the same buffer (e.g.
`linear → relu` writing to a 32-element vector), they touch the same memory
twice. *Within a single kernel* you can't fuse them, but inside a kernel
that writes its output you can apply the activation before storing:

```c
output[m*N + n] = acc > 0.0f ? acc : 0.0f;  /* fused linear+relu */
```

This is only legal if the kernel signature is changed — i.e. the *skeleton*
generator decides to fuse, not the kernel itself. Don't do this inside a
single-op kernel; it would break the contract.

### 5. Specialize for known small K

If K is small and constant at the call site (e.g. `K = 16`), the compiler
won't unroll because it doesn't have constant propagation across the call.
Manually unroll the inner loop. Combine with multiple accumulators for
maximum effect.

### 6. Avoid pointer aliasing churn

`const float *input` and `float *output` are different types but the
compiler must still assume they may alias unless told otherwise (`restrict`
helps, but the function signature is fixed by the kernel ABI). Inside the
kernel, write through `output[...]` exactly once per element — don't
read-then-write the same element in a loop.

### 7. ReLU and other elementwise ops

Trivial loops — `-O2` already produces near-optimal code. The only useful
rewrite is to support in-place safely (`input == output` is allowed by
spec, so don't accumulate into output before reading input).

---

## Anti-patterns that hurt at `-O2`

- **`#pragma`s and compiler-specific hints**: forbidden by the output rules.
- **Manual common-subexpression caching** of values the compiler already
  CSEs (e.g. `int idx = m*K + k;` then using `idx` once). Pure noise.
- **Splitting a single linear loop into nested loops** without changing the
  iteration order or unrolling. No effect, just clutter.
- **Using `double` for accumulators**: causes f32→f64 conversion on every
  add and an f64→f32 store at the end. On rv64fd this is more cycles, not
  fewer. Stay in `float` unless explicitly requested.
- **Unrolling by very large factors** (e.g. 16 or 32) on small K: hurts
  i-cache and forces large tail loops.

---

## Numerical-equivalence reminders

The verify harness compares against the reference within
`atol=1e-4, rtol=1e-3`. Reordering FP operations within a dot-product
(multiple accumulators, tiling) is allowed and expected. Anything that
changes the math (using `<` instead of `<=` in ReLU, dropping a bias term,
clamping outputs) will fail verify and the variant will be discarded.
