# MBP (pext) kernel optimization guide

Reusable instruction set for the LLM kernel-generation/optimization stage when the target is
`pext`: hart 0 of the PYNQ-Z1 Rocket (rv64imac, **no FPU**) with the MBP packed-integer
extension. Kernels are compiled with `-O2` by `riscv64-zephyr-elf-gcc` and linked into a Zephyr
application that runs on spike (which models the MBP instructions) and on the FPGA.

The MBP instructions are exposed as `static inline` C functions by `pext.h`. That header is
automatically `#include`d at the top of `kernels.c`, so do **not** add it yourself. On the target
(`-DMB_PEXT_HW=1`) each function compiles to one custom-0 instruction; on the verification host it
compiles to a bit-identical software model, so the same source is checked on both. They are NOT
compiler builtins: there is no `__builtin_riscv_*` for them.

The compiler at `-O2` will not use the MBP on its own. Using it is your job.

---

## The instructions

Each operand is a 64-bit register holding eight int8 lanes; byte 0 is the least significant byte.

| intrinsic | semantics |
|---|---|
| `int64_t mb_pext_dot8(int64_t a, int64_t b)` | `sum_{i=0..7} (int8)a.byte[i] * (int8)b.byte[i]`, exact, as a signed 64-bit value. Not accumulating: add the result into your own accumulator. |
| `int64_t mb_pext_max8(int64_t a, int64_t b)` | LANE-WISE: `r.byte[i] = max((int8)a.byte[i], (int8)b.byte[i])` for i = 0..7. |
| `int64_t mb_pext_relu8(int64_t a)` | `mb_pext_max8(a, 0)` |
| `int64_t mb_pext_qmul(int64_t acc, int64_t mult)` | scalar: `((int64)(int32)acc * (int64)(int32)mult + 2^30) >> 31` (int32 accumulator times the Q0.31 multiplier, round-half-up) |
| `int64_t mb_pext_clip8(int64_t q)` | scalar: `clamp(q, -128, 127)`, sign-extended to 64 bits |

One instruction does the work of eight scalar multiply-adds or eight compare/selects, so the hot
loop should be built around these, with 8-byte-aligned, contiguous operands.

### CRITICAL: `mb_pext_max8` compares bytes, not values

Never use it to clamp a single scalar to `activation_min`/`activation_max`: that compares bytes,
not the value. The one scalar use that is exact is `mb_pext_relu8(v)` on a `v` that is already in
[-128, 127] (e.g. a `mb_pext_clip8` result), which gives `max(v, 0)`.

## Loading and storing 8 bytes: alignment is required

Use the header's macros:
- `MB_PEXT_LD8(p)` loads 8 consecutive int8 values at `p` as one `int64_t` (byte 0 = `p[0]`).
- `MB_PEXT_ST8(p, v)` stores 8 bytes.
- `MB_PEXT_ALIGNED8(p)` is true when `p` is 8-byte aligned.

### CRITICAL: a misaligned 8-byte access traps

`MB_PEXT_LD8`/`MB_PEXT_ST8` compile to a single `ld`/`sd` and REQUIRE an 8-byte-aligned address:
the core raises a misaligned-address exception rather than emulating. A misaligned access does not
return a wrong answer, it traps and the run hangs. The model's tensors are plain `int8_t` arrays
with no alignment guarantee, so test the actual pointers with `MB_PEXT_ALIGNED8` at run time, take
the MBP path only when every 8-byte access is aligned (base pointer aligned AND row stride a
multiple of 8), and otherwise run the scalar reference expression. The rule holds for EVERY
`MB_PEXT_LD8`/`MB_PEXT_ST8` address, including offsets into your own scratch buffers:
`scratch + 2*ow` is not 8-aligned for odd `ow`.

Do not use `memcpy(&v, p, 8)` from an `int8_t *`: on this target GCC turns it into eight byte
loads plus eight byte stores and a reload, which is slower than the scalar code.

## Requantization (linear_s8, conv2d_s8): must match the reference bit for bit

The reference computes, per output:
```c
int32_t acc = bias[n] + sum_k (input + input_offset) * (weight + filter_offset);
int64_t prod = ((int64_t)acc * output_multiplier + (1LL << 30)) >> 31;
int32_t scaled = (int32_t)prod;
if (output_shift > 0) scaled = ((int64_t)scaled + (1LL << (output_shift - 1))) >> output_shift;
else if (output_shift < 0) scaled = scaled << (-output_shift);
scaled += output_offset;
clamp scaled to [activation_min, activation_max]; output = (int8_t)scaled;
```
With MBP: `p = mb_pext_qmul(acc, output_multiplier)` replaces the `prod` line exactly. Then do
the shift in scalar code with an arithmetic (signed) shift; both roundings are round-half-up
(add, then shift), not round-to-even and not away-from-zero; anything else is 1 LSB out on
about half the negative outputs. `MB_PEXT_ROUND(s)` is the rounding constant `1 << (s-1)`,
valid for `s >= 0` only, so hoist it out of the loop under `if (output_shift > 0)`. Keep the
`output_shift < 0` left-shift branch and add `output_offset` before clamping.
`mb_pext_clip8(q)` may replace the final clamp only when `activation_min == -128` and
`activation_max == 127`; otherwise clamp with scalar compares, as the reference does.

`mb_pext_dot8` multiplies raw int8 lanes, so it cannot absorb a non-zero `input_offset` or
`filter_offset` (an offset added to a byte no longer fits in a byte). Take the DOT8 path only
when both offsets are 0, and run the scalar reference otherwise.

## Reduction-style kernels (linear_s8)

`input` is [M,K] and `weight` is [N,K], both row-major, so the reduction axis K is contiguous in
both operands and one output is one row-by-row dot product of DOT8s. Row `n` of the weight starts
at `weight + n*K`, which is 8-aligned for every `n` only if `weight` is aligned and `K % 8 == 0`,
so the weight rows are read where they are and the INPUT row is shifted to match (the
pext_row_dot8 scheme):
1. Scratch: `static int8_t xs[4096] __attribute__((aligned(8)));` declared INSIDE the function
   (a function-local static, not a global). Take this path only if `input_offset == 0`,
   `filter_offset == 0`, `K > 0` and `K + 15 <= 4096`; otherwise run the scalar reference.
   Never size the scratch by anything smaller than `K + 15` bytes: the verifier runs your
   kernel inside the generator process, so an out-of-bounds write (e.g. `int8_t scratch[8]`
   on the stack) kills the whole run with a segmentation fault instead of failing one attempt.
2. `r = (uintptr_t)(weight + n*K) & 7` depends on `n` only through `n*K mod 8`, so it has
   `period = 8 / gcd(K, 8)` values (1 when `K % 8 == 0`). For each `m`, for each
   `t = 0 .. min(period, N)-1`: compute `r` for row `t`, `L = (r + K + 7) & ~7`, and fill
   `xs[0..r) = 0`, `xs[r..r+K) = input row m`, `xs[r+K..L) = 0`. Then for
   `n = t, t+period, t+2*period, ... < N` (all share that `r`), with
   `a = weight + n*K - r` (8-aligned), `acc = bias ? bias[n] : 0` and for `g = 0, 8, .. < L`:
   `acc += mb_pext_dot8(MB_PEXT_LD8(xs + g), MB_PEXT_LD8(a + g))`. The bytes of the
   neighbouring rows that the 8-byte groups also cover are multiplied by the zeros in `xs`.
3. Edges: at `n*K - r + g < 0` (first row) or `n*K - r + g + 8 > N*K` (last rows) the group
   lies partly outside the weight tensor and must not be loaded. For such a group, add
   `xs[j] * a[j]` in scalar code for only the `j` in the group with
   `0 <= n*K - r + j < N*K`. Compute these bounds with integer offsets, not pointer compares.
4. Requantize `acc` as above and write `output[m*N + n]`.

## Maxpool: lane mapping

NCHW, so W is contiguous: eight adjacent columns of one input row are one 8-byte operand, eight
channels are not. For a 2x2, stride-2, unpadded, undilated pool with `IW % 8 == 0` and an aligned
`input`: `v = mb_pext_max8(LD8(row0 + 8g), LD8(row1 + 8g))` gives the vertical maxima of 8
columns; then `m = mb_pext_max8(v, (int64_t)((uint64_t)v >> 8))` leaves the four horizontal pair
maxima in bytes 0, 2, 4 and 6 of `m`, i.e. outputs `(int8_t)m`, `(int8_t)(m >> 16)`,
`(int8_t)(m >> 32)`, `(int8_t)(m >> 48)`. Every other window shape, any padding (padding is
INT8_MIN-filled) or an unaligned input takes the scalar reference expression.

---

## Constraints: same as scalar, with these additions

- **DO** use the MBP functions and macros from `pext.h`. The header is already in scope; do NOT
  add `#include` yourself.
- **DO** keep the scalar reference expression as the fallback for every shape, offset or
  alignment the MBP path does not handle. The verifier tests shapes the model does not use.
- **DO NOT** use floating point. The core has no FPU, so every float operation is a software
  call that costs hundreds of cycles.
- **DO NOT** write inline asm or `.insn` yourself. The intrinsics emit the right encoding.

## Numerical-equivalence reminders

Every `pext` kernel is **bit-exact**: the verifier compares against the reference with
`atol = 0`. An MBP path must compute exactly what the reference computes, and every shape it
cannot accelerate falls through to the reference expression verbatim. There is no tolerance to
trade for speed.
