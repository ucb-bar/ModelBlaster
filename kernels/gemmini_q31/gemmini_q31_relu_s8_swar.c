/* source: curated */
/* algorithm: swar */
/* accuracy_class: bit_exact */
/* origin: hand-written. int8 ReLU is a sign-bit test, and eight of them fit
 *         in one 64-bit register.
 *
 *   WHY THIS FILE EXISTS. relu_s8's only AlgorithmCandidate is
 *   gemmini_resadd_relu, whose kernel lives in kernels/gemmini_q31/archive/
 *   because it declared accuracy_class=bit_exact and was not (kernel_opt_log
 *   ids 1100-1108). With it withdrawn, relu_s8 on a Gemmini target had no
 *   file to probe for and ran the scalar reference.
 *
 *   MEASURED, spike, dronet, target gemmini_q31, 1 dispatch, n=2048:
 *   26,582 cycles = 13.0 cycles per byte for a compare and a store, 0.5% of
 *   the model. Small, and reported as small -- this is a coverage hole, not
 *   a bottleneck, and it is closed so that a shard boundary landing on
 *   relu_s8 is not priced at the reference's rate.
 *
 *   HOW. Load eight int8s as one uint64. `sign` isolates each byte's sign
 *   bit; `sign - (sign >> 7)` cannot borrow across bytes (0x80 - 0x01 =
 *   0x7f, and a non-negative byte contributes 0 to both terms), so
 *   `sign | (sign - (sign >> 7))` is 0xff in exactly the negative bytes and
 *   0x00 elsewhere. Masking with its complement zeroes the negative lanes
 *   and leaves the rest untouched.
 *
 *   THE POINTER TYPE IS THE MEASUREMENT. The first version of this file
 *   moved the eight bytes with `memcpy(&x, input + i, 8)` after aligning the
 *   pointers in a byte loop. It measured 31,797 cycles against the
 *   reference's 26,582 -- 20% SLOWER. The reason is that int8_t* carries
 *   alignment 1, so GCC cannot prove the copy is aligned and expands it into
 *   an unaligned-safe byte-assembly sequence rather than one `ld`; the
 *   run-time alignment check bought nothing because the compiler never saw
 *   it. Loading through a may_alias uint64_t* under an explicit alignment
 *   GATE is what actually emits `ld`/`sd`.
 *
 *   Buffers that are not mutually 8-aligned (offset aliases from chunk2_c1
 *   or a split tile can land anywhere) take the plain scalar loop, which is
 *   the reference expression with no extra test in it.
 *
 *   BIT-EXACT: `v > 0 ? v : 0` and "zero the byte iff its sign bit is set"
 *   agree on all 256 int8 values, zero included. MB_DRIFT_ATOL must NOT be
 *   set for this op. */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef uint64_t mb_u64_alias __attribute__((may_alias, aligned(8)));

void kernel_relu_s8(const int8_t *input, int8_t *output, int n)
{
    int i = 0;
    if (n >= 8 && ((((uintptr_t)input | (uintptr_t)output) & 7u) == 0)) {
        const mb_u64_alias *src = (const mb_u64_alias *)(const void *)input;
        mb_u64_alias *dst = (mb_u64_alias *)(void *)output;
        int blocks = n >> 3;
        for (int b = 0; b < blocks; b++) {
            uint64_t x = src[b];
            uint64_t sign = x & 0x8080808080808080ull;
            uint64_t mask = sign | (sign - (sign >> 7));
            dst[b] = x & ~mask;
        }
        i = blocks << 3;
    }
    for (; i < n; i++) { int8_t v = input[i]; output[i] = v > 0 ? v : 0; }
}
