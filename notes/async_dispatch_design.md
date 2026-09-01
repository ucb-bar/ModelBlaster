# Phase G2c — Async dispatch: overlap accelerator with next kernel setup

## Why

From v10 G1 attribution:

```
gemmini  wall=494 ms  kernel=109 ms  dep_wait=384 ms  (78% of wall blocked)
rvv_opu  wall=571 ms  kernel=509 ms  dep_wait=62 ms
```

The gemmini hart's 384 ms `dep_wait` is the upper bound on what
async dispatch can save: every microsecond gemmini spends waiting
for rvv_opu to publish an intermediate buffer is a microsecond
nothing else useful is happening on the Rocket core. Reducing it
requires either (a) making rvv_opu faster (Phase G6 RVV kernels)
or (b) letting gemmini overlap its own setup work with rvv_opu's
ongoing compute.

(a) is in flight (v11d). (b) is this design note.

## Current synchronous flow per kernel

```
walker:
    [issue tile setup]      ← scalar, ~50 µs typical
    tiled_matmul_auto(...)  ← issues RoCC commands to Gemmini
    gemmini_fence()         ← BLOCKS scalar until DMA out drains
    gemmini_flush(0)        ← reset state for next kernel
    return                  ← walker moves to next entry
```

While `tiled_matmul_auto` is running on the accelerator, the
scalar Rocket pipe is sitting in `gemmini_fence` waiting for the
RoCC busy bit to clear. For our matmul shapes that's typically
30–80 µs of pure wait per kernel. Across 105 gemmini entries on
the canonical workload that's 3–8 ms of pure scalar-wait.

## Proposed async flow

Split each Gemmini kernel into two halves:

```c
void kernel_<op>_issue(state_t *s, args...);  /* issues RoCC, no fence */
void kernel_<op>_wait(state_t *s);            /* gemmini_fence + flush */
```

Walker emits a 2-phase dispatch:

```
walker for entry N:
    if entry N-1 was issue-only on the same kind:
        kernel_*_wait(N-1)         ← drains the previous in-flight kernel
    kernel_*_issue(N)              ← issues current kernel, returns immediately
    [scalar setup for entry N+1]   ← runs in parallel with N's accelerator work
    advance i_
```

The fence happens only when (a) the next consumer needs the
in-flight kernel's output, or (b) we hit a non-Gemmini op on the
same hart. In the canonical hybrid schedule the gemmini hart is
~100% Gemmini-only between dronet/mlp instances, so this overlaps
almost every kernel with its successor's setup.

## Implementation pieces

### 1. Kernel-side split (per Gemmini kernel)

Touch `kernels/gemmini_q31/*.c`. Each kernel currently ends with
`gemmini_fence(); gemmini_flush(0);` — move those two lines into a
`<name>_wait()` sibling that takes the same state ptr. The "issue"
function keeps the same name + signature so non-async dispatch
sites work unchanged.

Variants to emit per Gemmini kernel:
- `kernel_linear_s8_gemmini_*` → split into _issue + _wait
- `kernel_conv2d_s8_gemmini_*` → split
- `kernel_batchnorm2d_s8_gemmini_*` → split
- elementwise / pool / activation: these don't use Gemmini RoCC,
  no split needed.

### 2. Walker codegen

`pipeline/generate_xpurt_main.py` worker loop changes:
- Track `prev_issued_entry_id` per-hart.
- Before dispatching entry N:
  - If `prev_issued_entry_id != -1`, emit `kernel_*_wait()` for
    that entry (drains in-flight).
  - If entry N's kind matches the prev and N's deps don't include
    `prev_issued_entry_id`, skip the wait — N's dispatch can stack
    on top of prev's drain.
- After issuing entry N (call _issue not the full kernel):
  - Record N as `prev_issued_entry_id`.

### 3. Dispatch table metadata

Add a per-entry `int issue_only;` bit at codegen time, set to 1
when the (op, kind) has a registered _issue/_wait pair. Walker
reads this to decide between full-kernel and issue-only dispatch.

### 4. Dispatch fn table

Per-backend now needs two function arrays:
- `MODEL_<UMID>_DISPATCH_FNS_<BS>[]` — the full kernel (for sync
  fallback and for the LAST entry of any periodic instance)
- `MODEL_<UMID>_DISPATCH_FNS_WAIT_<BS>[]` — the _wait sibling

Indexed by the same `dispatch_id`. Entries that don't support
async (kernel without _issue/_wait split) have NULL in the WAIT
table; the walker treats those as "synchronous, no follow-up wait
needed".

## Expected savings

- Gemmini side: ~30–80 µs scalar overlap per kernel × 105 entries ≈
  3–8 ms.
- Indirect rvv_opu side: gemmini's earlier dep posts mean rvv_opu's
  `dep_wait` (62 ms in v10) shrinks proportionally — bounded by how
  often rvv_opu was waiting on gemmini, which the hart_acc breakdown
  shows is small in v10.

Net: probably 5–15 ms wall savings on top of whatever v11 lands.
Not huge in isolation but cheap to add once the kernel split lands
and additive with the Phase G2d fanout patch.

## Out of scope

- RVV / OPU outer-product kernels on rvv_opu hart do NOT need this
  split — their "accelerator" is in-register state (OPU matrix m0..m3)
  drained synchronously inside the kernel. The scalar pipe and
  vector pipe issue serially through the same thread, so async
  dispatch has nothing to overlap there.

## Risk

- A consumer that issues right after a producer needs the
  producer's *output* (not just its issue). Walker must fence
  before any dispatch that reads buffers the in-flight kernel
  writes. The schedule's dep graph already encodes this — the
  walker's existing dep-take loop covers it as long as we keep
  the `_wait()` call inside the loop before the next dispatch
  issues.

- Gemmini's internal queue depth is bounded. If we issue 8 kernels
  ahead without waiting, the queue overflows. The 1-deep async
  scheme (max one in-flight at a time per hart) is conservative
  enough to never overflow.

## Status

Design note only. Implementation deferred until v11 measurements
land — the 384 ms gemmini dep_wait may collapse on its own once
rvv_opu is no longer the critical path. If post-v11 wall is
< 100 ms total, this whole work item probably won't move the
needle and can be deleted.
