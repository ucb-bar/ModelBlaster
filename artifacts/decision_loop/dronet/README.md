# Decision loop on dronet (the right workload)

`mlp_control` was a convenience target (working build path, deterministic
spike output) but its sub-100 µs total runtime makes split/fuse
decisions effectively noise. The conv-heavy networks — **dronet** and
**yolov8** — are where the decisions actually matter. This directory
runs the loop on dronet first.

## Baseline

```bash
BACKEND=reference RUNNER=spike TARGET=rvv_opu QUANT=int8 \
    uv run bash examples/dronet/run.sh
```

**454,544,606 cycles total = 4.55 ms** for one dronet inference. That's
**4,400× more compute than mlp_control's 102k cycles**. Heaviest single
op is `conv_modules.8` at 74.5M cycles (16.4% of total).

## Loop run (round_005_final)

Granularity advisor proposes 5 fuses + 3 splits. After the
`--network dronet` filter, 3 candidates remain — all `split_heavy_dispatch`
on `dronet[0..2]_dispatch_0` (the first conv: IC=3 OC=32 112×112).

| Candidate | Predicted Δµs | Measured Δcyc | Decision |
|:---|---:|---:|:---:|
| split_dronet0_dispatch_0 | +3.06 | −158 | REJECT (within noise) |
| split_dronet1_dispatch_0 | +3.06 | −158 | REJECT (within noise) |
| split_dronet2_dispatch_0 | +3.06 | −158 | REJECT (within noise) |

Predicted said splits would *hurt* makespan by 3 µs; measured says
they're effectively cycle-neutral. **Both views agree splitting on a
sequential single-core simulator doesn't help** — the win is at
multi-core wall-clock when tiles can run on different accelerators.

The loop refuses to accept. **That's the right call given what spike
can measure.** A FireSim run on the hetero bitstream or the
wall-clock scheduler gate (round 8 work, blocked on a separate
scheduler-side ingestion gap) would be the next experiment.

## Two real skeleton-emitter gaps surfaced + fixed

Getting to a passing measurement required fixing two bugs in
`pipeline/generate_skeleton.py` that were latent in the linear-split
path but only manifested on multi-output-channel conv2d splits:

### Gap 1 — tile outputs trampled each other

**Symptom:** first split attempt → max_abs_err=24, max_rel_err=0.43
on dronet conv2d.

**Root cause:** the split rewrite generated `<orig>.tile_0` and
`<orig>.tile_1` as separate output tensors. The skeleton's buffer
allocator gave each its own buffer, but downstream consumers of the
ORIGINAL output tensor name read from an unallocated location (the
buffer allocator gap from the prior session). After registering the
tile tensors as proper buffer entries, the consumers still got wrong
data because both tiles wrote to OFFSET 0 of their respective buffers,
not OC-strided offsets of one combined buffer.

**Fix:** use the existing `offset_aliases` mechanism that `chunk2_c1`
uses for slicing. Each tile output aliases to the original tensor's
buffer at the right element offset:
- conv2d (axis=OC): `base + tile * tile_oc * OH * OW`
- linear (axis=N): `base + tile * tile_n`

### Gap 2 — tile weights all started from OC=0

**Symptom:** max_abs_err 24 → 2, but still non-zero — clear improvement
but not bit-exact.

**Root cause:** the kernel call site in model.c passed the SAME
`weight` and `bias` pointers to both tiles. Each tile thought it was
computing OC=0..tile_oc and read weight[OC=0..tile_oc] for ITS
computation. So tile_1 used the wrong weight slice.

**Fix:** detect `split_from.axis="OC"` at the conv2d_s8 call site and
offset the pointers:
- `weight + t*tile_oc*IC*KH*KW`
- `bias + t*tile_oc`

After both fixes: **max_abs_err=0 max_rel_err=0 PASS** on all 3 dronet
split candidates.

## What's still missing (for the splits to actually accept)

The per-op cycle gate the decision loop currently uses correctly
rejects splits — that's the right verdict for "is the total work
smaller." It can't see parallelism win because spike is sequential.

To get a measured ACCEPT we need ONE of:

1. **FireSim on hetero bitstream**, where tile_0 lands on Gemmini and
   tile_1 on Saturn-OPU+RVV. Wall-clock saving = ~max(tile_0_cyc,
   tile_1_cyc) ≈ orig_cyc/2. **Blocked on bitstream stability**
   (notes/firesim_measured_status.md).
2. **Wall-clock scheduler gate**, where the predicted scheduler is
   re-run after applying the split and the makespan delta is the
   accept signal. **Blocked on a separate scheduler-side mlp_control
   duration ingestion gap** (artifacts/decision_loop/README.md round 8).

Both blockers are out of scope for THIS round, which was about
getting the build path to work on real workloads. **Round 5
successfully proves: with the two skeleton fixes, the conv2d_s8
splits realize cleanly all the way to spike-bit-exact output. The
agent's measurement framework now applies to the right networks.**

## Why dronet, then yolov8

Dronet (10 conv2d_s8 ops, 4.55 ms total) was the smaller of the two
conv-heavy networks. yolov8_nano has 63 conv2d_s8 ops — same fixes
apply. Once the wall-clock gate or FireSim escalation unblocks, the
same loop runs on yolov8 with no further code changes; the
realizability filter, the IR rewriter, and the measurement
primitive are network-agnostic.
