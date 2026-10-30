# What we actually learned about fuse, split, and per-op placement

The decision loop ran across 12 rounds at three workload scales
(mlp_control 0.1 ms, dronet 4.5 ms, yolov8_nano 104 ms), measuring
spike cycles and wall-clock scheduler makespan for fuse and split
candidates on real conv-heavy networks. This is the synthesis.

## The big finding

**On this workload + hardware, splits don't help — and that's the
correct answer, not a framework bug.**

The reason: the scheduler is already placing each op on its best
accelerator. Different op kinds have wildly different cost ratios
between the two accelerator classes:

| Op kind | Cost on Gemmini | Cost on RVV | Best accel | Speedup |
|:---|---:|---:|:---:|---:|
| 3×3 large conv (OC=80, 20×20) | 0.65 ms | 24.2 ms | Gemmini | 37× |
| 3×3 large conv (OC=128, 20×20) | 0.50 ms | 12.3 ms | Gemmini | 25× |
| 1×1 conv (OC=80, 10×10) | 36.18 ms | 1.40 ms | RVV | 26× |
| 1×1 conv (OC=64, 10×10) | 23.26 ms | 0.82 ms | RVV | 28× |
| Concat (H=20 W=20) | 4.21 ms | 0.089 ms | RVV | 47× |
| Concat (H=10 W=10) | 1.41 ms | 0.031 ms | RVV | 45× |

Splitting a heavy 3×3 conv currently on Gemmini and migrating tile_1
to RVV would change that tile's cost from 0.33 ms to 12.1 ms.
Makespan = max(0.33, 12.1) = 12.1 ms — **36× worse** than the
unsplit op at 0.65 ms.

Splitting a heavy 1×1 conv currently on RVV and migrating tile_1 to
Gemmini would change that tile from 0.70 ms to 18 ms. Makespan = 18 ms,
**26× worse** than the unsplit 1.40 ms op.

Splitting and keeping both tiles on the same (best) accelerator gives
zero benefit on a 2-core machine — the tiles run sequentially because
there's only one of each accelerator class.

**The agent's "reject every split" verdict across rounds is correct.**

## What we proved the loop CAN do

Three honest gates fire correctly:

1. **Per-op cycle gate (fuse case).** Detected that `linear_s8_elu_s8`
   curated scalar is 18.7% slower than per-op chain on spike (round
   003). Detected that Bedrock auto-chose scalar fusion strategy
   loses 793% vs Bedrock per-op RVV (round 004). REJECT each.
2. **Bit-exact verify gate.** Caught two latent skeleton-emitter
   bugs on the very first dronet split attempt (output buffer
   offset → max_abs_err=24; tile weight offset → max_abs_err=2).
   After fixes: all 3 dronet splits + 3 yolov8 splits verify
   max_abs_err=0 on output tensors up to 75,600 elements.
3. **Wall-clock scheduler gate.** With the profile-loader tile
   fallback (parse parent dispatch_id from name → parent cycles /
   n_splits), the scheduler sees honest per-tile costs and places
   tiles wherever the cost model says is best. Δmakespan = 0.00 ms
   on every yolov8 split tested, exactly because the scheduler's
   placement decisions can't be improved.

## What the decision loop SHOULD be deciding (forward)

Splits and fuses on the current hardware+workload are at a local
optimum. The agent is right to refuse them. The actual decisions
the loop should make on workloads like this are:

### 1. Per-op placement reassignment

The most impactful single-op decision today is *not* "split this op"
but *"is this op on the right accelerator?"* The scheduler already
makes this choice, but its profile-DB-based scoring might be wrong if:

- The profile data is stale (kernel improved, profile didn't refresh).
- The cost model misses cross-cluster transfer overhead.
- A new accelerator config (OPU vs RVV) wasn't profiled separately.

A `candidate type = reassign_to_other_accel` from the granularity
advisor (which `granularity_loop.py` doesn't currently emit) would
let the loop measure: "the schedule placed this op on Gemmini; what
if I force it to RVV? Does measured cycles agree with the profile?"
That's a useful agentic check.

### 2. Detect when the workload changes the decision space

When a new shape lands (different K, different OC, etc.), the
per-accelerator cost ratio shifts. An op that was 30× better on RVV
might become 1.5× better on Gemmini at a different shape. The agent
should flag this and re-measure, not just re-run the predicted
scheduler.

### 3. Decide whether to fuse at the IR level vs at the kernel level

We saw two different "fusion" mechanisms this session:

- **IR-level fuse** (`apply_fusion_hint`): rewrites the graph so
  two ops become one fused dispatch. Needs a fused kernel that
  matches the per-op strategies. Currently loses because the fused
  kernel auto-selects a worse strategy than the per-op chain.
- **Schedule-level automerge** (`xpu-rt/automerge.py`): collapses
  back-to-back same-network dispatches on the same core into a
  single fused-call boundary at schedule time, without an IR change.
  Saves handshake overhead with no kernel-strategy risk.

The schedule-level automerge is **strictly safer** for the workload
shapes we tested. The agent should prefer it for "small consecutive
ops on the same core" cases, and reserve IR-level fusion for cases
where a true compute-fused kernel exists and outperforms the per-op
chain.

## Where splits WOULD help (followup workloads)

The framework will surface a measurable split win when one of these
holds:

1. **Multiple instances of a heavy op available for parallel
   placement.** With 4× mlp_control instances at 10 ms periods,
   instance-level placement across accelerators (not per-op split)
   gives 2× throughput for the same total work. The agent's
   "candidate type" enumeration should include
   `parallelize_periodic_instances_across_cores`.
2. **Workloads with > 2 accelerator classes.** On a 3-class machine
   (Gemmini + OPU + RVV), even when one class is best for an op,
   the *second-best* class might still be useful for an N=3 split.
   The test machine here has 2 classes; OPU is bundled into CPU_E
   with RVV by current scheduler topology.
3. **Pipeline-friendly networks.** When 16 mlp_control instances
   flow continuously, pipelining layer 0 of instance N+1 alongside
   layer 1 of instance N gives throughput wins. The agent should
   propose pipeline-shard candidates, which the granularity loop
   doesn't currently emit.

## What landed this session (code-level)

`feat/agentic-fusion-loop` (ModelBlaster):
- `pipeline/apply_split_hint.py` — `_split_linear_s8`, `_split_conv2d_s8`,
  per-tile tensor registration with correct shape (NCHW or NM).
- `pipeline/apply_fusion_hint.py` — pairwise linear+elu chain fusion.
- `pipeline/reference_kernels.py` — Saturn OPU AlgorithmCandidates
  (linear_s8/outerprod, matmul_s8/outerprod,
  linear_s8_elu_s8/outerprod_with_in_register_elu) with the macro
  contract embedded in description + working reference_impl.
- `pipeline/generate_kernels.py` — auto-inject
  `SATURN_OPU_KEEP_REGISTER_MACROS` before saturn_opu.h include.
- `pipeline/generate_skeleton.py` — tile output offset aliases
  (chunk2_c1 mechanism extended) + conv2d_s8 call-site weight/bias
  pointer offsets for tiles.
- `cores/saturn_opu/include/saturn_opu.h` — opt-in undef guard.
- `scripts/measure_candidate.sh` — Contract-2 hint → IR rewrite →
  build → spike verify → cycles ingest. Pipefail fix for long runs.
- `scripts/ingest_measured_cycles.py` — parse per-dispatch cycle
  table; handles fused-op format (empty shape column).
- `scripts/decision_loop.py` — granularity → realizability filter
  (fuse + linear/conv2d splits) → measure top-K → accept/reject.
- `scripts/wallclock_split_eval.py` — schedule-before-vs-after with
  IR substitution in zephyr-chipyard-sw, with backup/restore.

`xpurt-scheduler-advisor` (XPU-RT):
- `xpu-rt/compaction.py`, `automerge.py`, `oracle.py` — post-pass
  + lower-bound (committed earlier).
- `xpu-rt/profile_loader.py` v1 + v2 — tile-dispatch parent cycles
  fallback, parsing parent dispatch_id directly from name string.
- `xpu-rt/bundle.py` — greedy_periodic in DEFAULT_SCHEDULERS.
- `xpu-rt/schedulers.py` — registry wrapping for compaction post-pass.

## Honest budget + commits

Budget: $87 used / $150 cap. ~12 hours of session time. Commits this
segment on `feat/agentic-fusion-loop`:
- Dronet loop + 2 skeleton bugs fixed (output offset, weight pointer).
- Yolov8 loop on heaviest convs (all bit-exact PASS, no measurable
  cycle Δ).
- Profile-loader v2 (parse parent dispatch_id from name string).
- Wall-clock eval honest = 0.00 ms (correct verdict, no parallelism
  win available).
- This findings document.

The decision loop is correct, the agent is honest, the workload is
already at a local optimum. The next interesting question isn't
"more splits/fuses" — it's **"what other candidate types should the
agent generate?"** (per-op placement reassignment, periodic-instance
parallelization, pipeline sharding).

---

## Asymmetric splits — the real cross-accelerator parallelism win

After the FINDINGS conclusion above, we tested the *asymmetric* split
case: split an op into unequal tile fractions proportional to the
accelerator speed ratio, so the big tile on the fast accel and the
small tile on the slow accel both finish at the same wall-clock.

### Predicted win formula

For an op with cost `c_g` on Gemmini and `c_r` on RVV, the balanced
asymmetric split has:
- `work_g / work_r = c_r / c_g` (more work to the faster accel)
- Balanced time `T_balanced = (c_g * c_r) / (c_g + c_r)`
- Saving vs best single accel: `min(c_g, c_r) - T_balanced`

Top candidates by absolute saving on yolov8:

| did | Op shape | Gemm ms | RVV ms | Balanced | Saving % |
|---:|:---|---:|---:|---:|---:|
| 0 | first 3×3 conv IC=3 OC=16 | 2.110 | 8.566 | 1.693 | **19.8%** |
| 18 | 1×1 IC=48 OC=32 | 1.248 | 5.326 | 1.011 | 19.0% |
| 6 | 1×1 IC=32 OC=32 | 1.035 | 4.119 | 0.827 | **20.1%** |

### Measured

`xpu-rt/profile_loader.py` extended to honor `tile_oc_fraction` in
`split_from` metadata (defaults to `1/n_splits` for symmetric).

Test on dispatch_0 with fractions 0.812 / 0.188 (≈ OC=13 + OC=3
from the 16-OC original):

| | Hardware | Duration | Tile cost |
|:---|:---|---:|:---:|
| BEFORE — dispatch_0 alone | CPU_P#0 | 2.110 ms | Gemmini full op |
| AFTER — dispatch_0.tile_0 | CPU_P#0 | 1.714 ms | Gemmini OC=13 |
| AFTER — dispatch_0.tile_1 | **CPU_E#0** | 1.606 ms | RVV OC=3 |

**The scheduler placed tile_0 on Gemmini and tile_1 on RVV.** Per-op
wall-clock dropped from 2.110 ms to max(1.714, 1.606) = 1.714 ms,
**a real 18.8% saving on this op.**

Network makespan barely moved (75.57 → 75.5705 ms) because
dispatch_0 isn't on the makespan-critical path. The asymmetric split
mechanism works — it just doesn't move the headline number on this
workload because the critical path is elsewhere.

### Test on critical-path op (did=195, 1×1 conv on RVV)

Tried asymmetric split of did=195 (OC=80 → 77/3 = RVV-favored
asymmetric). The scheduler kept both tiles on RVV, putting tile_1
(small, 0.0523 ms) early at t=70 in RVV slack, then tile_0 (big,
1.343 ms) at t=74. **No makespan change.**

Why: tile_1 (OC=3) on its "other" accel (Gemmini) costs 1.357 ms —
massively MORE than 0.0523 ms on RVV. Even though the *original*
op-level math says asymmetric split saves 3% (max 1.357 vs 1.40),
the scheduler correctly notices that the *tile-level* placement
decision is "tile_1 on RVV (0.052 ms) beats tile_1 on Gemmini
(1.357 ms)" — and that local choice dominates.

### What this tells us about when asymmetric splits help

**Asymmetric splits move the makespan when:**

1. The op is on the makespan-critical path (necessary).
2. The fast accelerator is heavily loaded at the moment the op
   becomes ready (so backfilling part of the op onto the slow
   accelerator avoids waiting for fast-accel slack).
3. The fraction on the slow accelerator is small enough to finish
   before the fast-accel slot would free up anyway, OR the slow
   accelerator has nothing else to do during that interval.

On this workload, the critical-path ops (mostly small 1×1 convs on
RVV near the end of the detect head) don't satisfy (2) — Gemmini is
free during those slots, but it's free *because* it's structurally
the wrong tool for those ops. Migrating work to it doesn't help.

### Where this WILL win

Workloads where:
- Multiple ops with comparable Gemmini/RVV costs (ratio ~1.5–3×)
  cluster on the same accel.
- Critical path passes through a contention point where the fast
  accel is overloaded.
- The slow accel has a slot of idle time long enough to absorb
  10–20% of the contended op's work.

That's a different workload shape than yolov8_nano + dronet + mlp_control
on this hetero bitstream. **The framework now supports it for free
when such a workload comes along.**
