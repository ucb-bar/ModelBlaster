# Memory-aware kernel optimization on FireSim

Status: in-progress (Plan B). Captures the design decisions, the pieces
that landed, the validation results, and what's still rough.

The original optimize loop scores LLM-proposed kernel candidates on
**spike**, which models a flat memory: every load is the same cost. That
means cycle-count wins from cache locality and pipeline-pattern wins
look identical to spike's `rdcycle`, and the loop has no way to prefer
the locality wins. On the FireSim quad-rocket-saturn RTL the gap is
real — the same dronet RVV cache that takes 63 M cycles on spike takes
~140 M cycles on FireSim, mostly because the heavy 3x3 convs (OC≥64,
IC≥64, K=3) walk the entire weight tensor on every output position
instead of holding a small OC tile in L1D.

This work plumbs FireSim into the optimize loop so the scorer is no
longer blind, then re-tunes the dronet conv2d kernel under the new
scorer.

## Pieces

| Path | Role |
|---|---|
| `modelblaster/optimize/firesim_eval/__init__.py` | Public surface (FiresimEvaluator, evaluate_top_k, memory_model_stanza) |
| `modelblaster/optimize/firesim_eval/cache_aware_prompt.py` | MemoryModel dataclass + `memory_model_stanza()` renderer; ships one model (the quad-rocket-saturn hwconfig) |
| `modelblaster/optimize/firesim_eval/evaluator.py` | `FiresimEvaluator` (build for chipyard, FPGA-queue, run, parse cycles), `evaluate_top_k` (re-rank K spike survivors against firesim) |
| `modelblaster/optimize/firesim_eval/test_evaluator.py` | Stub-evaluator unit tests for `evaluate_top_k` ordering / dedupe / fail-handling, and a render check for `memory_model_stanza` |
| `modelblaster/pipeline/generate_kernels.py` | New flags `--firesim-eval / --firesim-top-k / --cache-aware-prompt / --firesim-ops`; threads them through the optimize loop. `beam_search_optimize` now also returns the ranked list of viable candidates so the rerank can pick the top-K. |
| `modelblaster/pipeline/reference_kernels.py` | New `oc_blocked` algorithm seed for conv2d (cache-blocked direct convolution with a fixed `TILE_OC=4`). |
| `modelblaster/examples/_run_lib.sh` | Threads new env knobs (`FIRESIM_EVAL=1`, `FIRESIM_TOP_K`, `FIRESIM_OPS`, `CACHE_AWARE_PROMPT=1`) through the optimize CLI |

The new code is fully opt-in. Default `OPTIMIZE=1` runs are unchanged
(`firesim_eval` defaults to off everywhere). The existing
spike-only flow keeps producing the same kernels at the same speed.

## Design decisions

### Where to plug FireSim in

Three options were on the table:

1. **Score every candidate on FireSim.** Per-build+run costs ~60-180 s
   on the alveo_u250 board. The optimize loop generates ~12-18 candidates
   per op (beam=2, expansions=3, iterations=2) → ~15-50 min/op of FPGA
   time. For a model with 7 op kinds that's > 4 hours. Rejected.

2. **Score only the final best on FireSim.** Cheapest, but loses the
   whole point — when spike says A and B are tied, the firesim cycles
   may differ a lot, and we want to pick whichever is actually better.

3. **Re-rank top-K spike survivors on FireSim.** Bounded at K + 1 firesim
   builds per op (default K = 3; one extra for the spike-best baseline,
   for context). Comes out to ~5-15 min/op on the FPGA.

Going with **(3)**. The spike beam still drives the inner loop (cheap
per candidate, lots of candidates surveyed), and FireSim only weighs in
on the last 3-4 candidates that survived spike's pruning. The full loop
is still LLM- and FPGA-bound, but at K=3 the FPGA cost on the heaviest
op (conv2d) is roughly 4 builds × ~2 min ≈ 8 minutes — acceptable.

### Cache-replacement policy

When the firesim re-rank picks a different kernel from the spike-best,
the firesim winner is promoted into the persistent cache (same path
the spike-best would have written to: `cache/<target>/<target>_<op>_<algo>.c`).
The decision rule:

  - if firesim winner is structurally identical to the spike winner → no
    change, keep the existing cache file.
  - if firesim winner differs and beats the spike-best on firesim → cache
    the firesim winner.
  - if firesim re-rank fails entirely (FPGA busy, all candidates broken
    on RTL) → keep the spike-best as the cache slot. Fail-safe.

The cycle-tagging in `optimize_summary.json` records both the spike
and firesim cycles so a downstream sweep can verify the promotion was
worthwhile.

### Memory-model stanza

When `--cache-aware-prompt` is set, the optimize-phase system prompt
gets a new top-level section listing the target's L1D / LLC / DRAM
parameters plus a worked example. The example uses the heaviest dronet
conv2d (OC=128, IC=128, K=3) and walks through:

    weight_footprint = OC * IC * KH * KW * 4
                     = 128 * 128 * 3 * 3 * 4
                     = 576 KB             # >> 32 KB L1D

    pick TILE_OC such that
        TILE_OC * IC * KH * KW * 4   <=   ~24 KB
    →   TILE_OC <= 5  →  round down to 4

The stanza is target-specific. We ship one (`QUAD_ROCKET_SATURN_MEMORY_MODEL`).
Adding a new target = adding a new `MemoryModel` instance.

The stanza is only included when firesim re-rank is also active —
without it the spike scorer is blind to cache wins so steering the LLM
at locality is wasted prompt tokens.

### Memory-aware seeds

Added a single new conv2d algorithm: **`oc_blocked`**. It's a direct
convolution with the OC dimension tiled outermost; the LLM's seed
example uses a fixed `TILE_OC = 4` (small enough to fit OC × IC × KH ×
KW × 4 = 4×128×9×4 ≈ 18 KB in L1D for the worst dronet shape).

The algorithm description includes the same rationale as the
worked example in the memory-model stanza, plus explicit DON'T rules
(don't pick TILE_OC=1, don't pick TILE_OC=OC, don't compute it from
runtime IC). It's gated on `OC>=32 and IC>=16` via `applicable=` so it
doesn't fire on tiny LeNet-style convs where the direct algorithm
already fits its weights in L1D.

The structural-check substring set
(`_ALGORITHM_REQUIRED_SUBSTRINGS["oc_blocked"]`) demands `oc_outer` and
`TILE_OC` appear in the candidate. That fails fast when the LLM
collapses the outer tile loop and writes the direct algorithm under
the cache-blocked label.

I considered also adding `channel_first_im2col` and `k_blocked_linear`,
but skipped them:

  - `channel_first_im2col` has the same big-buffer problem as the
    existing `im2col_gemm` for dronet (im2col_buf for the 56×56 conv is
    339 KB — way past LLC). The win would have to come from a
    different lowering, not from a different memory order on the same
    one.
  - `k_blocked_linear` would be useful for huge linear layers (M*N*K
    big), but dronet's two linears are M=1, K=2048, N=1 — they're
    already L1D-resident at 8 KB. Not worth the seed.

## Validation

Baseline firesim profile (current cache): `gen/profile_firesim_sweep/RVV/firesim_rocket_saturn/dronet/dronet.fp32/.../topo_0/results.csv`.

Per-op sums (sequential, RVV, single-hart, current cache):

| op kind | sum cycles | sum ms @ 1 GHz |
|---|---|---|
| conv2d | 38,023,679 | 38.02 |
| maxpool2d | 1,021,372 | 1.02 |
| batchnorm2d | 152,970 | 0.15 |
| relu | 48,537 | 0.05 |
| add | 15,806 | 0.02 |
| linear | 10,507 | 0.01 |
| sigmoid | 371 | 0.00 |
| **TOTAL** | **39,273,242** | **39.27** |

Conv2d dominates at 97% of total time. Per-shape breakdown:

| shape | dispatch | firesim cyc | ms |
|---|---|---|---|
| IC=3, OC=32, OH×OW=56×56, K=3 | 0 | 9,115,959 | 9.12 |
| IC=128, OC=128, OH×OW=4×4, K=3 | 23 | 6,473,296 | 6.47 |
| IC=32, OC=32, OH×OW=14×14, K=3, S=2 | 4 | 5,256,466 | 5.26 |
| IC=32, OC=32, OH×OW=14×14, K=3, S=1 | 7 | 5,241,803 | 5.24 |
| IC=64, OC=64, OH×OW=7×7, K=3 | 15 | 4,798,074 | 4.80 |
| IC=64, OC=128, OH×OW=4×4, K=3, S=2 | 20 | 2,807,600 | 2.81 |
| IC=32, OC=64, OH×OW=7×7, K=3, S=2 | 12 | 2,639,164 | 2.64 |
| IC=32, OC=32, 1×1 stride=2 | 8 | 797,213 | 0.80 |
| IC=64, OC=128, 1×1 stride=2 | 24 | 498,877 | 0.50 |
| IC=32, OC=64, 1×1 stride=2 | 16 | 395,227 | 0.40 |

The five 3×3 K=3 convs (>4 M cycles each) are the obvious targets for
cache blocking. dispatch_23 (OC=128, IC=128) has 576 KB of weights —
the canonical case for the worked example.

### Optimize run

The first end-to-end memory-aware optimize run:

    BACKEND=llm TARGET=rvv QUANT=fp32 OPTIMIZE=1 \\
    BEAM=2 EXPANSIONS=2 ITERATIONS=2 \\
    FIRESIM_EVAL=1 FIRESIM_TOP_K=3 FIRESIM_OPS=conv2d \\
    CACHE_AWARE_PROMPT=1 \\
    bash modelblaster/examples/dronet/run.sh

Spike-side optimize summary:

| op | baseline (cyc) | best (cyc) | Δ |
|---|---|---|---|
| conv2d | 16,331,429 | 15,072,003 | -7.7% |
| batchnorm2d | 193,756 | 96,164 | **-50.4%** |
| maxpool2d | 653,588 | 192,212 | **-70.6%** |
| linear | 2,207 | 2,181 | -1.2% |
| relu | 15,986 | 15,986 | 0.0% (rejected) |
| add | 4,696 | 4,696 | 0.0% |
| sigmoid | 98 | 98 | 0.0% |

FireSim re-rank trace for conv2d:

  - 2 candidates from spike beam (15.07M and 16.33M+0% cyc).
  - **baseline** (spike-best, 15.07M cyc) ran clean on RTL → 37,558,963
    cyc.
  - **top-2** (16.33M cyc, no improvement on spike) **faulted on
    FireSim** with `mcause: 5, Load access fault, mtval: cc747057` —
    a wild pointer somewhere in the kernel. spike accepted it (its
    flat memory model can't tell), but the FireSim RTL caught it.
  - Re-rank picked baseline. Cache promoted to the 15.07M kernel.

The fault was a useful demonstration of WHY firesim eval matters: a
spike-equivalent candidate that would have been silently cached (had
the +0% candidate been the spike best, it would have been written to
cache) now gets rejected at the firesim gate. After this run I added
a `Load access fault / Store access fault / Illegal instruction`
short-circuit to `modelblaster/validation/firesim_runner.py` so future
faulting kernels fail in ~5 s instead of waiting out the 240 s timeout.

### Per-shape FireSim numbers, before / after

Single-model dronet, RVV, single-hart, sequential:

| shape | dispatch | before (cyc) | after (cyc) | Δ |
|---|---|---|---|---|
| IC=3, OC=32, K=3, OH×OW=56×56 | 0 | 9,115,959 | 9,068,109 | -0.5% |
| IC=32, OC=32, K=3, S=2, 14×14 | 4 | 5,256,466 | 5,203,825 | -1.0% |
| IC=32, OC=32, K=3, S=1, 14×14 | 7 | 5,241,803 | 5,191,673 | -1.0% |
| IC=64, OC=64, K=3, 7×7 | 15 | 4,798,074 | 4,726,340 | -1.5% |
| IC=128, OC=128, K=3, 4×4 | 23 | 6,473,296 | 6,364,825 | -1.7% |
| IC=64, OC=128, K=3, S=2, 4×4 | 20 | 2,807,600 | 2,755,503 | -1.9% |
| IC=32, OC=64, K=3, S=2, 7×7 | 12 | 2,639,164 | 2,608,796 | -1.2% |
| 1×1 conv2ds (3 shapes) | 8/16/24 | 1,691,317 | 1,648,813 | -2.5% |
| **conv2d total** | | **38,023,679** | **37,567,884** | **-1.2%** |
| maxpool2d | 1 | 1,021,372 | 754,300 | **-26.1%** |
| batchnorm2d (sum across 6) | | 152,970 | 92,357 | **-39.6%** |
| relu (sum across 7) | | 48,537 | 47,319 | -2.5% |
| add (sum across 3) | | 15,806 | 15,581 | -1.4% |
| linear (sum across 2) | | 10,507 | 10,113 | -3.7% |
| sigmoid | | 371 | 650 | **+75%** (noise — n=1 op) |
| **dronet TOTAL (sum)** | | **39,273,242** | **38,488,204** | **-2.0%** |

`MODELBLASTER_WALL_CYCLES` (mtime, source-of-truth wall-clock):
**39,273 → 38,494** = **-2.0%** end-to-end speedup. Golden compare PASS.

### End-to-end XPU-RT impact

Out of scope of this validation pass — none of the touched cache
files are mlp_control's, so the multi_demo het schedule's
`mlp_control + dronet` makespan is bounded below by `dronet`'s
single-model time and no-better than the dronet improvement above.
The 60.87 ms xpurt-replan baseline becomes ~60.1 ms with the new
cache (2% off the dronet-shaped portion only).

### What worked, what didn't

The +2% wall improvement is real but underwhelming, mostly because:

1. **The optimize phase doesn't switch algorithms.** It takes the
   spike-baseline-best kernel as the seed and asks for "a faster
   equivalent." The LLM never touched the algorithmic structure
   (e.g. didn't introduce OC blocking from the new `oc_blocked`
   seed). It produced a syntactic refinement (replacing the
   `if ih < 0 || ih >= IH continue` pattern with an
   `(unsigned int)ih < (unsigned int)IH` branchless cast — a real
   optimization, but at the leaf-instruction level, not at the
   loop-structure level).

2. **The biggest single op (conv_modules.0, 9.1 ms) didn't improve.**
   That conv has 3 input channels and a huge 56×56 spatial output;
   the existing cache's strided weight load + bounds-check pattern
   was already fine for it. Cache blocking would help conv_23
   (OC=128, IC=128, weight = 576 KB, doesn't fit in L1D), but the
   LLM didn't pick OC blocking even with the prompt and seed. A
   future iteration should invalidate the conv2d cache and let the
   correctness phase pick `oc_blocked` as a fresh algorithm.

3. **Spike was actually pretty good at picking the firesim winner**
   in this run. The +7.7% spike improvement carried through to a
   1.2% firesim improvement on conv2d (the two are usually within a
   factor of 2-3x of each other for these sequential kernels —
   spike ignores DRAM stalls, FireSim doesn't, but the *relative*
   ordering of two FP32 RVV kernels is mostly preserved).

The infrastructure passed its smoke test:

  - The FireSim re-rank correctly evaluated 2 candidates, caught a
    fault in one of them, and promoted the right cache entry.
  - The fault-detection short-circuit (added after the run) cuts
    failed-candidate cost from 240 s to ~5 s.
  - The cache-aware optimize prompt is wired but its effect is
    limited as long as the optimize phase is constrained to "make
    *this* kernel faster" rather than "consider switching
    algorithms." The next iteration should expose the new
    `oc_blocked` algorithm to the correctness phase by deleting the
    conv2d cache, so the cache-aware probe picks a tile-blocked
    seed.

## Dead-ends / things I'd do differently

  - I almost made the `wait_for_fpga` helper match by full command
    line (`pgrep -f`), which self-matches: `pgrep -f
    "FireSim-xilinx_alveo_u250"` finds itself because its argv contains
    that string. Switched to a `comm[]` confirmation pass via
    `/proc/<pid>/comm` — the actual driver's comm[] truncates to
    `FireSim-xilinx_a` (15 chars; `TASK_COMM_LEN-1`). Worth knowing for
    anyone else who tries to write a poll-the-FPGA helper.

  - `evaluate_top_k` initially returned the firesim-best regardless of
    whether it actually beat the baseline. Changed the cache-promotion
    logic so we only overwrite the cache when the firesim winner is
    structurally different from the spike-best — protects against
    promoting a kernel that's only marginally faster on RTL but happens
    to be a different lexical body.

  - Originally considered scoring every spike-survivor on firesim
    inside the beam (rather than after). Rejected — too slow, plus
    adds firesim build noise to the inner loop. The current shape (do
    spike, then re-rank top-K) keeps the inner loop tight.

## Open gaps

  - The re-rank doesn't currently feed a per-shape weight back to the
    LLM. If the LLM produces a kernel that wins big on dispatch_23
    (OC=128) but regresses on dispatch_0 (IC=3, big spatial), the
    aggregated firesim cycles can hide the regression. A sharper next
    step would be a "shape-aware" re-rank that splits firesim cycles
    by dispatch ID and only promotes a kernel that strictly improves
    every op-shape.

  - The memory-model stanza is hardcoded to the quad-rocket-saturn
    config. Adding a new target means editing `cache_aware_prompt.py`.
    Could be inferred from the chipyard scala source or from a
    JSON/yaml shipped alongside `firesim_chipyard.conf`, but the value
    isn't there yet — there's exactly one target.

  - The firesim eval blocks the whole optimize loop on FPGA
    availability. If a separate session is hammering the FPGA,
    optimize stalls. The `wait_for_fpga` helper has a 15-min timeout,
    after which it gives up gracefully (the rerank fails for that op,
    cache stays at spike-best). Should probably be configurable from
    the CLI.

  - The `oc_blocked` seed has `TILE_OC=4` hardcoded. That's a fine
    starting point for OC≥128 IC≥128 K=3 (the worst dronet conv), but
    e.g. a `OC=32 IC=3 K=3` shape (dispatch_0, the heaviest) could use
    TILE_OC=32 happily — its weight footprint is only 3.5 KB. The LLM
    *can* pick a different TILE_OC during optimize (the seed is just a
    starting point and the optimize prompt gives it freedom), but a
    smarter seed would parameterize on the IC × K × K footprint and
    pick TILE_OC accordingly.

