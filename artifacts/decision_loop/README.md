# Phase 3 — measured-grounded fuse/split decision loop

The research question this directory answers:

> Given the current schedule, **when does fusing a dispatch help, when does
> splitting a dispatch help, and how do we drive that decision?**

This is the report. All numbers below are **measured cycles on the spike
simulator** (no profile-DB bookkeeping, no predicted scaling, no faked
deltas — the bug from `artifacts/agentic_fuse_split/WARNING.md` is
explicitly avoided by gating accept/reject on spike-measured cycle
counts).

---

## Setup

- **Workload:** Dima's reference
  `networks_mlp10_dronet20_yolov8_firesim_static_q31profile.json`
  (16×mlp_control @ 10 ms + 8×dronet @ 20 ms + 1×yolov8_nano).
- **Baseline solver:** `decomposed` (Dima's choice — periods honored,
  see `artifacts/periodic_solvers/README.md` for why other solvers
  weren't valid candidates).
- **Realization target:** `mlp_control / int8 / rvv_opu` (Saturn OPU
  bitstream) — the network we have a working IR rewrite path for.
- **Backend:** `reference` (curated kernels). This is a deliberate
  choice — see "On backend choice" below.
- **Inner verify oracle:** spike (cheap, deterministic). FireSim
  escalation is wired in via `--runner firesim` but not used here
  because the 16+8+1 workload's bitstream stability is a separate
  blocker (notes/firesim_measured_status.md).

Driver: `scripts/decision_loop.py`.
Measurement primitive: `scripts/measure_candidate.sh` →
`scripts/ingest_measured_cycles.py`.

---

## Result table

### Fusion candidates (round_003, --K 3, decomposed baseline)

| candidate | predicted Δµs | measured Δcycles | measured Δ% | accept? | reason |
|:---|---:|---:|---:|:---:|:---|
| baseline (no hint)                | — | — | — | — | 730,069 cycles total mlp_control |
| fuse_chain_mlp_control0[0..5]     | +0.00 | **−136,434** | **−18.7%** | ✗ REJECT | curated `linear_s8_elu_s8` (scalar) is slower than per-op variants |
| fuse_chain_mlp_control1[0..5]     | +0.00 | −136,434 | −18.7% | ✗ REJECT | functionally identical to control0 — same IR template |
| fuse_chain_mlp_control2[0..5]     | +0.00 | −136,434 | −18.7% | ✗ REJECT | functionally identical to control0 — same IR template |

### Split candidates (manual probe)

| candidate | result | reason |
|:---|:---|:---|
| split_linear_s8 mlp_control dispatch_2 (N=128 → 2×64) | ✗ BUILD FAIL | `generate_skeleton` doesn't allocate buffers for tile outputs — model.c references `buf_mlp_2_tile_0/1` that don't exist |
| split_conv2d_s8 dronet (granularity loop's top split) | ⊘ NOT REALIZABLE | `apply_split_hint` Phase 1e covers `linear_s8` only — conv2d weight surgery is a follow-up |

---

## What this tells us

### When does fusion help?

On `mlp_control / int8 / rvv_opu / BACKEND=reference`: **it doesn't.**
The curated `linear_s8_elu_s8` is a straight scalar combined-loop
(linear MAC → Q0.31 requantize → ELU per element). It does the same
total work as `kernel_linear_s8 + kernel_elu_s8` chained, *plus* the
ELU is now inside the linear loop where it can't be hoisted out by the
compiler. Net: 730k → 866k cycles, **−18.7% slower**.

Fusion **could** help if:
1. The fused kernel does *true compute fusion* — accumulates in
   register and applies the ELU without ever writing the intermediate
   tensor to memory. The curated `linear_s8_elu_s8` does this for the
   sub-`mlmax` case, but on this workload (N up to 256) it falls back
   to the scalar path.
2. The fused kernel uses target-specific hardware (e.g. an
   LLM-generated OPU outer-product matmul with the ELU emitted as a
   `VRGATHER` lookup inside the same dispatch). We don't have this
   kernel today — Bedrock generated a per-op OPU `linear_s8` this
   session (7.1× speedup measured) but not the fused `linear_s8_elu_s8`
   variant.
3. Many sub-µs dispatches whose worker-handshake overhead dominates.
   In this workload, the smallest mlp_control dispatch (`mlp.6`, K=64
   N=4) is 2,755 cycles ≈ 3 µs — already above the handshake-dominated
   regime.

**The decision loop's verdict is correct:** at the current kernel
state, fusion costs cycles. The agent rejects.

### When does splitting help?

On the only candidate the granularity advisor proposed (dronet
conv2d_s8), **we can't tell yet — the realization path doesn't ship
conv2d_s8 splits**. The agent did the right thing: it logged the
candidate as "not yet realizable" (`apply_split_hint.py` Phase 1e
covers `linear_s8` along N only) instead of pretending to score it.

On a manual linear_s8 split probe (mlp_control dispatch_2, N=128 →
2×64), we hit a different layer's gap: the IR rewrite is correct (the
Phase 1e unit tests pass) but `generate_skeleton.py` doesn't allocate
per-tile output buffers, so the harness build fails at link time. This
is a small skeleton-emitter fix that would unblock measured-split
testing.

### How do we drive the decision?

The driver in `scripts/decision_loop.py`:

1. **Generate candidates** (`granularity_loop.py` — predicted scoring,
   330 candidates per round on this workload).
2. **Filter by realizability** — only consider candidates one of our
   IR rewriters can emit. Skipped candidates are logged + counted, not
   silently dropped (this session's verification: 5 fuses realizable +
   3 splits proposed-but-not-realizable, of which 3/5 fuses measured
   and 0/3 splits measured).
3. **Measure top-K via the primitive** (`measure_candidate.sh`):
   apply hint → swap IR → build harness → spike verify (max_abs_err=0)
   → ingest per-dispatch cycles from the run output.
4. **Accept iff** measured Δcycles > `epsilon_cycles` (default 1,000,
   roughly the noise floor of an mtime-counter mlp_control run) AND
   verify PASS. Reject otherwise, with the reason logged.
5. **No multi-round loop yet** — this report is round 1. Multi-round
   iteration (accept a candidate, re-baseline on the accepted IR,
   re-run advisor) is a follow-up; the per-round primitive works,
   the loop wrapper around it is a ~30-line addition.

### On backend choice

Why `BACKEND=reference` and not `BACKEND=llm`?

- Bedrock generated the OPU per-op `linear_s8` kernel this session
  (102k cycles total mlp_control, 7.1× speedup vs scalar). It did
  *not* generate a `linear_s8_elu_s8` fused-OPU kernel; the
  AlgorithmCandidate exists but the LLM attempt this round produced
  invalid macro usage (forgot the `SATURN_OPU_KEEP_REGISTER_MACROS`
  guard despite the prompt — Bedrock output is stochastic).
- An apples-to-apples comparison requires both baseline and
  candidate use the same backend. With `BACKEND=llm` and no fused
  LLM kernel cached, the candidate falls through to a different
  code path than the baseline → unfair comparison.
- The cleanest experiment is **both reference** (this report's
  numbers) or **both LLM** (would need Bedrock to successfully
  generate a fused-OPU kernel). The honest comparison
  `LLM-per-op vs LLM-fused` is the follow-up question and would
  almost certainly show fusion helping (because both halves get
  OPU acceleration).

---

## Reproducing the result

```bash
cd /scratch2/agustin/ModelBlaster

# Clean cache so the kernel pick re-runs
rm -rf examples/mlp_control/int8/build/rvv_opu \
       examples/mlp_control/int8/cache/rvv_opu \
       examples/mlp_control/int8/generated/rvv_opu/kernels.c

source scripts/setup_benchmark_env.sh  # west, spike, ZEPHYR_BASE
PYTHONPATH=/scratch2/agustin/XPU-RT/xpu-rt \
    /scratch2/agustin/miniforge3/envs/merlin-dev/bin/python \
    scripts/decision_loop.py \
    --networks-json /scratch2/agustin/XPU-RT/data/toplevel/networks_mlp10_dronet20_yolov8_firesim_static_q31profile.json \
    --baseline-solver decomposed \
    --network mlp_control --quant int8 --target rvv_opu \
    --backend reference \
    --K 3 \
    --out-dir artifacts/decision_loop/round_003
```

Inspect:
- `round_003/summary.json` — full structured outcome.
- `round_003/cand_*/spike.log` — per-candidate spike output with the
  raw cycle table.
- `round_003/baseline/measured_cycles.json` — baseline ingested cycles.
- `round_003/cand_*/measured_cycles.json` — per-candidate ingested
  cycles.

---

## Open follow-ups (not blocking the decision driver)

1. **Bedrock-generated `linear_s8_elu_s8` for OPU** — would let us
   answer "does LLM-fused beat LLM-per-op". The prompt guard is the
   issue (`SATURN_OPU_KEEP_REGISTER_MACROS`); tightening the seed to
   make the macro guard mandatory would help.
2. **`generate_skeleton.py` tile-buffer emission** — unblocks measured
   linear_s8 split experiments.
3. **conv2d_s8 split realization** — the agent will keep proposing
   conv2d splits on dronet/yolov8; today the loop logs them as
   not-yet-realizable. Phase 1f of the older plan covers this.
4. **Multi-round iteration** — accept a candidate, re-baseline on the
   accepted IR, repeat. The per-round primitive works; this is a wrapper.
5. **FireSim escalation** — `--runner firesim` is wired in. Currently
   blocked on the same bitstream stability issue as #163. When that
   clears, the same decision loop runs with FireSim measurements.

---

## Round 4 (post-prompt-tightening): measured BACKEND=llm

After the `SATURN_OPU_KEEP_REGISTER_MACROS` guard was made mechanical
(injected at `emit_kernels_c` time in `generate_kernels.py`, no longer
LLM-stochastic), we re-ran the loop with **both** baseline and fuse
candidate using `BACKEND=llm` (Bedrock-generated kernels):

| | Strategy | Cycles |
|:---|:---|---:|
| BASELINE (per-op) | Bedrock OPU `linear_s8` (outerprod) + scalar `elu_s8` | **96,597** |
| FUSE candidate | Bedrock scalar `linear_s8_elu_s8` (auto-chose scalar fusion) | 862,925 |
| Δ | | **−793 % worse** |

**Decision: REJECT.** The agent correctly refused the fuse hint
because the measured cost is dramatically higher.

### Why Bedrock chose scalar fusion (real finding)

The `linear_s8_elu_s8` AlgorithmCandidate didn't have an OPU-targeted
seed in the prompt — its reference_impl is the curated scalar chain.
When Bedrock got the `linear_s8` AlgorithmCandidate it had the
`outerprod` seed (OPMVINBCAST / VOPACC / VMV_VR pattern) and
reproduced it bit-exact. With no equivalent seed for the fused op, it
defaulted to "just chain the scalar implementations."

This is the actual research answer:

> **Fusion helps when the fused kernel reaches for the same hardware
> features the per-op kernels do.** On Saturn OPU + `linear_s8_elu_s8`,
> the fused kernel that the LLM auto-generates does NOT reach for OPU
> macros — so fusion collapses to scalar and loses 9× to the per-op
> variants. The agent correctly rejects this in 1 round.

### What would change the verdict

Adding an OPU-targeted `linear_s8_elu_s8` AlgorithmCandidate with the
right seed (outerprod accumulator + in-register ELU via `vfmacc` or
table lookup before drain). That's an LLM-prompt change in
`pipeline/reference_kernels.py`, not a kernel that needs to be
hand-written — Bedrock can produce it, given the right seed. Follow-up.

