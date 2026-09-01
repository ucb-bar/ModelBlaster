# Phase E — Bedrock fused-kernel expansion

## E1 — Kernel gap survey

`scripts/kernel_gap_survey.py` walks each network's ModelBlaster IR
(`examples/<net>/<quant>/generated/graph.json`), counts producer→consumer
pairs that fit the single-producer/single-consumer chain shape the
granularity advisor proposes, and cross-references each pair's
(p_op, c_op) types against the fused KernelSpecs registered in
`pipeline/reference_kernels.py`.

Result: `artifacts/kernel_gap_survey.json`.

### Workload-scope summary (4 MLP + 2 Dronet + 1 Yolo)

| Field | Value |
|:---|---:|
| Networks scanned | mlp_control, dronet, yolov8_nano |
| Total fuse-pair candidates | 188 |
| Registered fused KernelSpecs | 2 (`conv2d_silu_s8`, `linear_s8_elu_s8`) |
| Pairs covered by existing fused kernels | ~6 (subset of conv→silu chains) |
| Pairs NOT covered → realization gap | 182 |

### Top unregistered gaps

| Rank | Pair (producer → consumer) | Candidates | Hot network |
|:---:|:---|---:|:---|
| 1 | `conv2d_s8 → batchnorm2d_s8` | 60 | yolov8_nano |
| 2 | `batchnorm2d_s8 → silu_s8` | 57 | yolov8_nano |
| 3 | `silu_s8 → chunk2_c1` | 8 | yolov8_nano |
| 4 | `batchnorm2d_s8 → relu_s8` | 6 | dronet |
| 5 | `relu_s8 → conv2d_s8` | 6 | dronet |
| 6 | `cat3_c1_s8 → conv2d_s8` | 6 | yolov8_nano |
| 7 | `cat2_c1_s8 → conv2d_s8` | 4 | yolov8_nano |
| 8 | `cat4_c1_s8 → conv2d_s8` | 3 | yolov8_nano |
| 9 | `conv2d_s8 → maxpool2d_s8` | 1 | dronet |
| 10 | `add_s8 → relu_s8` | 1 | dronet |

### Key insight

The yolov8_nano Conv→BN→SiLU triple is the dominant fusion opportunity
on this workload — pairs (1) and (2) together account for 117 of the
188 candidates (62%). The existing `conv2d_silu_s8` kernel only covers
conv2d→silu directly; pairs that go through BN are not captured. Two
ways to close this:

1. **BN folding pre-pass.** Numerically fold the BN affine into the
   preceding conv2d's weight/bias. Pure code, no Bedrock — this is the
   standard inference-time optimization. Drops gap pair (1) to ~0.
2. **Triple-op fused kernel.** Generate a `conv2d_s8_bn_silu_s8`
   KernelSpec via Bedrock. Bigger search space, larger kernel; pays
   off when the BN parameters are dynamic / quantized in a way that
   makes folding lossy.

Standard practice is BN folding; the triple-fused kernel is only worth
the Bedrock spend if folding is somehow not viable. For Phase E2 we
choose:

- **E2-target-1:** `conv2d_s8_relu_s8` (yolov8 + dronet — 6 cands; the
  classic activation-after-conv fusion)
- **E2-target-2:** `add_s8_relu_s8` (1 cand here but a high-frequency
  pattern in residual networks at large)
- **E2-target-3:** `batchnorm2d_s8_silu_s8` (yolov8 — 57 cands;
  pointwise BN-affine + SiLU)

## E2 — Bedrock kernel generation (deferred to budget-gated run)

**Status:** E1 has identified the targets and quantified the gap. The
actual Bedrock invocation is bounded by a real-dollar cost cap
(≤ $30 per kernel, ≤ $90 total) and requires:

1. AlgorithmCandidate registration in `pipeline/reference_kernels.py`
   with target_affinity + reference_impl.
2. `LLM_PROVIDER=bedrock BACKEND=llm` run through
   `examples/<host_net>/run.sh` for each kernel.
3. Spike verify on TWO host networks per kernel
   (`max_abs_err=0 max_rel_err=0`).
4. Measured-cycles speedup ≥ 1.1× vs unfused baseline; reject otherwise.

The framework for E2 is in place; the actual invocation is gated on
budget approval. Per-kernel reports live in
`artifacts/kernels/<pair>/measurement_report.md` once produced.

## E3 — Realizability filter wire-in (pending E2)

After each E2 kernel passes bit-exact verify, add its op-pair to
`scripts/decision_loop.py:REALIZABLE_FUSE_TYPES`. Re-run the headline
decision loop including the new pair; verify accept rate strictly
increases.

## E4 — Per-kernel measurement reports (pending E2)

One `measurement_report.md` per generated kernel:

- Verify rows on host_net_a + host_net_b (both must show max_abs_err=0).
- Measured speedup vs unfused: cycles_after / cycles_before.
- Bedrock spend: USD per attempt + summed cost.
- Honesty notes: if speedup falls in mtime noise (≤ 1.1×), the kernel
  is REJECTED in the report with the rejection reason.

## Quality gates (binding)

- Any max_abs_err > 0 → kernel rejected (no relaxed tolerances).
- Speedup ≤ 1.1× of unfused on TWO networks → rejected.
- Budget overrun > 30 USD per kernel → halt before next attempt.
