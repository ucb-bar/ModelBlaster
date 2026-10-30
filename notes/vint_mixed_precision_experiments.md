# ViNT Mixed-Precision Experiments

Log of mixed-precision configurations tried on ViNT, with cos_sim and
physical waypoint deltas vs PyTorch fp32 ground truth.

**Methodology.** Each row runs `extract_graph_export.py --quant int8
--per-channel --num-calibration 16` against IDSIA calibration samples,
with a per-model `get_precision_spec()` override naming which ops get
promoted to fp16. The auto-cast pass inserts cast_{i8_to_f16,f16_to_i8}
at dtype boundaries. Built for spike_riscv64 (Zfh-enabled spike for
fp16 ops). Spike runs against IDSIA calibration sample 0 (matches
io.npz baked-in input).

**Metrics** (all vs fp32 reference forward of the same input):

- `linear cos_sim`: goal-encoder output (1×512); the wide-range tensor
  (|max|≈181) that motivated mixed precision in the first place
- `linear_24 cos_sim`: final waypoint deltas (1×20), reshaped to 5×4
  (x, y, sin_θ, cos_θ) and post-processed via cumsum + L2-normalize
  in the harness tail
- `wp4 Δpos`: euclidean distance between fp32 ground truth and spike
  output at the 5th (furthest) waypoint
- `wp4 Δθ`: heading angle difference at wp4
- `Pilot ω@wp2 Δ`: difference in pilot steering signal computed from
  wp2 (the navigation-critical metric)

## Results

| # | Configuration | linear cos | linear_24 cos | wp4 Δpos | wp4 Δθ | Pilot ω Δ | Notes |
|---|---|---|---|---|---|---|---|
| 1 | All-int8 (per-channel, silu+mul_c1 fixed) | 0.972 | 0.986 | 0.87m | 32° | 0.328 | baseline |
| 2 | Mixed: only `linear` → fp16 (v1) | 0.972 | 0.986 | 0.86m | 33° | — | identical to baseline; cast back to i8 with same wide scale nullifies fp16 gain |
| 3 | Mixed: linear + cat_1 + layer_norm → fp16 (v2) | 0.972 | 0.986 | 0.86m | 33° | — | same as #2; cat/layer_norm at the boundary doesn't propagate the precision |
| 4 | Mixed: goal_encoder → fp16 (v3, **buggy** slice/pad emit) | 0.960 | 0.965 | 1.12m | 48° | 0.358 | regressed below baseline! cause: slice/pad/relu/add/maxpool ignored `_quant_for(n)` so the "promoted" region had int8 islands inside with i8↔f16 cast pairs around each |
| 5 | Mixed: **goal_encoder → fp16 (v4, fixed)** | **0.999** | **0.982** | **0.75m** | 37° | **0.224** | first real mixed-precision win. linear at fp16-ceiling accuracy. pilot ω error -32% vs all-int8 |
| 6 | Mixed: **goal_encoder + output_head (linear_19..22 + relus) → fp16 (v5)** | **0.999** | **0.9998** | **0.148m** | **4.5°** | **0.060** | huge step. linear_24 at near-perfect direction (0.9998); wp4 position 6× better than all-int8 (0.15m vs 0.87m), heading 7× better. NOTE: linear_23/linear_24 deliberately stay int8 because vint_action_post composite expects int8 inputs |
| 7 | Pure fp16 | 0.991 | 0.993 | 0.26m | 2° | ~0.000 | ceiling; ~2× cycles vs int8 |
| 8 | PyTorch fp16 reference | (~0.999) | (~0.999) | 0.006m | 0° | 0.000 | impossible-to-beat |

## Key observation from row 6

**Mixed v5 beats pure fp16 on wp4 position** (0.148m vs 0.26m). The
output-head promotion is doing more than just "match pure fp16" — it's
benefiting from the int8 transformer body's stable calibration AND
fp16 precision in the output MLP, getting the best of both. The
heading (Δθ 4.5°) is still slightly worse than pure fp16's 2°, but
better than goal-encoder-only mixed's 37° by an order of magnitude.

The configuration in row 6 is the *recommended deployment* for ViNT:
near-fp16 accuracy at roughly half the fp16 op count (transformer
body + obs_encoder stay int8 = ~280 ops vs ~600 if pure fp16).

## Architectural lessons

1. **Single-op promotions are useless when surrounded by int8.** The
   auto-cast pass inserts i8→f16 before the promoted op and f16→i8
   after — but both casts use the int8 calibration scale on either
   side, so any precision the fp16 op gains internally gets thrown
   away at the boundary. The cast-back uses the same wide-range scale
   that was the original problem.

2. **Region promotions need every op in the region to actually emit at
   the promoted precision.** The Phase 6 bug — `_emit_slice_c`,
   `_emit_pad`, etc. ignoring `_quant_for(n)` — meant the "promoted"
   region had int8 islands that the auto-cast pass faithfully
   surrounded with casts, destroying the precision claim. After the
   Phase 7 fix, the goal_encoder genuinely runs end-to-end in fp16,
   and `linear` cos_sim jumps from 0.960 → 0.999.

3. **`_record_tensor` must update dtype on re-record.** Fold-classified
   ops (pad, batchnorm) get their tensor recorded by `_visit` BEFORE
   the emit method runs. If `_record_tensor` short-circuits on
   "already recorded", the tensor stays at walker default dtype even
   when the actual op emits at a different precision. Fix: when an
   explicit dtype is passed and differs from the recorded value, update.

4. **Downstream calibration mismatch is a real but second-order
   concern.** Mixed precision can change the values that flow into
   int8 ops downstream of the fp16 region. Those int8 ops were
   calibrated against the fp32 forward, which matches what a
   properly-implemented fp16 region produces (closer to fp32 than int8
   would). So this is actually FINE — fp16 islands hand off values
   closer to the calibration target. The earlier Phase 6 conclusion
   ("downstream calibration mismatch") was a red herring; the real
   issue was the buggy emit methods.

## Next experiments to try

- **Goal encoder + output head fp16** (Phase 8): does promoting
  `linear_19..linear_24` recover the wp4 heading? predicted: yes
  for Δθ but maybe at the cost of slightly more cycles
- **Goal encoder + transformer layer_norms fp16**: targets the
  zero-mean signal that int8 handles worst
- **Output head only fp16**: cheap test, isolated win in the heading
  channel

## Implementation references

- Plan: modelblaster/notes/mixed_precision_plan.md
- Walker: modelblaster/pipeline/extract_graph_export.py `_ExportWalker`,
  `_quant_for`, `insert_casts`, `_resolve_op_precision`
- Per-model API: `get_precision_spec()` in `modelblaster/models/<name>.py`
- CLI overrides: `--fp16-ops`, `--fp16-patterns`
