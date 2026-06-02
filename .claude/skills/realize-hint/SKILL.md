---
name: realize-hint
description: Apply an XPU-RT axis-C fusion hint (modelblaster.fusion_hints/v1) to a network's IR — rewrite graph.json to collapse the listed dispatch chains into one synthetic __fused__ op per group, emit a graph.fused.json. Use when XPU-RT's granularity_loop emitted a fusion hint and you need ModelBlaster to realize it before re-profiling.
---

# realize-hint

Apply a Contract-2 fusion hint produced by XPU-RT's `granularity_loop.py`
to a ModelBlaster IR. The output is a rewritten `graph.fused.json` that
collapses each `fuse_group` (a topologically-ordered chain of
dispatch_ids) into a single synthetic op named
`__fused__<op0>__<op1>__...__<opN>` carrying the originals under
`sub_ops`. Downstream `depends_on` is rewired and dispatch_ids are
reassigned contiguously, so the rewritten IR is immediately consumable
by `pipeline/emit_dispatch_graph.py`.

The IR-rewrite pass is the foundational piece for the
predicted-vs-measured granularity loop with XPU-RT — see
`notes/plans/agentic_fusion_loop.md`.

## Steps

1. **Locate the hint** (typically produced by XPU-RT's
   `scripts/granularity_loop.py --emit-hint <path>`):
   ```bash
   HINT=/scratch2/agustin/XPU-RT/artifacts/iterate/granularity_hint.json
   jq -r '.networks[] | "\(.network): fuse_groups=\(.fuse_groups | length) n_tiny=\(.n_tiny)"' "$HINT"
   ```

2. **For each network in the hint, apply the rewrite:**
   ```bash
   python3 -m pipeline.apply_fusion_hint \
       --hint  "$HINT" \
       --model mlp_control \
       --ir    examples/mlp_control/int8/generated/graph.json \
       --out   examples/mlp_control/int8/generated/graph.fused.json
   ```
   The CLI prints `wrote <out> (<N> fused ops, <M> total dispatches; input had <K>)`.
   Re-run per network listed in the hint (e.g. `mlp_control`,
   `dronet`, `yolov8_nano` if all three are present).

3. **Verify shape** by inspecting the fused op's
   `inputs` / `outputs` / `internal_tensors` / `fused_from`:
   ```bash
   jq '.ops[] | select(.fused_from != null) |
       {dispatch_id, op, inputs, outputs, internal_tensors, fused_from}' \
       examples/mlp_control/int8/generated/graph.fused.json
   ```
   - `internal_tensors` are the per-stage outputs that live entirely
     inside the chain (stack-local in the eventual fused kernel).
   - `outputs` must include any tensor consumed by an op outside the
     group OR named as the model's output.

4. **Next steps** (handled by downstream pipeline stages):
   - `python3 -m pipeline.emit_dispatch_graph` re-emits the fused
     dispatch graph that XPU-RT will see as `K-J+1` dispatches
     instead of `K`.
   - Codegen (`generate_kernels.py` / `generate_skeleton.py`)
     special-cases `__fused__` ops to emit a chained sub-kernel
     wrapper. (Integration in progress — see plan doc.)

## Rules

- **Linear chains only** in this pass. The validator rejects
  fuse_groups whose ops have circular or out-of-order dependencies.
  If XPU-RT proposes a branchy group, surface the
  `FusionHintError` to the user rather than silently dropping it.
- **Disjoint groups only.** One op cannot appear in two fuse_groups
  — the validator enforces this.
- **Do not edit the source `graph.json` in place.** Always write
  `graph.fused.json` alongside it so the original IR survives for
  diff and for axis-A/B candidates that don't fuse.
- **Mention the contract version** (`modelblaster.fusion_hints/v1`)
  when reporting failures — a mismatch is XPU-RT being on a newer
  schema and the right fix is to bump this side, not silently
  proceed.
