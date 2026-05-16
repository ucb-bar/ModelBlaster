# Mixed-Precision Support — Architecture & Implementation Plan

Status: planned, phases 1–5 to implement.
Motivation: ViNT's int8 path hits ~0.97 cos_sim on the goal-encoder
output (`linear`, |max|=181). The wide dynamic range forces a coarse
per-tensor scale that wastes resolution on small typical values and
that error propagates into `linear_24` (final waypoints, cos_sim
0.986 vs 0.993 for full-fp16). Percentile clipping made things worse
(clipping the wide-range signal — turns out it's not outliers, it's
real signal). The cleanest accuracy win is to keep that **one op** in
fp16 while the rest of the network stays int8 — but the pipeline has
no way to express that today.

## Design principles

1. **Annotation-driven.** Per-model precision spec, parallel to
   `get_calibration_spec()`. No automated heuristics in phase 1 —
   model authors and PTQ tuning decide which ops to promote.
2. **IR-baked.** Dtype transitions materialize as explicit `cast` ops
   in the IR, not implicit kernel behavior. Runtime stays simple.
3. **Backwards compatible.** No spec ⇒ existing all-int8 or all-fp16
   behavior, byte-identical IR.
4. **Per-op granularity, region-level ergonomics.** Spec accepts
   explicit names + glob patterns + structural tags (regions —
   future).

## IR additions

### Op record
```jsonc
{
  "name": "linear",
  "op": "linear_s8_pc",
  "precision": "int8",   // NEW — explicit so consumers don't infer from suffix
  ...
}
```

### New op kinds
* `cast_i8_to_f16` — int8 input + per-tensor `scale_in` → fp16 output
  (`out = (float)in * scale_in`, cast to `_Float16`).
* `cast_f16_to_i8` — fp16 input + per-tensor `scale_out` (asymmetric
  zero-point optional) → int8 output (`q = round(in * inv_scale_out)`
  clamped to `[-128, 127]`).

### Tensor meta
Each tensor still has one `dtype` — whichever the producer emits.
Consumers expecting a different dtype read through an inserted cast.

## Per-model precision spec

In `agents/models/<name>.py`:

```python
def get_precision_spec() -> dict:
    return {
        "default": "int8",       # walker baseline
        "fp16_ops": [
            "linear",            # explicit names — aten node names
            "linear_22",
            "linear_23",
            "linear_24",
        ],
        "fp16_patterns": [       # fnmatch globs on aten node names
            "layer_norm_*",
        ],
        # "fp16_regions": [...]  # phase 6 (structural tags)
    }
```

CLI override (additive): `--fp16-ops linear,linear_24` /
`--fp16-patterns 'layer_norm_*'`. Without a spec, mixed precision is
off and `--quant` decides the global mode (current behavior).

## Walker changes

### Constructor
```python
def __init__(self, ..., default_quant="int8",
             op_precision: dict[str, str] | None = None):
    self.default_quant = default_quant
    self.op_precision = op_precision or {}     # node_name → "int8" | "fp16"
```

### Helper
```python
def _quant_for(self, n) -> str:
    return self.op_precision.get(n.name, self.default_quant)
```

Drop `self.quant`, `self.op_suffix`, `self.tensor_dtype` as walker-
wide invariants. Each `_emit_*` method picks its own precision /
suffix / dtype from `_quant_for(n)`.

`_record_tensor` derives dtype per op output, not from walker state.

## Auto-cast pass

After all walker emits, run a one-shot IR pass that examines each
op's input tensors and inserts cast ops when producer dtype !=
consumer expected dtype:

```python
def _insert_casts(walker):
    new_ops = []
    cast_intermediates = {}    # (src, dst_dtype) → intermediate name
    for op in walker.ops:
        consumer_dtype = "f16" if op["precision"] == "fp16" else "i8"
        for i, in_name in enumerate(op["inputs"]):
            producer_dtype = walker.tensors_meta[in_name]["dtype"]
            if producer_dtype == consumer_dtype:
                continue
            cast_out = cast_intermediates.get((in_name, consumer_dtype))
            if cast_out is None:
                cast_out = f"{in_name}__cast_{consumer_dtype}"
                new_ops.append(_make_cast_op(in_name, cast_out,
                                              producer_dtype, consumer_dtype,
                                              walker))
                walker.tensors_meta[cast_out] = {
                    "shape": walker.tensors_meta[in_name]["shape"],
                    "dtype": consumer_dtype,
                }
                cast_intermediates[(in_name, consumer_dtype)] = cast_out
            op["inputs"][i] = cast_out
        new_ops.append(op)
    walker.ops = new_ops
```

Mixed precision becomes a property of the **assembled IR**, not the
walker's per-op logic — cleaner because per-emit code stays
single-precision-per-op.

For calibration: `cast_f16_to_i8`'s `scale_out` comes from
`walker.scales[src_name]` (already calibrated from the fp32
reference). `cast_i8_to_f16`'s `scale_in` is the same scale (just
dequantizing).

## Cast kernels (reference impls)

```c
void kernel_cast_i8_to_f16(const int8_t *in, _Float16 *out,
                           int n, float scale) {
    for (int i = 0; i < n; i++)
        out[i] = (_Float16)((float)in[i] * scale);
}

void kernel_cast_f16_to_i8(const _Float16 *in, int8_t *out,
                           int n, float inv_scale) {
    for (int i = 0; i < n; i++) {
        float v = (float)in[i] * inv_scale;
        int32_t q = (int32_t)(v >= 0 ? v + 0.5f : v - 0.5f);
        if (q > 127) q = 127; if (q < -128) q = -128;
        out[i] = (int8_t)q;
    }
}
```

Codegen handlers in `generate_skeleton.py` are trivial — single-input,
single-output, two scalar params.

## Codegen changes

* Drop the `len(out_dtypes) != 1: raise NotImplementedError` check in
  `emit_model`. Casts at the surface handle mixed-dtype surface
  outputs cleanly.
* Buffer allocation is already per-tensor dtype — works as-is.
* Existing per-op codegen handlers don't change — they consume
  tensors at their declared dtype.

## Phased rollout

### Phase 1 — foundation refactor (no behavior change)
* Per-op precision storage in walker; default → walker-wide `--quant`
  (current behavior preserved).
* Per-tensor dtype derived from producing op.
* Test: extract ViNT int8 + fp16, ensure byte-identical IR vs
  current.

### Phase 2 — cast kernels + codegen handlers
* `cast_i8_to_f16` / `cast_f16_to_i8` KernelSpecs + ref impls.
* Codegen handlers via the standard handler-table.
* Smoke-test via the kernel host-verify path.

### Phase 3 — auto-cast IR pass
* Implement `_insert_casts` and wire into the extract flow.
* Drop generate_skeleton's mixed-dtype-output check.
* Test: ViNT with one op manually promoted to fp16 (`linear`) — verify
  1 cast inserted before linear, 1 after.

### Phase 4 — user-facing API
* `get_precision_spec()` per-model hook in `agents/models/vint.py`.
* CLI: `--fp16-ops`, `--fp16-patterns` (additive to model spec).
* `_load_model` plumbs the spec into walker init.

### Phase 5 — ViNT validation
* Promote `linear` (goal_encoder out) to fp16. Predict linear_24
  cos_sim 0.986 → ~0.995.
* Try wider islands (transformer, output head) and report a small
  table of promotions × accuracy gain × extra fp16 cycles.

### Phase 6 (future) — auto-promote heuristic
A pass that suggests fp16 promotions based on:
  (a) wide dynamic range relative to median activation magnitude,
  (b) zero-mean tensors (post-LayerNorm),
  (c) tensors whose downstream cos_sim is observed to be low.

## Open design decisions

1. **`--quant` semantics with mixed precision.** Keep `int8` / `fp16`
   as the **default** for ops without an explicit override. A
   precision spec only carves out exceptions. Result: simple cases
   stay one-flag.
2. **Asymmetric (zero-point) quant for casts of post-LayerNorm
   tensors.** Probably worth doing — symmetric int8 wastes ~half the
   range on the zero-mean bulge. Defer to phase 5 measurement.
3. **Mixed precision *within* a kernel** (e.g. fp16 weights + int8
   activations). Out of scope here — that's an internal kernel
   implementation choice; IR keeps one dtype per tensor.

## Estimated effort

* Phases 1+2+3+4: 1.5–2 days
* Phase 5: short, bounded by 2× spike runs (~30 min each)

## Expected wins

Based on the int8 drift map (see `vint_int8_drift_diagnosis` in
chat / inspect_intermediates results):

| Promotion | Δ cos_sim on linear_24 (predicted) | Cost |
|---|---|---|
| `linear` (goal enc out) | 0.986 → 0.995 | 1 linear, ~260K MACs |
| `linear_22..linear_24` (output head) | + small | 3 tiny linears |
| Transformer layer_norms (×8) | + moderate | 8 layer_norms |
| Full SDPA (×4) | up to fp16-ceiling (0.993) | ~25% of cycles |
