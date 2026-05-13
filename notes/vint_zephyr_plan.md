# Plan — ViNT on the Zephyr/agents flow

Goal: take the same ViNT checkpoint we validated in IsaacLab
(`sims/scripts/pilot/pilot_forest_with_vint.py`) and run it under the
agents pipeline on FireSim Saturn-Gemmini-Q31, eventually scheduled
alongside the existing dronet/yolov8/mlp_control workloads.

## 0. What ViNT actually is

Pulled from `sims/external/visualnav-transformer/train/config/vint.yaml`
and the pilot script:

* **Inputs**
  - `obs`: 6 FPV frames (context_size+1=5+1), each 3×H×W (image_size = 85×64 W×H),
    stacked along channels → tensor `(1, 18, 64, 85)`.
  - `goal`: 1 image, same size → `(1, 3, 64, 85)`.
* **Backbone**: 7× EfficientNet-B0 (shared weights) → 512-d obs/goal tokens.
* **Decoder**: cross-attention transformer, 4 attention heads × 4 layers,
  ff_dim_factor=4 → 2048-d FFN.
* **Head**: linear → `(len_traj_pred=5) × 4` action coords (dx, dy, sin θ, cos θ)
  + 1 temporal-distance scalar.
* **Forward pass**: ~5 GMAC fp32 / inference. Already exported to ONNX
  and int8-PTQ'd by `sims/scripts/utils/quantize_vint.py`:
  - `vint_fp32.onnx` — 96 MB
  - `vint_int8.onnx` — 26 MB (QDQ, per-tensor sym act / per-channel sym weight)

So we already have a bit-exact int8 golden and the calibration pipeline.

## 1. Missing ops in our pipeline today

`agents/pipeline/extract_graph.py` + `reference_kernels.py` +
`verify_kernel.py` cover the dronet/yolov8/mlp_control op set. ViNT adds:

| op | source | first appears in | priority | notes |
|---|---|---|---|---|
| `depthwise_conv2d_s8` | EfficientNet MBConv blocks | every stage | **P0** | not a degenerate `conv2d_s8(groups=IC)` — needs its own kernel for both correctness and speed (gemmini systolic is poorly matched here; RVV wins) |
| `global_avg_pool2d_s8` | SE block, post-backbone reduction | every MBConv with `se_ratio > 0` | **P0** | one-shot reduction over H×W, fast on RVV |
| `sigmoid_s8` | SE gate, attention softmax-of-zero edge | SE blocks | **P0** | we have `silu_s8`; sigmoid is `1/(1+exp(-x))` LUT |
| `mul_s8` (elementwise) | SE gating + attention weighting | SE blocks, attention | **P0** | per-element s8×s8 → s8 with rescale; reuse `add_s8` quant plumbing |
| `matmul_s8` | scaled dot-product attention (Q·Kᵀ, ·V) | transformer decoder | **P0** | bigger M/K/N than linear_s8; on gemmini_q31 use `tiled_matmul_auto`, on RVV use blocked-K tiled kernel |
| `softmax_s8` | attention | transformer decoder | **P0** | numerically-stable subtract-max + exp + normalize; LUT for exp |
| `layernorm_s8` | transformer decoder | every transformer layer | **P1** | row-reduce mean + var, then scale-shift; RVV-friendly |
| `gelu_s8` or `swish_s8` | EfficientNet uses Swish/SiLU (already have); transformer uses GELU | transformer FFN | **P1** | LUT |
| `cat_seq_s8` (cat over a 7-token sequence dim) | obs+goal token sequence builder | between backbone and decoder | **P1** | we already have `cat{2,3,4}_c1_s8`; extend to 7, or generalize to `cat_n_dim_s8` |
| `pos_embed_add_s8` | learned positional embedding | transformer input | **P2** | could be folded into the cat seq output as a static add |
| `unsqueeze_s8` / `reshape_s8` | layout shuffles between backbone and transformer | various | **P2** | zero-cost in extract_graph terms (alias) — but the dispatch table needs to know about them to keep dispatch_ids contiguous; we already filter zero-cost in codegen, just need to plumb the cases |

P0 = required for correctness, P1 = required for performance baseline,
P2 = nice-to-have.

## 2. Phased delivery

### Phase A — Get the model into the agents IR (no kernel work yet)

**Source-of-truth decision.** The published `vint_int8.onnx` is QDQ:
int8 values flow between ops, but inside each op the compute typically
re-dequantizes to fp32 and re-quantizes. That's a useful correctness
oracle but **not** our deployment target — we want true integer-only
compute (int8 in/out, int32 accumulator, Q0.31 requantize at the
boundary), same shape as the dronet / yolov8 / mlp_control int8 flows.
So we quantize ViNT ourselves in PyTorch and run it through our
existing `extract_int8` walker.

Use the QDQ ONNX as a cross-check oracle (per-op scales should agree
within rounding), not as the input.

Steps:

1. **PyTorch PTQ for ViNT** — new
   `sims/scripts/utils/quantize_vint_torch.py`. Three op classes:
   - **Conv / Linear / Pointwise / Depthwise (EfficientNet body)**
     — `torch.ao.quantization` eager-mode (or FX graph mode). Per-tensor
     symmetric activations, per-channel symmetric weights — same recipe
     as dronet / yolov8.
   - **MatMul (Q·Kᵀ, ·V in attention)** — `torch.ao.quantization`
     doesn't cleanly cover bare `torch.matmul`. Either rewrap the
     attention layer so the matmuls flow through a wrapped
     `nn.Linear` (cleanest, model-source patch in
     `sims/external/visualnav-transformer/.../self_attention.py`) or
     attach a forward-hook observer at each `matmul` site and capture
     input/output scales manually.
   - **LayerNorm / Softmax / Sigmoid / GELU** — torch AO leaves these
     in fp32. Capture per-tensor input/output scales with forward-hook
     observers during the calibration pass, then synthesize them into
     `module.input_scale` / `module.output_scale` attributes that
     `extract_int8` already reads. Same pattern we used when adding
     `silu_s8`.
2. **Calibration data.** Reuse the 200-frame IDSIA loop already in
   `quantize_vint.py` — same calibration set keeps our quantization
   apples-to-apples with the published ONNX (we can sanity-check
   per-tensor scales against the ONNX QDQ scales as a cross-check).
3. **Run `extract_int8` over the quantized PyTorch model.** Extend
   `extract_graph.py::extract_int8` to handle the new modules
   (Sigmoid / Softmax / LayerNorm / MatMul + EfficientNet's
   depthwise-conv pattern). Output: the same
   `(graph.json, weights.npz, io.npz)` triple every other model has.
4. **Op-coverage check.** Inspect the emitted `graph.json` and confirm
   it lists only the ops in §1's table. Anything else means we missed
   an extract case.
5. **(Optional) Bit-compat cross-check.** Run the quantized PyTorch
   model and `vint_int8.onnx` on the same input, compare waypoint MAE.
   Won't be bit-exact (different quant rounding, possibly different
   per-tensor scales) but should be small. Useful sanity step.

Exit criterion: `agents/examples/vint/int8/generated/scalar/graph.json`
exists, lists only the ops in §1, and a fresh `BACKEND=reference`
spike run matches PyTorch waypoint MAE within a small tolerance.

### Phase B — Reference kernels (spike bit-exact)

For each P0 op (depthwise conv, gap, sigmoid, mul, matmul, softmax):

1. Reference impl in `agents/pipeline/reference_kernels.py` (numpy
   ground truth + atol_class registration).
2. Verify input/output generators in `agents/pipeline/verify_kernel.py`.
3. Skeleton emitter snippet in `agents/pipeline/generate_skeleton.py`
   for the universal-direct scalar kernel.
4. Spike harness verify against onnxruntime golden, per op shape that
   actually appears in ViNT (extract the shape set from §A.3).

Exit criterion: `BACKEND=reference QUANT=int8 bash
agents/examples/vint/run.sh` produces a `PASS` on spike with
`max_abs_err ≤ atol_class`.

### Phase C — Curated kernels for FireSim performance

For each P0 op (and P1 layernorm), write at minimum:

| op | rvv kernel | gemmini_q31 kernel |
|---|---|---|
| depthwise_conv2d_s8 | rvv tiled OC=1, K×K | scalar fallback (gemmini poor fit) |
| global_avg_pool2d_s8 | rvv whole-tile reduce | scalar fallback |
| sigmoid_s8 | rvv vectorized LUT | scalar |
| mul_s8 | rvv pointwise | gemmini resadd-style |
| matmul_s8 | rvv blocked-K | tiled_matmul_auto full_C=true (bit-exact Q0.31) |
| softmax_s8 | rvv row-reduce + exp LUT | scalar |
| layernorm_s8 | rvv row-reduce + reciprocal LUT | scalar |

Tag each with `/* accuracy_class: bit_exact */` where possible (LUT ops
typically ≤1 LSB drift → `numeric_drift`).

Exit criterion: `TARGET=gemmini_q31` and `TARGET=rvv` runs both pass
`MAX_ACCURACY_CLASS=numeric_drift` verify on spike at every shape that
appears in ViNT.

### Phase D — Single-model harness on FireSim

1. Add `agents/examples/vint/` with `run.sh` (mirror dronet's pattern).
2. **Memory layout.** 26 MB int8 weights + worst-case activations
   (~6 MB for the 7-frame EfficientNet stack + 4 transformer layers)
   need `CONFIG_HEAP_MEM_POOL_SIZE` ≈ 40 MB. Saturn FireSim's DDR is
   256 MB, so this is fine, but bump from current 8 MB.
3. **Rolling 6-frame context.** The harness today runs single-shot
   per model. For ViNT we either:
   - (a) ingest pre-stacked 18-channel obs (host pre-stacks frames),
     keep the harness single-shot. Simpler; lets us reuse all the
     existing scaffolding. **Recommended first pass.**
   - (b) maintain in-binary rolling buffer that the runtime pops/pushes
     each iter. Adds harness complexity; revisit once (a) works.
4. Profile per-op cycles on FireSim Saturn-Gemmini-Q31 (same flow as
   dronet/yolov8 today: `results.csv` per backend+target).

Exit criterion: `RUNNER=firesim TARGET=gemmini_q31 QUANT=int8 bash
agents/examples/vint/run.sh` produces a clean trace + matches the
onnxruntime int8 golden within atol.

### Phase E — Multi-model + scheduler integration

1. Generate a workload spec
   `data/toplevel/networks_vint_dronet_mlp_firesim.json` with ViNT at
   1 Hz, dronet at 50 ms, mlp_control at 10 ms (or whatever the
   real-robot rates are).
2. Use the profile data from Phase D to feed the scheduler. Likely
   outcome: scheduler splits EfficientNet stages onto gemmini and
   transformer attention layers onto RVV. The cross-attention matmuls
   are the interesting placement decision — gemmini is bit-exact via
   `tiled_matmul_auto(full_C=true)` but startup cost dominates for
   small per-head dimensions; RVV wins below a threshold.
3. Run via `agents/examples/xpurt_demo/run.sh` with the new spec.
4. Compare against a fixed-pinning microros baseline (mirror the
   existing dronet/yolov8/mlp_control comparison in `ROS_FLOW.md`).

Exit criterion: clean xpurt run on FireSim, predicted-vs-actual plot,
end-to-end runtime + a meaningful speedup over the pinned baseline.

### Phase F — Close the loop with IsaacLab

1. Run the FireSim xpurt schedule's outputs through the pilot script as
   the policy source (`--vint_onnx <path to embedded inference output
   replay>`, or write a tiny socket adapter that pipes actions from a
   running FireSim instance into the env).
2. Verify the drone still navigates the forest trail with the FireSim
   ViNT in the loop — not pixel-identical to onnxruntime-on-host but
   close enough that the steering controller doesn't see degraded
   waypoints.

## 3. Decisions to make up-front

| decision | recommendation | rationale |
|---|---|---|
| Re-do PTQ via our pipeline or ingest the existing `vint_int8.onnx`? | **Re-do PTQ in PyTorch** (Phase A above). | The published ONNX is QDQ (int8-tested, fp32-compute inside each op), not integer-only. Our gemmini_q31 / RVV kernels are integer-only, so we need a quant flow that produces real int8 ops — same as the existing dronet/yolov8/mlp_control int8 path. The QDQ ONNX stays useful as a cross-check oracle. |
| Inputs from host (pre-stacked 18ch) or rolling on-target? | **Pre-stacked first; rolling later.** | Decouples model bring-up from the runtime state plumbing. The rolling buffer is a runtime feature, not a model feature. |
| EfficientNet on gemmini or RVV? | **Mostly RVV (depthwise-heavy).** | Gemmini wins on dense conv2d but loses on depthwise (no reduction → poor utilization). Pointwise (1×1) is fine on either. Let the scheduler pick. |
| matmul_s8 for attention on gemmini or RVV? | **Both, scheduler picks.** | gemmini_q31 has a bit-exact `tiled_matmul_auto(full_C=true)` path already proven on yolov8; competitive once M ≥ 32. RVV wins at small head_dim. |
| Cross-frame rolling buffer in `agents_pool` or in harness? | **Harness.** | This is application state, not a kernel concern. Keep `agents_pool` model-agnostic. |

## 4. Risks

1. **Heap fragmentation.** 26 MB of static weights + activation
   bookkeeping is bigger than anything we've run. Likely need to
   manually place weights in `.rodata` (already do for dronet) — but
   the depthwise conv's per-channel scale tables blow up the
   `.rodata` symbol count by 50–100× the dronet level. We may hit
   linker symbol-count limits before runtime issues.
2. **HTIF dump bandwidth.** A full per-dispatch trace for ViNT is ~5×
   the size of yolov8's. Already mitigated by the buffered HTIF
   `_write(1,…)` path landed today, but worth a measurement before
   sizing FIRESIM_TIMEOUT.
3. **Softmax / LayerNorm numerics.** Per-tensor PTQ scales on these
   ops produce visible drift even at numeric_drift class. Plan to
   loosen verify atol for these specific ops and validate end-to-end
   via waypoint MAE rather than per-op LSB.
4. **Saturn vlse strided-bug recurrence.** Depthwise conv2d wants
   strided loads of weights. We hit `vlse8` corruption on yolov8 (see
   `v512_yolov8_rvv_open` memory entry); workaround is scalar
   gather + `vle8`. Apply the same pattern to the depthwise kernel
   from day one.
5. **Schedule solver scale.** ViNT's IR is ~250–400 ops vs yolov8's
   200. The decomposed-MILP solver may need `time_limit` bumped from
   60 s and `restrict_makespan_to_nonperiodic` tuned. Start with the
   greedy_periodic solver; fall back to decomposed only if the
   greedy result has obvious slack.

## 5. Estimated effort

| phase | LOC | notes |
|---|---|---|
| A | ~800 (mostly extract_graph_onnx.py) | reuses verify_kernel + reference_kernels scaffolding |
| B | ~1200 across reference + verify + skeleton | 6 new ops |
| C | ~1500 across curated rvv + gemmini_q31 | depthwise + matmul are the bulk |
| D | ~200 (example dir + run.sh + heap bump) | mostly config |
| E | ~100 (workload spec + schedule run) | tooling exists |
| F | ~150 (replay adapter) | one-shot validation script |

About 1–2 weeks of focused work to Phase D PASS, another few days for
E + F.

## 6. Where each artifact will live

* `agents/examples/vint/` — example dir (run.sh, fp32/, int8/)
* `agents/pipeline/extract_graph_onnx.py` — new ONNX ingest
* `agents/pipeline/reference_kernels.py` — new ref impls (extend)
* `agents/pipeline/verify_kernel.py` — new input/output gens (extend)
* `agents/pipeline/generate_skeleton.py` — new op snippets (extend)
* `agents/kernels/rvv/rvv_{depthwise_conv2d,gap,sigmoid,mul,matmul,softmax,layernorm}_s8_direct.c`
* `agents/kernels/gemmini_q31/gemmini_q31_matmul_s8_*.c` (and any
  other gemmini-friendly ones)
* `data/toplevel/networks_vint_dronet_mlp_firesim.json` — Phase E spec
* `docs/end_to_end_xpurt_firesim.md` — add ViNT to the list of
  worked examples once Phase E lands
