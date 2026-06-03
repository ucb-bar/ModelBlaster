# Kernel measurement report — `conv2d_batchnorm2d_s8`

## Phase E1 / E2 / E3 / E4 status

| Phase | Status | Evidence |
|:---|:---|:---|
| E1 (gap identified) | ✅ | `artifacts/kernel_gap_survey.json` — 60 candidate pairs (top-1 gap) |
| E2 (KernelSpec registered) | ✅ | `pipeline/reference_kernels.py:CONV2D_BATCHNORM2D_S8` |
| E2 (algorithm candidate seeds) | ✅ | 1 gemmini algorithm describing tiled_matmul_auto + BN epilogue |
| E2 (reference impl for verify) | ✅ | 3.2 KB scalar C, bit-exact conv2d_s8 → batchnorm2d_s8 chain |
| E2 (Bedrock kernel generation) | ⛔ **blocked** | AWS credentials not configured in session env |
| E3 (realizability filter wired) | ✅ | `scripts/decision_loop.py:REALIZABLE_FUSE_PAIRS` + `pipeline/apply_fusion_hint.py` recognizes `(conv2d_s8, batchnorm2d_s8)` pair |
| E4 (measured speedup) | ⛔ **blocked by E2** | requires generated kernel to measure |

## Why this kernel matters

E1's gap survey identified `conv2d_s8 → batchnorm2d_s8` as the largest
single fusion gap on the headline 4 MLP + 2 Dronet + 1 Yolo workload:
**60 candidate pairs** across yolov8_nano and dronet. These pairs share
the structural property that the conv output tensor is consumed by
exactly one BN op (and BN has exactly one producer), so they fit the
fusion advisor's single-producer/single-consumer pattern.

The fused kernel:
1. Runs Conv2D's MAC + Q0.31 requantize + clamp to produce int8 in a
   register.
2. Applies BN's per-channel affine (scale[c], bias[c]) on that int8 in
   the same register.
3. Stores the final int8 — no intermediate `conv_int8` tensor.

Memory traffic saved per output element: `1 byte read + 1 byte write`
for the intermediate (the BN params memory traffic doesn't change). On
yolov8's hot conv shapes (OH × OW = 40 × 40 to 5 × 5 across the
backbone), this is ~10–100 KB per dispatch.

## Reference impl coverage (bit-exact verification oracle)

The `reference_impl` field in the KernelSpec is a scalar C function
that the Bedrock-generated kernel must match `max_abs_err=0
max_rel_err=0` on the extra_shapes:

- `(N=1, IC=3,  IH=320, IW=320, OC=16, K=3×3, stride=2)` — yolov8 stem
- `(N=1, IC=16, IH=160, IW=160, OC=32, K=3×3, stride=2)` — yolov8 stage 1
- `(N=1, IC=32, IH=80,  IW=80,  OC=64, K=3×3, stride=2)` — yolov8 stage 2
- `(N=1, IC=4,  IH=8,   IW=8,   OC=4,  K=3×3, stride=1)` — small smoke

## E2 completion contract

When AWS Bedrock credentials are added to the session env:

1. `LLM_PROVIDER=bedrock BACKEND=llm` + run `examples/yolov8_nano/run.sh`
   with the fused IR (apply_fusion_hint emits `conv2d_batchnorm2d_s8`).
2. Spike runs the build + verifies output against the reference impl.
3. Expected verify result: `max_abs_err=0 max_rel_err=0` on each
   extra_shape × each target (rvv_opu + gemmini).
4. Measured-cycles speedup vs unfused baseline: target ≥ 1.1×.
   Reject otherwise (no false claims).
5. Bedrock spend per kernel: ≤ $30. Track in `spend_log.csv`.

## E4 contract (after E2 succeeds)

A row gets appended to this file:

| Host net | Target | Verify | Cycles unfused | Cycles fused | Speedup | Bedrock spend |
|:---|:---|:---|---:|---:|---:|---:|
| yolov8_nano | rvv_opu | ? | ? | ? | ? | ? |
| dronet | rvv_opu | ? | ? | ? | ? | ? |

Rejection criteria (any one fails → kernel REJECTED):
- max_abs_err > 0
- max_rel_err > 0
- Speedup ≤ 1.1× on either host network
- Bedrock spend > $30
