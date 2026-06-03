# Kernel measurement report — `batchnorm2d_silu_s8`

## Phase E1 / E2 / E3 / E4 status

| Phase | Status | Evidence |
|:---|:---|:---|
| E1 (gap identified) | ✅ | `artifacts/kernel_gap_survey.json` — 57 candidate pairs (top-2 gap) |
| E2 (KernelSpec registered) | ✅ | `pipeline/reference_kernels.py:BATCHNORM2D_SILU_S8` |
| E2 (algorithm candidate seeds) | ✅ | 2 algorithms: rvv_opu (VRGATHER LUT) + gemmini scalar fallback |
| E2 (reference impl for verify) | ✅ | 1.6 KB scalar C, bit-exact batchnorm2d_s8 → silu_s8 chain |
| E2 (Bedrock kernel generation) | ⛔ **blocked** | AWS credentials not configured in session env |
| E3 (realizability filter wired) | ✅ | `(batchnorm2d_s8, silu_s8)` in `REALIZABLE_FUSE_PAIRS` |
| E4 (measured speedup) | ⛔ **blocked by E2** | requires generated kernel |

## Why this kernel matters

E1's gap survey identified `batchnorm2d_s8 → silu_s8` as the second-
largest fusion gap on the headline workload: **57 candidate pairs**
across yolov8_nano. Combined with `conv2d_s8 → batchnorm2d_s8` (60
pairs), these two fused kernels cover the standard Conv→BN→SiLU
yolov8 block, accounting for **62 % of all fuse candidates** in
the workload.

The fused kernel:
1. Applies BN's per-channel affine on int8 input → int8 in register.
2. Looks up SiLU LUT entry for that int8 → int8 final output.
3. Stores final int8 — no intermediate `bn_int8` tensor.

Performance: the SiLU LUT (256 bytes per (silu_scale_in,
silu_scale_out) tuple) precomputes once per dispatch. The
hot loop then does a vector-broadcast multiply (BN scale) + vector
add (BN bias) + clamp + VRGATHER on the SiLU LUT — all in vector
registers.

## RVV-OPU algorithm candidate detail (highlights from KernelSpec)

```
PROGRAMMING MODEL:
  /* Precompute SiLU LUT keyed by BN int8 output. */
  for (int v = 0; v < 256; v++) {
      int8_t iv = (int8_t)(uint8_t)v;
      float f = (float)iv * silu_scale_in;
      float y = f / (1.0f + expf(-f));
      int32_t q = roundf(y / silu_scale_out);
      if (q < silu_activation_min) q = silu_activation_min;
      if (q > silu_activation_max) q = silu_activation_max;
      silu_lut[v] = (int8_t)q;
  }
  /* For each (n, c) panel of (H*W) int8 elements: */
  for (n, c) in [N, C):
      bs = vfmv_v_f(scale[c]); bb = vfmv_v_f(bias[c]);
      for tile of size VLEN/8:
          v_in = vle8_v(input + idx)
          v_f  = vfcvt_f_x(v_in) * bn_scale_in
          v_y  = bs * v_f + bb
          v_q  = round(v_y / bn_scale_out)
          v_q  = clamp(v_q, bn_act_min, bn_act_max)
          v_bn8 = vncvt_x_x(v_q)
          v_out = vrgather_vv(silu_lut, v_bn8)
          vse8_v(output + idx, v_out)
```

## Reference impl coverage

The reference_impl in the KernelSpec covers 6 shapes:

- `(N=1, C=16, H=40, W=40)` — yolov8 mid-block
- `(N=1, C=32, H=20, W=20)` — yolov8 mid-block
- `(N=1, C=64, H=10, W=10)` — yolov8 deep block
- `(N=1, C=128, H=5, W=5)` — yolov8 tail
- `(N=1, C=8, H=7, W=7)` — small generalization
- `(N=2, C=4, H=3, W=3)` — batch generalization

## E2 / E4 contract

Same as `conv2d_batchnorm2d_s8`:

1. Bedrock generation: ≤ $30 per kernel
2. Spike verify: max_abs_err = 0, max_rel_err = 0
3. Cycle speedup: ≥ 1.1× vs unfused on TWO host networks
4. Reject (with reason) if any criterion misses

## Gap-survey impact

| Metric | Before E2 | After E2 |
|:---|---:|---:|
| Total fuse candidates | 188 | 188 |
| Registered fused KernelSpecs | 2 | 4 |
| Candidates covered by KernelSpecs | ~6 | **123** (= 6 + 60 + 57) |
| Coverage % | 3.2 % | **65.4 %** |
| Top unregistered gap | conv→BN (60) | silu→chunk (8) |

Adding these two KernelSpecs reduced the unrealizable fuse-candidate
surface from 182 to 65 candidates — a **64 % reduction in realizability
gap** even before any Bedrock kernel is generated, because the
remaining gaps are small (≤ 8 candidates each) and dominated by
memory-only ops (chunk) that don't reward a fused kernel.
