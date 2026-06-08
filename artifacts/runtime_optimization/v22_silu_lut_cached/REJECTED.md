# v22 — silu_s8 LUT cache across calls — REJECTED

## Hypothesis

On yolov8 there are 54 silu calls. Each call rebuilds a 256-entry LUT
at ~20k cycles (256 × (expf + roundf + clip)). If consecutive calls
share `(scale_in, scale_out, activation_min, activation_max)`, a
kernel-local static cache could skip the rebuild and save up to ~1 M
rdcycles on the rvv_opu hart.

## Implementation

Added kernel-local static cache to
`examples/yolov8_nano/int8/cache/rvv_opu/rvv_opu_silu_s8_rvv_lut_gather.c`
(mirrored to dronet) — bit-exact: cache entries use the same scalar
math as the reference.

## Measured result

| Metric          | v20b      | v22       | delta        |
|:----------------|----------:|----------:|-------------:|
| gemmini wall    | 176,760 µs| 175,545 µs| -1,215 µs    |
| rvv_opu wall    | 183,446 µs| 182,226 µs| -1,220 µs    |
| yolov8 mtime    | 156,924   | 155,687   | -1,237 ticks |
| mlp err         | 0         | 0         | unchanged    |
| dronet err      | 72        | 72        | unchanged    |
| yolov8 err      | 22        | 22        | unchanged    |

Delta is within FireSim mtime noise floor (~±2 ms run-to-run).
Spot-check on the first 10 rvv_opu silu calls: rdcycle counts in v22
differ from v20b by ±0.5%. The cache invalidates on every call —
yolov8's per-layer dequant/requant scales rarely repeat between
consecutive silu invocations, so the cache never hits.

## Conclusion

Kernel optimization journey ends at v20b 183 ms wall. The remaining
gap to predicted ~70 ms lives in the runtime (sync, dep_wait, gemmini
cfg emission), not in kernel cycles. Reverted v22 changes:

```bash
git checkout -- examples/yolov8_nano/int8/cache/rvv_opu/rvv_opu_silu_s8_rvv_lut_gather.c \
                examples/dronet/int8/cache/rvv_opu/rvv_opu_silu_s8_rvv_lut_gather.c
```

Phase G plan items that survive past v22:
- G3 — PDB re-ingestion against the calibrated runtime (no FireSim
  needed, just `ingest_measured_cycles.py` against v20b's per-op
  rdcycle CSV).
- G4 — final confirmation FireSim run after G3 recalibration.
