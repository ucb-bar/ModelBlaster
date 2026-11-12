# maxpool2d_s8 (K3)

Median us, reps 21, bit-exact (max_abs_err 0) on every shape.

| shape | X100 old | X100 new | A100 old | A100 new |
|---|---|---|---|---|
| dronet maxpool1 C32 56x56 k3 s2 | 437 | 64.5 | 6297 | 219 |
| yolov8_nano l9.m1 C128 5x5 k5 s1 p2 | 275 (same-run; before_*.json said 455) | 40.5 | 634 (same-run; before_*.json said 1439) | 203 |

"new" = kernels/rvv_k3_a100/rvv_k3_a100_maxpool2d_s8_direct.c (X100 numbers: built with
--backend rvv_k3 --kernel-file, not yet promoted). Shared kernels/rvv fix (cap vl at 256 so
tmp[256] cannot overflow) is perf-neutral on both cores.

Technique: per (n,c,oh) vertical max over valid kh rows (unit-stride e8m8) into a -128-padded
scratch row, then horizontal window max with unit-stride loads; stride 2 loads 2*vl bytes and
de-interleaves with vnsrl (shift 0 / 8; kw pairs share one load). No strided loads (except a
generic SW>=3 fallback), no scalar boundary paths, no whole-register spills (objdump clean).
Extra synthetic shapes (pad, dilation, k2 s2, long rows IW 900/1200) also bit-exact.
