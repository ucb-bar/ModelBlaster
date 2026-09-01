# v13: LUT batchnorm picked up successfully but REGRESSED

Result vs v11g:
  mlp_control: 77,316 -> 83,806   (+8.4%)
  dronet:     145,168 -> 153,964   (+6.1%)
  yolov8_nano: 383,653 -> 444,848 (+16.0%)
  rvv_opu wall: 531,250 -> 601,276 (+13.2%)
  batchnorm2d_s8 rvv_opu sum (60 calls): 73 M -> 121 M rdcycle (+48 M)

Root cause: LUT-build per channel is ~256 iters of (cast/mul/FMA/div/
roundf/clamps). On yolov8's small-spatial layers (5x5 = 25 px, 10x10 =
100 px), the LUT build is 3-12x larger than the work it amortizes.

Fix landed in the kernel: guard `if (spatial >= LUT_BREAKEVEN)` — fall
through to reference inner for small-spatial layers. Re-tested
bit-exact across all 7 shapes. Will measure in v14.

KEY LESSON: spike-microbench speedup numbers don't generalize. The
silu LUT amortized because silu's reference uses expf() (very expensive,
hundreds of cycles). batchnorm's reference uses div+roundf (much
cheaper) so the LUT's amortization point is much higher.
