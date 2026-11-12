# conv2d_batchnorm2d_silu_s8: K3 A100 IME (vmadot 8x16x8) vs rvv

New kernel: `kernels/ime_k3_a100/ime_k3_a100_conv2d_batchnorm2d_silu_s8_ime_vmadot_8x16x8.c` (shared code in `kernels/ime_k3_a100/ime_k3_a100_common.h`).
Baseline: `kernels/rvv/rvv_conv2d_batchnorm2d_silu_s8_rvv_oc_blocked_bn_silu_epilogue.c` built with the rvv_k3_a100 flags (zvl1024b), run on CPU 8 (A100). X100 column: the same rvv kernel on CPU 0 (rvv_k3, zvl256b).

Median us over 7 reps, steady state (weight pack and BN/SiLU table caches warm; the first call per layer pays ~0.1-0.4 ms to build them).
`max_abs_err` is 0 on every shape, both the full-range and the small-value regime. Raw data: `after_ime_cpu8.json`, `before_rvv_k3_cpu0_full.json`.

| shape | A100 rvv (us) | A100 IME (us) | speedup | X100 rvv (us) | err |
|---|---|---|---|---|---|
| yolov8_nano:l0.conv N1 IC3 OC16 IH160 IW160 KH3 KW3 SH2 PH1 | 9836 | 654 | 15.0x | 1314 | 0 |
| yolov8_nano:l1.conv N1 IC16 OC32 IH80 IW80 KH3 KW3 SH2 PH1 | 8541 | 315 | 27.1x | 1483 | 0 |
| yolov8_nano:l2.cv1.conv N1 IC32 OC32 IH40 IW40 KH1 KW1 SH1 PH0 | 4401 | 166 | 26.5x | 797 | 0 |
| yolov8_nano:l2.m0.cv1.conv N1 IC16 OC16 IH40 IW40 KH3 KW3 SH1 PH1 | 6730 | 108 | 62.2x | 1264 | 0 |
| yolov8_nano:l2.m0.cv2.conv N1 IC16 OC16 IH40 IW40 KH3 KW3 SH1 PH1 | 6731 | 108 | 62.1x | 1265 | 0 |
| yolov8_nano:l2.cv2.conv N1 IC48 OC32 IH40 IW40 KH1 KW1 SH1 PH0 | 5253 | 204 | 25.7x | 963 | 0 |
| yolov8_nano:l3.conv N1 IC32 OC64 IH40 IW40 KH3 KW3 SH2 PH1 | 4717 | 184 | 25.6x | 1734 | 0 |
| yolov8_nano:l4.cv1.conv N1 IC64 OC64 IH20 IW20 KH1 KW1 SH1 PH0 | 2260 | 117 | 19.3x | 1037 | 0 |
| yolov8_nano:l4.m0.cv1.conv N1 IC32 OC32 IH20 IW20 KH3 KW3 SH1 PH1 | 3555 | 79 | 44.8x | 857 | 0 |
| yolov8_nano:l4.m0.cv2.conv N1 IC32 OC32 IH20 IW20 KH3 KW3 SH1 PH1 | 3555 | 79 | 45.1x | 856 | 0 |
| yolov8_nano:l4.m1.cv1.conv N1 IC32 OC32 IH20 IW20 KH3 KW3 SH1 PH1 | 3555 | 80 | 44.7x | 868 | 0 |
| yolov8_nano:l4.m1.cv2.conv N1 IC32 OC32 IH20 IW20 KH3 KW3 SH1 PH1 | 3556 | 79 | 44.7x | 860 | 0 |
| yolov8_nano:l4.cv2.conv N1 IC128 OC64 IH20 IW20 KH1 KW1 SH1 PH0 | 3241 | 171 | 18.9x | 1393 | 0 |
| yolov8_nano:l5.conv N1 IC64 OC128 IH20 IW20 KH3 KW3 SH2 PH1 | 2498 | 157 | 15.9x | 1797 | 0 |
| yolov8_nano:l6.cv1.conv N1 IC128 OC128 IH10 IW10 KH1 KW1 SH1 PH0 | 1140 | 89 | 12.8x | 819 | 0 |
| yolov8_nano:l6.m0.cv1.conv N1 IC64 OC64 IH10 IW10 KH3 KW3 SH1 PH1 | 2001 | 76 | 26.2x | 748 | 0 |
| yolov8_nano:l6.m0.cv2.conv N1 IC64 OC64 IH10 IW10 KH3 KW3 SH1 PH1 | 2010 | 76 | 26.3x | 766 | 0 |
| yolov8_nano:l6.m1.cv1.conv N1 IC64 OC64 IH10 IW10 KH3 KW3 SH1 PH1 | 2006 | 77 | 26.0x | 778 | 0 |
| yolov8_nano:l6.m1.cv2.conv N1 IC64 OC64 IH10 IW10 KH3 KW3 SH1 PH1 | 2001 | 76 | 26.3x | 749 | 0 |
| yolov8_nano:l6.cv2.conv N1 IC256 OC128 IH10 IW10 KH1 KW1 SH1 PH0 | 1753 | 135 | 13.0x | 1203 | 0 |
| yolov8_nano:l7.conv N1 IC128 OC256 IH10 IW10 KH3 KW3 SH2 PH1 | 2181 | 155 | 14.1x | 1909 | 0 |
| yolov8_nano:l8.cv1.conv N1 IC256 OC256 IH5 IW5 KH1 KW1 SH1 PH0 | 833 | 89 | 9.4x | 659 | 0 |
| yolov8_nano:l8.m0.cv1.conv N1 IC128 OC128 IH5 IW5 KH3 KW3 SH1 PH1 | 1100 | 72 | 15.3x | 719 | 0 |
| yolov8_nano:l8.m0.cv2.conv N1 IC128 OC128 IH5 IW5 KH3 KW3 SH1 PH1 | 1099 | 72 | 15.2x | 706 | 0 |
| yolov8_nano:l8.cv2.conv N1 IC384 OC256 IH5 IW5 KH1 KW1 SH1 PH0 | 1087 | 108 | 10.1x | 961 | 0 |
| yolov8_nano:l9.cv1.conv N1 IC256 OC128 IH5 IW5 KH1 KW1 SH1 PH0 | 428 | 55 | 7.8x | 298 | 0 |
| yolov8_nano:l9.cv2.conv N1 IC512 OC256 IH5 IW5 KH1 KW1 SH1 PH0 | 1340 | 126 | 10.6x | 1197 | 0 |
| yolov8_nano:l12.cv1.conv N1 IC384 OC128 IH10 IW10 KH1 KW1 SH1 PH0 | 2336 | 178 | 13.1x | 1761 | 0 |
| yolov8_nano:l12.m0.cv1.conv N1 IC64 OC64 IH10 IW10 KH3 KW3 SH1 PH1 | 2001 | 77 | 26.0x | 754 | 0 |
| yolov8_nano:l12.m0.cv2.conv N1 IC64 OC64 IH10 IW10 KH3 KW3 SH1 PH1 | 2008 | 76 | 26.2x | 766 | 0 |
| yolov8_nano:l12.cv2.conv N1 IC192 OC128 IH10 IW10 KH1 KW1 SH1 PH0 | 1425 | 112 | 12.7x | 999 | 0 |
| yolov8_nano:l15.cv1.conv N1 IC192 OC64 IH20 IW20 KH1 KW1 SH1 PH0 | 4267 | 225 | 18.9x | 1792 | 0 |
| yolov8_nano:l15.m0.cv1.conv N1 IC32 OC32 IH20 IW20 KH3 KW3 SH1 PH1 | 3550 | 80 | 44.4x | 867 | 0 |
| yolov8_nano:l15.m0.cv2.conv N1 IC32 OC32 IH20 IW20 KH3 KW3 SH1 PH1 | 3554 | 80 | 44.5x | 858 | 0 |
| yolov8_nano:l15.cv2.conv N1 IC96 OC64 IH20 IW20 KH1 KW1 SH1 PH0 | 2743 | 144 | 19.0x | 1208 | 0 |
| yolov8_nano:l16.conv N1 IC64 OC64 IH20 IW20 KH3 KW3 SH2 PH1 | 2038 | 105 | 19.4x | 797 | 0 |
| yolov8_nano:l18.cv1.conv N1 IC192 OC128 IH10 IW10 KH1 KW1 SH1 PH0 | 1425 | 112 | 12.7x | 1000 | 0 |
| yolov8_nano:l18.m0.cv1.conv N1 IC64 OC64 IH10 IW10 KH3 KW3 SH1 PH1 | 2001 | 77 | 26.0x | 748 | 0 |
| yolov8_nano:l18.m0.cv2.conv N1 IC64 OC64 IH10 IW10 KH3 KW3 SH1 PH1 | 2008 | 77 | 26.0x | 760 | 0 |
| yolov8_nano:l18.cv2.conv N1 IC192 OC128 IH10 IW10 KH1 KW1 SH1 PH0 | 1425 | 112 | 12.7x | 1000 | 0 |
| yolov8_nano:l19.conv N1 IC128 OC128 IH10 IW10 KH3 KW3 SH2 PH1 | 1096 | 96 | 11.4x | 776 | 0 |
| yolov8_nano:l21.cv1.conv N1 IC384 OC256 IH5 IW5 KH1 KW1 SH1 PH0 | 1087 | 108 | 10.0x | 963 | 0 |
| yolov8_nano:l21.m0.cv1.conv N1 IC128 OC128 IH5 IW5 KH3 KW3 SH1 PH1 | 1099 | 73 | 15.1x | 719 | 0 |
| yolov8_nano:l21.m0.cv2.conv N1 IC128 OC128 IH5 IW5 KH3 KW3 SH1 PH1 | 1099 | 73 | 15.1x | 716 | 0 |
| yolov8_nano:l21.cv2.conv N1 IC384 OC256 IH5 IW5 KH1 KW1 SH1 PH0 | 1087 | 108 | 10.1x | 962 | 0 |
| yolov8_nano:detect.cv2_0_0.conv N1 IC64 OC64 IH20 IW20 KH3 KW3 SH1 PH1 | 8053 | 196 | 41.0x | 2777 | 0 |
| yolov8_nano:detect.cv2_0_1.conv N1 IC64 OC64 IH20 IW20 KH3 KW3 SH1 PH1 | 8069 | 196 | 41.1x | 2819 | 0 |
| yolov8_nano:detect.cv3_0_0.conv N1 IC64 OC80 IH20 IW20 KH3 KW3 SH1 PH1 | 8566 | 239 | 35.8x | 4197 | 0 |
| yolov8_nano:detect.cv3_0_1.conv N1 IC80 OC80 IH20 IW20 KH3 KW3 SH1 PH1 | 10324 | 375 | 27.5x | 5354 | 0 |
| yolov8_nano:detect.cv2_1_0.conv N1 IC128 OC64 IH10 IW10 KH3 KW3 SH1 PH1 | 3696 | 208 | 17.8x | 1283 | 0 |
| yolov8_nano:detect.cv2_1_1.conv N1 IC64 OC64 IH10 IW10 KH3 KW3 SH1 PH1 | 2007 | 158 | 12.7x | 757 | 0 |
| yolov8_nano:detect.cv3_1_0.conv N1 IC128 OC80 IH10 IW10 KH3 KW3 SH1 PH1 | 3876 | 249 | 15.6x | 1837 | 0 |
| yolov8_nano:detect.cv3_1_1.conv N1 IC80 OC80 IH10 IW10 KH3 KW3 SH1 PH1 | 2576 | 206 | 12.5x | 1287 | 0 |
| yolov8_nano:detect.cv2_2_0.conv N1 IC256 OC64 IH5 IW5 KH3 KW3 SH1 PH1 | 1729 | 153 | 11.3x | 568 | 0 |
| yolov8_nano:detect.cv2_2_1.conv N1 IC64 OC64 IH5 IW5 KH3 KW3 SH1 PH1 | 508 | 110 | 4.6x | 198 | 0 |
| yolov8_nano:detect.cv3_2_0.conv N1 IC256 OC80 IH5 IW5 KH3 KW3 SH1 PH1 | 1827 | 182 | 10.0x | 844 | 0 |
| yolov8_nano:detect.cv3_2_1.conv N1 IC80 OC80 IH5 IW5 KH3 KW3 SH1 PH1 | 652 | 135 | 4.8x | 334 | 0 |
| **total** | **175537** | **8030** | **21.9x** | **67706** | |

## Notes
- No shape falls back to RVV: IME wins every real shape (smallest win 2.0x, dronet conv_modules.9 1x1 stride 2; ~4.3x on the 5x5 detect convs).
- In-kernel scalar fallback (never hit by the real graphs) when vlenb != 128, input_offset != 0, filter_offset != 0 or SW not in {1,2}.
- Design and the A100 micro-costs that shaped it: header comment of `ime_k3_a100_common.h` and `artifacts/k3_probes/ime/layout_cpu8.txt` (probe `scripts/k3_ime_layout_probe.c`).
- Edge-shape sweep (odd OW, OC % 8 != 0, IC % 16 != 0, N=2, 5x5/7x7/2x2/3x1 kernels, stride 2 with and without padding, OW > 128): all bit-exact.
- Where the time goes (IC80 OC80 20x20 3x3, ~300 us): NHWC16 transpose ~60 us, vmadot k-loop + epilogue ~170 us, BN/SiLU table pass ~40 us.
