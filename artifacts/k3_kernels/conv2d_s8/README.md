# conv2d_s8: K3 A100 IME (vmadot 8x16x8) vs rvv

New kernel: `kernels/ime_k3_a100/ime_k3_a100_conv2d_s8_ime_vmadot_8x16x8.c` (shared code in `kernels/ime_k3_a100/ime_k3_a100_common.h`).
Baseline: `kernels/rvv/rvv_conv2d_s8_rvv_vsmul_vnclip.c` built with the rvv_k3_a100 flags (zvl1024b), run on CPU 8 (A100). X100 column: the same rvv kernel on CPU 0 (rvv_k3, zvl256b).

Median us over 7 reps, steady state (weight pack and BN/SiLU table caches warm; the first call per layer pays ~0.1-0.4 ms to build them).
`max_abs_err` is 0 on every shape, both the full-range and the small-value regime. Raw data: `after_ime_cpu8.json`, `before_rvv_k3_cpu0_full.json`.

| shape | A100 rvv (us) | A100 IME (us) | speedup | X100 rvv (us) | err |
|---|---|---|---|---|---|
| dronet:conv_modules.0 N1 IC3 OC32 IH112 IW112 KH3 KW3 SH2 PH1 | 2330 | 337 | 6.9x | 570 | 0 |
| dronet:conv_modules.2 N1 IC32 OC32 IH14 IW14 KH3 KW3 SH1 PH1 | 1011 | 35 | 29.0x | 232 | 0 |
| dronet:conv_modules.3 N1 IC32 OC32 IH27 IW27 KH1 KW1 SH2 PH0 | 198 | 55 | 3.6x | 55 | 0 |
| dronet:conv_modules.5 N1 IC64 OC64 IH7 IW7 KH3 KW3 SH1 PH1 | 520 | 28 | 18.5x | 226 | 0 |
| dronet:conv_modules.6 N1 IC32 OC64 IH14 IW14 KH1 KW1 SH2 PH0 | 58 | 21 | 2.8x | 28 | 0 |
| dronet:conv_modules.8 N1 IC128 OC128 IH4 IW4 KH3 KW3 SH1 PH1 | 331 | 44 | 7.5x | 366 | 0 |
| dronet:conv_modules.9 N1 IC64 OC128 IH7 IW7 KH1 KW1 SH2 PH0 | 40 | 19 | 2.0x | 31 | 0 |
| yolov8_nano:detect.cv2_0_2 N1 IC64 OC64 IH20 IW20 KH1 KW1 SH1 PH0 | 769 | 87 | 8.8x | 388 | 0 |
| yolov8_nano:detect.cv3_0_2 N1 IC80 OC80 IH20 IW20 KH1 KW1 SH1 PH0 | 940 | 115 | 8.2x | 683 | 0 |
| yolov8_nano:detect.cv2_1_2 N1 IC64 OC64 IH10 IW10 KH1 KW1 SH1 PH0 | 198 | 33 | 6.0x | 97 | 0 |
| yolov8_nano:detect.cv3_1_2 N1 IC80 OC80 IH10 IW10 KH1 KW1 SH1 PH0 | 231 | 43 | 5.3x | 170 | 0 |
| yolov8_nano:detect.cv2_2_2 N1 IC64 OC64 IH5 IW5 KH1 KW1 SH1 PH0 | 65 | 12 | 5.2x | 24 | 0 |
| yolov8_nano:detect.cv3_2_2 N1 IC80 OC80 IH5 IW5 KH1 KW1 SH1 PH0 | 76 | 16 | 4.7x | 43 | 0 |
| **total** | **6766** | **845** | **8.0x** | **2911** | |

## Notes
- No shape falls back to RVV: IME wins every real shape (smallest win 2.0x, dronet conv_modules.9 1x1 stride 2; ~4.3x on the 5x5 detect convs).
- In-kernel scalar fallback (never hit by the real graphs) when vlenb != 128, input_offset != 0, filter_offset != 0 or SW not in {1,2}.
- Design and the A100 micro-costs that shaped it: header comment of `ime_k3_a100_common.h` and `artifacts/k3_probes/ime/layout_cpu8.txt` (probe `scripts/k3_ime_layout_probe.c`).
- Edge-shape sweep (odd OW, OC % 8 != 0, IC % 16 != 0, N=2, 5x5/7x7/2x2/3x1 kernels, stride 2 with and without padding, OW > 128): all bit-exact.
- Where the time goes (IC80 OC80 20x20 3x3, ~300 us): NHWC16 transpose ~60 us, vmadot k-loop + epilogue ~170 us, BN/SiLU table pass ~40 us.
