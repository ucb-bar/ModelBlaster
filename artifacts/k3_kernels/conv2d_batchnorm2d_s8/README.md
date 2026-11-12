# conv2d_batchnorm2d_s8: K3 A100 IME (vmadot 8x16x8) vs rvv

New kernel: `kernels/ime_k3_a100/ime_k3_a100_conv2d_batchnorm2d_s8_ime_vmadot_8x16x8.c` (shared code in `kernels/ime_k3_a100/ime_k3_a100_common.h`).
Baseline: `kernels/rvv/rvv_conv2d_batchnorm2d_s8_rvv_oc_blocked_bn_epilogue.c` built with the rvv_k3_a100 flags (zvl1024b), run on CPU 8 (A100). X100 column: the same rvv kernel on CPU 0 (rvv_k3, zvl256b).

Median us over 7 reps, steady state (weight pack and BN/SiLU table caches warm; the first call per layer pays ~0.1-0.4 ms to build them).
`max_abs_err` is 0 on every shape, both the full-range and the small-value regime. Raw data: `after_ime_cpu8.json`, `before_rvv_k3_cpu0_full.json`.

| shape | A100 rvv (us) | A100 IME (us) | speedup | X100 rvv (us) | err |
|---|---|---|---|---|---|
| dronet:conv_modules.1 N1 IC32 OC32 IH27 IW27 KH3 KW3 SH2 PH1 | 1728 | 74 | 23.5x | 486 | 0 |
| dronet:conv_modules.4 N1 IC32 OC64 IH14 IW14 KH3 KW3 SH2 PH1 | 572 | 34 | 16.6x | 250 | 0 |
| dronet:conv_modules.7 N1 IC64 OC128 IH7 IW7 KH3 KW3 SH2 PH1 | 400 | 47 | 8.5x | 263 | 0 |
| **total** | **2700** | **155** | **17.4x** | **999** | |

## Notes
- No shape falls back to RVV: IME wins every real shape (smallest win 2.0x, dronet conv_modules.9 1x1 stride 2; ~4.3x on the 5x5 detect convs).
- In-kernel scalar fallback (never hit by the real graphs) when vlenb != 128, input_offset != 0, filter_offset != 0 or SW not in {1,2}.
- Design and the A100 micro-costs that shaped it: header comment of `ime_k3_a100_common.h` and `artifacts/k3_probes/ime/layout_cpu8.txt` (probe `scripts/k3_ime_layout_probe.c`).
- Edge-shape sweep (odd OW, OC % 8 != 0, IC % 16 != 0, N=2, 5x5/7x7/2x2/3x1 kernels, stride 2 with and without padding, OW > 128): all bit-exact.
- Where the time goes (IC80 OC80 20x20 3x3, ~300 us): NHWC16 transpose ~60 us, vmadot k-loop + epilogue ~170 us, BN/SiLU table pass ~40 us.
