# Verified ModelBlaster codegen, published for machines that cannot verify

ModelBlaster picks a curated kernel only if it passes a per-op check against the
scalar reference. That check *runs* the kernel, which for a RISC-V accelerator
target means spike — and spike ships with the **Chipyard** backend while codegen
runs in the **Zephyr** one. A bench-tier machine has no Chipyard at all, so it
cannot run the gate, and `rb modelblaster build` refuses rather than trusting a
kernel's self-declared accuracy class.

This directory is the way out: codegen run on a silicon-tier host with the gate
working, committed so a Mac can build the same image without a simulator.

    rb modelblaster build --board arty200t \
        --config RocketArty200TDroneGemmini16At40Config \
        --gen-dir artifacts/modelblaster/dronet/int8/gemmini_q31

`.rb-codegen.json` records that verification ran and where the spike that ran it
came from. Regenerating on a machine that cannot verify overwrites the marker
and the refusal returns, which is the intent.

## Four directories, two shells

For `RocketArty200TDroneGemmini16At40Config` — Gemmini only, full scalar FPU:

| directory | accepted at | conv2d_s8 | error | measured |
| --- | --- | --- | --- | --- |
| `dronet/int8/gemmini_q31-drift2` | 2 LSB | **Gemmini** | **2** | **188 ms, 5.31 FPS** |
| `dronet/int8/gemmini_q31` | bit-exact | scalar reference | **0** | 15,792 ms, 0.06 FPS |

For `RocketArty200TDroneGemminiSaturnFp16At35Config` — the 93% Gemmini+Saturn
shell, whose scalar FPU implements only Zfh, so the image is built `CONFIG_FPU=n`
and every kernel must inherit that soft-float ABI:

| directory | accepted at | on Gemmini | error | measured |
| --- | --- | --- | --- | --- |
| `dronet/int8/gemmini_q31_softfp` | 2 LSB | maxpool, relu, sigmoid | ≤2 | not yet run |
| `dronet/int8/scalar` | bit-exact | nothing | **0** | not yet run |

**Those two are the slow way onto the 93% shell.** `conv2d_s8` is on scalar
reference in both of them and it is ~99% of the network. The fast way is the
next section, which is what the Gemmini+Saturn shells actually run.

## Two more directories: Dima's recipe on the Gemmini+Saturn shells

Grayscale input, conv0 and maxpool fused into one Gemmini call, NHWC islands,
and the `gemmini_q31_rvv` kernels compiled for an integer-only vector unit
(`zve64x`, never `v`, which GCC expands to imply F and D). The C in the two
directories is identical apart from the backend name; the targets differ in
flags, and the noPermute one keeps `linear_s8` on the scalar reference
because the RVV one compiles to `vmv4r.v`, which that Saturn decodes as
illegal. Both were accepted at 3 LSB.

| directory | shell | kernel flags | measured on the board (2026-09-24) |
| --- | --- | --- | --- |
| `dronet/int8/gemmini_q31_rvv_softfp-gray-nhwc` | `RocketArty200TDroneGemminiSaturnFp16At35Config` | `rv64imac_zve64x` / lp64, image `CONFIG_FPU=n` | **13.36 fps** (WALL 2620-2622 at 35 kHz), err 3, `[114,127]`, 3/3 |
| `dronet/int8/gemmini_q31_rvv_zve64x-gray-nhwc` | `RocketArty200TDroneGemminiSaturnIntAt35Config` | `rv64imafdc_zve64x` / lp64d | **14.57 fps** (WALL 2401-2402), err 3, `[114,127]`, 3/3 |

For comparison, `gemmini_q31-drift2` on Gemmini16At40 measured 5.32 fps
(WALL 7515-7526 at 40 kHz) in the same session.

    rb modelblaster build --board arty200t \
        --config RocketArty200TDroneGemminiSaturnIntAt35Config \
        --gen-dir artifacts/modelblaster/dronet/int8/gemmini_q31_rvv_zve64x-gray-nhwc \
        --ram-console

`rb modelblaster generate` cannot express this recipe, so it has its own
script: `tools/modelblaster/dronet_gray_nhwc.py --target <target>` regenerates
either directory, and on 2026-09-24 it reproduced both byte for byte from a
clean clone (`kernel_picks.json` excepted: it now names each kernel by its path
inside ModelBlaster rather than by the codegen host's scratch directory). It
is silicon-tier only, because its gate runs spike with a Q0.31 Gemmini model
the script builds itself; the script's docstring says why Chipyard's own
`libgemmini.so` cannot serve.

The gate's per-op results for dronet / int8 / gemmini_q31, measured on spike
against an all-reference baseline — each kernel swapped in alone, so these are
per-kernel facts and not collateral from one bad neighbour:

| op | algorithm | result |
| --- | --- | --- |
| `conv2d_s8` | `gemmini_tiled_conv` | misses exact by **2 LSB** |
| `conv2d_s8` | `gemmini_im2col_full_C` | **broken**, 56 LSB |
| `linear_s8` | `gemmini_tiled_matmul` | **broken**, 56 LSB |
| `relu_s8` | `gemmini_resadd_relu` | **broken**, 56 LSB (archived; `swar` used) |
| `add_s8`, `batchnorm2d_s8`, `maxpool2d_s8`, `sigmoid_s8` | curated | exact, 0 |

### Why the 2 LSB is worth having

`conv2d_s8` dominates DroNet, and the difference between rejecting and accepting
that drift is not marginal:

```
conv2d_s8   460,351,223 cycles   scalar reference   (bit-exact variant)
conv2d_s8     1,989,282 cycles   gemmini_tiled_conv (drift2 variant)
```

On hardware, both variants run back to back on the same board in one session:

| | drift2 | bit-exact |
| --- | --- | --- |
| `MODELBLASTER_VERIFY` | max_abs_err **2** | max_abs_err **0** |
| inference | **188.4 ms** | 15,791.8 ms |
| frames/s | **5.31** | 0.06 |
| Gemmini share of kernel time | 99.3% | 0.7% |

**83.9x end to end**, and the A/B isolates cleanly: `conv2d_s8` goes
627,380,812 -> 3,241,262 cycles (**193.6x**) while every other op moves by less
than 1%. The error lands at exactly the accepted bound, not above it.

(The 231x quoted earlier was measured on spike, 460,351,223 -> 1,989,282. The
193.6x here is the same comparison on hardware, whose scalar baseline is larger.
Prefer the hardware figure.)

**This is not the mistake that produced wrong numbers earlier.** That run had no
gate at all and used `gemmini_im2col_full_C`, the 56-LSB kernel, with nothing
measuring it. Here the gate runs, measures the error, and accepts a bound that
is recorded in `.rb-codegen.json` and printed at build time. The difference is
between a known 2 and an unknown anything.

Whether 2 LSB is acceptable is a judgement about the airframe, not about the
tooling. DroNet's outputs are a steering angle and a collision probability; 2 of
255 is ~0.8% on each. `linear_s8`, which produces those two numbers, stays on
scalar reference in both variants, so the drift enters through the convolution
stack rather than the final layer.

`linear_s8` and `gemmini_im2col_full_C` are broken by 56 LSB and no tolerance
setting should be used to admit them. Fixing those is kernel work.


## Reading the output of a fast workload

At 188 ms this workload finishes before a human can attach a console, and
`rb debug` starts the program the instant it runs. Output goes to channel B as
it is produced, so anything printed before you attach is gone — which on the
bench looked like a failing UART: the 15.79 s run yielded 978 lossy bytes and
the 188 ms run yielded 15, the reproducible reset glitch and none of the
program's output.

The console was working both times. Use `--ram-console`, which buffers into a
RAM ring read back over the same JTAG link, or `rb debug --no-run` and
`continue` once the console is up. `rb debug` now says so before it starts
anything.


## Why the 93% shell cannot yet match 188 ms

`gemmini_q31` pins `-march=rv64imafdc -mabi=lp64d`. The FP16 shell's FPU
implements only Zfh, so that ABI cannot be linked against its image and those
instructions cannot be executed. `gemmini_q31_softfp` drops the two flags so
kernels inherit whatever the image uses — and with the flags gone, four kernels
change behaviour:

| op | under `gemmini_q31` | under `gemmini_q31_softfp` |
| --- | --- | --- |
| `conv2d_s8` | 2 LSB (accepted) | **56 — rejected** |
| `add_s8` | 0 | **56 — rejected** |
| `linear_s8` | 56 (rejected) | 56 — rejected |
| `batchnorm2d_s8` | 0 | **56 — rejected** |
| `maxpool2d_s8`, `relu_s8`, `sigmoid_s8` | 0 | 0 |

The kernels are identical C. Something in those four depends on `-march`/`-mabi`
rather than merely being compiled under them, and until that is understood the
soft-float shell keeps conv2d on the scalar reference. The gate catches it in
both directions, which is the only reason this is a known limitation rather than
another wrong number.
