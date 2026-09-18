# DroNet KU040 dual-core (2-hart) reproduction

End-to-end reproduction of the **heterogeneous 2-hart DroNet** that runs on the
riskybird v3 KU040 board (`RocketKU040DroneDualConfig`): grayscale, fully-integer,
conv on the 32x32 Gemmini + everything else on the Saturn RVV vector unit, driven
by an XPU-RT capability schedule. **Measured on-board: `MODELBLASTER_VERIFY
max_abs_err=2` (int8-mvout), ~18-20 fps.**

## 1. Hardware / bitstream
- Bitstream `RocketKU040DroneDualConfig` (garden chipyard, lorenshung `ku040-codesign-cnn`):
  - **hart0** = Rocket + Saturn V128D128 RVV (+ FcRoCC, unused here), FP32-only FPU.
  - **hart1** = Rocket + `Q31Ws32x32AccGemminiConfig` Gemmini: **DIM=32**, WS-only,
    Q0.31 acc_scale, 128 KB accumulator, `dma_buswidth=256`.
- Because the harts are FP32-only (misa F, no D), the firmware is **soft-float** and
  uses the **eager-V + kernel-only** pattern so hart1 (no V) never traps.

## 2. The one bug that masqueraded as a HW fault
The 32x32 Gemmini looked like it produced wrong int32 (`max_abs_err=183` on the DroNet
conv, `1346` on a bare matmul) while Spike showed 0. Root cause was **NOT hardware**:
the software `gemmini_params.h` was **DIM=16** while the bitstream mesh is **DIM=32**.
A DIM=16 SW config drives the 32x32 array with wrong tile bounds + mvin/mvout addressing.
Spike agreed only because its model matched the (wrong) DIM=16 software.
Fingerprint of this class of bug: **Spike=0, real-HW != 0, RVV=0** -> chase the header,
not the fabric.

**Fix = the per-config header mechanism.** `MODELBLASTER_GEMMINI_CONFIG=q31ws_32x32_acc`
selects `cores/gemmini/include/per_config/q31ws_32x32_acc/gemmini_params.h` (DIM=32,
BANK_ROWS=2048, MAX_BLOCK_LEN_ACC=1), placed on the kernel `-isystem` line **before**
`cores/gemmini/include`, so it shadows the top-level default. The top-level
`cores/gemmini/include/gemmini_params.h` is intentionally **left at DIM=16** (the default
16x16 config) -- it is irrelevant to this build because per_config shadows it. Only
`ACC_ROWS` is unchanged across the two headers because the 32x32 config doubles
`acc_capacity` 64->128 KB precisely to keep the BRAM geometry constant when DIM doubles.

## 3. Model generation (grayscale, fused, NCHW)
Env (torch + west + numpy; the `zephyr` conda env has all three):
```
source <conda>/etc/profile.d/conda.sh && conda activate zephyr
export ZEPHYR_BASE=<zcs>/zephyr_ws/zephyr
export ZEPHYR_SDK_INSTALL_DIR=<zcs>/tools-manual/zephyr-sdk-1.0.0-beta1
export ZEPHYR_TOOLCHAIN_VARIANT=zephyr
export PATH=/usr/bin:$ZEPHYR_SDK_INSTALL_DIR/gnu/riscv64-zephyr-elf/bin:$PATH  # keep Vitis cmake off PATH
export PYTHONPATH=$(dirname <modelblaster>)
export MODELBLASTER_GEMMINI_CONFIG=q31ws_32x32_acc   # DIM=32
export MODELBLASTER_CURATED_VERIFY=0                  # grayscale golden is random-init (see below)
export MODELBLASTER_DRONET_CHANNELS=1                 # grayscale conv0 (IC=1)
export MB_ENABLE_FUSION=1                             # fuse conv0 -> maxpool1
export MODELBLASTER_GEMMINI_SPIKE=<...>/riscv-tools/bin/spike
export MODELBLASTER_GEMMINI_LIB=<...>/riscv-tools/lib/libgemmini.so
```
- `extract_graph --model dronet --quant int8 --num-calibration 1 --fusion-target gemmini_q31_rvv`
  -> 20-op **NCHW** graph, op0 = fused `conv2d_pool_s8` (IC=1). `conv2d_pool_fuse=1`.
- **Do NOT run `assign_layouts`** for the two-backend build -- NHWC islands insert
  relayout ops that only the single hetero `gemmini_q31_rvv` backend has kernels for.
  Two-backend `gemmini_q31`+`rvv` stays NCHW end-to-end (no relayouts).
- `generate_skeleton` + `generate_kernels` for **both** `gemmini_q31` and `rvv`.
  Kernel picks (no `im2col_full_C` anywhere):
  - `gemmini_q31`: conv2d_pool_s8 -> `gemmini_tiled_conv_pool` (HW conv+pool, path A),
    conv2d_s8 -> `gemmini_tiled_conv` (HW im2col, int8-mvout Q0.31 scale).
  - `rvv`: batchnorm/relu/linear -> `direct`, add -> `rvv_frm_rmm`, sigmoid -> `rvv_memo_lut_gather`.
- **CURATED_VERIFY=0**: the grayscale conv0 is random-init (no trained checkpoint), so the
  per-kernel spike verify-gate fails against a badly-conditioned random-int8 golden and
  silently falls back to scalar. Disabling it selects the fast curated kernels by pick.
  Accuracy is judged on-board instead (below).
- **Do NOT bake the batchnorm** (`bake_bn_int.py`): BN runs on the rvv hart via the
  `direct` kernel, not fused into a gemmini conv, so baking is inapplicable and it broke
  32x32 gemmini tiling in earlier attempts.

## 4. Schedule (capability-driven, no solver)
`gen_hetero_schedule.py --policy gemmini_main_opu_skip` then pin by op:
- **hart1 / gemmini_q31 (10 ops):** conv0 (`conv2d_pool_s8`) + conv1..conv9 (`conv2d_s8`).
- **hart0 / rvv (10 ops):** batchnorm2d, add, relu, linear1/2, sigmoid.
- Every V-using kernel is on hart0; only Gemmini kernels on hart1 -> hart1 cannot trap on V.
- `ingest_xpurt_schedule --cpu-p-kind gemmini_q31 --cpu-e-kind rvv` -> dispatch table with
  **`.impl = gemmini_q31` / `.impl = rvv`** (matching the runner's worker kinds -- an
  `impl=gemmini_q31_rvv` shortcut FATALs at runtime: the walker dispatches by impl and the
  binary only has `rvv`/`gemmini_q31` per-model tables).
- `generate_xpurt_main --core-kinds rvv,gemmini_q31 --backends rvv,gemmini_q31` (kind[i]->backend[i]).
The committed schedule + dispatch table + main are in `examples/dronet/int8/ku040_dual/`.

## 5. Build (two-backend, soft-float, eager-V + V_KERNEL_ONLY)
`harness_xpurt` compiles the model **twice** (`generated/gemmini_q31` + `generated/rvv`,
symbol-mangled `_gemmini_q31` / `_rvv`); the schedule picks per-op hart+impl. Key -D flags:
- `MODEL_BACKENDS=gemmini_q31,rvv`, `MODEL_DIRS_BASE=examples/dronet/int8/generated`
- `MODELBLASTER_KERNEL_CFLAGS_GEMMINI_Q31 = -march=rv64imac_zicsr_zifencei -mabi=lp64
  -isystem .../per_config/q31ws_32x32_acc ... -DGEMMINI_ROCC -DMODELBLASTER_GEMMINI_Q31_ACC_SCALE=1`
- `MODELBLASTER_KERNEL_CFLAGS_RVV = -march=rv64imac_zve64x -mabi=lp64 -DMODELBLASTER_RVV_IHWOC_WEIGHTS=1`
  (Zve64x = integer-vector subset; avoids bare `v` re-admitting D under soft-float.)
- Kconfig: `CONFIG_FPU=n` (soft-float), `CONFIG_RISCV_ISA_EXT_V=y` + `V_LAZY=n` (eager),
  `CONFIG_RISCV_V_KERNEL_ONLY=y`, `CONFIG_MP_MAX_NUM_CPUS=2`, `CONFIG_RV_BOOT_HART=0`.
- Console overlay `ku040_dronedual_console.overlay` (**devicetree** -- enables uart0 @ 50 MHz;
  the `.conf` alone is not enough, the DTS overlay is required or the console device is undefined),
  115200 baud, 1 GiB RAM.
- Harness prints per-op cycles + `MODELBLASTER_VERIFY` (see `harness/src/main.c`
  `MODELBLASTER_PROFILE_ITERS`: iter 0 = WARMUP, discard; average the STEADY iters).

One-shot: **`examples/dronet/int8/build_ku040_dual.sh`** (paths as vars at top; runs
steps 3-5; `REGEN_MODEL=1` regen models, `REGEN_SCHED=1` regen schedule).

### Env gotcha
Do **NOT** `source set_envvars_sdk.sh` under `set -e`/`pipefail` -- its internal `find|head`
SIGPIPEs and aborts the script. Set `ZEPHYR_BASE` / `ZEPHYR_SDK_INSTALL_DIR` /
`ZEPHYR_TOOLCHAIN_VARIANT` explicitly. Launch long jobs from a script file via `setsid`.

## 6. Deploy + measure (on-bench JTAG, SRAM load, no reflash)
SMP boots hart0 with hart1 as the secondary, so **resume BOTH harts**:
```
openocd -f riskybird/scripts/openocd/ku040_dual_smp.cfg \
  -c "init; reset halt; load_image {zephyr.elf}; <set both harts pc=0x80000000>; resume; sleep ...; shutdown"
```
Console on `/dev/ttyUSB3 @ 115200`. Entry 0x80000000.

## 7. Measured results
- **Accuracy:** `MODELBLASTER_VERIFY max_abs_err=2` end-to-end (the int8-mvout HW-scale drift
  on conv1..conv9; <= a few LSB, deliberately accepted). conv0 (path A `tiled_conv_pool`)
  is bit-exact on-board (errA=0).
- **Throughput:** ~18-20 fps.
- Per-op HW-im2col Gemmini cycle costs (not the inflated SW-`im2col_full_C` numbers) come
  from the on-board `MODELBLASTER_PROFILE` output of this same ELF.
- **load-once conv0** (`conv2d_pool_loadonce.h`, path E) is *not* used here: it lives only in
  the `gemmini_q31_rvv` NHWC backend; the two-backend NCHW build uses path A
  (`tiled_conv_pool`), which is bit-exact and sufficient.

## 8. What is committed vs regenerable
- **Committed (this branch):** `cores/gemmini/include/per_config/` (DIM=32 headers),
  `cores/chipyard_ku040_dronedual_q31.json` (registry), `examples/dronet/int8/ku040_dual/`
  (schedule + dispatch table + main), `examples/dronet/int8/build_ku040_dual.sh`,
  `scripts/run_xpurt_scheduler.py` (codegen->IR CSV remap), `pipeline/profile_kernel.py`
  (MODEL_DIR abspath + portable toolchain default), `harness/src/main.c` (PROFILE_ITERS),
  `harness_conv0_lo/` + `harness_gemmini_mm_sanity/` (validation harnesses, src only).
- **Regenerable (gitignored):** `examples/*/*/generated/` model.c/weights/*.bin,
  `build/`, `cache/`, ELFs. Rebuild with `build_ku040_dual.sh`.
