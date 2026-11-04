# RiskyBird model development

This directory owns the model recipes and generated baselines formerly stored in
RiskyBird's integration and PCB repositories. The owning checkout is the single
`backends/zephyr-chipyard-sw/modelblaster` submodule; board firmware/overlays belong
in its Zephyr parent and Scala hardware configs belong in Chipyard.

## Preserved histories and outputs

The migration merges lab `riskybird-fps-repro` (`aaf0d44`), the fork's Arty support
(`1293334`), and opt-in placement (`66da197`). `artifacts/` preserves all 79 tracked
files from the integration repository's `artifacts/modelblaster/`, including the
three measured Arty baselines and their verification metadata. These sources
were copied without regeneration; moving them is not a fresh hardware test.

`dronet_temporal/recovery-manifest.json` identifies the recovered PCB source
commit and original SHA256 values. `int8_sim.py` was recovered from the readable
local research workspace. Its source is now committed alongside `hybrid_sim.py`.
The recovery commit retains the original scripts before portability edits.

## Arty grayscale source generation

Activate a Python/Zephyr environment providing torch, numpy and west, set
`ZEPHYR_BASE`, `ZEPHYR_SDK_INSTALL_DIR` and `ZEPHYR_TOOLCHAIN_VARIANT`, then run:

```sh
python riskybird/dronet_gray_nhwc.py --target gemmini_q31_rvv_softfp \
  --riscv-tools /path/to/chipyard/.conda-env/riscv-tools --work /path/to/scratch
```

`RB_RISCV_TOOLS` is an alternative to `--riscv-tools`; `--python` selects the
interpreter for generation. The script inherits the activated environment and
has no dependency on RiskyBird's `rb` package. It builds the pinned Q31 Spike
extension and verifies curated kernels. Generated markers retain the `.rb-codegen.json`
contract for the integration CLI. No passed verification means a failing command.

## Temporal training and evaluation

External dataset paths are explicit. Use the PULP-DroNet v3 layout documented at
https://zenodo.org/records/13348430, and retain the dataset provenance with results.

```sh
export MODELBLASTER_DRONET_STACK3_DATA=/path/to/Dataset_PULP_Dronet_v3
python riskybird/dronet_temporal/train_stack.py --N 3 --epochs 45 --out /path/to/run
N=3 N_CALIB=64 N_SUB=6000 python riskybird/dronet_temporal/eval_stack_sweep.py
```

The sweep defaults to the committed temporal checkpoint for N=3. Set
`MODELBLASTER_DRONET_STACK3_CKPT` to evaluate another checkpoint (required for other
N values). `train_himax.py` provides the single-frame baseline. `fp16_char.py`
requires `MODELBLASTER_DRONET_CKPT` and either Himax data with `DATASET=himax` or
`MODELBLASTER_SIMFOREST_DATA` with `DATASET=simforest`.

Training uses the in-tree `models/dronet_arch.py`; no XPU-RT checkout is needed.
For the 112x112 temporal deployment, its outputs were checked against the old
XPU-RT architecture after loading identical trained weights. The recovered hybrid simulator is a steering-only evaluator: it intentionally
bypasses the unused collision-head sigmoid. Its steering output matched the
recovered int8 simulator on an extracted trained temporal model; the collision
output is not a verification oracle. The recovered
single-frame characterization accidentally selected the generic wrapper's
random-init grayscale model; it now loads the supplied checkpoint explicitly.
This correction means its old accuracy figures must be measured again.

`--model dronet_stack3` loads committed trained weights and real temporal
calibration samples. Missing data is an error. Synthetic samples require explicit
`MODELBLASTER_DRONET_STACK3_ALLOW_SYNTHETIC=1` and are not an accuracy evaluation.
The historical per-frame dataset split can put validation frames close in time
to training frames. Re-evaluate by whole acquisition/time block before relying
on absolute task-accuracy claims.

## KU040 build recipes

Activate the Zephyr build environment and provide `ZEPHYR_SDK_INSTALL_DIR`,
`CONSOLE_CONF`, and `CONSOLE_OVERLAY`. `ZCS` defaults to ModelBlaster's parent;
use it to select another Zephyr workspace. These recipes build ELFs only:

```sh
bash examples/dronet/int8/build_ku040_dual.sh       # single-frame NCHW baseline
bash examples/dronet/int8/build_ku040_dual_nhwc.sh  # single-frame NHWC baseline
bash riskybird/dronet_temporal/stack3_build.sh     # historical temporal benchmark
```

The scripts stop on generation/build errors. The temporal script preserves the
measured legacy policy: 14 Gemmini operations, 14 RVV operations, trained temporal
checkpoint, synthetic input, and curated verification disabled. NCHW-to-NHWC
relayouts run on RVV; reverse relayouts run on Gemmini. The original script's
nonexistent `nchw_to_stack3_s8` op spelling caused this placement; the new code
makes the existing policy explicit rather than changing the benchmark silently.
The newer real-data `dronet_stack3` model is not substituted into this recipe.
Unifying that recipe requires a fresh numeric and hardware timing baseline.

The canonical single-frame NHWC policy remains 18 Gemmini / 10 RVV operations.
These scripts intentionally retain `MODELBLASTER_CURATED_VERIFY=0` to preserve
the recorded selection; successful build is not numeric validation. The board
must use DIM=32 Gemmini headers and matching integer-vector/soft-float flags.

## Validation and remaining work

Run `python -m unittest discover -s riskybird/tests -v` in the model environment.
The host checks cover artifact assembly relocation, required datasets, temporal
ordering/partition boundaries, simulator agreement, and failure propagation.
Shell parsing and Python compilation were also checked during migration.

No new FPGA build, deployment, retraining, broad accuracy sweep or hardware
performance run is claimed. Recover/republish the exact KU040 ELF, bitstream,
raw capture and calibration hashes before considering its historical FPS
reproducible from the new tree. The 22.29-fps note's `91a4ab0` evidence object
was not present in the audited reachable trees or local reproduction worktree.
The unrelated sibling checkout's IME and uncommitted NVFP4 work remains separate.
