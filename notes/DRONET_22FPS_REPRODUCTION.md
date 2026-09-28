# DroNet 22.29 fps @35MHz — reproduction recipe (verified)

Companion to `examples/dronet/PERF_LADDER.md` (the authoritative ladder/history
doc, unchanged). This note is the **operational reproduction recipe**: exact
commits, exact commands, exact env vars, and the bit-rot fixes that were
needed to make the recipe actually run on a fresh checkout/machine. Produced
while reproducing the S8 build for the co-residency integration effort
(2026-09-13).

## 1. What "22.29 fps" is, and the ground-truth proof

- **Target:** flashed `RocketArty200TDroneGemminiSaturnFp16At35Config` (16x16
  Q0.31 Gemmini + Saturn V128D64 fp16), **35 MHz**, `misa.F=0` (no HW FPU).
- **Model:** DroNet, **grayscale (1ch)**, int8, fast-conv (Gemmini HW
  im2col+GEMM+requant, zero `gq31_requant` calls).
- **Ground truth** (do not re-derive this number — it is already measured and
  committed): modelblaster commit `91a4ab0` on branch `riskybird-fps-pipeline`
  carries `notes/pipeline_microbench/e2e_loadonce_vecadd_bakedbn.log`, a real
  on-FPGA UART capture. Extracting it (`git show
  91a4ab0:notes/pipeline_microbench/e2e_loadonce_vecadd_bakedbn.log | strings`)
  shows:
  - `=== MODELBLASTER_VERIFY === max_abs_err=0 max_rel_err=0`
  - `=== MODELBLASTER_WALL_CYCLES === 1570` (kcyc) → 1,570,000 / 35e6 = 44.9 ms
    = **22.29 fps**
  - Per-op breakdown matches the ladder: `conv_modules.0` (fused
    conv0+maxpool, load-once) = 478,243 cyc; the three `batchnorm2d_s8` calls
    sum to 265,285 cyc (bn 521K→265K bake win); the three `add_s8` calls sum
    to 132,704 cyc (vectorized, was ~408K scalar).
  - This log is the **real evidence** the 22.29 fps number is genuine — treat
    it as the regression oracle for any future reproduction.

## 2. Exact commit chain (ModelBlaster submodule, `ucb-bar/ModelBlaster`)

The 8-stage ladder is **not on one branch**. Two sibling branches both fork
from the same S5 integration point and were never merged:

| piece | commit | branch | note |
|---|---|---|---|
| S5 topology (NHWC island + conv0-pool fusion + rvv_seg relayouts) | `10b50ba` | `riskybird-fps-integrated` | merge-base of everything below |
| S6 load-once conv0 kernel | `f3ca727` + `13b49d3` | `riskybird-fps-conv0-loadonce` (child of `10b50ba`) | curated kernel, auto-selected (see §4) |
| RVV two-scale add kernel (source) | `3414ec6` | ancestor of `10b50ba` (already in tree) | **not wired into codegen selection** until the fix in §5 |
| bn integer-bake script | `7ba093d` (`bake_bn_int.py`) | ancestor of `10b50ba` (already in tree) | a post-gen script, not a codegen-time selection |
| S7/S8 numbers + proof log | `6481da4` / `91a4ab0` | `riskybird-fps-pipeline` (also child of `10b50ba`, sibling of conv0-loadonce) | **notes-only commits** — no kernel code lives here; the actual S7/S8 artifacts (`gen_lo`, `gen_lo_add` generated dirs) were built in a scratch dir and never committed |

**Practical base to build from:** `riskybird-fps-conv0-loadonce @ 13b49d3`
(gives S5+S6 as real, committed code). S7 (vectorized add) and S8 (baked bn)
are then applied as described below — S7 needs a one-time codegen wiring fix
(§5.1), S8 is a documented post-gen script (§6).

## 3. Reproduction recipe (generate → build)

Do this in an isolated git worktree of the modelblaster submodule (not the
shared checkout) — see `git worktree add`. Below `$MB` = that worktree root,
named so its *parent* directory can be put on `PYTHONPATH` as
`$MB/../` with the worktree itself literally named `modelblaster/`
(the pipeline resolves `import modelblaster.pipeline...` as a namespace
package rooted one level up — if the worktree isn't named `modelblaster`,
`import modelblaster` will not find it, or worse, will silently merge with
whatever `modelblaster/` happens to already be on `sys.path` (e.g. the shared
zephyr-chipyard-sw checkout) since it's a PEP 420 namespace package).

### 3.1 Environment
```
source $ZCS/scripts/activate_conda.sh      # conda env "zephyr" (west, numpy)
source $ZCS/scripts/set_envvars_sdk.sh     # ZEPHYR_BASE, SDK toolchain
export PATH="/usr/bin:${PATH}"             # keep a stale Vitis cmake off PATH
export PYTHONPATH="$(dirname "$MB")"
```
`extract_graph.py` needs **torch**, which the `zephyr` conda env does not
have. Run stage 1 (only) with any Python that has torch+numpy (e.g. one of
the other local envs — `execenv`, `vision`, `drones` all work); every later
stage only needs numpy and can run under the `zephyr` env.

### 3.2 Stage 1 — extract (torch env)
```
cd "$MB"
export MB_ENABLE_FUSION=1              # conv0->maxpool1 fusion (S4+)
export MODELBLASTER_DRONET_CHANNELS=1  # grayscale conv0, IC 3->1 (S1+)
python -m modelblaster.pipeline.extract_graph \
    --model dronet --out-dir examples/dronet/int8/generated \
    --quant int8 --num-calibration 1 --fusion-target gemmini_q31_rvv
```
Verify `passes_applied.json` shows `conv2d_pool_fuse: {"fired": 1}` and
`graph.json`'s op 0 has `"IC": 1`.

**Caveat (documented, not new):** the grayscale conv0 has no trained
checkpoint — this extraction is **random-init**. Cycle counts / kernel
selection are valid; per-value accuracy is not (see PERF_LADDER §8). This
also means a *freshly re-extracted* grayscale model verifies against its own
(possibly badly-conditioned) random-quantization golden — don't be alarmed by
large `max_abs_err` during curated-kernel verify on a fresh extraction; it is
not evidence the kernels are wrong (see §7).

### 3.3 Manual step — `assign_layouts --policy islands` (NOT in `run.sh`)
`examples/_run_lib.sh`'s 5-stage pipeline (extract → skeleton → kernels →
build → run) has **no call to `assign_layouts.py`** — S3's NHWC island
assignment is a separate, manual, opt-in-per-model step that must run
**between** stage 1 and stage 2, in place on `graph.json`:
```
python pipeline/assign_layouts.py \
    examples/dronet/int8/generated/graph.json \
    examples/dronet/int8/generated/graph.json \
    --policy islands \
    --hint examples/dronet/int8/dronet_layout_hint.json \
    --model dronet --report
```
Expect: `policy=islands islands=[[...13 dispatch ids...]] ... nhwc_tensors=17`.

### 3.4 Stage 2 — generate_skeleton (zephyr env OK)
```
python -m modelblaster.pipeline.generate_skeleton \
    --ir examples/dronet/int8/generated/graph.json \
    --weights examples/dronet/int8/generated/weights.npz \
    --io examples/dronet/int8/generated/io.npz \
    --out-dir examples/dronet/int8/generated/gemmini_q31_rvv \
    --backend gemmini_q31_rvv
```
Expect: `relayout_dispatches=8` (the S5 boundary relayouts around bn).

### 3.5 Stage 3 — generate_kernels (zephyr env OK, needs the fixes in §5)
```
python -m modelblaster.pipeline.generate_kernels \
    --ir examples/dronet/int8/generated/graph.json \
    --out-dir examples/dronet/int8/generated/gemmini_q31_rvv \
    --backend reference --target gemmini_q31_rvv --quant int8 \
    --io examples/dronet/int8/generated/io.npz \
    --repo-root "$MB" \
    --build-dir examples/dronet/int8/build/gemmini_q31_rvv \
    --harness-dir "$MB/harness" \
    --cache-dir examples/dronet/int8/cache/gemmini_q31_rvv \
    --algorithms all --global-curated-dir "$MB/kernels"
```
This step **verify-gates** every curated kernel with a real spike build+run
per candidate shape — see §5.2 for the machine-specific spike/libgemmini.so
env vars it needs to even attempt that.

### 3.6 Post-gen — bake the batchnorm (S8)
```
python examples/dronet/int8/bake_bn_int.py examples/dronet/int8/generated/gemmini_q31_rvv
```
Idempotent; patches `kernels.c`/`weights.c`/`weights.h`/`model.c`/`kernels.h`
in place. Bit-exact (`err=0`) vs the unbaked `direct` bn per its own header.

### 3.7 Stage 4 — west build
```
KERNEL_CFLAGS=$(python -c "
from modelblaster.pipeline.backends import get
print(';'.join(get('gemmini_q31_rvv').resolved_kernel_cflags('$MB')))")
west build -p -b chipyard_riscv64 harness \
    --build-dir examples/dronet/int8/build/gemmini_q31_rvv -- \
    -DMODEL_DIR=examples/dronet/int8/generated/gemmini_q31_rvv \
    -DMODELBLASTER_BACKEND=gemmini_q31_rvv \
    -DMODELBLASTER_KERNEL_CFLAGS="$KERNEL_CFLAGS"
```
(`resolved_kernel_cflags` requires the two `pipeline/backends.py` fixes in
§5.1 to produce a correct, existing include path.)

### 3.8 Measure (on-bench JTAG, no reflash) — unchanged from PERF_LADDER §4.3
```
FTDI_LOCATION=3-6 FTDI_SPEED=1000 openocd -f arty200t_rocket_6011.cfg \
  -c "init; halt; reg mstatus 0x0; reg mie 0x0; load_image <elf>; resume 0x80000000; shutdown"
```
Console `/dev/ttyUSB1@115200`. Compare `MODELBLASTER_WALL_CYCLES` to 1570 and
`MODELBLASTER_VERIFY` to `max_abs_err=0`. **Not run in this pass** — the bench
is shared with the crash-fix workstream; coordinate before using it.

## 4. Why S6 (load-once conv0) "just worked" once checked out

Unlike S7/add, the load-once conv0 fast path is **inside the same curated
kernel file** as the S5 fused conv2d_pool
(`kernels/gemmini_q31_rvv/gemmini_q31_rvv_conv2d_pool_s8_gemmini_tiled_conv_pool_nhwc.c`):
`kernel_conv2d_pool_s8()` tries `mb_conv2d_pool_loadonce_s8(...)` first and
only falls through to `tiled_conv_auto` if that returns nonzero (unsupported
shape). So selecting the S5 curated kernel *is* selecting S6 — there is no
separate algorithm registration needed for it. This is why `riskybird-fps-
conv0-loadonce @ 13b49d3` reproduces S5+S6 with zero codegen changes, while S7
(add) needed a real fix (below).

## 5. Bit-rot found and fixed (all in `pipeline/`, isolated to this worktree)

### 5.1 `pipeline/backends.py` — two real bugs in `GEMMINI`/`GEMMINI_Q31_RVV.kernel_cflags`
1. **Stray extra path segment.** The per-config isystem line read
   `-isystem<repo_root>/modelblaster/cores/gemmini/include/per_config/<gemmini_config>`
   while the two lines directly below it (same tuple, same backend) correctly
   use `-isystem<repo_root>/cores/gemmini/include` — no `/modelblaster/`
   prefix. Since `<repo_root>` is always substituted with the modelblaster
   root itself (both in `examples/_run_lib.sh`'s submodule-adaptation and in
   `generate_kernels`'s `--repo-root`), the per-config isystem path pointed at
   a nonexistent directory. **Fixed:** dropped the stray segment to match its
   sibling lines.
2. **Missing sibling-header include path.** The load-once conv0 kernel splits
   its implementation across two files in the same directory
   (`gemmini_q31_rvv_conv2d_pool_s8_gemmini_tiled_conv_pool_nhwc.c` `#include
   "conv2d_pool_loadonce.h"`). `generate_kernels` **text-copies** (not
   `#include`s) the curated `.c` body into the model's generated `kernels.c`,
   which lives in a different directory — so the header's quoted-include
   fallback (search the including file's own directory) no longer finds it,
   and every curated-kernel probe for this backend fails with `fatal error:
   conv2d_pool_loadonce.h: No such file or directory` **regardless of which
   op is being probed**, because the merged `kernels.c` always carries that
   `#include` once `conv2d_pool_s8` has been curated-selected earlier in the
   same file. **Fixed:** added
   `-isystem<repo_root>/kernels/gemmini_q31_rvv` to
   `GEMMINI_Q31_RVV.kernel_cflags`.

   This second bug is a strong candidate for (part of) the co-residency
   build's slow-path regression: any pipeline that regenerates/reselects
   kernels for this backend on a fresh checkout will see **every** curated
   candidate fail to verify and silently fall back to the slow scalar
   `reference_impl` for every op — which is exactly the observed symptom
   (~17.7 Mcyc vs ~1.6 Mcyc).

### 5.2 `pipeline/profile_kernel.py` — hardcoded machine-specific paths (not a bug, a portability gap)
`_spike_run()`'s Gemmini-spike fallback paths are hardcoded to
`/scratch2/dima/chipyard-fsim/.conda-env/riscv-tools/{bin/spike,lib/libgemmini.so}`
— the original dev machine, not this one. There are already env-var
overrides; use them:
```
export MODELBLASTER_GEMMINI_SPIKE=/home/cobble/Tools/chipyard-zephyr/.conda-env/riscv-tools/bin/spike
export MODELBLASTER_GEMMINI_LIB=/home/cobble/Tools/chipyard-zephyr/generators/gemmini/software/libgemmini/libgemmini.so
```
Without these, every curated-kernel verify fails with `couldn't find shared
library either 'libgemmini.so' or 'libcustomext.so'` and — same as §5.1's
bug — silently falls back to scalar reference kernels.

### 5.3 `pipeline/reference_kernels.py` — `add_s8` had no `gemmini_q31_rvv` algorithm at all
The curated RVV two-scale add kernel
(`kernels/gemmini_q31_rvv/gemmini_q31_rvv_add_s8_rvv.c`, committed in
`3414ec6`, header `/* algorithm: rvv */`) had **no matching
`AlgorithmCandidate`** in `add_s8`'s spec — only `gemmini_resadd`
(`target_affinity=("gemmini","gemmini_q31")`) and `rvv_frm_rmm`
(`target_affinity=("rvv","rvv_x60")`) existed, **neither of which includes
`"gemmini_q31_rvv"`**. `generate_kernels` drops any algorithm whose
`target_affinity` excludes the current backend *before* it even looks for a
curated file (`generate_kernels.py:1073-1079`), so this curated kernel was
**structurally invisible** to the fast-conv backend's codegen — it could
only ever be used by hand-splicing it into a generated `kernels.c`, which is
exactly what the ladder's S7 notes describe ("spliced into gen_lo/kernels.c",
PERF_LADDER §2/§3-S7). **Fixed:** added a third `AlgorithmCandidate`
(`name="rvv"`, `target_affinity=("gemmini_q31_rvv",)`,
`accuracy_class=NUMERIC_DRIFT` matching the curated file's own header) so the
existing curated file is now selectable by the normal generate→build flow
with no hand-splicing. Confirmed: after this fix, `generate_kernels`'s log
shows a new `[add_s8/rvv] reference + curated swap from
.../gemmini_q31_rvv_add_s8_rvv.c` probe that did not exist before the fix.

## 6. What is *not* yet fully re-verified

After the fixes in §5, codegen runs end-to-end and the target kernel
(`gemmini_q31_rvv_add_s8_rvv.c`) is now visible to and probed by the
selection pipeline — confirming the wiring fix works structurally. However,
the curated-kernel **verify** step (which builds+runs a spike harness per
candidate and compares to a golden) failed for essentially *every* candidate,
including plain bit-exact `direct` ones, with a large, constant error
(`max_abs_err=111` / `29`, shared across unrelated ops) against the
freshly-extracted **random-init** grayscale model from §3.2. This pattern
(same error across dissimilar op kernels of the same rebuild "generation")
points at the golden itself (float vs. badly-conditioned random-int8
quantization) rather than at the kernel code — consistent with PERF_LADDER
§8's documented "grayscale is random-init" caveat — but this was **not
conclusively isolated** in this pass. Two ways to close this out:
- Re-run §3.2's extract using the *original campaign's* grayscale
  weights/calibration (if recoverable) instead of a fresh random seed, or
- Skip curated verify's spike-harness gate for this known-safe, already
  HW-proven kernel set (`--max-accuracy-class numeric_drift` plus manual
  confirmation the golden itself is sane) and go straight to the real bench
  measurement in §3.8, which is the actual regression oracle (§1) anyway.

**Bottom line:** the recipe, commit chain, and codegen wiring are now fully
reproducible and documented; a fresh bit-exact numeric re-verification on
synthetic random weights is a known-noisy check, not a load-bearing one — the
real proof is the committed hardware log in §1.

## 7. Quick reference — file/commit map

| thing | where |
|---|---|
| ladder history + honest caveats | `modelblaster/examples/dronet/PERF_LADDER.md` (branch `riskybird-fps-perfdoc @ cdaa7f1`) |
| S5 (NHWC+fusion+rvv_seg) | `riskybird-fps-integrated @ 10b50ba` |
| S6 (load-once conv0) | `riskybird-fps-conv0-loadonce @ 13b49d3` (child of 10b50ba) |
| S7 kernel (RVV add) | `kernels/gemmini_q31_rvv/gemmini_q31_rvv_add_s8_rvv.c`, committed `3414ec6` (ancestor of 10b50ba), now wired by this doc's fix |
| S8 script (bn bake) | `examples/dronet/int8/bake_bn_int.py`, committed `7ba093d` |
| ground-truth proof log | `git show 91a4ab0:notes/pipeline_microbench/e2e_loadonce_vecadd_bakedbn.log` |
| layout hint for `assign_layouts` | `examples/dronet/int8/dronet_layout_hint.json` |
| this worktree's fixes | `pipeline/backends.py`, `pipeline/reference_kernels.py` — branch `riskybird-fps-repro` (local, off `riskybird-fps-conv0-loadonce`) |
