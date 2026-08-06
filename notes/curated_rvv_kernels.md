# Curated RVV kernels: build, verify, and pick (reproducible)

How to add a hand-written ("curated") RVV kernel for an op, verify it **bit-exact
on the V-spike simulator**, and get it auto-selected by `generate_kernels`. Written
up after landing curated `conv2d_s8_pc` + `linear_s8_pc` for the fused drone-nav
model (commit c1dc2f2, branch rose-2-dev).

See also: `notes/mixed_precision_plan.md`, and the project memory
`rose-modelblaster-spike-build` / `rose-modelblaster-multiinput-lstm`.

---

## 0. TL;DR of the gotchas (read these first)

1. **`--out-dir` for `generate_kernels --verify` MUST be `<IR_DIR>/<target>/`.**
   The verify's `profile_kernel.build_and_run` re-derives the model name by looking
   for `graph.json` next to `model_dir` (in it, or its parent). It does **not** read
   the name from `--ir`. If it can't find `graph.json`, `model_name=None`, kernel
   symbol mangling becomes a no-op, and `kernels.c` emits the **unmangled**
   `kernel_<op>` while `model.c` calls the **mangled** `kernel_<op>_<modelname>`
   → undefined reference at link → the verify reports a generic
   *"spike-harness build/run failed"* with the real error buried (it only keeps
   `stdout/stderr[-3000:]`, which is the cmake FATAL wrapper, not the cc1/link line).

2. **The verify's `MODEL_DIR` must already contain the skeleton.** `generate_kernels`
   only writes `kernels.{c,h}`. You must run `generate_skeleton` into the **same**
   out-dir first, so `model.c/model.h/weights.c/buffers.c/test_io.{h,S}/test_golden.bin`
   are present — CMakeLists.txt hard-requires them.

3. **Two Python environments.** `generate_*` needs the torch/`requests` env
   (`.../envs/xpurt`); the spike build needs `west`+SDK from the in-tree zephyr env.
   Source the zephyr env (puts `west` on PATH) and invoke the pipeline with the
   **xpurt python by absolute path** — the `west` subprocess inherits the env.

4. **`rvv` vs `rvv_f16`.** If the plain `rvv` backend won't build on ISA grounds,
   use `rvv_f16` (`-march=rv64gcv_zfh_zvfh -mabi=lp64d`). Curated files live at
   `kernels/<target>/<target>_<op>_<algo>.c` and are probed via `--global-curated-dir`.

5. **spike's cost model charges vector loads ~per byte.** Pre-widening int8 weights
   to int16 in memory was *slower* than an int8 load + widening `vwmacc`. Optimize for
   fewer **bytes loaded**, not fewer instructions. (Details in the conv section below.)

---

## 1. Environment

```bash
MB=/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/ModelBlaster
ZCS=/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw
PY=/scratch2/dima/miniforge3/envs/xpurt/bin/python     # has torch + requests

cd $ZCS
source scripts/activate_conda.sh      # conda env 'zephyr' -> provides west
source scripts/set_envvars_sdk.sh     # ZEPHYR_BASE, SDK, PATH (west)
export ZEPHYR_TOOLCHAIN_VARIANT=zephyr
cd $MB
```

## 2. Write the curated kernel

Create `kernels/<target>/<target>_<op>_<algo>.c` for **both** the target you build and,
if you also verify under `rvv_f16`, its dir. For the "direct" (synthesized, affinity-free)
algorithm the file is `<target>_<op>_direct.c`.

- The function is named `kernel_<op>` (unmangled — the pipeline mangles it on emit).
- Match the reference `KernelSpec.signature` **exactly** (arg order/types); `_check_signature`
  rejects mismatches before it ever builds.
- Header comment lines are read as metadata:
  `/* source: curated */  /* algorithm: direct */  /* accuracy_class: bit_exact */`
- Keep it **bit-exact** vs the reference impl (same integer requant math) unless you
  intend a `numeric_drift`/`approximate` class.
- `#include <riscv_vector.h>`; you may define file-local `static` helpers (they land in
  the same TU as the other ops' kernels, so give them unambiguous names).

## 3. Extract the IR (once)

```bash
IR=$MB/scratch/fused_full/int8            # <-- pick any dir; graph.json lands HERE
PYTHONPATH=$MB/src $PY -m modelblaster.pipeline.extract_graph \
  --model fused_full --quant int8 --per-channel --num-calibration 8 --out-dir $IR
# writes $IR/graph.json, $IR/io.npz, $IR/weights.npz
```

## 4. Generate skeleton + kernels INTO `<IR>/<target>/`  (gotchas #1, #2)

```bash
TARGET=rvv_f16
GEN=$IR/$TARGET                            # <-- parent ($IR) has graph.json. REQUIRED.
mkdir -p $GEN

PYTHONPATH=$MB/src $PY -m modelblaster.pipeline.generate_skeleton \
  --ir $IR/graph.json --weights $IR/weights.npz --io $IR/io.npz \
  --out-dir $GEN --backend $TARGET

PYTHONPATH=$MB/src $PY -m modelblaster.pipeline.generate_kernels \
  --ir $IR/graph.json --out-dir $GEN --backend reference --target $TARGET --quant int8 \
  --repo-root $MB --build-dir $IR/kverify --io $IR/io.npz \
  --harness-dir harness --global-curated-dir $MB/kernels
```

`--backend reference` = the *source* of the baseline kernels (reference C); the curated
files override per-op when present + verified. Passing `--repo-root/--build-dir/--io/
--harness-dir` turns on the **spike-harness verify**: each curated kernel is swapped into
the full model, built for `spike_riscv64`, run on the V-spike, and its output compared to
the PyTorch/numpy golden in `io.npz`. On PASS the pick is written to
`$GEN/kernel_picks.json` as `source=curated`.

Expected log per curated op:
```
[conv2d_s8_pc/direct] reference + curated swap from .../rvv_f16_conv2d_s8_pc_direct.c
[conv2d_s8_pc/direct] verify curated at 7 shape(s) vs reference_impl
[conv2d_s8_pc/direct] curated verify PASS — spike-harness ok (max_abs_err=0 ...,
                       cycles={'conv2d_s8_pc': ..., 'linear_s8_pc': ..., ...})
```
The `cycles={...}` dict is the per-op spike cycle profile — this is how you measure a
kernel's speedup (compare the op's entry with curated vs reference in the pick).

## 5. Build the full model with the picked kernels + run

**The verify run (step 4) already wrote a deployable `$GEN/kernels.c`** with each
PASSing curated kernel swapped in (`grep -c vwmacc $GEN/kernels.c` to confirm; the
top-of-file `/* source: reference */` is just the baseline label — per-op curated
swaps are applied under it, and `kernel_picks.json` records `source=curated`). So
just build `$GEN` directly. Do **not** re-run `generate_kernels` without the verify
args to "finalize" — for a spike-harness backend it errors out (*"verify_method=
spike_harness ... requires --repo-root, --build-dir, --io"*) and writes nothing, so
you'd either error or, worse, regenerate a reference-only `kernels.c`.

```bash
west build -p -b spike_riscv64 harness --build-dir $IR/build \
  -- -DMODEL_DIR=$GEN -DMODELBLASTER_BACKEND=$TARGET \
     "-DMODELBLASTER_KERNEL_CFLAGS=-march=rv64gcv_zfh_zvfh;-mabi=lp64d"

PYTHONPATH=$MB/src $PY -m modelblaster.validation.spike_runner \
  --elf $IR/build/zephyr/zephyr.elf --io $IR/io.npz --spike $(which spike) --timeout 300
# int8 io.npz golden == int8 numpy sim => expect max_abs_err=0 (bit-exact).
```

## 6. Debugging a verify build failure

The verify swallows the root error. To see it, reproduce the build manually against the
completed `$GEN` (it has the skeleton + a picked/curated `kernels.c`):

```bash
west build -p -b spike_riscv64 harness --build-dir $IR/manual \
  -- -DMODEL_DIR=$GEN -DMODELBLASTER_BACKEND=rvv_f16 \
     "-DMODELBLASTER_KERNEL_CFLAGS=-march=rv64gcv_zfh_zvfh;-mabi=lp64d" 2>&1 | tee /tmp/b.log
grep -nE "error:|undefined reference|CMakeLists.txt:[0-9]" /tmp/b.log
```
- `CMakeLists.txt:48 ... missing required file` → skeleton not in `$GEN` (gotcha #2).
- `undefined reference to kernel_<op>` → mangling no-op (gotcha #1: fix out-dir layout).
- A standalone kernel compile check: `riscv64-zephyr-elf-gcc -march=rv64gcv_zfh_zvfh
  -mabi=lp64d -O2 -c kernels/rvv/<file>.c` (SDK gcc under
  `$ZCS/tools-manual/zephyr-sdk-*/gnu/riscv64-zephyr-elf/bin/`).

---

## 7. conv2d_s8_pc — algorithm study (how the methods compared)

conv is ~87% of this model's compute, so it got the most attention. All variants
are **bit-exact** (max_abs_err=0) vs the reference; only the spike cycle profile
differs. Measured on the pure-int8 per-channel `fused_full` (6 stride-2 / stride-1
convs, R=IC·KH·KW ∈ [25,576], OC ∈ [16,64]), `conv2d_s8_pc` entry of the verify
`cycles={}` dict:

| # | Method | conv cyc | vs ref | why |
|---|--------|---------:|:------:|-----|
| 0 | reference scalar triple-loop | 12,305,824 | 1.00× | baseline |
| 1 | OC-vectorized, **strided** `vlse8` over OC (native OCIHW) | 12,305,824 | 1.00× | strided/gather loads are charged **per element** → zero win |
| 2 | im2col patch + unit-stride **dot** per oc (`vredsum`) | 9,983,681 | 1.23× | one `vredsum` **per oc per pixel**; reduction overhead dominates short R |
| 3 | **OC-vectorized GEMV**: transpose W→[R,OC], `vwmacc` accumulate OC lanes | 7,653,261 | 1.61× | no per-oc reduction; but re-loads the whole weight set **per pixel** → memory-bound |
| 4 | #3 with weights **pre-widened to i16** in memory | 7,788,810 | 1.58× | **slower**: i16 load = 2× bytes; confirms loads charged per-byte |
| 5 | **#3 + pixel tiling TILE=4** (register-blocked over OW) | **5,981,395** | **2.06×** | each weight vector loaded once, reused across 4 pixels → weight byte-loads ÷4 ← **chosen** |
| 6 | #5 + scalar all-zero-tap skip | 6,466,933 | 1.90× | **slower**: per-r branch overhead > the rare padded-edge savings |

**Takeaways**

- **spike's cost model charges vector loads ~per byte** (#1 vs #3, and #4). Optimize
  for *bytes moved*, not instruction count. Strided/indexed loads are the worst —
  always arrange for unit-stride (that's why the weight transpose in #3 exists).
- The **arithmetic floor** is the `vwmacc` count = OH·OW·⌈OC/VL⌉·R, invariant across
  #3/#5/#6. Pixel tiling (#5) attacks the *other* term — weight reloads — cutting
  them by the tile factor. Est. floor ≈ 5.4M, so TILE=4 (5.98M) is within ~10% of it.
- **TILE=4 is register-bound**: 4 × i32m4 accumulators = 16 vector regs, + the i8/i16
  weight temporaries, fits the 32-register file. TILE=6–8 would need either >32 regs
  or a smaller LMUL (which halves VL and doubles weight-load passes, cancelling the
  tiling gain). Modelled benefit past TILE=4 is <0.3M cyc — not worth the register
  pressure / code size.
- **"iGEMM prepared ahead of time"** (precompute the full im2col matrix once, then a
  clean tiled GEMM) is numerically identical to #5 and would land at the same cycle
  count — it just relocates the same per-pixel gather to a prepass. It costs extra
  memory (OH·OW·R buffer) and doesn't lower the `vwmacc` floor, so it wasn't adopted
  here; it *would* help if the same im2col buffer were reused across multiple weight
  sets (not the case in a single forward). The weight transpose W[OC,R]→[R,OC] **is**
  the "prepared ahead of time" part we kept (amortized over all pixels).

**Where the remaining time goes / further ideas (not done):** the `vwmacc` floor is
fundamental to int8 conv on a vector unit; beating it needs a different compute
primitive — a systolic/matrix engine (Gemmini `conv2d_s8_pc`, or the Saturn OPU
matrix ext, both separate targets) rather than RVV element MACs. On RVV the only
remaining RVV-side levers are larger LMUL for the weight load (bandwidth) and fusing
the requant into the drain — both single-digit-percent.
