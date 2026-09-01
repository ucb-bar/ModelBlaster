# Comparing the two PyTorch→RISC-V/Zephyr flows

Two independent lowering paths in this repo take a PyTorch model to a
Zephyr RISC-V ELF that runs on spike. They were not directly comparable
before; `scripts/compare_flows.sh` makes them so for a shared model.

## The two flows

| | **ModelBlaster** (`modelblaster/`) | **ExecuTorch** (`samples/executorch/`) |
|---|---|---|
| Front end | model `.py` → `extract_graph` (own FX/int8 sim) | `torch.export` → `to_edge` → XNNPACK partition |
| Artifact | per-op C (`kernels.c`, `weights.c`, `model.c`) | serialized `.pte` (flatbuffer) → `model_pte.c` |
| Runtime | AOT: generated C `run_model()` calls kernels directly | interpreted: ExecuTorch runtime walks the `.pte`, dispatches XNNPACK-delegated subgraphs |
| Quant | fp32 / fp16 / **int8 PTQ** / mixed | fp32 / fp16 (int8 quantizer commented out in `gen_pte.py`) |
| Kernels | reference / hand-curated / LLM, per HW target | XNNPACK micro-kernels (RVV / Gemmini toggles) |
| Profiling | per-kernel `rdcycle` CSV + wall_clock | whole-program (added `EXECUTORCH_EXECUTE_CYCLES` rdcycle bracket) |
| Verify | bit-exact / tol vs golden baked in binary | prints `Output[i][j]`; compare to PyTorch offline |
| Models | mlp, lenet, mobilenet_v2, dronet, yolov8_nano, vint | torchvision (mobilenet/squeezenet/alexnet/…) + inline lenet |

**Why they weren't comparable:** disjoint model sets, disjoint quant
(MB int8 vs ET fp32), disjoint metric surfaces, disjoint runtime models.
The bridge is a model both support (**LeNet**, defined identically in
`modelblaster/models/lenet.py` and `samples/executorch/model/gen_pte_lenet.py`,
input `1×1×28×28`) run **fp32 on RVV** on both, comparing spike cycles,
ELF size, model-payload size, and each flow's output vs its own PyTorch ref.
Cycle counts are architecture-determined (dense conv/linear compute does not
branch on data), so they compare even though the two flows use independently
initialized weights.

## Running the comparison

```bash
source tools/miniforge3/etc/profile.d/conda.sh && conda activate zephyr
source scripts/set_envvars_sdk.sh
bash scripts/compare_flows.sh              # MODEL=lenet by default
# knobs: MODEL, MB_TARGET, ET_PYTHON, FLATC_DIR, SKIP_MB=1 / SKIP_ET=1,
#        SKIP_ET_BUILD=1 (reuse build/zephyr.elf; the pristine build is ~7 min)
```

## Measured — LeNet fp32 on spike (RVV), 2026-07-28

| metric | ModelBlaster | ExecuTorch 1.0.1 |
|---|---:|---:|
| runtime model | AOT per-op C kernels | interpreted `.pte` + XNNPACK |
| execute cycles (rdcycle) | 5,749,356 | 797,661,486 |
| zephyr.elf | 681,360 B | 105,231,840 B (static XNNPACK) |
| model payload | weights.c 713,619 B | model.pte 186,560 B |
| output[0][0] | (own ref) | 0.014086 (sane) |

### FireSim (real RTL, dual-rocket Saturn V256D128 + Gemmini-Q31 bitstream)

Run via the standard infrasetup+runworkload (post-reboot; `firesim_runner` for
ModelBlaster, `/tmp/run_et_firesim.sh` for ExecuTorch). LeNet int8:

**LeNet int8 execute cycles, CLEAN (no per-op profiling), FireSim:**

| flow | FireSim execute cycles |
|---|---:|
| ModelBlaster int8 | 866,429 (bit-exact PASS) |
| **ExecuTorch, MP_MAX_NUM_CPUS=1** | **~690,000 warm** (1.19M cold) |
| ExecuTorch, MP_MAX_NUM_CPUS=2 | ~120,400,000 warm |

**Two corrections to the earlier (wrong) FireSim numbers.** The 537M/645M figures
reported before were BOTH artifacts:

1. **Profiling pollution.** `ENABLE_XNNPACK_PROFILING` emits an `ET_LOG(">>,<op>,
   <cyc>")` per op every invoke over the HTIF console (~millions of cyc/write on
   FireSim), *inside* the `method->execute()` bracket. It added ~500M and, being
   constant, masked everything (both 1- and 2-core looked ~537M). Build
   `-DMB_XNN_PROFILE=OFF` for real timing.
2. **Core count.** Once profiling is off, `MP_MAX_NUM_CPUS` dominates: the
   XNNPACK pool is sized to `sysconf(_SC_NPROCESSORS_ONLN)`, so 2 cores → every
   op pays cross-hart threadpool dispatch (~120M total for LeNet), while 1 core
   runs every op INLINE (no dispatch) → ~690k. **~175x.**

**Corrected takeaway:** with the right config (1 core / inline pool, no
profiling), **ExecuTorch's XNNPACK RVV int8 is competitive with — even a hair
faster than — ModelBlaster** (~690k vs 866k) on real RTL. The huge gaps we saw
were measurement/config artifacts, not the kernels. For a tiny model, avoid
multi-core threadpool dispatch (MP_MAX_NUM_CPUS=1); parallelism only pays when
ops are big enough to amortize the per-op dispatch.

_(Earlier spike/profiled runs, kept for context: on spike, spin looked 86x
faster than vanilla — a functional-model artifact; on FireSim with profiling ON,
spin 537M / vanilla 645M / 1-core 537M all collapsed together because the
profiling HTIF logging dominated.)_ Per-op profiling
(rdcycle timer) on the FireSim spin run shows the **actual XNNPACK compute is
tiny**: Conv QC8 IGEMM ~106k, Max Pool ~107k/50k, FC-GEMM ~45k/34k/9k, Convert
~7k, Transpose ~9k → **~0.5M cycles of real compute**, competitive with
ModelBlaster's 866k. So the ~537M is ~99% **threadpool dispatch overhead**
(~90M cyc/op) for this tiny 6-op model, not compute.

**Takeaway:** ExecuTorch's XNNPACK RVV int8 kernels are genuinely competitive
with ModelBlaster's hand-written kernels at the op level (~0.5M vs ~0.87M cyc);
the order-of-magnitude gap is entirely the general runtime's per-op threadpool
dispatch, which dominates for small models. Neither threadpool variant is good
here — spin busy-burns, vanilla sleeps (a single-threaded pool would be best
for a model this small). This is the real, RTL-grounded comparison; the spike
numbers are functional-model artifacts.

**Vanilla on FireSim = 644,964,385 cycles** (clean infrasetup rc=0; a couple of
prior attempts hit flaky `infrasetup` rc=1 → ELF loaded into a bad target state
→ FESVR "bad syscall" crash in ~8s — a FireSim-infra flakiness after repeated
reprograms, resolved by retrying infrasetup until rc=0). So on real RTL vanilla
(645M) ≈ spin (537M) — same order, spin ~20% better — versus spike's misleading
86x. Bottom line, RTL-grounded: **ModelBlaster 0.87M ≪ ExecuTorch ~537–645M**,
but ExecuTorch's *compute* is ~0.5M — the 600x gap is per-op threadpool dispatch
overhead on a tiny model, near-identical for both pool variants.

### Substantial models — FireSim, int8, 1-core, CLEAN (no profiling)

LeNet is tiny (6 ops), so it stresses runtime overhead, not kernels. Extending
to bigger convnets confirms the finding where it matters — real compute:

| model (int8) | ModelBlaster rvv | ExecuTorch (MP=1) | ET/MB |
|---|---:|---:|---:|
| LeNet         | 866,429 (bit-exact) | ~690k warm (1.19M cold) | 0.80x |
| **DroNet**    | **15,868,769 (bit-exact PASS)** | **~13.67M warm** (15.26M cold) | **0.86x** |
| MobileNetV2   | — (int8 extract rejects ReLU6) | ~107.7M warm (116.6M cold) | — |

**DroNet is the apples-to-apples substantial int8 point** — both flows run the
*identical* architecture (ExecuTorch imports `modelblaster.models.dronet`).
ModelBlaster's curated `conv2d_s8_rvv_vsmul_vnclip` is bit-exact
(`max_abs_err=0`). ExecuTorch's XNNPACK RVV int8 comes out ~1.16x faster
(13.67M vs 15.87M) — the LeNet "ET a hair faster" result **holds and grows on a
real conv workload** (DroNet is ~18x LeNet's cost), so it is the kernels, not a
small-model fluke. MB's per-op profile: convs dominate (conv0 3.79M/23.9%,
conv8 2.17M/13.7%, conv1/2 ~1.78M each, conv5 1.72M) — 9 conv2d_s8 = ~91% of the
15.87M; adds/bn/pool/linear are the rest.

**Per-operator breakdown (plot: `plots/dronet_firesim_compare.png`).** From an
XNNPACK-profiling FireSim run (`MB_XNN_PROFILE=ON`; per-op `>>` lines are clean
rdcycle deltas even though that build's *total* is HTIF-logging-polluted, so the
full-model total above is from the profiling-off run). By category, MB (clean
in-binary rdcycle) vs ET (warm-averaged):

| category | ModelBlaster | ExecuTorch |
|---|---:|---:|
| Convolution | 14.08M | 4.04M |
| MaxPool | 0.76M | 2.04M |
| ReLU / Clamp | 0.009M | 0.53M |
| Residual Add | 0.92M | 0.05M |
| BatchNorm | 0.10M | 0 *(fused into conv)* |
| Transpose (NCHW↔NHWC) | 0 | 1.09M |
| Convert (quantize) | 0 | 0.34M |
| Reshape / Setup | 0 | 0.53M |
| Dispatch / other runtime | 0 | 4.96M |

The kernels tell opposite stories that nearly cancel: **ExecuTorch's XNNPACK
convolutions are ~3.5x faster than ModelBlaster's** (4.04M vs 14.08M — better
GEMM microkernels), but ExecuTorch spends the saving back on overhead
ModelBlaster structurally avoids — **layout Transposes** (XNNPACK runs NHWC
inside an NCHW graph), **quantize Converts**, **Reshape/Setup**, a slower
MaxPool/Clamp, and ~5M of **ExecuTorch interpreter/delegate dispatch** across
the 6 XNNPACK segments (the gap between the ~8.7M profiled op-sum and the 13.67M
clean total). ModelBlaster stays NCHW-int8 end-to-end with hand-curated kernels,
so its 15.87M is ~89% raw convolution and essentially zero format/dispatch
overhead. Net on real RTL: ET 1.16x faster overall, but for very different
reasons per stage.

**MobileNetV2 is ExecuTorch-only** here: ModelBlaster's int8 extractor doesn't
support `ReLU6` (`NotImplementedError ... ReLU6 at features_0_2`), so it can't be
the cross-flow model. As a stand-alone ET substantial data point, 1-core int8 =
~107.7M warm — and this is REAL compute (big ops, not threadpool-dispatch-bound
like LeNet), which is why 1-core already costs 100M+ where LeNet was 690k.

**MobileNetV2 2-core FAULTS (deterministic).** Built with MP_MAX_NUM_CPUS=2, it
dies during setup with `mcause 5 Load access fault, mtval 0xffffff8100000020`
(a corrupted pointer) inside `xnn_pack_qs8_gemm_goi_w` weight-packing
(`unaligned_int32_t::operator int()`, packing.cc:52) — right after the planned
buffer alloc, on **both** independent runs with identical mtval/PC. The
*identical* PTE runs clean on 1-core (107.7M), so this is an SMP-specific bug in
XNNPACK's packing path under the 2-hart pool, not a model or transient-FPGA
issue. Combined with the LeNet ~175x 2-core slowdown, the guidance is firm:
**run ExecuTorch/XNNPACK with MP_MAX_NUM_CPUS=1 on this SoC** — multi-hart
neither helps (small models: dispatch dominates) nor is even stable (big models:
packing faults).

### 2x2 matrix — LeNet execute cycles on spike (after threadpool fix)

| quant | ModelBlaster | ET vanilla pthreadpool | ET spin pthreadpool |
|---|---:|---:|---:|
| fp32 | 5,749,356 | 799,609,343 | 21,873,597 |
| int8 | 863,996 | 599,652,118 | 6,981,533 |

With the spin threadpool, ExecuTorch is ~3.8x (fp32) / ~8.1x (int8) ModelBlaster
— a believable general-runtime-vs-hand-tuned gap. The vanilla→spin jump (~37x
fp32, ~86x int8) is the per-op tick-sleep described below.

**Do not read the raw vanilla cycle ratio as a verdict.** The executorch 1.0.1
build produces a *sane* output (0.014) — it really ran the graph — but ~797M
cycles for a LeNet (~0.4 MFLOP) is ~140× ModelBlaster and far too high for a
vectorized run. Almost certainly XNNPACK's RVV micro-kernels are **not being
selected at runtime**: XNNPACK picks kernels via cpuinfo hardware detection,
which can't probe the ISA on bare-metal Zephyr, so it falls back to scalar
reference micro-kernels. (The earlier 0.5 build reported 5.27M cycles but with
a *garbage* output — it was almost certainly not doing the real work, so that
apparent "~9% apart" was an artifact.) A trustworthy cycle comparison needs
XNNPACK's RISC-V vector path confirmed as engaged (force the microkernel ISA /
fix the cpuinfo probe), and identical inputs fed to both flows. Until then the
solid results are: **both flows build and run the same model to completion on
spike from one env, and ModelBlaster is ~140× fewer cycles** — consistent with
hand-tuned RVV kernels vs an unvectorized general runtime.

## ExecuTorch build — single-env setup (executorch 1.0.1)

The repo was migrated so ONE env — the local `tools/miniforge3/envs/zephyr`
(torch 2.9) — does ModelBlaster AND the full ExecuTorch flow. The pieces:

0. **Env (torch 2.9 line).** executorch 1.0.1 pins `torch>=2.9,<2.10`, and
   ModelBlaster only needs `torch>=2.4`, so the env moved 2.10→2.9:
   `pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.0
   torchvision==0.24.0 && pip install executorch==1.0.1 torchao==0.14.0
   "cmake>=3.29"`. (executorch 1.1.0 supports torch 2.10 but the ucb-bar
   RVV/Zephyr XNNPACK fork isn't ported past 1.0.1.) DroNet int8 re-verified
   bit-exact on torch 2.9 + numpy 2.x.

1. **Submodule on `zephyr-1.0.1`** (executorch 1.0.1a0). Unlike the old 0.5
   `zephyr` branch, 1.0.1 self-contains codegen (`python -m codegen.gen` →
   `executorch.codegen`), so it needs NO `torchgen.gen_executorch` from torch
   (which torch≥2.9 dropped) and NO buck2 source-gen. Nested submodules to init
   (beyond the XNNPACK stack + flatbuffers/flatcc/gflags/prelude/googletest):
   **`third-party/{json,pocketfft,ao}`** — `json` is required by
   `third-party/CMakeLists.txt`. XNNPACK stays on the ucb-bar fork (RVV kept).

2. **cmake ≥ 3.29** (executorch 1.0.1 requires it). `/usr/bin/cmake` is 3.28
   and the activated env otherwise resolves to the stale Vitis cmake 3.3.2, so
   `pip install "cmake>=3.29"` into the env (gives 4.4.0) and put the env bin
   FIRST on PATH. Also: the XNNPACK microkernels target was renamed
   `microkernels-prod` → `xnnpack-microkernels-prod` (fixed in the sample
   CMakeLists link line).

3. **Target ABI flags for the ExecuTorch subbuild.** ExecuTorch/XNNPACK are
   added via `add_subdirectory` and do NOT link Zephyr's `zephyr_interface`,
   so they compile with the SDK toolchain's default multilib (**rv32
   soft-float**) and fail the final link against Zephyr's rv64/lp64d objects
   (`file in wrong format`, `ELFCLASS32 incompatible with ELFCLASS64`, and
   `R_RISCV_CALL_PLT out of range`). Fixed in the sample CMakeLists by forcing
   `-march=rv64imafdc -mabi=lp64d -mcmodel=medany` into `CMAKE_C/CXX_FLAGS`
   before `add_subdirectory`. This is the "patched RISC-V toolchain" the
   README alludes to — the root issue is flag propagation, not the compiler.

4. **Enable the RISC-V vector context in Zephyr.** XNNPACK's RVV micro-kernels
   read `vlenb` and issue vector ops, but the runner's `prj.conf` did not enable
   the V extension, so the first vector CSR read trapped at boot (`mcause=2`
   Illegal instruction, `mtval=0xc22022f3` = `csrr vlenb`). Added to
   `executor_runner/prj.conf`: `CONFIG_RISCV_ISA_EXT_V=y` +
   `CONFIG_RISCV_ISA_EXT_V_LAZY=n` (eager V save/restore, matching the
   ModelBlaster RVV harness).

Run: `spike -p4 --isa=rv64gcv_zicntr build/zephyr/zephyr.elf`
(the runner prints `EXECUTORCH_EXECUTE_CYCLES=…` and `Output[i][j]: …`).

## Operator-level profiling & RVV validation (spike)

**ModelBlaster** already emits a per-kernel cycle CSV (`rdcycle` around each
kernel; see the table in any run). LeNet fp32: conv2 3.32M (58%), conv1 1.92M
(33%), fc1 0.25M — convs dominate, total 5.75M.

**ExecuTorch** operator profiling is enabled via `ENABLE_XNNPACK_PROFILING`
(`add_compile_definitions` in the sample CMakeLists), which creates the XNNPACK
runtime with `XNN_FLAG_BASIC_PROFILING` and logs `>>, <op>, <cycles>` per op
every invoke (`XNNProfiler::end()→log_operator_timings`). Two fixes were needed
in the fork: (a) `XNNPACKBackend.cpp` called a non-existent
`executor->print_avg_op_timings()` — neutralized (per-op logging already
happens per invoke); (b) XNNPACK's `xnn_read_timer` used tick-granular
`clock_gettime` (~100µs floor on Zephyr) — patched to `rdcycle` on `__riscv`
(`src/runtime.c` + `src/xnnpack/subgraph.h`) so op timings are in cycles.

**RVV validation:** the ELF contains the RVV microkernels
(`xnn_f32_gemm/igemm/dwconv..._rvv`, 3767 vector instrs) AND scalar ones;
`XNN_ENABLE_RISCV_VECTOR=1` is defined, so `hardware-config.c` sets
`xnn_arch_riscv_vector` and `gemm-config.c`/`igemm` register the **`_rvv`**
ukernels. At runtime the conv executes as `Convolution (NHWC, F32) IGEMM`
(the rvv-registered path). So RVV is engaged — the LeNet slowness is **not** a
scalar fallback.

**Where ExecuTorch's ~600–800M cycles went — the threadpool.** Per-op profiling
showed a recurring **~100M cycles on _every_ operator** regardless of op (a
`Max Pooling` or a `Transpose` cost the same ~99,975,000). Root cause: XNNPACK
parallelizes every op through **pthreadpool**, whose wait loop spins
`PTHREADPOOL_SPIN_WAIT_ITERATIONS` (=1000) then falls to a **futex wait**; on
bare-metal Zephyr that futex blocks until the next **100 Hz system tick
(~10 ms ≈ 100M cycles)**. The pool is sized to `sysconf(_SC_NPROCESSORS_ONLN)`
= 4 (SMP), so N ops × ~1 tick dominates (6 ops→600M int8, 8 ops→800M fp32).
This is the "setup problem" — not missing RVV.

**Threadpool variant toggle (the fix + the comparison).** `modelblaster`'s
runtime deliberately avoids this with a lightweight spin rendezvous
(`modelblaster/runtime/modelblaster_pool` + the `spin_overrides/` microbench:
pthreadpool ~13M cyc/dispatch vs raw spin ~20k). We can't drop
`modelblaster_pool` into XNNPACK (it implements only `parallelize_1d`; XNNPACK
uses 30+ `parallelize_*` variants), so the sample CMakeLists exposes
`-DMB_XNNPACK_SPIN_THREADPOOL=ON` which brings the *property* — bump the spin
ceiling (via the microbench's `spin_inject.h` `-include` override) so the pool
spins/`sched_yield`s instead of sleeping to a tick. Default OFF = vanilla
pthreadpool (baseline). Measured LeNet int8 on spike: **vanilla 599.7M → spin
6.98M cycles (~86×)**, identical output. Per-op then reads sensibly (Max Pool
65k, FC-GEMM 61k/52k/7k, Convert ~6k, Transpose 7k).

**Important:** spike is a *functional* ISA model — absolute cycle counts
(esp. the tick-sleep blow-up) don't reflect real pipeline/memory timing. The
vanilla-vs-spin gap is a Zephyr-scheduler artifact that FireSim (real RTL +
real timing) will render differently — which is exactly why the FireSim phase
matters.

## FireSim status (blocked — needs the matching chipyard config)

FireSim here is a real Alveo U250 (FPGA present: Xilinx `903f`, `/dev/xdma0_*`)
with prebuilt bitstreams incl. `alveo_u250_firesim-dual-rocket-saturn-gemmini-q31`
(Saturn V256D128 RVV) — the ideal RVV target for both flows. But `firesim
enumeratefpgas`/`infrasetup` force a **driver rebuild**, which fails: FireChip's
`TargetConfigs.scala` references `chipyard.REFV256D128DualRocketGemminiQ31Config`
(and ~11 sibling Saturn/REF configs) that are **no longer in the chipyard
package** — `generators/chipyard/.../SaturnConfigs.scala` was stripped (−370
lines) by an in-progress "FP-precision-stripping / OPU" refactor. The config
that matches the prebuilt bitstream (built 2026-05-06) is not in HEAD, any
recent commit, or the stash — it lived only in an uncommitted working-tree
state that has since been overwritten. So the driver for the RVV bitstream
can't be re-elaborated, and no git state cleanly restores it. Resolving this
needs the user (restore/point at the matching SaturnConfigs.scala, or rebuild
the bitstream from the current configs). The compiled driver + bitstream from
the last good build DO exist under `sims/firesim/output/.../`.

**RESOLVED (driver rebuild) via non-destructive artifact-touch.** The cached
driver/`.fir`/generated-src/`firechip.jar` all match the prebuilt bitstream
(built together in May). `make` only re-elaborated because the modified
SaturnConfigs.scala looked newer. `touch`-ing the build artifacts newer than
sources makes `make` report "Nothing to be done" and reuse the
bitstream-matching driver — no source/config change, survives reboot.

**Correct run path = the shared job queue.** This is a shared single-FPGA host
(`/scratch2/agustin/firesim_queue`, per-user fair share, daemon holds
`fpga.lock`). Run via `firesim-queue runworkload-full --chipyard
/scratch2/dima/chipyard-fsim --workload zephyr --stage-from <elf>` (see
`scripts/firesim_submit.sh`). NEVER run raw `firesim infrasetup` outside the
queue: doing so contended with the daemon and its FPGA-reprogram
(`firesim-fpga-util.py --bitstream`) dropped the Alveo off PCIe, needing a
reboot. `firesim enumeratefpgas` is likewise off-limits (rewrites the FPGA DB).

**Staged and ready** (post-reboot, all three built for the RVV Q31 bitstream):
`modelblaster/examples/lenet/int8/build/rvv_firesim/zephyr/zephyr.elf`,
`/tmp/fs_elfs/et_int8_{spin,vanilla}.elf` (ExecuTorch on
`chipyard_riscv64/rocketchip_virt_riscv64` + `firesim.conf` overlay). The only
remaining blocker: `queue.db` is `agustin:agustin` 644 and the daemon is down —
needs agustin to restart the daemon + grant write access, then
`scripts/firesim_submit.sh` collects the real-RTL cycle numbers for all three.

## Status / caveats

- **int8 vs int8** is not yet comparable: ExecuTorch's int8 path here is
  commented out (`gen_pte.py`), and the quantized portable-ops codegen needs
  the same pinned torchgen. fp32-vs-fp32 is the apples-to-apples axis today.
- ModelBlaster fp32 LeNet fails its *default* tolerance verify (near-zero
  logits inflate relative error); the int8 path is bit-exact after the RVV
  conv weight-layout fix (see `project_dronet_int8_rvv_verify_fail`).
- ExecuTorch inputs default to the runner's `prepare_input_tensors` fill; the
  harness saves the PyTorch reference IO (`lenet_io.npz`) for offline output
  comparison but does not yet feed identical inputs into the ELF.
