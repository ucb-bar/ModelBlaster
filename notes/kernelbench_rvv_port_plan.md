# Porting KernelBench-on-RVV into the current (submodule) structure

Goal: run KernelBench **level1** benchmark problems on **RVV targets** (spike
`spike_riscv64` first, then FireSim Saturn), through the current
`modelblaster/` submodule flow — extract → skeleton → generate_kernels →
build → run → compare-vs-PyTorch.

This ports the prototype from the stale local `kernelbench` branch (parent repo,
`agents/…` layout, 86 commits behind `dev`, never pushed, last touched
2026-04-30) into the current `modelblaster/` layout, and extends it from the
branch's *scalar/reference* first cut to **RVV**.

## What already exists on `main` (do NOT re-port)
- KernelBench **Phase-2 activation ops** in `pipeline/reference_kernels.py`
  (`leaky_relu, tanh, swish, gelu, gelu_exact, selu, hardsigmoid, softplus,
  softsign, hardtanh`) + their verify/skeleton wiring.
- Multi-model bundling: `harness_multi/` + `pipeline/generate_multi_main.py`.
- fp32 extraction (`--quant fp32`) and the `rvv` backend (fp32 `rv64gcv`,
  `--isa=rv64gcv_zicntr`) with reference impls that compile for RVV.

## What must be ported (orphaned on the branch)
| Branch artifact (`agents/…`) | Port target (`modelblaster/…`) | Notes |
|---|---|---|
| `pipeline/extract_graph.py`: `_load_kernelbench()`, `--bench-file`, `--bench-max-elements`, functional `sigmoid`/`elu` handlers | `pipeline/extract_graph.py` | **Re-port logic, not the diff** — file evolved (fusion passes, IHWO, ReLU6). |
| `examples/kernelbench/run_all.sh`, `run_one.sh` | `examples/kernelbench/` | Adapt to the submodule `_run_lib.sh` (REPO_ROOT retarget already in local WIP). |
| `validation/runner_common.py` (`io_paths` override), `validation/spike_runner.py` (`--io-path NAME=PATH`) | same files under `modelblaster/validation/` | ~28 lines total. |
| `examples/_run_lib.sh` (+19) | already adapted locally; reconcile | Only if the bench flow needs the deltas. |
| **level1 corpus** (100 `.py`) — **NOT committed** | new `bench/level1/` (vendored) or via a submodule | Source from `ScalingIntelligence/KernelBench`; or check if the `third_party/KernelBlaster` submodule vendors them. |

## RVV / quant decision (central)
KernelBench models are **fp32**; the curated RVV kernels here are all **int8
(`_s8`)**. So the RVV path is:
- **Baseline**: `--quant fp32 --target rvv --backend reference` → reference C
  impls compiled with `-march=rv64gcv` (auto-vectorized). Gives a running,
  correct RVV binary + cycle baseline for every covered op.
- **Optimized (the KernelBench point)**: `--backend llm` → LLM-generated RVV
  fp32 kernels, verified vs the reference golden, measured on RVV. This is what
  KernelBench actually evaluates (kernel-gen quality).
- **Optional int8 axis** (later): `--quant int8` reuses the existing curated
  int8 RVV kernels, but KernelBench goldens are fp32 so this is a separate,
  lower-priority track (would need per-op int8 tolerance).

## Plan (phased)

### Phase 0 — corpus + triage (prep, ~2–3 days)
1. Vendor the level1 corpus (decide: `bench/level1/` in-repo vs. the
   `KernelBlaster` submodule — **first check whether KernelBlaster already
   supersedes this effort** to avoid duplicate work).
2. Triage the 100 ops into: (a) **runnable now** on RVV via existing ops +
   reference impls (single-input, no matmul/loss — the branch's Phase-1 scope);
   (b) needs a new op/handler; (c) out of scope (multi-input matmuls, losses).
   Expect ~30 in bucket (a) for the first cut.

### Phase 1 — loader (`extract_graph`, ~2–3 days)
3. Re-port `_load_kernelbench` + `--bench-file` + `--bench-max-elements` onto
   the current `extract_graph.py`; add functional handlers not already present.
4. Unit-check: `--bench-file <op>.py --quant fp32 --out-dir …` emits
   `graph.json`/`weights.npz`/`io.npz` with a PyTorch-matching golden, host-side.

### Phase 2 — RVV run harness (~3–4 days)
5. Port `run_all.sh`/`run_one.sh` to `examples/kernelbench/`, wired for
   `TARGET=rvv BACKEND=reference RUNNER=spike`; bundle via `generate_multi_main`
   + `harness_multi`; board `spike_riscv64` (has V).
6. Port `--io-path` into `spike_runner.py`/`runner_common.py`.
7. Green-light the branch's original 6-bench suite on **RVV** (reference),
   bit-exact / within-tolerance vs PyTorch.

### Phase 3 — coverage to ~30 ops on RVV  ✅ DONE (2026-07-29)
8. Walked bucket (a): **34/34 extractable level1 benches PASS** on RVV/spike
   fp32 (reference backend), covering **25 distinct ops** (relu family, gelu,
   selu, softplus, softsign, batchnorm2d, frobenius/l2 norm, maxpool2d,
   sum/mean/max/min/argmax/argmin reductions, conv2d, conv2d_dw). Two layout
   bugs fixed: fp32 `conv2d` and `conv2d_dw` reference impls now honor the
   rvv IHWO weight packing (`MODELBLASTER_RVV_IHWOC_WEIGHTS`). `42_Max_Pooling`
   dilation passes on current main.
9. Per-op PASS + spike cycles recorded in
   `examples/kernelbench/results/rvv_fp32.md` (auto) +
   `results/rvv_fp32_report.md` (cycles + triage).
   Harness: `run_all.sh` gained a `JOBS=N` parallel mode (isolated per-bench
   dirs; 34 benches in ~1:45 at JOBS=8 vs ~15 min sequential; JOBS forced to 1
   on firesim). Env pinned to this working copy via scratch `mbenv.sh`;
   `_run_lib.sh` exports `PYTHONPATH=<parent>` for the submodule layout.

### Phase 3.5 — extend op coverage (2026-07-30) — 67/100 extractable, 66 PASS
Beyond the first 34 single-input benches, extended the loader + op set.
Final full-suite spike run: **66 / 67 PASS** (only 10_3D fails — a 3D×2D
broadcast matmul the plain matmul op computes wrong).

Commits (in order):
- **Matmul family** (`babc532`): relaxed `_load_kernelbench` to return
  multi-input `forward(A,B)`; extract()'s `packed_inputs` path already handled
  it. 12/18 matmul benches PASS (matmul, matmul_ta/tb/tatb, bmm).
- **softmax / log_softmax / avgpool2d / layer_norm** (`926a9d2`): unlocks 23,
  24, 45, 40.
- **group_norm + rms_norm** (`8325630`): one group_norm kernel covers
  nn.GroupNorm and nn.InstanceNorm2d (G==C); rms_norm via compound-fusion
  matcher. Unlocks 34, 35, 36.
- **conv_transpose2d** (`243bd35`): general gather form (stride/pad/
  output_pad/groups/dilation), IHWO-aware weight. 7/7 ConvTranspose2d benches.
- **1D conv/pool** (`4308eab`): Conv1d/ConvTranspose1d/MaxPool1d/AvgPool1d map
  to 2D kernels with a unit height dim. 6 benches (dilated Conv1d 76 deferred).
- **3D conv/pool** (`1a8f4da`): conv3d, conv_transpose3d, maxpool3d, avgpool3d
  (new 5D NCDHW kernels; 5D weights stay OIDHW — no repack). 13/13 benches.

- **Matmul variants** (`8b0f599`): triu/tril mask ops, diag_matmul fusion
  (diag(A)@B row-scale), and N-D@2D matmul + einsum '...l,lk->...k'. Unlocked
  11/12/14/15 and fixed 10_3D.

Extractable corpus: **34 → 84 / 100** (84/84 PASS after the matmul-variant batch).

Still out (16): losses (7); cumsum/cumprod/flip/select (5); abs(L1Norm 38);
dilated conv2d (76, 80); SDPA (97, hardcodes cuda).

Note on losses: a scalar output is NOT a blocker — the harness flattens outputs
and compares element-wise, so a size-1 result works. Losses are deferred purely
by effort: each needs a new reduction-to-scalar op (mse_loss, huber, kldiv) or a
compound (cross_entropy = log_softmax + NLL), and they're multi-input (preds +
targets) — which the packed_inputs path already supports. Implementable, not
architecturally out of scope.

### Phase 4 — optimized RVV + real RTL (~1 week)
10. `BACKEND=llm` on the covered ops → LLM RVV kernels, verified + measured
    (reference-vs-llm speedup per op) — the actual KernelBench metric.
11. Re-measure the suite on **FireSim** (real Saturn cycles) via
    `firesim_runner` for the headline numbers.

## Effort
Matches the branch's own estimate: **~3–4 weeks for ~30 ops**, 6–8 weeks for
all 100. The pipeline doesn't fight you; the work is mostly mechanical
(handlers + op coverage).

## Risks / decisions to make first
- **KernelBlaster overlap**: `third_party/KernelBlaster` (declared, not
  checked out) may be the intended successor — evaluate before re-porting.
- **fp32 vs int8 on RVV**: fp32-reference is the correct baseline; "real" RVV
  perf needs LLM/curated kernels (no curated fp32 RVV kernels exist today).
- **extract_graph drift**: 85 upstream commits + fusion passes + IHWO — port
  the loader logic carefully and re-verify each op end-to-end.
- **Memory**: stock KernelBench shapes overflow spike's 256 MB when bundled;
  `--bench-max-elements` (halving) is already designed for this.
- **Submodule-layout WIP**: the `_run_lib.sh` REPO_ROOT retarget is currently
  local uncommitted WIP — the bench harness depends on it resolving paths.
