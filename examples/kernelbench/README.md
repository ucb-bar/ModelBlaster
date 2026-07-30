# KernelBench level1 on RISC-V via ModelBlaster

Run the [KernelBench](https://github.com/ScalingIntelligence/KernelBench) level-1
single-operator benchmarks on RISC-V hardware/simulators through the ModelBlaster
pipeline. Each PyTorch op is lowered to a self-contained Zephyr ELF, executed on
spike / FireSim / native, and its output is verified bit-exact (fp32) or within
tolerance (fp16) against a PyTorch golden.

**Status:** 100/100 level-1 problems PASS on RVV/spike in **fp32** and **fp16**
(reduced dims, `BENCH_MAX_ELEMENTS=65536`). Stock (full) dims validated on
`native` and on spike with an enlarged RAM region. See `results/`.

---

## 1. How the flow works

```
bench/level1/<n>_<Op>.py                     (PyTorch nn.Module benchmark)
        │  extract_graph.py --bench-file  (loads Model+get_inputs, runs it,
        ▼                                  captures graph.json + weights + io.npz golden)
   examples/kernelbench/kb_<name>/<quant>/generated/
        │  generate_skeleton.py            (model.c, kernels.h, test_io.S)
        │  generate_kernels.py <backend>   (per-op kernel impls for the target ISA)
        ▼
   west build -b <board> harness/          (links harness + model + kernels → zephyr.elf)
        │
        ▼
   spike / firesim / native runner          (runs ELF, parses printed output)
        │  validation/<runner>_runner.py
        ▼
   compare vs io.npz golden  →  PASS / FAIL (max_abs_err)
```

The whole per-op sequence is driven by `examples/_run_lib.sh`; the three wrapper
scripts here just set env and call into it.

* **`run_one.sh`** — one op → one ELF (cleanest per-op cycle count).
* **`run_all.sh`** — a set of ops, one ELF each, optional `JOBS` parallelism, writes a results table.
* **`run_multi.sh`** — *fuses* N ops into **one** ELF + **one** run. On FireSim this
  amortizes the (slow) per-problem `infrasetup`/`runworkload` over N problems.

---

## 2. One-time prerequisites

These are bootstrapped by the repo setup and are **not** part of the per-run flow
(a teammate sets them up once):

* The repo + submodules initialized (`modelblaster`, `zephyr_ws/zephyr`, third-party).
* Zephyr **SDK 1.0.0-beta1** (required for the RVV intrinsics; do not downgrade).
* The `zephyr` conda env under `tools/miniforge3/envs/zephyr` (provides `python`, `west`, `cmake>=3.20`).
* KernelBench level-1 sources present at `modelblaster/bench/level1/*.py` (committed).

## 3. Per-session environment

Run from the **parent repo root** (`zephyr-chipyard-sw/`). This exact recipe is
what the working runs use:

```bash
cd /path/to/zephyr-chipyard-sw
REPO=$PWD
source scripts/set_envvars_sdk.sh                        # SDK + west workspace env
export PATH="$REPO/tools/miniforge3/envs/zephyr/bin:$PATH"
export WEST_PYTHON="$REPO/tools/miniforge3/envs/zephyr/bin/python"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"                 # so `modelblaster` imports resolve
```

> ⚠️ **Disk / `TMPDIR`.** On this box `/` (which holds `/tmp`) runs near-full.
> `cc1`, FireSim bitstream extraction, etc. write to `/tmp` and fail with ENOSPC
> (FireSim shows `tar xvf ... rc=2`). Redirect temp to a roomy disk:
> `export TMPDIR=/scratch/$USER/tmp` (create it first). `/scratch` has TBs free.

---

## 4. Quick start

### One op
`BENCH_FILE` must be an **absolute** path (`_run_lib.sh` chdir's before extract, so
a relative path won't resolve). `run_all.sh`/`run_multi.sh` already pass absolute paths.
```bash
BENCH_FILE=$PWD/modelblaster/bench/level1/19_ReLU.py \
  TARGET=rvv QUANT=fp32 RUNNER=spike \
  bash modelblaster/examples/kernelbench/run_one.sh
```
Look for a `PASS` line and `max_abs_err=...`.

### A set of ops (parallel)
```bash
# spike is single-thread per run, so fan out with JOBS. NEVER JOBS>1 on firesim.
BENCHES="19_ReLU,31_ELU,42_Max_Pooling_2D,63_conv_standard_2D__square_input__square_kernel" \
  TARGET=rvv QUANT=fp32 RUNNER=spike JOBS=4 \
  bash modelblaster/examples/kernelbench/run_all.sh
# → results/rvv_fp32.md  (with no BENCHES it runs a 6-op default set)
```

### All 100, fp16
```bash
BENCHES="$(cd modelblaster/bench/level1 && ls *.py | sed 's/\.py$//' | paste -sd,)" \
  TARGET=rvv QUANT=fp16 RUNNER=spike JOBS=6 \
  bash modelblaster/examples/kernelbench/run_all.sh          # → results/rvv_fp16.md
```

### Fused: N ops, one ELF, one run (best for FireSim)
```bash
export TMPDIR=/scratch/$USER/fs_tmp                          # see disk note above
BENCHES="19_ReLU,31_ELU,63_conv_standard_2D__square_input__square_kernel,95_CrossEntropyLoss" \
  TARGET=rvv QUANT=fp32 RUNNER=firesim FORCE_EXTRACT=1 FIRESIM_TIMEOUT=1800 \
  bash modelblaster/examples/kernelbench/run_multi.sh
```
Each model prints its own name-tagged block + cycle count; the runner compares
each against its `io.npz` golden (passed via `--io-paths`, since the flat
`kb_<name>` tag doesn't map to an `examples/<tag>/` dir).

---

## 5. Environment knobs

| Var | Default | Meaning |
|---|---|---|
| `BENCH_FILE` | — | (run_one) level-1 `.py` path. Required. |
| `BENCHES` | 6-op set | (run_all/run_multi) comma-list of level-1 basenames (prefix match OK). |
| `TARGET` | `rvv` | ISA backend family: `rvv`, `scalar`, `gemmini`, `gemmini_q31`. |
| `QUANT` | `fp32` | `fp32` or `fp16`. `TARGET=rvv`+`QUANT=fp16` → `rvv_f16` backend. |
| `RUNNER` | `spike` | `spike`, `firesim`, or `native`. |
| `BACKEND` | `reference` | `reference` (curated kernels) or `llm` (generated). |
| `BENCH_MAX_ELEMENTS` | `65536` | Shrink largest module dim so io fits spike RAM. `0` = **stock dims**. |
| `CMODEL_LARGE` | `0` | `1` = build with the RISC-V **large** code model (`CONFIG_RISCV_CMODEL_LARGE=y`). Lifts the ±2 GiB medany PC-relative (`R_RISCV_PCREL_HI20`) span limit so stock-dim baked io >2 GB can link. Needs SDK gcc ≥14. Verified PASS on spike + FireSim. |
| `JOBS` | `1` | (run_all) concurrent benches. Forced to 1 on firesim (shared FPGA). |
| `FORCE_EXTRACT` | `1` (multi/all) | Re-run extract even if `generated/` exists. |
| `MB_KB_DATA_ROOT` | `/scratch/dima/mb_kb` | Symlink target for the heavy per-bench dir (keeps it off the repo partition; parallel-safe). Set empty to disable. |
| `SPIKE_HARTS` | `4` | spike `-p` core count. |
| `SPIKE_RAM_SIZE` | — | Hex 32-bit byte count for the `ram0` DTS region (base stays `0x80000000`). Needed for stock dims on spike. |
| `SPIKE_MEM_MB` | — | spike `-m` memory size (MiB). Must be ≥ `SPIKE_RAM_SIZE`. |
| `SPIKE_TIMEOUT` | `1200` | spike run timeout (s). |
| `FIRESIM_TIMEOUT` | runner default | FireSim run timeout (s). |
| `TMPDIR` | system | **Set to `/scratch/...`** — see disk note. |

---

## 6. Runners

| Runner | What | Use it for | Limits |
|---|---|---|---|
| **spike** | RISC-V functional sim | default; fast iteration, exact cycle-ish counts | compute-bound (~100–300 MIPS). Reduced dims fit the default 256 MB `ram0`; bump with `SPIKE_RAM_SIZE`/`SPIKE_MEM_MB` for stock. |
| **firesim** | FPGA-accelerated cycle-exact | real cycle counts; fused `run_multi.sh` runs | shared single FPGA → `JOBS=1`; needs `infrasetup`/`runworkload`; set `TMPDIR=/scratch`. |
| **native** | host x86 `native_sim` binary | **stock (full) KernelBench dims** with no RAM limit | scalar `TARGET` only; not a RISC-V measurement. |

**Stock-dim ceilings** (baked-io bare-metal RISC-V): io > ~2 GB hits
`R_RISCV_PCREL_HI20` relocation truncation; io > 4 GB hits the GNU `ar`
archive-member limit. Those problems only run on `native`.

---

## 7. Backends

Registered in `modelblaster/pipeline/backends.py` (resolve with `backends.get('<name>')`):
`scalar`, `rvv` (`-march=rv64gcv`), `rvv_f16` (`_zfh_zvfh`), `scalar_f16`,
`gemmini`, `gemmini_q31`, plus `rvv_hetero` / `rvv_opu`. `QUANT=fp16` appends
`_f16` to the `TARGET` automatically.

---

## 8. Results & docs

* `results/<target>_<quant>.md` — auto-generated PASS/FAIL tables from `run_all.sh`.
* `../../notes/kernelbench_rvv_port_plan.md` — porting plan, phases, op coverage.
* `../../notes/conv_kernel_gap_analysis.md` — why XNNPACK int8 conv outpaces the MB kernel.

## 9. Adding / debugging an op

1. Extraction fails (`NotImplementedError`): add an extract handler + a `KernelSpec`
   in `pipeline/reference_kernels.py`, and a call-emitter in `generate_skeleton.py`.
2. fp16: `_make_f16_variant()` auto-derives `<op>_f16` from the fp32 spec — usually nothing to do.
3. Verify FAIL with a conv: check weight layout — RVV conv weights are repacked to
   IHWO (`MODELBLASTER_RVV_IHWOC_WEIGHTS`); 5-D (conv3d) and fp16 conv weights stay OIHW.
4. Inspect a single run's log at `/tmp/kb_<bench>.log` (written by `run_all.sh`).
