# harness_shared_input — design draft

## Purpose

Run **N kernel implementations of the SAME KernelBench problem** in ONE
Zephyr boot. Shares the input tensor + weights + golden in rodata
(baked once via `test_io.S`), allocates per-variant output buffers in
ram0, invokes each variant's `run_model_<mid>_<tag>()` in sequence,
verifies against the shared golden, and emits per-variant
`MODELBLASTER_WALL_CYCLES [<mid>@<tag>]` markers that
`spike_runner --models <mid>@<t0>,<mid>@<t1>,...` /
`firesim_runner` parses without changes.

## Amortization payoff (why this exists)

RL sweeps generate N candidate kernels of the same problem. Under
harness_multi today, each candidate needs its own generated dir with
distinct mangled names — impossible without regenerating the whole
skeleton per candidate — so the RL loop falls back to N sequential
FireSim boots. With this harness:

- **Boot cost amortized N-way** — one infrasetup + one runworkload
  instead of N. On a 3-4 minute boot, N=8 collapses ~30 minutes into
  ~4 minutes.
- **Rodata shared** — one 100+ MiB baked input (from `test_input.bin`)
  instead of N copies. ram0 budget is `sizeof(input) + sizeof(golden) +
  N * sizeof(output)` instead of `N * (sizeof(input) + sizeof(golden) +
  sizeof(output))`.

## Design decisions

### Per-variant output buffers (not shared+clear)
Each variant gets its own `out_<tag>[MODEL_OUTPUT_SIZE]` in ram0.
Bounded cross-variant contamination is impossible — variant N cannot
read stale data from variant N-1 because their storage is disjoint.
Cost: `N * sizeof(output)` ram0. For the 25M-element ReLU case that's
`N * 96 MiB`; auto-ram0 sizing (upstream in `_run_lib.sh`) needs to
account for this. See "ram0 sizing formula" below.

### Symbol namespacing via per-variant `-include` header
Each variant's OBJECT lib is compiled with `-include <tag>_renames.h`
prepended. That header contains one `#define <sym> <sym>_<tag>` line
per mangled runtime symbol (enumerated by `gen_variant_renames.py`).
Zero source-level modifications — the LLM-emitted kernels.c stays
canonical, gcc's preprocessor does the renaming as it sees the source.

The alternative (`objcopy --redefine-syms` on each .a) would work but
requires an extra post-compile step per variant and doesn't cover
inline static symbols the same way.

### Weights + input + golden are NOT renamed
`weights.c`, `test_io.S`, and `test_io.h` contain identical bytes in
every variant — same weights, same baked I/O. Renaming their symbols
would N-plicate the rodata. `gen_variant_renames.py` explicitly
excludes any identifier that appears in those files from the rename
set.

### Verify + WALL_CYCLES markers per variant, PROFILE deferred
The generated `shared_input_main.c` emits
`MODELBLASTER_VERIFY [<mid>@<tag>]` + `MODELBLASTER_WALL_CYCLES
[<mid>@<tag>]` per variant. Per-op `MODELBLASTER_PROFILE_*` is a
future extension — needs the rename of `records_[]` / `n_` file-static
in `model.c` to be exposed to `shared_input_main.c` via
`model_<mid>_profile_records_<tag>()`. Wire this after the smoke
validates.

## Build inputs (CMakeLists.txt contract)

| Flag | Required? | Meaning |
|---|---|---|
| `MODELBLASTER_BACKEND` | no (default scalar) | Shared HW backend for all variants |
| `MODEL_DIR` | yes | Single per-problem `generated/<target>/` dir with the baseline sources |
| `VARIANT_SOURCES` | yes | Semicolon list of variant kernels.c paths (each has the LLM's implementation) |
| `VARIANT_TAGS` | no (default `v0;v1;...`) | Parallel semicolon list of tag strings for the emitted markers |
| `MODELBLASTER_KERNEL_CFLAGS` | no | Applied ONLY to VARIANT_SOURCES (same convention as harness_multi) |

## ram0 sizing formula

For upstream `_run_lib.sh`'s auto-ram0 logic to handle this harness,
the formula becomes:
```
ram0 >= INPUT_BYTES + GOLDEN_BYTES + N * OUTPUT_BYTES + FIXED_OVERHEAD
```
where `FIXED_OVERHEAD` covers stack, heap pool, Zephyr .bss, and the
variant TU merges (~128 MiB matches harness_multi's).

`_run_lib.sh`'s current formula assumes each variant has its own
input+golden — for shared-input, it over-allocates by
`(N-1) * (INPUT_BYTES + GOLDEN_BYTES)`. That's safe (upper bound) but
wastes room for large N. Optional followup: teach `_run_lib.sh` to
recognize the shared-input case and use the tighter formula.

## Runtime interface (spike_runner / firesim_runner)

The multi-model runners already accept `--models kb_A,kb_B,...` and
scan uartlog for `MODELBLASTER_WALL_CYCLES [<name>@<quant>]` markers,
using the tag content between the brackets as the model key. Emitting
`MODELBLASTER_WALL_CYCLES [kb_19_ReLU@v3] === 12345` from this
harness maps to `--models kb_19_ReLU@v0,kb_19_ReLU@v1,...` on the CLI
without any runner changes. Confirm at bring-up.

## Testing plan (before wiring KernelBlaster)

1. Manually stage one KernelBench problem (e.g. `19_ReLU`) with the
   modelblaster pipeline.
2. Duplicate its `kernels.c` twice — one unchanged reference, one with
   a trivially different implementation (say `output = input` — should
   fail verify, useful for the marker exercise).
3. `west build modelblaster/harness_shared_input ... -DMODEL_DIR=<stage>
    -DVARIANT_SOURCES=<c1>;<c2> -DVARIANT_TAGS=ref;bogus` → check the
   generated `<tag>_renames.h` files under build/modelblaster_shared/.
4. Run the resulting elf on spike; confirm both markers appear with
   distinct WALL_CYCLES, and the bogus variant emits VERIFY FAIL.

## KernelBlaster wiring (deferred)

Once bring-up passes on spike, wire from the KernelBlaster side:
1. `scripts/riscv/multi_link.sh` — wrapper that takes a manifest of
   `<tag> <kernels_c>` on stdin + `MODEL_DIR` in env, invokes west
   build with the assembled `VARIANT_SOURCES` + `VARIANT_TAGS`,
   copies fused ELF to `$FUSED_OUT`.
2. `spike_exec.py` / `firesim_exec.py`: `_link_batch_elf` writes each
   job's source .c to a temp path, passes `<tag> <path>` manifest to
   the fuse script.
3. `ExecBatchClient.submit_riscv`: pass the source .c path in
   `kernel_files` alongside io.npz, plus the base stage dir in
   `env_vars` so the strategy knows which MODEL_DIR to reference.
4. Multi-agent RL orchestrator OR single-agent variant that batches
   its own concurrent rollouts (harder — bandit needs to know the
   grouping).

## Open items for review

1. **PROFILE block emit** — punted for now; wire per-variant
   `model_<mid>_profile_records_<tag>()` accessor once we know it's
   needed for the RL loop's per-op cycle attribution.
2. **Fallback when a variant's kernel misbehaves** — currently the
   dispatcher continues to the next variant; alternative is to sample
   `k_cycle_get_64()` on a watchdog thread and TIMEOUT bad variants.
   Not needed for correctness (verify catches wrong output), but a
   pathological infinite loop would waste the whole boot.
3. **`_run_lib.sh` auto-ram0 update** — the shared-input case wants
   the tighter formula. Ship harness first, tune ram0 sizing later.
