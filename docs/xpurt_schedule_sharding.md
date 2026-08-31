# XPU-RT schedule-selected sharding

This document is the contract between an XPU-RT schedule and the
ModelBlaster Linux runtime when one dispatch occupies several harts at once.
It describes the path exercised by the K1 exact-cycle experiment; it is not a
proposal.

## Sharding is not splitting

Splitting creates several dispatches and changes the graph. Sharding keeps one
dispatch, one dispatch ID, and the same dependency edges, but runs that
dispatch cooperatively on a composite target such as:

```json
{
  "hardware_target": "CPU_P#0+CPU_P#1+CPU_P#2+CPU_P#3",
  "impl": "rvv"
}
```

`hardware_target` says **where and how wide** the dispatch runs. `impl` says
**which kernel implementation** runs there. They are independent fields.

There are two ways a `shard_factor` can reach code generation:

1. `apply_shard_hint.py` applies explicit
   `modelblaster.shard_hints/v1` advice to an IR.
2. The scheduled-runtime path derives the width directly from each composite
   `hardware_target` with `schedule_shards.py`.

The second path is authoritative for an XPU-RT schedule. It lets the scheduler
choose a width per dispatch from measured 1/2/4/8-hart profiles instead of
forcing one `MB_SHARD_FACTOR` on a whole model.

## End-to-end path

```text
measured topo_0 / topo_0_1 / topo_0_1_2_3 profiles
    -> XPU-RT machine_combination_mode="shard"
    -> schedule.json composite hardware_target
    -> ingest_xpurt_schedule.py preserves every selected hart
    -> schedule_shards.py annotates the codegen-only IR
    -> generate_skeleton.py emits per-shard packed convolution weights
    -> generate_xpurt_main.py creates and selects exact-hart pools
    -> modelblaster_pool runs caller + pinned helper threads
```

`scripts/run_xpurt_k1.sh` drives these steps. For every model it writes
`graph.xpurt_schedule.json`; the original `graph.json` remains unchanged so
profile provenance and later schedules still refer to the extracted graph.

## Scheduler requirements

The workload must enable composite combinations and must price each width
with its own measured profile:

```json
{
  "hardware": {
    "machines": {"cpu_p": 4, "cpu_e": 4},
    "profile": {"topo_tag_override": false}
  },
  "scheduler": {
    "machine_combination_mode": "shard",
    "enforce_same_processor_combinations": true
  }
}
```

`topo_tag_override: false` is load-bearing. Setting it to `true` can charge a
four-hart candidate the single-hart cost and makes the solve invalid.
Combinations must stay within one runtime kind; ModelBlaster refuses a
composite target that mixes kinds, repeats a physical hart, or names an
unbound hart.

The generated model has one packed-weight layout per dispatch, so all periodic
instances of a packed convolution dispatch must select the same width.
`schedule_shards.py` refuses invocation-dependent widths instead of silently
building the wrong layout.

## Code-generation rules

The explicitly supported shardable operations are:

- `conv2d_s8`
- `conv2d_batchnorm2d_s8`
- `conv2d_batchnorm2d_silu_s8`
- `conv2d_silu_s8`
- `linear_s8`
- `matmul_s8`

Packed convolution weights are re-packed into one output-channel slice per
shard. Output channels must divide evenly by the selected width; disagreement
with an explicit `shard_factor` or a non-divisible width is a hard error.
Linear and matmul weights remain row-major and are sliced by their parallel
wrappers at runtime.

Curated cross-model kernels live under `kernels/<backend>/`. Successful
model-specific generated kernels live under
`examples/<model>/<quant>/cache/<backend>/`. Aggregate `kernels.c`, weights,
objects, and binaries are generated build products and are intentionally not
committed.

## Runtime ownership and exclusion

The runtime creates one persistent `modelblaster_pool` for every distinct
composite hart set in the schedule. The scheduler worker on the first hart is
the master and participates as slice zero; helpers are pinned to the remaining
harts in the exact order named by the schedule. Singleton entries use the
serial/NULL-pool path.

Before a composite dispatch starts, its master acquires per-hart locks in
numeric order. This prevents another scheduler worker or an overlapping pool
from using any reserved hart concurrently. Numeric acquisition order prevents
lock-order cycles. Dispatch-local model state selects the entry's pool, so two
concurrent entries never race by rewriting a shared `.pool` pointer.

At startup the harness prints every pool, including width and physical harts.
The board evaluator checks those declarations, scheduler-worker affinity,
requested and observed real-time policy, trace completeness, numerical
goldens, and deadlines.

## Build and run

From the ModelBlaster checkout nested in XPU-RT:

```bash
env CORE_KINDS=rvv,ime,rvv_c1 \
  CROSS=../tools/riscv-tools-spacemit/spacemit-toolchain-linux-glibc-x86_64-v1.1.2/bin/riscv64-unknown-linux-gnu- \
  PY="$PWD/.venv/bin/python" \
  MODELBLASTER_K1_RT_PRIORITY=80 \
  MODELBLASTER_KERNEL_CC=../tools/riscv-tools-spacemit/spacemit-toolchain-linux-glibc-x86_64-v1.1.2/bin/riscv64-unknown-linux-gnu-gcc \
  ./scripts/run_xpurt_k1.sh \
    --schedule ../schedules/scheduled_networks_k1_tri_exact_100ms_feedback_greedy_profiled.json \
    --models mlp_control,fused_full,dronet,ffn_block \
    --backends rvv_x60,ime_x60,rvv_x60 --jobs 4
```

Model-specific checkpoint or calibration variables still apply; see
`../../docs/the_loop.md` for the complete K1 command and evidence protocol.

## Tests and evidence

The focused contracts are covered by:

- `pipeline/tests/test_schedule_shards.py`
- `pipeline/tests/test_per_dispatch_impl.py`
- `pipeline/tests/test_linux_xpurt_cmake.py`
- `runtime/modelblaster_pool/test_app/`

The checked-in exact-cycle schedules and twenty audited K1 runs are under
`XPU-RT/results/k1_feedback_exact/`. They exercise two- and four-hart blocks,
per-dispatch implementation selection, affinity, pool creation, numerical
goldens, cyclic completion, and deadline validation.
