# The XPU-RT schedule walker on Linux (host and boards such as the SpacemiT K1)

This is the original Chipyard/Zephyr walker, built as an ordinary hosted
executable. Nothing about the walker's design changes: the same generated
`xpurt_main.c`, the same dispatch table from `ingest_xpurt_schedule`, the same
per-(model, backend) kernels, the same `modelblaster_pool`, the same marker
protocol on stdout. The Zephyr kernel calls it makes are replaced by one small
header, `runtime/mb_posix.h`.

## What is reused unchanged

| Piece | File |
|---|---|
| Schedule -> dispatch table | `pipeline/ingest_xpurt_schedule.py` |
| Walker source generator | `pipeline/generate_xpurt_main.py` |
| Per-(model, backend) kernels, weights, symbol renames | `generate_skeleton.py`, `backend_rename.py` |
| Intra-op parallel-for | `runtime/modelblaster_pool/` |
| Output / profile / wall-cycle markers | emitted by the walker, parsed by the existing runners |

## What replaces the Zephyr APIs (`runtime/mb_posix.h`)

| Zephyr | POSIX |
|---|---|
| `k_sem_init/take/give` | `sem_t` on Linux; mutex + condition variable elsewhere (saturating at `limit`, like Zephyr) |
| `k_cycle_get_64()` | `CLOCK_MONOTONIC_RAW`, in **nanoseconds** |
| `CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC` | `1e9`, so the walker's cycles-per-ms gate stays correct |
| `k_yield`, `k_thread_cpu_pin`, `k_sched_lock` | `sched_yield`, `pthread_setaffinity_np` (Linux), no-op |
| `sys_reboot` | `fflush` + `exit(0)` |
| `CONFIG_BOARD_TARGET`, `CONFIG_POSIX_THREADS_AFFINITY` | defined by the header |

`rdcycle` traps from user mode on riscv64 Linux (SIGILL on the K1), so the
timer in the generated `model.c` uses `k_cycle_get_64` under
`-DMB_PLATFORM_POSIX`.

Build flags for every translation unit: `-D_GNU_SOURCE -DMB_PLATFORM_POSIX`
(`build_posix.sh` sets both).

## The one walker change: per-hart workers

The Chipyard walker spawns **one worker per core kind**. On Chipyard that is
right: each kind is a single hart or an intra-op pool. A K1 has eight identical
harts that the schedule drives independently (`CPU_P#0` ... `CPU_P#7`), so one
worker per kind would serialise them.

`generate_xpurt_main.py --worker-granularity hart` makes the emitted walker find
the distinct `(core_kind, hart)` pairs in the dispatch table at start-up and
spawn one pinned worker for each. Each worker takes only the entries scheduled
on its hart; dependency waits, the start-time gate, tracing and output are the
same code. Intra-op pools are not used in this mode (every hart already has its
own worker). The default, `--worker-granularity kind`, emits **exactly** the
original walker, byte for byte.

A hart is whatever the registry says: `ingest_xpurt_schedule` resolves
`CPU_P#n` to the n-th registry core of the mapped kind and records that core's
`harts[0]` in the table. To use all eight K1 harts, list them as eight
single-hart cores of the same kind in the registry.

## Flow

```bash
# 1. schedule -> dispatch table   (unchanged)
python -m modelblaster.pipeline.ingest_xpurt_schedule \
    --schedule sched.json --registry cores/<registry>.json \
    --ir net:graph.json --out out/sched.c --name sched

# 2. walker source                (--worker-granularity hart on multi-hart targets)
python -m modelblaster.pipeline.generate_xpurt_main \
    --schedule sched.json --out out/sched_main.c --name sched \
    --dispatch-table-header sched.h --core-kinds rvv --worker-granularity hart

# 3. build as a Linux executable  (native on the board, or cross with CC=)
MODEL_NAMES=net MODEL_DIRS_BASE=/abs/gen MODEL_BACKENDS=rvv \
XPURT_SCHEDULE_C=out/sched.c XPURT_MAIN_C=out/sched_main.c \
XPURT_INCLUDE_DIR=out OUT=out/walker bash harness_xpurt/build_posix.sh

# 4. run; stdout carries the usual === MODELBLASTER_* === markers
./out/walker > run.log
```

## Testing

`tests/test_xpurt_walker_host.py` runs steps 1 to 4 for real on a synthetic
network (real ingest, real generator in both modes, real `modelblaster_pool`
and `mb_posix.h`, only the kernels are fixtures). It checks dependency order,
that kind mode serialises, that hart mode overlaps independent harts and is
faster, that every hart gets a pinned worker, and that the default emission is
unchanged. It needs only a C compiler and runs on Linux and macOS.

## Targets whose core kinds are not backend names

On Chipyard a core kind is named after its backend (`rvv`, `gemmini`). On the
K1/K3 it is not: the K1 registry's kinds are `rvv`/`rvv_c1`, both run by the
`rvv_x60` backend; the K3's are `x100`/`a100`. When `--backends` has one entry
per `--core-kinds` entry and the names differ, entry *i* runs kind *i*:

```bash
--core-kinds rvv,rvv_c1 --backends rvv_x60,rvv_x60
```

In every other case the emitted walker is unchanged (byte-identical across the
schedule fixtures). Without this pairing every dispatch on such a target is
skipped with a `WARN unknown core_kind` and the run still exits 0.

Composite targets (`CPU_P#4+CPU_P#5+...`, one dispatch over several harts) are
accepted by `ingest_xpurt_schedule` and run on their **first** hart; ingest
prints how many there were. Spreading such a dispatch over its harts is not
implemented.

For a static or cross build pass `LDFLAGS=-static` (and `CC=`) to
`build_posix.sh`; QEMU user mode and board deploys need a static binary.

## Verified

- x86-64 Linux host, real ingest and generator, a real 3-model schedule (826
  dispatches, 46 instances), both worker modes: every dispatch ran, outputs
  bit-exact against the golden for `mlp_control`/`dronet`.
- riscv64 under QEMU user mode (`-cpu max,vlen=256`) and on a SpacemiT K1
  (Bianbu, Linux 6.6), static cross build with the `rvv_x60` backend, both
  worker modes: exit 0, 46/46 instances, outputs bit-exact for all three
  networks.

## What is not covered yet

- **Composite targets run on one hart** (see above).
- **Intermediate buffers are still shared per model**, as in the original
  walker: overlapping periodic instances of one network can race on them. A
  fix belongs in `generate_skeleton.py` (per-instance buffers), not here.
- No runner script that builds and executes in one step, or copies to a board.
  Steps 1 to 4 are manual.
