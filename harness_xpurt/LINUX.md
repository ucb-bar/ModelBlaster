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
own worker). The default, `--worker-granularity kind`, emits the original
walker; the only difference is how it stops on an error (see "Errors stop the
run" below).

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
schedule fixtures). Without this pairing no backend runs the target's core
kinds, and the run stops at the first such dispatch (see below).

## Per-dispatch implementations

A schedule solved with XPU-RT's `scheduler.enable_impls` writes an `impl` on
each dispatch (`"rvv"`, `"ime"`): `hardware_target` names the core a dispatch
runs on, `impl` the implementation it runs with. Ingest carries `impl` into
the dispatch table (it defaults to the entry's core kind), and the walker picks
the kernel table by it. This is how one K3 A100 core runs a GEMM on IME and the
next op on its vector unit.

Map each (core kind, impl) to a backend with `--impl-backends`; every backend
named there must be built as well:

```bash
--core-kinds x100,a100 --backends rvv_k3,ime_k3_a100 \
--impl-backends a100:ime=ime_k3_a100,a100:rvv=rvv_k3_a100,x100:rvv=rvv_k3
```

A matched dispatch keeps its core's state, so its pool follows the core and its
kernel follows the implementation. Dispatches without a match fall back to the
`--core-kinds`/`--backends` pairing, with `impl` compared against the kind
names; a schedule without `impl` therefore runs as before. The one-step runners
take the same `--impl-backends` and build the extra backends.

## Composite targets

A composite target (`CPU_P#4+CPU_P#5+...`) is one dispatch over several harts.
`ingest_xpurt_schedule` records the whole hart set (`.n_harts`, `.harts`, the
first hart first). With `--worker-granularity hart`:

- the worker of the first hart runs the dispatch on an intra-op pool created
  at start-up on exactly that hart set (one pool per distinct set; helpers
  place themselves, so A100 sets work too);
- every entry locks the harts it runs on, in ascending order, around its
  kernel call, so a composite dispatch has its harts to itself and two
  composites cannot deadlock.

A dispatch spreads over its harts only if its kernel has a pool path
(`parallel_<op>`): `linear`, `linear_s8`, `conv2d`, `conv2d_s8`. On the RVV
backends, whose conv weights are packed IHWOC, `parallel_conv2d_s8` shards a
repacked copy of the weights (built once per weight tensor and pool size), so
the result stays bit-exact. Other ops run on the first hart while the rest of
the set is held. In kind mode a composite dispatch uses its kind's pool.

## Per-instance buffers

Every intermediate buffer has `MB_N_BUF_SETS` copies (`buffers.c`), and a
dispatch uses copy `s->buf_set`. The walker passes
`instance % model_<m>_n_buf_sets`, and `build_posix.sh` builds each network
with one copy per instance in the schedule (ingest writes
`XPURT_N_INSTANCES_<NET>` into the table header). So overlapping instances of
one network never share scratch. Builds that do not set `MB_N_BUF_SETS`
(Zephyr, the single-model harness) keep one shared set.

## SpaceMiT K3 (X100 + A100)

The K3 has 16 CPUs: X100 = 0-7 (VLEN 256), A100 = 8-15 (VLEN 1024). A thread may
run on an A100 CPU only after writing its own TID to `/proc/set_ai_thread`, and
that registration is one-way. `runtime/mb_placement.{h,c}` (Sam Bai) handles it:

- Every worker places itself on its hart as its first action
  (`mb_place_current_thread`: register if the CPU is an AI CPU, pin, verify).
  Pthread-attribute affinity cannot do this, so on Linux it is not used.
- `main` pins itself to the first non-AI CPU before any thread exists, so pools
  and workers start on the X100 cluster and main never registers.
- Pool helpers place themselves the same way, and a kind's pool runs on that
  kind's harts from the registry (`--registry`), so an A100 pool uses 8-15.

Build with `MODELBLASTER_AI_CPUS=8-15` and the K3 backends paired with the
registry's kinds (`cores/spacemit_k3.json`):

```bash
--core-kinds x100,a100 --backends rvv_k3,rvv_k3_a100     # or ime_k3_a100 for the A100
```

`build_posix.sh` compiles each backend's kernels with its `kernel_cflags` from
`pipeline/backends.py` when `CC` targets RISC-V: `rvv_k3` at `zvl256b`, the A100
backends at `zvl1024b`, which is illegal on the X100, so they must only run on
CPUs 8-15. `KERNEL_CFLAGS_<BACKEND>` overrides. On the K1 and the host
`MODELBLASTER_AI_CPUS` is unset and placement is a plain pin.

## One-step board runs

`scripts/run_xpurt_k1.sh` (K1) and `scripts/run_xpurt_k3.sh` (K3) run every
step above for a schedule: generate each (network, backend), ingest, generate
the walker, cross-build a static binary with `build_posix.sh`, copy it to the
board, run it, and pull back stdout and the trace CSV.

```bash
eval "$(../XPU-RT/scripts/setup_spacemit_toolchain.sh)"     # sets CROSS
scripts/run_xpurt_k1.sh --schedule <sched.json> --models mlp_control,dronet,yolov8_nano
MODELBLASTER_K3_HOST=k3 scripts/run_xpurt_k3.sh --schedule <sched.json> \
    --models mlp_control,dronet,yolov8_nano
```

The board is reached through the ssh host `MODELBLASTER_K1_HOST` (default
`k1`) or `MODELBLASTER_K3_HOST` (default `spacemitk3`). The K3 runner builds
`rvv_k3` for the X100 and `ime_k3_a100` for the A100, takes
`MODELBLASTER_AI_CPUS` from the registry, and launches under `taskset -c 0-7`.
`MB_XPURT_WORKERS=kind` selects kind workers (default `hart`);
`--staged-ir <network>:<dir>` uses an IR that is already built, such as a
rewritten one, instead of extracting it.

## Warm-up, placement record, repeated runs (hosted Linux, opt-in)

The runners take three knobs; each is off by default and leaves the emitted
walker unchanged when off.

- `MB_XPURT_WARMUP=1` (`generate_xpurt_main --warmup`): every worker runs its
  entries once, untimed and without the start-time gate, then all workers meet
  at a barrier, the completion semaphores are drained, `run_t0` is reset and
  the timed pass runs as before. Trace, wall-cycle and profile records come
  from the timed pass. (Sam Bai's `--warmup` idea.)
- `MB_XPURT_PLACEMENT=1` (`--placement`): each entry records `sched_getcpu()`
  before and after its kernel; the block is saved as `<sched>_placement.csv`
  (`entry_id,network,instance,dispatch_id,core_kind,impl,hart,n_harts,harts,
  cpu_before,cpu_after`). XPU-RT's `scripts/check_k3_placement.py` checks it.
- `MB_XPURT_REPEATS=N`: run the deployed binary N times; with N > 1 each run's
  stdout, trace and placement files get an `_r<k>` suffix.

## Errors stop the run

A dispatch whose implementation has no backend in the build prints
`FATAL: xpurt: entry <n> of <network> asks for impl '<impl>', which this
binary was not built with (has: ...)` and stops the run, as do the walker's other fatal errors (too many workers, a failed pool or
thread create). On Linux they exit with status 2; on Zephyr they reboot, as
before. A skipped dispatch would otherwise report the runtime of a different
workload. `--core-kinds`, `--backends` and `--impl-backends` that do not cover
the schedule are the usual cause.

For a static or cross build pass `LDFLAGS=-static` (and `CC=`) to
`build_posix.sh`; QEMU user mode and board deploys need a static binary.

## Verified

Three int8 networks (`mlp_control`, `dronet`, `yolov8_nano`), extracted and
generated with this repo's pipeline, 46 instances, both worker modes. Outputs
are compared with each network's golden (`yolov8_nano` on its 16 printed
samples):

- x86-64 Linux host: the walker tests in `tests/test_xpurt_walker_host.py`,
  including a run with one CPU treated as an AI CPU.
- riscv64 under QEMU user mode (`-cpu max,vlen=256`), static cross build,
  `rvv_x60`: 46/46 instances, all three networks bit-exact.
- SpacemiT K1 (Bianbu, Linux 6.6), the same binaries: bit-exact, outputs
  identical to QEMU.
- SpacemiT K3 (Bianbu, Linux 6.18), X100 + A100 (`MODELBLASTER_AI_CPUS=8-15`),
  `rvv_k3` on the X100 with either `ime_k3_a100` or `rvv_k3_a100` on the A100:
  46/46 instances, all three networks bit-exact, outputs identical to the K1.
  The A100 workers run on CPUs 8-15 after registering their own TIDs.
- SpacemiT K3 with per-dispatch implementations: the same 1054-dispatch
  schedule with `impl` set (A100 dispatches alternating `ime` and `rvv`, a
  third of the X100 dispatches `rvv`, the rest unset), built with `rvv_k3`,
  `ime_k3_a100` and `rvv_k3_a100`: both worker modes exit 0, 46/46 instances,
  all three networks bit-exact. Every profile row's backend is the one the
  schedule chose for that dispatch, and both A100 backends ran in the same run.
- The one-step runners, end to end on the K1 and the K3 with the same three
  networks and a 1054-dispatch schedule (554 composite targets): exit 0,
  1054 trace rows, all three networks bit-exact.

## What is not covered yet

- **Composite speedup depends on the kernel**: only ops with a pool path
  spread over the hart set (see "Composite targets").
- **Start gate**: each instance's first dispatch waits for its scheduled start;
  later dispatches, including other DAG roots, start when their dependencies
  finish.
- **Warm-up**: off by default, so each network's first instance runs cold
  inside the timed window unless `MB_XPURT_WARMUP=1` is set (see below).
- **Output check**: the walker prints the output markers; the runners compare
  them against the golden. There is no in-binary golden check.
