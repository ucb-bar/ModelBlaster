# xpurt walker semantics

How the FireSim/spike harness drives an XPU-RT schedule end-to-end. Cross-references:

- IR / ingest:  `agents/pipeline/ingest_xpurt_schedule.py`
- Codegen:     `agents/pipeline/generate_xpurt_main.py`
- Harness:     `agents/harness_xpurt/{prj.conf, CMakeLists.txt}`
- Pool runtime: `agents/runtime/agents_pool/`

## 1. Inputs

Three artifacts feed the harness:

1. **`scheduled_*.json`** — XPU-RT scheduler output. Map of
   `<job_name>_dispatch_<id>` to record:
   ```
   id, ordinal/total, dependencies[], hardware_target ("CPU_P#0" | "CPU_E#0"),
   start_time (ms), duration (ms), job_name, time_dependency
   ```
   `job_name` is `<network>` for one-shot networks or `<network><instance_idx>`
   for periodic networks (e.g. `dronet0`, `dronet1`, … with period=200 ms).
2. **Per-network IR `*_dispatch_graph.json`** — the static op graph. Each op
   has `dispatch_id`, `op`, `name`, plus shape metadata.
3. **Core registry `agents/cores/*.json`** — abstract `CPU_P/CPU_E#n` slots
   resolve into `(core_name, core_kind, hart)`. For our dual-rocket-saturn-gemmini
   bitstream, `cpu_p → gemmini@hart0`, `cpu_e → rvv@hart1`.

## 2. Ingest → C dispatch table

> **Important — IR-vs-codegen dispatch_id remap.** The IR (`graph.json`)
> assigns a contiguous `dispatch_id` 0..N-1 to **every** op including
> zero-cost aliases (`view`, `chunk2_c1`). `generate_skeleton.py`
> filters those out and emits a flat `MODEL_<UMID>_DISPATCH_FNS_<BS>[]`
> table sized to the count of *non-zero-cost* ops, indexed
> sequentially. So an IR `dispatch_id` greater than the codegen index
> of its op (or any IR id corresponding to a skipped op) is **not** a
> valid table index. Indexing past the table reads adjacent static
> data — typically int8 weights — and the kernel jalrs garbage. We
> hit exactly this on yolov8_nano (212 IR ops, 8 zero-cost; codegen
> table size 204; schedule had IDs 0..211; out-of-range entries
> jumped to `0x01e2f185f0fe15ec` and faulted with `mcause=1
> Instruction Access fault`). The fix lives in ingest: it builds an
> `ir_dispatch_id → codegen_idx` remap from each network's IR and
> rewrites every entry's `dispatch_id` to the codegen-space index
> (zero-cost ops get `-1` and the runtime skips them but still posts
> their completion sem).

`ingest_xpurt_schedule.py::load`:

1. Sort dispatches stably by `(start_time, schedule_key)` → assigns each one
   a contiguous `entry_id` 0..N-1. **All later cross-references are by
   `entry_id`, not the original key.**
2. Validate every `(network, dispatch_id)` against the IR. Multiple instances
   of the same network share one IR.
3. For each dispatch resolve `dependencies[]` (intra-job data edges) and
   `time_dependency` (cross-job ordering edge from the scheduler) into
   `entry_id` indices.
4. Resolve `hardware_target` to `(core_name, core_kind, hart)`.

`emit_table` writes a flat `.c` array of `xpurt_sched_entry_t`:

```c
typedef struct {
    int            entry_id;
    const char    *network;          // e.g. "dronet"
    int            instance;         // 0..K-1 across periodic instances
    int            dispatch_id;      // IR-side
    const char    *job_name;         // "dronet0" / "yolov8_nano"
    const char    *op;               // "conv2d" / "linear"
    const char    *name;             // IR node name
    const char    *core_name;        // "rocket0" / "rvv0"
    const char    *core_kind;        // "rvv" | "gemmini" | "scalar"
    int            hart;             // preferred hart, -1 if unbound
    float          start_time_ms;    // schedule-issued start
    float          duration_ms;      // schedule-modeled duration
    int            n_deps;
    const int     *deps;             // entry_ids of intra-job data deps
    int            time_dep_entry_id; // cross-job ordering edge, -1 if none
} xpurt_sched_entry_t;
```

The table is monotonic in `start_time_ms` — workers walk it in this order.

## 3. Generated `<schedule>_main.c`

`generate_xpurt_main.py::_emit` produces a single C source. Layout:

### 3.1 Per-network state, per-entry sync, per-kind pool

```c
static model_<m>_state_t  s_<m>;           // ONE state struct per network m
static <out_t>            out_<m>[...];    // shared output buffer

static struct k_sem completion_sems[N_ENTRIES];   // one per entry
static void *pools[N_KINDS] = { NULL };           // per-kind agents_pool
static uint64_t run_t0;                            // mtime baseline at worker spawn
```

Crucial specs (and where the current code gets them wrong — see §4.5):

- **Concurrent ops within a network are a designed feature.** A network's
  static op DAG (ResNet skip branches, YOLOv8's parallel detection heads,
  etc.) has independent ops with **no `deps[]` edge between them**. The
  scheduler is free to assign such independent ops to different kinds and
  start them concurrently — in fact this is precisely the heterogeneity
  the system aims to exploit. The runtime **must** allow that.
- **Intermediate buffers are file-static, model-level, and disjoint per IR
  edge.** `model.c` declares (via `extern` from `buffers.c`) one
  `buf_<mid>_<node>` per IR node (e.g. `buf_dronet_conv_modules_0`,
  `buf_dronet_maxpool1`, …). A dispatch reads from its predecessor's
  `buf_*` and writes to its own. Two independent ops in the same network
  necessarily touch different `buf_*` arrays, so no buffer-level race
  exists *as long as* the IR's `deps[]` correctly captures the RAW edges.
  This is what task #82 ("Fix per-backend buffer aliasing") nailed down —
  the buffers TU is per-model not per-backend, so cross-kind dispatches
  in one network see the same scratch.
- **`s_<m>` is shared across all workers** that dispatch into network m.
  It is *not* single-writer. Read-only fields (`input`, `output` pointer)
  are fine; mutable scratch fields (notably `s_<m>.pool`) are not (§4.5).
- **`completion_sems`** are sized `(0, 64)` — a binary completion event with
  a high count limit so multiple consumers can each take their own
  "completed" reading without blocking each other.

### 3.2 Per-kind worker thread

One Zephyr-pthread worker per distinct `core_kind` (e.g. one for `gemmini`,
one for `rvv`). Each worker:

1. Pins itself to the first hart the table assigns to its kind via
   `pthread_attr_setaffinity_np` (vendored Phase-A patch). Singleton-machine
   schedules: all entries of one kind share a hart.
2. Walks `<UPPER>_TABLE[0..N-1]` in `entry_id` order (== `start_time` order).
3. Skips entries whose `core_kind` doesn't match.

### 3.3 Per-entry execution (the core loop body)

For each matching entry `e_` at index `i_`:

```c
// (a) DATA DEP WAIT — intra-job edges
for (int d = 0; d < e_->n_deps; d++) {
    k_sem_take(&completion_sems[e_->deps[d]], K_FOREVER);
    k_sem_give(&completion_sems[e_->deps[d]]);   // re-post for other waiters
}

// (b) TIME DEP WAIT — cross-job ordering edge from scheduler
if (e_->time_dep_entry_id >= 0) {
    k_sem_take(&completion_sems[e_->time_dep_entry_id], K_FOREVER);
    k_sem_give(&completion_sems[e_->time_dep_entry_id]);
}

// (c) PER-INSTANCE START GATE — only on dispatch_id==0
if (e_->dispatch_id == 0) {
    uint64_t target = run_t0 +
        (uint64_t)(e_->start_time_ms * XPURT_CYCLES_PER_MS);
    while ((uint64_t)k_cycle_get_64() < target) k_yield();
    model_<m>_reset_profile_<bs>();    // reset all backends' counters
    wall_start_<m> = k_cycle_get_64();
}

// (d) STEER INTRA-OP POOL
s_<m>.pool = (void *)pools[my_kind_idx];   // NULL if 1-hart kind

// (e) DISPATCH BY core_kind → per-backend table
if (strcmp(e_->core_kind, "gemmini") == 0) {
    MODEL_<UMID>_DISPATCH_FNS_GEMMINI[e_->dispatch_id](&s_<m>);
} else if (strcmp(e_->core_kind, "rvv") == 0) {
    MODEL_<UMID>_DISPATCH_FNS_RVV[e_->dispatch_id](&s_<m>);
}

// (f) WALL-CYCLE FINALIZE — last dispatch in instance
if (e_->dispatch_id == MODEL_<UMID>_OP_COUNT - 1) {
    wall_cycles_<m> = k_cycle_get_64() - wall_start_<m>;
}

// (g) RELEASE — unblock dependents
k_sem_give(&completion_sems[i_]);
```

## 4. Synchronization model

### 4.1 Data dependencies (`deps[]`)

Generated by the IR (intra-network producer/consumer edges). The walker's
take/re-give pattern (step **(a)**) makes the sem behave as a one-shot latch:
once any consumer takes it, every other consumer that races in still sees it
posted because each take is followed by a give. With `init=(0, 64)` the
counter never wraps for plausible fan-outs.

**Invariant:** entry `i_` runs only after every `entry_id ∈ deps[i_]` has
posted its completion. Within a network, `deps[]` forms a DAG — **not a
linear chain**. Independent ops of the same network (e.g. ResNet skip
branches, YOLOv8 detection-head fanout) have no edge between them and are
free to run concurrently on different kinds' workers.

### 4.2 Time / cross-job edges (`time_dep_entry_id`)

The scheduler emits these to serialize ordering between independent jobs that
happen to share a resource (e.g. periodic `dronet1` should not preempt
`yolov8_nano`'s gemmini run). Same take/re-give protocol as data deps, but
across networks. `-1` means "no cross-job edge for this entry".

### 4.3 Per-kind worker walk-order

Each worker walks the *same* table but only acts on its kind's entries. This
is deadlock-free because:

- An entry `i_` is only blocked on `deps` and `time_dep` whose `entry_id < i_`
  (the table is start-time-sorted; XPU-RT only emits backward edges).
- Therefore `entry_id`-monotone walk over the worker's slice cannot create a
  hold-and-wait cycle with any other worker — each worker only ever waits on
  *earlier* entries, regardless of kind.

### 4.4 Intra-op parallelism (`agents_pool`)

**Spec.** Inside a dispatched kernel, `parallel_<op>(s->pool, ...)` should
fan out onto helper threads pinned to other harts of the *kind that is
running this dispatch*. Pool size per kind = `harts_of_kind` (caller +
helpers). `0` ⇒ NULL pool, kernel runs synchronously on the scheduler
worker.

### 4.5 Race conditions and what's been fixed

Concurrent cross-kind dispatch into one network exposed several races
in the previous emitter. Status as of now:

1. **`s_<m>.pool` write race — FIXED.** Previously the worker did
   `s_<m>.pool = pools[my_kind_idx];` per-dispatch on a single shared
   `s_<m>`. The emitter now declares one state struct per
   (network, kind) — `s_<m>_<kind>` — with `.pool` bound once in
   `main()` after pool creation. The dispatch path uses
   `&s_<m>_<core_kind>` and never writes the pool field; concurrent
   workers across kinds reference disjoint state. Intermediate
   buffers stay in the per-model `buffers.c` (extern from every
   `(network, kind)` state's owner), so cross-kind data flow is
   unchanged.

2. **Profile counters race within a backend — STILL LATENT.**
   `model.c`'s file-static `int n_;` and `records_[n_++]` are
   per-`(model, backend)` TU but not per-thread. Today only one
   worker per kind exists, so no two concurrent ops share a single
   `(model, backend)` `n_`. **Will break** the moment we add
   multi-hart-per-kind workers. Track in task #64 — fix shape: atomic
   `n_` or per-worker record arrays merged at end.

3. **`wall_start_<m>` / `wall_cycles_<m>` per-instance race — FIXED.**
   The emitter now allocates `wall_start_<m>[K]` and
   `wall_cycles_<m>[K]` arrays where K = max instance index + 1
   (computed from the schedule). Each entry indexes by
   `e_->instance`, so back-to-back periodic instances cannot clobber
   each other. The walker prints `=== AGENTS_WALL_CYCLES_INST
   [<net>#<inst>] === <cycles>` per instance, plus the existing
   single `=== AGENTS_WALL_CYCLES [<net>] === <max_cycles>` line per
   network for streamed-runner end-of-run sentinels.

4. **`completion_sems` re-give pattern relies on no-overflow.** Step
   (a) does `take; give;` so other waiters re-see "done." With
   `init=(0, 64)` the count limit is 64. If a single completion has
   more than 64 fanout consumers all racing through the take-then-give
   in close succession, the `give` after the 64th can spuriously
   `k_sem_give` past the limit (silently capped, but other waiters now
   may starve). Currently 64 is plenty; flag if op-fanout grows.

5. **The `s_<m>_<kind>.input/output` pointers** are set once in
   `main()` and read-only thereafter. Safe.

### 4.6 Per-kind worker walk-order

Each worker walks the *same* table but only acts on its kind's entries.
This is deadlock-free because:

- An entry `i_` is only blocked on `deps` and `time_dep` whose
  `entry_id < i_` (the table is start-time-sorted; XPU-RT only emits
  backward edges in `entry_id` order).
- Therefore `entry_id`-monotone walk over the worker's slice cannot
  create a hold-and-wait cycle — each worker only waits on *earlier*
  entries, regardless of kind.

**Validated at ingest time.** `ingest_xpurt_schedule.py::load` now
asserts every `dep` and `time_dep` resolves to an `entry_id` strictly
less than the current entry's. Out-of-spec scheduler output fails at
ingest with a clear error instead of deadlocking the harness mid-walk.

## 5. Periodic timing

The scheduler emits one job per period instance: `dronet0` at `start_time=0`,
`dronet1` at `start_time=200`, etc. Inside `dronet1`'s table entries the
ordering is identical to `dronet0`'s — same dispatch_ids, same op DAG —
just shifted in `start_time`.

Two mechanisms enforce the period:

1. **Soft start gate** (step **(c)**, only on `dispatch_id==0`): busy-yield
   until `k_cycle_get_64() >= run_t0 + start_time_ms*cycles_per_ms`. Models
   "task wakes at deterministic boundary, then runs as fast as it can" —
   subsequent ops are NOT gated, they chain via data deps and run back-to-back.
2. **Cross-job edges** (`time_dep_entry_id`): scheduler-issued. If a periodic
   instance must complete before some other-network entry runs, the edge
   serializes them via the completion sem.

There is intentionally **no** post-instance "sleep until next period"
behavior — the schedule's `start_time` is the contract, the walker honors it
on entry, and any slack between actual completion and the next period is just
idle time on the worker. If an instance overruns its period the next instance
still gates on its own absolute `start_time`, so the schedule self-corrects
once execution catches up; a chronically late workload is a scheduler bug,
not a runtime one.

## 5.5 Spec → implementation validation

The desired spec, in one sentence:

> **The runtime must let the scheduler dispatch independent ops within
> one network onto different kinds at overlapping times, with correct
> data flow, correct intra-op fanout, and correct profile output.**

Auditing `generate_xpurt_main.py` + `model.c` against that:

| # | Spec property | Impl status | Evidence |
|---|---|---|---|
| 1 | Independent ops within a network can start concurrently across kinds. | **OK** — workers walk independently, gated only by `deps[]`/`time_dep_entry_id`. No global "one network at a time" lock. | `generate_xpurt_main.py:_emit` outer worker loop |
| 2 | Data flow correct under concurrent dispatch. | **OK** — intermediate buffers live in a shared per-model `buffers.c` (one TU per model, not per backend), so cross-kind dispatches in the same network read each other's writes. | `agents/examples/dronet/int8/generated/rvv/model.c:218` extern decls |
| 3 | Intra-op fanout uses the *running kind's* pool. | **OK** (was broken; landed). The walker now declares `s_<m>_<kind>` per (network, kind), `.pool` is bound once in `main()`, and the dispatch path passes `&s_<m>_<core_kind>` so two kinds dispatching the same network never share a state struct. | `generate_xpurt_main.py` state_decls / state_pool_assigns |
| 4 | Profile records correctly attribute cycles to their backend. | **OK for current shape** — `n_`/`records_[]` are file-static per `(model, backend)` TU, and we run one worker per kind, so no two concurrent ops share a `(model, backend)`'s `n_`. **Will break** if we ever add multi-hart-per-kind workers. | `model.c:254` |
| 5 | Periodic-instance wall cycles correctly attributed. | **OK** (was broken; landed). `wall_start_<m>[K]` / `wall_cycles_<m>[K]` arrays indexed by `e_->instance`; per-instance lines printed under `AGENTS_WALL_CYCLES_INST [<net>#<inst>]`, summary line under the existing `AGENTS_WALL_CYCLES [<net>]` keeps streamed-runner counters intact. | `generate_xpurt_main.py` `wall_decls`, print_blocks |
| 6 | Deadlock-free walk. | **OK** — invariant `time_dep_id < entry_id` (and `dep_id < entry_id`) is now asserted at ingest time and fails fast on out-of-spec scheduler output. | `ingest_xpurt_schedule.py::load` |
| 7 | Per-instance start time honored on `dispatch_id==0`. | **OK** — busy-yield on `k_cycle_get_64()`. The emitter still resets every backend's `model_<m>_reset_profile_<bs>()` at the gate, which is correct (each backend's `records_/n_` arrays get a clean slate per instance, even if the instance crosses kinds). | step (c) in §3.3 |
| 8 | Single-kind-multi-hart concurrency. | **Not supported.** Single worker per kind. Today's schedules don't need it; if a registry declares `harts=[a,b]` for one kind, the walker only uses one of them. Combined with the row-4 caveat: a future multi-hart-per-kind change requires both more workers AND atomic profile counters. | `_emit` thread spawn loop |
| 9 | Per-bitstream hart count comes from one place. | **OK** — `harness_xpurt/prj.conf` no longer sets `CONFIG_MP_MAX_NUM_CPUS`; `run.sh` always applies a per-target overlay (`spike_quad.conf` / `firesim_chipyard.conf` / `firesim_chipyard_dual_gemmini.conf`) and fails fast if none is found. | `agents/harness/backends/*.conf` + `xpurt_demo/run.sh` |
| 10 | IR-vs-codegen `dispatch_id` are kept consistent. | **OK** — ingest now remaps each schedule entry's IR `dispatch_id` to the codegen-space index (zero-cost ops → -1 sentinel). The walker short-circuits on `dispatch_id < 0` (still posts the completion sem so dependents unblock). Closes the "table indexed past its end → garbage `jalr` to `0x01e2f185f0fe15ec`" crash that was masquerading as a sync bug on FireSim. | `ingest_xpurt_schedule.py::load` remap, `_emit` early-out |

### Validation

Re-ingested + rebuilt + ran the 422-entry FireSim schedule
(7 dronet instances + 1 yolov8_nano, kinds=gemmini+rvv) on the
chipyard spike with `--extension=gemmini`. All 422 dispatches walked
to completion. No crash, no deadlock. Both networks produced output
and per-network/per-instance wall-cycles printed. Numerical accuracy
is a separate downstream concern (dronet `max_abs_err=1`,
yolov8_nano `max_abs_err=84` — gemmini float-scale-requantize drift,
covered by `Backend.atol_override=8` for individual-network verifies
but not currently propagated to the xpurt-harness `--atol`). Track
in a follow-up.

## 6. Profile + verify

After both workers `pthread_join`:

- Each worker has dispatched its kind's entries and posted its completion sems.
- Per-network output buffers `out_<m>` hold the final tensors.
- Each `(model, backend)` pair maintains its own profile-record array
  (separate object libs via `backend_rename.py` → `_<bs>` suffix on every
  externally-visible symbol). The walker prints them tagged with `<bs>` so
  the host can split per-op cycles by backend.
- `spike_runner` / `firesim_runner` parse the
  `=== AGENTS_OUTPUT_BEGIN [<network>] ===` markers and check against a
  PyTorch golden.

## 7. Where things can go wrong

A short field guide while debugging on spike (where this should be deterministic):

| Symptom | Likely cause |
|---|---|
| `pthread_create` returns `EINVAL` (rc=22) | `pthread_attr_init` failed to allocate the dynamic stack: `CONFIG_HEAP_MEM_POOL_SIZE` < `CONFIG_DYNAMIC_THREAD_STACK_SIZE * n_kinds`. |
| `mcause=1 Instruction Access Fault` mid-run | Worker stack overflow (DroNet's `im2col_buf` ≈ 295 KB; conv VLAs blow 16 KB stacks). Bump `CONFIG_DYNAMIC_THREAD_STACK_SIZE`. |
| `mcause=2 Illegal Instruction` on `vsetivli` | Lazy-V trap on a secondary hart. Workaround: `CONFIG_RISCV_ISA_EXT_V_LAZY=n`. |
| Output mismatch only on cross-backend networks | Per-backend buffer aliasing — different backends share the same intermediate buffer name; check `backend_rename.py` covered the renamed symbol. |
| Periodic instance landing way past its `start_time` | `k_cycle_get_64()` baseline (`run_t0`) is captured *before* worker spawn. If one worker's dispatch starves the hart, instance waits. Check pinning + `parallel_<op>` fanout. |
| Deadlock partway through walk | Cross-job edge pointing forward in `entry_id` (forbidden by sort). Check `ingest_xpurt_schedule.py` validation didn't get bypassed. |

## 8. Spike-first debug recipe

For runtime/architectural bugs (sync, races, ordering) we should **always
debug on spike first** — it is deterministic, fast to iterate (~seconds vs.
~minutes per FireSim run), and rules out FireSim bitstream / RTL issues:

```bash
# 1. Profile each network on spike to get cycle counts.
runtime/scripts/profile_remote.sh --target spike --hw RVV   --models dronet,yolov8_nano
runtime/scripts/profile_remote.sh --target spike --hw scalar --models dronet,yolov8_nano

# 2. Generate a schedule using spike-profiled times. See
#    data/toplevel/networks_periodic_dronet_yolov8_spike.json (hardware.profile.target=spike).
python scripts/run_xpurt_schedule.py \
    --top-level-json data/toplevel/networks_periodic_dronet_yolov8_spike.json

# 3. Run the schedule on spike via the xpurt harness.
SCHEDULE_JSON=schedules/scheduled_networks_periodic_dronet_yolov8_spike_profiled.json \
MODELS=dronet,yolov8_nano \
REGISTRY=agents/cores/spike_quad_rvv_scalar.json \
BACKENDS=rvv,scalar  CPU_P_KIND=rvv  CPU_E_KIND=scalar \
RUNNER=spike  AGENTS_POOL_THREADS=2 \
bash zephyr-chipyard-sw/agents/examples/xpurt_demo/run.sh
```

Once spike passes, the same schedule rebuilt against the firesim_rocket_saturn
profile should reproduce on FireSim.

## 9. Validation results (May 2026 checkpoint)

End-to-end runs after landing all the above changes. Bitstream:
`alveo_u250_firesim-dual-rocket-saturn-gemmini` (cpu_p=gemmini@hart0,
cpu_e=RVV@hart1, MP_MAX_NUM_CPUS=2 via overlay).

### Schedule shape A — 1 dronet (period=200 ms) + 1 yolov8_nano

```
schedule:        scheduled_networks_periodic_dronet_yolov8_firesim_greedy_profiled.json
entries:         242    (1 × 30 dronet + 1 × 212 yolov8_nano IR ops, with
                         8 chunk2_c1 zero-cost remapped to dispatch_id=-1)
predicted:       141.63 ms greedy makespan (non-periodic only; matches
                  best-HW critical path 106 ms + greedy gap)
actual FireSim:  135.21 ms (0.95× predicted)
dronet output:   max_abs_err=6 vs PyTorch golden  (< gemmini.atol_override=8) ✓
per-instance wall_cycles_dronet[0] = 39167 cycles = 39.17 ms
```

### Schedule shape B — 4 dronets (period=50 ms) + 1 yolov8_nano

```
schedule:        scheduled_networks_periodic_dronet50ms_yolov8_firesim_greedy_profiled.json
entries:         332    (4 × 30 dronet + 1 × 212 yolov8_nano)
predicted:       188.64 ms (after 3-iter greedy convergence:
                  1→3→4 dronets as actual contention pushes makespan out)
actual FireSim:  183.99 ms (0.98× predicted)
dronet output (last instance): same max_abs_err=6 ✓
per-instance wall_cycles_dronet = [39167, 16292, 27310, 16291]
                                  (varying with cross-kind contention,
                                   not a single shared scalar)
```

### Greedy lower-bound sanity check

Best-HW per-op routing for yolov8_nano on this bitstream:
- Sequential best-HW (sum-of-min):  129.63 ms
- Parallel best-HW (max hart load): 109.90 ms (gemmini 109.90 / rvv 19.73)
- Critical-path best-HW:            106.05 ms
- Greedy actual:                    141.63 ms (33.6% over CP — typical greedy gap)

Heterogeneous routing alone gets us 4× speedup over gemmini-standalone
(422 ms) and 7× over RVV-standalone (999 ms), mostly by routing each op
away from its slow-fallback path: silu/sigmoid go to RVV's LUT, convs
go to gemmini's tiled_conv_auto.

### Predicted-vs-actual diff per-op (yolov8 in shape A)

```
mean (actual − predicted): −0.039 ms      (essentially zero bias)
median actual / predicted: 1.001
ops where actual < predicted: 84 / 204 (41%)
ops with actual ≤ 1 µs (mtime granularity): 0 / 204
```

There's no per-op systematic offset — the dispatch wrapper's
`rdcycle()` reads + `records_[n_++]` bookkeeping live in the same
generated `model.c` for both the profile harness (`multi_demo`) and
the xpurt harness, so they cancel. The 5% makespan-level edge that
xpurt has comes from cross-kind interleaving keeping working sets
warm in L2 (better than profile's sequential single-network passes)
and DRAM-bank parallelism with two harts active.

### Per-target overlay summary (post-checkpoint)

`harness_xpurt/prj.conf` no longer sets `CONFIG_MP_MAX_NUM_CPUS`.
`xpurt_demo/run.sh` selects an overlay for every target and fails
fast if none is found:

| RUNNER | overlay | MP_MAX_NUM_CPUS | spike_args |
|---|---|---|---|
| spike  | `spike_quad.conf`                  | 4 | `--isa=rv64gcv_zicntr [--extension=gemmini]` |
| firesim (default) | `firesim_chipyard.conf`            | 4 | n/a |
| firesim + gemmini | `firesim_chipyard_dual_gemmini.conf` | 2 | n/a |
