# Multi-Graph Context Refactor Plan

> **STATUS (2026-05-12): IMPLEMENTED BUT NOT IN USE.**
>
> The refactor itself works end-to-end: bundles build, the runtime walks
> the schedule, all dispatches return `QNN_SUCCESS`. **However**, loading
> a multi-graph context binary silently crashes the cDSP user process
> domain on this firmware (QRB5165 v66). The host runtime cannot see the
> crash and reports a clean run while the DSP has been dead for most of
> it. Confirmed via on-board ramdumps and kernel `fastrpc_mem_map_to_dsp`
> failures — see `qnn_models/QRB5165_MULTIGRAPH_CDSP_CRASH_FORENSICS.md`.
>
> **Do not enable in production.** The canonical schedule continues to
> use single-graph contexts at budget=9 (2561 ms / 1.24× speedup).
>
> The pipeline (`bundles_from_schedule.py` → on-board build script →
> `deploy_and_run.sh`) is preserved end-to-end so that, when the
> firmware is fixed (or we move to a chip with a usable op-package),
> enabling multi-graph is a one-line schedule-side change. The
> ~2272 ms / 1.40× target documented below remains the *theoretical*
> ceiling — not the current achievable.

## Why

QRB5165 v66 firmware caps simultaneous DSP/HTA contexts at ~30 and
cumulative `contextCreateFromBinary` calls at ~45 per FastRPC PD. Today's
runtime creates ONE context per .bin file, so 69 unique DSP contexts for
the full HTA-bundle-DSP vision schedule overflow both caps and force us
down to the budget=9 config (1.24× speedup).

The fix is structurally simple and proven (commit 8e0347a, see
`multi_graph_test.cpp`): **`qnn-context-binary-generator` accepts a
comma-separated list of DLCs and produces ONE multi-graph context binary
holding up to ~10 graphs**. The runtime loads it as one context, fetches
each graph by name. Per-graph execute timings match single-graph exactly
(88 ms B-type DSP, 156 ms A-type DSP). Cumulative-create budget goes from
~45 single-graph creates to ~45 multi-graph creates × 10 graphs = ~450
effective graphs available. **The cap effectively disappears.**

## Target

Realize the full **2272 ms / 1.40× speedup** for the v3 vision schedule
(all 23 inner segments routed to HTA-bundle-DSP) — the predicted-but-
previously-unrealizable config.

## File-by-file changes

### 1. `qnn_models/smolVLA/build_multi_graph_ctx.py` (NEW)

A host-side script that takes a list of existing DLCs (already
quantized, e.g. `vision_slices_v3/dlc/*_quantized.dlc`) and produces
multi-graph context binaries chunked at N graphs each.

```
Input:
  --dlc-dir   path/to/dlcs
  --backend   { Cpu | Dsp | Hta }
  --out-dir   path/to/multi_ctx_bins
  --chunk     N (default 10)
  --filter    glob pattern for which DLCs to bundle (e.g. "dsp_seg_*_quantized.dlc")
  --board     ssh target (default $QNN_BOARD_HOST)

Output:
  multi_ctx_bins/multi_<glob>_<chunk_i>__<Be>.bin   (board-side build)
  multi_ctx_bins/manifest.json                       (which graphs went where)
```

The manifest is the crucial bridge: it records every (`graph_name`,
`ctx_bin_path`, `chunk_idx`, `graph_idx_within_chunk`) tuple. The
runtime+staging step uses this to map graph-name → which .bin to load.

Implementation: shell out to `qnn-context-binary-generator` on the board
(it has to run there for the target-specific backend `.so` to be present).
We already do this in `profile_vision_v3_correct.sh` and
`profile_trampolines_dsp.sh`; this script generalizes the same pattern.

### 2. `qnn_models/runtime/generate_runtime.py` — runtime refactor

Today's data structures:
```cpp
struct LoadedCtx {
    Qnn_ContextHandle_t ctx;
    Qnn_GraphHandle_t   graph;        // <- ONE graph per context
    std::string         graphName;
    // ... I/O tensors, buffers ...
};
static std::unordered_map<std::string, std::shared_ptr<LoadedCtx>> g_ctx;
static std::unordered_map<int, std::string> g_seg_to_ctx_key;
```

Needed:
```cpp
struct GraphInfo {
    Qnn_GraphHandle_t                graph;
    std::string                      graphName;
    std::vector<Qnn_Tensor_t>        inputs;
    std::vector<Qnn_Tensor_t>        outputs;
    std::vector<std::vector<uint8_t>> inputBufs;
    std::vector<std::vector<uint8_t>> outputBufs;
    std::vector<std::vector<uint32_t>> dimStorage;
    std::vector<std::string>          nameStorage;
    std::mutex                       m;   // guards graphExecute
};
struct LoadedCtx {
    std::string                        label;
    std::shared_ptr<SharedBackend>     sb;
    Qnn_ContextHandle_t                ctx;
    std::unordered_map<std::string, std::unique_ptr<GraphInfo>> graphs;  // by graph name
};
static std::unordered_map<std::string, std::shared_ptr<LoadedCtx>> g_ctx;
static std::unordered_map<int, std::pair<std::string, std::string>> g_seg_to_graph;
    // seg_id -> (ctx_key, graph_name)
```

Key changes inside `bringup()`:
- Walk ALL graphs in the binary (we already enumerate them in `multi_graph_test.cpp`)
- For each: `graphRetrieve(ctx, name)` + allocate I/O buffers, store in `bc.graphs[name]`

Key changes in worker:
- Today: `auto bc = ensure_ctx_loaded(ctx_key); graphExecute(bc.graph, ...)`
- New: `auto [ctx_key, graph_name] = g_seg_to_graph[seg_id];
         auto bc = ensure_ctx_loaded(ctx_key);
         auto& gi = bc->graphs[graph_name];
         std::lock_guard<std::mutex> lock(gi.m);
         graphExecute(gi.graph, gi.inputs.data(), ..., gi.outputs.data(), ...);`

The per-segment lock moves from `LoadedCtx::m` → `GraphInfo::m` so two
workers can concurrently execute different graphs within the same ctx
(matches reality — graphExecute is per-graph, the context is just a
container).

Eviction stays at the ctx level: free a whole multi-graph ctx and all
its graphs go with it. Use counts now tracked per (ctx_key, graph_name):
when the LAST graph in a ctx has remaining_uses=0, the ctx becomes
LRU-evictable.

### 3. `qnn_models/smolVLA/stage_v3_bundles_ctx.py` — symlink wiring

Today the script maps each dispatch's `(seg_name, hw)` to one .bin file
on disk. With multi-graph, multiple dispatches share one .bin.

New behaviour:
- Read the manifest from build_multi_graph_ctx.py
- For each dispatch, find the (ctx_bin_path, graph_name_in_bin) tuple
- Emit symlinks: `ctx_<network>_<label>_seg<id>__<Be>.bin → <ctx_bin>`
  (multiple symlink names point at the same source ctx_bin)
- Also emit a sidecar JSON consumed by the runtime that maps each
  symlink name → the graph name to retrieve inside that ctx binary

OR (simpler): bake the (ctx_key, graph_name) mapping directly into the
generated `dispatch_table.h`. The dispatch table already has a column
per entry — add `const char* graph_name` and use it instead of the
hardcoded `bc.graphName`.

### 4. `qnn_models/smolVLA/build_v3_bundles.py` — minor

No functional change. Optionally add a comment noting that the bundle
plan's dispatch-graph + results.csv outputs are now consumed by both
the schedule generator AND `build_multi_graph_ctx.py` (which uses the
list of unique DLCs the schedule references to decide chunking).

### 5. New helper script `qnn_models/smolVLA/stage_multi_graph_pipeline.sh`

End-to-end wrapper: from the bundle plan to a runnable runtime.
```
build_v3_bundles.py            # plan + results.csv
run_xpurt_schedule.py          # schedule
build_multi_graph_ctx.py       # multi-graph ctx binaries on board
generate_runtime.py            # emits runtime + dispatch_table.h
stage_v3_bundles_ctx.py        # symlinks
build_and_run.sh               # compile + run on board
```

## Validation strategy

Each step has a test:

1. **build_multi_graph_ctx.py**: assert produced .bin files exist + match
   expected graphs (via systemContextGetBinaryInfo). Reuse the existing
   `multi_graph_test.cpp` to confirm per-graph correctness and timings.

2. **Runtime refactor**: a "regression run" with the existing budget=9
   vision schedule but using multi-graph ctx layout (chunk size 1, so
   each ctx still holds one graph). Should produce IDENTICAL trace to
   commit 8e0347a's `runs/v3_bundles_dsp9/trace.csv`. If not, fix bugs
   before scaling up.

3. **Scaled run**: same schedule but all 23 segments DSP-tramp, chunk
   size 10. Expect ~2272 ms wallclock matching the prediction.

4. **Plot update**: extend `plot_v3_bundles_vs_baseline.py` with a new
   strip showing the realized multi-graph run.

## Rollback path

The refactor is gated by the manifest-based loading. If the new path has
a bug, set chunk size to 1 in `build_multi_graph_ctx.py` and the runtime
falls back to the legacy "one graph per ctx" layout — identical to
today's behaviour. We can also keep the old `generate_runtime.py`
checked in at the current SHA and revert.

## Risk and unknowns

| Risk | Likelihood | Mitigation |
|---|---|---|
| Per-graph perf degrades when ctx holds many graphs | Low — multi_graph_test showed identical timings at N=10 for DSP and HTA | Re-measure each segment in its multi-graph ctx as part of bringup validation |
| HTA conv multi-graph hits a smaller limit (we tested 10, didn't push higher) | Medium | If HTA caps lower, just use a smaller chunk size for HTA than DSP |
| graphRetrieve fails on certain DLCs in a chunk because of name collisions | Low — graph names come from the DLC's internal `graph_name`, unique per source segment | Detect at manifest-build time; rename if conflict |
| Total context binary size exceeds the firmware's "loadBinary" allocator | Medium — we saw 18-graph multi-bin (64 MB) fail to load while 10-graph (35 MB) worked | Cap chunk size at 10 by default; can lower per backend |

## Estimated effort

| Step | Time |
|---|---|
| 1. build_multi_graph_ctx.py | ~45 min |
| 2. Runtime refactor (GraphInfo/LoadedCtx + worker + bringup) | ~90 min |
| 3. Staging script update | ~20 min |
| 4. Wrapper + plumbing | ~15 min |
| 5. Validation runs + debug | ~60 min |
| **Total** | **~3.5 hours** |

## After landing

Re-run end-to-end with the unconstrained DSP-tramp schedule (all 23 inner
segs DSP-bundle, 141 dispatches). Expected outcome:

| Config | Pre-refactor | Post-refactor (predicted) |
|---|---:|---:|
| Vision-only makespan | 2563 ms (1.24×) | **~2272 ms (1.40×)** |
| Unrolled10 full-pipeline makespan | 4833 ms (1.17×) | **~4540 ms (1.24×)** |
| DSP firmware ctx count | 27 | **~3** (well under 30 cap) |
| HTA firmware ctx count | 18 | **~5** (well under 30 cap) |

After that, decode/prefill RMSNorm is still a separate blocker for
moving them off CPU — multi-graph helps with ctx accounting but doesn't
fix the QNN_DSP_BACKEND_ERROR_OP_PACKAGE_NOT_FOUND for the
Pow/ReduceMean/Sqrt/Div decomposition.
