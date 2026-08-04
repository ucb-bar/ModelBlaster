# Bundle-Aware v3 Runtime — Findings and Next Steps

Status: end-to-end pipeline working; the budgeted-context runtime ran the
v3 bundle-aware schedule to completion with measured wall close to
prediction. See `plots/v3_bundles_vs_baseline.png` for the full comparison.

> **Multi-graph context binaries are NOT enabled in the canonical run.**
> The refactor exists end-to-end (build pipeline + runtime support) but
> triggers a cDSP user-PD crash on this firmware. Documented in
> `MULTIGRAPH_REFACTOR_PLAN.md` and `qnn_models/QRB5165_MULTIGRAPH_CDSP_CRASH_FORENSICS.md`.
> The working demo at 2561 ms / 1.24× uses single-graph contexts (one
> graph per `.bin` file).

## Canonical run

The "budgeted" runtime configuration that we'll keep as the working
demo is **eager DSP-tramp budget=9**:

| Metric | Value |
|---|---:|
| Schedule | `schedules/scheduled_networks_smolvla_vision_v3_bundles_qrb5165_greedy_profiled.json` |
| Dispatches | 97 (13 CPU-mono, 3 HTA-bundle-CPU, 9 HTA-bundle-DSP, 24 cpu_seg, 49 inner = 97 total) |
| Predicted | 2564.9 ms |
| Measured | **2561.0 ms** (error -0.15%) |
| Speedup vs CPU-mono baseline | 1.24× (3172 ms → 2561 ms) |
| DSP contexts loaded | 27 (eager, well under the 30-context simul. limit) |

Build/run command:
```bash
# Plan + scheduler
python qnn_models/smolVLA/build_v3_bundles.py --dsp-tramp-budget 9
python scripts/run_xpurt_schedule.py \
    --networks-json data/toplevel/networks_smolvla_vision_v3_bundles_qrb5165.json \
    --solver greedy --profiled

# Runtime sources
python qnn_models/runtime/generate_runtime.py \
    --schedule schedules/scheduled_networks_smolvla_vision_v3_bundles_qrb5165_greedy_profiled.json \
    --out-dir qnn_models/runtime/gen/qrb5165_smolvla_v3_bundles \
    --backend-map CPU_P=HTA:libQnnHta.so,CPU_E=DSP:libQnnDsp.so,CPU_X=CPU:libQnnCpu.so \
    --from-segmented-schedule \
    --ctx-dir /root/qnn_runtime_ctx_v3

# On-board ctx staging + run
python qnn_models/smolVLA/stage_v3_bundles_ctx.py
ssh root@$BOARD "bash /tmp/stage_v3_bundles_ctx.sh"
BOARD_DIR=/root/qnn_runtime_v3_bundles CTX_DIR=/root/qnn_runtime_ctx_v3 \
    bash qnn_models/runtime/build_and_run.sh qnn_models/runtime/gen/qrb5165_smolvla_v3_bundles
```

## QRB5165 firmware limits we hit

| Limit | Value | Symptom |
|---|---|---|
| Max simultaneous DSP contexts | **30** | `QnnDsp <E> Skel side failed when loading context binary` at the 31st `contextCreateFromBinary` |
| Max cumulative DSP context creates | **~45** | Same error, this time after eviction (so the firmware leaks per-create resources, not just per-simultaneous) |
| Max simultaneous HTA contexts | **~32** | `deserialize failed` / `Fail to load cache context error: 5005` at HTA context #33 |

`contextFree()` releases the QNN-side handle but the cDSP firmware doesn't
fully reclaim its slot. So evict-then-reload buys only ~15 additional context
creates beyond the simultaneous cap, then it's done.

## Lazy-load + LRU runtime: works but slow

The runtime's `XPURT_DSP_CTX_BUDGET` / `XPURT_HTA_CTX_BUDGET` env vars enable
lazy loading. Implementation in `qnn_models/runtime/generate_runtime.py`:

- Prefetch the first N (budget) contexts per backend at startup, in
  schedule start_time order.
- Workers call `ensure_ctx_loaded(ctx_key)` synchronously when they hit a
  dispatch whose context isn't loaded. Bringup happens inline on the worker
  thread, blocking that lane.
- `release_ctx_after_use(ctx_key)` decrements a per-ctx use counter; when
  it hits zero the ctx becomes the prime LRU eviction candidate.
- Eviction picks the LRU entry with `remaining_uses == 0` (we never evict an
  in-flight context).

We ran the 14-seg DSP-tramp schedule (42 unique DSP contexts, budget=20
simultaneous → ~22 evict+reload cycles) to completion. Result: 3791 ms
wall, vs 2412 ms predicted. The +1.4 sec gap is the inline bringup cost
(50–200 ms per miss × ~30 misses).

So in practice, **for this QRB5165 schedule the lazy mechanism is slower
than just sizing the schedule to fit eagerly**. It's still architecturally
useful for two scenarios:

1. Mixed workloads where total context count exceeds simultaneous cap but
   stays under cumulative cap, AND the misses can be hidden by overlapping
   compute (see prefetcher idea below).
2. Schedules where we can't predict context demand statically (e.g.,
   periodic instances with input-dependent paths).

## Combining the two open improvements

### A. Continuous prefetcher thread — IMPLEMENTED

`XPURT_PREFETCH_LOOKAHEAD=K` spawns a background thread that watches an
atomic `g_schedule_cursor` (updated by workers as they start dispatches)
and lazy-loads the next K contexts in start_time order. The structures
needed already existed (`g_ctx_bringup_specs`, per-backend LRU vectors,
`g_lazy_ctx_mu`); the prefetcher just calls `ensure_ctx_loaded` ahead
of workers so they hit the fast path. Works as designed in tests.

On its own the prefetcher can't help our specific QRB5165 v66 schedule,
because the bottleneck is the firmware cumulative-create cap (~45 DSP
contexts), not the per-miss bringup latency.

### A'. Periodic backend reset — IMPLEMENTED

`XPURT_DSP_RESET_THRESHOLD=N` and `XPURT_HTA_RESET_THRESHOLD=N` cause
`ensure_ctx_loaded` to call `QnnBackend_free()` + `QnnBackend_create()`
on the relevant backend when cumulative creates since the last reset
exceed N. Workers coordinate via a per-backend `std::shared_mutex`:
they take shared lock during `graphExecute`, the resetter takes
exclusive lock to drain in-flight execs before tearing down.

**Findings:**
- Backend reset DOES reset the firmware's cumulative-create state
  (confirmed: 141/141 dispatches completed where pure-eager and
  pure-lazy both crashed). This validates the per-PD-session model
  hinted at in `QnnDspBackend.h`.
- BUT each reset costs **~5 seconds** of wallclock on QRB5165 v66
  (frees ~24 ctxs, backendFree, backendCreate, then re-bringup the
  next ~24 contexts under exclusive lock — ~150-200 ms per ctx ×
  24 = ~3.6-4.8 sec plus backend recreate overhead).
- With threshold=28 and a schedule that needs 67 unique DSP ctxs:
  2 resets fire, total wall = **15.7 sec** for a 2.3 sec compute
  workload. That's a 0.20× "speedup" — net-negative.
- threshold=42 (closer to the firmware cap): only 1 reset before
  hitting the firmware ceiling anyway, run failed. Likely the firmware
  doesn't quite cleanly reset all per-PD state, OR our software counter
  isn't matched to whatever the firmware counts.

**Conclusion:** on this v66 firmware, the reset cost dwarfs any per-segment
routing savings. The mechanism is correct and useful as a fallback for
future workloads with different cost ratios, but doesn't unlock the
predicted 2272 ms for our v3 vision schedule.

### B. DLC sharing across same-shape segments

The 12 B-type segments all have identical Conv shapes, identical
trampoline-phase op structure, and identical tensor shapes — only their
quantized weight values differ per layer. Right now each gets its own
DLC → its own context binary → its own DSP firmware slot. 12 segments × 3
trampoline phases × 1 conv pair × 2 convs = 60+ contexts that all "look the
same" at the graph-topology level.

Possible approaches to share:

1. **Weight-streaming context binary**: build ONE DLC for the B-type
   trampoline + one for B-type conv1 + one for B-type conv2, with weights
   exposed as graph inputs instead of baked-in initializers. The runtime
   then binds the appropriate weight buffer per dispatch. Cuts B-type DSP
   contexts from 36 to 3. Same trick for A-types (12 segments × 3 phases =
   36 → 3). Total DSP contexts: 6 (down from ~69). HTA convs: 6 (down from 46).
   This brings us well under the firmware cumulative limit.
   - Cost: needs to write the weight-binding harness. QNN supports tensor-
     rebinding at runtime via `set_tensor_buffer`-style APIs (we already
     use it for I/O). Whether the backend allows the weights tensor's
     `clientBuf` to be swapped per call is the question — need a quick
     experiment.

2. **Manually pack multiple layers into one DLC**: extend the trampoline
   ONNX to take a "layer index" input that selects between weight tensors
   pre-loaded as a 3D weight array. Adds a Gather op. May or may not
   compile cleanly on DSP/HTA.

3. **Accept the firmware limit**: route the unfittable segments to CPU-mono
   or HTA-bundle-CPU. This is what `--dsp-tramp-budget=9` does today.

### Combined recipe (A + A' + B)

The prefetcher (A) and reset (A') are now in place. The remaining piece
is B: collapse 69 unique DSP contexts → 6 by sharing DLCs across same-
shape segments, with weights bound at runtime via `set_tensor_buffer`.

**B has been experimentally ruled out for v66.** See `qnn_models/
weight_rebind_test/` for the test rig. Findings:

| Backend | Conv1x1 with weight-as-graph-input | Verdict |
|---|---|---|
| CPU | Compiles + rebind works | ✓ But CPU is the path we're trying to *avoid* |
| DSP | Compiles, but rebind has no effect — DSP snapshots weights at `contextCreateFromBinary` time, ignoring later `clientBuf` updates | ✗ Useless |
| HTA | `qnn-context-binary-generator` segfaults (SIGSEGV/139) when weight is declared as a graph input | ✗ Can't even compile |

The DSP behaviour was surprising: the graph input tensor IS provided per
`graphExecute` call (the binding mechanism works for activation tensors),
but DSP's compiler appears to detect that the tensor is the second input
to a Conv op and re-classifies it as a constant, baking the first call's
bytes into the context state. There's no documented `QnnDspContext_Config`
option to disable this behaviour.

So on QRB5165 v66 / QAIRT 2.45, **DLC sharing via weight rebinding is
fundamentally unavailable for the backends we care about**. The
2272 ms target is therefore unreachable with this firmware.

Paths forward (not pursued):
1. **Newer chip**: HTP on v68+ exposes `QNN_HTP_CONTEXT_CONFIG_OPTION_
   LORA_WEIGHT_SHARING_ENABLED` and `_REFERENCE_WEIGHT_SHARING_ENABLED`
   which are explicit weight-sharing-across-contexts APIs. The whole
   architecture changes there.
2. **Multi-process**: spawn a child process per inference batch so the
   FastRPC PD is fresh each time. Heavy infrastructure for a small win.
3. **Compiler trick**: emit one DLC per layer-shape that has the same
   topology but pre-baked DIFFERENT weights, then alias many segments to
   the same .bin file at the runtime layer. We were already doing this
   conceptually for the conv ops — but each layer's weights are distinct
   so this doesn't reduce ctx count, just file count.

The 1.24× speedup (eager DSP-tramp budget=9) remains the achievable
ceiling on this hardware.

## Other observations

- The trampoline phases on DSP took int8 quantization with random
  calibration data (`gen_trampoline_calibration.py`). Random calibration
  is fine for *timing* profiling because DSP execution time is data-
  distribution invariant; it would be wrong for end-to-end numeric
  validation. When we add that, we'd use boundary tensors captured via
  `gen_vision_slice_calibration.py`.
- The plot shows the over-optimistic conv-only HTA prediction (1083 ms)
  vs the actual realizable best (2272 ms predicted, 2561 ms measured).
  The 2.4× gap between the optimistic prediction and reality came from
  treating Conv compute as the whole segment cost — the trampolines turn
  out to dominate, especially for A-type segments with the heavy Q×K^T
  MatMul.
- The HTA conv-only Gantt strip in the over-optimistic panel (top of the
  plot) is misleading green because the *predicted* HTA segments include
  only the conv times; the actual realizable hybrid execution looks much
  more like the budget=9 measured panel below.
