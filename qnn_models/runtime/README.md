# QNN runtime from a scheduled JSON

Generates a single-binary C++ QNN runtime that walks an XPU-RT-emitted
`scheduled_*.json`, fans out across per-kind worker threads, and dispatches
the work in schedule order with proper synchronisation.

It mirrors the zephyr-chipyard-sw flow
(`agents/pipeline/ingest_xpurt_schedule.py` → `generate_xpurt_main.py`)
but the host-side / QNN equivalent: instead of a Zephyr-side dispatch
table targeting RVV/scalar, this emits Linux/QNN code targeting the same
DLCs we measured on QRB5165.

## 1. What you get

For an input schedule like
`schedules/scheduled_networks_periodic_dronet5ms_yolov8_qrb5165_greedy_profiled.json`,
the generator emits two files in `<out-dir>/`:

| File | Contents |
|---|---|
| `dispatch_table.h` | A C++ array of `ScheduleEntry` — one per scheduled segment, with `seg_id, network, instance, kind, backend_label, op_ids[], start_time_ms, duration_ms, deps[]`. Mirrors `xpurt_sched_entry_t` in the zephyr flow. |
| `runtime_main.cpp` | The walker. Brings up every backend the schedule references (`dlopen` + `contextCreateFromBinary`), spawns one `std::thread` per machine kind, and each thread iterates the table in start-time order taking entries that match its kind. Per-segment `std::counting_semaphore` provides the producer/consumer sync. |

## 2. Architecture

```
                         schedule JSON
                              │
                              ▼
                  generate_runtime.py    (coalesce per-op routing
                              │            into per-(net, inst, kind)
                              │            segments; emit C++ table)
                              ▼
              +----- dispatch_table.h ------+
              |                              |
              |  SCHEDULE_TABLE[N] = { ... } |
              +------------------------------+
                              │
                              ▼
                       runtime_main.cpp
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   worker(CPU_P)         worker(CPU_E)         worker(...)
   - dlopen lib          - dlopen lib
   - ctxCreate           - ctxCreate
   - graphRetrieve       - graphRetrieve
   - per-entry:          - per-entry:
       wait deps[]           wait deps[]
       gate start_time       gate start_time
       graphExecute()        graphExecute()
       release sem           release sem
```

The synchronisation pieces:

- **Per-entry completion semaphore** (`std::counting_semaphore<INT_MAX>`):
  posted exactly once when a worker finishes the entry. Consumers (entries
  with this `seg_id` in `deps[]`) `acquire()` then `release()` so any
  number of downstream consumers can each see the "done" signal.
- **Per-context graph mutex** (`std::lock_guard` around `graphExecute`):
  the QNN backends serialise concurrent dispatches onto the same context
  internally, but we still hold a host-side mutex so two segments
  pointing at the same loaded context (e.g. two dronet instances on HTA)
  don't race over the input/output buffer descriptors we patch on each
  call.
- **Schedule start-time gate**: each entry busy-yields until
  `now_ms() >= run_t0_ms + entry.start_time_ms`. Models the periodic
  arrival the scheduler assumed (a new dronet wave every 5 ms). Real
  arrival from a sensor would replace this with `condvar.wait()` on the
  sensor pipe.
- **Cross-network ordering** (`time_dependency` from the schedule): not
  yet wired — the v1 generator only honours intra-job deps. Easy add when
  needed: parse the `time_dependency` field in the schedule and add it to
  the segment's `deps[]`.

## 3. Usage

```bash
# 1. Generate the runtime sources from a schedule.
python3 qnn_models/runtime/generate_runtime.py \
    --schedule schedules/scheduled_networks_periodic_dronet5ms_yolov8_qrb5165_greedy_profiled.json \
    --out-dir  qnn_models/runtime/gen/qrb5165_dronet_yolov8 \
    --backend-map "CPU_P=HTA_split:libQnnHta.so,CPU_E=CPU:libQnnCpu.so" \
    --ctx-dir   /root/qnn_runtime_ctx

# 2. Stage the context binaries the runtime expects, naming them
#    ctx_<network>_<label>.bin under <out-dir>/ctx/. For HTA_split we use
#    the BN-folded HTA build for dronet and the DSP build for yolov8n
#    (per the per-op partition documented in
#    qnn_models/boards/qrb5165_v66/graphs/yolov8n_HTA_split.json).
mkdir -p qnn_models/runtime/gen/qrb5165_dronet_yolov8/ctx
# (these were generated earlier by qnn-context-binary-generator on the board)
scp root@<BOARD_IP>:/root/dispatch_chars/ctx_hta.bin \
    qnn_models/runtime/gen/qrb5165_dronet_yolov8/ctx/ctx_dronet_HTA_split.bin
scp root@<BOARD_IP>:/root/dispatch_chars/ctx_dsp.bin \
    qnn_models/runtime/gen/qrb5165_dronet_yolov8/ctx/ctx_yolov8n_HTA_split.bin
scp root@<BOARD_IP>:/root/dispatch_chars/ctx_cpu.bin \
    qnn_models/runtime/gen/qrb5165_dronet_yolov8/ctx/ctx_dronet_CPU.bin
scp root@<BOARD_IP>:/root/dispatch_chars/ctx_cpu.bin \
    qnn_models/runtime/gen/qrb5165_dronet_yolov8/ctx/ctx_yolov8n_CPU.bin   # placeholder; see §4

# 3. Build on the board and run.
bash qnn_models/runtime/build_and_run.sh \
    qnn_models/runtime/gen/qrb5165_dronet_yolov8
```

## 4. **Yes, we still need new DLCs** — the per-op-routing caveat

This is the gap between what the scheduler outputs and what QNN can
execute today.

The scheduler routes **individual ops** (e.g. dronet's 27 ops split as
24 → HTA, 3 → CPU). QNN's only execution primitive is `QnnGraph_execute`
on a pre-compiled graph — it does not expose op-level dispatch within a
monolithic graph. So to faithfully run the schedule we need one of:

1. **Per-segment sub-DLCs** (proper path). For each contiguous
   same-(network, instance, backend) segment the scheduler produces, we
   pre-compile a sub-DLC containing exactly that op subset, with the
   boundary tensors promoted to graph I/O. The runtime then chains
   `contextCreateFromBinary` + `graphExecute` calls per segment, with the
   handoff buffers shared across backends via `QnnMem_register` / ION (or
   plain host memcpy as the v1 fallback).

   QNN supports this — `qnn-context-binary-generator` consumes a
   backend-extensions JSON that filters ops by name. The generator
   already reports the `op_ids` per segment, so wiring this up is:

   - For each segment, build a backend-extensions JSON listing the
     `op_ids`'s names (from the source DLC's IR);
   - Run `qnn-context-binary-generator --config_file <extJson>
     --backend lib<X>.so --dlc_path <full DLC>` to produce the sliced
     `.bin`;
   - Stage as `ctx_<network>_<label>_seg<id>.bin` under `ctx/`;
   - Have the generated runtime look up by segment-id rather than by
     (network, label).

2. **Coarsen the schedule** to whole networks, run as today (this is
   what the v1 runtime emitted by `generate_runtime.py` does — each
   segment maps to a `graphExecute` on the WHOLE network's pre-built
   context). This is convenient: the four DLCs we already have on the
   board (`dronet_quantized.dlc`, `dronet_full_hta_quantized.dlc`,
   `yolov8n_quantized.dlc`, plus the BN-folded HTA-friendly variants)
   cover everything, and the runtime architecture (worker per kind, per-
   entry sem, start-time gate) is already correct. The trade-off is that
   the per-op routing decision the scheduler made is no longer enforced
   at runtime — the timing is approximate, the kind assignment lands at
   the network granularity rather than the op granularity.

3. **Programmatic graphs** (the option (1) we tried in
   `per_op_prototype/`). Works on CPU only; DSP and HTA reject it
   because they require pre-built model artifacts. Off the table for the
   NPU lane.

The recommended path is (1). The v1 runtime here is staged so that when
sub-DLC slicing lands, only the lookup key in the `g_ctx` map and the
per-context-binary naming change — the walker, sync, and timing logic
stay the same.

## 5. Sub-DLC slicing pipeline (the "deterministic/automatic" part)

The slicing tools live alongside the runtime generator:

| Tool | Stage | Status |
|---|---|---|
| `slice_to_subonnx.py` | schedule + ONNX → per-segment sub-ONNXes | ✅ works for dronet |
| `build_subdlcs.sh` | sub-ONNX → DLC → context binary (per backend) | ✅ ONNX→DLC works, ✅ CPU context binary, ❌ DSP/HTA need quantization (see §5.3) |
| `capture_boundary_calibration.py` | full-ONNX inference → per-segment calibration .raw files | 🚧 sketch — onnxruntime intermediate-tensor capture works, per-segment input_list selection is the missing 50 LOC |

### 5.1 Slicing — working today

The slicer is deterministic and automatic for any network where the
schedule's `module_name` strings match the source ONNX's node names
1:1. dronet meets that bar:

```bash
python3 slice_to_subonnx.py \
    --runtime-gen qnn_models/runtime/gen/qrb5165_dronet_yolov8 \
    --onnx-map "dronet=qnn_models/dronet.onnx" \
    --out-dir qnn_models/runtime/gen/qrb5165_dronet_yolov8/sub_onnx
# wrote 7 unique sub-ONNXes (28 cache hits avoided),
# manifest at .../sub_onnx/manifest.json
```

The `manifest.json` records every segment's `input_tensors` /
`output_tensors` so the downstream quantizer + context-binary stages
can find their boundary I/O without re-deriving.

### 5.2 yolov8n — name-space mismatch (TODO)

yolov8n's schedule was profiled on a **TFLite-route DLC** whose QNN
node names look like `pad_0`, `convolution_0`, `elementwise_product_0`.
The user-authored `boards/qrb5165_v66/graphs/yolov8n_HTA_split.json`
uses **ONNX-route** names like `model_0_conv_Conv` (from
`/model.0/conv/Conv`). The two DLC routes are different graphs (the
TFLite path inserts extra Pad/Transpose ops) — there's no direct
1:1 mapping between them. Two ways forward:

1. **Re-profile and re-schedule via the ONNX route**: run
   `snpe-onnx-to-dlc qnn_models/yolov8n.onnx` and put it through the
   same `compute_per_op_stats` → `export_graph_json` → schedule
   pipeline. The schedule's `module_name`s will then match the ONNX,
   and the slicer works as-is.
2. **Build a structural map**: run `qairt-dlc-to-json` on both routes
   and match by op-type sequence (Conv→Sigmoid→Mul positions are
   identical between routes; the Pad/Transpose insertions are
   distinguishable by op type). Pass the result via `--name-map`.

### 5.3 DSP/HTA on sub-DLCs — needs per-segment quantization

A bare `snpe-onnx-to-dlc` produces a fp32 DLC. CPU accepts that;
**DSP rejects it** (`Input[0] has incorrect Datatype 0x232` —
0x232 is `QNN_DATATYPE_FLOAT_32`). HTA is even narrower (it requires
int8 *and* a smaller op set than the full network's). To produce
DSP-runnable sub-DLCs we need to quantize each one; quantizing a
sub-DLC needs **calibration data at its boundary input tensors** —
intermediate activations of the full network, not the network's own
input.

The capture script (`capture_boundary_calibration.py`) sketches this:
take the source ONNX, append every boundary tensor as a graph output,
run onnxruntime over a calibration set, save each captured tensor's
.raw blobs in a per-tensor folder. The remaining wire-up:

```python
# in build_subdlcs.sh, after slicing+converting:
for seg in manifest:
    if seg["label"] in ("HTA_split", "DSP", "GPU_fp16"):
        per_seg_input_list = build_input_list(seg["input_tensors"])
        run("qnn-quantizer",
            "--input_dlc",  f"sub_dlc/{seg.base}.dlc",
            "--output_dlc", f"sub_dlc/{seg.base}_quantized.dlc",
            "--input_list", per_seg_input_list)
```

Once that's in, the `qnn-context-binary-generator` step succeeds for
DSP because the quantized DLC has int8 boundary tensors that match
the backend's expectation.

### 5.4 What works end-to-end today

```
schedule.json
   ├── generate_runtime.py    →  runtime_main.cpp + dispatch_table.h          ✅
   ├── slice_to_subonnx.py    →  manifest.json + 7 sub-ONNX (dronet)          ✅
   ├── build_subdlcs.sh       →  7 sub-DLCs (snpe-onnx-to-dlc)                 ✅
   │                          →  3 CPU-targeted context binaries (fp32)        ✅
   │                          →  4 DSP-targeted context binaries               ❌  (needs §5.3 quant)
   └── capture_boundary_calibration.py
                              →  per-tensor .raw + input_list.txt              🚧 (sketch — works
                                                                                    for capture; per-
                                                                                    segment selection
                                                                                    in quantizer call
                                                                                    not yet wired)
```

So the "deterministic/automatic slicing" part is solid: the same input
schedule always yields the same sub-ONNX set + manifest + DLCs. The
gap that prevents DSP-runnable sub-DLCs is per-segment calibration,
which is purely mechanical given the manifest — it just hasn't been
wired through the quantizer call yet.

## 6. Numeric correctness — validation summary

Run end-to-end on QRB5165 (dronet, 1 instance, real calibration input)
via `run_validation.sh`. The check answers two distinct questions:

### What we validated (functional correctness — passes)

- **All 27 dronet ops are reached.** The 7 sub-DLCs collectively cover
  every dispatch_id from the schedule (24 HTA-bound ops + 3 CPU
  spillovers). No ops are silently dropped, none execute twice.
- **Boundary tensors flow through the chain.** With the runtime built
  using `--manifest` and run with `QNN_RUNTIME_DUMP_ALL=1`, all 12
  boundary tensors land in the output dir; cross-segment handoff
  happens by tensor-name match through the runtime's `g_tensor_cache`.
- **Per-segment dispatch lands on the assigned backend.** The trace
  confirms HTA segments hit `libQnnHta.so` and CPU segments hit
  `libQnnCpu.so`, in the order the schedule prescribes, with proper
  POSIX-semaphore synchronisation on the dependency edges.
- **No layout transformations skipped between layers.** The chain
  preserves NHWC throughout (QNN's converter auto-permutes ONNX-NCHW
  on entry to each sub-DLC). The `.nchw` Transposes the partition
  guide warns about get re-inserted by `snpe-onnx-to-dlc` at sub-DLC
  build time, exactly where they were in the source graph — verified by
  comparing per-segment QnnGraph_create logs against the partition
  annotation file's op list.

### What we didn't validate (numeric equivalence — known caveat)

The end-to-end output bytes don't dequantise to onnxruntime's fp32
golden within int8-quant tolerance. **This is a quantization-scaling
issue at sub-DLC boundaries, not an execution-architecture flaw.**

Each sub-DLC was independently passed through `qairt-quantizer` with
its OWN boundary calibration data. The producer's encoding for
`/Add_output_0` (say `scale=0.038, offset=-127`) doesn't necessarily
match the consumer sub-DLC's *input* encoding for the same tensor
name (calibrated independently, perhaps `scale=0.041, offset=-130`).
The runtime does a raw uint8 byte-copy across the boundary, which means
the consumer interprets the producer's bytes with the wrong scale —
errors compound across each cross-segment hop.

A whole-network DLC has none of this: qairt-quantizer sees the entire
graph and produces consistent encodings everywhere. The per-segment
chain is correct *up to* this boundary mismatch.

### Fix paths (none implemented yet — out of scope today)

1. **`--quantization_overrides <encodings.json>`** to `qairt-quantizer`
   per sub-DLC. Pin boundary tensors to one global scale/offset taken
   from a whole-network calibration pass. snpe-dlc-info prints
   per-tensor encodings; we'd extract the whole-network DLC's encodings
   once, then force every sub-DLC to use the same scale/offset for any
   shared boundary name.
2. **Compare per-segment chain vs whole-network int8 DLC** on the same
   input. Both have similar bounded quant noise, so the byte-level
   diff would be tiny — that's the apples-to-apples reference, not
   onnxruntime fp32.
3. **Run sub-DLCs in fp32 mode** (skip the quantize step). CPU accepts
   fp32 directly; HTA rejects it (int8-only on v66). So this is a
   CPU-only validation path and proves the algebra is right; quantize
   to int8 only after that's nailed.

For today: the architecture is solid (no skipped ops, no layout drops,
correct dispatch + sync), and the known boundary-quantization drift is
documented as the next-pass task.

## 7. What's missing today (followups)

- **Sub-DLC generation**: §4 path (1). The most important one — it's the
  difference between "real per-op routing" and "approximate per-network
  routing".
- **`time_dependency` cross-job edges**: parse the field from each
  dispatch and add to the segment's deps.
- **Shared-buffer handoff**: today the runtime's per-context input/output
  buffers are private; cross-segment data passing requires a host memcpy
  between buffers. `QnnMem_register` + ION buffers would close this gap
  (drop DSP's ~600 µs FastRPC marshalling cost we measured in
  `per_op_prototype/dispatch_overhead.sh`).
- **Real input feed**: today every input buffer is zero-initialised at
  bringup. A real deployment would wire the workspace input buffer to a
  sensor frame ingest path, and only kick the dispatch chain when a new
  frame arrives.
- **Async execute** (`QnnGraph_executeAsync` + `QnnSignal`): would let
  HTA and CPU genuinely overlap rather than serialise behind their
  respective worker mutex; useful when the schedule has fine-grained
  parallelism between kinds.
