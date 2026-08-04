# SmolVLA Vision v3 — Profiling Infrastructure

## Correct Profiling Methodology

The scheduler's cost model requires **wallclock around QnnGraph_execute** — the same
measurement the generated runtime emits in its trace. This captures launch + RPC +
dispatch + compute, which is the true cost the runtime pays per segment.

**Wrong:** `qnn-net-run --profiling_level detailed` → parses QNN profiling CSV for
BACKEND/NETRUN EXECUTE events. This captures only the backend's self-reported compute
time, not the full dispatch overhead the runtime actually sees.

**Right:** `profile_segments.cpp` → directly calls `QnnGraph_execute` via `dlopen`'d
backend lib on a pre-built **context binary** (`.bin`), measures `steady_clock::now()`
wallclock per call over N iterations.

## Pipeline (3-Stage)

```
ONNX → DLC → quantized DLC → context binary (.bin) → profile_seg wallclock
```

### Stage 1+2: Host (Docker)

```bash
# ONNX → DLC (snpe-onnx-to-dlc, NOT qnn-onnx-converter which produces .cpp/.bin model-lib)
snpe-onnx-to-dlc --input_network <seg>.onnx --output_path <seg>.dlc

# DLC → quantized DLC
qairt-quantizer --input_dlc <seg>.dlc --output_dlc <seg>_quantized.dlc \
    --input_list <calib_list> --act_bitwidth 8 --weights_bitwidth 8
```

### Stage 3: On-Board (QRB5165)

```bash
# quantized DLC → per-backend context binary
qnn-context-binary-generator \
    --backend $QNN/lib/target/libQnn<Backend>.so \
    --model $QNN/lib/target/libQnnModelDlc.so \
    --dlc_path <seg>_quantized.dlc \
    --binary_file ctx_<seg>__<Backend> --output_dir .
# Produces: ctx_<seg>__<Backend>.bin
```

### Stage 4: On-Board — Profiling

```bash
# Build the profiler (once)
g++ -std=c++2a -O2 -pthread -I$QNN/include profile_segments.cpp -o profile_seg -ldl

# Profile one segment
export LD_LIBRARY_PATH=$QNN/lib/target
export ADSP_LIBRARY_PATH="$QNN/lib/hexagon-v66;/dsp/cdsp;/dsp"
./profile_seg ctx_<seg>__<Backend>.bin $QNN/lib/target/libQnn<Backend>.so 50
```

Output (one JSON line to stdout):
```json
{"dlc":"ctx_...bin","backend":"libQnnDsp.so","status":"ok","graph":"...",
 "iters":50,"init_us":1234.5,
 "mean_us":145000.00,"median_us":144500.00,"min_us":140000.00,
 "max_us":152000.00,"std_us":3200.00,"p99_us":151000.00}
```

## Key Files

| File | Purpose |
|------|---------|
| `qnn_models/runtime/profile_segments.cpp` | C++ profiler: loads context binary, runs graphExecute N times, reports wallclock stats |
| `qnn_models/runtime/profile_sweep.sh` | Orchestrates sweep: pushes DLCs, builds ctx binaries, runs profiler per (seg,backend) |
| `qnn_models/runtime/build_subdlcs.sh` | 3-stage ONNX→DLC→quantized DLC→context binary pipeline |
| `qnn_models/runtime/generate_runtime.py` | Generates C++ runtime from a schedule (dispatch_table.h + runtime_main.cpp) |
| `qnn_models/runtime/build_and_run.sh` | Compiles + runs generated runtime on board, captures trace |

## Applying to SmolVLA Vision v3

The v3 vision encoder has 49 segments (25 DSP + 24 CPU) in a linear chain.
Existing DLCs are at `vision_slices_v3/dlc/`:
- DSP: `dsp_seg_XX_quantized.dlc` (25 segments, conv1x1-rewritten, int8)
- CPU: `cpu_seg_XX.dlc` (24 segments, fp32)

To profile:
1. Push quantized DLCs to board
2. Generate context binaries per (segment, backend):
   - DSP segments → try libQnnHta.so, libQnnDsp.so, libQnnCpu.so
   - CPU segments → libQnnCpu.so only (softmax/tanh ops)
3. Run `profile_seg` on each successful context binary
4. Aggregate into `segment_perf.json`

## Output Format for XPURT Scheduler

The scheduler consumes `gen/profile/<HW>/<target>/<model>/<model>.int8/topo_0/results.csv`:
```
dispatch_id,module_name,mean_time,mean_unit
0,dsp_seg_00,150000.00,us
1,cpu_seg_00,42671.00,us
...
```

The `emit_vision_v3_profile.py` script reads the `segment_perf.json` and writes these CSVs.

## HTA Status (v66) — COMPLETE

All 50 Conv ops pass HTA context-binary generation and profiling (2026-05-10).
The key requirement was proper int8 quantization (qairt-quantizer with --input_list).

Shape families (all profiled successfully):
- (768, 768, 1, 1) × 12 — output projection: ~3.1–3.3 ms each
- (3072, 768, 1, 1) × 12 — fc1: ~9.3 ms each
- (768, 3072, 1, 1) × 12 — fc2: ~15.5–16.2 ms each
- (2304, 768, 1, 1) × 12 — QKV projection: ~6.1–6.2 ms each
- (768, 3, 16, 16) × 1 — patch embed Conv16×16: 289 ms (HTA-unfriendly shape)
- (960, 12288, 1, 1) × 1 — final head projection: 119 ms (too large for HTA tiling)

Per-segment HTA totals (sum of constituent convs):
- B-type (odd, proj+fc1): ~12.4 ms → 7× faster than DSP (88 ms)
- A-type (even, fc2+QKV): ~22 ms → 5× faster than DSP (156 ms)
- seg_00 (patch+QKV): 295 ms → CPU wins (68 ms)
- seg_24 (fc2+head): 135 ms → CPU wins (78 ms)

## Environment

- Board: QRB5165 (Hexagon v66), reachable at `$QNN_BOARD_HOST` (default: root@<BOARD_IP>)
- QNN SDK: QAIRT 2.45.0.260326
- On-board QNN: `/root/qairt/` (libs at `lib/target/`, tools at `bin/target/`)
- Docker image: `qnn-convert` (host conversion/quantization)
- Python env: `<PATH_TO_CONDA_ENV>/bin/python`
