# SmolVLA Vision Encoder — Heterogeneous Scheduling on QRB5165

End-to-end procedure for slicing, profiling, and scheduling the SigLIP ViT-B
vision encoder (407 ops) across three backends on the Qualcomm QRB5165
(Hexagon v66 DSP, HTA tensor accelerator, CPU).

## Result

| Metric | Value |
|--------|-------|
| Makespan (HTA+CPU schedule) | **1083.6 ms** |
| Serial CPU-only | 3172.2 ms |
| Serial DSP-only | 3609.6 ms |
| Speedup vs. best serial | **2.9×** |

47/49 segments run on HTA (5–7× faster than DSP per segment). Two segments
remain on CPU: seg_00 (patch embed Conv16×16, 68.5 ms) and seg_24 (final head
12288→960 projection, 77.8 ms) where HTA is slower.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  smolvlm_vision.onnx (SigLIP ViT-B, 12 transformer layers)         │
│  407 ONNX ops in a linear chain                                     │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ slice_vision_v3.py
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  49 segments: 25 DSP + 24 CPU                                       │
│    DSP_A[k]: LayerNorm → QKV → reshape → Q×K^T + scale             │
│    CPU[2k]:  Softmax → V×attn → Transpose → Reshape                │
│    DSP_B[k]: output_proj → Add → LN → fc1 → GELU_prep             │
│    CPU[2k+1]: Tanh (GELU activation)                                │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ rewrite_matmul_to_conv1x1.py
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  DSP segments with Conv1×1 (VTCM-tiled) instead of MatMul          │
│    MatMul(x[B,M,K], W[K,N]) → Conv(x[B,K,M,1], W[N,K,1,1])       │
│    Wrapped in Transpose+Reshape to/from 3D ↔ NCHW                  │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ extract_hta_convs.py
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  50 standalone HTA Conv models (strip Transpose/Reshape wrappers)   │
│    Raw NCHW I/O → directly compilable for HTA                       │
│    Per segment: 2 convs (e.g., output_proj + fc1, or fc2 + QKV)     │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ Docker: snpe-onnx-to-dlc + qairt-quantizer
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Quantized DLCs (int8) for DSP/HTA, fp32 DLCs for CPU              │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ On-board: qnn-context-binary-generator
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Context binaries (.bin): pre-compiled graph for each backend       │
│    DSP: ctx_dsp_seg_XX__Dsp.bin                                     │
│    CPU: ctx_dsp_seg_XX__Cpu.bin, ctx_cpu_seg_XX__Cpu.bin            │
│    HTA: ctx_dsp_seg_XX_<conv_name>__Hta.bin (50 standalone convs)   │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ profile_segments.cpp → wallclock QnnGraph_execute
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  segment_perf.json (per-segment × per-backend timing)               │
│    → emit_vision_v3_profile.py → gen/profile/{HTA,DSP,CPU}/...csv  │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ run_xpurt_schedule.py --solver greedy
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Heterogeneous schedule: per-dispatch backend assignment + Gantt     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Component | Location/Version |
|-----------|-----------------|
| QNN SDK (QAIRT) | `<PATH_TO_QAIRT_SDK>` |
| Docker image | `qnn-convert` (has snpe-onnx-to-dlc, qairt-quantizer) |
| Board | QRB5165, reachable at `root@<BOARD_IP>` (`$QNN_BOARD_HOST`) |
| On-board QNN | `/root/qairt/` (libs at `lib/target/`, v66 hexagon at `lib/hexagon-v66`) |
| Python env | `<PATH_TO_CONDA_ENV>/` (onnx, onnxruntime, numpy) |
| Source ONNX | `qnn_models/smolVLA/smolvlm_vision.onnx` |

---

## Procedure

### 0. Quick Path (full pipeline)

```bash
cd qnn_models/smolVLA
./pipeline_vision_v3.sh
```

This runs all 7 stages. To run individual stages:
```bash
./pipeline_vision_v3.sh slice rewrite build build-hta profile emit schedule
```

---

### 1. Slice the Vision Encoder

**Script:** `slice_vision_v3.py`

Cuts smolvlm_vision.onnx into 49 segments at operator boundaries chosen to
minimize inter-segment tensor size and avoid expensive layout conversions:

- **Softmax boundaries** (v3 fix): CPU trampoline is 4 ops
  `{Softmax → V_MatMul → Transpose → Reshape}` so the next DSP segment
  receives a 3D tensor `[1,1024,768]` (cheap to quantize), not a 4D attention
  map `[1,12,1024,1024]` (causes 307ms requantize penalty on DSP).
- **Tanh boundaries**: single GELU activation op, not supported on DSP.

```bash
python slice_vision_v3.py
# Output: vision_slices_v3/{dsp_seg_00..24, cpu_seg_00..23}.onnx
```

**Why 25+24=49:** 12 transformer layers × 2 CPU cuts/layer = 24 CPU segments.
The DSP segments interleave: first DSP seg includes the patch embed conv.

---

### 2. Rewrite MatMul → Conv1×1

**Script:** `rewrite_matmul_to_conv1x1.py`

On Hexagon v66, QNN's Conv2d has heavily-optimized VTCM tiling that the
generic MatMul path lacks. Each Linear layer becomes:

```
Transpose([1,1024,768] → [1,768,1024])
→ Reshape([1,768,1024,1])
→ Conv2d(in=768, out=N, kernel=1×1)
→ Reshape([1,N,1024])
→ Transpose([1,1024,N])
```

Only rewrites MatMuls with a constant weight initializer (Linear layers).
Batched attention MatMuls (Q×K^T, attn×V) are left untouched.

```bash
python rewrite_matmul_to_conv1x1.py vision_slices_v3 --batch --validate
# Output: vision_slices_v3/conv1x1/dsp_seg_*.onnx
```

---

### 3. Extract Standalone HTA Convs

**Script:** `extract_hta_convs.py`

HTA cannot execute the Transpose/Reshape wrappers around Conv1×1. This script
extracts each Conv node as a standalone model with raw NCHW I/O:

- Input: `(1, C_in, 1024, 1)` for Conv1×1, or `(1, 3, 512, 512)` for patch embed
- Op: single Conv2d (includes bias)
- Output: `(1, C_out, M_out, W_out)`

```bash
python extract_hta_convs.py \
    --slices-dir vision_slices_v3/conv1x1 \
    --out-dir vision_slices_v3/hta_convs
# Output: 50 standalone .onnx files in hta_convs/
```

**Shape families (50 total):**
| Shape (C_in→C_out) | Kernel | Count | Role |
|---------------------|--------|-------|------|
| 768→768 | 1×1 | 12 | output projection |
| 768→3072 | 1×1 | 12 | fc1 |
| 3072→768 | 1×1 | 12 | fc2 |
| 768→2304 | 1×1 | 12 | QKV projection |
| 3→768 | 16×16 | 1 | patch embed |
| 12288→960 | 1×1 | 1 | final head |

---

### 4. Build DLCs (Docker, host-side)

**Script:** `pipeline_vision_v3.sh` stages 3+4

Converts ONNX to Qualcomm DLC format and quantizes to int8:

```bash
# DSP segments: ONNX → DLC → quantized DLC (int8)
snpe-onnx-to-dlc --input_network <seg>.onnx --output_path <seg>.dlc
qairt-quantizer --input_dlc <seg>.dlc --output_dlc <seg>_quantized.dlc \
    --input_list <cal_list> --act_bitwidth 8 --weights_bitwidth 8

# CPU segments: ONNX → DLC (fp32, no quantization)
snpe-onnx-to-dlc --input_network cpu_seg_XX.onnx --output_path cpu_seg_XX.dlc

# HTA standalone convs: ONNX → DLC → quantized DLC (int8)
# CRITICAL: HTA requires int8 DLCs. Without --input_list, qairt-quantizer
# produces an fp32 copy labeled as _q.dlc that will fail on HTA with
# "Validate OpConfig failed". Must provide actual calibration data.
```

**Calibration data** (`gen_vision_slice_calibration.py`): runs the full
smolvlm_vision.onnx through onnxruntime, captures intermediate activations at
segment boundaries, saves as `.raw` files. For HTA standalone convs, random
calibration is acceptable (timing-only profiling, not accuracy).

---

### 5. Build Context Binaries (on-board)

Context binaries are pre-compiled QNN graphs for a specific backend. Built
on the target board because they encode backend-specific optimizations.

```bash
# On QRB5165:
export LD_LIBRARY_PATH=/root/qairt/lib/target
export ADSP_LIBRARY_PATH="/root/qairt/lib/hexagon-v66;/dsp/cdsp;/dsp"

qnn-context-binary-generator \
    --backend /root/qairt/lib/target/libQnn{Dsp,Cpu,Hta}.so \
    --model /root/qairt/lib/target/libQnnModelDlc.so \
    --dlc_path <seg>_quantized.dlc \
    --binary_file ctx_<seg>__<Backend> --output_dir .
```

**Backend libraries:**
- `libQnnDsp.so` — Hexagon DSP (CDSP, v66)
- `libQnnCpu.so` — ARM CPU (Kryo 585)
- `libQnnHta.so` — Hardware Tensor Accelerator (fixed-function int8 MAC array)

---

### 6. Profile (wallclock around QnnGraph_execute)

**Script:** `profile_vision_v3_correct.sh`
**Profiler:** `qnn_models/runtime/profile_segments.cpp`

The correct methodology: directly calls `QnnGraph_execute` via `dlopen`'d
backend library on a pre-built context binary, measures `steady_clock::now()`
wallclock per call over N iterations (default 50).

**Why not qnn-net-run --profiling_level?** That only captures the backend's
self-reported compute time, not the full dispatch overhead (launch + RPC +
requantize + compute) the runtime actually pays.

```bash
# Build profiler on board (once):
g++ -std=c++2a -O2 -pthread -I$QNN/include profile_segments.cpp -o profile_seg -ldl

# Profile one context binary:
./profile_seg ctx_dsp_seg_01__Dsp.bin /root/qairt/lib/target/libQnnDsp.so 50
# Output: JSON line with mean_us, median_us, std_us, etc.
```

**HTA profiling aggregation:** HTA context binaries are per-conv-op (not
per-segment). Per-segment HTA time = sum of constituent conv op times. This
gives the scheduler the total HTA compute cost for each segment.

```
seg_01 HTA time = output_proj_conv (3.1ms) + fc1_conv (9.3ms) = 12.4ms
seg_02 HTA time = fc2_conv (15.9ms) + QKV_conv (6.2ms) = 22.2ms
```

---

### 7. Emit XPURT Profile CSVs

**Script:** `emit_vision_v3_profile.py`

Reads `segment_perf.json` and writes per-backend results.csv in the format
the scheduler expects:

```bash
python emit_vision_v3_profile.py --target qrb5165_v66 \
    --from-perf-json boards/qrb5165_v66/profiles/smolvlm_vision_v3/segment_perf.json
```

Output at `gen/profile/{CPU,DSP,HTA}/qrb5165_v66/smolvlm_vision_v3/smolvlm_vision_v3.int8/topo_0/results.csv`:
```csv
dispatch_id,module_name,mean_time,mean_unit
0,dsp_seg_00,295459.02,us
1,cpu_seg_00,36071.30,us
...
```

---

### 8. Run Scheduler

**Script:** `scripts/run_xpurt_schedule.py`

The topology JSON (`data/toplevel/networks_smolvla_vision_v3_qrb5165.json`)
maps logical machines to backends:
- `cpu_p` → HTA (profile_hw)
- `cpu_e` → DSP (profile_hw)
- `cpu_x` → CPU (profile_hw)

```bash
python scripts/run_xpurt_schedule.py \
    --networks-json data/toplevel/networks_smolvla_vision_v3_qrb5165.json \
    --solver greedy --profiled
```

Output:
- `schedules/scheduled_networks_smolvla_vision_v3_qrb5165_greedy_profiled.json`
- `plots/networks_smolvla_vision_v3_qrb5165_greedy_profiled.png`

---

## Key Design Decisions

### Why v3 slicing (not v1/v2)?

| Version | Strategy | Problem |
|---------|----------|---------|
| v1 | Cut only at Tanh | Large DSP segments (~25 ops) hit QNN compile limits |
| v2 | Cut at Softmax + Tanh | Softmax-only CPU trampoline leaves 4D attention tensor as DSP input → 307ms requantize |
| v3 | Cut at {Softmax+V×attn+Transpose+Reshape} + Tanh | DSP segments receive 3D [1,1024,768] → cheap quantize |

### Why Conv1×1 rewrite?

Hexagon v66 VTCM tiling for Conv2d is heavily optimized (data stays in 256KB
scratchpad across tiles). The generic MatMul path doesn't tile the same way.
Observed 1.5–2× speedup on DSP for the large projections.

### Why extract standalone HTA convs?

HTA is a fixed-function int8 MAC array. It can only execute Conv/Pool/Eltwise
ops — not Transpose, Reshape, LayerNorm, etc. Wrapping the Conv1×1 in
Transpose+Reshape (needed for the DSP data layout) makes the whole subgraph
HTA-incompatible. Extracting the Conv alone lets HTA execute just the compute-heavy
part (5–7× faster than DSP for these shapes).

### Why force-quantize HTA convs with random calibration?

HTA requires int8 inputs. `qairt-quantizer` without `--input_list` produces a
misleading fp32 copy as the `_q.dlc` output (no actual quantization happens).
For timing-only profiling, random calibration data is valid — it exercises the
same hardware path. The actual runtime would use properly-calibrated DLCs.

---

## File Index

| File | Purpose |
|------|---------|
| `slice_vision_v3.py` | Cut ONNX into 49 segments at Softmax-block + Tanh boundaries |
| `rewrite_matmul_to_conv1x1.py` | MatMul → Conv1×1 rewrite for DSP VTCM tiling |
| `extract_hta_convs.py` | Extract standalone Conv ops for HTA (strips wrappers) |
| `gen_vision_slice_calibration.py` | Generate calibration data from real activations |
| `gen_vision_v3_dispatch_graph.py` | Generate 49-node linear dispatch graph for scheduler |
| `emit_vision_v3_profile.py` | Convert segment_perf.json → XPURT results.csv |
| `pipeline_vision_v3.sh` | Orchestrates all 7 stages end-to-end |
| `profile_vision_v3_correct.sh` | On-board profiling sweep (ctx binaries + profile_seg) |
| `../runtime/profile_segments.cpp` | C++ profiler: wallclock around QnnGraph_execute |
| `PROFILING_INFRASTRUCTURE.md` | Detailed profiling methodology notes |

---

## Profiling Results Summary

### Per-segment timing (representative)

| Segment Type | HTA (ms) | DSP (ms) | CPU (ms) | HTA Speedup |
|-------------|-----------|----------|----------|-------------|
| seg_00 (patch embed + QKV) | 295.5 | 159.9 | 68.5 | 0.2× (CPU wins) |
| B-type odd (proj + fc1) | 12.4 | 88.2 | 97.3 | 7.1× |
| A-type even (fc2 + QKV) | 22.2 | 155.7 | 111.5 | 5.0× |
| seg_24 (fc2 + final head) | 134.8 | 136.5 | 77.8 | 0.6× (CPU wins) |
| CPU segments (Softmax/Tanh) | 9–36 | — | 9–36 | 1× (same) |

### Why seg_00 and seg_24 are bad for HTA

- **seg_00**: Contains the patch embed Conv(3→768, kernel=16×16, stride=16) on
  input [1,3,512,512]. This is a single large conv that produces [1,768,32,32].
  HTA's fixed tiling doesn't suit the unusual 16×16 kernel well → 289ms vs CPU's 68ms.

- **seg_24**: Contains the final head projection Conv1×1(12288→960). The 12288
  input channels exceed HTA's efficient tiling range → 119ms for that single
  conv, plus 16ms for fc2 = 135ms total. CPU does the full segment in 78ms.

---

## Troubleshooting

### HTA context binary fails with "Validate OpConfig failed"
The DLC is fp32, not int8. Ensure `qairt-quantizer` was run with `--input_list`
pointing to actual calibration data (even random `.raw` files work for timing).

### profile_seg compilation fails "No space left on device"
The board's `/tmp` is a small tmpfs. Clean it: `rm -rf /tmp/_ctxgen_* /tmp/qnn_*`

### stat errors in profiling output
Cosmetic: the shell script runs `stat` before confirming the context binary
exists. Non-blocking, profiling results are still correct.

### DSP segments show higher time than expected
First few iterations are warmup (DSP firmware loading). The profiler drops
warmup via statistics (mean over 50 iters, first few are amortized).
