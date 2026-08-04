# QRB5165 (Hexagon v66) Layer Mapping — CNNs and SmolVLA Components

A working report of what we observed while taking the CNN benchmark suite
(dronet / yolov8 / mobilenet) and SmolVLA's component networks (vision
encoder, decode / prefill expert, projectors, text expert) and mapping
them onto QRB5165's four backends: **CPU**, **GPU (Adreno 650)**,
**HTA**, and **DSP (Hexagon v66 cDSP)**. All numbers are from
`qnn_models/boards/qrb5165_v66/profiles/`.

---

## 1. Backend cheat-sheet

| Backend | Lib | Dtype native | Strengths | Weaknesses |
|---|---|---|---|---|
| CPU | `libQnnCpu.so` | fp32 (+ `CPU_int8` weighted) | Supports every op; no firmware caps; no allocator caps | 5–80× slower than accelerators on conv-heavy graphs |
| GPU | `libQnnGpu.so` (fp32) / fp16_half2 (`GPU_fp16`) | fp16 / fp32 | High peak GOPS for matmul-heavy nets | Big OpenCL kernel-launch + compile overhead; loses on tiny graphs; int8 path crashes on QRB5165 |
| HTA | `libQnnHta.so` | int8 only | Lowest-latency for pure conv stacks (dronet, mobilenet) | Op coverage is the smallest of all 4 — won't accept anything outside Conv/Pool/Add/ReLU/etc.; hard simultaneous-context cap; quantization mandatory |
| DSP | `libQnnDsp.so` | int8 / int16 | Best general-purpose accelerator on this chip; supports Softmax / MatMul / LayerNorm / Reshape | Per-context init is the slowest of all 4 (80–160 ms); firmware leaks per-create resources |

---

## 2. Layer taxonomy we encountered

### 2.1 CNN-style ops (dronet, yolov8, mobilenet, action/state projectors)

Every backend handles these. Differences are pure performance.

| Op family | Where it appears | Lands on |
|---|---|---|
| Conv2d 3×3 / depthwise | dronet, yolov8 backbone, mobilenet | All four; HTA wins when whole graph is int8 conv |
| Conv2d 1×1 | yolov8 neck, projector heads, **also rewritten matmuls** | HTA / DSP near-equivalent |
| Pool (Max/Avg) | yolov8, mobilenet | All four |
| ReLU / SiLU / GELU | Throughout | All four (DSP fuses with Conv) |
| BatchNorm | Folded into Conv at compile-time | n/a |
| Sigmoid (yolov8 detect head) | yolov8 | All four |

### 2.2 Transformer ops (SmolVLA vision encoder = SigLIP-base ViT, SmolVLA decode/prefill = SmolVLM-360M expert)

Coverage drops sharply once we leave the CNN comfort zone.

| Op | CPU | GPU | DSP | HTA | Notes |
|---|---|---|---|---|---|
| Linear / `MatMul(act, const_W)` | ✓ | ✓ | ✓ (slow as MatMul) | ✗ as MatMul, ✓ when rewritten to Conv1x1 | See §3.1 |
| Batched MatMul `Q·Kᵀ`, `attn·V` | ✓ | ✓ | ✓ | ✗ | DSP keeps these as MatMul; HTA can't host them at all |
| LayerNorm | ✓ | ✓ | ✓ | ✗ | DSP path is fine |
| RMSNorm (no explicit op — *decomposed* in ONNX) | ✓ | ✓ | ✗ | ✗ | Decoder hits `QNN_DSP_BACKEND_ERROR_OP_PACKAGE_NOT_FOUND` (§3.4) |
| Softmax | ✓ | ✓ | ✓ but layout-expensive (§3.2) | ✗ | We cut Softmax out of DSP segments |
| GELU / Tanh | ✓ | ✓ | ✓ | ✗ | We cut Tanh out as a CPU trampoline |
| Reshape / Transpose / Cast | Free | Free | Free | "free" only if shapes match HTA's NHWC4 expectation; otherwise refused |  |
| ScatterND (KV cache writes) | ✓ | ✓ | ✗ | ✗ | Decode-only; forces the whole rotary+KV-cache block to CPU |
| Where, Sin/Cos (rotary) | ✓ | ✓ | ✗ | ✗ | Decode-only |
| Embedding / Gather | ✓ | ✓ | ✓ | ✗ |  |

### 2.3 Op-coverage scoreboard (smolvlm_vision encoder, 1 frame)

| Backend | Native execute | Mean latency |
|---|---|---:|
| CPU (fp32) | All 196 ops | 3 567 ms |
| CPU_int8 | All 196 ops | 3 323 ms |
| GPU_fp16 | All 196 ops | **12 869 ms** ← OpenCL kernel-launch overhead dominates |
| DSP whole-graph | **fails** (Softmax + Reshape layout); we slice it ourselves | n/a |
| HTA whole-graph | **fails** (every non-conv op rejected) | n/a |

This is why we never run SmolVLA's vision encoder as a single QNN graph.
The actual deployment path slices it.

---

## 3. Transforms required to fit each backend

### 3.1 MatMul → Conv1×1 rewrite

**Source**: `qnn_models/smolVLA/rewrite_matmul_to_conv1x1.py`.

Every weight-times-activation MatMul (Linear in PyTorch terms) is
mathematically:

```
  MatMul(x[B,M,K], W[K,N])  ≡  Conv2d(x_4d[B,K,M,1], W_4d[N,K,1,1])
```

We rewrite, in ONNX, in two places:

* **SigLIP ViT layers** — fc1, fc2, output proj, and QKV (the only
  ones where one operand is a constant initializer). The batched
  attention matmuls `Q·Kᵀ` and `attn·V` are left as MatMul because
  both operands are activations.
* **Decode/prefill expert** — output_proj, fc_gate, fc_up, fc_down
  per layer.

Why: HTA refuses MatMul outright. DSP accepts MatMul but the Conv2d
path has heavily-optimized VTCM tiling that the MatMul path lacks.

ViT Linear shapes that get rewritten:

| Shape (C_in→C_out) | Kernel | Count | Role |
|---|---|---:|---|
| 768→768 | 1×1 | 12 | output projection |
| 768→3072 | 1×1 | 12 | fc1 |
| 3072→768 | 1×1 | 12 | fc2 |
| 768→2304 | 1×1 | 12 | QKV |
| 3→768 | 16×16 | 1 | patch embed |
| 12288→960 | 1×1 | 1 | head |

Plus a Reshape that turns `[1,1024,768]` into `[1,768,1024,1]` immediately
before the conv, and a Transpose that reverses it after — these become
CPU "trampolines" because HTA can't handle a Reshape that doesn't preserve
its NHWC4 layout.

### 3.2 Slicing the vision encoder ("v3 slicing")

**Source**: `qnn_models/smolVLA/slice_vision_v3.py`.

Per ViT layer, we cut at two boundaries:

```
   DSP_A: [LayerNorm → QKV split → reshape heads → Q×Kᵀ + scale]   (pre-Softmax)
   CPU:   [Softmax → V matmul → Transpose → Reshape]               (attention core)
   DSP_B: [output proj → Add residual → LayerNorm → fc1 → GELU prep]
   CPU:   [Tanh]                                                    (GELU activation)
```

Why we include `V·attn → Transpose → Reshape` in the CPU trampoline (and
not just Softmax): cutting only at Softmax leaves a 4D `[1,12,1024,1024]`
attention map as the DSP segment's input, which requires a 307 ms
requantize+layout pass on entry. Cutting four ops in keeps the DSP
boundary at a 3D `[1,1024,768]` tensor, which quantizes cheaply.

Result: **25 DSP segments + 24 CPU segments + 24 Tanh segments = 49 slice
graphs** per vision-encoder forward.

Slicing history (where prior tries failed):

| Version | Strategy | Problem |
|---|---|---|
| v1 | Cut only at Tanh | DSP segments hit ~25 ops, exceeded QNN compile limits |
| v2 | Cut at Softmax + Tanh | 307 ms layout penalty per DSP segment entry |
| v3 | Cut at {Softmax+V·attn+Transpose+Reshape} + Tanh | DSP boundary is 3D, cheap quantize |

### 3.3 HTA conv extraction (the inner inner-loop)

**Source**: `qnn_models/smolVLA/extract_hta_convs.py`.

Even after MatMul→Conv1×1 rewrite, the DSP segments still contain the
Transpose/Reshape "tramp" ops that HTA can't run. So for each DSP
segment we extract the *pure Conv ops only* as standalone HTA-targeted
sub-graphs. The Transpose/Reshape wrappers become CPU trampoline
graphs (`<seg>_tramp_p1`, `<seg>_tramp_p2`).

Three-tier dispatch per ViT layer:

```
  CPU-trampoline p1   →   HTA Conv1x1 (or fallback DSP/CPU)   →   CPU-trampoline p2
```

### 3.4 Decode-encoder RMSNorm — unresolved blocker

**Source**: `qnn_models/smolVLA/fuse_rmsnorm.py` (attempt; insufficient).

The SmolVLM decoder uses **RMSNorm**, which in ONNX is decomposed into
`Pow → ReduceMean → Sqrt → Reciprocal → Mul`. DSP rejects the whole
segment with `QNN_DSP_BACKEND_ERROR_OP_PACKAGE_NOT_FOUND`.

We tried `Reciprocal → Div(1, x)` in `fuse_rmsnorm.py` — still failed.
Other ops in the decomposition are also unsupported. The proper fix is
a custom QNN op-package that fuses the pattern back into a single
`RMSNorm` op, which would require chip support we don't have. Path A
for decode/prefill is **blocked** on QRB5165 v66.

Workaround we use today: run the SmolVLA decoder on `CPU_int8` end-to-end
(133 ms vs 476 ms fp32), and let the scheduler keep DSP+HTA free for the
vision encoder's frame.

---

## 4. Speedups observed per backend

### 4.1 CNN benchmark suite (whole-graph EXECUTE, one frame)

| Model | CPU | GPU_fp32 | GPU_fp16 | DSP | HTA |
|---|---:|---:|---:|---:|---:|
| dronet | 8.0 ms | 4.1 ms | 6.5 ms | **1.6 ms** | 3.0 ms |
| mobilenet_v2 | 38.0 ms | 8.9 ms | 8.1 ms | 7.3 ms | **4.7 ms** |
| yolov8n | 330 ms | 106 ms | 106 ms | **67 ms** | (fails) |
| yolov8s | 744 ms | 231 ms | 242 ms | **105 ms** | (fails) |

Observations:
* DSP wins on the conv-heavy yolov8 sizes despite HTA being the
  theoretical conv accelerator — HTA refuses yolov8's detect-head ops
  (Sigmoid+Concat shapes).
* Dronet is small enough that HTA's per-graph fixed overhead (~1.4 ms)
  closes most of its win against DSP.
* GPU is always 1.5–3× slower than DSP on CNNs; OpenCL kernel-launch
  amortization fails on networks this small.

### 4.2 SmolVLA components

| Component | CPU | CPU_int8 | DSP | HTA | GPU_fp16 |
|---|---:|---:|---:|---:|---:|
| smolvlm_vision (full) | 3 567 ms | 3 323 ms | (slice) | (slice) | 12 869 ms |
| smolvlm_expert_decode | 476 ms | **133 ms** | (RMSNorm blocked) | (RMSNorm blocked) | 916 ms |
| smolvlm_expert_prefill | 647 ms | **441 ms** | (RMSNorm blocked) | (RMSNorm blocked) | 2 923 ms |
| smolvlm_text | 0.097 ms | 0.029 ms | 0.77 ms | – | 6.1 ms |
| action_in_projector | 2.1 ms | 9.8 ms | 6.9 ms | **2.0 ms** | 7.5 ms |
| action_out_projector | 2.4 ms | 2.3 ms | **0.82 ms** | 2.4 ms | 5.3 ms |
| state_projector | 0.14 ms | 0.12 ms | 0.68 ms | 2.2 ms | 0.63 ms |
| time_in_projector | 14.4 ms | 8.6 ms | **1.7 ms** | 2.1 ms | 13.5 ms |
| time_out_projector | 7.2 ms | 4.1 ms | **1.1 ms** | 2.2 ms | 9.6 ms |

Pattern: tiny graphs (state_projector, smolvlm_text) get crushed by
accelerator fixed-cost overheads. Mid-sized graphs (time / action
projectors) win on DSP. HTA is competitive on action_in_projector but
its per-graph overhead floor (~2 ms) makes it a wash for everything
smaller.

### 4.3 Sliced vision encoder (v3) — sub-segment

After v3 slicing, sum-of-mean over the **25 DSP segments**:

| Backend dispatched on | Sum |
|---|---:|
| Pure CPU | 2 627 ms |
| Pure DSP | 3 065 ms (init dominates — see §5.2) |
| Pure HTA (sum of inner conv1x1s only) | **823 ms** |

The headline number: when we route the inner 1×1 convs of each DSP
segment to HTA (the rest stays on CPU as trampolines), conv compute
drops 3.2× vs CPU on the same segments. End-to-end vision-encoder
makespan with the v3 schedule:

| Schedule | Makespan | Speedup vs serial-CPU |
|---|---:|---:|
| Serial CPU (49 segments on CPU) | 3 172 ms | 1.0× |
| Serial DSP (49 segments on DSP) | 3 610 ms | 0.88× (init+layout penalty) |
| Bundle-aware (DSP-tramp budget=9, eager) | **2 561 ms** | **1.24×** |
| HTA+CPU schedule (no DSP) | 1 084 ms | 2.9× — but ignores firmware caps (§5.1) |

The 1.24× number is the realistic-budgeted result. The 2.9× number is
an aspirational ceiling that the firmware cap prevents us from hitting
on real hardware.

### 4.4 Trampolines — DSP vs CPU on the small wrapper graphs

74 trampoline graphs (3 per DSP segment, minus a few edges) running on
CPU vs DSP:

| Backend | Sum mean | Speedup |
|---|---:|---:|
| Tramp on CPU | 2 376 ms | 1.0× |
| Tramp on DSP | 1 270 ms | **1.87×** |

So even the Reshape/Transpose-heavy trampolines benefit from DSP when
the firmware slot is available, which is why the scheduler decides
which trampolines get a DSP slot (the `dsp-tramp-budget` knob).

---

## 5. Overheads — what actually costs us latency

### 5.1 Firmware context caps (the biggest constraint)

QRB5165 v66 firmware imposes these hard limits, all of which we
hit while building the bundle-aware vision schedule:

| Limit | Value | Failure mode |
|---|---|---|
| Max **simultaneous** DSP contexts | **~30** | `QnnDsp <E> Skel side failed when loading context binary` |
| Max **cumulative** DSP context creates (leaky!) | **~45** | Same error after evict-then-reload. The cDSP firmware *does not fully reclaim* a freed slot. |
| Max simultaneous HTA contexts | **~32** | `deserialize failed` / `Fail to load cache context error: 5005` |

Implications:
* You cannot get around the cumulative cap by `contextFree`-ing and
  reloading. After ~15 evict+reload cycles past the simul cap, the
  firmware just stops accepting new binaries until reboot.
* Vision-v3 alone wants 49 ctx binaries (25 DSP-seg × 2 trampolines + ~12
  HTA conv segs + ~12 CPU segs) — well over the cumulative cap if you
  load them naively as separate ctxs.

**Mitigation**: multi-graph context binaries.
`qnn-context-binary-generator --dlc_path a.dlc,b.dlc,...` produces a
*single* ctx binary holding multiple graphs that all share one firmware
slot. We validated 10 graphs per DSP binary and 10 per HTA binary. After
bundling the v3 schedule, the runtime needs only:

| Backend | Per-graph load count (before) | After multi-graph bundling |
|---|---:|---:|
| DSP firmware ctxs | 27 | **~3** |
| HTA firmware ctxs | 18 | **~5** |

Cloud QRB5165 differs from the physical board: it caps DSP at only **2
simultaneous** contexts but `contextFree` *does* reclaim (validated by
`/tmp/probe_dsp_ctx_cap.cpp` — 20 load+free cycles ran clean). So on
the cloud instance we additionally lean on an LRU-evict runtime path,
while the physical board is locked to the eager + bundling strategy.

### 5.2 Per-context init time

`INIT + COMPOSE + FINALIZE` time per backend, single small graph:

| Backend | Init (ms) | Notes |
|---|---:|---|
| CPU | ~20 ms | mostly Linux loader |
| GPU | ~50 ms | OpenCL kernel compile |
| HTA | ~100 ms | accelerator config |
| DSP | **80–160 ms** | FastRPC handshake + skel load |

For graphs whose EXECUTE time is in the same ballpark as init (e.g.
state_projector at 0.7 ms exec but ~100 ms DSP init), you must keep the
context loaded across many invocations to amortize. This is what drives
the `dsp-tramp-budget=9` choice: budget=20 needs 22 evict+reload cycles,
adding ~3.3 sec of inline bringup overhead and making the schedule
**slower** than budget=9 (3 791 ms wall vs 2 561 ms wall).

### 5.3 Layout / quantization at backend boundaries

When the schedule dispatches consecutive segments to different
backends, the runtime requantizes + relays the boundary tensor.

* DSP→CPU on a 3D `[1,1024,768]` tensor: a few hundred µs.
* DSP→CPU on a 4D attention map `[1,12,1024,1024]`: **307 ms** —
  this single boundary was why v2 vision slicing was unworkable.
* HTA→CPU on `[1,C,M,1]`: a few hundred µs (HTA's native layout).

Lesson: *which 3D shape* you cut at matters more than where in the
op-list you cut.

### 5.4 HTIF-equivalent / RPC channel throughput

Each DSP `EXECUTE` does ~1 round-trip FastRPC. On the SmolVLA vision
schedule (97 dispatches total), inter-segment FastRPC latency
contributes ~2 ms aggregate. Not the bottleneck on this chip — but it
would be on FireSim-style HTIF-bottlenecked platforms.

### 5.5 Cloud SSH-tunnel timing penalty (workflow, not chip)

While iterating on the cloud QRB5165 we found per-`ssh`-roundtrip
overhead drove the bundle-build script wall time. Bundling all 6 DSP
chunks via per-chunk SSH took ~minutes and occasionally hit
"Connection reset by peer" mid-stream. We mitigated by writing a
single board-side build script (`/tmp/build_all_mg_on_board.sh`) and
invoking it once via SSH.

---

## 6. What we'd want next (chip-side wishlist)

These would directly unblock things we cannot do today on QRB5165 v66:

1. **Custom op-package for RMSNorm on DSP.** Currently blocks the
   entire SmolVLA decoder/prefill from accelerator dispatch.
2. **Increased simultaneous-context cap, or `contextFree` that actually
   reclaims.** Would remove the "must bundle + must evict carefully"
   tightrope.
3. **HTA op coverage beyond Conv/Pool/Add.** Specifically: MatMul
   (no need for Conv1×1 rewrite), Reshape (no need for trampolines).
4. **DSP MatMul backend with VTCM tiling at parity with Conv2d.**
   Would remove the rewrite/extract pipeline entirely for the linear
   layers; the batched attention matmuls would still need a separate
   path because both operands are activations.

---

## Appendix A — file references

| Concern | File |
|---|---|
| Backend EXECUTE timings (raw) | `qnn_models/boards/qrb5165_v66/profiles/*.csv` |
| Vision v3 per-segment perf | `qnn_models/boards/qrb5165_v66/profiles/smolvlm_vision_v3/segment_perf.json` |
| Trampoline perf (DSP vs CPU) | `qnn_models/boards/qrb5165_v66/profiles/smolvlm_vision_v3/trampolines_*_perf.json` |
| MatMul→Conv1×1 rewrite | `qnn_models/smolVLA/rewrite_matmul_to_conv1x1.py` |
| HTA conv extraction | `qnn_models/smolVLA/extract_hta_convs.py` |
| Vision-encoder slicer (v3) | `qnn_models/smolVLA/slice_vision_v3.py` |
| Decode slicer (v1, blocked) | `qnn_models/smolVLA/slice_decode_v1.py` |
| RMSNorm-fusion attempt | `qnn_models/smolVLA/fuse_rmsnorm.py` |
| Multi-graph bundle build | `qnn_models/smolVLA/build_multi_graph_ctx.py` |
| Bundle-aware runtime | `qnn_models/runtime/generate_runtime.py` |
| Run notes / canonical schedule | `qnn_models/smolVLA/BUNDLE_RUNTIME_NOTES.md` |
| Multi-graph refactor plan | `qnn_models/smolVLA/MULTIGRAPH_REFACTOR_PLAN.md` |
