# YOLOv8n — architectural divergence from vanilla ultralytics

Documents what `agents/models/yolov8_nano.py` changes relative to the
upstream ultralytics `yolov8n` reference. The divergences are all in
service of getting a pure-PyTorch model that **traces cleanly through
`torch.fx.symbolic_trace`** — which the agents-flow extract pass
requires — without changing the trained-weight numerics.

The COCO-pretrained weights from `yolov8n.pt` stream into this model
unchanged; backbone / neck channel counts and per-layer Conv shapes
are byte-for-byte identical. The PyTorch forward pass of this model
matches ultralytics' to within fp32 rounding (validated against the
agents-flow's PyTorch-vs-spike golden compare).

## Why we needed a wrapper

Two specific spots in the upstream ultralytics codebase break
`torch.fx.symbolic_trace`:

1. **`ultralytics.nn.modules.block.C2f.forward`**:
   ```python
   def forward(self, x):
       y = list(self.cv1(x).chunk(2, 1))
       y.extend(m(y[-1]) for m in self.m)
       return self.cv2(torch.cat(y, 1))
   ```
   The generator expression `m(y[-1]) for m in self.m` defeats FX —
   it's a Python list comprehension with side-effectful module calls.
   FX can't symbolically unroll `len(self.m)` because the bottleneck
   count `n` is read from the YAML config, not derived from the input
   tensor.

2. **`ultralytics.nn.tasks.DetectionModel._predict_once`**: a
   Python `for m in self.model:` loop with conditional skip-connection
   routing (some layers consume `y[m.f]` lookups into a list of
   intermediate tensors keyed by layer index). FX can't trace
   data-dependent `if`-branches.

We could patch `ultralytics` in-place, but that's brittle and
version-locks us. Reimplementing the 23 layers as a static module is
~250 LOC and self-contained.

## Divergence list

### 1. C2f → static `_C2fN1` / `_C2fN2`

Upstream `C2f(c1, c2, n, shortcut)` parameterizes the bottleneck count
`n` at construction time. We split into two concrete classes:

```python
class _C2fN1(nn.Module):   # one bottleneck — used everywhere except
                           # backbone layers 4 and 6
class _C2fN2(nn.Module):   # two bottlenecks — backbone layers 4 and 6
                           # of YOLOv8n only
```

`yolov8n.yaml`'s depth_multiple=0.33 cuts the upstream
`[1,3,6,6,3]` per-stage counts → 1,1,2,2,1, so only n=1 and n=2 occur.
Static unroll is safe (no other variants used) and FX-traceable.

### 2. Flat top-level `forward` instead of YAML-driven dispatch

Upstream `DetectionModel.forward` iterates `self.model` (an
`nn.ModuleList` populated from the YAML), reading routing metadata
from each layer's `f` (from-index) attribute. Tensors get parked in
a `y` list keyed by layer index so later layers can pull from there:

```python
# upstream — simplified
y = []
for m in self.model:
    if m.f != -1:
        x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
    x = m(x)
    y.append(x if m.i in self.save else None)
```

We replace this with a single hand-written `forward` that names every
intermediate tensor and routes them explicitly:

```python
def forward(self, x):
    x  = self.l0(x);  x  = self.l1(x);  x  = self.l2(x);  x  = self.l3(x)
    x4 = self.l4(x);  x  = self.l5(x4)                      # save P3 source
    x6 = self.l6(x);  x  = self.l7(x6); x  = self.l8(x)     # save P4 source
    x9 = self.l9(x)                                          # save P5 source
    u10 = self.l10(x9); cat11 = torch.cat([u10, x6], 1)
    x12 = self.l12(cat11); u13 = self.l13(x12); cat14 = torch.cat([u13, x4], 1)
    ...
    return self.detect(x15, x18, x21)
```

Every `torch.cat` is explicit (matches upstream's `Concat` modules
which have no parameters). Every skip connection is a named local
variable.

### 3. Detection head outputs raw, no DFL / NMS

Upstream `Detect.forward` does:
```
per scale: cv2 (box) + cv3 (cls) → cat
in training: return tuple of raw outputs
in eval:  decode DFL → multiply by stride grid → cat into single
          (B, 4+nc, anchors) tensor → optional end2end NMS / TopK
```

The DFL decode (`Conv2d(reg_max, 1, 1)` with hardcoded
`weight = arange(reg_max)` and Softmax) plus the NMS / TopK path are
post-processing dressed up as PyTorch modules. They wouldn't trace
cleanly (NMS has data-dependent control flow), and even if they did
they'd be a waste of on-device compute (NMS belongs in the host post-
process where it can be hand-tuned for the deployment scenario).

Our `_DetectHead.forward` returns `(out0, out1, out2)` — the three
raw P3/P4/P5 feature maps with channel layout
`[reg_max*4 (box) | nc (cls)]`. The harness's PyTorch golden runs
the same model so the compare is apples-to-apples; the actual NMS /
DFL decode runs in Python on the host receiving the network output
(out of scope for the on-spike binary).

This matches the deployment pattern most teams use — on-device
inference produces raw detection feature maps; NMS is a CPU-side
or even tiny-NN post-process.

### 4. Input resolution defaults to 160, not 640

Upstream YOLOv8n is trained and inferenced at 640×640. Our default
is `AGENTS_YOLOV8N_INPUT=160` because:

- 640²×3 = 1.2 MB per input frame; the agents harness allocates that
  on stack (or in the buffer arena) per dispatch — workable but
  expensive on Zephyr stack.
- The intermediate-tensor pyramid is ~7× larger at 640: P3 is
  80×80×64=512 KB instead of 20×20×64=25 KB, P4 1280 KB vs 50 KB,
  etc. Total buffer footprint exceeds the 4 MB stack default.
- At 640 spike inference takes >30 minutes per frame. 160 fits in
  ~2 minutes — useful for iteration.

The model is **resolution-agnostic** by design (`AGENTS_YOLOV8N_INPUT`
must be a multiple of 32 since the stride-32 head requires it).
Real deployments would set this to 320 or 640 depending on the
target latency budget; the trained weights work at any resolution.

### 5. BatchNorm eps + momentum match ultralytics, not PyTorch defaults

```python
self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
```

Ultralytics uses these non-default values throughout. The standard
PyTorch defaults (`eps=1e-5, momentum=0.1`) would silently corrupt
the pretrained weights' numerics by 1–2 orders of magnitude per BN
layer. Worth flagging in the docstring because it's the kind of
silent bug that's painful to chase later.

### 6. SiLU explicit, no fused Conv-BN-Act

Upstream `Conv` has the same shape, but ultralytics also ships a
fused-inference path that folds BN into the conv weights and bias.
We keep them separate `nn.Conv2d` + `nn.BatchNorm2d` + `nn.SiLU`
because:

- The agents extract pass folds BN into the preceding conv at
  IR-emission time (see `_fold_bn_into_conv` in `extract_graph.py`),
  so by the time the C codegen sees the model, BN is gone anyway.
- Keeping BN as its own module makes the FX graph more uniform
  (one `Conv2d` op per conv layer in the IR; the BN-fold is a
  pass, not a structural choice).

### 7. Detect head submodules flattened (no `nn.ModuleList`)

Upstream `Detect` stores `cv2: nn.ModuleList[nn.Sequential]` and
`cv3: nn.ModuleList[nn.Sequential]`, indexed by scale at forward
time. FX-tracing a ModuleList indexed by a runtime int variable
is fragile.

We flatten to `cv2_0_0`, `cv2_0_1`, `cv2_0_2`, `cv2_1_0`, ... —
one named submodule per `(scale, depth)` slot. The weight-loading
translator (`_ultra_to_local_key`) rewrites
`detect.cv2.0.0.conv.weight` → `detect.cv2_0_0.conv.weight` etc.,
so pretrained weights still stream in cleanly.

## What stays IDENTICAL

These are deliberate equalities. If you ever notice a delta in any
of these, the trained weights stop loading correctly:

| property | value |
|---|---|
| total convs in backbone+neck+head | 63 conv2d_s8 dispatches in IR |
| channel widths per stage | [16, 32, 64, 128, 256] (depth_multiple=0.33 of upstream nano) |
| C2f bottleneck residual | shortcut=True in backbone, shortcut=False in neck (matches upstream) |
| SPPF kernel | 5×5, 3 chained maxpools |
| Detect channel layout | [reg_max*4 | nc] = [64 | 80] = 144 per scale |
| reg_max, nc | 16, 80 (COCO defaults) |
| BatchNorm eps, momentum | 1e-3, 0.03 (ultralytics) |
| activation | SiLU |
| stride pattern (backbone) | 2,2,1,2,1,2,1,2 |

## Op inventory (int8 PTQ pass)

The full agents-flow IR at default `AGENTS_YOLOV8N_INPUT=160`:

```
conv2d_s8                63
batchnorm2d_s8           57    ← BN-fold pass would erase these but
                                 we keep them as standalone ops in
                                 the int8 path for now
silu_s8                  57
chunk2_c1                 8    ← C2f's "chunk(2, dim=1)" — split
                                 into two halves of channels
cat2_c1_s8                7    ← C2f's "cat([y0, y1, y2], 1)" for n=1
cat3_c1_s8                6    ← cat for n=2 / SPPF (4 inputs counted
                                 elsewhere)
cat4_c1_s8                3    ← SPPF's 4-input cat
add_s8                    6    ← bottleneck residuals
maxpool2d_s8              3    ← SPPF
upsample_nearest_s8       2    ← neck PAN-FPN upsamples
                        ───
total dispatches        212
```

Compare to dronet's 30 dispatches and ViNT's 605. YOLOv8n sits in
the middle of the model-complexity spectrum supported by the
agents flow.

## Weight loading

`_ultra_to_local_key` maps every ultralytics `yolov8n.pt` state_dict
key to our flat-name layout:

```
upstream key                            our key
─────────────────────────               ────────────────────────────
model.0.conv.weight                  →  l0.conv.weight
model.4.m.0.cv2.bn.weight            →  l4.m0.cv2.bn.weight
model.9.cv1.conv.weight              →  l9.cv1.conv.weight
model.22.cv2.0.0.conv.weight         →  detect.cv2_0_0.conv.weight
model.22.cv2.0.2.weight              →  detect.cv2_0_2.weight
model.22.dfl.conv.weight             →  (skipped — DFL not in our model)
```

Layer 10 (Upsample) and layers 11/14/17/20 (Concat) are
parameter-less so they don't appear in the state_dict — only our
forward references them.

Verified end-to-end: `get_model()` with `AGENTS_YOLOV8N_PRETRAINED=1`
copies 235 of ultralytics' tensors (the full backbone + neck + Conv
layers of the head; the DFL and end2end branches are intentionally
skipped). PyTorch forward of the loaded model matches ultralytics'
`yolo.model.eval()(x)` on the box+cls outputs.

## Implications for deployment

Anyone integrating this on-device needs to know:

1. **The harness output is raw P3/P4/P5 feature maps**, NOT decoded
   detections. The host (or a follow-up small NN) must run:
   - DFL: split box channels into 4 × reg_max=16, softmax along
     reg_max, dot with [0, 1, ..., 15] to get the bounding-box
     deltas in cell units.
   - Stride decode: multiply box deltas by per-scale stride (8, 16,
     32) and add the anchor-cell grid offset.
   - Class scoring: sigmoid the cls channels.
   - NMS: with whatever per-class threshold the deployment scenario
     wants.

2. **Calibration data matters.** The default int8 PTQ uses ONE
   random-init synthetic frame (see `get_sample_input`). Real
   deployment needs a representative calibration set (the agents
   flow's `--num-calibration` knob takes care of this once the
   harness has a dataset loader for it). Without calibration, the
   int8 accuracy degrades visibly; not catastrophic for detection
   feature maps but worth running through `--per-channel
   --num-calibration 32` for production.

3. **Input resolution drives compute linearly.** Bumping to 320
   from 160 costs ~4× cycles on spike. The trained weights are
   accuracy-good at any resolution that's a multiple of 32; the
   higher you go the better the small-object recall.

## References

- Upstream YAML: `ultralytics/cfg/models/v8/yolov8.yaml`
- Upstream layers: `ultralytics.nn.modules.{conv,block,head}`
- Our model: `agents/models/yolov8_nano.py`
- Example runner: `agents/examples/yolov8_nano/run.sh`
- BN-fold pass (where the 57 BatchNorm dispatches in the int8 IR get
  collapsed for the fp32 path): `agents/pipeline/extract_graph.py`
- IR + weights: `agents/examples/yolov8_nano/<quant>/generated/`
