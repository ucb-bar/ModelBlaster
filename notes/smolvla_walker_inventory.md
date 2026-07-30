# SmolVLA walker — what blocks int8 lowering today

Captured 2026-05-25 against the `feat/benchmark-harness` branch on a
clean `uv sync --extra smolvla` install. Recorded so the next attempt
to unblock SmolVLA starts from data, not speculation.

## TL;DR

SmolVLA is multi-week work. The admission doc was right to call int8
"a separate research project." Two real blockers, not one:

1. **The entire forward is wrapped opaque.** `torch.export(strict=True)`
   wraps SmolVLA's forward in a single `wrap_with_set_grad_enabled`
   call because `transformers.utils.output_capturing._active_collector`
   is touched during the forward (counts as a side effect). The walker's
   existing `wrap_with_set_grad_enabled` handling targets
   EfficientNet's 2-op MemoryEfficientSwish sub-graph; it has no
   mechanism to recurse into a 4629-node sub-graph.
2. **~50 distinct aten op kinds inside the wrap are unclassified.** Even
   if the walker recursed, most of the inner ops are in the `UNKNOWN`
   classification bucket.

## How the graph actually looks

```
torch.export(strict=True) on SmolVLAOneStepNoCacheWrapper:

Outer graph: 519 nodes
  ├─ 515 placeholders  (each parameter is a separate input)
  ├─ 1   output
  └─ 2   call_function
       ├─ wrap_with_set_grad_enabled  → submod_1  (the whole model)
       └─ getitem                                  (extract output)

submod_1: 5145 nodes
  ├─ ~5145 - 4629 = 516 placeholders / get_attr / output
  └─ 4629 call_function nodes — the actual SmolVLA compute
```

The `wrap_with_set_grad_enabled` wrap is triggered by
`G['__import_transformers_dot_utils_dot_output_capturing']._active_collector`
(per dynamo's compile warning). That is a module-global the
transformers internals touch during forward.

## Inner inventory (after recursing into submod_1)

Counts of aten ops, grouped by the existing `_classify` taxonomy:

| Class | Count | Notes |
|---|---|---|
| supported / new | small | already known to walker but unreachable from outer graph |
| alias | 517 | reshape, permute, split, clone, select, ... |
| noop | 176 | getitem, copy_ |
| tail | 42 | div.Tensor, cumsum |
| **UNKNOWN** | **2063** | 50 distinct op kinds — top items below |

Top unknown ops inside submod_1 (full list runs to 50 kinds; this is the head):

| Op | Count | Disposition |
|---|---|---|
| `_assert_tensor_metadata.default` | 601 | trivial — classify as noop |
| `to.dtype` | 589 | trivial — classify as alias (dtype-cast passthrough) |
| `pow.Tensor_Scalar` | 66 | LayerNorm component |
| `mean.dim` | 66 | LayerNorm component |
| `rsqrt.default` | 66 | LayerNorm component |
| `add_.Tensor` | 64 | trivial — alias of `add.Tensor` |
| `arange.default` | 58 | constant tensor (alias-class) |
| `expand.default` | 56 | trivial — alias |
| `sub.Tensor` | 49 | LayerNorm component / arithmetic |
| `matmul.default` | 48 | already has MATMUL spec — walker hookup needed |
| `pow.Scalar` / `sin` / `cos` | 41 × 3 | RoPE — primitive or pattern-match |
| `silu.default` | 33 | already has SILU spec — walker hookup needed |
| `mul_.Tensor` | 24 | trivial — alias of mul |
| `where.ScalarOther` | 24 | conditional |
| `softmax.int` | 24 | already has SOFTMAX_F16/S8 — walker hookup needed |
| `embedding.default` | 4 | new — embedding lookup primitive |
| (~30 more low-count ops) | ~50 | mixed |

## Path to unblock

### A. The wrap problem (must fix first)

Pick one:

- **A1: Disable the transformers output_capturing side effect** before
  export so torch.export traces a flat graph. Likely a monkey-patch on
  `transformers.utils.output_capturing._active_collector` to look
  inert. Clean result; potentially version-fragile.
- **A2: Make the walker recurse into `wrap_with_set_grad_enabled`
  sub-modules generally**, not just the EfficientNet-specific swish
  pattern. More invasive but transformers-version-independent.

### B. Op classification + walker hookup (~1.5-2 weeks for fp16)

| Sub-task | Effort |
|---|---|
| Trivial reclassifications (~10 op kinds → alias/noop) | 2-3 hours |
| LayerNorm pattern matching (mean.dim + sub + pow + rsqrt + mul + add) | 1-2 days |
| RoPE primitive + ROTARY_EMB_F16 KernelSpec | 1-2 days |
| Walker hookup for existing specs (matmul, silu, softmax) inside submod_1 | 1-2 days |
| End-to-end accuracy verification vs PyTorch fp16 | 1 day |

### C. int8 attention research (additional ~3-5 weeks)

- Quantization scheme for the attention block (Q, K, V activation
  calibration; attention-mask handling under int8).
- int8 LayerNorm numerics.
- int8 RoPE numerics (or keep RoPE fp16 + cast around it).
- Curated attention kernels for accelerator targets (open-ended).

This is what `notes/smolvla_admission.md` meant by "separate research
project."

## Reproducing this diagnostic

One-time:

```bash
uv sync --extra smolvla
export PI0_ROOT=/scratch2/agustin/merlin/third_party/Understanding-PI0
export LEROBOT_ROOT=/scratch2/agustin/merlin/third_party/lerobot
export HF_HOME=/scratch2/agustin/hf_cache
```

Outer inventory (only sees the wrap):

```bash
uv run python -m modelblaster.pipeline.extract_graph_export \
    --model smolvla --quant fp16 --inventory-only \
    --out-dir /tmp/smolvla-inventory
cat /tmp/smolvla-inventory/op_inventory.txt
```

Inner inventory (recurse into submod_1) — a quick script until the
walker grows recursion:

```python
import torch
from collections import Counter
from modelblaster.models import smolvla
from modelblaster.pipeline.extract_graph_export import _classify, _op_name

m = smolvla.get_model()
inp = smolvla.get_sample_input()
ep = torch.export.export(m, inp, strict=True)
submod = ep.graph_module.submod_1

by_class = {}
for n in submod.graph.nodes:
    if n.op in ("placeholder", "get_attr", "output"):
        continue
    cls = _classify(_op_name(n))
    by_class.setdefault(cls, Counter())[_op_name(n)] += 1

for cls in ("supported", "new", "swish", "alias", "fold",
            "noop", "tail", "UNKNOWN"):
    if cls not in by_class: continue
    print(f"--- {cls} ({sum(by_class[cls].values())}) ---")
    for name, n in by_class[cls].most_common(20):
        print(f"  {n:5d}  {name}")
```
