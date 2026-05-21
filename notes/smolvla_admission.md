# SmolVLA — admission status

Status: **loads, exports, runs**. Full int8 codegen is downstream work.

Adds `models/smolvla.py`, a thin wrapper around the canonical
``understanding_pi0.smolvla_mx`` loader and one-step no-cache module.
Loads ``lerobot/smolvla_base`` (450M params, vision encoder + cross-
attention + action head) and exposes the standard model-module API
(``get_model`` / ``get_sample_input`` / ``get_calibration_spec``).

## What works today

* ``get_model()`` builds the policy on CPU, wraps it in
  ``SmolVLAOneStepNoCacheWrapper``, and returns it in eval mode.
* ``get_sample_input()`` returns a flat 11-tensor tuple
  (``2 * num_cams + 5`` for ``num_cams=3``): three camera frames
  ``(1, 3, 512, 512) bfloat16``, three image masks ``(1,) bool``,
  ``lang_tokens (1, 48) int64``, ``lang_masks (1, 48) bool``,
  ``state (1, 32) float32``, ``noisy_actions (1, 50, 32) float32``,
  ``timestep (1,) float32``.
* End-to-end forward returns ``(1, 50, 32) float32``
  (``chunk_size=50``, ``max_action_dim=32``).
* ``torch.export.export(model, sample)`` traces in strict mode without
  raising. The non-inventory-only path through
  ``pipeline/extract_graph_export.py --model smolvla`` is now valid;
  the inventory matches what ``--model vint`` looked like at admission
  time (vision-encoder ops dominate, attention via
  ``scaled_dot_product_attention``, lots of ``layer_norm`` /
  ``linear``).

## Dependencies

The lerobot policy lives in a sibling clone (default at
``/scratch2/agustin/merlin/third_party/lerobot``); ``Understanding-PI0``
lives at ``/scratch2/agustin/merlin/third_party/Understanding-PI0``.
Both are wired in via ``sys.path`` injection at module import time
rather than as pip dependencies — lerobot's full dep set
(``transformers``, ``diffusers``, ``av``, ``opencv``, ``wandb``,
``rerun-sdk``, ``draccus``, ``num2words``, ``torchcodec``, ...) is too
heavy to drop on every modelblaster venv. The ``[smolvla]`` extra in
``pyproject.toml`` installs them when needed:

```bash
uv sync --extra smolvla
```

This raises the project's Python floor to 3.12 (lerobot requires it).

A leftover note for new contributors: the project ships a local
``mb_datasets/`` package (originally ``datasets/``, renamed to avoid
shadowing Hugging Face's ``datasets`` pip package once the smolvla
extra installs it). All ``from modelblaster.datasets import ...`` call
sites moved to ``from modelblaster.mb_datasets import ...``; the
runtime path string ``datasets/idsia/...`` for image dirs is
unchanged (it was always a runtime-resolved data path, not a Python
import).

## What's NOT in scope for admission

* **int8 PTQ on SmolVLA.** The walker in ``extract_graph_export.py``
  handles the aten ops ViNT/YOLOv8 emit; SmolVLA emits attention via
  ``scaled_dot_product_attention``, ``rotary_emb``, masked
  embeddings, and a softmax-heavy cross-attention block that the
  walker does not lower to int8 yet. fp16 codegen is the realistic
  near-term path; int8 is a separate research project.
* **Real calibration data.** ``get_calibration_spec`` declares the
  source as ``dummy_processed_inputs``; this is the same dummy batch
  ``Understanding-PI0`` uses for one-shot tracing. Replace with a
  LeRobot dataset hook when accuracy tuning becomes the focus.
* **Cross-attn `KernelSpec` entries.** The aten attention block does
  not yet have a matching curated kernel for any RISC-V backend.
  Until that lands, ``BACKEND=reference`` is the only viable path on
  ``rvv_opu`` / ``gemmini*``.

## Bringing it up

One-time prereqs (Python 3.12+, lerobot + smolvla deps,
Understanding-PI0 sibling, HF cache for ``lerobot/smolvla_base``):

```bash
uv sync --extra smolvla
export PI0_ROOT=/scratch2/agustin/merlin/third_party/Understanding-PI0
export LEROBOT_ROOT=/scratch2/agustin/merlin/third_party/lerobot
export HF_HOME=/scratch2/agustin/hf_cache
```

Trace + inventory:

```bash
uv run python -m modelblaster.pipeline.extract_graph_export \
    --model smolvla --quant fp16 --inventory-only \
    --out-dir /tmp/smolvla-trace
```

Direct policy-load smoke (no IR emit; useful for confirming the
sibling-repo wiring on a new machine):

```bash
uv run python -c "
from modelblaster.models import smolvla
m = smolvla.get_model()
inputs = smolvla.get_sample_input()
import torch
with torch.inference_mode(): out = m(*inputs)
print('out:', tuple(out.shape), out.dtype)
"
```

## Override env vars

| env var                           | default                    | what       |
|-----------------------------------|----------------------------|------------|
| ``MODELBLASTER_SMOLVLA_MODEL_ID`` | ``lerobot/smolvla_base``   | HF repo id |
| ``MODELBLASTER_SMOLVLA_PROMPT_LEN`` | ``8``                    | dummy prompt token count |
| ``MODELBLASTER_SMOLVLA_IMAGE_HW`` | ``256x256``                | source resolution before SmolVLM resize |
| ``MODELBLASTER_SMOLVLA_DEVICE``   | ``cpu``                    | export requires CPU |
| ``PI0_ROOT`` / ``UNDERSTANDING_PI0_ROOT`` | (fallback path) | Understanding-PI0 repo |
| ``LEROBOT_ROOT``                  | (fallback path)            | lerobot repo |
| ``HF_HOME``                       | ``/scratch2/agustin/hf_cache`` (fallback) | HF snapshot dir |
