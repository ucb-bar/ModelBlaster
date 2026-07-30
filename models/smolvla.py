"""SmolVLA model wrapper for the modelblaster flow.

Loads ``lerobot/smolvla_base`` through the canonical ``understanding_pi0``
``SmolVLAOneStepNoCacheWrapper`` (one-step action inference, no KV
cache). Two upstream packages are consumed off-tree:

* ``Understanding-PI0`` at ``PI0_ROOT`` / ``UNDERSTANDING_PI0_ROOT``,
  shipping the SmolVLA loader + the no-cache wrapper.
* ``lerobot`` at ``LEROBOT_ROOT`` (the canonical clone is the merlin
  sibling checkout at ``$MERLIN_DIR/third_party/lerobot``, default
  ``../merlin/third_party/lerobot``), shipping the ``SmolVLAPolicy``
  class and constant names.

Both are pulled in via ``sys.path`` injection rather than as pip
dependencies; the upstream pyproject deps would force diffusers /
accelerate / transformers into every modelblaster venv whether the
caller needs SmolVLA or not. ``HF_HOME`` should point at the same
cache that holds the ``models--lerobot--smolvla_base`` snapshot
(defaults to ``~/.cache/huggingface``); the wrapper does not download
weights itself.

Tracing: SmolVLA can NOT be FX-symbolically-traced — the policy uses
``len()`` on dynamic tensors, list comprehensions over an unknown
number of camera streams, and an HF Transformers model whose internals
are not FX-traceable. The walker that consumes this module must go
through ``pipeline/extract_graph_export.py`` (torch.export), same as
ViNT.

Inputs: ``get_sample_input()`` returns a flat tuple of
``2 * num_cams + 5`` tensors:

    [img_0 … img_{C-1}, mask_0 … mask_{C-1},
     lang_tokens, lang_masks, state, noisy_actions, timestep]

Sized per the published ``smolvla_base`` config: ``num_cams`` is
inferred from the policy's image processor (typically 3 cameras at
256×256), ``prompt_len`` defaults to 8 tokens.

Environment overrides:

* ``MODELBLASTER_SMOLVLA_MODEL_ID``  default ``lerobot/smolvla_base``
* ``MODELBLASTER_SMOLVLA_PROMPT_LEN`` default ``8``
* ``MODELBLASTER_SMOLVLA_IMAGE_HW``   default ``256x256``
* ``MODELBLASTER_SMOLVLA_DEVICE``     default ``cpu`` (export needs CPU)
* ``PI0_ROOT`` / ``UNDERSTANDING_PI0_ROOT`` — Understanding-PI0 repo root
* ``LEROBOT_ROOT`` — lerobot repo root (with src/lerobot inside)
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Optional

import torch


_DEFAULT_MODEL_ID = "lerobot/smolvla_base"
_DEFAULT_PROMPT_LEN = 8
_DEFAULT_IMAGE_HW = (256, 256)
_DEFAULT_DEVICE = "cpu"

# Canonical sibling clones on this dev box. Used as fallbacks when the
# corresponding *_ROOT env vars are not set. Both are real upstream
# checkouts, not vendored copies.
_MERLIN_ROOT = Path(os.environ.get("MERLIN_DIR", str(Path(__file__).resolve().parents[2] / "merlin")))
_FALLBACK_PI0_ROOT = _MERLIN_ROOT / "third_party" / "Understanding-PI0"
_FALLBACK_LEROBOT_ROOT = _MERLIN_ROOT / "third_party" / "lerobot"
_FALLBACK_HF_HOME = Path.home() / ".cache" / "huggingface"


def _resolve_root(env_vars: tuple[str, ...], fallback: Path,
                  label: str) -> Path:
    for var in env_vars:
        v = os.environ.get(var)
        if v:
            p = Path(v).expanduser().resolve()
            if p.exists():
                return p
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"{label}: none of the env vars {env_vars} resolve to an "
        f"existing directory, and the fallback {fallback} does not "
        f"exist either."
    )


def _prepend_sys_path(p: Path) -> None:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def _stub_groot_submodules() -> None:
    """``lerobot.policies.__init__`` imports
    ``lerobot.policies.groot.*`` whose dataclass surface is currently
    broken (upstream issue). Pre-register empty stub modules so the
    parent ``__init__.py`` succeeds without dragging in the broken
    GR00T tree. Idempotent: only stubs entries that are not already
    real modules."""
    stubs = (
        "lerobot.policies.groot",
        "lerobot.policies.groot.configuration_groot",
        "lerobot.policies.groot.modeling_groot",
        "lerobot.policies.groot.groot_n1",
    )
    for name in stubs:
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        mod.__path__ = []  # type: ignore[attr-defined]
        # Some lerobot top-level imports reach for ``GrootConfig`` even
        # without instantiating it; provide a sentinel dataclass-like
        # attribute so the import-time read succeeds.
        if name.endswith(".configuration_groot"):
            mod.GrootConfig = type("GrootConfig", (), {})  # type: ignore[attr-defined]
        sys.modules[name] = mod


def _ensure_environment() -> None:
    """Resolve PI0 + lerobot + HF_HOME and wire them onto sys.path /
    env. Runs at import time so any downstream import of lerobot
    inside this module sees the correct paths."""
    pi0_root = _resolve_root(
        ("PI0_ROOT", "UNDERSTANDING_PI0_ROOT"),
        _FALLBACK_PI0_ROOT, "Understanding-PI0 repo",
    )
    _prepend_sys_path(pi0_root)

    lerobot_root = _resolve_root(
        ("LEROBOT_ROOT",),
        _FALLBACK_LEROBOT_ROOT, "lerobot repo",
    )
    # The lerobot package directory is at <root>/src/lerobot when the
    # repo was checked out with its modern src-layout, or at
    # <root>/lerobot for older layouts.
    src_layout = lerobot_root / "src"
    _prepend_sys_path(src_layout if src_layout.is_dir() else lerobot_root)

    if not os.environ.get("HF_HOME") and _FALLBACK_HF_HOME.exists():
        os.environ["HF_HOME"] = str(_FALLBACK_HF_HOME)


# Side effect at import time -- match how vint.py prepares vint_train.
_ensure_environment()
_stub_groot_submodules()


def _image_hw() -> tuple[int, int]:
    raw = os.environ.get("MODELBLASTER_SMOLVLA_IMAGE_HW")
    if not raw:
        return _DEFAULT_IMAGE_HW
    try:
        h_s, w_s = raw.lower().split("x", 1)
        return int(h_s), int(w_s)
    except ValueError as e:
        raise ValueError(
            f"MODELBLASTER_SMOLVLA_IMAGE_HW must be HxW (e.g. 256x256); "
            f"got {raw!r}"
        ) from e


def _device() -> str:
    return os.environ.get("MODELBLASTER_SMOLVLA_DEVICE", _DEFAULT_DEVICE)


def _build_policy_and_wrapper():
    """One-shot: load the upstream policy + wrap it in the no-cache
    one-step module. Returns (wrapper, flat_inputs, num_cams)."""
    from understanding_pi0.smolvla_mx.loader import (  # type: ignore[import]
        build_dummy_processed_inputs,
        load_smolvla_policy,
    )
    from understanding_pi0.smolvla_mx.wrappers import (  # type: ignore[import]
        SmolVLAOneStepNoCacheWrapper,
        flatten_processed_inputs,
    )

    model_id = os.environ.get(
        "MODELBLASTER_SMOLVLA_MODEL_ID", _DEFAULT_MODEL_ID,
    )
    prompt_len = int(
        os.environ.get("MODELBLASTER_SMOLVLA_PROMPT_LEN",
                       _DEFAULT_PROMPT_LEN)
    )
    image_hw = _image_hw()
    device = _device()

    policy = load_smolvla_policy(model_id=model_id, device=device)
    processed = build_dummy_processed_inputs(
        policy, batch_size=1, image_hw=image_hw,
        prompt_len=prompt_len, device=device,
    )
    flat_inputs = flatten_processed_inputs(processed)
    num_cams = (len(flat_inputs) - 5) // 2
    wrapper = SmolVLAOneStepNoCacheWrapper(policy, num_cams=num_cams).eval()
    return wrapper, flat_inputs, num_cams


# Cached so callers that ask for both the model and the sample input
# back-to-back do not pay the load cost twice (the policy weights are
# ~hundreds of MB; instantiation is the dominant cost).
_BUNDLE: Optional[tuple[torch.nn.Module, tuple[torch.Tensor, ...], int]] = None


def _bundle():
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = _build_policy_and_wrapper()
    return _BUNDLE


def get_model() -> torch.nn.Module:
    """Return the SmolVLAOneStepNoCacheWrapper in eval mode."""
    wrapper, _, _ = _bundle()
    return wrapper


def get_sample_input() -> tuple[torch.Tensor, ...]:
    """Return a flat tuple of ``2 * num_cams + 5`` tensors suitable
    for ``torch.export``. Order matches the wrapper's ``forward``
    contract: image cams, image masks, then lang_tokens / lang_masks /
    state / noisy_actions / timestep."""
    _, flat_inputs, _ = _bundle()
    return tuple(flat_inputs)


def get_calibration_spec(num_samples: int = 1) -> dict:
    """Calibration spec for the modelblaster/datasets loader.

    SmolVLA admission is intentionally limited to fp16 (no int8 PTQ)
    in this phase; the spec exists so the extractor still emits the
    declarative metadata that the rest of the pipeline expects. The
    spec describes a single deterministic dummy sample sized per the
    image / prompt env overrides.
    """
    h, w = _image_hw()
    prompt_len = int(
        os.environ.get("MODELBLASTER_SMOLVLA_PROMPT_LEN",
                       _DEFAULT_PROMPT_LEN)
    )
    return {
        "num_samples": int(num_samples),
        "source": "dummy_processed_inputs",
        "image_hw": [h, w],
        "prompt_len": prompt_len,
    }
