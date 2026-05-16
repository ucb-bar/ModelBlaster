"""ViNT model wrapper for the agents flow.

Sources the canonical ViNT PyTorch class from the vendored
``visualnav-transformer`` repo (under ``sims/external/``) and loads the
published pretrained checkpoint by default. Architecture is taken from
``train/config/vint.yaml``:

    context_size       = 5         # past frames in the rolling window
    len_traj_pred      = 5         # predicted future waypoints
    learn_angle        = True      # waypoints include (sinθ, cosθ)
    obs_encoder        = efficientnet-b0
    obs_encoding_size  = 512
    late_fusion        = False     # goal_encoder takes obs[-1]+goal stacked
    mha attention      = 4 heads × 4 layers, ff_dim_factor=4
    image_size         = (85, 64)  # W, H

Override the checkpoint with ``AGENTS_VINT_CKPT``; override the config
yaml with ``AGENTS_VINT_CFG``.

NOTE: ViNT cannot be traced with ``torch.fx.symbolic_trace`` —
EfficientNet's internals use ``len(...)`` calls plus ViNT.forward has
in-place index assignments and the TransformerEncoder uses
``nn.MultiheadAttention`` whose lowered form is not FX-traceable.
``torch.export`` does trace cleanly (1690 aten nodes), so the
agents-pipeline extractor that consumes this model has to use the
export-based ingest path, not the FX-based one. See
``agents/pipeline/extract_graph_export.py`` (Phase A of the ViNT
plan in ``agents/notes/vint_zephyr_plan.md``).

Calibration: ``get_sample_input()`` returns a single
``(obs, goal)`` tuple of shape ``(1, 18, H, W)`` and ``(1, 3, H, W)``
suitable for one-shot tracing. For real PTQ, the caller should pass
multiple calibration samples (from the IDSIA forest-trail dataset, or
rendered from the IsaacLab forest env) — the extractor accepts a
list of samples via its ``--calibration-glob`` flag.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[3]
_VINT_TRAIN = _REPO_ROOT / "sims/external/visualnav-transformer/train"
_DEFAULT_CFG = _VINT_TRAIN / "config/vint.yaml"
_DEFAULT_CKPT = (
    _REPO_ROOT / "sims/external/visualnav-transformer"
    / "deployment/model_weights/vint.pth"
)

# Make vint_train importable. The vendored repo lives outside the
# zephyr-chipyard-sw subtree, so this is a one-time path injection.
if str(_VINT_TRAIN) not in sys.path:
    sys.path.insert(0, str(_VINT_TRAIN))


def _load_config() -> dict:
    cfg_path = Path(os.environ.get("AGENTS_VINT_CFG", _DEFAULT_CFG))
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"ViNT config yaml not found at {cfg_path}. Set AGENTS_VINT_CFG "
            f"to override."
        )
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _build_module(cfg: dict):
    # Local import so the rest of this module doesn't pay the import cost
    # unless someone actually calls get_model().
    from vint_train.models.vint.vint import ViNT  # noqa: PLC0415
    return ViNT(
        context_size=cfg["context_size"],
        len_traj_pred=cfg["len_traj_pred"],
        learn_angle=cfg["learn_angle"],
        obs_encoder=cfg["obs_encoder"],
        obs_encoding_size=cfg["obs_encoding_size"],
        late_fusion=cfg["late_fusion"],
        mha_num_attention_heads=cfg["mha_num_attention_heads"],
        mha_num_attention_layers=cfg["mha_num_attention_layers"],
        mha_ff_dim_factor=cfg["mha_ff_dim_factor"],
    )


def get_model() -> torch.nn.Module:
    """Load ViNT with pretrained weights (or warn if the checkpoint is
    missing) and return the eval-mode module."""
    cfg = _load_config()
    model = _build_module(cfg).eval()

    ckpt_path = Path(os.environ.get("AGENTS_VINT_CKPT", _DEFAULT_CKPT))
    if not ckpt_path.is_file():
        print(
            f"[vint.get_model] WARN: checkpoint not found at {ckpt_path}; "
            f"using RANDOM-INIT weights (extract output will be meaningless). "
            f"Download via "
            f"https://drive.google.com/drive/folders/"
            f"1a9yWR2iooXFAqjQHetz263--4_2FFggg",
            flush=True,
        )
        return model

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    loaded = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state_dict = (
        loaded.module.state_dict()
        if hasattr(loaded, "module")
        else loaded.state_dict()
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[vint.get_model] {len(missing)} missing keys "
              f"(e.g. {missing[:3]})", flush=True)
    if unexpected:
        print(f"[vint.get_model] {len(unexpected)} unexpected keys "
              f"(e.g. {unexpected[:3]})", flush=True)
    return model


def get_sample_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Return one ``(obs, goal)`` tuple sized per the published config.

    Single sample is fine for graph tracing. For PTQ calibration the
    caller should use ``get_calibration_samples(n)`` instead — random
    gaussian doesn't activate ViNT's learned filters anywhere near
    the deployment distribution, and per-tensor max_abs scales
    computed from torch.randn() are essentially noise.
    """
    cfg = _load_config()
    W, H = cfg["image_size"]
    n_obs_channels = 3 * (cfg["context_size"] + 1)
    # Use a fixed seed so the calibration is reproducible across runs.
    g = torch.Generator().manual_seed(0)
    obs = torch.randn(1, n_obs_channels, H, W, generator=g)
    goal = torch.randn(1, 3, H, W, generator=g)
    return obs, goal


def get_calibration_spec(num_samples: int = 16) -> dict:
    """Return a declarative calibration spec for the agents/datasets
    loader. The walker materializes (obs, goal) tuples from this and
    serializes the spec into the generated/ dir for reproducibility.

    Two inputs:
     * obs_img: rolling 6-frame stack along channel dim — drawn from
       IDSIA still images by default. Override the dir with
       ``AGENTS_VINT_OBS_DATASET`` (path under datasets/).
     * goal_img: one image per sample — IsaacLab forest renders by
       preference (matches the deployment distribution; the
       goal_encoder was the main int8 drift source when calibrated
       with IDSIA stills, see inspect_intermediates.py results).
       Falls back to IDSIA if the IsaacLab render dir doesn't exist
       (with a clear warning).
    """
    import os
    cfg = _load_config()
    W, H = cfg["image_size"]
    ctx = cfg["context_size"]
    obs_dir = os.environ.get(
        "AGENTS_VINT_OBS_DATASET", "datasets/idsia/samples/sc")
    # _REPO_ROOT (= parents[3]) is already FreshScheduler; the renders
    # live directly under it, not under its parent.
    goal_render_dir = _REPO_ROOT / "datasets/isaaclab_forest_renders"
    if goal_render_dir.is_dir():
        goal_input = {
            "loader": "isaaclab_forest_render",
            "path": str(goal_render_dir),
            "image_size": [W, H],
        }
    else:
        print(f"[vint.get_calibration_spec] WARN: {goal_render_dir} not "
              f"found — falling back to IDSIA images for goal calibration. "
              f"For real deployment accuracy, render IsaacLab forest goals "
              f"via sims/scripts/utils/render_vint_calibration.py.",
              flush=True)
        goal_input = {
            "loader": "image_dir",
            "path": obs_dir,
            "image_size": [W, H],
        }
    return {
        "num_samples": num_samples,
        "inputs": {
            "obs_img": {
                "loader": "image_dir",
                "path": obs_dir,
                "image_size": [W, H],
                "compose": {
                    "kind": "rolling_window",
                    "frames_per_sample": ctx + 1,
                },
            },
            "goal_img": {
                **goal_input,
                "compose": {"kind": "one_per_sample"},
            },
        },
    }


def get_precision_spec() -> dict:
    """Default per-op precision overrides for the ViNT extract path.

    The int8 PTQ flow is accurate everywhere except the goal-encoder's
    final ``linear`` op: that tensor has |max|≈181 vs typical magnitudes
    ~5, so per-tensor symmetric int8 wastes 36× of its representable
    resolution on the long tail. Per-op fp16 promotion lets one linear
    keep its dynamic range while the rest of the network stays in int8.

    See ``agents/notes/mixed_precision_plan.md`` for the architecture.
    The auto-cast pass in ``extract_graph_export`` inserts
    ``cast_i8_to_f16`` before this op and ``cast_f16_to_i8`` after it
    (when its downstream consumer is back in int8 land).

    The CLI ``--fp16-ops a,b,c`` flag is additive to this spec.
    """
    return {
        "default": "int8",
        # Promote a contiguous region from goal-encoder output through
        # the transformer's first LayerNorm. A single-op promotion of
        # `linear` alone gave zero accuracy benefit because the auto-
        # cast pass immediately re-quantizes the f16 output back to i8
        # (with the same wide-range per-tensor scale that was the
        # original drift source). Keeping cat_1 + the first
        # layer_norm in f16 too lets the LayerNorm zero-mean its input
        # *before* we cast back to int8 — the cast then runs against a
        # tight-range tensor, not the 181-magnitude goal output.
        "fp16_ops": [
            "linear",          # goal-encoder output projection
            "cat_1",           # obs+goal token concatenation
            "layer_norm",      # transformer's first LayerNorm
        ],
    }


def get_calibration_samples(
    n_samples: int = 32,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Materialize (obs, goal) calibration tuples via the agents.datasets
    spec resolver. Returns ``[(obs (1, 18, H, W), goal (1, 3, H, W)), ...]``
    in the order the model's forward expects.

    Back-compat wrapper around ``get_calibration_spec`` for callers
    that don't want to plumb the spec themselves.
    """
    from agents.datasets import materialize_calibration_samples  # noqa: PLC0415
    spec = get_calibration_spec(n_samples)
    materialized = materialize_calibration_samples(spec)
    return [(d["obs_img"], d["goal_img"]) for d in materialized]
