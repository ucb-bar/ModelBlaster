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


def _load_idsia_image(path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    """ImageNet-normalized resize → tensor pipeline (matches ViNT's training
    preprocessing — see pilot_forest_with_vint.py::_vint_transform)."""
    # Local imports keep the module light when only get_model() is needed.
    from PIL import Image as PILImage  # noqa: PLC0415
    from torchvision import transforms  # noqa: PLC0415
    W, H = image_size  # yaml stores as (W, H)
    tfm = transforms.Compose([
        transforms.Resize((H, W)),  # torchvision Resize takes (H, W)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return tfm(PILImage.open(path).convert("RGB"))


def get_calibration_samples(
    n_samples: int = 32,
    sample_dir: Optional[str] = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return a list of ``(obs, goal)`` calibration tuples drawn from the
    IDSIA forest-trail samples that the pilot script uses.

    Each sample's obs is a rolling 6-frame stack along channel dim
    (matching ViNT's ``context_size+1=6`` input) and the goal is a
    fixed end-of-trail image. The first ``context_size`` frames of
    obs are repeats of the first image (to bootstrap the rolling
    window without padding artifacts).

    Override the source directory with the ``AGENTS_VINT_CALIB_DIR``
    env var. Falls back to ``torch.randn()`` samples if the dir is
    missing — useful for CI / smoke tests on machines without the
    IDSIA data.
    """
    import os
    cfg = _load_config()
    W, H = cfg["image_size"]
    ctx = cfg["context_size"]
    n_obs_channels = 3 * (ctx + 1)

    sample_dir = sample_dir or os.environ.get(
        "AGENTS_VINT_CALIB_DIR",
        str(_REPO_ROOT / "datasets/idsia/samples/sc"),
    )
    samples_root = Path(sample_dir)
    if not samples_root.is_dir():
        print(f"[vint.get_calibration_samples] WARN: {samples_root} not "
              f"found; falling back to torch.randn calibration. The "
              f"extracted activation scales will reflect noise, not "
              f"real navigation imagery.", flush=True)
        g = torch.Generator().manual_seed(0)
        return [(
            torch.randn(1, n_obs_channels, H, W, generator=g),
            torch.randn(1, 3, H, W, generator=g),
        ) for _ in range(n_samples)]

    image_paths = sorted(samples_root.glob("*.jpg"))
    if not image_paths:
        image_paths = sorted(samples_root.glob("*.png"))
    if len(image_paths) < 2:
        raise RuntimeError(
            f"{samples_root} has fewer than 2 images; need at least one "
            f"frame for the goal and one for the rolling obs window.")
    images = [_load_idsia_image(p, (W, H)) for p in image_paths]
    samples: list[tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(n_samples):
        # Build a rolling 6-frame context. When the calibration count
        # exceeds the available image count we cycle with stride so
        # consecutive obs windows differ.
        anchor = (i * max(1, len(images) // n_samples)) % len(images)
        context = []
        for k in range(ctx + 1):
            idx = max(0, anchor - (ctx - k)) % len(images)
            context.append(images[idx])
        obs = torch.cat(context, dim=0).unsqueeze(0)  # (1, 3*(ctx+1), H, W)
        # Vary the goal too — earlier versions pinned it to images[-1]
        # for "every sample sees the end of trail", but that meant the
        # goal_encoder calibration only saw ONE input distribution.
        # On ViNT the goal pipeline produces large-magnitude activations
        # (max_abs ~164 at the compress output vs ~28 for obs); without
        # diverse goal samples the per-tensor scale fits poorly and the
        # int8 forward of the goal encoder collapses information
        # (cos-sim vs PyTorch fp32 → ~0 at the goal_encoder output).
        # Use a stride-offset goal so each calibration sample sees a
        # different goal image while still covering the IDSIA spread.
        goal_idx = (anchor + len(images) // 2) % len(images)
        goal = images[goal_idx].unsqueeze(0)
        samples.append((obs, goal))
    return samples
