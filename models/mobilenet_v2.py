"""MobileNetV2 model wrapper for the modelblaster flow.

Uses torchvision's MobileNetV2 with a small width_mult and modest input
resolution to keep the on-spike runtime bounded — the architecture (DW +
PW conv blocks, ReLU6 activations, global avg pool, classifier head)
matches the canonical published model. Override the channel multiplier
and input resolution via env vars; defaults pick a configuration that's
small enough to verify end-to-end on spike in single-digit minutes.

Env knobs:
  MODELBLASTER_MOBILENETV2_WIDTH_MULT   default 0.25 (8x channel reduction
                                  vs the standard 1.0)
  MODELBLASTER_MOBILENETV2_INPUT        default 96 (input is 1x3xNxN); pick a
                                  multiple of 32 so the strided blocks
                                  don't fight rounding
  MODELBLASTER_MOBILENETV2_NUM_CLASSES  default 1000
"""

from __future__ import annotations

import os

import torch


def _cfg() -> tuple[float, int, int]:
    width_mult = float(os.environ.get("MODELBLASTER_MOBILENETV2_WIDTH_MULT", 0.25))
    input_size = int(os.environ.get("MODELBLASTER_MOBILENETV2_INPUT", 96))
    num_classes = int(os.environ.get("MODELBLASTER_MOBILENETV2_NUM_CLASSES", 1000))
    return width_mult, input_size, num_classes


def get_model(seed: int = 0):
    import torchvision.models as tv

    torch.manual_seed(seed)
    width_mult, _input_size, num_classes = _cfg()

    # Random init — we don't have a trained checkpoint at this width, and
    # numerical correctness vs the same-init PyTorch reference doesn't
    # care about the actual weight values. The downstream verify pass
    # snapshots the same random init via this seed.
    m = tv.mobilenet_v2(
        weights=None,
        width_mult=width_mult,
        num_classes=num_classes,
    )
    m.eval()
    return m


def get_sample_input(seed: int = 1) -> torch.Tensor:
    """Synthetic NCHW frame at the configured resolution."""
    _, input_size, _ = _cfg()
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 3, input_size, input_size, generator=g)
