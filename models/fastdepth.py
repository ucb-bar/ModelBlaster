"""FastDepth — MobileNet encoder + NNConv5 decoder, monocular depth.

The canonical embedded depth network (Wofk et al., ICRA 2019): a MobileNet
feature extractor followed by a light upsampling decoder built from depthwise
separable convolutions and nearest-neighbour upsampling. It is the natural
depth companion to the drone models already in this tree — DroNet does
collision/steering, ViNT does goal-conditioned navigation, and this does the
dense depth those policies normally consume.

Reuses torchvision's MobileNetV2 features as the encoder rather than vendoring
the original paper's MobileNet, because models/mobilenet_v2.py already
establishes that path in this repo and it keeps the op set to things the
ModelBlaster backends already cover.

Shape/size knobs, same convention as the other models here:

  MODELBLASTER_FASTDEPTH_INPUT       default 128 (input is 1x3xNxN). Must be a
                                     multiple of 32: the encoder downsamples
                                     five times, and the decoder's five
                                     nearest-neighbour upsamples have to land
                                     back on the input resolution exactly.
  MODELBLASTER_FASTDEPTH_WIDTH_MULT  default 0.25, matching mobilenet_v2.py --
                                     keeps weights small enough for the SoC's
                                     DRAM budget and the int8 calibration fast.
  MODELBLASTER_FASTDEPTH_DECODER_CH  default 64, channels at the widest decoder
                                     stage; halves at each subsequent stage.

The decoder deliberately uses `nn.Upsample(scale_factor=2, mode='nearest')`
rather than a transposed convolution: nearest-neighbour upsampling is what the
paper's NNConv5 uses, and it is also the op ModelBlaster and XNNPACK both
already handle, whereas conv-transpose was the one op class the ExecuTorch
numerics sweep flagged as diverging.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn


def _cfg() -> tuple[int, float, int]:
    input_size = int(os.environ.get("MODELBLASTER_FASTDEPTH_INPUT", 128))
    width_mult = float(os.environ.get("MODELBLASTER_FASTDEPTH_WIDTH_MULT", 0.25))
    dec_ch = int(os.environ.get("MODELBLASTER_FASTDEPTH_DECODER_CH", 64))
    if input_size % 32:
        raise ValueError(
            f"MODELBLASTER_FASTDEPTH_INPUT={input_size} must be a multiple of 32: "
            f"the encoder downsamples 5x and the decoder upsamples 5x, so anything "
            f"else lands the output on a different resolution than the input.")
    return input_size, width_mult, dec_ch


def _sep_conv(cin: int, cout: int) -> nn.Sequential:
    """Depthwise-separable 5x5, the decoder block from NNConv5.

    Depthwise then pointwise, each with its own BN+ReLU. The 5x5 depthwise is
    the paper's choice -- a wider receptive field costs almost nothing when the
    convolution is per-channel.
    """
    return nn.Sequential(
        nn.Conv2d(cin, cin, kernel_size=5, padding=2, groups=cin, bias=False),
        nn.BatchNorm2d(cin),
        nn.ReLU(inplace=False),
        nn.Conv2d(cin, cout, kernel_size=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=False),
    )


class FastDepth(nn.Module):
    def __init__(self, width_mult: float = 0.25, dec_ch: int = 64):
        super().__init__()
        from torchvision.models import mobilenet_v2
        # No pretrained weights: this runs as a shape/throughput benchmark and
        # against its own PyTorch golden, so random init is honest. Anything
        # claiming depth ACCURACY would need the paper's NYUv2 checkpoint.
        self.encoder = mobilenet_v2(weights=None, width_mult=width_mult).features
        enc_out = self.encoder[-1].out_channels

        chs = [dec_ch, dec_ch // 2, dec_ch // 4, dec_ch // 8, dec_ch // 16]
        chs = [max(c, 8) for c in chs]
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = _sep_conv(enc_out, chs[0])
        self.dec2 = _sep_conv(chs[0], chs[1])
        self.dec3 = _sep_conv(chs[1], chs[2])
        self.dec4 = _sep_conv(chs[2], chs[3])
        self.dec5 = _sep_conv(chs[3], chs[4])
        self.head = nn.Conv2d(chs[4], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.dec1(self.up(x))
        x = self.dec2(self.up(x))
        x = self.dec3(self.up(x))
        x = self.dec4(self.up(x))
        x = self.dec5(self.up(x))
        return self.head(x)


def get_model(seed: int = 0):
    torch.manual_seed(seed)
    _input_size, width_mult, dec_ch = _cfg()
    m = FastDepth(width_mult=width_mult, dec_ch=dec_ch)
    m.eval()
    return m


def get_sample_input(seed: int = 1) -> torch.Tensor:
    input_size, _wm, _dc = _cfg()
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 3, input_size, input_size, generator=g)
