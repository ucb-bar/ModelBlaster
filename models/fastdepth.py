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
  MODELBLASTER_FASTDEPTH_SKIPS       default 0. 1 adds the paper's additive
                                     skip connections from the encoder into
                                     each decoder stage. OFF by default so the
                                     already-measured latency graphs reproduce
                                     unchanged; the TRAINED checkpoint needs
                                     them, because a decoder fed only by the
                                     1/32-resolution bottleneck cannot recover
                                     depth edges and scores far worse.
  MODELBLASTER_FASTDEPTH_PRETRAINED  default 0. 1 initialises the encoder from
                                     torchvision's ImageNet weights, which
                                     exist ONLY for width_mult=1.0 -- any other
                                     width raises rather than silently
                                     returning a randomly-initialised encoder.
  MODELBLASTER_FASTDEPTH_CKPT        path to a trained state_dict. This is what
                                     turns the model from a shape/throughput
                                     benchmark into something whose depth
                                     output means anything.
  MODELBLASTER_FASTDEPTH_NYU_DIR     directory of the original NYU HDF5
                                     frames. Preferred over _CALIB when both
                                     are set: a spec plus a dataset is
                                     reproducible, whereas an .npz of
                                     preprocessed floats has to be trusted.
                                     Needs h5py; _CALIB does not.
  MODELBLASTER_FASTDEPTH_CALIB       path to an .npz of REAL, already
                                     ImageNet-normalised NYU frames (key
                                     "samples", NxCxHxW). Used for int8
                                     activation calibration and as the golden
                                     anchor. Without it the calibration set is
                                     torch.randn, whose activation ranges have
                                     nothing to do with the ranges a trained
                                     encoder actually sees -- the int8 model
                                     would still verify bit-exact against its
                                     own golden while being quantised to the
                                     wrong scales.

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


#: torchvision mobilenet_v2 feature indices whose OUTPUT sits at 1/2, 1/4,
#: 1/8 and 1/16 of the input. The decoder upsamples 1/32 -> 1/16 -> ... -> 1/1,
#: so these are consumed in reverse. Taken by running the encoder rather than
#: by arithmetic on width_mult: _make_divisible rounds channel counts, and
#: guessing them is how a skip projection silently gets the wrong width.
_TAPS = (1, 3, 6, 13)


class FastDepth(nn.Module):
    def __init__(self, width_mult: float = 0.25, dec_ch: int = 64,
                 skips: bool = False, pretrained: bool = False):
        super().__init__()
        from torchvision.models import mobilenet_v2
        if pretrained and abs(width_mult - 1.0) > 1e-9:
            raise ValueError(
                f"MODELBLASTER_FASTDEPTH_PRETRAINED=1 needs width_mult=1.0; "
                f"torchvision ships ImageNet weights only for the full-width "
                f"MobileNetV2, and got width_mult={width_mult}. Silently "
                f"falling back to random init would look like a trained model.")
        self.encoder = mobilenet_v2(
            weights="IMAGENET1K_V1" if pretrained else None,
            width_mult=width_mult).features
        enc_out = self.encoder[-1].out_channels
        self.skips = bool(skips)

        chs = [dec_ch, dec_ch // 2, dec_ch // 4, dec_ch // 8, dec_ch // 16]
        chs = [max(c, 8) for c in chs]
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = _sep_conv(enc_out, chs[0])
        self.dec2 = _sep_conv(chs[0], chs[1])
        self.dec3 = _sep_conv(chs[1], chs[2])
        self.dec4 = _sep_conv(chs[2], chs[3])
        self.dec5 = _sep_conv(chs[3], chs[4])
        self.head = nn.Conv2d(chs[4], 1, kernel_size=1)

        if self.skips:
            # 1x1 projections so an encoder tap can be ADDED to the decoder
            # stage at the same resolution. Widths are measured, not derived.
            # eval() for the probe (BatchNorm rejects a 1x1 spatial map in
            # train mode) and 64x64 so the 1/32 stage is still 2x2. The module
            # is constructed in train mode, so restore it afterwards.
            was_training = self.encoder.training
            self.encoder.eval()
            with torch.no_grad():
                widths, h = [], torch.zeros(1, 3, 64, 64)
                for i, blk in enumerate(self.encoder):
                    h = blk(h)
                    if i in _TAPS:
                        widths.append(h.shape[1])
            self.encoder.train(was_training)
            # decoder stage i lands at the resolution of tap (3 - i)
            self.skip_proj = nn.ModuleList([
                nn.Conv2d(widths[3 - i], chs[i], kernel_size=1, bias=False)
                for i in range(4)])

    def _encode(self, x):
        taps = []
        for i, blk in enumerate(self.encoder):
            x = blk(x)
            if i in _TAPS:
                taps.append(x)
        return x, taps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.skips:
            x = self.encoder(x)
            x = self.dec1(self.up(x))
            x = self.dec2(self.up(x))
            x = self.dec3(self.up(x))
            x = self.dec4(self.up(x))
            x = self.dec5(self.up(x))
            return self.head(x)
        x, taps = self._encode(x)
        for i, dec in enumerate((self.dec1, self.dec2, self.dec3, self.dec4)):
            x = dec(self.up(x)) + self.skip_proj[i](taps[3 - i])
        x = self.dec5(self.up(x))
        return self.head(x)


def get_model(seed: int = 0):
    torch.manual_seed(seed)
    _input_size, width_mult, dec_ch = _cfg()
    m = FastDepth(width_mult=width_mult, dec_ch=dec_ch,
                  skips=os.environ.get("MODELBLASTER_FASTDEPTH_SKIPS", "0") == "1",
                  pretrained=os.environ.get("MODELBLASTER_FASTDEPTH_PRETRAINED", "0") == "1")
    ckpt = os.environ.get("MODELBLASTER_FASTDEPTH_CKPT", "")
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu")
        sd = sd.get("model", sd)
        # strict: a checkpoint trained with a different width or skip setting
        # would otherwise load partially and quantize to a model that is part
        # trained and part random, which no metric would flag.
        m.load_state_dict(sd, strict=True)
    m.eval()
    return m


#: ImageNet statistics. These MUST match what the training pipeline applied
#: (experiments/fastdepth_train/scripts/train_fastdepth.py): the encoder is
#: pretrained, and feeding it a different normalisation silently degrades
#: every downstream number without failing anything.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_calibration_spec(num_samples: int = 8) -> "dict | None":
    """Declarative calibration spec, the convention the rest of this tree uses.

    Single-input model; the input tensor is named ``x`` in the IR. Returns
    None -- meaning "I have no calibration data configured, use
    get_sample_input()" -- when neither source env var is set, which is the
    untrained shape/throughput benchmark path.

    Two sources, preferring the reproducible one:

      MODELBLASTER_FASTDEPTH_NYU_DIR   a directory of the original NYU HDF5
                                       frames. Preferred: the spec plus the
                                       dataset regenerate the exact scales,
                                       which an .npz of preprocessed floats
                                       cannot promise on its own.
      MODELBLASTER_FASTDEPTH_CALIB     the pre-baked .npz bank. Portable
                                       fallback for a build machine that has
                                       neither the dataset nor h5py.

    Both go through modelblaster/mb_datasets/nyu_depth.py, which applies the
    same crop/resize/normalise to either, so the two sources are
    interchangeable rather than two subtly different calibration sets.

    NOTE ON SIZE. More calibration frames is NOT automatically better here,
    and which way it goes depends on the range rule. Activation scales are
    per-tensor MAX-ABS, so an extra frame can only WIDEN a range, never
    tighten one, and a wider range is a coarser int8 step. Measured on the
    trained epoch-0 FastDepth over all 654 NYU val frames
    (experiments/fastdepth_int8):

      plain max-abs ranges      1 frame  d1 0.7221          32 frames  0.7118
      clamp-aware ranges        1 frame  d1 0.7238 / rmse 0.6663
                                8 frames    0.7251 / 0.6514
                               32 frames    0.7228 / 0.6511

    i.e. under the stock rule 32x the data COSTS 0.010 d1, while with
    clamp-aware ranges (which cap each range at what the consumer's ReLU/ReLU6
    can pass) the extra frames buy a real 0.015 m of RMSE and then stop. 8 is
    the measured sweet spot; the default is small deliberately.
    """
    import os  # noqa: PLC0415
    _input_size, _wm, _dc = _cfg()
    nyu_dir = os.environ.get("MODELBLASTER_FASTDEPTH_NYU_DIR", "")
    calib = os.environ.get("MODELBLASTER_FASTDEPTH_CALIB", "")
    if nyu_dir:
        src = {"path": nyu_dir,
               "image_size": [_input_size, _input_size],
               "normalize": "imagenet", "spread": True}
    elif calib:
        # The bank is already cropped, resized and ImageNet-normalised, so it
        # declares no image_size/normalize -- restating them would invite the
        # two sources to drift apart.
        src = {"npz": calib}
    else:
        return None
    return {
        "num_samples": num_samples,
        "inputs": {
            "x": dict(src, loader="nyu_depth", n_take=num_samples,
                      compose={"kind": "one_per_sample"}),
        },
    }


def get_calibration_samples(n: int = 8):
    """Real NYU frames for int8 activation calibration.

    Back-compat wrapper around get_calibration_spec (same shape as
    yolov8_nano's), so the two entry points cannot disagree about what the
    calibration set is. Falls back to get_sample_input when nothing is
    configured, leaving the untrained benchmark path unchanged.
    """
    spec = get_calibration_spec(n)
    if spec is None:
        return [get_sample_input()]
    from modelblaster.mb_datasets.base import materialize_calibration_samples  # noqa: PLC0415
    return [d["x"] for d in materialize_calibration_samples(spec)]


def get_sample_input(seed: int = 1) -> torch.Tensor:
    input_size, _wm, _dc = _cfg()
    spec = get_calibration_spec(1)
    if spec is not None:
        # A trained model's golden should be a real frame: randn would pin the
        # io.npz anchor to an input the network was never trained on. Routed
        # through the SAME spec the calibration set comes from, so the golden
        # anchor is calibration sample 0 whichever source is configured.
        from modelblaster.mb_datasets.base import materialize_calibration_samples  # noqa: PLC0415
        return materialize_calibration_samples(spec)[0]["x"]
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 3, input_size, input_size, generator=g)
