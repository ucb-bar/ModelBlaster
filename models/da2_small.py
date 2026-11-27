"""Depth-Anything-V2-Metric-Outdoor-Small (HF transformers) at 280x378 as a ModelBlaster --model, the sized-down
depth model of the K3 sizing study. Input: image
[1, 3, 280, 378] (already normalised as the HF processor does); output: predicted metric depth [1, 280, 378].
MB_DA2_H / MB_DA2_W change the size; MB_DA2_IMAGES: a directory of jpg/png frames for calibration / samples."""
import glob
import os

import numpy as np
import torch
import torch.nn as nn

REPO = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
H, W = int(os.environ.get("MB_DA2_H", 280)), int(os.environ.get("MB_DA2_W", 378))
IMAGES = os.environ.get("MB_DA2_IMAGES", "")
_M = {}


class Wrap(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, img):
        return self.n(pixel_values=img).predicted_depth


def get_model():
    if "m" not in _M:
        from transformers import AutoModelForDepthEstimation
        _M["m"] = Wrap(AutoModelForDepthEstimation.from_pretrained(REPO).eval()).eval()
    return _M["m"]


def frames(n, offset=0):
    from PIL import Image
    if not IMAGES:
        raise RuntimeError("da2_small: set MB_DA2_IMAGES to a directory of jpg/png frames")
    fs = sorted(glob.glob(f"{IMAGES}/*.jpg") + glob.glob(f"{IMAGES}/*.png"))
    out = []
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    for f in fs[offset:offset + n]:
        a = np.asarray(Image.open(f).convert("RGB").resize((W, H), Image.BICUBIC), np.float32) / 255.0
        out.append(torch.from_numpy(((a - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)))
    return out


def get_sample_input():
    return frames(1, offset=19)[0]


def get_calibration_samples(n):
    return frames(n, offset=0)


def get_int8_mixed_policy():
    pol = {"base": "linear", "alpha": float(os.environ.get("MB_DA2_ALPHA", 0.5)), "keep_fp32": []}
    if os.environ.get("MB_DA2_CONVQ", "1") == "1":       # DPT head convs as int8 linears (int8_mixed.conv_as_linear)
        pol["conv_as_linear"] = True
    return pol
