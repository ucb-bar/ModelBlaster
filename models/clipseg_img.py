"""CLIPSeg (CIDAS/clipseg-rd64-refined) image path at 352x352 as a ModelBlaster --model: the CLIP ViT-B/16
vision tower + the FiLM-conditioned decoder, given the prompt's conditional embedding (the text tower runs
once per prompt on the host). Inputs: pixel_values [1, 3, 352, 352] (CLIPSeg processor normalisation),
cond [1, 512] (CLIP text embedding of the prompt, default 'a dirt path'); output: logits [1, 352, 352].
MB_CLIPSEG_IMAGES: frames for calibration / samples."""
import glob
import os

import numpy as np
import torch
import torch.nn as nn

RID = "CIDAS/clipseg-rd64-refined"
PROMPT = os.environ.get("MB_CLIPSEG_PROMPT", "a dirt path")
IMAGES = os.environ.get("MB_CLIPSEG_IMAGES", "")
_M = {}


class Wrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values, cond):
        return self.m(pixel_values=pixel_values, conditional_embeddings=cond).logits


def _load():
    if "m" not in _M:
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
        _M["p"] = CLIPSegProcessor.from_pretrained(RID)
        _M["m"] = CLIPSegForImageSegmentation.from_pretrained(RID).eval()
        t = _M["p"](text=[PROMPT], return_tensors="pt")
        with torch.no_grad():
            _M["cond"] = _M["m"].clip.get_text_features(**t)
    return _M


def get_model():
    return Wrap(_load()["m"]).eval()


def frames(n, offset=0):
    from PIL import Image
    p = _load()["p"]
    if not IMAGES:
        raise RuntimeError("clipseg_img: set MB_CLIPSEG_IMAGES to a directory of jpg/png frames")
    fs = sorted(glob.glob(f"{IMAGES}/*.jpg") + glob.glob(f"{IMAGES}/*.png"))[offset:offset + n]
    return [p(images=[Image.open(f).convert("RGB")], return_tensors="pt")["pixel_values"] for f in fs]


def get_sample_input():
    return [frames(1, offset=15)[0], _load()["cond"]]


def get_calibration_samples(n):
    return [[f, _load()["cond"]] for f in frames(n, offset=0)]


def get_int8_mixed_policy():
    return {"base": "linear", "alpha": 0.5, "keep_fp32": [], "conv_as_linear": True}
