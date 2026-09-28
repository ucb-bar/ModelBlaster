"""Temporal (FRAME-STACKED, N=3) DroNet wrapper for the modelblaster flow.

Sources the canonical PyTorch class from the vendored dronet_arch.py (see that
module's docstring) and loads the trained TEMPORAL checkpoint by default. The
temporal DroNet stacks N=3 consecutive HM01B0 (Himax) grayscale frames as N
input channels (conv0 depth 1 -> 3); everything downstream is the SAME
DronetTorch arch, so it stays in the int8-Gemmini envelope with zero new
operators. Trained config:

    img_dims = (112, 112)   img_channels = 3   output_dim = 1   small = True
    -> conv_modules.0 weight [32, 3, 3, 3], linear_in = 2048.

Override the checkpoint path via MODELBLASTER_DRONET_STACK3_CKPT, and the Himax
(PULP-DroNet v3) dataset root (for calibration / the golden anchor) via
MODELBLASTER_DRONET_STACK3_DATA.

The frame-stacking, partition split, grayscale load and normalization below
mirror the temporal trainer (dronet_temporal/train_stack.py) and its faithful
int8 eval (dronet_temporal/eval_stack.py) EXACTLY, so get_calibration_samples
produces tensors byte-identical in preprocessing to what the model was trained
on (deploy / eval path: no random crop or hflip augmentation).
"""

from __future__ import annotations

import csv
import glob
import os

import torch

from . import dronet_arch

# Committed alongside this module (mirrors models/dronet.py and models/gtsrb.py)
# so the checkpoint travels with the checkout.
_DEFAULT_CKPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints", "dronet_stack3", "best.pt",
)

# Fixed temporal deploy config: N=3 stacked grayscale frames at 112x112.
_N = 3
_IMG_H = 112
_IMG_W = 112
_NORM = 90.0            # label_yaw_rate is deg/s in +/-90; /90 to normalize.
_HUGE = 1e12            # drop anomalous huge-int filenames (tii uint64 overflow).

# Himax PULP-DroNet v3 dataset root. Defaults to the garden copy the temporal
# checkpoint was trained/evaluated against; override with
# MODELBLASTER_DRONET_STACK3_DATA to point at any PULP-DroNet v3 layout
# (directories each containing labels_partitioned.csv + images/).
_DEFAULT_DATA = ("/scratch2/dima/misc_sw/XPU-RT/datasets/pulp_dronet_himax/"
                 "raw/Dataset_PULP_Dronet_v3")
_DATA_ROOT = os.environ.get("MODELBLASTER_DRONET_STACK3_DATA", _DEFAULT_DATA)

# Cache the leakage-free stack lists per partition so repeated calls
# (get_sample_input + get_calibration_samples) don't rescan the whole dataset.
# Caching the (paths, label) list does not change the produced tensors.
_STACKS_CACHE: dict[str, list] = {}


def get_model(seed: int = 0):
    """DronetTorch (3-channel temporal) with the trained weights loaded, in
    eval()."""
    torch.manual_seed(seed)
    m = dronet_arch.DronetTorch(
        img_dims=(_IMG_H, _IMG_W),
        img_channels=_N,
        output_dim=1,
        small=True,
    )
    ckpt_path = os.environ.get("MODELBLASTER_DRONET_STACK3_CKPT", _DEFAULT_CKPT)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Temporal DroNet checkpoint not found at {ckpt_path}. "
            f"Set MODELBLASTER_DRONET_STACK3_CKPT to override."
        )
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(
            f"unexpected keys in temporal DroNet checkpoint: {unexpected}")
    if missing:
        # linear2 (the collision head) is the only key the steering-only
        # temporal checkpoints may lack; it is at random init in the model and
        # unused by steering inference. Any OTHER missing key is a real error.
        non_linear2 = [k for k in missing if not k.startswith("linear2.")]
        if non_linear2:
            raise RuntimeError(
                f"temporal DroNet checkpoint missing weights: {non_linear2[:8]}"
            )
    m.eval()
    return m


def _load_stack(paths, size: int = _IMG_H) -> torch.Tensor:
    """Load N grayscale frames -> (N, size, size) tensor in [0, 1], channel
    order oldest->newest. Byte-identical to eval_stack.py's load_stack (the
    deploy / eval path: no augmentation)."""
    from PIL import Image, ImageFile  # noqa: PLC0415
    import torchvision.transforms.functional as TF  # noqa: PLC0415
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    chans = []
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("L").resize((size, size), Image.BILINEAR)
        chans.append(TF.to_tensor(im))          # (1, size, size)
    return torch.cat(chans, dim=0)              # (N, size, size), oldest->newest


def _build(partition: str) -> list:
    """Leakage-free N-frame stacks for one official partition. Mirrors
    train_stack.py / eval_stack.py build(): frames time-ordered by int(filename)
    within an acquisition, stack for target i = [i-(N-1) .. i] (oldest->newest),
    all N frames must share the target's partition, never across acquisition
    boundaries; drop huge-int filenames. Returns [(tuple_of_N_paths, yaw_norm)].
    """
    if partition in _STACKS_CACHE:
        return _STACKS_CACHE[partition]
    stacks = []
    csvs = sorted(glob.glob(
        os.path.join(_DATA_ROOT, "**", "labels_partitioned.csv"),
        recursive=True))
    for c in csvs:
        d = os.path.dirname(c)
        imgd = os.path.join(d, "images")
        keyed = []
        for r in csv.DictReader(open(c)):
            try:
                k = int(r["filename"].rsplit(".", 1)[0])
            except Exception:  # noqa: BLE001
                continue
            if k > _HUGE:
                continue
            try:
                y = max(-1.0, min(1.0, float(r["label_yaw_rate"]) / _NORM))
            except Exception:  # noqa: BLE001
                continue
            p = os.path.join(imgd, r["filename"])
            if not os.path.exists(p):
                continue
            keyed.append((k, r["partition"], p, y))
        keyed.sort(key=lambda t: t[0])
        parts = [t[1] for t in keyed]
        for i in range(len(keyed)):
            if parts[i] != partition or i - (_N - 1) < 0:
                continue
            win = keyed[i - (_N - 1):i + 1]
            if any(w[1] != partition for w in win):
                continue
            stacks.append((tuple(w[2] for w in win), keyed[i][3]))
    _STACKS_CACHE[partition] = stacks
    return stacks


def _train_tensors(n: int) -> list:
    """`n` real stacked TRAIN frames, each a [1, 3, 112, 112] tensor,
    preprocessed identically to the temporal trainer's eval path. Indices are
    evenly spaced (np.linspace) over the train split -- the same selection
    eval_stack.py used, so index 0 (== the golden anchor) is train stack 0."""
    import numpy as np  # noqa: PLC0415
    tr = _build("train")
    if not tr:
        raise RuntimeError(
            f"no temporal train stacks found under {_DATA_ROOT}; set "
            f"MODELBLASTER_DRONET_STACK3_DATA to a PULP-DroNet v3 root.")
    idx = np.linspace(0, len(tr) - 1, n).astype(int)
    return [_load_stack(tr[int(i)][0]).unsqueeze(0) for i in idx]


def get_sample_input(seed: int = 1) -> torch.Tensor:
    """One real stacked TRAIN frame, NCHW [1, 3, 112, 112] in [0, 1].

    Real stacked input makes a meaningful io.npz golden even at
    --num-calibration 1. Falls back to a deterministic torch.randn only if the
    Himax dataset isn't available (with a warning); with --num-calibration > 1
    the golden anchor is get_calibration_samples()[0] anyway."""
    try:
        return _train_tensors(1)[0]
    except Exception as e:  # noqa: BLE001
        print(f"[dronet_stack3.get_sample_input] WARN: couldn't load Himax "
              f"stacks from {_DATA_ROOT} ({e}); falling back to torch.randn. "
              f"Set MODELBLASTER_DRONET_STACK3_DATA to fix.")
        g = torch.Generator().manual_seed(seed)
        return torch.randn(1, _N, _IMG_H, _IMG_W, generator=g)


def get_calibration_samples(n: int = 32) -> list:
    """`n` real stacked TRAIN frames for int8 PTQ activation calibration,
    preprocessed IDENTICALLY to the temporal trainer's eval path (grayscale,
    112x112 BILINEAR, ToTensor [0,1], N=3 stacked oldest->newest).

    Each sample is a [1, 3, 112, 112] tensor; the first becomes the io.npz
    golden anchor. More samples widen the per-tensor activation ranges so the
    int8 scales reflect the true distribution (not a single frame)."""
    return _train_tensors(n)
