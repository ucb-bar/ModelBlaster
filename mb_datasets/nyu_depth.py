"""NYU Depth v2 loader — the calibration source for the depth models.

Two sources behind one spec, because the two things a calibration set has to
be are in tension:

  ``path``  a directory of the original FastDepth HDF5 frames (keys ``rgb``
            uint8 [3, H, W] and ``depth`` float32 metres). This is the
            REPRODUCIBLE source: given the dataset and the spec, anyone
            recomputes the same activation scales. Needs ``h5py``, imported
            lazily so this module still imports (and the npz source still
            works) where h5py is not installed.
  ``npz``   a pre-baked bank of already-preprocessed frames (key ``samples``,
            float32 NCHW, ImageNet-normalised). This is the PORTABLE source:
            one file, no dataset and no h5py on the build machine.

Both must produce IDENTICAL tensors for the same frames, so the preprocessing
here is the same square centre crop -> bilinear resize -> ImageNet normalise
that experiments/fastdepth_train/scripts/{make_calib,eval_fastdepth}.py apply.
That equality is the whole point of having both: the npz bank stops being an
opaque binary and becomes a cache of a declared computation.

Spec fields:
    loader:      "nyu_depth"            (required)
    path:        "<dir of *.h5>"        (one of path / npz is required)
    npz:         "<bank.npz>"
    image_size:  [W, H]                 (h5 only; default [224, 224])
    normalize:   "imagenet" | "none"    (h5 only; default "imagenet")
    n_take:      int                    (optional, default all)
    spread:      bool                   (h5 only, default True) -- take n_take
                                        frames evenly across the split rather
                                        than the first N, which would be one
                                        contiguous run from a single scene.

Example:

    {"loader": "nyu_depth",
     "path": "experiments/fastdepth_int8/data/nyu_h5",
     "image_size": [224, 224], "n_take": 32}
"""

from __future__ import annotations

from pathlib import Path

import torch

from modelblaster.mb_datasets.base import DatasetItem, register_loader
from modelblaster.mb_datasets.image_dir import _resolve_path

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _preprocess(rgb_chw_u8, image_size, normalize: str) -> torch.Tensor:
    """uint8 [3, H, W] -> float32 [3, Hout, Wout], the training preprocessing.

    Square centre crop first, THEN resize: the NYU frames are 640x480 and a
    direct resize to a square would change the aspect ratio, so the network
    would calibrate on differently-shaped scenes than it was trained on.
    """
    import numpy as np  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415
    W, H = image_size
    # Op order is copied from experiments/fastdepth_train/scripts/make_calib.py
    # deliberately, down to cropping in HWC and permuting afterwards. It is not
    # arbitrary: F.interpolate dispatches on the input's memory layout, so the
    # same crop expressed CHW-first lands on a different bilinear kernel and
    # the two disagree in the last few ulps (measured: 4e-5). Small, but it
    # would mean the "reproducible" source did not in fact reproduce the bank.
    a = np.asarray(rgb_chw_u8, dtype=np.float32).transpose(1, 2, 0) / 255.0
    h, w = a.shape[:2]
    c = min(h, w)
    y0, x0 = (h - c) // 2, (w - c) // 2
    t = torch.from_numpy(a[y0:y0 + c, x0:x0 + c]).permute(2, 0, 1)[None]
    t = F.interpolate(t, (H, W), mode="bilinear", align_corners=False)[0]
    if normalize == "imagenet":
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        t = (t - mean) / std
    elif normalize != "none":
        raise ValueError(f"normalize={normalize!r} not in {{imagenet, none}}")
    return t


def _load_npz(spec: dict) -> list[DatasetItem]:
    import numpy as np  # noqa: PLC0415
    path = _resolve_path(spec["npz"])
    if not path.is_file():
        raise FileNotFoundError(f"nyu_depth: npz bank {path} does not exist")
    arr = np.load(path)["samples"]
    n_take = spec.get("n_take")
    if n_take is not None:
        arr = arr[: int(n_take)]
    return [DatasetItem(data=torch.from_numpy(arr[i]).float(),
                        meta={"source": f"{path}#{i}"})
            for i in range(arr.shape[0])]


def _load_h5(spec: dict) -> list[DatasetItem]:
    import h5py  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    path = _resolve_path(spec["path"])
    if not path.is_dir():
        raise FileNotFoundError(
            f"nyu_depth: {path} does not exist or is not a directory")
    files = sorted(path.glob("**/*.h5"))
    if not files:
        raise FileNotFoundError(f"nyu_depth: no *.h5 under {path}")
    n_take = spec.get("n_take")
    if n_take is not None and int(n_take) < len(files):
        n = int(n_take)
        if spec.get("spread", True):
            idx = np.linspace(0, len(files) - 1, n).astype(int)
        else:
            idx = np.arange(n)
        files = [files[i] for i in idx]
    image_size = spec.get("image_size", [224, 224])
    normalize = spec.get("normalize", "imagenet")
    out: list[DatasetItem] = []
    for fp in files:
        with h5py.File(fp, "r") as f:
            rgb = np.asarray(f["rgb"])          # uint8 [3, H, W]
        out.append(DatasetItem(data=_preprocess(rgb, image_size, normalize),
                               meta={"source": str(fp)}))
    return out


def load(spec: dict) -> list[DatasetItem]:
    if spec.get("path") and spec.get("npz"):
        raise ValueError("nyu_depth: pass exactly one of path / npz, not both")
    if spec.get("npz"):
        return _load_npz(spec)
    if spec.get("path"):
        return _load_h5(spec)
    raise ValueError("nyu_depth: spec needs either 'path' (*.h5 dir) or 'npz'")


register_loader("nyu_depth", load)
