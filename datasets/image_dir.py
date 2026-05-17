"""Generic image-directory loader.

Spec fields:
    loader:         "image_dir"           (required)
    path:           "<path/to/dir>"       (required; can be absolute or
                                           repo-relative)
    image_size:     [W, H]                (required; output size in px)
    pattern:        "*.jpg"               (optional, default "*.jpg" then
                                           falls back to "*.png" if no jpgs)
    normalize:      "imagenet" | "none"   (optional, default "imagenet")
    n_take:         int                   (optional, default all)
    sort:           "name" | "none"       (optional, default "name")

Example (for the IDSIA forest-trail straight-corridor samples):

    {"loader": "image_dir",
     "path":   "datasets/idsia/samples/sc",
     "image_size": [85, 64],
     "normalize":  "imagenet"}
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from modelblaster.datasets.base import DatasetItem, register_loader


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    # Try repo-root-relative first, then current working dir.
    for root in (_REPO_ROOT, _REPO_ROOT.parent, Path.cwd()):
        cand = root / p
        if cand.exists():
            return cand
    return pp  # let downstream raise if it doesn't exist


def _make_transform(image_size, normalize: str):
    from torchvision import transforms  # noqa: PLC0415
    W, H = image_size
    layers = [
        transforms.Resize((H, W)),  # torchvision Resize is (H, W)
        transforms.ToTensor(),
    ]
    if normalize == "imagenet":
        layers.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]))
    elif normalize != "none":
        raise ValueError(f"normalize={normalize!r} not in {{imagenet, none}}")
    return transforms.Compose(layers)


def load(spec: dict) -> list[DatasetItem]:
    from PIL import Image  # noqa: PLC0415
    path = _resolve_path(spec["path"])
    if not path.is_dir():
        raise FileNotFoundError(
            f"image_dir: {path} does not exist or is not a directory")
    image_size = spec["image_size"]
    pattern = spec.get("pattern", "*.jpg")
    n_take = spec.get("n_take")
    sort_kind = spec.get("sort", "name")

    candidates = list(path.glob(pattern))
    if not candidates and pattern == "*.jpg":
        candidates = list(path.glob("*.png"))
    if sort_kind == "name":
        candidates.sort()
    if n_take is not None:
        candidates = candidates[: int(n_take)]
    if not candidates:
        raise FileNotFoundError(
            f"image_dir: no files matching {pattern} in {path}")
    tfm = _make_transform(image_size, spec.get("normalize", "imagenet"))
    out: list[DatasetItem] = []
    for fp in candidates:
        img = Image.open(fp).convert("RGB")
        tensor = tfm(img)
        out.append(DatasetItem(data=tensor, meta={"source": str(fp)}))
    return out


register_loader("image_dir", load)
