"""BridgeData episode-pickle loader.

Octo's activation scales cannot be fitted to `torch.randint(0, 256)`: the
stems apply weight standardisation and GroupNorm, whose output ranges are a
property of real scene statistics, and a scale fitted to noise verifies
bit-exact against its own golden while being the wrong scale (see
experiments/octo_port/NOTES.md on why this spec exists).

Source format -- a pickle holding a list of episodes, each a dict:

    {"images":  ndarray (T, H, W, 3) uint8,
     "instr":   str,
     "actions": ndarray (T, action_dim) float32}

which is what the upstream Octo benchmarks used
(`/scratch2/dima/misc_sw/octo_work/bridge_episodes.pkl`).

Spec fields:
    loader:      "bridge_episodes"        (required)
    path:        "<path/to/pkl>"          (required)
    key:         "image_primary" | "image_wrist" | "images"
                                          (optional, default "image_primary")
    image_size:  [W, H]                   (required; output size in px)
    domain:      "unit" | "raw"           (optional, default "unit")
                 "unit" = [-1, 1], matching normalize_images(x) = x/127.5 - 1,
                 which is the input contract when the model's own
                 normalisation is moved out of the graph
                 (MODELBLASTER_OCTO_NORM=0). "raw" leaves [0, 255].
    n_take:      int                      (optional, default all frames)

`key` is matched loosely: an episode dict that only has "images" serves any
camera key, because these episodes are single-camera. A wrist request against
a single-camera episode therefore reuses the primary frames rather than
failing -- calibrating the wrist stem on primary frames is closer to right
than calibrating it on noise, and it is reported in the item meta.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from modelblaster.mb_datasets.base import DatasetItem, register_loader

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    for root in (_REPO_ROOT, _REPO_ROOT.parent, Path.cwd()):
        cand = root / p
        if cand.exists():
            return cand
    return pp


def _frames_for(ep: dict, key: str):
    """The (T, H, W, 3) uint8 stack this key should use, or None."""
    for k in (key, key.replace("image_", ""), "images"):
        v = ep.get(k)
        if isinstance(v, np.ndarray) and v.ndim == 4:
            return v, k
    return None, None


def load(spec: dict) -> list[DatasetItem]:
    path = _resolve_path(str(spec["path"]))
    if not path.exists():
        raise FileNotFoundError(f"bridge_episodes: no such file: {path}")
    key = str(spec.get("key", "image_primary"))
    W, H = (int(x) for x in spec["image_size"])
    domain = str(spec.get("domain", "unit"))
    if domain not in ("unit", "raw"):
        raise ValueError(f"bridge_episodes: domain={domain!r} must be "
                         f"'unit' or 'raw'")
    with open(path, "rb") as fh:
        episodes = pickle.load(fh)
    if isinstance(episodes, dict):
        episodes = [episodes]

    items: list[DatasetItem] = []
    for ep_i, ep in enumerate(episodes):
        if not isinstance(ep, dict):
            continue
        frames, used_key = _frames_for(ep, key)
        if frames is None:
            continue
        for t in range(frames.shape[0]):
            # uint8 HWC -> float CHW, resized. Bilinear, antialiased: these
            # are 256x256 source frames and a stem input may be smaller
            # (the wrist camera is 128), so a plain stride-subsample would
            # alias and change exactly the high-frequency content the
            # stem's first stride-2 conv responds to.
            a = torch.from_numpy(
                np.ascontiguousarray(frames[t])).permute(2, 0, 1).float()
            if a.shape[-2:] != (H, W):
                a = torch.nn.functional.interpolate(
                    a.unsqueeze(0), size=(H, W), mode="bilinear",
                    align_corners=False, antialias=True).squeeze(0)
            if domain == "unit":
                a = a / 127.5 - 1.0
            items.append(DatasetItem(
                data=a,
                meta={"source": str(path), "episode": ep_i, "frame": t,
                      "key": used_key, "requested_key": key,
                      "instr": ep.get("instr", ""), "domain": domain},
            ))
    if not items:
        raise ValueError(
            f"bridge_episodes: {path} yielded no frames for key {key!r}; "
            f"episodes carry {[sorted(e) for e in episodes[:1]]}")
    n_take = int(spec.get("n_take", 0))
    if n_take > 0:
        items = items[:n_take]
    return items


register_loader("bridge_episodes", load)
