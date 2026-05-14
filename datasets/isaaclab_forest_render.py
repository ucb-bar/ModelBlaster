"""IsaacLab forest-trail goal-image renders.

Same loader semantics as image_dir, with a stricter contract about
provenance: the images are produced by
``sims/scripts/utils/render_vint_calibration.py`` which spawns
IsaacLab's forest-trail env, teleports the drone to varied trail
positions, and captures the FPV camera output as PNG. The resulting
distribution matches what ViNT's goal_encoder will actually see at
deployment time (synthetic 3-D renders with the env's lighting +
textures), not generic 2-D outdoor photographs like IDSIA.

Spec fields:
    loader:         "isaaclab_forest_render"
    path:           dir containing the *.png renders (default:
                    "datasets/isaaclab_forest_renders")
    image_size:     [W, H]                (required)
    n_take:         optional int

Run the renderer script first to populate the dir:
    bash sims/scripts/utils/render_vint_calibration.sh \\
        --out-dir datasets/isaaclab_forest_renders --n-samples 32

If the dir doesn't exist this loader raises with a clear message
pointing at the renderer — no silent fallback to noise.
"""

from __future__ import annotations

from pathlib import Path

from agents.datasets.base import DatasetItem, register_loader
from agents.datasets.image_dir import _resolve_path, _make_transform


def load(spec: dict) -> list[DatasetItem]:
    from PIL import Image  # noqa: PLC0415
    path = _resolve_path(spec.get(
        "path", "datasets/isaaclab_forest_renders"))
    if not path.is_dir():
        raise FileNotFoundError(
            f"isaaclab_forest_render: {path} does not exist. Generate "
            f"renders first via:\n"
            f"    bash sims/scripts/utils/render_vint_calibration.sh "
            f"--out-dir {path}\n"
            f"or override the path in the calibration spec.")
    images = sorted(path.glob("*.png")) + sorted(path.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(
            f"isaaclab_forest_render: no images in {path}")
    n_take = spec.get("n_take")
    if n_take is not None:
        images = images[: int(n_take)]
    tfm = _make_transform(spec["image_size"], spec.get("normalize", "imagenet"))
    out: list[DatasetItem] = []
    for fp in images:
        img = Image.open(fp).convert("RGB")
        out.append(DatasetItem(
            data=tfm(img),
            meta={"source": str(fp), "kind": "isaaclab_render"}))
    return out


register_loader("isaaclab_forest_render", load)
