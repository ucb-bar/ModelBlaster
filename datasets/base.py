"""Dataset protocol + calibration-spec resolver for the modelblaster pipeline.

Goals:
 1. Calibration data is a *declared spec*, not hidden Python code per model.
    The spec serializes to JSON alongside the IR artifacts so a model
    extracted in one session is reproducible by anyone with the same
    spec + the same source data.
 2. Multiple datasets can compose into one calibration sample (e.g.
    ViNT's obs_img comes from a rolling window of IDSIA images, while
    goal_img comes from IsaacLab forest renders).
 3. Adding a new data source is one file: drop a loader module under
    ``modelblaster/datasets/<name>.py`` exporting ``load(spec) -> list[DatasetItem]``.

Spec schema (saved to ``<example>/<quant>/generated/calibration_spec.json``):

    {
      "num_samples": 16,
      "inputs": {
        "<input_tensor_name>": {
          "loader": "<module under modelblaster.datasets>",
          # rest is loader-specific
          "image_size": [85, 64],
          ...
          # optional sample-composition wrapper:
          "compose": {
            "kind": "rolling_window",
            "frames_per_sample": 6,
            ...
          }
        },
        ...
      }
    }

The walker calls ``materialize_calibration_samples(spec)`` and gets back
a list of dicts ``[{input_name: torch.Tensor(1, C, H, W), ...}, ...]``
suitable for feeding directly into the model's forward.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable

import torch


@dataclass
class DatasetItem:
    """One image (or other tensor) from a dataset, with provenance.

    ``data`` is the preprocessed tensor in the model's input dtype
    (typically float32, NCHW). ``meta`` carries source info for
    debugging / reproducibility reports — never the raw bytes.
    """
    data: torch.Tensor
    meta: dict = field(default_factory=dict)


# Each registered loader implements: load(spec: dict) -> list[DatasetItem]
_LOADERS: dict[str, Callable[[dict], list[DatasetItem]]] = {}


def register_loader(name: str, fn: Callable[[dict], list[DatasetItem]]) -> None:
    """Register a loader callable under ``name``. Called from each
    loader module's import-time top-level."""
    _LOADERS[name] = fn


def load_dataset(spec: dict) -> list[DatasetItem]:
    """Dispatch on spec["loader"] to load DatasetItems.

    Loader modules are auto-imported on first reference; they register
    themselves under their canonical name on import. So
    ``{"loader": "image_dir", ...}`` triggers import of
    ``modelblaster.datasets.image_dir``.
    """
    name = spec["loader"]
    if name not in _LOADERS:
        # Lazy import: each loader module registers itself at top level.
        importlib.import_module(f"modelblaster.datasets.{name}")
    if name not in _LOADERS:
        raise KeyError(
            f"loader {name!r} not registered; check that "
            f"modelblaster/datasets/{name}.py calls register_loader at top level")
    return _LOADERS[name](spec)


def _compose_rolling_window(items: list[DatasetItem], spec: dict,
                            n_samples: int) -> list[torch.Tensor]:
    """Build N rolling-window samples by stacking K consecutive items
    along the channel dim. Used for ViNT-style obs_img (6 context
    frames stacked into a single (1, 18, H, W) tensor)."""
    k = int(spec["frames_per_sample"])
    if not items:
        raise ValueError("rolling_window: empty item list")
    out = []
    for i in range(n_samples):
        anchor = (i * max(1, len(items) // n_samples)) % len(items)
        frames = []
        for j in range(k):
            idx = max(0, anchor - (k - 1 - j)) % len(items)
            frames.append(items[idx].data)
        out.append(torch.cat(frames, dim=0).unsqueeze(0))  # (1, k*C, H, W)
    return out


def _compose_one_per_sample(items: list[DatasetItem], spec: dict,
                            n_samples: int) -> list[torch.Tensor]:
    """One item per sample; cycles with stride if fewer items than
    samples. Used for the goal image (one per inference)."""
    if not items:
        raise ValueError("one_per_sample: empty item list")
    stride = max(1, len(items) // n_samples)
    return [items[(i * stride) % len(items)].data.unsqueeze(0)
            for i in range(n_samples)]


_COMPOSERS = {
    "rolling_window": _compose_rolling_window,
    "one_per_sample": _compose_one_per_sample,
}


def materialize_calibration_samples(
        spec: dict) -> list[dict[str, torch.Tensor]]:
    """Top-level entry: resolve the spec into N dicts of
    {input_tensor_name: shaped torch.Tensor}.

    The N samples are aligned across inputs by ordering — i.e.
    samples[i] has each input's i-th composed tensor. The composer's
    cycling logic handles inputs with fewer items than samples.
    """
    n = int(spec.get("num_samples", 16))
    composed: dict[str, list[torch.Tensor]] = {}
    for input_name, input_spec in spec["inputs"].items():
        # Strip compose to pass the rest to the loader.
        compose_spec = dict(input_spec.get("compose", {"kind": "one_per_sample"}))
        loader_spec = {k: v for k, v in input_spec.items() if k != "compose"}
        items = load_dataset(loader_spec)
        kind = compose_spec.pop("kind", "one_per_sample")
        composer = _COMPOSERS.get(kind)
        if composer is None:
            raise KeyError(f"unknown composer {kind!r}; "
                           f"options: {list(_COMPOSERS)}")
        composed[input_name] = composer(items, compose_spec, n)

    # Sanity: every input list has n entries.
    for name, lst in composed.items():
        if len(lst) != n:
            raise RuntimeError(
                f"composed input {name!r} produced {len(lst)} samples, "
                f"expected {n}")
    return [
        {name: composed[name][i] for name in composed}
        for i in range(n)
    ]
