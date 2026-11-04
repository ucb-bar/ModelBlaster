"""Synthetic-tensor loader: the inputs a calibration spec has no data for.

A calibration spec has to cover EVERY forward input, because
`materialize_calibration_samples` builds each sample's positional tuple from
the spec's keys -- a spec covering only the cameras produces a 2-tuple and the
model call fails. Some inputs genuinely have no dataset behind them (Octo's
language embedding is the output of a T5 encoder that is not part of the port;
its diffusion timestep is a loop index), so this loader states that in the
spec instead of leaving it implicit in Python.

Spec fields:
    loader: "synthetic"                   (required)
    shape:  [d0, d1, ...]                 (required; NO batch dim -- the
                                           composer adds it)
    kind:   "gaussian" | "uniform" | "const"   (optional, default "gaussian")
    scale:  float                         (optional, default 1.0; stddev for
                                           gaussian, half-range for uniform,
                                           the value for const)
    n_items: int                          (optional, default 64)
    seed:   int                           (optional, default 0)

Anything drawn here is noise, so it is honest for tensors whose real
distribution is unknown and WRONG for anything whose range the model's
accuracy depends on. Prefer a real loader wherever one can be written.
"""

from __future__ import annotations

import torch

from modelblaster.mb_datasets.base import DatasetItem, register_loader


def load(spec: dict) -> list[DatasetItem]:
    shape = tuple(int(d) for d in spec["shape"])
    kind = str(spec.get("kind", "gaussian"))
    scale = float(spec.get("scale", 1.0))
    n = int(spec.get("n_items", 64))
    g = torch.Generator().manual_seed(int(spec.get("seed", 0)))
    items = []
    for i in range(n):
        if kind == "gaussian":
            t = torch.randn(shape, generator=g) * scale
        elif kind == "uniform":
            t = (torch.rand(shape, generator=g) * 2.0 - 1.0) * scale
        elif kind == "const":
            t = torch.full(shape, scale)
        else:
            raise ValueError(f"synthetic: kind={kind!r} must be one of "
                             f"gaussian / uniform / const")
        items.append(DatasetItem(data=t, meta={"loader": "synthetic",
                                               "kind": kind, "index": i}))
    return items


register_loader("synthetic", load)
