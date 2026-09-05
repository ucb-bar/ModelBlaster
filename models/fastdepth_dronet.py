"""FastDepth -> DroNet: a genuinely data-dependent perception pipeline.

RGB in, dense depth in the middle, a navigation command out. DroNet does not
run alongside FastDepth here -- it consumes FastDepth's depth map, so every
DroNet dispatch is transitively dependent on every FastDepth dispatch. That is
the point: the scheduler sees ONE graph with a real dependency chain across two
backends, rather than two independent networks it may interleave freely.

Why fuse rather than express it as two networks: the workload JSON schema has
no cross-network dependency edge -- a network entry carries only
dispatch_deps_path, id, identifier and num_instances, all intra-network. So a
producer/consumer relationship between two networks is not representable to the
scheduler, and the honest way to schedule one is to hand it a single graph.
models/fused_full.py and fused_depth.py set that precedent.

The coupling is 1-channel: FastDepth emits [1,1,H,W] and DroNet is constructed
with img_channels=1 at the same H,W, so the depth map is consumed directly with
no replication or rescaling in between. Anything else would put a reshape on
the dependency edge and blur what is being measured.

  MODELBLASTER_FASTDEPTH_DRONET_INPUT   default 128, must be a multiple of 32
                                        (FastDepth downsamples/upsamples 5x).

fp32 only for now: FastDepth cannot take the int8 route because
extract_graph.py refuses grouped convolution, which is the defining op of its
MobileNet encoder.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn


def _cfg() -> int:
    n = int(os.environ.get("MODELBLASTER_FASTDEPTH_DRONET_INPUT", 128))
    if n % 32:
        raise ValueError(f"input {n} must be a multiple of 32")
    return n


class FastDepthDroNet(nn.Module):
    def __init__(self, size: int = 128):
        super().__init__()
        from models.fastdepth import FastDepth
        from models.dronet_arch import DronetTorch
        self.depth = FastDepth()
        # img_channels=1: DroNet reads the depth map itself. DroNet probes its
        # own trunk for the linear width, so any multiple-of-32 size works.
        self.nav = DronetTorch(img_dims=(size, size), img_channels=1, output_dim=2)

    def forward(self, x: torch.Tensor):
        d = self.depth(x)          # [1,3,H,W] -> [1,1,H,W] dense depth
        return self.nav(d)         # depth -> (steering, collision)


def get_model(seed: int = 0):
    torch.manual_seed(seed)
    m = FastDepthDroNet(size=_cfg())
    m.eval()
    return m


def get_sample_input(seed: int = 1) -> torch.Tensor:
    n = _cfg()
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 3, n, n, generator=g)
