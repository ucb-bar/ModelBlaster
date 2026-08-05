"""ModelBlaster wrapper: FULL FusedSensorNet (CNN variant) — best-effort probe.

This exists only to DOCUMENT where ModelBlaster's int8 lowering breaks on the
full net (LSTM / heterogeneous fusion). It wraps the full net behind a single
image tensor input (depth + state slots are zero-filled internally), so the FX
trace reaches the nn.LSTM and head, isolating the LSTM as the failure point
rather than the dict-input tracing issue.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from modelblaster.models._fused_loader import load_fused_cnn

_H, _W = 60, 90


class _FullSingleInput(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, front_grey: torch.Tensor) -> torch.Tensor:
        # Inline the fused forward (batch=1) to bypass the collaborator's
        # data-dependent shape guard in _vision, so the FX trace reaches the
        # nn.LSTM / head fusion — the part we want to document.
        net = self.net
        vision_feat = net.vision_fc(torch.flatten(net.vision_cnn(front_grey), 1))
        depth_feat = net.depth_fc(
            torch.flatten(net.depth_conv(torch.zeros(1, 4, 8, 8)), 1)
        )
        state = torch.zeros(1, 15 + 6)  # STATE_DIM + N_STATE_GROUPS
        fused = torch.cat([vision_feat, depth_feat, state], dim=1).unsqueeze(0)
        out, _ = net.lstm(fused)
        return net.head(out.squeeze(0))


def get_model(seed: int = 0):
    net = load_fused_cnn(seed)
    m = _FullSingleInput(net)
    m.eval()
    return m


def get_sample_input(seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 1, _H, _W, generator=g)
