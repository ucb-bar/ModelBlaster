"""ModelBlaster wrapper: FULL FusedSensorNet (CNN v12) — multi-input, end-to-end.

Trace-clean, multi-input lowering of the whole deployable net: the vision CNN,
the ToF-cross depth conv, the [vision|depth|lowdim] fuse, the 3-layer LSTM, and
the output head — no zero-filled slots. Uses the multi-input pipeline (3 typed
inputs) + the int8 lstm_s8 kernel.

Trace-clean choices (FX symbolic_trace can't handle the collaborator's forward):
  * inputs are 3 separate tensors, not a dict (no dict iteration).
  * no `if img.shape[-2:] != (60,90)` guard — the caller feeds exactly 60x90.
  * torch.flatten(x, 1), not x.flatten(1) (the latter traces as a call_method
    the int8 extractor rejects).
  * the fused (1, 597) vector is fed to nn.LSTM UNBATCHED (seq=1) so its output
    is (1, 128) directly — no unsqueeze/squeeze around the LSTM.

`lowdim` (1, 21) is the state groups (flow2, down_tof1, baro2, quat4, body_rates3,
desired_vel3 = 15) concatenated with the 6 per-group validity flags, matching the
collaborator's `state = cat(parts + flags)`. The host builds it; on the SoC it is
one contiguous sensor vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from modelblaster.models._fused_loader import load_fused_cnn

_H, _W = 60, 90
_LOWDIM = 15 + 6   # STATE_DIM (flow2+tof1+baro2+quat4+rates3+desired3) + 6 flags


class FusedFull(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.vision_cnn = net.vision_cnn
        self.vision_fc = net.vision_fc
        self.depth_conv = net.depth_conv
        self.depth_fc = net.depth_fc
        self.lstm = net.lstm
        self.head = net.head

    def forward(self, front_grey, tof_cross, lowdim):
        v = self.vision_fc(torch.flatten(self.vision_cnn(front_grey), 1))  # (1,512)
        d = self.depth_fc(torch.flatten(self.depth_conv(tof_cross), 1))    # (1,64)
        fused = torch.cat([v, d, lowdim], dim=1)   # (1, 597)
        out, _ = self.lstm(fused)                  # (1, 128) unbatched (seq=1)
        return self.head(out)                      # (1, out_dim)


def get_model(seed: int = 0):
    net = load_fused_cnn(seed)
    m = FusedFull(net)
    m.eval()
    return m


def _sample(seed: int):
    g = torch.Generator().manual_seed(seed)
    # front_grey, tof_cross in [0,1] like the trained normalization; lowdim is a
    # modest-range vector (flags in [0,1], the rest ~unit) — representative, not
    # a specific logged frame.
    front = torch.rand(1, 1, _H, _W, generator=g)
    tof = torch.rand(1, 4, 8, 8, generator=g)
    low = 0.5 * torch.randn(1, _LOWDIM, generator=g)
    low[:, -6:] = 1.0   # all sensor groups valid
    return (front, tof, low)


def get_sample_input(seed: int = 1):
    return _sample(seed)


def get_input_dtypes():
    # front_grey, tof_cross, lowdim — all int8-quantized (each its own scale).
    return ["i8", "i8", "i8"]


def get_calibration_samples(n: int):
    # Widen activation-scale calibration across a few representative inputs.
    return [_sample(100 + i) for i in range(n)]


def get_precision_spec() -> dict:
    """Recommended HYBRID precision map (see docs / the modelblaster memo): keep
    the int8 encoders (vision CNN + depth conv + their FCs) on the int8/Gemmini
    path (95% of compute, ~9% feature error), and promote the recurrent head —
    the fuse, the 3-layer LSTM, and the output head — to fp16, since the int8
    3-layer LSTM was the entire accuracy loss (full-int8 0.45 L2 vs 0.0029 for
    this split) and it is only ~5% of compute. The extractor inserts
    cast_i8_to_f16 at the encoder->fuse boundary automatically.
    """
    import os
    tail = ["cat", "lstm_l0", "lstm_l1", "lstm_l2", "head"]
    # The int8 encoder-FC OUTPUTS (vision_fc 512, depth_fc 64) carry the ~9%
    # feature error that the LSTM amplifies; promoting these two cheap linears to
    # fp16 (conv bulk stays int8) recovers most of it. Toggle with MB_FUSED_FP16_FC=0
    # to A/B the conv-only-int8 split.
    if os.environ.get("MB_FUSED_FP16_FC", "1") != "0":
        tail = ["vision_fc", "depth_fc"] + tail
    return {"default": "int8", "fp16_ops": tail}
