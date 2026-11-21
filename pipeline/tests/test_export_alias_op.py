"""extract_graph_export classifies aten.alias.default (emitted by torch >= 2.15's export,
k3-work TASK-019) as a zero-compute pass-through, like clone.default."""
from modelblaster.pipeline import extract_graph_export as ege


def test_alias_default_is_a_pass_through():
    assert ege._classify("alias.default") == "alias"
    assert ege._classify("clone.default") == "alias"


def test_bias_safe_scales_only_raise_overflowing_channels():
    # A per-channel weight scale is raised only where the int32 bias would overflow
    # (a near-zero filter with a bias); every other channel keeps its scale.
    import numpy as np
    import torch
    w_scales = np.array([0.01, 1e-8 / 127.0, 0.02])
    b = torch.tensor([3.0, 5.0, -2.0])
    out = ege._bias_safe_scales(w_scales, b, in_scale=0.05)
    assert out[0] == w_scales[0] and out[2] == w_scales[2]
    assert abs(5.0 / (0.05 * out[1])) < 2**31 - 1
    assert ege._bias_safe_scales(w_scales, None, 0.05) is w_scales
