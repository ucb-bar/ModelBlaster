"""models/vint.py's precision spec: MODELBLASTER_VINT_ALL_INT8 (k3-work TASK-019).

The default (mixed: goal encoder and output head in fp16) must not change; the switch
gives an all-int8 spec. No ViNT checkout or weights are needed: only the spec is read.
"""
from __future__ import annotations

import pytest

vint = pytest.importorskip("modelblaster.models.vint")

# The spec exactly as it was before the switch existed (k3-rd d230765).
_DEFAULT = {
    "default": "int8",
    "fp16_upstream_of": ["linear"],
    "fp16_ops": ["linear_19", "linear_20", "linear_21", "linear_22",
                 "relu", "relu_1", "relu_2", "relu_3", "relu_4"],
}


@pytest.mark.parametrize("value", [None, "", "0"])
def test_default_spec_unchanged(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("MODELBLASTER_VINT_ALL_INT8", raising=False)
    else:
        monkeypatch.setenv("MODELBLASTER_VINT_ALL_INT8", value)
    assert vint.get_precision_spec() == _DEFAULT


def test_all_int8_switch(monkeypatch):
    monkeypatch.setenv("MODELBLASTER_VINT_ALL_INT8", "1")
    assert vint.get_precision_spec() == {"default": "int8", "fp16_upstream_of": [], "fp16_ops": []}
