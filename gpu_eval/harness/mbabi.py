"""ModelBlaster kernel ABI for the GPU harnesses.

Signature parsing and IR-op binding are ModelBlaster's own
(pipeline/kernel_abi.py, loaded by path so the GPU box needs only this
checkout, not an installed ModelBlaster); this adds the ctypes side.
"""
from __future__ import annotations

import ctypes
import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "kernel_abi", Path(__file__).resolve().parents[2] / "pipeline" / "kernel_abi.py")
kernel_abi = importlib.util.module_from_spec(_spec)
sys.modules["kernel_abi"] = kernel_abi  # dataclasses resolve their module by name
_spec.loader.exec_module(kernel_abi)
BindError, WEIGHT_ROLES = kernel_abi.BindError, kernel_abi.WEIGHT_ROLES  # re-exported


class Param(kernel_abi.Param):
    def ctypes_type(self):
        if self.is_ptr:
            return ctypes.c_void_p
        return {"int": ctypes.c_int, "float": ctypes.c_float, "long": ctypes.c_long,
                "int64_t": ctypes.c_int64, "int32_t": ctypes.c_int32,
                "double": ctypes.c_double}[self.base_type]


def parse_signature(sig: str) -> tuple[str, list[Param]]:
    name, params = kernel_abi.parse_signature(sig)
    return name, [Param(p.ctype, p.name) for p in params]


def bind(op: dict, params: list[Param], ptr_of_tensor, ptr_of_weight) -> list:
    """Positional ctypes-ready args for calling the kernel on `op`;
    ptr_of_tensor(name) / ptr_of_weight(key) return device addresses."""
    out = []
    for a in kernel_abi.bind(op, params):
        if a.kind in ("out", "in"):
            out.append(ptr_of_tensor(a.value))
        elif a.kind == "weight":
            out.append(ptr_of_weight(a.value))
        elif a.kind == "null":
            out.append(None)
        else:
            out.append(a.value)
    return out
