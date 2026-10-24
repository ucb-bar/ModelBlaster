"""Bind an IR op to its kernel's C signature by parameter name.

`KernelSpec.signature` is the kernel ABI for every backend. Most ops have a
hand-written call in generate_skeleton's per-op chain, but the binding is
mechanical, so ops without one (the generic ops in reference_kernels_ext,
and any CUDA walker) derive the call from (signature, op):

  * non-const pointer      -> the op's next output tensor
  * const pointer named weight/scale/bias/gamma/beta, when the op has the
    matching weight field -> that weights.npz key (None = NULL)
  * any other const pointer -> the op's next input tensor; `in<k>` params
    past the last input bind NULL (variadic kernels such as catn)
  * scalar                 -> op["shape"][name], then op[name] (or its
    alias, e.g. `s` <- op["scalar"]), then shape["C_inputs"][k] for c<k>,
    then a few ABI defaults (DH/DW = 1), then per-op defaults
    (_OP_DEFAULTS); `Ca`/`Cb` are C_inputs[0]/[1] like `c0`/`c1`
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

WEIGHT_ROLES = {"weight": "weight", "scale": "weight", "bias": "bias",
                "gamma": "gamma_key", "beta": "beta_key"}
_DEFAULTS = {"DH": 1, "DW": 1, "alpha": 1.0}
# hand-written `_f16` specs with extra parameters (the values generate_skeleton
# passes): softmax_f16's fused input scale, and the output size of
# adaptive_avg_pool2d_f16 -- MB's adaptive_avg_pool2d is the global (1x1) pool
_OP_DEFAULTS = {"softmax_f16": {"input_scale": 1.0},
                "adaptive_avg_pool2d_f16": {"OH": 1, "OW": 1}}
_C_INPUT_ALIASES = {"Ca": 0, "Cb": 1}
# signature param name -> op field holding its value, where they differ
_FIELD_ALIASES = {"s": "scalar"}


@dataclass
class Param:
    ctype: str
    name: str

    @property
    def is_ptr(self) -> bool:
        return "*" in self.ctype

    @property
    def is_const_ptr(self) -> bool:
        return self.is_ptr and "const" in self.ctype

    @property
    def base_type(self) -> str:
        return self.ctype.replace("const", "").replace("*", "").strip()


@dataclass
class Arg:
    kind: str          # "out" | "in" | "weight" | "null" | "scalar"
    value: Any         # tensor name, weights key, or scalar value
    param: Param


class BindError(Exception):
    pass


def parse_signature(sig: str) -> tuple[str, list[Param]]:
    m = re.match(r"\s*void\s+(\w+)\s*\((.*)\)\s*;?\s*$", sig, re.S)
    if not m:
        raise ValueError(f"unparseable signature: {sig}")
    params = []
    for raw in m.group(2).split(","):
        raw = " ".join(raw.split())
        if not raw or raw == "void":
            continue
        pm = re.match(r"(.*?[\s\*])(\w+)$", raw)
        params.append(Param(pm.group(1).replace(" *", "*").strip(), pm.group(2)))
    return m.group(1), params


def bind(op: dict, params: list[Param]) -> list[Arg]:
    ins = list(op.get("inputs", []))
    outs = list(op.get("outputs", []))
    shape = op.get("shape", {}) or {}
    args: list[Arg] = []
    for p in params:
        if p.is_ptr and not p.is_const_ptr:
            if not outs:
                raise BindError(f"{op['op']}: no output left for {p.name}")
            args.append(Arg("out", outs.pop(0), p))
        elif p.is_ptr:
            role = WEIGHT_ROLES.get(p.name)
            if role is not None and role in op:
                key = op.get(role)
                args.append(Arg("weight", key, p) if key else Arg("null", None, p))
            elif ins:
                args.append(Arg("in", ins.pop(0), p))
            elif re.fullmatch(r"in\d+", p.name):
                args.append(Arg("null", None, p))
            else:
                raise BindError(f"{op['op']}: no input left for {p.name}")
        else:
            if p.name in shape:
                v = shape[p.name]
            elif p.name in op:
                v = op[p.name]
            elif _FIELD_ALIASES.get(p.name) in op:
                v = op[_FIELD_ALIASES[p.name]]
            elif re.fullmatch(r"c\d", p.name) and "C_inputs" in shape:
                v = shape["C_inputs"][int(p.name[1])]
            elif p.name in _C_INPUT_ALIASES and "C_inputs" in shape:
                v = shape["C_inputs"][_C_INPUT_ALIASES[p.name]]
            elif p.name in _DEFAULTS:
                v = _DEFAULTS[p.name]
            elif p.name in _OP_DEFAULTS.get(op["op"], {}):
                v = _OP_DEFAULTS[op["op"]][p.name]
            else:
                raise BindError(f"{op['op']}: no value for scalar {p.ctype} {p.name}")
            args.append(Arg("scalar", v, p))
    if ins or outs:
        raise BindError(f"{op['op']}: unbound tensors ins={ins} outs={outs}")
    return args
