"""Activation-layout resolution, shared by the kernel picker and the codegen.

Stage 3 of docs/IR_TENSOR_LAYOUT_DESIGN.md. Two consumers have to agree, exactly,
on one question -- "does backend B have a NATIVE kernel for op O in the layout
this tensor declares?" -- and they run in different processes:

  * `generate_kernels` picks which curated C ends up in kernels.c, and must not
    pick an NCHW kernel for an NHWC tensor;
  * `generate_skeleton.emit_model` emits the call, and must wrap it in a
    relayout shim exactly when the pick was NOT native.

If those two disagree the build still links and still runs -- NCHW and NHWC are
size-identical (design doc section 2.3c) -- and returns a plausible wrong answer.
So the predicate lives here, once, and both import it.

The layout of a TENSOR is `ir["tensors"][t]["layout"]`, absent meaning "nchw".
Absent must keep meaning exactly what it means today: a graph with no layout key
anywhere produces byte-identical output to before any of this existed.
"""
from __future__ import annotations

from typing import Any, Optional

NCHW = "nchw"
NHWC = "nhwc"

#: The two relayout ops. They are the only ops that legitimately touch two
#: layouts at once, so the contract gate checks them per ROLE (input vs output)
#: rather than exempting them.
RELAYOUT_OPS = {
    "nchw_to_nhwc_s8": (NCHW, NHWC),   # (input layout, output layout)
    "nhwc_to_nchw_s8": (NHWC, NCHW),
}

#: The relayout op that converts FROM a layout TO nchw / TO nhwc.
RELAYOUT_TO = {NCHW: "nhwc_to_nchw_s8", NHWC: "nchw_to_nhwc_s8"}

#: Ops whose kernel signature carries a flat element count and no axis order,
#: so the same C is correct in any layout. Derived by inspection of
#: KernelSpec.signature (design doc section 3): a `int n` signature cannot
#: express an axis order, therefore cannot get one wrong.
#:
#: Kept as an explicit list rather than sniffed from the signature string at
#: import time so that adding an op here is a deliberate act -- an op that grows
#: an N,C,H,W signature later must not silently stay on this list.
LAYOUT_AGNOSTIC_OPS = {
    "add_s8", "relu_s8", "relu6_s8", "sigmoid_s8", "silu_s8", "gelu_s8",
    "mul_s8", "elu_s8", "leaky_relu_s8", "tanh_s8",
    "add_f16", "relu_f16", "mul_f16", "sigmoid_f16", "silu_f16",
}


def tensor_layout(ir: dict[str, Any], name: str) -> str:
    """Declared layout of one IR tensor. Absent means nchw."""
    t = ((ir.get("tensors") or {}).get(name)) or {}
    return t.get("layout", NCHW)


def op_act_tensors(op: dict[str, Any]) -> list[tuple[str, str]]:
    """[(role, tensor_name)] over an op's ACTIVATION operands.

    Weights and biases are separate IR keys (`op["weight"]`, `op["bias"]`), not
    entries in inputs/outputs, so this really is the activation surface. Their
    layout is a different, already-solved problem (`weight_layout` /
    `_backend_pack_weight`).
    """
    out: list[tuple[str, str]] = []
    for t in (op.get("inputs") or []):
        out.append(("input", t))
    for t in (op.get("outputs") or []):
        out.append(("output", t))
    return out


def op_layouts(ir: dict[str, Any], op: dict[str, Any]) -> set[str]:
    """Set of layouts appearing on an op's activation tensors."""
    return {tensor_layout(ir, t) for _, t in op_act_tensors(op)}


def graph_layouts_for_kind(ir: dict[str, Any], kind: str) -> set[str]:
    """Union of activation layouts over every op of `kind` in the graph.

    A singleton set means one kernel can serve every dispatch of that kind. A
    two-element set means the model mixes layouts for this op, and the codegen
    has to shim the odd ones -- see `plan_for_op`.
    """
    lays: set[str] = set()
    for op in ir.get("ops") or []:
        if op.get("op") != kind:
            continue
        lays |= op_layouts(ir, op)
    return lays


def native_layouts(kind: str, backend_name: Optional[str]) -> set[str]:
    """Activation layouts backend `backend_name` has a NATIVE kernel for.

    Reads the algorithm table, which is the same table `generate_kernels`
    selects from, so the two cannot drift. An algorithm counts only if it is
    affined to this backend: a universal-affinity algorithm is the scalar
    reference, which is NCHW by construction.

    nchw is always in the result -- every op has a reference_impl, and
    KernelSpec.__post_init__ guarantees the algorithm queue is never empty.
    """
    from modelblaster.pipeline.reference_kernels import KERNEL_SPECS
    lays = {NCHW}
    if kind in LAYOUT_AGNOSTIC_OPS:
        return {NCHW, NHWC}
    if kind in RELAYOUT_OPS:
        return {NCHW, NHWC}
    spec = KERNEL_SPECS.get(kind)
    if spec is None or not backend_name:
        return lays
    for a in spec.algorithms:
        if not a.target_affinity or backend_name not in a.target_affinity:
            continue
        lays |= set(getattr(a, "act_layouts", None) or (NCHW,))
    return lays


def kernel_layout_for_kind(ir: dict[str, Any], kind: str,
                           backend_name: Optional[str]) -> str:
    """Which layout THE ONE kernel emitted for `kind` on `backend_name` reads.

    There is exactly one `kernel_<op>_<mid>` per (model, backend) -- kernels.c
    holds one implementation per op kind -- so this is a per-kind decision even
    though layout is a per-tensor property. The rule:

      * every op of this kind is nchw            -> "nchw"  (today, unchanged)
      * every op is nhwc AND this backend has a native nhwc kernel -> "nhwc"
      * anything else                            -> "nchw", and emit_model
        shims the ops that disagree.

    The "anything else" arm is what makes an arbitrary layout assignment
    CORRECT on an arbitrary backend rather than merely fast on one: a backend
    with no nhwc conv still gets a working conv, it just pays two conversions
    for it. That is the whole reason the shim exists (design doc section 8.1).
    """
    if kind in RELAYOUT_OPS or kind in LAYOUT_AGNOSTIC_OPS:
        return NCHW          # meaningless for these; nothing branches on it
    lays = graph_layouts_for_kind(ir, kind)
    if lays == {NHWC} and NHWC in native_layouts(kind, backend_name):
        return NHWC
    return NCHW


def plan_for_op(ir: dict[str, Any], op: dict[str, Any],
                backend_name: Optional[str]) -> str:
    '''"native" | "shim" | "none" for one op on one backend.

    * "none"   -- every activation is nchw; today's code path, unchanged.
    * "native" -- the op's activations match the layout this backend's kernel
                  for that op reads. Emit the ordinary call.
    * "shim"   -- they do not. The codegen must relayout into per-hart scratch,
                  call the kernel, and relayout the result back.

    A single op carrying TWO different layouts across its operands is not
    representable by either path; only the relayout ops may do that.
    '''
    kind = op.get("op")
    lays = op_layouts(ir, op)
    if lays <= {NCHW}:
        return "none"
    if kind in RELAYOUT_OPS or kind in LAYOUT_AGNOSTIC_OPS:
        return "native"
    if len(lays) != 1:
        raise SystemExit(
            f"activation layout: op {op.get('name')!r} ({kind}) mixes layouts "
            f"{sorted(lays)} across its own operands. Only the relayout ops "
            f"({sorted(RELAYOUT_OPS)}) may do that; every other kernel reads "
            f"and writes one layout. assign_layouts should have put a relayout "
            f"dispatch on this edge.")
    want = next(iter(lays))
    return "native" if want == kernel_layout_for_kind(ir, kind, backend_name) \
        else "shim"


def assert_relayout_roles(ir: dict[str, Any], op: dict[str, Any]) -> None:
    """A relayout op must have its input in the FROM layout and its output in
    the TO layout. Getting this backwards is a pure permutation applied in the
    wrong direction: same size, no error, wrong answer.
    """
    kind = op.get("op")
    want_in, want_out = RELAYOUT_OPS[kind]
    got_in = tensor_layout(ir, (op.get("inputs") or [None])[0])
    got_out = tensor_layout(ir, (op.get("outputs") or [None])[0])
    if got_in != want_in or got_out != want_out:
        raise SystemExit(
            f"relayout direction contradicts the declared layouts:\n"
            f"  op      {op.get('name')!r} ({kind})\n"
            f"  input   {(op.get('inputs') or [None])[0]!r} layout={got_in!r} "
            f"(expected {want_in!r})\n"
            f"  output  {(op.get('outputs') or [None])[0]!r} layout={got_out!r} "
            f"(expected {want_out!r})\n"
            f"A relayout applied in the wrong direction is a permutation, so "
            f"it is size-identical and silent. Fix the layout assignment or "
            f"use {RELAYOUT_TO[got_out]!r} instead.")


def relayout_shape(ir: dict[str, Any], op: dict[str, Any]) -> dict[str, int]:
    """(N, C, H, W) for a relayout op, from its own shape or its tensor."""
    sh = op.get("shape") or {}
    if all(k in sh for k in ("N", "C", "H", "W")):
        return {k: int(sh[k]) for k in ("N", "C", "H", "W")}
    t = (op.get("inputs") or [None])[0]
    shape = list(((ir.get("tensors") or {}).get(t) or {}).get("shape") or [])
    if len(shape) != 4:
        raise SystemExit(
            f"relayout op {op.get('name')!r} needs a 4-D tensor shape; "
            f"{t!r} has shape {shape}")
    return {"N": int(shape[0]), "C": int(shape[1]),
            "H": int(shape[2]), "W": int(shape[3])}
