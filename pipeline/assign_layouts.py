#!/usr/bin/env python3
"""Assign an activation layout to every IR tensor.

Stage 2 of docs/IR_TENSOR_LAYOUT_DESIGN.md. **This pass assigns nothing yet.**
It exists so the plumbing -- the IR key, the codegen gate, the splitter guard --
can land and be exercised while every tensor is still nchw, which is the point:
a graph without a `layout` key must produce byte-identical output to before any
of this existed, and that is only credible if the pass is present and inert
rather than absent.

  assign_layouts.py IN.json OUT.json [--policy none|islands] [--report]

`none` (default) writes the graph back unchanged apart from provenance.
`islands` is stage 3 and currently refuses.

Why this is its own pass rather than a flag on the splitter: layout is a
property of a TENSOR, not of an op or a backend. buffers.c is compiled once per
model and shared by every backend (generate_skeleton.py:112-123), so a tensor
has exactly one physical layout no matter which hart the schedule lands its
producer on. Weights are the opposite -- weights.c is per backend, which is why
_backend_pack_weight can permute them per backend and this cannot.
"""
import argparse, collections, json, sys

#: Ops that can carry a non-nchw activation layout once stage 3 lands. Kept here
#: rather than inferred so that adding one is a deliberate act with a kernel
#: behind it -- the codegen gate is deny-by-default and will refuse anything
#: whose declared act_layouts do not cover what this pass assigned.
NHWC_CAPABLE: set[str] = set()          # stage 3 populates this

#: Layout-agnostic: pure elementwise, same bytes in any order, so they never
#: force a conversion and never need a variant.
LAYOUT_AGNOSTIC = {"add_s8", "relu_s8", "mul_s8", "sigmoid_s8", "silu_s8",
                   "relu6_s8", "gelu_s8", "add_f16", "relu_f16", "mul_f16"}


def report(ir):
    ops = [o for o in ir.get("ops", []) if o.get("dispatch_id") is not None]
    lay = collections.Counter(
        (ir.get("tensors") or {}).get(t, {}).get("layout", "nchw")
        for o in ops for t in (o.get("outputs") or []))
    kinds = collections.Counter(o["op"] for o in ops)
    agn = sum(v for k, v in kinds.items() if k in LAYOUT_AGNOSTIC)
    cap = sum(v for k, v in kinds.items() if k in NHWC_CAPABLE)
    print(f"  {len(ops)} dispatches; output-tensor layouts: {dict(lay)}")
    print(f"  layout-agnostic ops: {agn}   nhwc-capable ops: {cap}"
          f"   would need a conversion or a variant: {len(ops) - agn - cap}")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("inp"); a.add_argument("out")
    a.add_argument("--policy", default="none", choices=("none", "islands"))
    a.add_argument("--report", action="store_true")
    a = a.parse_args()
    ir = json.load(open(a.inp))
    if a.report:
        report(ir)
    if a.policy == "islands":
        sys.exit("[assign_layouts] policy 'islands' is stage 3 and not "
                 "implemented. Landing it requires, in order: nhwc entry points "
                 "for the island's ops (populating NHWC_CAPABLE), relayout "
                 "dispatches on the island boundary edges, and shim generation "
                 "in emit_model for backends with no native nhwc path. Until "
                 "all three exist, assigning nhwc would trip the deny-by-default "
                 "gate in generate_kernels.assert_act_layout_contract -- which "
                 "is the intended behaviour, not a bug to work around.")
    ir.setdefault("_rewrite", []).append(
        {"pass": "assign_layouts", "policy": a.policy, "assigned": 0})
    json.dump(ir, open(a.out, "w"), indent=1)
    print(f"[assign_layouts] policy={a.policy} assigned=0 -> {a.out}")


main()
