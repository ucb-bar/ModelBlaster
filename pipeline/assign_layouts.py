#!/usr/bin/env python3
"""Assign an activation layout to every IR tensor, and materialise the boundaries.

Stage 3 of docs/IR_TENSOR_LAYOUT_DESIGN.md, and the "decide" layer of the
three-layer design in its section 8:

  declare      AlgorithmCandidate.act_layouts        (reference_kernels.py)
  DECIDE       ir["tensors"][t]["layout"]            (this file)
  materialise  nchw_to_nhwc_s8 / nhwc_to_nchw_s8 dispatches on island edges
  enforce      generate_kernels.assert_act_layout_contract  +  emit_model shims

  assign_layouts.py IN.json OUT.json [--policy none|islands]
                    [--hint HINT.json --model NAME] [--report]

`none` (the default) writes the graph back unchanged apart from provenance, so
this pass is inert unless asked. `islands` is opt-in per model behind an explicit
hint file (`modelblaster.layout_hints/v1`), the same shape as
`modelblaster.split_hints/v1` in apply_split_hint.py and
`modelblaster.shard_hints/v1` in apply_shard_hint.py:

    {"contract": "modelblaster.layout_hints/v1",
     "reason": "87% of gemmini conv time is NCHW<->NHWC conversion (fq 475)",
     "networks": [{"network": "dronet", "layout": "nhwc"}]}

Why this is its own pass rather than a flag on the splitter: layout is a
property of a TENSOR, not of an op or a backend. buffers.c is compiled once per
model and shared by every backend (generate_skeleton.py:112-123), so a tensor has
exactly one physical layout no matter which hart the schedule lands its producer
on. Weights are the opposite -- weights.c is per backend, which is why
_backend_pack_weight can permute them per backend and this cannot.

WHAT AN ISLAND IS. A maximal connected run of ops that can all consume and
produce NHWC, with a relayout dispatch on every edge that crosses out of it. The
model input and output surfaces stay NCHW, always: that is what keeps io.npz,
test_golden.bin, the in-binary compare and every host parser working unmodified,
and it is what makes an island BIT-IDENTICAL end to end rather than merely close
(a permutation is a bijection; no arithmetic changes). Design doc section 7.

Inserting relayout dispatches renumbers every op after the first one, so the
output graph carries `id_remap` the same way apply_split_hint's does -- one
entry per original dispatch_id, list-valued.
"""
import argparse, collections, copy, json, sys

from typing import Any

LAYOUT_CONTRACT = "modelblaster.layout_hints/v1"

#: Ops with a native NHWC kernel on at least one backend today. Kept here rather
#: than inferred from act_layouts so that adding one is a deliberate act with a
#: kernel behind it -- though the codegen gate is deny-by-default and will refuse
#: anything whose declared act_layouts do not cover what this pass assigned, so
#: a stale entry here is caught rather than shipped.
NHWC_CAPABLE: set[str] = {"conv2d_s8", "maxpool2d_s8", "batchnorm2d_s8"}

#: Layout-agnostic: pure elementwise, same bytes in any order, so they never
#: force a conversion and never need a variant. They join an island for free.
LAYOUT_AGNOSTIC = {"add_s8", "relu_s8", "mul_s8", "sigmoid_s8", "silu_s8",
                   "relu6_s8", "gelu_s8", "add_f16", "relu_f16", "mul_f16"}

#: Ops that must never be inside an island even though they look harmless.
#: `view` is a codegen alias and stays layout-preserving -- but the thing it
#: aliases INTO (a flatten feeding a linear) reads the buffer as a flat vector,
#: so the element ORDER it sees changes under NHWC. chunk2_c1 and cat*_c1 are
#: channel-axis slices/concats whose codegen assumes contiguous planes, which is
#: exactly what NHWC removes. Design doc section 10.
ISLAND_FORBIDDEN = {"view", "chunk2_c1", "chunk2_c1_f16", "chunk2_c1_s8",
                    "cat2_c1_s8", "cat3_c1_s8", "cat4_c1_s8",
                    "cat2_c1_f16", "slice_c_s8", "slice_c_f16",
                    "linear_s8", "linear", "linear_s8_pc"}


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


def _is_4d(ir, t):
    return len(list(((ir.get("tensors") or {}).get(t) or {}).get("shape") or [])) == 4


def _eligible(ir, op) -> bool:
    """Can this op sit INSIDE an island?

    Refusals here are the safety half of the pass, and each has a specific
    failure it prevents rather than a general caution:

      * a split or shard tile along OC -- an OC slice is a contiguous plane
        range only in NCHW; under NHWC the channel is innermost, so the
        codegen's `elem_offset = oc0*OH*OW` alias would select the wrong
        elements. Same size, plausible wrong answer. apply_split_hint carries
        the mirror guard (stage 2) for the other order of operations.
      * an OH tile -- the reverse case is FINE in principle (a row band is
        contiguous under NHWC, which is the whole point of section 6) but the
        NHWC gemmini kernel does not implement the windowed walk the NCHW one
        carries, so an OH tile would link against a kernel that ignores the
        window. Refused until that path exists.
      * anything whose activations are not 4-D -- a relayout has nothing to
        permute.
    """
    kind = op.get("op")
    if kind in ISLAND_FORBIDDEN:
        return False
    if kind in LAYOUT_AGNOSTIC:
        return True
    if kind not in NHWC_CAPABLE:
        return False
    if op.get("split_from") or op.get("shard_from"):
        return False
    for t in (op.get("inputs") or []) + (op.get("outputs") or []):
        if not _is_4d(ir, t):
            return False
    return True


def _islands(ir, ops):
    """Maximal runs of eligible ops, connected through their tensors.

    Connectivity is on the dataflow graph, not on program order: an op joins an
    island if it is eligible AND it shares a tensor with it. A layout-agnostic
    op is eligible but must not form an island on its OWN -- an island of three
    add_s8 would convert into NHWC and straight back out for no reason -- so a
    component is kept only if it contains at least one NHWC_CAPABLE op.
    """
    prod = {}
    for o in ops:
        for t in (o.get("outputs") or []):
            prod[t] = o["dispatch_id"]
    elig = {o["dispatch_id"]: o for o in ops if _eligible(ir, o)}
    parent = {d: d for d in elig}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for d, o in elig.items():
        for t in (o.get("inputs") or []):
            p = prod.get(t)
            if p is not None and p in elig:
                union(p, d)
    comps = collections.defaultdict(list)
    for d in elig:
        comps[find(d)].append(d)
    out = []
    for _, members in comps.items():
        if any(elig[d]["op"] in NHWC_CAPABLE for d in members):
            out.append(sorted(members))
    return sorted(out)


def assign_islands(graph: dict[str, Any]) -> dict[str, Any]:
    """Rewrite `graph`: mark island tensors nhwc and insert boundary relayouts."""
    out = copy.deepcopy(graph)
    tensors = out.setdefault("tensors", {})
    ops = [o for o in out["ops"] if o.get("dispatch_id") is not None]
    islands = _islands(out, ops)
    if not islands:
        raise SystemExit(
            "[assign_layouts] policy 'islands' found no island in this graph. "
            "Every candidate op was refused by _eligible() -- an OC split, a "
            "non-4-D activation, or an op kind with no nhwc kernel. Assigning "
            "nothing here would produce a correct-but-NCHW build that looks "
            "like a success, so this is an error rather than a no-op.")

    in_names = set(out["input"].get("tensors")
                   or ([out["input"]["tensor"]] if out["input"].get("tensor") else []))
    of = out["output"]
    out_names = set(of["tensors"]) if of.get("tensors") else (
        {of["tensor"]} if of.get("tensor") else set())

    island_ids = {d for isl in islands for d in isl}
    by_id = {o["dispatch_id"]: o for o in ops}
    # Consumers of each tensor, including the non-dispatch `view` aliases: a
    # flatten feeding a linear reads the buffer as a flat vector, so it is very
    # much a consumer for layout purposes even though it emits no kernel.
    consumers = collections.defaultdict(list)
    for o in out["ops"]:
        for t in (o.get("inputs") or []):
            consumers[t].append(o)
    producer = {}
    for o in ops:
        for t in (o.get("outputs") or []):
            producer[t] = o["dispatch_id"]

    new_ops: list[dict[str, Any]] = []
    n_head = n_tail = 0
    # Head conversions: a tensor an island op READS that no island op produces.
    head_src: dict[str, str] = {}     # original tensor -> nhwc tensor
    for d in sorted(island_ids):
        for t in (by_id[d].get("inputs") or []):
            if producer.get(t) in island_ids or t in head_src:
                continue
            head_src[t] = f"{t}.nhwc"
    # Tail conversions: a tensor an island op WRITES that anything outside the
    # island reads, or that is a model output. The island op is retargeted to a
    # new nhwc tensor and a relayout restores the original name, so every
    # downstream consumer -- including the codegen's view aliases and the output
    # surface -- is untouched.
    tail_src: dict[str, str] = {}     # original tensor -> nhwc tensor
    for d in sorted(island_ids):
        for t in (by_id[d].get("outputs") or []):
            outside = [c for c in consumers.get(t, [])
                       if c.get("dispatch_id") not in island_ids]
            if outside or t in out_names:
                tail_src[t] = f"{t}.nhwc"

    def _mk_tensor(base, name):
        meta = copy.deepcopy(tensors[base])
        meta["layout"] = "nhwc"
        meta.pop("split_from", None)
        meta.pop("elem_offset", None)
        tensors[name] = meta

    def _relayout_op(kind, src, dst, shape_src, name):
        sh = list(tensors[shape_src]["shape"])
        return {"op": kind, "name": name, "inputs": [src], "outputs": [dst],
                "dispatch_id": -1,
                "shape": {"N": int(sh[0]), "C": int(sh[1]),
                          "H": int(sh[2]), "W": int(sh[3])},
                "depends_on": [], "hardware_target": "any",
                "layout_boundary": True}

    for t, nm in head_src.items():
        _mk_tensor(t, nm)
    for t, nm in tail_src.items():
        _mk_tensor(t, nm)
    # Interior tensors -- produced by an island op and read only inside it --
    # simply become nhwc in place. No new buffer, no copy.
    for d in island_ids:
        for t in (by_id[d].get("outputs") or []):
            if t not in tail_src:
                tensors[t]["layout"] = "nhwc"

    # Rewire the island ops onto the nhwc names.
    for d in island_ids:
        o = by_id[d]
        o["inputs"] = [head_src.get(t, t) for t in (o.get("inputs") or [])]
        o["outputs"] = [tail_src.get(t, t) for t in (o.get("outputs") or [])]

    # Emit in the original order, dropping a head relayout in front of the first
    # island op that needs it and a tail relayout straight after the island op
    # that produces it. Both are correct by construction because ir["ops"] is in
    # topological order (extract_graph emits it that way and every rewrite
    # preserves it).
    emitted_head: set[str] = set()
    for o in out["ops"]:
        d = o.get("dispatch_id")
        if d in island_ids:
            for t, nm in head_src.items():
                if nm in (o.get("inputs") or []) and t not in emitted_head:
                    emitted_head.add(t)
                    new_ops.append(_relayout_op(
                        "nchw_to_nhwc_s8", t, nm, t, f"relayout_in.{t}"))
                    n_head += 1
            new_ops.append(o)
            for t, nm in tail_src.items():
                if nm in (o.get("outputs") or []):
                    new_ops.append(_relayout_op(
                        "nhwc_to_nchw_s8", nm, t, t, f"relayout_out.{t}"))
                    n_tail += 1
        else:
            new_ops.append(o)

    # Renumber and rewire depends_on. Same contract as apply_split_hint: the
    # published id_remap is list-valued so a consumer never has to branch on the
    # value type, and anything keyed on dispatch_id (profile CSVs, cost DBs,
    # Gantt labels) must translate through it before joining across the rewrite.
    id_remap: dict[int, list[int]] = {}
    nxt = 0
    for o in new_ops:
        if o.get("dispatch_id") is None:
            continue            # view / chunk aliases: zero-cost, never numbered
        old = o["dispatch_id"]
        o["dispatch_id"] = nxt
        if old >= 0:            # -1 marks a relayout this pass just created
            id_remap[old] = [nxt]
        nxt += 1
    # depends_on: rebuild from dataflow rather than remapping, because the
    # relayouts are new nodes with no old id to remap through and they sit ON
    # the dependency edges they were inserted into.
    prod_now = {}
    for o in new_ops:
        if o.get("dispatch_id") is None:
            continue
        for t in (o.get("outputs") or []):
            prod_now[t] = o["dispatch_id"]
    alias = {}
    for o in new_ops:
        if o.get("op") == "view":
            alias[o["outputs"][0]] = o["inputs"][0]

    def _resolve(t):
        seen = set()
        while t in alias and t not in seen:
            seen.add(t)
            t = alias[t]
        return t

    for o in new_ops:
        if o.get("dispatch_id") is None:
            continue
        deps, seen = [], set()
        for t in (o.get("inputs") or []):
            p = prod_now.get(_resolve(t))
            if p is not None and p != o["dispatch_id"] and p not in seen:
                seen.add(p)
                deps.append(p)
        o["depends_on"] = sorted(deps)

    out["ops"] = new_ops
    if isinstance(out.get("dispatches"), list):
        out["dispatches"] = [o["dispatch_id"] for o in new_ops
                             if o.get("dispatch_id") is not None]
    out["id_remap"] = {str(k): v for k, v in sorted(id_remap.items())}
    out["_layout"] = {
        "policy": "islands",
        "islands": islands,
        "head_relayouts": n_head,
        "tail_relayouts": n_tail,
        "nhwc_tensors": sorted(t for t, v in tensors.items()
                               if v.get("layout") == "nhwc"),
    }
    return out


def _load_hint(path, model):
    hint = json.loads(open(path).read())
    if hint.get("contract") != LAYOUT_CONTRACT:
        sys.exit(f"[assign_layouts] hint contract is {hint.get('contract')!r}, "
                 f"expected {LAYOUT_CONTRACT!r}")
    for e in hint.get("networks", []):
        if e.get("network") == model:
            return e
    return None


def main():
    a = argparse.ArgumentParser()
    a.add_argument("inp"); a.add_argument("out")
    a.add_argument("--policy", default="none", choices=("none", "islands"))
    a.add_argument("--hint", default=None)
    a.add_argument("--model", default=None)
    a.add_argument("--report", action="store_true")
    a = a.parse_args()
    ir = json.load(open(a.inp))
    if a.report:
        report(ir)

    policy = a.policy
    if a.hint:
        model = a.model or ir.get("name")
        entry = _load_hint(a.hint, model)
        if entry is None:
            print(f"[assign_layouts] no layout hint for {model!r} in {a.hint} "
                  f"-- copying IR through unchanged", file=sys.stderr)
            policy = "none"
        elif entry.get("layout", "nhwc") != "nhwc":
            sys.exit(f"[assign_layouts] hint asks for layout "
                     f"{entry.get('layout')!r}; only 'nhwc' is implemented.")
        else:
            policy = "islands"
    elif policy == "islands":
        sys.exit("[assign_layouts] policy 'islands' is opt-in per model: pass "
                 "--hint with a modelblaster.layout_hints/v1 file naming this "
                 "network. Gating it behind a hint file is the same discipline "
                 "apply_split_hint and apply_shard_hint follow, and it is what "
                 "keeps every model that does not ask for this byte-identical.")

    if policy == "islands":
        ir = assign_islands(ir)
        n = len(ir["_layout"]["nhwc_tensors"])
        print(f"[assign_layouts] policy=islands islands={ir['_layout']['islands']} "
              f"head={ir['_layout']['head_relayouts']} "
              f"tail={ir['_layout']['tail_relayouts']} nhwc_tensors={n}")
    else:
        n = 0
        print(f"[assign_layouts] policy=none assigned=0 -> {a.out}")
    ir.setdefault("_rewrite", []).append(
        {"pass": "assign_layouts", "policy": policy, "assigned": n})
    json.dump(ir, open(a.out, "w"), indent=1)


main()
