"""Composite seeds: a structural variant starts at PARITY with the incumbent.

A structural hint (``fuse_attention``, ``fold_layout``) replaces a sub-graph of the current graph by one op of a new
kind (``attention_f16``, ``bmm_strided_f16``, ``fused__bmm_strided_f16__...``). That kind has no tuned kernel, and
its hand seed is a correctness anchor that runs 7-13x slower than the sub-graph it replaces
(l3_farm/analysis/attention_fusion.md), so a variant seeded with it loses the measure-and-accept gate long before
KernelBlaster can make it fast.

The composite seed of the new op instead *is* the incumbent: the current-best kernels of every op the rewrite
absorbed (the q/k/v split / transpose copies, the score bmm (+ scale + mask fusion), the softmax, the PV bmm, the
output transpose), called back to back inside the new slot's entry with the same shapes, in graph order. The
intermediates the rewrite deleted from the graph (copies, the score tensor, the probabilities) live in a slot-owned
workspace, so the composite needs nothing the new kind's fixed signature does not pass:

  * tensors bound to the new op's pointer params (q/k/v roots, the output, a fused group's scratch) are those
    params, with view offsets folded in (``view`` ops are pointer arithmetic, as in kb_export);
  * a constant the rewrite turned into a flag (the causal mask -> ``causal = 1``) is re-materialised once in the
    workspace (only constants the matcher verified, ``verified``);
  * every other tensor gets a workspace region; regions are reused once their tensor is dead (linear scan), so the
    workspace is the peak of live intermediates, not their sum (43: 512 MiB, the score + probability pair is never
    live at once with the copies; 31: 2 x 4 GiB). Workspaces above ``cap_bytes`` are refused (the hand seed is used,
    with the reason recorded) -- see ``variant_discovery.md`` for the batch-chunked alternative.

The workspace is one device allocation per slot TU, made on the first call (``cudaMalloc`` under
``cudaStreamCaptureModeRelaxed``, so a first call inside a CUDA-graph capture is legal) and grown, never freed while
the process runs: the timed calls are exactly the incumbent's kernel launches. Each composite branch is guarded on
the instance's int shape params (the cache's shape guard); any other shape falls through to the kind's hand seed, so
the slot stays correct for every instance and for KernelBlaster's own shape probes.

The header comment of the source tells the agent what it is looking at (the target structure, the intermediates it
may eliminate, the correctness reference). This module is pure Python (no torch): unit-tested offline.
"""
from __future__ import annotations

import hashlib
import re

from .kernel_abi import bind, parse_signature

ELEM_BYTES = {"f16": 2, "bf16": 2, "f32": 4, "i32": 4, "i64": 8, "u8": 1, "bool": 1, "i8": 1}
DEFAULT_CAP_BYTES = 8 << 30
ALIGN = 256

GOALS = {
    "attention_f16": (
        "TARGET: one flash-style kernel (per (batch*head, query tile): loop over K/V tiles in shared memory, "
        "S = Q K^T with mma.sync m16n8k16 + ldmatrix, online softmax with a running max / rescaled sum in fp32 "
        "registers, P re-packed to fp16 A-fragments, O += P V on tensor cores, normalise once; causal: skip key "
        "tiles above the diagonal, mask the diagonal tile; act==1: relu, no normalisation).\n"
        "ELIMINATE: the q/k/v split + head-transpose copies (read Q/K/V in place through q_*/k_*/v_* strides), the "
        "[M,N] score / masked / probability tensors (never materialise them), the output transpose (write O through "
        "o_* strides). When all of them are gone the workspace is unused: drop mb_ws() entirely.\n"
        "CORRECTNESS: the slot reference is EXACT fp32 attention rounded once at the output (not this fp16-rounded "
        "chain); keep the shape guard + fallback for other shapes."),
    "bmm_strided_f16": (
        "TARGET: one tensor-core GEMM that reads A/B and writes C through the given strides "
        "(fold_transpose_into_load: a_sk==1 -> row-major A tile, a_sm==1 -> ldmatrix.trans; b_sn==1 / b_sk==1 "
        "likewise; c_sm/c_sn pick the epilogue store order; two batch levels B0 x B1 index heads directly).\n"
        "ELIMINATE: the strided_copy kernels below and their workspace buffers (each one is a full read + write of "
        "an operand that the GEMM can address in place).\n"
        "CORRECTNESS: bit-for-bit the same math as the bmm (fp32 accumulate, fp16 store); keep the fallback."),
}


def _goal(kind: str) -> str:
    if kind in GOALS:
        return GOALS[kind]
    if kind.startswith("fused__bmm_strided"):
        return GOALS["bmm_strided_f16"] + "\nThe epilogue stages of the fused group stay fused as they are."
    return "TARGET: one kernel for the whole sub-graph; ELIMINATE the intermediates held in the workspace."


def absorbed_names(new_op: dict, base_names: set) -> list[str]:
    """Names of the base-graph ops a rewritten op replaced: attention_fuse records ``absorbed`` + ``folded``;
    layout_fold keeps the gemm's name and records ``folded`` copies."""
    names = list(new_op.get("absorbed") or [])
    names += [n for n in new_op.get("folded") or [] if n not in names]
    if new_op["name"] in base_names and new_op["name"] not in names:
        names.append(new_op["name"])
    return [n for n in names if n in base_names]


def diff_ops(g0: dict, g1: dict) -> tuple[list[str], list[dict]]:
    """(base op names removed or changed by a rewrite, the new / changed ops of g1)."""
    a = {o["name"]: o for o in g0["ops"]}
    b = {o["name"]: o for o in g1["ops"]}

    def same(x, y):          # dispatch numbering (apply_hints renumbers) is bookkeeping, not the op
        return {k: v for k, v in x.items() if k not in _BOOKKEEPING} == {k: v for k, v in y.items()
                                                                       if k not in _BOOKKEEPING}
    removed = [n for n, o in a.items() if n not in b or not same(b[n], o)]
    new = [o for n, o in b.items() if n not in a or not same(a[n], o)]
    return removed, new


_BOOKKEEPING = {"dispatch_id", "depends_on"}


def _aliases(graph: dict) -> dict:
    al = {}
    for o in graph["ops"]:
        if o["op"] == "view":
            al[o["outputs"][0]] = (o["inputs"][0], int(o.get("offset", 0) or 0))
    return al


def _resolve(t: str, al: dict) -> tuple[str, int]:
    off = 0
    seen = set()
    while t in al and t not in seen:
        seen.add(t)
        t, o = al[t]
        off += o
    return t, off


def _numel(graph, t) -> int:
    n = 1
    for d in graph["tensors"][t]["shape"]:
        n *= int(d)
    return n


def _esize(graph, t) -> int:
    return ELEM_BYTES.get(str(graph["tensors"][t].get("dtype", "f32")), 4)


def _bind_any(op: dict, specs: dict):
    if "sub_ops" in op:
        from .kb_proposals import bind_fused
        return bind_fused(op, specs)
    return bind(op, parse_signature(specs[op["op"]].signature)[1])


def _cuda_sig(spec) -> str:
    from .cuda_dtypes import cuda_signature
    return cuda_signature(spec.signature)


def _c_scalar(base, v):
    from .kb_export import _c_scalar as cs
    return cs(base, v)


def plan(base_graph: dict, new_op: dict, specs: dict, *, verified_consts: set | None = None,
         cap_bytes: int = DEFAULT_CAP_BYTES) -> dict:
    """The composite's calls and workspace for one new op. -> {"ok", "why", "calls": [(op, [(param, expr)])],
    "ws_bytes", "fills": [(offset, numel, R, C)], "stages": [names]}. Pure (no sources needed)."""
    by_name = {o["name"]: o for o in base_graph["ops"]}
    names = absorbed_names(new_op, set(by_name))
    order = [o for o in base_graph["ops"] if o["name"] in set(names)]
    kops = [o for o in order if o["op"] != "view"]
    if not kops:
        return {"ok": False, "why": "no absorbed kernel ops"}
    al = _aliases(base_graph)
    try:
        new_args = _bind_any(new_op, specs)
        stage_args = [(o, _bind_any(o, specs)) for o in kops]
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "why": f"bind: {type(e).__name__}: {e}"}
    proot = {}
    for a in new_args:
        if a.kind in ("in", "out", "scratch") and a.value in base_graph["tensors"]:
            r, off = _resolve(a.value, al)
            proot.setdefault(r, (a.param.name, off, _esize(base_graph, a.value)))
    # workspace: every non-param root a stage touches; liveness over the stage sequence
    first, last, consts = {}, {}, {}
    for i, (o, args) in enumerate(stage_args):
        for a in args:
            if a.kind == "weight":
                return {"ok": False, "why": f"{o['name']}: weights inside an absorbed op are not supported"}
            if a.kind not in ("in", "out", "scratch"):
                continue
            r, _ = _resolve(a.value, al)
            if r in proot:
                continue
            td = base_graph["tensors"].get(r) or {}
            if td.get("const"):
                if not verified_consts or td["const"] not in verified_consts:
                    return {"ok": False, "why": f"constant {r} ({td['const']}) is not a verified mask"}
                if str(td.get("dtype")) != "f16" or len(td["shape"]) < 2:
                    return {"ok": False, "why": f"constant {r}: only f16 [.., R, C] masks are re-materialised"}
                consts[r] = True
                first[r], last[r] = -1, 1 << 30
                continue
            if a.kind == "in" and r not in first:
                return {"ok": False, "why": f"{o['name']} reads {r}, which nothing in the composite produces"}
            first.setdefault(r, i)
            last[r] = i
    # linear-scan placement (first fit by offset among blocks live at the same time)
    place, blocks = {}, []
    for r in sorted(first, key=lambda r: (first[r], r)):
        size = -(-_numel(base_graph, r) * _esize(base_graph, r) // ALIGN) * ALIGN
        live = sorted((b for b in blocks if not (last[b[2]] < first[r] or last[r] < first[b[2]])), key=lambda b: b[0])
        off = 0
        for bo, bs, _ in live:
            if off + size <= bo:
                break
            off = max(off, bo + bs)
        place[r] = off
        blocks.append((off, size, r))
    ws = max((o + s for o, s, _ in blocks), default=0)
    if ws > cap_bytes:
        return {"ok": False, "why": f"workspace {ws / 2**20:.0f} MiB > cap {cap_bytes / 2**20:.0f} MiB", "ws_bytes": ws}

    def expr(t):
        r, off = _resolve(t, al)
        es = _esize(base_graph, r)
        if r in proot:
            p, poff, pes = proot[r]
            d = (off - poff) * es
            return f"((char *)({p}) + {d}L)" if d else f"((char *)({p}))"
        return f"(ws + {place[r] + off * es}L)"

    calls = []
    for o, args in stage_args:
        cargs = []
        for a in args:
            if a.kind in ("in", "out", "scratch"):
                cargs.append(("ptr", a.param.ctype, expr(a.value)))
            elif a.kind == "null":
                cargs.append(("raw", a.param.ctype, "nullptr"))
            else:
                cargs.append(("raw", a.param.ctype, _c_scalar(a.param.base_type, a.value)))
        calls.append((o, cargs))
    fills = []
    for r in consts:
        sh = [int(x) for x in base_graph["tensors"][r]["shape"]]
        fills.append((place[r], _numel(base_graph, r), sh[-2], sh[-1]))
    guard = [(a.param.name, a.param.base_type, a.value) for a in new_args
             if a.kind == "scalar" and a.param.base_type in ("int", "long", "int64_t", "size_t")]
    return {"ok": True, "why": None, "calls": calls, "ws_bytes": ws, "fills": fills, "guard": guard,
            "stages": [o["name"] for o in kops], "stage_kinds": [o["op"] for o in kops],
            "workspace": {r: [place[r], _numel(base_graph, r) * _esize(base_graph, r)] for r in place}}


def op_source(o: dict, specs: dict, sources: dict, sub_sources: dict, lib) -> str:
    """The source a base-graph op runs today (as kb_export seeds it): its slot's cached winner, else a fused op's
    stages back to back (cached stage winners), else the hand kernel."""
    from .kb_export import base_slot_id, slot_id_of
    sid, bsid = slot_id_of(o, specs), base_slot_id(o["op"], specs)
    for k in (sid, bsid, o["op"]):
        if k in (sources or {}):
            return sources[k]
    spec = specs[o["op"]]
    if hasattr(spec, "cuda_seed"):
        return spec.cuda_seed(lib, sub_sources=(sub_sources or {}).get(sid, {}))
    return lib.slot_source(parse_signature(spec.signature)[0])


def build_source(new_kind: str, spec, branches: list[dict], stage_sources: dict, fallback_src: str) -> str:
    """One slot source for ``new_kind``: per branch (one per distinct instance shape) the composite under its
    shape guard, else the hand seed. ``stage_sources``: {op name: source} for every stage of every branch."""
    from .kb_export import namespaced

    entry, params = parse_signature(_cuda_sig(spec))
    includes, blocks, ns_of = [], [], {}
    for src in dict.fromkeys(stage_sources.values()):
        h = "cmp_" + hashlib.sha1(src.encode()).hexdigest()[:10]
        ns_of[src] = h
        inc, body = namespaced(h, src)
        includes += [i for i in inc if i not in includes]
        blocks.append(body)
    inc, fb = namespaced("cmp_fallback", fallback_src)
    includes += [i for i in inc if i not in includes]
    blocks.append(fb)
    plist = ", ".join(f"{p.ctype} {p.name}" for p in params)
    pnames = ", ".join(p.name for p in params)
    body = []
    for k, br in enumerate(branches):
        cond = " && ".join(f"{n} == {_c_scalar(bt, v)}" for n, bt, v in br["guard"]) or "true"
        lines = [f"    if ({cond}) {{   // {' -> '.join(br['stages'])}",
                 f"        char *ws = mb_ws({br['ws_bytes']}L);"]
        for off, n, R, C in br["fills"]:
            lines += [f"        static char *filled_{k}_{off} = nullptr;",
                      f"        if (filled_{k}_{off} != ws) {{ mb_fill_triu<<<(int)(({n}L + 255) / 256), 256>>>("
                      f"(__half *)(ws + {off}L), {n}L, {R}, {C}); filled_{k}_{off} = ws; }}"]
        for o, cargs in br["calls"]:
            src = stage_sources[o["name"]]
            e = parse_signature(_stage_sig(o))[0]
            av = [f"({ct.replace('_Float16', '__half')})({ex})" if kind == "ptr" else ex for kind, ct, ex in cargs]
            lines.append(f"        {ns_of[src]}::{e}({', '.join(av)});   // {o['name']} ({o['op']})")
        lines += ["        return;", "    }"]
        body += lines
    body.append(f"    cmp_fallback::{entry}({pnames});   // other shapes: the kind's hand seed")
    ws_total = max((b["ws_bytes"] for b in branches), default=0)
    goal = _goal(new_kind)
    header = (f"// COMPOSITE SEED for {new_kind}: the incumbent's current-best kernels of the sub-graph this op "
              f"replaced,\n// called back to back (starts at parity with the unfused graph). Workspace: "
              f"{ws_total / 2**20:.1f} MiB of intermediates.\n" +
              "".join(f"// {l}\n" for l in goal.splitlines()))
    helpers = (
        "#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n"
        "static char *mb_ws(long bytes) {\n"
        "    static char *ws = nullptr; static long cap = 0;\n"
        "    if (bytes > cap) {   // first call (or a larger shape): allocate; legal inside a relaxed graph capture\n"
        "        cudaStreamCaptureMode m = cudaStreamCaptureModeRelaxed;\n"
        "        cudaThreadExchangeStreamCaptureMode(&m);\n"
        "        void *p = nullptr;\n"
        "        if (cudaMalloc(&p, (size_t)bytes) == cudaSuccess) { ws = (char *)p; cap = bytes; }\n"
        "        cudaThreadExchangeStreamCaptureMode(&m);\n"
        "    }\n"
        "    return ws;\n"
        "}\n"
        "__global__ void mb_fill_triu(__half *m, long n, int R, int C) {   // causal mask: 1 above the diagonal\n"
        "    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (i < n) { long rc = i % ((long)R * C); m[i] = __float2half((rc % C) > (rc / C) ? 1.0f : 0.0f); }\n"
        "}\n")
    return (header + "\n".join(includes) + "\n" + helpers + "\n" + "\n\n".join(blocks) +
            f"\n\nvoid {entry}({plist}) {{\n" + "\n".join(body) + "\n}\n")


_STAGE_SPECS: dict = {}


def _stage_sig(o: dict) -> str:
    return _cuda_sig(_STAGE_SPECS[o["op"]])


def composite_sources(base_graph: dict, variant_graph: dict, specs: dict, *, sources: dict, sub_sources: dict,
                      lib, kinds: tuple = ("attention_f16", "bmm_strided_f16", "fused__bmm_strided_f16"),
                      verified_consts: set | None = None, cap_bytes: int = DEFAULT_CAP_BYTES) -> tuple[dict, dict]:
    """{slot id: composite source} for the variant's new ops of ``kinds`` (prefix match), and a report
    {slot id: {ok, why, stages, ws_bytes}}. Instances of one slot with different shapes get one guarded branch
    each; a slot whose composite cannot be built is left out (the hand seed / fused stages stay)."""
    from .kb_export import slot_id_of
    from .kb_proposals import with_fused_specs

    specs = with_fused_specs(variant_graph, with_fused_specs(base_graph, specs))
    _STAGE_SPECS.clear()
    _STAGE_SPECS.update(specs)
    _, new_ops = diff_ops(base_graph, variant_graph)
    per_slot: dict[str, list] = {}
    for o in new_ops:
        if not any(o["op"] == k or o["op"].startswith(k + "__") for k in kinds):
            continue
        per_slot.setdefault(slot_id_of(o, specs), []).append(o)
    out, rep = {}, {}
    for sid, ops in per_slot.items():
        branches, stage_src, why = [], {}, None
        seen_guards = set()
        for o in ops:
            p = plan(base_graph, o, specs, verified_consts=verified_consts, cap_bytes=cap_bytes)
            if not p["ok"]:
                why = p["why"]
                break
            g = tuple((n, v) for n, _, v in p["guard"])
            if g in seen_guards:
                continue
            seen_guards.add(g)
            branches.append(p)
            for so, _ in p["calls"]:
                stage_src[so["name"]] = op_source(so, specs, sources, sub_sources, lib)
        kind = ops[0]["op"]
        if why is None:
            fb = op_source({**ops[0], "slot_variant": None}, specs, {}, {}, lib)
            out[sid] = build_source(kind, specs[kind], branches, stage_src, fb)
        rep[sid] = {"ok": why is None, "why": why, "kind": kind,
                    "stages": [b["stages"] for b in branches], "ws_bytes": max((b["ws_bytes"] for b in branches),
                                                                                default=0)}
    return out, rep


def semantics_note(kind: str, rep: dict) -> str:
    """Appended to the slot's semantics in the variant's manifest (the agent's task statement)."""
    st = "; ".join(" -> ".join(s) for s in rep.get("stages", []))
    return ("\n[composite seed] The seed of this slot runs the incumbent's tuned kernels of the replaced sub-graph "
            f"back to back ({st}), with the intermediates in a {rep.get('ws_bytes', 0) / 2**20:.1f} MiB workspace: "
            "it starts at the unfused graph's speed and only a real fusion makes it faster.\n" + _goal(kind))


def is_composite(src: str) -> bool:
    return bool(re.search(r"^// COMPOSITE SEED", src or "", re.M))
