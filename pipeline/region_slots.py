"""Opt-in multi-op persistent *region* slots (A2 capabilities, item 4; cuda_vs_flow.md recommendation 4).

A region slot owns a whole connected sub-graph (an RNN stack, an attention block, or the whole model) as ONE op of
the IR, so the slot optimiser may replace the sub-graph by one persistent kernel ("megakernel"): Claude's CUDA-only
baseline got -8% on 41_GRUBidirectional (one kernel for 6 layers x 2 directions + concats) and about -17% on
43_MinGPTCausalAttention (QKV GEMM -> attention -> projection with an atomic task queue) on top of per-op kernels.

IR form (``fuse_region``): ``{"op": "region__<pattern>__<sha10>", "region": true, "sub_ops": [members in IR order,
views included], "inputs": [external tensors], "outputs": [member outputs read outside or returned],
"internal": [non-alias member outputs read only inside], "shape": {"m0": member 0 shape, ...}}``, placed at the
position of its last member. The kind hashes the members' kinds, shapes and wiring, so a cached region winner is
shape-guarded like any slot (``kb_ingest``) and never applies to a different sub-graph.

Fixed signature (``RegionKernelSpec``): one pointer per distinct external tensor and per internal tensor
(harness-allocated scratch, not checked), names ``<role>_<tensor>``; member scalars (shapes, strides, flags) are
compile-time constants of the region (the same values every call), listed in the semantics. Outputs are checked
per slot (isolated, against the reference walk) and in the whole model; internal tensors are scratch a fused
kernel need not write. The slot may own more scratch / flags / counters (static, lazily allocated, reused across
calls with epoch tags; instance-safe through lanes when concurrency is on).

Composite seed (``cuda_seed``): every member's CURRENT kernel (per member: its slot variant's cached winner when
given, else the hand kernel) called in order inside the region's entry point, each distinct source in its own
namespace; views are pointer arithmetic. So a region starts at the incumbent's speed (parity) and the optimiser
can only improve from there.

Reference: the walker runs the members (``walk_torch`` recurses into ``sub_ops``); the C reference shown to the LLM
is the members' references in order.

Co-residency: a megakernel usually spins across blocks. Exports containing a region slot get the deadlock
watchdog and ``mb_sm_budget()`` of ``kb_concurrency`` (concurrency forced to ``on`` for that partition unless the
caller passed ``concurrency="off"`` explicitly): a hung replay prints ``MB_CONC_DEADLOCK`` and the callers' existing
sequential fallback / failed-candidate handling applies. The slot semantics state the co-residency rule.

Switch: ``--region-slots off|rnn_stack|attention_block|all`` (comma list; default off: no region is ever proposed
or built, every export unchanged).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from dataclasses import dataclass, field

from .kernel_abi import Param, bind, parse_signature
from .reference_kernels import KernelSpec

ALIAS_OPS = ("view", "chunk2_c1")
PREFIX = "region__"
PATTERNS = ("rnn_stack", "attention_block", "all")
ENV = "MB_REGION_SLOTS"
RECURRENT = ("gru_seq", "lstm_seq", "lstm")
MAX_MEMBERS = 256                     # the default cap on a region's members (views included)
ENV_MAX = "MB_REGION_MAX_MEMBERS"     # raise / lower it (region windows, docs/region_windows.md); unset = MAX_MEMBERS


def max_members() -> int:
    """The member cap of ``fuse_region``: ``$MB_REGION_MAX_MEMBERS`` (a positive int), else :data:`MAX_MEMBERS` (256)."""
    v = os.environ.get(ENV_MAX, "").strip()
    if not v:
        return MAX_MEMBERS
    try:
        n = int(v)
    except ValueError:
        raise ValueError(f"{ENV_MAX}={v!r}: expected a positive integer") from None
    if n < 2:
        raise ValueError(f"{ENV_MAX}={v!r}: expected a positive integer >= 2")
    return n


def enabled(mode: str | None = None) -> set[str]:
    m = (mode if mode is not None else os.environ.get(ENV, "off")).strip().lower()
    if m in ("", "off", "0"):
        return set()
    out = {x.strip() for x in m.split(",") if x.strip()}
    bad = out - set(PATTERNS)
    if bad:
        raise ValueError(f"region slots: unknown pattern(s) {sorted(bad)} (off|{'|'.join(PATTERNS)})")
    return out


def is_region(kind_or_op) -> bool:
    k = kind_or_op["op"] if isinstance(kind_or_op, dict) else kind_or_op
    return isinstance(k, str) and k.startswith(PREFIX)


def _base(kind: str) -> str:
    return kind[:-4] if kind.endswith("_f16") else kind


def _ident(s: str) -> str:
    s = re.sub(r"\W", "_", s)
    return s if not s[:1].isdigit() else "_" + s


# ------------------------------------------------------------------ the spec

@dataclass
class RegionKernelSpec(KernelSpec):
    members: list = field(default_factory=list)        # member op dicts (IR order)
    params: list = field(default_factory=list)         # [(Param, role, tensor name or weight key)]

    def cuda_seed(self, lib, sub_sources: dict | None = None, parallel: bool | None = None) -> str:
        return region_seed(self, lib, sub_sources or {}, parallel)


def _member_args(m: dict, specs: dict):
    _, ps = parse_signature(specs[m["op"]].signature)
    if "sub_ops" in m:
        from .kb_proposals import bind_fused
        return bind_fused(m, specs)
    return bind(m, ps)


def region_spec(op: dict, specs: dict) -> RegionKernelSpec:
    """The spec of a region op (from its members; the kind name only identifies it)."""
    from .cuda_dtypes import cuda_signature
    from .kb_proposals import with_fused_specs
    members = op["sub_ops"]
    specs = with_fused_specs({"ops": members}, specs)
    internal = set(op.get("internal", []))
    outputs = set(op["outputs"])
    alias_out = {t for m in members if m["op"] in ALIAS_OPS for t in m["outputs"]}
    params, seen = [], {}
    for m in members:
        if m["op"] in ALIAS_OPS:
            continue
        for a in _member_args(m, specs):
            if a.kind in ("in", "out", "scratch"):
                t = a.value
                if t in alias_out or t in seen:
                    continue
                role = "out" if t in outputs else "scratch" if t in internal else "in"
                if a.kind != "in" and role == "in":
                    role = "scratch"                    # written by a member, read nowhere outside
                base = a.param.ctype.replace("const", "").strip()
                ctype = f"const {base}" if role == "in" else base
                seen[t] = len(params)
                params.append((Param(cuda_signature(ctype), f"{role}_{_ident(t)}"), role, t))
            elif a.kind == "weight":
                key = f"w:{a.value}"
                if key in seen:
                    continue
                seen[key] = len(params)
                base = a.param.ctype.replace("const", "").strip()
                params.append((Param(cuda_signature(f"const {base}"), f"w_{_ident(a.value)}"), "weight", a.value))
    # a view of an external input (e.g. h0 slices) needs its base: external alias inputs are passed as tensors
    for m in members:
        if m["op"] in ALIAS_OPS:
            src = m["inputs"][0]
            if src not in seen and src not in alias_out and not any(src in x["outputs"] for x in members):
                base = _alias_ctype(members, m, specs)
                seen[src] = len(params)
                params.append((Param(f"const {base}", f"in_{_ident(src)}"), "in", src))
    kind = op["op"]
    sig = f"void kernel_{kind}(" + ", ".join(f"{p.ctype} {p.name}" for p, _, _ in params) + ");"
    kinds = [m["op"] for m in members if m["op"] not in ALIAS_OPS]
    outs = [p.name for p, r, _ in params if r == "out"]
    scr = [p.name for p, r, _ in params if r == "scratch"]
    sem = (f"REGION slot: {len(kinds)} ops ({', '.join(dict.fromkeys(kinds))}) of the model computed by ONE entry point "
           f"(members in order: " + "; ".join(_member_line(m) for m in members) + "). "
           f"Checked outputs: {', '.join(outs)}. Scratch (harness-allocated, not checked, a fused kernel need not "
           f"write them): {', '.join(scr) or 'none'}. Every scalar of the members is a constant of this region (listed "
           "per member). You may replace the whole region by one persistent kernel (e.g. all layers / both "
           "directions / QKV->attention->projection with an on-device task queue and ready flags); own any extra "
           "scratch, flags or counters as static, lazily allocated device memory reused across calls with epoch / "
           "step tags (no per-call memset races). Co-residency: every block of a spin-waiting grid must be resident "
           "at once; size such grids with mb_sm_budget() (SMs available to this op) and the occupancy API, never "
           "assume more. A replay that hangs is killed by the harness watchdog and the candidate fails.\n" +
           "\n".join(f"[{k}] {specs[k].semantics}" for k in dict.fromkeys(kinds)))
    refs = "\n\n".join(specs[k].reference_impl for k in dict.fromkeys(kinds))
    ref = f"/* reference = the {len(kinds)} member reference kernels in order (see the semantics) */\n{refs}\n"
    spec = RegionKernelSpec(op=kind, signature=sig, semantics=sem, reference_impl=ref,
                            members=copy.deepcopy(members), params=params)
    _KNOWN[kind] = spec
    return spec


_KNOWN: dict = {}           # region kind -> spec, for every region built in this process (kb_ingest looks here)


def known_spec(kind: str):
    """The spec of a region kind built earlier in this process (exports always build it), else None."""
    return _KNOWN.get(kind)


def _alias_ctype(members, view, specs) -> str:
    """Element type of a view's base: the pointer type of the member that reads the view's output."""
    t = view["outputs"][0]
    for m in members:
        if m["op"] in ALIAS_OPS or t not in m["inputs"]:
            continue
        for a in _member_args(m, specs):
            if a.kind == "in" and a.value == t:
                return a.param.ctype.replace("const", "").replace("*", "").strip() + " *"
    return "__half *"


def _member_line(m: dict) -> str:
    sh = m.get("shape", {})
    return f"{m['name']}: {m['op']} {json.dumps(sh, sort_keys=True)[:300]}"


def seed_parallel() -> bool:
    """Region seeds run independent members concurrently (helper-thread streams, fork/join; $MB_REGION_SEED_PARALLEL,
    default on): a region must not start slower than the per-op graph with concurrent branches."""
    return os.environ.get("MB_REGION_SEED_PARALLEL", "1").strip().lower() not in ("0", "off", "false")


def _levels(members: list[dict]) -> list[int]:
    """Dependency level of each member (aliases resolve to their base's producer); -1 for alias members."""
    prod, lv = {}, []
    for i, m in enumerate(members):
        if m["op"] in ALIAS_OPS:
            p = prod.get(m["inputs"][0])
            for t in m["outputs"]:
                if p is not None:
                    prod[t] = p
            lv.append(-1)
            continue
        deps = [prod[t] for t in m["inputs"] if t in prod]
        lv.append(1 + max((lv[d] for d in deps), default=-1))
        for t in list(m["outputs"]) + list(m.get("internal", [])):
            prod[t] = i
    return lv


def region_seed(spec: RegionKernelSpec, lib, sub_sources: dict, parallel: bool | None = None) -> str:
    """Composite seed: each member's kernel (its source from ``sub_sources`` by member name, slot id (variant
    included) or kind; else the hand kernel), views as pointer arithmetic. Members are issued level by level of the
    region's dependency DAG; with ``parallel`` (default: :func:`seed_parallel`) the members of one level run
    concurrently (stage 0 on the caller's stream, the others on helper threads' per-thread streams, fork/join with
    events: works eagerly and inside a CUDA-graph capture), each concurrent member with its own copy of its source
    (own statics, like the walker's lanes)."""
    from .cuda_dtypes import cuda_signature
    from .kb_export import _c_scalar, namespaced
    from .kb_proposals import _all_specs, _par_calls, with_fused_specs
    specs = with_fused_specs({"ops": spec.members}, _all_specs())
    parallel = seed_parallel() if parallel is None else parallel
    pname = {v if r != "weight" else f"w:{v}": p.name for p, r, v in spec.params}
    lv = _levels(spec.members)
    width = {}
    for x in lv:
        if x >= 0:
            width[x] = width.get(x, 0) + 1
    blocks, includes, ns_of, hand = [], [], {}, set()
    local = {}                                         # alias tensor -> C expression
    by_level: dict[int, list[str]] = {}
    lane_of_src: dict = {}
    for i, m in enumerate(spec.members):
        if m["op"] in ALIAS_OPS:
            src = m["inputs"][0]
            base = local.get(src, pname.get(src))
            if m["op"] == "view":
                local[m["outputs"][0]] = f"({base} + {int(m.get('offset', 0))})"
            else:                                      # chunk2_c1: outputs k at k * c_each * hw
                if "hw" not in m.get("shape", {}):
                    raise ValueError("chunk2_c1 inside a region needs shape.hw")
                for k, t in enumerate(m["outputs"]):
                    local[t] = f"({base} + {k * int(m['shape']['c_each']) * int(m['shape']['hw'])})"
            continue
        kind = m["op"]
        entry, _ = parse_signature(specs[kind].signature)
        src = None
        for key in (m["name"], _sid(m, specs), kind):
            if key in sub_sources:
                src = sub_sources[key]
                break
        if src is None and ("sub_ops" in m or kind.startswith("fused__")) and hasattr(specs.get(kind), "cuda_seed"):
            # a fused member with no cached source (no kernel round has produced a winner for its kind yet, e.g. a graph-first
            # round, and a region op the cache has never seen): the member's own stage-by-stage seed, its sub-ops' sources
            # where sub_sources has them else the hand kernels -- the seed the non-region fused slot path (kb_export) builds
            fs = specs[kind]
            subs = {k: sub_sources[k] for k in getattr(fs, "sub_ops", ()) if k in sub_sources}
            src = fs.cuda_seed(lib.of(kind) if hasattr(lib, "of") else lib, sub_sources=subs)
        if src is not None:
            # concurrent members of one level never share a copy (statics); lane k = k-th member of the level
            # using this source, copies reused across levels
            lane = 0
            if parallel and width[lv[i]] > 1:
                used = lane_of_src.setdefault((lv[i], src), 0)
                lane, lane_of_src[(lv[i], src)] = used, used + 1
            key = (src, lane)
            ns = ns_of.get(key)
            if ns is None:
                ns = ns_of[key] = f"member{len(ns_of)}"
                inc, body = namespaced(ns, src)
                includes += [x for x in inc if x not in includes]
                blocks.append(f"// current kernel of {kind} ({m['name']}{', lane %d' % lane if lane else ''})\n{body}")
            callee = f"{ns}::{entry}"
        elif "sub_ops" in m or kind.startswith("fused__"):
            raise ValueError(f"{m['name']}: a fused member needs its source in sub_sources")
        else:
            hand.add(entry)
            callee = entry
        args = []
        for a in _member_args(m, specs):
            if a.kind in ("in", "out", "scratch"):
                args.append(local.get(a.value) or pname[a.value])
            elif a.kind == "weight":
                args.append(pname[f"w:{a.value}"])
            elif a.kind == "null":
                args.append("nullptr")
            else:
                args.append(_c_scalar(a.param.base_type, a.value))
        by_level.setdefault(lv[i], []).append(f"    {callee}({', '.join(args)});  // {m['name']}")
    if hand:
        names = {n for e in hand for n in lib.closure(e)}
        blocks.insert(0, "\n\n".join(lib.items[k].text for k in lib.items if k in names))
    lines, n_calls, conc = [], 0, False
    for lvl in sorted(by_level):
        calls = by_level[lvl]
        n_calls += len(calls)
        if parallel and len(calls) > 1:
            conc = True
            lines.append("    {")
            lines += ["    " + c for c in _par_calls(calls)]
            lines.append("    }")
        else:
            lines += calls
    if conc:
        includes += [x for x in ("#include <thread>", "#include <cuda_runtime.h>") if x not in includes]
    entry, params = parse_signature(cuda_signature(spec.signature))
    plist = ", ".join(f"{p.ctype} {p.name}" for p in params)
    note = (f"// Seed: the region's {n_calls} member ops with their current kernels, level by level of the region's "
            f"dependency DAG\n// ({'independent members of a level run concurrently on helper-thread streams; ' if conc else ''}"
            f"parity with the per-op graph).\n// One persistent kernel may compute the whole region; scratch_* need "
            f"not be written.\n")
    return ("\n".join(includes) + ("\n\n" if includes else "") + "\n\n".join(blocks) + "\n\n" + note +
            f"void {entry}({plist}) {{\n" + "\n".join(lines) + "\n}\n")


def _sid(m, specs):
    from .kb_export import slot_id_of
    try:
        return slot_id_of(m, specs)
    except Exception:
        return m["op"]


def bind_region(op: dict, specs: dict) -> list:
    """kernel_abi Args of a region op in its signature's order (in / out / scratch / weight)."""
    from .kernel_abi import Arg
    spec = specs.get(op["op"]) or region_spec(op, specs)
    kinds = {"in": "in", "out": "out", "scratch": "scratch", "weight": "weight"}
    return [Arg(kinds[r], v, p) for p, r, v in spec.params]


def with_region_specs(graph: dict, specs: dict) -> dict:
    """``specs`` plus a RegionKernelSpec for every region op of ``graph``."""
    add = {o["op"]: region_spec(o, specs) for o in graph["ops"] if is_region(o) and o["op"] not in specs}
    return {**specs, **add} if add else specs


# ------------------------------------------------------------------ building regions

def _graph_outputs(g):
    return g["output"].get("tensors") or [g["output"]["tensor"]]


def region_kind(members: list[dict], pattern: str) -> str:
    """``region__<pattern>__<sha10>[_f16]``: the suffix follows the members' storage dtype (cuda_dtypes.op_dtype)."""
    sig = json.dumps([[m["op"], m.get("shape", {}), m.get("offset"), m["inputs"], m["outputs"]] for m in members],
                     sort_keys=True)
    from .cuda_dtypes import FP16, op_dtype
    f16 = all(op_dtype(m["op"]) == FP16 for m in members if m["op"] not in ALIAS_OPS)
    return f"{PREFIX}{pattern}__{hashlib.sha1(sig.encode()).hexdigest()[:10]}" + ("_f16" if f16 else "")


def region_class(kind: str, recurrent: bool = False) -> str:
    """Op class of a region kind (op_classes): by its pattern."""
    pat = kind[len(PREFIX):].split("__")[0]
    return {"rnn_stack": "recurrent" if recurrent else "gemm", "attention_block": "attention"}.get(pat, "other")


def fuse_region(graph: dict, names: list[str], specs: dict, pattern: str = "ops") -> tuple[dict, RegionKernelSpec]:
    """(new graph, spec) with the ops ``names`` replaced by one region op at the last member's position.

    Refused (ValueError) when: < 2 non-alias members; a member is already a region; an op between the first and the
    last member that is not a member reads a member's output (it would run before the region); an alias member's
    output is read outside the region or returned (aliases inside a region are pointer arithmetic only)."""
    g = copy.deepcopy(graph)
    ops = g["ops"]
    pos = {o["name"]: i for i, o in enumerate(ops)}
    idx = sorted(pos[n] for n in dict.fromkeys(names))
    mem = [ops[i] for i in idx]
    cap = max_members()
    if sum(m["op"] not in ALIAS_OPS for m in mem) < 2 or len(mem) > cap:
        raise ValueError(f"region needs 2..{cap} ops")
    if any(is_region(m) for m in mem):
        raise ValueError("nested regions")
    mset = set(idx)
    from . import graph_prune
    dead = set(graph_prune.dead_ops(g))                  # dead readers (never run) do not make an output
    produced = {t for m in mem for t in list(m["outputs"]) + list(m.get("internal", []))}
    for j in range(idx[0], idx[-1]):
        if j not in mset and j not in dead and produced & set(ops[j]["inputs"]):
            raise ValueError(f"{ops[j]['name']} reads a region output before the region ends")
    outside_reads = {t for j, o in enumerate(ops) if j not in mset and j not in dead for t in o["inputs"]}
    gouts = set(_graph_outputs(g))
    alias_out = {t for m in mem if m["op"] in ALIAS_OPS for t in m["outputs"]}
    if alias_out & (outside_reads | gouts):
        raise ValueError("an alias inside the region is read outside it")
    outs = [t for m in mem for t in m["outputs"] if t not in alias_out and (t in outside_reads or t in gouts)]
    internal = [t for m in mem for t in list(m["outputs"]) + list(m.get("internal", []))
                if t not in alias_out and t not in outs]
    ins = list(dict.fromkeys(t for m in mem for t in m["inputs"] if t not in produced))
    members = []
    for m in mem:
        c = copy.deepcopy(m)
        for k in ("hint", "depends_on", "dispatch_id"):
            c.pop(k, None)
        members.append(c)
    kind = region_kind(members, pattern)
    rop = {"name": f"region:{mem[0]['name']}..{mem[-1]['name']}", "op": kind, "region": True, "pattern": pattern,
           "inputs": ins, "outputs": outs, "internal": internal, "sub_ops": members,
           "shape": {f"m{i}": m.get("shape", {}) for i, m in enumerate(members)}}
    # Dead non-member ops between the members that read a member's output (or each other's) must not stay in front
    # of the region: the region produces those tensors at ITS position, and the walker (walk_torch.walk) still runs
    # every IR op in list order (dead ones included) => KeyError on the tensor (49_Mamba2: dead getitem_3 reads
    # einsum_1; 35/37/38/40/42: dead h_n/c_n readers). They move behind the region (the walk's env then holds the
    # region's internals; the export prunes them anyway). MB_REGION_DEAD_MOVE=0: legacy behaviour.
    moved: list[int] = []
    if os.environ.get("MB_REGION_DEAD_MOVE", "1") != "0":
        avail = set(produced)
        for j in range(idx[0], idx[-1]):
            if j not in mset and j in dead and avail & set(ops[j]["inputs"]):
                moved.append(j)
                avail.update(ops[j]["outputs"])
    mv = set(moved)
    new_ops = []
    for j, o in enumerate(ops):
        if (j in mset and j != idx[-1]) or j in mv:
            continue
        new_ops.append(o)
        if j == idx[-1]:
            new_ops[-1] = rop
            new_ops.extend(ops[k] for k in moved)
    g["ops"] = new_ops
    late = _read_before_defined(g)
    if late:                                              # never hand the walker/exporter a mis-ordered IR
        raise ValueError(f"region would leave {late[0][0]} reading {late[0][1]} before it is produced")
    return g, region_spec(rop, specs)


def _read_before_defined(g: dict) -> list[tuple[str, str]]:
    """(op, tensor) pairs where an op (members of fused/region ops included) reads a tensor that some LATER op
    defines: the order walk_torch.walk executes (list order, sub_ops recursed)."""
    produced: dict[str, int] = {}

    def outs(o):
        yield from o.get("outputs", [])
        yield from o.get("internal", [])
        for s in o.get("sub_ops", []):
            yield from outs(s)
    for i, o in enumerate(g["ops"]):
        for t in outs(o):
            produced.setdefault(t, i)
    bad = []
    for i, o in enumerate(g["ops"]):
        for t in o.get("inputs", []):
            if produced.get(t, -1) > i:
                bad.append((o["name"], t))
    return bad


def find_regions(graph: dict, pattern: str, specs: dict) -> list[list[str]]:
    """Member name lists for a pattern (dead ops never included).

    rnn_stack        every live op from the first to the last recurrent op (gru_seq / lstm_seq / lstm, sibling-fused
                     ones included) that is a recurrent op, a concat / copy between them, or a view they read:
                     the whole stack incl. the per-layer concats (41: 12 GRUs + 6 concats + 12 h0 views)
    attention_block  from the first GEMM (linear / matmul / bmm) before an attention core (attention / softmax) to
                     the first linear after it, everything in between (43: QKV -> ... -> projection)
    all              every live non-alias op plus the views they read (one megakernel for the model)"""
    from . import graph_prune
    ops = graph["ops"]
    dead = set(graph_prune.dead_ops(graph))
    live = [i for i in range(len(ops)) if i not in dead]

    def base(o):
        k = o["op"]
        if k.startswith("fused__"):
            parts = [p for p in k.split("__")[1:] if p and p != "par"]
            k = parts[0] if parts else k
        return _base(k)

    sel: list[int] = []
    if pattern == "rnn_stack":
        rec = [i for i in live if base(ops[i]) in RECURRENT]
        if len(rec) < 2:
            return []
        lo, hi = rec[0], rec[-1]
        # extend to the concat right after the last recurrence if it only joins recurrent outputs
        for j in range(hi + 1, len(ops)):
            if j in dead:
                continue
            if base(ops[j]) in ("catn", "cat2_c1") and all(
                    any(t in ops[i]["outputs"] for i in rec) for t in ops[j]["inputs"]):
                hi = j
            break
        sel = [i for i in range(lo, hi + 1) if i not in dead and (
            base(ops[i]) in RECURRENT + ("catn", "cat2_c1", "strided_copy") or ops[i]["op"] in ALIAS_OPS)]
    elif pattern == "attention_block":
        core = [i for i in live if base(ops[i]) in ("attention", "softmax", "sdpa")]
        if not core:
            return []
        gemm = ("linear", "matmul", "bmm", "bmm_strided")
        lo = max((i for i in live if i < core[0] and base(ops[i]) in gemm and base(ops[i]) == "linear"),
                 default=None)
        hi = min((i for i in live if i > core[-1] and base(ops[i]) == "linear"), default=None)
        if lo is None or hi is None:
            return []
        sel = [i for i in range(lo, hi + 1) if i not in dead]
    elif pattern == "all":
        sel = list(live)
    else:
        raise ValueError(pattern)
    # views read by members but defined before the range come along (h0 slices); alias ops outside the members'
    # reach are dropped
    want = {t for i in sel for t in ops[i]["inputs"]}
    views = [i for i in live if ops[i]["op"] in ALIAS_OPS and i < (sel[0] if sel else 0)
             and set(ops[i]["outputs"]) & want and not (set(ops[i]["inputs"]) & {
                 t for j in sel for t in ops[j]["outputs"]})]
    sel = sorted(set(sel) | set(views))
    # a view at the front whose base comes from outside stays outside (pointer to an external tensor is fine
    # either way); keep views only if they are read by members
    sel = [i for i in sel if ops[i]["op"] not in ALIAS_OPS or set(ops[i]["outputs"]) & want]
    if sum(ops[i]["op"] not in ALIAS_OPS for i in sel) < 2:
        return []
    names = [ops[i]["name"] for i in sel]
    try:
        fuse_region(graph, names, specs, pattern)
    except (ValueError, KeyError) as e:
        return []
    return [names]


def apply_regions(graph: dict, patterns, specs: dict) -> tuple[dict, dict, list[dict]]:
    """Apply every pattern in order: (graph, added specs, [{pattern, kind, members}])."""
    g, added, made = graph, {}, []
    for p in patterns:
        for names in find_regions(g, p, {**specs, **added}):
            g, spec = fuse_region(g, names, {**specs, **added}, p)
            added[spec.op] = spec
            made.append({"pattern": p, "kind": spec.op, "members": names})
    return g, added, made


def seed_sources(region_op: dict, sources: dict, specs: dict) -> dict:
    """{member name: source} for a region seed from a {slot id / kind: source} map (e.g. a cache plan's sources):
    each member gets its own variant's source when present, else its base slot's / kind's."""
    from .kb_export import base_slot_id
    out = {}
    for m in region_op["sub_ops"]:
        if m["op"] in ALIAS_OPS:
            continue
        for key in (_sid(m, specs), base_slot_id(m["op"], specs) if m["op"] in specs else None, m["op"]):
            if key and key in sources:
                out[m["name"]] = sources[key]
                break
    return out
