"""Fusion beyond producer -> elementwise (``--fusion-extended off|siblings|epilogue|attention|all``).

The flow's fusion proposals (``graph_hints.propose_from_profile``) only merge a producer with the elementwise op
that is the only reader of its output. The CUDA-only baselines won with three other kinds of fusion
(l3_farm/analysis/cuda_vs_flow.md), added here behind one flag, each decided by the unchanged measure-and-accept
loop (``kb_graph_opt``: original vs variant problem, same KernelBlaster budget, paired re-timing, ``accept_gain``):

* ``siblings`` -- independent ops of one kind that read the same activation (the two directions of a bidirectional
  RNN layer, Q/K/V projections of one input) become one ``fused__par__<k>__<k>...`` op (``kb_proposals``
  ``find_siblings`` / ``fuse_siblings`` / ``make_par_spec``): every stage's parameters with its stage prefix, every
  output checked, no intermediates. Hint kind ``fuse_siblings`` (id ``fuse_siblings:<fused kind>``, ``groups`` of
  original op names). Reference semantics: each stage's reference on its own arguments (the torch walker runs the
  stages in order; independent, so the order is irrelevant); per-slot check = every stage output, model-level check
  unchanged.
* ``epilogue`` -- a pool (max / avg / adaptive-avg) as the epilogue of a conv / GEMM, directly or through one
  elementwise op (conv -> relu -> maxpool): ordinary ``fuse`` chain hints (the chain machinery is generic; the
  pool stage reads the previous stage's output through its input parameter). Reference: the stages in sequence.
* ``attention`` -- bmm -> scale -> causal mask -> softmax|relu -> bmm (and the copies around q/k/v/o) become one
  ``attention_f16`` op with an explicit ``causal`` flag (``attention_fuse``, from the variant-discovery prototype;
  the mask must be proven ``triu(1)`` on its constant values at proposal time). Hint kind ``fuse_attention``.
  Reference: exact fp32 attention rounded once (an ``approx`` rewrite: ``verify_rewrite`` compares it to the
  original walk within the fp16 output tolerance); the variant starts from a composite seed (``composite_seed``).

**Per-shape inheritance** (any mode but off): a fused slot's seed per instance is the composition of that
instance's best stage kernels and the cache keeps per-shape fused winners. ``prepare_hint`` builds the variant
with the cache's coverage split (``cache_plan(split=True)``): a fused slot whose first stage has per-shape winners
is split into one slot per covering winner (``slot_variant``), each seeded with its own stage kernels
(``FusedKernelSpec.cuda_seed`` with that group's ``sub_sources``), and an accepted variant ingests every split
slot guarded on its own shapes. Structural kinds (attention) get per-shape composite seeds (one guarded branch per
instance shape). The sweep's 11_VGG16 conv+ReLU rejection (one fresh fused kernel for 9 shapes) is that case.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

MODES = ("off", "siblings", "epilogue", "attention", "all")
HINT_KINDS = {"siblings": "fuse_siblings", "attention": "fuse_attention"}
POOL_KINDS = ("maxpool2d", "maxpool2d_ex", "avgpool2d", "adaptive_avg_pool2d")


def enabled(mode: str | None) -> set[str]:
    m = (mode or "off").strip().lower()
    out: set[str] = set()
    for part in m.split(","):
        part = part.strip()
        if part not in MODES:
            raise ValueError(f"--fusion-extended: {part!r} not in {'|'.join(MODES)}")
        if part == "all":
            out |= {"siblings", "epilogue", "attention"}
        elif part != "off":
            out.add(part)
    return out


def decide_kinds(mode: str | None) -> tuple[str, ...]:
    """Hint kinds (besides fuse / split_slot) the decide step considers."""
    en = enabled(mode)
    return tuple(HINT_KINDS[k] for k in ("siblings", "attention") if k in en)


def inherit(mode: str | None) -> bool:
    """Fused variants keep per-shape stage kernels (cache coverage split) whenever an extended mode is on."""
    return bool(enabled(mode))


def _base(kind: str) -> str:
    for suf in ("_f16", "_s8_pc", "_s8", "_i8"):
        if kind.endswith(suf):
            return kind[: -len(suf)]
    return kind


# ---------------------------------------------------------------- proposals

def siblings_hint(kind: str, groups: list[list[str]], source: dict, status: str = "proposed") -> dict:
    return {"id": f"fuse_siblings:{kind}", "kind": "fuse_siblings", "target": "cuda", "fused_kind": kind,
            "groups": groups, "status": status, "source": [source], "evidence": []}


def propose_siblings(graph: dict, op_us: dict, specs: dict, *, total_us: float, min_gain_pct: float,
                     ir: str | None, done: set) -> list[dict]:
    from .graph_hints import _now, unique_names
    from .kb_proposals import find_siblings, fuse_siblings

    g = unique_names(copy.deepcopy(graph))
    by_kind: dict[str, dict] = {}
    for grp in find_siblings(g, specs):
        try:
            _, spec = fuse_siblings(g, grp, specs)
        except (ValueError, StopIteration, KeyError):
            continue
        ts = [op_us.get(n, 0.0) for n in grp]
        c = by_kind.setdefault(spec.op, {"groups": [], "gain_us": 0.0})
        c["groups"].append(grp)
        c["gain_us"] += sum(ts) - max(ts, default=0.0)        # what running them as one concurrent unit saves
    out = []
    for kind, c in sorted(by_kind.items(), key=lambda kv: -kv[1]["gain_us"]):
        pct = 100.0 * c["gain_us"] / (total_us or 1.0)
        if pct < min_gain_pct or f"fuse_siblings:{kind}" in done:
            continue
        out.append(siblings_hint(kind, c["groups"], {"by": "mb-ext", "ir": ir, "est_gain_us": round(c["gain_us"], 2),
                                                     "est_gain_pct": round(pct, 2), "ts": _now()}))
    return out


def propose_epilogue(graph: dict, op_us: dict, specs: dict, *, total_us: float, min_gain_pct: float,
                     ir: str | None, done: set, launch_us: float = 0.0) -> list[dict]:
    """conv/GEMM [-> elementwise] -> pool chains whose intermediates are read only by the next stage."""
    from .graph_hints import _now, atoms, fuse_hint, unique_names
    from .kb_proposals import fuse_chain, with_fused_specs
    from .op_classes import op_class

    g = unique_names(copy.deepcopy(graph))
    specs = with_fused_specs(g, specs)
    outs = set(g["output"].get("tensors") or [g["output"]["tensor"]])
    readers: dict[str, list[dict]] = {}
    for o in g["ops"]:
        for t in o["inputs"]:
            readers.setdefault(t, []).append(o)

    def only_reader(o):
        if len(o["outputs"]) != 1 or o["outputs"][0] in outs:
            return None
        r = readers.get(o["outputs"][0], [])
        return r[0] if len(r) == 1 and r[0]["op"] in specs else None

    cands: dict[str, dict] = {}
    used: set[str] = set()
    for a in g["ops"]:
        if a["op"] not in specs or op_class(a["op"]) not in ("conv", "gemm") or a["name"] in used:
            continue
        b = only_reader(a)
        chain = None
        if b is not None and _base(b["op"]) in POOL_KINDS:
            chain = [a, b]
        elif b is not None and op_class(b["op"]) == "elementwise" and "sub_ops" not in b:
            c = only_reader(b)
            if c is not None and _base(c["op"]) in POOL_KINDS:
                chain = [a, b, c]
        if chain is None or any(x["name"] in used for x in chain):
            continue
        try:
            _, spec = fuse_chain(g, [x["name"] for x in chain], specs)
        except (ValueError, StopIteration, KeyError):
            continue
        if f"fuse:{spec.op}" in done:
            continue
        used.update(x["name"] for x in chain)
        k = cands.setdefault(spec.op, {"chains": [], "gain_us": 0.0})
        k["chains"].append([n for x in chain for n in atoms(x["name"])])
        # the epilogue stages' own time plus their launches (the producer's work stays)
        k["gain_us"] += sum(op_us.get(x["name"], 0.0) + launch_us for x in chain[1:])
    out = []
    for kind, c in sorted(cands.items(), key=lambda kv: -kv[1]["gain_us"]):
        pct = 100.0 * c["gain_us"] / (total_us or 1.0)
        if pct < min_gain_pct:
            continue
        out.append(fuse_hint(kind, c["chains"], {"by": "mb-ext-epilogue", "ir": ir, "est_gain_us": round(c["gain_us"], 2),
                                                 "est_gain_pct": round(pct, 2), "ts": _now()}))
    return out


def propose(graph: dict, profile: dict, specs: dict, doc: dict, *, mode: str, min_gain_pct: float,
            ir: str | None = None) -> list[dict]:
    """Every enabled proposer on one IR's profile -> new hints (not merged; the caller merges)."""
    from .graph_hints import decided

    en = enabled(mode)
    if not en:
        return []
    op_us = {o["name"]: o.get("us") or 0.0 for o in profile.get("ops", [])}
    total = profile.get("e2e_us") or sum(op_us.values()) or 1.0
    done = decided(doc) if doc else set()
    out = []
    if "siblings" in en:
        out += propose_siblings(graph, op_us, specs, total_us=total, min_gain_pct=min_gain_pct, ir=ir, done=done)
    if "epilogue" in en:
        out += propose_epilogue(graph, op_us, specs, total_us=total, min_gain_pct=min_gain_pct, ir=ir, done=done,
                                launch_us=profile.get("launch_us") or 0.0)
    if "attention" in en:
        from .attention_fuse import const_lookup_for
        from .attention_fuse import propose as attn_propose
        cl = const_lookup_for(Path(profile["ir_dir"])) if profile.get("ir_dir") else None
        out += [h for h in attn_propose(graph, op_us, const_lookup=cl, total_us=total, min_gain_pct=min_gain_pct,
                                        ir=ir) if h["id"] not in done]
    return out


# ---------------------------------------------------------------- decide-step helpers

def scope_ops(h: dict, graph: dict, specs: dict) -> list[str]:
    """Current op names an extended hint replaces in ``graph`` (by applying it)."""
    from . import graph_hints as gh
    try:
        g2, _, prov = gh.apply_hints(graph, [h], specs, statuses=None)
    except Exception:  # noqa: BLE001
        return []
    if not prov["applied"]:
        return []
    keep = {o["name"] for o in g2["ops"]}
    return [o["name"] for o in graph["ops"] if o["name"] not in keep]


def new_kinds(h: dict, graph: dict, specs: dict) -> set[str]:
    from . import graph_hints as gh
    try:
        g2, _, prov = gh.apply_hints(graph, [h], specs, statuses=None)
    except Exception:  # noqa: BLE001
        return set()
    old = {o["name"] for o in graph["ops"]}
    return {o["op"] for o in g2["ops"] if o["name"] not in old}


def variant_seeds(h: dict, g0: dict, vplan: dict, base: dict, seeds: dict, tags: dict, specs: dict, *,
                  cap_mb: int = 8192) -> dict:
    """Composite seeds (the incumbent's kernels back to back, one guarded branch per instance shape) for a structural
    variant's new slots (attention_f16), in ``seeds`` / ``tags`` in place -> {slot: report}."""
    if h.get("kind") != "fuse_attention":
        return {}
    from .composite_seed import composite_sources
    from .kb_export import DEFAULT_HAND_CU, CudaLibrary, _hand_source, _problem_dtype

    dtype = _problem_dtype([o["op"] for o in g0["ops"] if o["op"] not in ("view", "chunk2_c1")])
    lib = CudaLibrary(_hand_source(DEFAULT_HAND_CU, dtype))
    srcs, crep = composite_sources(g0, vplan["graph"], specs, sources=base["sources"], sub_sources=base["sub_sources"],
                                   lib=lib, verified_consts=set(h.get("verified_masks") or []),
                                   cap_bytes=int(cap_mb) << 20)
    rep = {}
    for sid, r in crep.items():
        if sid in vplan["sources"]:
            rep[sid] = {"seed": "cache", **r}
            continue
        if sid in srcs:
            seeds[sid] = srcs[sid]
            tags[sid] = ("composite:" + "+".join(r["stages"][0]))[:200] if r["stages"] else "composite"
        rep[sid] = {"seed": "composite" if sid in srcs else "hand", **r}
    return rep


def note_composite(problem: Path, rep: dict) -> None:
    """The agent's task statement for composite-seeded slots (manifest semantics)."""
    from .composite_seed import semantics_note
    f = Path(problem) / "manifest.json"
    man = json.loads(f.read_text())
    for s in man["slots"]:
        r = rep.get(s["id"])
        if r and r.get("seed") == "composite" and r.get("kind"):
            s["semantics"] = (s.get("semantics") or "") + semantics_note(r["kind"], r)
    f.write_text(json.dumps(man, indent=1))
