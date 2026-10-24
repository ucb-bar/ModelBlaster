"""Unified graph hints (``modelblaster.graph_hints/v1``): one schema and one
apply path for every graph-level change ModelBlaster makes to an IR.

Every producer of graph changes writes the same hint records:

* KernelBlaster proposals (``kb-proposal/1``: fuse, split)   -> from_kb_proposals
* ModelBlaster's own cost-driven proposals (whole-model
  per-op profile)                                            -> propose_from_profile
* ModelBlaster's kernel cache (shape-guarded winners that
  cover only some instances of a slot)                        -> kb_ingest.cache_plan
* XPU-RT's RISC-V hints (fusion_hints/v1, split_hints/v1,
  shard_hints/v1, unfuse_hints/v1)                           -> from_xpurt
* the Phase 1 fusion registry (mb-fusion-registry/1)         -> load() reads it

A hints file::

    {"format": "modelblaster.graph_hints/v1",
     "hints": [
       {"id": "fuse:fused__conv2d__batchnorm2d", "kind": "fuse", "target": "cuda",
        "fused_kind": "fused__conv2d__batchnorm2d",
        "chains": [["conv1", "bn1"]],          # optional scope (original op names);
                                               # absent = everywhere the pattern matches
        "status": "proposed|accepted|rejected",
        "source": [{"by": "kb|mb-cost|mb-cache|xpurt|user", ...}],
        "evidence": [{"problem", "ir", "original_us", "variant_us", "gain", "decision", "ts"}]},
       {"id": "split:conv2d", "kind": "split_slot", "target": "cuda", "slot": "conv2d",
        "by": "shape_bucket", "groups": [{"bucket": "conv:N1,...", "ops": [...]}], ...},
       {"id": "unfuse:fused__linear__relu", "kind": "unfuse", "target": "cuda",
        "fused_kind": "...", "ops": [...]},
       {"id": "xpurt:fuse:mlp_control", "kind": "fuse", "target": "riscv", "network": "...",
        "dispatch_groups": [[0, 1, 2]]},
       {"id": "xpurt:tile_split:...", "kind": "tile_split", "target": "riscv", "split_ops": [...]},
       {"id": "xpurt:shard:...", "kind": "shard", "target": "riscv", "shard_ops": [...]},
       ...]}

The id is the hint's identity across rounds and models: a fusion is decided
per fused kind (``fuse:<kind>``), a split per slot (``split:<slot>``), so the
evidence from every problem that tried it accumulates on one record and the
status is the majority of its decisions.

:func:`apply_hints` rewrites an IR for one target and records provenance in
``graph["graph_hints"]`` (applied hints, what each became, what was skipped
and why); fused ops carry ``hint``. On CUDA:

* ``fuse`` -> a fused op (``kb_proposals.fuse_chain``; chains of any length,
  fused ops extend by fusing with their consumer);
* ``split_slot`` -> ``slot_variant`` annotations: instances of one C entry are
  given shape-specialised slots (``kb_export`` exports one slot per variant,
  the kernel cache's shape guard keys each variant's winner). The graph's
  ops and tensors are unchanged;
* ``unfuse`` -> the fused op's stages restored (fp32 sub_ops keep their
  registered tensors, so nothing needs synthesizing);
* the RISC-V kinds: ``fuse`` with ``dispatch_groups`` is translated to op-name
  chains (each group must be an exclusive producer->consumer chain); the
  others are skipped with the reason -- ``tile_split`` (OC/N tiles placed on
  several cores) and ``shard`` (an RVV codegen width for composite-hart
  dispatch) parallelise one op across cores, which a CUDA kernel already
  does across SMs inside the kernel; their CUDA analogue for heterogeneous
  shapes is ``split_slot``.

With ``target="riscv"`` the RISC-V kinds go through the existing appliers
(``apply_fusion_hint``, ``apply_split_hint``, ``apply_shard_hint``,
``apply_unfuse_hint``) unchanged.

:func:`verify_rewrite` walks the rewritten IR with the torch walker (which
runs fused ops' stages) against the original golden and the original walk,
and optionally builds and runs the whole-graph driver with seed kernels.

    python -m modelblaster.pipeline.graph_hints apply IR_DIR HINTS.json OUT_IR_DIR [--verify] [--build --arch 75]
    python -m modelblaster.pipeline.graph_hints import {kb,xpurt,registry} FILE... --into HINTS.json
    python -m modelblaster.pipeline.graph_hints show HINTS.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

FORMAT = "modelblaster.graph_hints/v1"
CUDA_KINDS = ("fuse", "split_slot", "unfuse")
NOT_ON_CUDA = {
    "tile_split": "OC/N tiles exist to run one op on several cores at once; a CUDA kernel already spans "
                  "every SM and the driver runs one stream, so tiles would serialise (use split_slot for "
                  "shape-specialised kernels)",
    "shard": "shard_factor is an RVV codegen width for composite-hart dispatch; CUDA parallelism is inside "
             "the kernel",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- documents

def new_doc() -> dict:
    return {"format": FORMAT, "hints": []}


def load(path: Path | None) -> dict:
    """A hints document; a missing file is empty, a Phase 1 fusion registry
    (``mb-fusion-registry/1``) is converted."""
    if path is None or not Path(path).exists():
        return new_doc()
    d = json.loads(Path(path).read_text())
    if d.get("format") == FORMAT:
        return d
    if d.get("format") == "mb-fusion-registry/1":
        doc = new_doc()
        for kind, e in d.get("fusions", {}).items():
            doc["hints"].append({"id": f"fuse:{kind}", "kind": "fuse", "target": "cuda", "fused_kind": kind,
                                 "status": e.get("status", "proposed"), "signature": e.get("signature"),
                                 "source": [{"by": "registry", "ref": str(path)}],
                                 "evidence": list(e.get("evidence", []))})
        return doc
    raise ValueError(f"{path}: not {FORMAT} (format={d.get('format')!r})")


def save(doc: dict, path: Path) -> None:
    """Write atomically (temp file + rename): a reader never sees a partial
    document, and two writers leave one complete document, not a mix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, path)


def get(doc: dict, hid: str) -> dict | None:
    return next((h for h in doc["hints"] if h["id"] == hid), None)


def merge(doc: dict, hints: list[dict]) -> list[dict]:
    """Add new hints; for known ids add the new sources and scope (a decided
    status is kept). -> the hints that were new."""
    added = []
    for h in hints:
        cur = get(doc, h["id"])
        if cur is None:
            doc["hints"].append(copy.deepcopy(h))
            added.append(h)
            continue
        cur.setdefault("source", []).extend(s for s in h.get("source", []) if s not in cur["source"])
        for key in ("chains", "groups"):
            if key in h and cur.get(key) is not None:
                cur[key] = cur[key] + [c for c in h[key] if c not in cur[key]]
    return added


def record_decision(doc: dict, hint: dict | str, evidence: dict) -> dict:
    """Append a decision to a hint (added if unknown); its status becomes the
    majority of its decisions (a tie rejects)."""
    hid = hint if isinstance(hint, str) else hint["id"]
    h = get(doc, hid)
    if h is None:
        if isinstance(hint, str):
            raise KeyError(f"unknown hint {hid}")
        h = copy.deepcopy(hint)
        doc["hints"].append(h)
    h.setdefault("evidence", []).append(evidence)
    wins = sum(1 for x in h["evidence"] if x["decision"] == "accepted")
    h["status"] = "accepted" if wins > len(h["evidence"]) - wins else "rejected"
    if h["status"] == "accepted":
        # an accepted change applies wherever its pattern matches (split: by
        # shape bucket); the instances it was measured on stay as evidence
        for key in ("chains", "groups"):
            if key in h:
                h.setdefault("tried_" + key, []).extend(x for x in h.pop(key) if x not in h.get("tried_" + key, []))
    return h


def decided(doc: dict) -> set[str]:
    return {h["id"] for h in doc["hints"] if h.get("status") in ("accepted", "rejected")}


# ---------------------------------------------------------------- shape buckets

# Same bucketing as KernelBlaster's knowledge/impl_store.shape_bucket (kept
# identical: split proposals name their groups by these keys): powers of 4 of
# the work dimensions, else of the element count.
_BUCKET_KEYS = {
    "gemm": ("M", "N", "K", "batch"),
    "conv": ("N", "IC", "OC", "OH", "OW", "KH", "KW", "G"),
    "attention": ("BH", "S", "D"),
}


def _mag(v) -> str:
    v = max(int(v), 1)
    return f"{4 ** int(math.log(v, 4)):d}"


def shape_bucket(op_class: str, shape: dict) -> str:
    if isinstance(shape.get("a"), dict):                 # fused op: its first stage's work
        shape = shape["a"]
    keys = _BUCKET_KEYS.get(op_class)
    if keys:
        parts = [f"{k}{_mag(shape[k])}" for k in keys if k in shape]
        if parts:
            return op_class + ":" + ",".join(parts)
    n = shape.get("n")
    if n is None:
        ints = [v for v in shape.values() if isinstance(v, int) and v > 0]
        n = math.prod(ints) if ints else 1
    return f"{op_class}:n{_mag(n)}"


# ---------------------------------------------------------------- producers

def _graph_of_problem(problem_dir: Path) -> dict:
    man = json.loads((Path(problem_dir) / "manifest.json").read_text())
    return json.loads((Path(man["source_ir"]) / "graph.json").read_text())


def unique_names(graph: dict) -> dict:
    """Make op names unique in place: the k-th repeat of a name becomes
    ``name#k``. Op names are the handle of every hint, proposal and manifest
    op, but a module called twice (one ``nn.ReLU`` used after two convs)
    gives two ops one name; tensor names stay as they are. Deterministic and
    idempotent, so every reader of an IR agrees on the names."""
    seen: dict[str, int] = {}
    taken = {o["name"] for o in graph["ops"]}
    for o in graph["ops"]:
        n = o["name"]
        k = seen.get(n, 0)
        seen[n] = k + 1
        if k:
            while f"{n}#{k}" in taken:
                k += 1
            o["name"] = f"{n}#{k}"
            taken.add(o["name"])
    return graph


def atoms(name: str) -> list[str]:
    """Original op names inside a (possibly fused, ``a+b+c``) op name."""
    return name.split("+")


def fuse_hint(kind: str, chains: list[list[str]] | None, source: dict, status: str = "proposed") -> dict:
    h = {"id": f"fuse:{kind}", "kind": "fuse", "target": "cuda", "fused_kind": kind, "status": status,
         "source": [source], "evidence": []}
    if chains is not None:
        h["chains"] = chains
    return h


def from_kb_proposals(doc: dict, graph: dict | None = None, specs: dict | None = None) -> list[dict]:
    """``kb-proposal/1`` -> hints. Fusion pairs are grouped by the fused kind
    they produce (a dry run of the fusion on the problem's IR, which also
    flattens fused producers into chains); infeasible pairs are dropped."""
    from .kb_proposals import _all_specs, fuse_pair

    specs = specs or _all_specs()
    graph = unique_names(copy.deepcopy(graph) if graph is not None else _graph_of_problem(Path(doc["problem_dir"])))
    names = {o["name"] for o in graph["ops"]}
    out: dict[str, dict] = {}
    for p in doc.get("proposals", []):
        src = {"by": "kb", "problem": doc.get("problem"), "arch": doc.get("arch"),
               **{k: p.get("evidence", {}).get(k) for k in ("est_gain_us", "est_gain_pct")},
               "rationale": p.get("rationale"), "ts": _now()}
        if p.get("kind") == "fuse":
            for a, b, _t in p["pairs"]:
                if a not in names or b not in names:
                    continue
                try:
                    _, spec = fuse_pair(graph, a, b, specs)
                except (ValueError, StopIteration, KeyError):
                    continue
                h = out.setdefault(spec.op, fuse_hint(spec.op, [], src))
                h["chains"].append(atoms(a) + atoms(b))
        elif p.get("kind") == "split":
            hid = f"split:{p['slot']}"
            out[hid] = {"id": hid, "kind": "split_slot", "target": "cuda", "slot": p["slot"],
                        "by": "shape_bucket", "groups": p["groups"], "status": "proposed",
                        "source": [src], "evidence": []}
    return list(out.values())


def from_xpurt(doc: dict, status: str = "accepted") -> list[dict]:
    """XPU-RT's Contract-2 hints -> RISC-V hints, one per network entry."""
    kinds = {"modelblaster.fusion_hints/v1": ("fuse", "fuse_groups", "dispatch_groups"),
             "modelblaster.split_hints/v1": ("tile_split", "split_ops", "split_ops"),
             "modelblaster.shard_hints/v1": ("shard", "shard_ops", "shard_ops"),
             "modelblaster.unfuse_hints/v1": ("unfuse", "unfuse_ops", "unfuse_ops")}
    contract = doc.get("contract")
    if contract not in kinds:
        raise ValueError(f"not an XPU-RT hint contract: {contract!r}")
    kind, field_in, field_out = kinds[contract]
    out = []
    for n in doc.get("networks", []):
        if not n.get(field_in):
            continue
        out.append({"id": f"xpurt:{kind}:{n['network']}", "kind": kind, "target": "riscv",
                    "network": n["network"], field_out: n[field_in], "status": status,
                    "source": [{"by": "xpurt", "contract": contract, "reason": doc.get("reason"), "ts": _now()}],
                    "evidence": []})
    return out


def propose_from_profile(graph: dict, op_us: dict[str, float], specs: dict, *, doc: dict | None = None,
                         min_gain_pct: float = 2.0, ir: str | None = None) -> list[dict]:
    """ModelBlaster's own fusion proposals from a whole-model per-op profile:
    every producer -> elementwise consumer pair where the consumer is the
    producer output's only reader (fused producers and consumers included, so
    accepted fusions grow into chains), grouped by fused kind; gain estimate =
    the consumers' measured time. Kinds already decided in ``doc`` are skipped."""
    from .kb_proposals import fuse_pair, with_fused_specs
    from .op_classes import op_class

    graph = unique_names(copy.deepcopy(graph))
    specs = with_fused_specs(graph, specs)
    total = sum(op_us.values()) or 1.0
    done = decided(doc) if doc else set()
    outs = set(graph["output"].get("tensors") or [graph["output"]["tensor"]])
    readers: dict[str, list[dict]] = {}
    for o in graph["ops"]:
        for t in o["inputs"]:
            readers.setdefault(t, []).append(o)
    cands: dict[str, dict] = {}
    used: set[str] = set()
    pairs = []
    for o in graph["ops"]:
        if len(o["outputs"]) != 1 or o["op"] not in specs:
            continue
        r = readers.get(o["outputs"][0], [])
        if len(r) != 1 or o["outputs"][0] in outs:
            continue
        b = r[0]
        if b["op"] not in specs or op_class(b["op"]) != "elementwise":
            continue
        pairs.append((op_us.get(b["name"], 0.0), len(pairs), o["name"], b["name"]))
    for gain, _, a, b in sorted(pairs, key=lambda p: (-p[0], p[1])):
        if a in used or b in used:
            continue
        try:
            _, spec = fuse_pair(graph, a, b, specs)
        except (ValueError, StopIteration, KeyError):
            continue
        if f"fuse:{spec.op}" in done:
            continue
        used.update((a, b))
        c = cands.setdefault(spec.op, {"chains": [], "gain_us": 0.0})
        c["chains"].append(atoms(a) + atoms(b))
        c["gain_us"] += gain
    out = []
    for kind, c in sorted(cands.items(), key=lambda kv: -kv[1]["gain_us"]):
        pct = 100.0 * c["gain_us"] / total
        if pct < min_gain_pct:
            continue
        out.append(fuse_hint(kind, c["chains"], {"by": "mb-cost", "ir": ir, "est_gain_us": round(c["gain_us"], 2),
                                                 "est_gain_pct": round(pct, 2), "ts": _now()}))
    return out


# ---------------------------------------------------------------- apply

def _locate(graph: dict, chain: list[str]) -> list[str]:
    """Current op names covering a chain of original names (consecutive
    atoms already fused into one op collapse to it)."""
    where = {a: o["name"] for o in graph["ops"] for a in atoms(o["name"])}
    if any(a not in where for a in chain):
        raise ValueError(f"{chain}: op(s) {[a for a in chain if a not in where]} not in the graph")
    return list(dict.fromkeys(where[a] for a in chain))


def _apply_fuse_cuda(graph, h, specs, prov):
    from .kb_proposals import find_chains, fuse_chain, spec_from_kind

    kind = h["fused_kind"]
    try:
        spec = spec_from_kind(kind, specs)
    except (ValueError, KeyError) as e:
        prov["skipped"].append({"id": h["id"], "why": str(e)})
        return graph, {}
    chains = h.get("chains")
    if chains is None:
        chains = find_chains(graph, kind, specs)
    used = {}
    for chain in chains:
        try:
            cur = _locate(graph, chain)
            if len(cur) < 2:
                raise ValueError("already one op")
            g2, s = fuse_chain(graph, cur, specs)
            if s.op != kind:
                raise ValueError(f"chain fuses to {s.op}, not {kind}")
        except (ValueError, StopIteration, KeyError) as e:
            prov["skipped"].append({"id": h["id"], "chain": chain, "why": str(e)})
            continue
        graph = g2
        new = "+".join(cur)
        next(o for o in graph["ops"] if o["name"] == new)["hint"] = h["id"]
        for n in chain:
            prov["renamed"][n] = new
        for n in cur:
            prov["renamed"][n] = new
        prov["applied"].append({"id": h["id"], "kind": "fuse", "op": new, "from": cur, "fused_kind": kind})
        used[kind] = s
    if not used and h.get("chains") is None:
        prov["skipped"].append({"id": h["id"], "why": "pattern does not occur"})
    return graph, ({kind: spec} if used else {})


def _apply_split_cuda(graph, h, specs, prov):
    from .kb_export import base_slot_id
    from .op_classes import op_class

    groups = h.get("groups")
    inst = [o for o in graph["ops"] if o["op"] in specs and base_slot_id(o["op"], specs) == h["slot"]]
    if groups is None:                                   # pattern level: by shape bucket
        by: dict[str, list[str]] = {}
        for o in inst:
            by.setdefault(shape_bucket(op_class(o["op"]), o.get("shape", {})), []).append(o["name"])
        groups = [{"bucket": b, "ops": n} for b, n in by.items()]
    names = {o["name"]: o for o in inst}
    live = []
    for grp in groups:
        ops = [prov["renamed"].get(n, n) for n in grp["ops"]]
        ops = [n for n in dict.fromkeys(ops) if n in names]
        if ops:
            live.append({**grp, "ops": ops})
    if len(live) < 2:
        prov["skipped"].append({"id": h["id"], "why": f"{len(live)} shape group(s) of slot {h['slot']} here"})
        return graph
    for k, grp in enumerate(live):
        for n in grp["ops"]:
            names[n]["slot_variant"] = f"v{k}"
            names[n]["hint"] = h["id"]
    prov["applied"].append({"id": h["id"], "kind": "split_slot", "slot": h["slot"],
                            "variants": {f"v{k}": g["ops"] for k, g in enumerate(live)}})
    return graph


def _apply_unfuse_cuda(graph, h, specs, prov):
    ops = graph["ops"]
    targets = [o for o in ops if "sub_ops" in o and (o["name"] in (h.get("ops") or [])
                                                     or o["op"] == h.get("fused_kind"))]
    if not targets:
        prov["skipped"].append({"id": h["id"], "why": "no matching fused op"})
        return graph
    for t in targets:
        missing = [x for x in t.get("internal", []) if x not in graph["tensors"]]
        if missing:
            prov["skipped"].append({"id": h["id"], "op": t["name"],
                                    "why": f"internal tensors {missing} are not registered (an int8 RISC-V "
                                           f"fused op: use target riscv / apply_unfuse_hint)"})
            continue
        i = ops.index(t)
        stages = [copy.deepcopy(s) for s in t["sub_ops"]]
        for s in stages:
            s.pop("depends_on", None)
            s["unfused_from"] = t["name"]
        ops[i:i + 1] = stages
        for s in stages:
            prov["renamed"][s["name"]] = s["name"]
        prov["applied"].append({"id": h["id"], "kind": "unfuse", "op": t["name"], "to": [s["name"] for s in stages]})
    return graph


def _xpurt_fuse_to_chains(graph: dict, h: dict) -> list[list[str]]:
    """XPU-RT dispatch groups -> op-name chains (validated by the RISC-V
    applier's closure check, then required to be a producer->consumer chain)."""
    from .apply_fusion_hint import _ops_by_id, _validate_fuse_group

    by_id = _ops_by_id(graph["ops"])
    chains = []
    for grp in h["dispatch_groups"]:
        _validate_fuse_group(grp, by_id, graph.get("name", "?"))
        chains.append([by_id[d]["name"] for d in grp])
    return chains


def _renumber(graph: dict, before: dict[str, int]) -> None:
    """Contiguous dispatch ids, depends_on from tensor producers (the rule
    extract_graph uses), and ``id_remap`` {old id: [new ids]} by the original
    op names each op is made of. View ops (no dispatch id) stay as they are."""
    if not any(v is not None for v in before.values()):        # an IR without dispatch ids
        return
    nid, producer, new_of_atom = 0, {}, {}
    for o in graph["ops"]:
        if "dispatch_id" in o and o["dispatch_id"] is None:          # a view: no dispatch
            continue
        o["dispatch_id"] = nid
        for a in atoms(o["name"]):
            new_of_atom.setdefault(a, []).append(nid)
        nid += 1
    for o in graph["ops"]:
        if o.get("dispatch_id") is None:
            continue
        o["depends_on"] = sorted({producer[t] for t in o["inputs"] if t in producer} - {o["dispatch_id"]})
        for t in o["outputs"]:
            producer[t] = o["dispatch_id"]
    remap = {}
    for name, old in before.items():
        if old is None:
            continue
        new = sorted({i for a in atoms(name) for i in new_of_atom.get(a, [])})
        for d in ([old] if not isinstance(old, list) else old):
            remap.setdefault(str(d), sorted(set(remap.get(str(d), [])) | set(new)))
    graph["id_remap"] = remap
    if isinstance(graph.get("dispatches"), list):
        graph["dispatches"] = [o["dispatch_id"] for o in graph["ops"] if o.get("dispatch_id") is not None]


def apply_hints(graph: dict, hints: list[dict], specs: dict | None = None, *, target: str = "cuda",
                statuses: tuple | None = ("accepted",)) -> tuple[dict, dict, dict]:
    """Apply ``hints`` (those whose status is in ``statuses``; None = all) to
    a copy of ``graph`` for ``target``.

    -> (graph, {fused kind: FusedKernelSpec} the CUDA graph now needs,
        provenance {"applied", "skipped", "renamed": {old op name: new}}).
    Fusions apply longest chain first, so an accepted conv->bn->relu wins over
    its accepted conv->bn prefix where both match."""
    from .kb_proposals import _all_specs, parse_kind

    specs = dict(specs or _all_specs())
    g = unique_names(copy.deepcopy(graph))
    prov = {"applied": [], "skipped": [], "renamed": {}}
    todo = [h for h in hints if statuses is None or h.get("status") in statuses]
    if target == "riscv":
        return _apply_riscv(g, todo, prov)
    before = {o["name"]: o.get("dispatch_id") for o in g["ops"]}

    def order(h):
        if h["kind"] == "fuse" and h.get("fused_kind"):
            try:
                return (0, -len(parse_kind(h["fused_kind"], specs)[0]))
            except (ValueError, KeyError):
                return (0, 0)
        return ({"fuse": 0, "unfuse": -1, "split_slot": 1}.get(h["kind"], 2), 0)

    added = {}
    for h in sorted(todo, key=order):
        if h.get("target", "cuda") == "riscv" and h["kind"] == "fuse" and "dispatch_groups" in h:
            try:
                chains = _xpurt_fuse_to_chains(g, h)
            except ValueError as e:
                prov["skipped"].append({"id": h["id"], "why": str(e)})
                continue
            from .kb_proposals import fuse_chain
            for ch in chains:
                try:
                    _, s = fuse_chain(g, ch, specs)
                except (ValueError, StopIteration, KeyError) as e:
                    prov["skipped"].append({"id": h["id"], "chain": ch, "why": f"not a fusable chain on cuda: {e}"})
                    continue
                g, sp = _apply_fuse_cuda(g, {**h, "fused_kind": s.op, "chains": [ch]}, specs, prov)
                added.update(sp)
                specs.update(sp)
        elif h.get("target", "cuda") != "cuda" and h["kind"] not in CUDA_KINDS:
            prov["skipped"].append({"id": h["id"], "why": "not applicable on cuda: " +
                                    NOT_ON_CUDA.get(h["kind"], f"kind {h['kind']}")})
        elif h["kind"] == "fuse":
            g, sp = _apply_fuse_cuda(g, h, specs, prov)
            added.update(sp)
            specs.update(sp)
        elif h["kind"] == "split_slot":
            g = _apply_split_cuda(g, h, specs, prov)
        elif h["kind"] == "unfuse":
            if h.get("target") == "riscv":
                prov["skipped"].append({"id": h["id"], "why": "int8 unfuse needs quant blocks: target riscv"})
            else:
                g = _apply_unfuse_cuda(g, h, specs, prov)
        else:
            prov["skipped"].append({"id": h["id"], "why": f"unknown kind {h['kind']}"})
    _renumber(g, before)
    _record(g, prov, target)
    return g, added, prov


def _apply_riscv(g: dict, hints: list[dict], prov: dict):
    from .apply_fusion_hint import apply_hint
    from .apply_shard_hint import apply_shard_hint
    from .apply_split_hint import apply_split_hint
    from .apply_unfuse_hint import apply_unfuse_hint

    net = g.get("name")
    for h in hints:
        if h.get("network") not in (None, net):
            prov["skipped"].append({"id": h["id"], "why": f"hint is for network {h.get('network')}"})
            continue
        if h["kind"] == "fuse" and "dispatch_groups" in h:
            g = apply_hint(g, h["dispatch_groups"], pairwise=bool(h.get("pairwise")))
        elif h["kind"] == "tile_split":
            g = apply_split_hint(g, h["split_ops"])
        elif h["kind"] == "shard":
            g = apply_shard_hint(g, h["shard_ops"])
        elif h["kind"] == "unfuse" and "unfuse_ops" in h:
            g = apply_unfuse_hint(g, h["unfuse_ops"])
        else:
            prov["skipped"].append({"id": h["id"], "why": f"{h['kind']} has no RISC-V applier"})
            continue
        prov["applied"].append({"id": h["id"], "kind": h["kind"]})
    _record(g, prov, "riscv")
    return g, {}, prov


def _record(g: dict, prov: dict, target: str) -> None:
    rec = g.setdefault("graph_hints", {"format": FORMAT, "rounds": []})
    rec["rounds"].append({"target": target, "applied": prov["applied"], "skipped": prov["skipped"], "ts": _now()})


# ---------------------------------------------------------------- verify

def verify_rewrite(ir_dir: Path, graph: dict, *, specs: dict | None = None, build: bool = False,
                   arch: str = "75", work: Path | None = None) -> dict:
    """Torch-walk the rewritten ``graph`` (weights/io of ``ir_dir``) against
    the original golden and the original walk; with ``build``, also export
    the whole rewritten graph (seed kernels) and run its driver."""
    import tempfile

    import torch

    from .kb_export import _harness

    mbir, walk_torch = _harness()
    ir = mbir.load_ir(ir_dir)
    with torch.no_grad():
        ref, _ = walk_torch.walk(ir, dev="cpu")
        new_ir = mbir.IR(ir.name, graph, ir.weights, ir.golden_in, ir.golden_out)
        out, _ = walk_torch.walk(new_ir, dev="cpu")
    res = {"golden": mbir.compare(out, ir.golden_out),
           "vs_original_walk": float((out - ref).abs().max()) if out.shape == ref.shape else None}
    res["ok"] = bool(res["golden"]["ok"] and res["vs_original_walk"] == 0.0)
    if build:
        from .kb_export import build_and_run, export_problem
        from .kb_proposals import write_variant_ir

        tmp = Path(work or tempfile.mkdtemp(prefix="mb_hints_"))
        vir = write_variant_ir(Path(ir_dir), graph, tmp / "ir")
        prob = export_problem(vir, tmp / "problem", specs=specs, strategy="whole")
        r = build_and_run(prob, arch=arch)
        res["driver"] = {k: r.get(k) for k in ("passed", "stage", "partition_us", "slot_us")}
        if not r.get("passed"):
            res["driver"]["log"] = r.get("log", "")[-1500:]
        res["ok"] = res["ok"] and bool(r.get("passed"))
    return res


def apply_to_ir(ir_dir: Path, hints_doc: dict, out_dir: Path, specs: dict | None = None,
                statuses: tuple | None = ("accepted",)) -> tuple[Path, dict, dict]:
    """Apply a hints document to an IR dir; writes the rewritten IR (weights
    and io symlinked) when anything applied. -> (IR dir to use, specs, prov)."""
    from .kb_proposals import _all_specs, write_variant_ir

    specs = dict(specs or _all_specs())
    g = json.loads((Path(ir_dir) / "graph.json").read_text())
    g2, added, prov = apply_hints(g, hints_doc["hints"], specs, statuses=statuses)
    specs.update(added)
    if not prov["applied"]:
        return Path(ir_dir), specs, prov
    return write_variant_ir(Path(ir_dir), g2, Path(out_dir)), specs, prov


def hints_sha(doc: dict) -> str:
    key = [(h["id"], h.get("status"), h.get("chains"), h.get("groups")) for h in doc["hints"]]
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("apply")
    a1.add_argument("ir_dir")
    a1.add_argument("hints")
    a1.add_argument("out_ir")
    a1.add_argument("--all", action="store_true", help="apply proposed hints too, not only accepted")
    a1.add_argument("--target", default="cuda", choices=["cuda", "riscv"])
    a1.add_argument("--verify", action="store_true")
    a1.add_argument("--build", action="store_true")
    a1.add_argument("--arch", default="75")
    a2 = sub.add_parser("import")
    a2.add_argument("source", choices=["kb", "xpurt", "registry"])
    a2.add_argument("files", nargs="+")
    a2.add_argument("--into", required=True)
    a3 = sub.add_parser("show")
    a3.add_argument("hints")
    a = ap.parse_args()
    if a.cmd == "apply":
        from .kb_proposals import _all_specs, write_variant_ir

        doc, specs = load(Path(a.hints)), dict(_all_specs())
        g = json.loads((Path(a.ir_dir) / "graph.json").read_text())
        g2, added, prov = apply_hints(g, doc["hints"], specs, target=a.target,
                                      statuses=None if a.all else ("accepted",))
        specs.update(added)
        write_variant_ir(Path(a.ir_dir), g2, Path(a.out_ir))
        out = {"applied": prov["applied"], "skipped": prov["skipped"]}
        if a.verify or a.build:
            out["verify"] = verify_rewrite(Path(a.ir_dir), g2, specs=specs, build=a.build, arch=a.arch)
        print(json.dumps(out, indent=1))
    elif a.cmd == "import":
        doc = load(Path(a.into))
        for f in a.files:
            d = json.loads(Path(f).read_text())
            new = (from_kb_proposals(d) if a.source == "kb" else from_xpurt(d) if a.source == "xpurt"
                   else load(Path(f))["hints"])
            merge(doc, new)
        save(doc, Path(a.into))
        print(json.dumps([h["id"] for h in doc["hints"]]))
    else:
        for h in load(Path(a.hints))["hints"]:
            ev = h.get("evidence", [])
            print(f"{h['id']:<55} {h.get('status', '?'):<9} {h.get('target', 'cuda'):<6} "
                  f"decisions={len(ev)} last_gain={ev[-1].get('gain') if ev else None}")


if __name__ == "__main__":
    main()
