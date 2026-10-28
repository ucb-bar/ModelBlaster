"""Open-ended graph rewrites in the kb_graph_opt round loop (docs/graph_rewrite.md): the flag, the proposer interface and
the plumbing that feeds candidates into the EXISTING hint decision.

``kb_graph_opt --graph-rewrite off|agent`` (default off; env ``MB_GRAPH_REWRITE`` when the flag is absent). With ``off``
nothing here runs, nothing is written, the round key and every output are unchanged (the hook is an inert object; the
only code the flag-off path executes is ``RewriteHook.decide_kinds() == ()`` / ``round_key() == {}`` / ``propose()`` returning
at its first line). With ``agent`` a :class:`RewriteProposer` is asked, once per IR per round, for candidate edits.

One round, at the point where MB's own hint proposals are made (right after ``caps.propose``)::

    candidates = proposer.propose(ir, profile, budget)        # RewriteCandidate: an edited IR dir + provenance + est. cost
    for c in candidates:
        verdict = graph_verify.verify(current IR, c.edited_ir_dir, orig_weights=source IR weights, tensor_map=...)
        not ok  -> a REJECTED hypothesis (hint record + evidence {stage: graph_verify, reasons}), never timed
        ok      -> a PROPOSED hint (source "agent-rewrite"):
                     edit_type "rewire"  -> kind "rewrite": the patch (graph_edit.diff) + derived weights
                     edit_type "region"  -> kind "fuse_region" with explicit members (the existing applier)
    ...the existing round loop decides them like any hint: prepare_hint exports the ORIGINAL and the VARIANT problem,
    KernelBlaster optimises both with the same budget, both are re-timed back to back (kb_proposals.remeasure),
    finish_hint / graph_search.decide_checkpoint accept at ``accept_gain``, record the evidence, ingest the new slots.

The outcome is a graph hypothesis: the hint-registry record (id, status, evidence list, source "agent-rewrite") that
KernelBlaster's knowledge v2 syncs (``sync_graph_hints``: accepted -> promoted, rejected -> retired, by hint id).

Edit types (docs/graph_rewrite.md): (1) REWIRE with existing op kinds + derived weights  [implemented here];
(2) REGION: arbitrary members collapsed to a ``region__`` op  [existing fuse_region; this module only routes it];
(3) NEW OP KINDS needing a kernel spec  [interface only: :class:`NewOpSpec`, rejected with a reason: a later phase].

Proposers shipped: ``test-merge-expand`` (deterministic: SqueezeNet's expand1x1 / expand3x3 pair -> one conv with a derived
zero-padded weight stack), ``test-region`` (deterministic: explicit members -> region edit), ``agent`` (the LLM proposer:
NOT IMPLEMENTED, raises NotImplementedError; its interface is specified in :class:`AgentProposer`).
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from . import graph_edit as ge
from . import weight_derivations as wd

ENV = "MB_GRAPH_REWRITE"
MODES = ("off", "agent")
SOURCE = "agent-rewrite"
EDIT_TYPES = ("rewire", "region", "new_op")


def resolve_mode(cfg=None) -> str:
    """``--graph-rewrite`` wins, else ``$MB_GRAPH_REWRITE``, else off. Validated."""
    v = getattr(cfg, "graph_rewrite", None)
    if v is None:
        v = os.environ.get(ENV, "off")
    v = str(v).strip().lower() or "off"
    if v in ("0", "false", "no"):
        v = "off"
    if v not in MODES:
        raise ValueError(f"--graph-rewrite must be one of {MODES}, not {v!r}")
    return v


# ------------------------------------------------------------------------------ the candidate and the proposer interface
@dataclass
class NewOpSpec:
    """Edit type 3 (a LATER phase): what a new op kind must bring. Not consumed anywhere yet; ``graph_verify`` rejects an
    edited graph that uses a kind without a spec, naming this requirement."""
    kind: str                       # new op kind, e.g. "fused_gate_up_silu" or "rope_apply"
    signature: str                  # C signature (kernel_abi.parse_signature), float / _Float16 / __bf16 spellings per dtype
    semantics: str                  # reference semantics in words + the reference implementation (C) the walkers' rule mirrors
    tolerance: dict = field(default_factory=dict)       # {"slot_rel", "slot_abs"} (cuda_dtypes.TOLERANCE keys)
    cuda_seed: str = ""             # a correct (slow) CUDA seed kernel for the slot loop to start from
    op_class: str = "elementwise"   # op_classes class (drives the KernelBlaster menu / roofline state)
    walker_rule: str = ""           # name of the walk_torch rule implementing the reference semantics


@dataclass
class RewriteCandidate:
    """One candidate edit, as a proposer hands it over.

    ``edited_ir_dir``: a complete IR dir (graph.json, weights.npz, io.npz) of the edited graph, derived from the IR the proposer
    was shown (``ir["dir"]``); derived weights are in graph.json ``weight_derivations`` (write them with
    ``kb_proposals.write_variant_ir`` after an ``Editor``). ``provenance``: free-form who/what/how (model, session id, script
    sha, tool calls ...) recorded in the hint's source. ``est_gain_pct``: the proposer's estimate (share of model time) used to
    rank candidates against the flow's own hints (None = the flow's minimum). ``est_cost``: {"llm_usd", "ops_before", "ops_after",
    ...}. ``tensor_map``: {old tensor: {"from": new tensor, "axis", "start", "stop"}} for tensors the edit removed but that are a
    slice of a new one (they are compared too). ``members``: op names, for edit_type "region"."""
    ir: str
    edited_ir_dir: Path
    edit_type: str = "rewire"
    description: str = ""
    provenance: dict = field(default_factory=dict)
    est_gain_pct: float | None = None
    est_cost: dict = field(default_factory=dict)
    tensor_map: dict = field(default_factory=dict)
    members: list | None = None
    new_ops: list = field(default_factory=list)          # [NewOpSpec], edit type 3 (rejected for now)
    id: str | None = None                                # set by the hook (content hash)


class RewriteProposer:
    """Interface of a graph-rewrite proposer (deterministic or LLM-backed).

    ``propose(ir, profile, budget) -> list[RewriteCandidate]``

    ``ir``      {"name", "dir" (the IR the round profiled: accepted hints and cached winners applied, derived weights
                materialised), "source_ir" (the original IR dir, whose weights.npz holds the original weights), "graph" (its
                graph.json as a dict)}
    ``profile`` the round's profile of that IR (kb_plan.profile_ir): "ops" [{"name", "op", "us", "shape"}], "e2e_us",
                "launch_us", "partition_us"
    ``budget``  {"max_candidates", "work" (a dir the proposer may write edited IRs under), "round", "llm_usd" (None = unlimited)}

    It must not modify ``ir["dir"]``; it may write anything under ``budget["work"]``. Raising any exception other than
    NotImplementedError loses only that IR's candidates (logged, recorded in the round); NotImplementedError aborts the run
    (``agent`` mode with the LLM proposer not built)."""
    name = "base"

    def propose(self, ir: dict, profile: dict, budget: dict) -> list[RewriteCandidate]:
        raise NotImplementedError


class AgentProposer(RewriteProposer):
    """The LLM proposer (NOT IMPLEMENTED: a later phase; the graph-agent pilot is its prototype).

    Intended shape (docs/graph_rewrite.md, "LLM proposer"): ``propose`` starts one agent session per IR (Claude Code, the pilot's
    isolation flags, no API key beyond the session's, a cost cap from ``budget["llm_usd"]``) in a scratch workspace holding the
    IR dir (read-only), the round profile and a typed tool set modelled on the pilot's ``gttool.py``:

      show / match     overview, ops by GPU time, dataflow levels, weights with provenance (original | derived), chain / sibling /
                       fan-out / repeated-window matches                                  (read-only, from graph.json + profile)
      region NAME OPS  collapse ANY op set into one region op (edit type 2; fuse_region with explicit members)
      rewrite SCRIPT   ``def rewrite(ed)`` on a ``graph_edit.Editor`` in a restricted interpreter (no imports / underscore attrs /
                       eval / open; timeout): graph surgery + derived weights ONLY through ``ed.derive`` (weight_derivations.Recorder)
      verify           ``graph_verify.verify`` on the pending candidate; the structured verdict (reasons, max errors) is the tool's
                       result, so the agent can repair a rejected rewrite
      commit           writes the verified candidate's IR under ``budget["work"]`` and returns it as a RewriteCandidate (est_gain_pct
                       from the agent's claim, est_cost from the session's cost.json)

    The agent never times anything: timing and the accept decision stay in the flow (paired variant-vs-original, ``accept_gain``)."""
    name = "agent"

    def propose(self, ir: dict, profile: dict, budget: dict) -> list[RewriteCandidate]:
        raise NotImplementedError("--graph-rewrite agent: the LLM rewrite proposer is not implemented yet (docs/graph_rewrite.md, "
                                  "'LLM proposer'); use a deterministic proposer (cfg.graph_rewrite_proposer) to exercise the path")


# ------------------------------------------------------------------------------ deterministic proposers (tests / the fixture path)
def _suffix_of(kind: str) -> str:
    return "_bf16" if kind.endswith("_bf16") else "_f16" if kind.endswith("_f16") else ""


class MergeExpandProposer(RewriteProposer):
    """SqueezeNet's fire-module rewrite, found by pattern: two stride-1 'same' convolutions reading one tensor, each followed
    (exclusively) by a relu, whose outputs are concatenated by a ``cat2_c1`` -> ONE convolution with the larger kernel whose
    weights are  concat([zero_pad(w_small to the large kernel), w_large], axis 0)  (bias concat), one relu, the cat's output
    name kept. Exact (structural derivations only). ``37 -> 21`` ops on SqueezeNet's 8 fire modules."""
    name = "test-merge-expand"

    def __init__(self, wrong: str | None = None):
        self.wrong = wrong                       # negative-test hook: "order" stacks the filters (and biases) in the wrong order

    def propose(self, ir: dict, profile: dict, budget: dict) -> list[RewriteCandidate]:
        g = ir["graph"]
        by_name = {o["name"]: o for o in g["ops"]}
        prod = ge.produced(g)
        users: dict[str, list] = {}
        for o in g["ops"]:
            for t in o.get("inputs", []):
                users.setdefault(t, []).append(o)
        us = {o["name"]: o.get("us") or 0.0 for o in (profile or {}).get("ops", [])}
        total = (profile or {}).get("e2e_us") or sum(us.values()) or 1.0
        out = []
        for cat in g["ops"]:
            if not cat["op"].startswith("cat2_c1") or len(cat["inputs"]) != 2:
                continue
            pair = []
            for t in cat["inputs"]:
                r = by_name.get(prod.get(t, ""))
                if r is None or not r["op"].startswith("relu") or len(users.get(t, [])) != 1:
                    break
                c = by_name.get(prod.get(r["inputs"][0], ""))
                if c is None or not c["op"].startswith("conv2d") or not c.get("weight") or len(users.get(r["inputs"][0], [])) != 1:
                    break
                pair.append((c, r))
            if len(pair) != 2 or pair[0][0]["inputs"] != pair[1][0]["inputs"]:
                continue
            cands = self._build(ir, g, cat, pair, budget, len(out))
            if cands is not None:
                c = cands
                rem = [x["name"] for x in (pair[0][0], pair[0][1], pair[1][0], pair[1][1], cat)]
                saved = sum(us.get(n, 0.0) for n in rem) - (us.get(pair[0][0]["name"], 0.0) + us.get(pair[1][0]["name"], 0.0))
                c.est_gain_pct = round(100.0 * max(saved, 0.0) / total, 3) if saved > 0 else None
                out.append(c)
            if len(out) >= int(budget.get("max_candidates", 3)):
                break
        return out

    def _build(self, ir, g, cat, pair, budget, k):
        import numpy as np
        (ca, ra), (cb, rb) = pair
        sa, sb = ca["shape"], cb["shape"]
        if sa.get("KH", 1) > sb.get("KH", 1):               # a = the smaller kernel
            (ca, ra), (cb, rb) = (cb, rb), (ca, ra)
            sa, sb = sb, sa
        same = all(sa.get(f) == sb.get(f) for f in ("N", "IC", "IH", "IW", "OH", "OW", "SH", "SW", "DH", "DW"))
        if not same or sa["SH"] != 1 or sa["SW"] != 1 or sa.get("KH") != sa.get("KW") or sb.get("KH") != sb.get("KW") \
                or sa["KH"] >= sb["KH"] or (sb["KH"] - sa["KH"]) % 2 or sb["PH"] != sb["KH"] // 2 or sa["PH"] != sa["KH"] // 2:
            return None
        if _suffix_of(ca["op"]) != _suffix_of(cb["op"]) or (ca.get("bias") is None) != (cb.get("bias") is None):
            return None
        tag = hashlib.sha1((ca["name"] + cb["name"] + cat["name"]).encode()).hexdigest()[:8]
        work = Path(budget["work"]) / f"merge_{tag}"
        src = Path(ir["dir"])
        z = np.load(src / "weights.npz")
        base = {key: z[key] for key in z.files}
        ed = ge.Editor(g, base)
        D = ed.derive
        d = (sb["KH"] - sa["KH"]) // 2
        pk = f"d.merge.{tag}"
        D.pad(f"{pk}.a_pad", ca["weight"], [[0, 0], [0, 0], [d, d], [d, d]])
        sw = self.wrong == "order"
        D.concat(f"{pk}.w", [cb["weight"], f"{pk}.a_pad"] if sw else [f"{pk}.a_pad", cb["weight"]], 0)
        if ca.get("bias") is not None:
            D.concat(f"{pk}.b", [cb["bias"], ca["bias"]] if sw else [ca["bias"], cb["bias"]], 0)
        oc = sa["OC"] + sb["OC"]
        old_cat = g["tensors"][cat["outputs"][0]]
        mo = ed.unique(f"mrg_{tag}_o")
        ed.remove_ops([ca["name"], ra["name"], cb["name"], rb["name"], cat["name"]])
        for t in (ca["outputs"][0], ra["outputs"][0], cb["outputs"][0], rb["outputs"][0]):
            ed.g["tensors"].pop(t, None)
        ed.add_tensor(mo, [sa["N"], oc, sa["OH"], sa["OW"]], old_cat["dtype"])
        ed.add_op({"name": ed.unique(f"mrg_{tag}"), "op": cb["op"], "inputs": list(cb["inputs"]), "outputs": [mo], "weight": f"{pk}.w",
                   "bias": f"{pk}.b" if ca.get("bias") is not None else None,
                   "shape": {**sb, "OC": oc}})
        rn = ra["op"]
        out_name = cat["outputs"][0]
        ed.add_op({"name": ed.unique(f"mrg_{tag}_relu"), "op": rn, "inputs": [mo], "outputs": [out_name],
                   "shape": {**(ra.get("shape") or {}), "n": sa["N"] * oc * sa["OH"] * sa["OW"]}})
        g1 = ed.result()
        out_ir = _write_ir(src, g1, work)
        tmap = {ra["outputs"][0]: {"from": out_name, "axis": 1, "start": 0, "stop": sa["OC"]},
                rb["outputs"][0]: {"from": out_name, "axis": 1, "start": sa["OC"], "stop": oc}}
        return RewriteCandidate(ir=ir["name"], edited_ir_dir=out_ir, edit_type="rewire", tensor_map=tmap,
                                description=f"merge {ca['name']} ({sa['KH']}x{sa['KW']}) and {cb['name']} ({sb['KH']}x{sb['KW']}) "
                                            f"+ relu + {cat['name']} into one {sb['KH']}x{sb['KW']} conv with a zero-padded weight stack",
                                provenance={"proposer": self.name, "pattern": "expand-merge", "ops": [ca["name"], cb["name"], cat["name"]]},
                                est_cost={"llm_usd": 0.0, "ops_before": len(g["ops"]), "ops_after": len(g1["ops"]),
                                          "derived_weights": sorted(wd.derivations(g1))})


class RegionProposer(RewriteProposer):
    """Deterministic edit type 2: ``members`` (op names) of the IR become one region op (``fuse_region`` with explicit
    members, applied by the flow's own applier, so the candidate IR is exactly what the hint will give)."""
    name = "test-region"

    def __init__(self, members: list[str] | None = None, est_gain_pct: float | None = None):
        self.members, self.est = members, est_gain_pct

    def propose(self, ir: dict, profile: dict, budget: dict) -> list[RewriteCandidate]:
        from . import graph_hints as gh
        from .kb_proposals import _all_specs, write_variant_ir
        members = self.members or [o["name"] for o in ir["graph"]["ops"]][:4]
        h = {"id": "region:tmp", "kind": "fuse_region", "target": "cuda", "members": list(members), "status": "accepted", "source": []}
        g2, _, prov = gh.apply_hints(ir["graph"], [h], dict(_all_specs()))
        if not prov["applied"]:
            return []
        tag = hashlib.sha1(",".join(members).encode()).hexdigest()[:8]
        d = write_variant_ir(Path(ir["dir"]), g2, Path(budget["work"]) / f"region_{tag}")
        return [RewriteCandidate(ir=ir["name"], edited_ir_dir=d, edit_type="region", members=list(members), est_gain_pct=self.est,
                                 description=f"collapse {len(members)} ops into one region op",
                                 provenance={"proposer": self.name}, est_cost={"llm_usd": 0.0, "ops_before": len(ir["graph"]["ops"]),
                                                                              "ops_after": len(g2["ops"])})]


def _write_ir(src: Path, graph: dict, out: Path) -> Path:
    """Edited IR dir: graph.json, weights.npz = the source's + derived, io.npz (the same golden)."""
    from .kb_proposals import write_variant_ir
    return write_variant_ir(src, graph, out)


PROPOSERS = {"agent": AgentProposer, "test-merge-expand": MergeExpandProposer, "test-region": RegionProposer}


def make_proposer(spec: str | None) -> RewriteProposer:
    """``""`` -> the LLM stub; a registered name; or ``module:Class`` (dotted path, no-argument constructor)."""
    spec = (spec or "agent").strip()
    if spec in PROPOSERS:
        return PROPOSERS[spec]()
    if ":" in spec:
        mod, cls = spec.split(":", 1)
        return getattr(importlib.import_module(mod), cls)()
    raise ValueError(f"unknown rewrite proposer {spec!r} (registered: {sorted(PROPOSERS)}, or module:Class)")


# ------------------------------------------------------------------------------ hints
def candidate_id(c: RewriteCandidate, patch: dict | None) -> str:
    if c.edit_type == "region":
        return "region:agent:" + hashlib.sha1(json.dumps(sorted(c.members or [])).encode()).hexdigest()[:10]
    key = json.dumps({k: patch[k] for k in ("remove_ops", "add_ops", "add_tensors", "weight_derivations")} if patch else c.description,
                     sort_keys=True, default=str)
    return f"rewrite:{c.ir}:" + hashlib.sha1(key.encode()).hexdigest()[:10]


def make_hint(c: RewriteCandidate, patch: dict | None, verdict, min_gain_pct: float, proposer: str, after_hints=None) -> dict:
    src = {"by": SOURCE, "proposer": proposer, "ir": c.ir, "edit_type": c.edit_type, "description": c.description,
           "est_gain_pct": c.est_gain_pct if c.est_gain_pct is not None else min_gain_pct, "est_cost": c.est_cost,
           "provenance": c.provenance,
           "verify": {"ok": verdict.ok, "checks": {k: v["ok"] for k, v in verdict.checks.items()},
                      "max_errors": verdict.max_errors, "warnings": [w["code"] for w in verdict.warnings]}}
    h = {"id": c.id, "target": "cuda", "status": "proposed", "source": [src], "evidence": []}
    if c.edit_type == "region":
        h.update(kind="fuse_region", members=list(c.members or []))
    else:
        h.update(kind="rewrite", rewrite={"patch": patch, "tensor_map": c.tensor_map, "edit_type": c.edit_type,
                                          "after_hints": list(after_hints or []),
                                          "derived_weights": sorted(patch.get("weight_derivations", {})) if patch else []})
    return h


# ------------------------------------------------------------------------------ the hook the round loop calls
class RewriteHook:
    """``kb_graph_opt``'s handle (same shape as ``capabilities.Caps``). Inert with mode ``off``."""

    def __init__(self, cfg, work: Path | None = None, proposer: RewriteProposer | None = None):
        self.cfg, self.work = cfg, work
        self.mode = resolve_mode(cfg)
        self.proposer = proposer
        if self.mode == "agent" and self.proposer is None:
            self.proposer = make_proposer(getattr(cfg, "graph_rewrite_proposer", "") or "")
        self.log: list[dict] = []

    def active(self) -> bool:
        return self.mode != "off"

    def round_key(self) -> dict:
        return {"graph_rewrite": self.mode} if self.active() else {}

    def decide_kinds(self) -> tuple:
        return ("rewrite", "fuse_region") if self.active() else ()

    def summary(self) -> dict:
        return {"mode": self.mode, "proposer": getattr(self.proposer, "name", None), "log": self.log}

    def propose(self, r: int, R: Path, prog, graphs: dict, profiles: list, doc: dict, specs: dict) -> list:
        """Ask the proposer, gate every candidate, merge the survivors into ``doc`` as proposed hints and record the rest as
        rejected hypotheses. -> the hints added (proposed). Resumable: the round's candidate decisions are stored in ``prog``."""
        if not self.active():
            return []
        from . import graph_hints as gh
        saved = prog.once(f"graph_rewrite_{r}", lambda: self._propose_all(r, R, graphs, profiles, doc, specs)) \
            if prog is not None else self._propose_all(r, R, graphs, profiles, doc, specs)
        new = []
        for rec in saved:
            h = rec["hint"]
            if rec["verdict"]["ok"]:
                new += gh.merge(doc, [h])
            elif gh.get(doc, h["id"]) is None or not any(e.get("stage") == "graph_verify" for e in gh.get(doc, h["id"]).get("evidence", [])):
                ev = {"problem": None, "ir": h["source"][0]["ir"], "stage": "graph_verify", "decision": "rejected",
                      "why": rec["why"], "reasons": rec["verdict"]["reasons"][:6], "ts": gh._now()}
                gh.record_decision(doc, h, ev)
        return new

    def _propose_all(self, r, R, graphs, profiles, doc, specs) -> list:
        from . import graph_hints as gh
        from . import graph_verify as gv
        out = []
        decided = gh.decided(doc)
        budget0 = {"max_candidates": int(getattr(self.cfg, "graph_rewrite_max", 3) or 3), "round": r,
                   "llm_usd": getattr(self.cfg, "graph_rewrite_usd", None)}
        for p in profiles:
            g = graphs.get(p["ir"])
            if g is None:
                continue
            ir = {"name": p["ir"], "dir": str(p["ir_dir"]), "source_ir": str(p["source_ir"]), "graph": copy.deepcopy(g)}
            work = Path(R) / "rewrite" / gh_safe(p["ir"])
            work.mkdir(parents=True, exist_ok=True)
            try:
                cands = list(self.proposer.propose(ir, p, {**budget0, "work": str(work)}) or [])
            except NotImplementedError:
                raise
            except Exception as e:  # noqa: BLE001 -- a proposer never breaks a round
                self.log.append({"round": r, "ir": p["ir"], "error": f"{type(e).__name__}: {e}"[:300]})
                print(f"[kb_graph_opt] rewrite proposer {getattr(self.proposer, 'name', '?')}: {type(e).__name__}: {e}", flush=True)
                continue
            for c in cands[:budget0["max_candidates"]]:
                out.append(self._gate(r, c, p, g, specs, decided, doc))
        return [x for x in out if x is not None]

    def _gate(self, r: int, c: RewriteCandidate, p: dict, g0: dict, specs: dict, decided: set, doc: dict):
        from . import graph_verify as gv
        patch = None
        try:
            g1 = json.loads((Path(c.edited_ir_dir) / "graph.json").read_text())
            patch = ge.diff(g0, g1, tensor_map=c.tensor_map or None)
        except Exception:  # noqa: BLE001 -- the gate reports it
            pass
        c.id = c.id or candidate_id(c, patch)
        if c.id in decided:
            return None
        proposer = getattr(self.proposer, "name", "?")
        if c.edit_type == "new_op" or c.new_ops:
            v = gv.Verdict()
            v.fail("new_op_kind", "edit type 3 (new op kinds with a kernel spec) is a later phase: only edits over existing op kinds "
                                  "(rewire + derived weights) and region nodes are accepted")
        else:
            v = gv.verify(p["ir_dir"], c.edited_ir_dir, orig_weights=Path(p["source_ir"]) / "weights.npz", specs=specs,
                          tensor_map=c.tensor_map or None, seeds=_seeds(self.cfg), device=getattr(self.cfg, "graph_rewrite_device", "cpu") or "cpu")
        h = make_hint(c, patch, v, getattr(self.cfg, "min_gain_pct", 2.0), proposer,
                      after_hints=[x["id"] for x in doc["hints"] if x.get("status") == "accepted"])
        self.log.append({"round": r, "id": c.id, "ir": c.ir, "edit_type": c.edit_type, "ok": v.ok, "verdict": v.summary()})
        return {"hint": h, "verdict": json.loads(json.dumps(v.to_dict(), default=str)), "why": "graph_verify: " + v.summary()}


def gh_safe(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


def _seeds(cfg) -> tuple:
    return tuple(range(int(getattr(cfg, "graph_rewrite_seeds", 2) or 2)))


# ------------------------------------------------------------------------------ the applier of a "rewrite" hint (graph_hints calls it)
def apply_rewrite_hint(graph: dict, h: dict, specs: dict, prov: dict):
    """``rewrite`` hint -> (graph, {fused/region specs the edited graph needs}). Skips with the reason when the patch does not
    apply to this graph (stale base: another accepted hint changed an op the rewrite removes, or an op name is taken)."""
    from .kb_proposals import with_fused_specs
    patch = (h.get("rewrite") or {}).get("patch")
    if not patch:
        prov["skipped"].append({"id": h["id"], "why": "rewrite hint without a patch"})
        return graph, {}
    try:
        g2, info = ge.apply_patch(graph, patch)
    except ge.PatchError as e:
        prov["skipped"].append({"id": h["id"], "why": f"rewrite does not apply: {e}"})
        return graph, {}
    sp = with_fused_specs(g2, specs)
    added = {k: v for k, v in sp.items() if k not in specs}
    new = [o for o in g2["ops"] if o["name"] in set(info["added"])]
    if new:
        new[0]["hint"] = h["id"]
    olds = info["removed"]
    for i, n in enumerate(olds):
        if info["added"]:
            prov["renamed"][n] = info["added"][min(i, len(info["added"]) - 1)]
    prov.setdefault("added", []).extend(info["added"])
    prov["applied"].append({"id": h["id"], "kind": "rewrite", "op": info["added"][0] if info["added"] else None, "from": olds,
                            "added": info["added"], "derived_weights": sorted(patch.get("weight_derivations", {}))})
    return g2, added
