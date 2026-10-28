"""Offline REWRITE DISCOVERY: run a proposer of free-form graph rewrites over problem IRs, apply each rewrite in the restricted sandbox,
gate it with graph_verify, optionally time it with a paired evaluator, and write the ACCEPTED ones as PATTERN PROPOSALS
(``rewrite_proposals.json``) for a human / agent to promote into real hint kinds (as ``merge_sibling_convs`` was promoted from the pilot's
``merge_expand_fused``). Nothing is ever applied to a flow, a hints file or an IR: this is a discovery tool, flag-gated, not part of the sweep.

    python -m modelblaster.pipeline.graph_rewrite_discover IR_DIR [IR_DIR ...] --out OUT_DIR --enable-discovery \\
        --proposer scripted:SCRIPT_DIR | module:Class  [--evaluator none | fake:100,80 | module:Class]
        [--accept-gain 0.03] [--allow-untimed] [--seeds 2] [--device cpu] [--max-candidates 3]

(or ``MB_REWRITE_DISCOVERY=on`` instead of ``--enable-discovery``). Outputs under OUT_DIR::

    rewrite_proposals.json   the accepted pattern proposals (format modelblaster.rewrite_proposals/v1; merged across runs by pattern id)
    rewrite_attempts.json    every attempt: sandbox / verify / timing outcome with reasons (the rejected hypotheses)
    work/<ir>/<script sha>/  the edited IR dirs

Interfaces (the LLM side lives elsewhere; this module only fixes the seam)
-------------------------------------------------------------------------
:class:`ToolLoop`      ``run(session) -> [ScriptProposal]``: an iterative proposer (an LLM over the relay in a tool loop, e.g. the region-loop
                       module) that may call ``session.show()`` and ``session.try_rewrite(source)`` (sandbox + structural gate, structured result
                       to repair against) any number of times within ``session.budget`` and returns the scripts it wants evaluated.
:class:`PairedEvaluator` ``measure(orig_ir, variant_ir, tag) -> {"original_us", "variant_us"}``: the paired variant-vs-original timing (the flow's
                       ``kb_graph_opt`` / ``kb_proposals.remeasure`` path on a GPU; :class:`FakeEvaluator` in tests; none = untimed).
:class:`ScriptedToolLoop` the stub: returns scripts it was given (tests, the CLI's ``scripted:DIR``, replay of an agent's saved scripts).
:class:`ToolLoopProposer` adapts a ToolLoop to ``graph_rewrite.RewriteProposer`` (so the same loop can serve ``--graph-rewrite agent`` through
                       ``module:Class``: verified sandbox results become RewriteCandidates).
"""
from __future__ import annotations

import abc
import argparse
import copy
import hashlib
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import graph_edit as ge
from . import graph_rewrite as grm
from . import graph_rewrite_sandbox as sb
from . import weight_derivations as wd

ENV = "MB_REWRITE_DISCOVERY"
PROPOSALS_FORMAT = "modelblaster.rewrite_proposals/v1"
SOURCE = "agent-rewrite"


# ------------------------------------------------------------------------------ interfaces
@dataclass
class ScriptProposal:
    """One rewrite a proposer wants evaluated."""
    name: str
    source: str
    description: str = ""
    est_gain_pct: float | None = None
    tensor_map: dict | None = None
    provenance: dict = field(default_factory=dict)       # who / how (model, session, tool calls): copied into the evidence


@dataclass
class DiscoverySession:
    """What a ToolLoop works with for one IR."""
    ir: dict                                             # {"name", "dir", "graph"}
    profile: dict | None
    budget: dict                                         # {"max_candidates", "llm_usd", "work"}
    weights: dict = field(repr=False, default_factory=dict)
    tries: list = field(default_factory=list)
    gate: bool = True                                    # try_rewrite also runs the structural gate

    def show(self) -> dict:
        """A compact read-only overview (ops by profile time when there is one, dataflow degree, derived-weight catalogue)."""
        g = self.ir["graph"]
        us = {o["name"]: o.get("us") or 0.0 for o in (self.profile or {}).get("ops", [])}
        ops = sorted(g["ops"], key=lambda o: -us.get(o["name"], 0.0))
        return {"name": self.ir["name"], "n_ops": len(g["ops"]), "e2e_us": (self.profile or {}).get("e2e_us"),
                "ops": [{"name": o["name"], "op": o["op"], "inputs": o["inputs"], "outputs": o["outputs"], "us": us.get(o["name"]),
                         "shape": o.get("shape"), "weight": o.get("weight"), "bias": o.get("bias")} for o in ops[:80]],
                "weights": {k: list(v.shape) for k, v in list(self.weights.items())[:200]}}

    def try_rewrite(self, source: str, tensor_map: dict | None = None) -> dict:
        """Sandbox the script and (structurally) gate the result: the structured outcome a tool loop repairs against. Never raises."""
        res = sb.run_rewrite(source, self.ir["graph"], self.weights, tensor_map=tensor_map)
        out = {"ok": res.ok, "error": res.error, "description": res.description, "log": res.log, "steps": res.steps}
        if res.ok:
            out.update(ops_removed=res.patch["remove_ops"], ops_added=[o["name"] for o in res.patch["add_ops"]], derivations=res.provenance)
            if self.gate:
                from . import graph_verify as gv
                d = Path(self.budget["work"]) / "try" / res.script_sha
                v = gv.verify(self.ir["dir"], sb.write_result_ir(res, self.ir["dir"], d), tensor_map=res.tensor_map or None, structural_only=True)
                out["structural"] = {"ok": v.ok, "reasons": v.reasons[:6], "warnings": [w["code"] for w in v.warnings]}
                out["ok"] = v.ok
        self.tries.append({k: out[k] for k in ("ok", "error")})
        return out


class ToolLoop(abc.ABC):
    """An iterative proposer of rewrite scripts (the LLM side). ``run`` may call ``session.show()`` / ``session.try_rewrite(src)`` freely
    within ``session.budget`` and returns the scripts to evaluate (the discovery pipeline sandboxes, gates and times them itself, so
    a loop that returns an unverified script costs nothing worse than a rejected attempt)."""
    name = "toolloop"

    @abc.abstractmethod
    def run(self, session: DiscoverySession) -> list[ScriptProposal]:
        raise NotImplementedError


class ScriptedToolLoop(ToolLoop):
    """Stub: returns the scripts it was given (for the IR names it is scoped to, all when ``by_ir`` is None)."""
    name = "scripted"

    def __init__(self, scripts: list[ScriptProposal] | None = None, by_ir: dict | None = None):
        self.scripts, self.by_ir = list(scripts or []), by_ir

    @classmethod
    def from_dir(cls, d) -> "ScriptedToolLoop":
        """Every ``*.py`` in ``d`` is one script; an optional ``# description: ...`` first-comment line is its hypothesis."""
        out = []
        for p in sorted(Path(d).glob("*.py")):
            src = p.read_text()
            desc = next((ln.split(":", 1)[1].strip() for ln in src.splitlines()[:5] if ln.lower().startswith("# description:")), "")
            out.append(ScriptProposal(name=p.stem, source=src, description=desc, provenance={"file": p.name}))
        return cls(out)

    def run(self, session: DiscoverySession) -> list[ScriptProposal]:
        s = self.by_ir.get(session.ir["name"], []) if self.by_ir is not None else self.scripts
        return list(s)[: int(session.budget.get("max_candidates", 3))]


class PairedEvaluator(abc.ABC):
    """Paired variant-vs-original timing of an edited IR (the flow's ``remeasure`` on a GPU). ``measure`` returns
    ``{"original_us": float, "variant_us": float}`` (medians of back-to-back runs) or raises / returns ``None`` when it cannot time."""
    name = "evaluator"

    @abc.abstractmethod
    def measure(self, orig_ir: Path, variant_ir: Path, tag: str) -> dict | None:
        raise NotImplementedError


class FakeEvaluator(PairedEvaluator):
    """Tests: a fixed original / variant time, or per-script ``{script_sha_or_tag: (orig, var)}``."""
    name = "fake"

    def __init__(self, original_us: float = 100.0, variant_us: float = 80.0, per_tag: dict | None = None):
        self.o, self.v, self.per = original_us, variant_us, per_tag or {}
        self.calls: list[str] = []

    def measure(self, orig_ir, variant_ir, tag):
        self.calls.append(tag)
        o, v = self.per.get(tag, (self.o, self.v))
        return {"original_us": o, "variant_us": v, "remeasured": {"original_us": [o], "variant_us": [v], "evaluator": "fake"}}


class ToolLoopProposer(grm.RewriteProposer):
    """``graph_rewrite.RewriteProposer`` over a ToolLoop: each verified sandbox result becomes a RewriteCandidate, so a tool loop serves
    ``--graph-rewrite agent --graph-rewrite-proposer module:Class`` as well as this module's discovery run."""
    name = "toolloop"

    def __init__(self, loop: ToolLoop | None = None):
        self.loop = loop

    def propose(self, ir, profile, budget):
        if self.loop is None:
            raise NotImplementedError("ToolLoopProposer needs a ToolLoop (the LLM tool loop lives in the region-loop module)")
        w = sb.load_ir(ir["dir"])[1]
        sess = DiscoverySession({"name": ir["name"], "dir": ir["dir"], "graph": ir["graph"]}, profile, budget, w)
        out = []
        for sp in self.loop.run(sess)[: int(budget.get("max_candidates", 3))]:
            res = sb.run_rewrite(sp.source, ir["graph"], w, tensor_map=sp.tensor_map)
            if not res.ok:
                continue
            d = sb.write_result_ir(res, ir["dir"], Path(budget["work"]) / res.script_sha)
            out.append(grm.RewriteCandidate(ir=ir["name"], edited_ir_dir=d, edit_type="rewire", description=sp.description or res.description,
                                            provenance={"proposer": self.loop.name, "script_sha": res.script_sha, **sp.provenance},
                                            est_gain_pct=sp.est_gain_pct, tensor_map=res.tensor_map,
                                            est_cost={"ops_before": len(ir["graph"]["ops"]), "ops_after": len(res.graph["ops"]),
                                                      "derived_weights": sorted(res.derivations)}))
        return out


# ------------------------------------------------------------------------------ pattern description
def _labels(ops: list[dict], boundary_in: dict, boundary_out: dict):
    """Local dataflow of an op set with tensors renamed: external inputs x0.., region outputs y0.., internals t0.. (by first appearance)."""
    names: dict[str, str] = {}
    cnt = {"x": 0, "y": 0, "t": 0}

    def lab(t):
        if t not in names:
            k = "y" if t in boundary_out else "x" if t in boundary_in else "t"
            names[t] = f"{k}{cnt[k]}"
            cnt[k] += 1
        return names[t]
    out = []
    for o in ops:
        out.append({"op": o["op"], "inputs": [lab(t) for t in o.get("inputs", [])], "outputs": [lab(t) for t in o.get("outputs", [])],
                    "weight": bool(o.get("weight")), "bias": bool(o.get("bias")),
                    "shape_params": sorted((o.get("shape") or {}).keys())})
    return out, names


def _same_params(removed: list[dict]) -> list[dict]:
    """Per op kind appearing at least twice among the removed ops: shape fields equal across them / differing (a detector's match constraints)."""
    by: dict[str, list[dict]] = {}
    for o in removed:
        by.setdefault(o["op"], []).append(o)
    out = []
    for kind, os_ in by.items():
        if len(os_) < 2:
            continue
        keys = sorted(set.intersection(*[set((o.get("shape") or {}).keys()) for o in os_]))
        eq = [k for k in keys if len({json.dumps(o["shape"][k]) for o in os_}) == 1]
        ne = [k for k in keys if k not in eq]
        out.append({"op": kind, "count": len(os_), "equal_fields": eq, "differing_fields": ne})
    return out


def _pad_formulas(res: sb.SandboxResult, removed: list[dict]) -> list[dict]:
    """Parameterise the pad amounts of derivations that pad a REMOVED conv's weight: ``(K_max - K_i) // 2`` per spatial axis when it matches."""
    out = []
    byw = {o.get("weight"): o for o in removed if o.get("weight")}
    kmax = {ax: max((o["shape"].get(ax, 1) for o in removed if "KH" in (o.get("shape") or {})), default=None) for ax in ("KH", "KW")}
    for k, rec in res.derivations.items():
        e = rec["expr"]
        if e.get("op") == "pad" and len(e.get("args", [])) == 1 and "w" in e["args"][0] and e["args"][0]["w"] in byw:
            o = byw[e["args"][0]["w"]]
            f = {}
            for axis, ax in ((2, "KH"), (3, "KW")):
                lo, hi = e["pads"][axis]
                if kmax[ax] is not None and lo == hi == (kmax[ax] - o["shape"][ax]) // 2:
                    f[f"axis{axis}"] = f"({ax}_max - {ax}[{o['name']}]) // 2"
            out.append({"derived": k, "pad_of": e["args"][0]["w"], "pads": e["pads"], "formula": f})
    return out


def describe_pattern(g0: dict, res: sb.SandboxResult) -> dict:
    """before / after op signatures, the replayable recipe and a minimal detector sketch of an accepted rewrite."""
    from .graph_hints import unique_names
    g = unique_names(copy.deepcopy(g0))
    by_name = {o["name"]: o for o in g["ops"]}
    removed = [by_name[n] for n in res.patch["remove_ops"] if n in by_name]
    added = copy.deepcopy(res.patch["add_ops"])
    rm_names = {o["name"] for o in removed}
    produced_in = {t for o in removed for t in o.get("outputs", [])}
    readers_out = set()
    for o in g["ops"]:
        if o["name"] not in rm_names:
            readers_out.update(t for t in o.get("inputs", []) if t in produced_in)
    gouts = set(g["output"].get("tensors") or [g["output"]["tensor"]])
    region_out = {t for t in produced_in if t in readers_out or t in gouts}
    ext_in = {t for o in removed for t in o.get("inputs", []) if t not in produced_in}
    before, names = _labels(removed, ext_in, region_out)
    add_out = {t for o in added for t in o.get("outputs", []) if t in region_out or t in gouts}
    add_in = {t for o in added for t in o.get("inputs", []) if t not in {x for a in added for x in a.get("outputs", [])}}
    after, names2 = _labels(added, add_in, add_out)
    # align the after-labels of the shared boundary tensors with the before-labels (same tensor name => same label)
    shared = {t: names[t] for t in names if t in names2 and (t in ext_in or t in region_out)}
    inv = {names2[t]: l for t, l in shared.items()}
    after = [{**o, "inputs": [inv.get(x, x) for x in o["inputs"]], "outputs": [inv.get(x, x) for x in o["outputs"]]} for o in after]
    sig = hashlib.sha1(json.dumps({"before": before, "after": after}, sort_keys=True).encode()).hexdigest()[:10]
    kinds = [o["op"] for o in removed]
    base = lambda k: k.removesuffix("_f16").removesuffix("_bf16")  # noqa: E731
    has_cat = any(base(k).startswith(("cat", "catn")) for k in kinds)
    convs = sum(base(k).startswith("conv2d") for k in kinds)
    suggested = "merge_sibling_convs_cat" if has_cat and convs >= 2 and sum(base(o["op"]).startswith("conv2d") for o in added) == 1 \
        else f"rewrite_{'_'.join(sorted({base(k) for k in kinds}))[:40]}_{sig[:6]}"
    implemented = None
    try:
        from . import merge_convs as mc
        plans, _ = mc.find(g0, mc.Policy(max_waste=1e9, compute_share=0.0))
        want = {o["name"] for o in removed}
        if any(set(p["convs"] + p["acts"] + [p["cat"]]) == want for p in plans):
            implemented = mc.KIND
    except Exception:  # noqa: BLE001
        pass
    anchor = next((o for o in reversed(removed) if any(t in region_out for t in o.get("outputs", []))), removed[-1] if removed else None)
    sketch = {
        "anchor_op": anchor["op"] if anchor else None,
        "match": [f"{o['op']}({', '.join(o['inputs'])}) -> {', '.join(o['outputs'])}" for o in before],
        "exclusive_tensors": sorted({v for v in names.values() if v.startswith("t")}),
        "shape_constraints": _same_params(removed),
        "notes": ["internal tensors (t*) must be read only inside the matched ops and not be graph outputs",
                  "boundary tensors: x* = inputs read from outside, y* = outputs the rest of the graph reads (names kept by the rewrite)"],
    }
    leaf_roles = {}
    for rec in res.provenance.values():
        for leaf in rec["leaves"]:
            o = next((r for r in removed if leaf in (r.get("weight"), r.get("bias"))), None)
            leaf_roles[leaf] = (f"{'weight' if o and o.get('weight') == leaf else 'bias'} of the removed {o['op']} op {o['name']!r}" if o
                                else "a model parameter the matched ops do not own")
    return {"signature": sig, "suggested_kind": suggested, "already_implemented": implemented,
            "pattern": {"description": res.description, "before": before, "after": after,
                        "ops_before": len(removed), "ops_after": len(added), "removed_ops": [o["name"] for o in removed],
                        "added_ops": [o["name"] for o in added]},
            "recipe": {"script": None, "script_sha": res.script_sha,
                       "derivations": json.loads(json.dumps(res.derivations)), "provenance": res.provenance, "leaf_roles": leaf_roles,
                       "pad_formulas": _pad_formulas(res, removed),
                       "exact": all(r["exact"] for r in res.provenance.values()) if res.provenance else True,
                       "parameterisation": "the script reads every shape from g / T / W, so it replays on other shapes; the derivations above are the "
                                           "concrete instance (kernel sizes / channel counts of THIS problem)"},
            "detector_sketch": sketch}


# ------------------------------------------------------------------------------ the discovery run
def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def merge_proposal(doc: dict, desc: dict, evidence: dict, script: str, proposer: str) -> dict:
    pid = "rwp:" + desc["signature"]
    p = next((x for x in doc["proposals"] if x["id"] == pid), None)
    if p is None:
        p = {"id": pid, "source": SOURCE, "status": "proposed", "suggested_kind": desc["suggested_kind"],
             "already_implemented": desc["already_implemented"], "pattern": desc["pattern"],
             "recipe": {**desc["recipe"], "script": script}, "detector_sketch": desc["detector_sketch"],
             "evidence": [], "proposers": [], "note": "a PROPOSAL: nothing here is applied; promote it to a hint kind (detector + applier + tests, like "
                                                      "merge_sibling_convs) after review"}
        doc["proposals"].append(p)
    p["evidence"].append(evidence)
    if proposer not in p["proposers"]:
        p["proposers"].append(proposer)
    return p


def load_proposals(path: Path) -> dict:
    if Path(path).exists():
        d = json.loads(Path(path).read_text())
        if d.get("format") == PROPOSALS_FORMAT:
            return d
    return {"format": PROPOSALS_FORMAT, "proposals": []}


def discover(ir_dirs, loop: ToolLoop, out_dir, *, evaluator: PairedEvaluator | None = None, accept_gain: float = 0.03,
             allow_untimed: bool = False, seeds: int = 2, device: str = "cpu", max_candidates: int = 3, llm_usd: float | None = None,
             profile_for=None, gate_try: bool = True) -> dict:
    """The discovery loop. -> summary {"attempts", "accepted", "proposals": path, ...}. Never raises on a bad candidate."""
    from . import graph_verify as gv
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prop_path = out / "rewrite_proposals.json"
    doc = load_proposals(prop_path)
    attempts = []
    for d in map(Path, ir_dirs):
        g0, w0 = sb.load_ir(d)
        name = g0["name"]
        prof = profile_for(d) if profile_for else (json.loads((d / "profile.json").read_text()) if (d / "profile.json").exists() else None)
        work = out / "work" / name
        sess = DiscoverySession({"name": name, "dir": str(d), "graph": g0}, prof, {"max_candidates": max_candidates, "llm_usd": llm_usd, "work": str(work)},
                                w0, gate=gate_try)
        work.mkdir(parents=True, exist_ok=True)
        try:
            scripts = list(loop.run(sess) or [])
        except NotImplementedError:
            raise
        except Exception as e:  # noqa: BLE001 -- a proposer never breaks the run
            attempts.append({"ir": name, "stage": "proposer", "decision": "error", "error": f"{type(e).__name__}: {e}"[:400]})
            continue
        for sp in scripts[:max_candidates]:
            att = {"ir": name, "script": sp.name, "description": sp.description, "provenance": sp.provenance, "ts": _now()}
            attempts.append(att)
            res = sb.run_rewrite(sp.source, g0, w0, tensor_map=sp.tensor_map)
            att["script_sha"] = res.script_sha
            if not res.ok:
                att.update(stage="sandbox", decision="rejected", error=res.error, log=res.log)
                continue
            edir = sb.write_result_ir(res, d, work / res.script_sha)
            v = gv.verify(d, edir, tensor_map=res.tensor_map or None, seeds=tuple(range(seeds)), device=device)
            att["verify"] = {"ok": v.ok, "summary": v.summary(), "checks": {k: c["ok"] for k, c in v.checks.items()}, "max_errors": v.max_errors,
                             "warnings": [x["code"] for x in v.warnings]}
            if not v.ok:
                att.update(stage="graph_verify", decision="rejected", reasons=v.reasons[:8])
                continue
            tim = None
            if evaluator is not None:
                try:
                    tim = evaluator.measure(d, edir, res.script_sha)
                except Exception as e:  # noqa: BLE001
                    att["timing_error"] = f"{type(e).__name__}: {e}"[:300]
            if tim and tim.get("original_us") and tim.get("variant_us"):
                gain = (tim["original_us"] - tim["variant_us"]) / tim["original_us"]
                att["timing"] = {**tim, "gain": gain}
                decision = "accepted" if gain >= accept_gain else "rejected"
                att.update(stage="timing", decision=decision, gain=gain)
                if decision == "rejected":
                    att["why"] = f"gain {gain:.3f} < accept_gain {accept_gain}"
                    continue
            elif allow_untimed:
                att.update(stage="verified_untimed", decision="accepted", gain=None)
            else:
                att.update(stage="verified_untimed", decision="not_accepted", why="no paired timing (pass an evaluator, or --allow-untimed to list verified rewrites)")
                continue
            desc = describe_pattern(g0, res)
            att["pattern_id"] = "rwp:" + desc["signature"]
            ev = {"problem": name, "ir": str(d), "original_us": (tim or {}).get("original_us"), "variant_us": (tim or {}).get("variant_us"),
                  "gain": att.get("gain"), "decision": att["decision"], "timed": bool(tim),
                  "verify": {"ok": True, "checks": att["verify"]["checks"], "max_errors": v.max_errors, "warnings": att["verify"]["warnings"]},
                  "script_sha": res.script_sha, "provenance": sp.provenance, "ts": att["ts"]}
            merge_proposal(doc, desc, ev, sp.source, getattr(loop, "name", "toolloop"))
    prop_path.write_text(json.dumps(doc, indent=1, default=str))
    (out / "rewrite_attempts.json").write_text(json.dumps({"format": "modelblaster.rewrite_attempts/v1", "attempts": attempts}, indent=1, default=str))
    acc = [a for a in attempts if a.get("decision") == "accepted"]
    return {"attempts": len(attempts), "accepted": len(acc), "rejected": sum(a.get("decision") == "rejected" for a in attempts),
            "by_stage": {s: sum(a.get("stage") == s for a in attempts) for s in sorted({a.get("stage") for a in attempts if a.get("stage")})},
            "proposals": str(prop_path), "proposal_ids": [p["id"] for p in doc["proposals"]], "attempts_file": str(out / "rewrite_attempts.json")}


# ------------------------------------------------------------------------------ CLI
def _load_obj(spec: str, what: str):
    mod, _, cls = spec.partition(":")
    try:
        return getattr(importlib.import_module(mod), cls)()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"cannot load {what} {spec!r}: {type(e).__name__}: {e}")


def make_loop(spec: str) -> ToolLoop:
    if spec.startswith("scripted:"):
        return ScriptedToolLoop.from_dir(spec.split(":", 1)[1])
    if spec in ("llm", "agent"):
        raise SystemExit("the LLM tool loop is not part of this module (it lives with the region-loop tool-loop module): pass module:Class "
                         "implementing graph_rewrite_discover.ToolLoop, or scripted:DIR for saved scripts")
    return _load_obj(spec, "proposer")


def make_evaluator(spec: str) -> PairedEvaluator | None:
    if spec in ("", "none"):
        return None
    if spec.startswith("fake"):
        nums = spec.split(":", 1)[1].split(",") if ":" in spec else ["100", "80"]
        return FakeEvaluator(float(nums[0]), float(nums[1]))
    return _load_obj(spec, "evaluator")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("irs", nargs="+", help="problem IR dirs (graph.json, weights.npz, io.npz; optional profile.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--proposer", required=True, help="scripted:DIR | module:Class (a ToolLoop with a no-argument constructor)")
    ap.add_argument("--evaluator", default="none", help="none | fake[:orig,var] | module:Class (a PairedEvaluator)")
    ap.add_argument("--enable-discovery", action="store_true", help=f"required (or ${ENV}=on): discovery is not part of the normal flow")
    ap.add_argument("--accept-gain", type=float, default=0.03)
    ap.add_argument("--allow-untimed", action="store_true", help="list verified rewrites as proposals without a paired timing")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-candidates", type=int, default=3)
    ap.add_argument("--llm-usd", type=float, default=None, help="cost cap handed to the proposer (budget['llm_usd'])")
    a = ap.parse_args(argv)
    if not (a.enable_discovery or os.environ.get(ENV, "off").lower() in ("on", "1", "true", "yes")):
        print(f"graph_rewrite_discover is an offline discovery mode, off by default: pass --enable-discovery (or set {ENV}=on)", file=sys.stderr)
        return 2
    s = discover(a.irs, make_loop(a.proposer), a.out, evaluator=make_evaluator(a.evaluator), accept_gain=a.accept_gain,
                 allow_untimed=a.allow_untimed, seeds=a.seeds, device=a.device, max_candidates=a.max_candidates, llm_usd=a.llm_usd)
    print(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
