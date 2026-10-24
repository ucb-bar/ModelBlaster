"""KernelSpec proposals from KernelBlaster: build, verify and decide fusions.

KernelBlaster proposes (format ``kb-proposal/1``) when its profile says two
adjacent slots would be faster as one kernel -- typically a GEMM/conv
followed by a memory-bound epilogue that is the only reader of its output.
ModelBlaster owns the graph, so it builds the variant:

* the fused op is an IR op ``fused__<A>__<B>[__<param>]`` with ``sub_ops``
  [A, B] and ``internal`` [t] (A's output, B's input);
* its KernelSpec signature is A's parameters (``a_`` prefixed) followed by
  B's parameters minus the one that read t (``b_`` prefixed). A's output
  pointer stays in the signature as scratch, so the seed -- call A's kernel,
  then B's -- is correct by construction, and a truly fused kernel may
  simply not write it;
* the torch IR walker runs ``sub_ops`` in order, so references (and the
  partition golden) come from the same oracle as every other op.

The variant partition is exported like any problem; KernelBlaster optimises
it; :func:`decide` compares the variant's best partition time against the
original's and records the outcome in the fusion registry
(``kernels/cuda/fusion_registry.json``). Accepted fusions are what a later
export may apply (``apply_fusions``); promoting one to ``KERNEL_SPECS``
stays a reviewed change.

    python -m modelblaster.pipeline.kb_proposals variant PROPOSALS.json ORIG_PROBLEM_DIR OUT_DIR
    python -m modelblaster.pipeline.kb_proposals decide ORIG_RESULT.json VARIANT_RESULT.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .kernel_abi import Param, bind, parse_signature
from .reference_kernels import KernelSpec

MB_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = MB_ROOT / "kernels" / "cuda" / "fusion_registry.json"
ACCEPT_GAIN = 0.03               # variant must beat the original by >= 3%


@dataclass
class FusedKernelSpec(KernelSpec):
    sub_ops: list = field(default_factory=list)          # [kind_a, kind_b]
    t_param: str = ""                                    # B's param that read A's output

    def cuda_seed(self, lib) -> str:
        """Seed slot source: A's and B's hand kernels, called back to back."""
        specs = _all_specs()
        a_spec, b_spec = specs[self.sub_ops[0]], specs[self.sub_ops[1]]
        a_entry, pa = parse_signature(a_spec.signature)
        b_entry, pb = parse_signature(b_spec.signature)
        names = lib.closure(a_entry) + [n for n in lib.closure(b_entry) if n not in lib.closure(a_entry)]
        names = [k for k in lib.items if k in set(names)]
        helpers = "\n\n".join(lib.items[k].text for k in names)
        entry, params = parse_signature(self.signature)
        a_out = next(p.name for p in pa if p.is_ptr and not p.is_const_ptr)
        a_args = ", ".join("a_" + p.name for p in pa)
        b_args = ", ".join(f"a_{a_out}" if p.name == self.t_param else "b_" + p.name for p in pb)
        plist = ", ".join(f"{p.ctype} {p.name}" for p in params)
        return (f"{helpers}\n\n// Seed: {self.sub_ops[0]} then {self.sub_ops[1]}; `a_{a_out}` is the\n"
                f"// intermediate (scratch). A fused kernel need not write it.\n"
                f"void {entry}({plist}) {{\n    {a_entry}({a_args});\n    {b_entry}({b_args});\n}}\n")


_SPECS_CACHE: dict = {}


def _all_specs() -> dict:
    if not _SPECS_CACHE:
        from .reference_kernels import KERNEL_SPECS
        _SPECS_CACHE.update(KERNEL_SPECS)
    return _SPECS_CACHE


def _t_param(b_op: dict, t: str, specs: dict) -> str:
    _, pb = parse_signature(specs[b_op["op"]].signature)
    return next(a.param.name for a in bind(b_op, pb) if a.kind == "in" and a.value == t)


def fused_kind(a_kind: str, b_kind: str, t_param: str, specs: dict) -> str:
    _, pb = parse_signature(specs[b_kind].signature)
    n_in = sum(1 for p in pb if p.is_const_ptr and p.name not in ("weight", "bias", "scale", "gamma", "beta"))
    return f"fused__{a_kind}__{b_kind}" + (f"__{t_param}" if n_in > 1 else "")


def make_fused_spec(a_kind: str, b_kind: str, t_param: str, specs: dict) -> FusedKernelSpec:
    a, b = specs[a_kind], specs[b_kind]
    _, pa = parse_signature(a.signature)
    _, pb = parse_signature(b.signature)
    kind = fused_kind(a_kind, b_kind, t_param, specs)
    params = [Param(p.ctype, "a_" + p.name) for p in pa] + \
             [Param(p.ctype, "b_" + p.name) for p in pb if p.name != t_param]
    sig = f"void kernel_{kind}(" + ", ".join(f"{p.ctype} {p.name}" for p in params) + ");"
    a_out = next(p.name for p in pa if p.is_ptr and not p.is_const_ptr)
    sem = (f"{a_kind} followed by {b_kind}: computes {a_kind} (parameters a_*) into a_{a_out}, then "
           f"{b_kind} (parameters b_*) reading that result as its `{t_param}`. Only the {b_kind} output "
           f"is checked; a_{a_out} is scratch.\n[{a_kind}] {a.semantics}\n[{b_kind}] {b.semantics}")
    ref = (f"/* reference = the two reference kernels in sequence */\n{a.reference_impl}\n\n{b.reference_impl}\n")
    return FusedKernelSpec(op=kind, signature=sig, semantics=sem, reference_impl=ref,
                           sub_ops=[a_kind, b_kind], t_param=t_param)


def fuse_pair(graph: dict, a_name: str, b_name: str, specs: dict) -> tuple[dict, FusedKernelSpec]:
    """Return (new graph, fused spec) with ops a_name -> b_name fused."""
    g = copy.deepcopy(graph)
    ops = g["ops"]
    ia = next(i for i, o in enumerate(ops) if o["name"] == a_name)
    ib = next(i for i, o in enumerate(ops) if o["name"] == b_name)
    A, B = ops[ia], ops[ib]
    shared = [t for t in A["outputs"] if t in B["inputs"]]
    if len(A["outputs"]) != 1 or len(shared) != 1:
        raise ValueError(f"{a_name}->{b_name}: need one producer output read by the consumer")
    t = shared[0]
    readers = [o["name"] for o in ops if t in o["inputs"]]
    outs = g["output"].get("tensors") or [g["output"]["tensor"]]
    if readers != [b_name] or t in outs or B["inputs"].count(t) != 1:
        raise ValueError(f"{t}: not exclusively read by {b_name}")
    tp = _t_param(B, t, specs)
    spec = make_fused_spec(A["op"], B["op"], tp, specs)
    fused = {"name": f"{a_name}+{b_name}", "op": spec.op, "inputs": A["inputs"] + [x for x in B["inputs"] if x != t],
             "outputs": list(B["outputs"]), "internal": [t], "sub_ops": [A, B],
             "shape": {"a": A.get("shape", {}), "b": B.get("shape", {})}}
    ops[ib] = fused
    del ops[ia]
    return g, spec


def bind_fused(op: dict, specs: dict) -> list:
    """kernel_abi.bind for a fused op, in the fused signature's order.

    A's output arg becomes kind ``scratch`` (the internal tensor)."""
    A, B = op["sub_ops"]
    _, pa = parse_signature(specs[A["op"]].signature)
    _, pb = parse_signature(specs[B["op"]].signature)
    t = op["internal"][0]
    aa = bind(A, pa)
    for x in aa:
        if x.kind == "out" and x.value == t:
            x.kind = "scratch"
    ab = [x for x in bind(B, pb) if not (x.kind == "in" and x.value == t)]
    return aa + ab


# ---------------------------------------------------------------- variants

def write_variant_ir(ir_dir: Path, graph: dict, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    (out / "graph.json").write_text(json.dumps(graph, indent=1))
    for f in ("weights.npz", "io.npz"):
        dst = out / f
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(Path(ir_dir).resolve() / f, dst)
    return out


def build_variant(proposals: dict, orig_problem: Path, out: Path) -> dict:
    """Apply every pair of a kb-proposal/1 fusion to the original partition's
    IR and export the variant partition problem under ``out/problem``."""
    from .kb_export import export_problem

    assert proposals.get("format") == "kb-proposal/1"
    man = json.loads((Path(orig_problem) / "manifest.json").read_text())
    ir_dir = Path(man["source_ir"])
    specs = dict(_all_specs())
    graph = json.loads((ir_dir / "graph.json").read_text())
    names = list(man["partition"]["op_names"])
    applied, skipped = [], []
    for prop in proposals["proposals"]:
        if prop.get("kind") != "fuse":
            continue
        for a, b, _t in prop["pairs"]:
            try:
                graph, spec = fuse_pair(graph, a, b, specs)
            except (ValueError, StopIteration) as e:
                skipped.append({"pair": [a, b], "why": str(e)})
                continue
            specs[spec.op] = spec
            k = names.index(b)
            names[k] = f"{a}+{b}"
            names.remove(a)
            applied.append({"pair": [a, b], "kind": spec.op, "signature": spec.signature})
    vir = write_variant_ir(ir_dir, graph, out / "ir")
    idx = [i for i, o in enumerate(graph["ops"]) if o["name"] in set(names)]
    prob = export_problem(vir, out / "problem", op_indices=idx, specs=specs, strategy="fusion_variant",
                          name=man["name"] + "__fused")
    rec = {"original_problem": str(orig_problem), "variant_problem": str(prob), "applied": applied,
           "skipped": skipped}
    (out / "variant.json").write_text(json.dumps(rec, indent=1))
    return rec


# ---------------------------------------------------------------- registry + decision

def spec_from_kind(kind: str, specs: dict) -> FusedKernelSpec:
    """Rebuild a fused spec from its kind name (fused__<A>__<B>[__<param>])."""
    parts = kind.split("__")
    if parts[0] != "fused" or len(parts) not in (3, 4):
        raise ValueError(f"not a fused kind: {kind}")
    a, b = parts[1], parts[2]
    if len(parts) == 4:
        tp = parts[3]
    else:
        _, pb = parse_signature(specs[b].signature)
        tp = next(p.name for p in pb if p.is_const_ptr)
    spec = make_fused_spec(a, b, tp, specs)
    assert spec.op == kind, (spec.op, kind)
    return spec


def accepted_fused_specs(registry: Path = DEFAULT_REGISTRY, specs: dict | None = None) -> dict:
    """{kind: FusedKernelSpec} for every accepted fusion in the registry."""
    specs = specs or _all_specs()
    reg = _load_registry(Path(registry))
    return {k: spec_from_kind(k, specs) for k, e in reg["fusions"].items() if e["status"] == "accepted"}


def apply_accepted(graph: dict, registry: Path = DEFAULT_REGISTRY, specs: dict | None = None):
    """Fuse every exclusive producer->consumer pair whose fused kind is
    accepted in the registry. -> (graph, {kind: spec}, [fused op names])."""
    specs = dict(specs or _all_specs())
    accepted = accepted_fused_specs(registry, specs)
    if not accepted:
        return graph, {}, []
    used, done = {}, []
    changed = True
    while changed:
        changed = False
        for o in graph["ops"]:
            for t in o["outputs"]:
                readers = [r for r in graph["ops"] if t in r["inputs"]]
                if len(readers) != 1 or "sub_ops" in o or "sub_ops" in readers[0]:
                    continue
                b = readers[0]
                try:
                    kind = fused_kind(o["op"], b["op"], _t_param(b, t, specs), specs)
                except (KeyError, StopIteration):
                    continue
                if kind not in accepted:
                    continue
                try:
                    graph, spec = fuse_pair(graph, o["name"], b["name"], specs)
                except ValueError:
                    continue
                used[kind] = spec
                done.append(f"{o['name']}+{b['name']}")
                changed = True
                break
            if changed:
                break
    return graph, used, done


def _load_registry(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"format": "mb-fusion-registry/1", "fusions": {}}


def _per_instance_us(result: dict, manifest: dict) -> dict:
    """op name -> its final time: measured per op (driver MB_OP_US) when the
    result has it, else its slot's time / number of instances."""
    if result.get("op_us"):
        return dict(result["op_us"])
    out = {}
    for sl in manifest["slots"]:
        us = (result["slots"].get(sl["id"]) or {}).get("final_us")
        if us is None:
            continue
        for inst in sl["instances"]:
            out[inst["op_name"]] = us / max(len(sl["instances"]), 1)
    return out


def decide(orig_result: Path, variant_result: Path, variant_json: Path | None = None,
           registry: Path = DEFAULT_REGISTRY) -> dict:
    """Accept a fused kind if, after both partitions were optimised, its fused
    slot is faster than the two slots it replaced by >= ACCEPT_GAIN of the
    original partition. Only the touched ops are compared: whole-partition
    bests also move with unrelated slots (e.g. the variant run happening to
    find a better GEMM), which would credit or blame the fusion. The
    partition times are recorded too."""
    o = json.loads(Path(orig_result).read_text())
    v = json.loads(Path(variant_result).read_text())
    vj = Path(variant_json) if variant_json else Path(v["problem_dir"]).parent / "variant.json"
    var = json.loads(vj.read_text())
    ou, vu = o["final_partition_us"], v["final_partition_us"]
    ok = bool(o.get("final_ok") and v.get("final_ok") and ou and vu)
    o_inst = _per_instance_us(o, json.loads((Path(o["problem_dir"]) / "manifest.json").read_text())) if ok else {}
    v_inst = _per_instance_us(v, json.loads((Path(v["problem_dir"]) / "manifest.json").read_text())) if ok else {}
    reg = _load_registry(Path(registry))
    per_kind = {}
    for a in var["applied"]:
        k = per_kind.setdefault(a["kind"], {"signature": a["signature"], "orig_us": 0.0, "fused_us": 0.0,
                                            "pairs": 0, "missing": False})
        x, y = a["pair"]
        f = f"{x}+{y}"
        if x in o_inst and y in o_inst and f in v_inst:
            k["orig_us"] += o_inst[x] + o_inst[y]
            k["fused_us"] += v_inst[f]
            k["pairs"] += 1
        else:
            k["missing"] = True
    out = {"original_us": ou, "variant_us": vu, "partition_gain": (ou - vu) / ou if ok else None, "kinds": {}}
    for kind, k in per_kind.items():
        gain = (k["orig_us"] - k["fused_us"]) / ou if ok and k["pairs"] and not k["missing"] else None
        status = "accepted" if gain is not None and gain >= ACCEPT_GAIN else "rejected"
        e = reg["fusions"].setdefault(kind, {"kind": kind, "signature": k["signature"],
                                             "status": "proposed", "evidence": []})
        e["evidence"].append({"problem": o["problem"], "original_us": ou, "variant_us": vu, "gain": gain,
                              "touched_original_us": round(k["orig_us"], 3), "fused_us": round(k["fused_us"], 3),
                              "pairs": k["pairs"], "decision": status,
                              "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        wins = sum(1 for x in e["evidence"] if x["decision"] == "accepted")
        e["status"] = "accepted" if wins > len(e["evidence"]) - wins else "rejected"
        out["kinds"][kind] = {"status": status, "gain": gain, "touched_original_us": round(k["orig_us"], 3),
                              "fused_us": round(k["fused_us"], 3), "pairs": k["pairs"]}
    Path(registry).parent.mkdir(parents=True, exist_ok=True)
    Path(registry).write_text(json.dumps(reg, indent=1))
    out["status"] = "accepted" if any(x["status"] == "accepted" for x in out["kinds"].values()) else "rejected"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("variant")
    v.add_argument("proposals")
    v.add_argument("orig_problem")
    v.add_argument("out")
    d = sub.add_parser("decide")
    d.add_argument("orig_result")
    d.add_argument("variant_result")
    d.add_argument("--variant-json")
    d.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    a = ap.parse_args()
    if a.cmd == "variant":
        print(json.dumps(build_variant(json.loads(Path(a.proposals).read_text()), Path(a.orig_problem),
                                       Path(a.out)), indent=1))
    else:
        print(json.dumps(decide(Path(a.orig_result), Path(a.variant_result),
                                Path(a.variant_json) if a.variant_json else None, Path(a.registry)), indent=1))


if __name__ == "__main__":
    main()
