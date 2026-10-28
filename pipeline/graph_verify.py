"""Verification gate for graph rewrites (docs/graph_rewrite.md): is the edited IR the same model as the original?

    python -m modelblaster.pipeline.graph_verify ORIG_IR EDITED_IR [--weights ORIG_WEIGHTS.npz] [--seeds 3]
        [--variants uniform,normal,wide,small] [--shape-variant ORIG_IR2:EDITED_IR2 ...] [--sol-bench FILE]
        [--exact auto|on|off] [--conditioning auto|on|off] [--device cpu|cuda] [--structural-only] [--json OUT]

exit status 0 = verdict ok, 1 = rejected. In code::

    v = graph_verify.verify(orig_dir, edited_dir, ...)       # never raises: any failure is a reason
    if not v.ok: reject(v.reasons)                           # [{"code", "msg", "where"}]; v.to_dict() is the record

The checks (every failed check is a reason; ``Verdict.checks`` has one entry per check with its numbers):

Structural (cheap, no walk)
  io          graph input / output declarations and their tensors unchanged; golden io.npz unchanged
  names       op names unique (duplicates the original already had are tolerated), one producer per tensor
  dag         ops in list order are topologically ordered (the walkers and the exporter run them in list order)
  tensors     every tensor registered with positive shape and a known dtype; a name present in both graphs keeps its
              shape and dtype (a name is a value: a rewrite that changes a value gives it a new name)
  dtype       the dtype contract (cuda_dtypes, docs/tensor_dtypes.md): float tensors of an op have the op kind's storage
              dtype; a ``cast_<a>_to_<b>`` reads a and writes b; a single-dtype IR keeps its one float dtype
  weights     every weight an op / constant names exists; each is an ORIGINAL weight (bit-identical to the original
              weights.npz) or a DERIVED one (``weight_derivations``) that replays from the original weights to the stored
              array bit for bit; anything else -- including a weight whose provenance is lost -- is rejected with
              "IR weights not mapped to the Model"
  kinds       every op kind has a kernel spec (fused / region kinds are built from their members) and every op binds
              to its signature; a kind with no spec is a type-3 rewrite (a new kernel spec is a later phase)
  shape       conv2d / linear weights and tensor sizes agree with the op's shape dict
  slots       slot ids are well formed and one kind per entry point (informational: new slots / cache keys)
  dead        ops that reach no output (pruned at export; a warning)

Numerical (torch walkers on the CPU by default; ``walk_torch.walk_free`` frees every activation after its last reader)
  golden      the edited walk on the stored golden input against the stored golden output and the original walk
  outputs     original vs edited on ``seeds`` x ``variants`` random inputs (float inputs drawn per distribution and rounded
              to the tensor's storage dtype; integer / index inputs drawn within the golden range): max |edited - original|
              <= output_rel * max|original| + output_abs + OUTPUT_COND * (the original walk's own error against the exact
              float64 walk)  -- the conditioning-aware bound of kb_export (``reference_error`` / ``exact_values``)
  sampled     SAMPLED INTERMEDIATES: tensors present in both graphs under one name must keep their value. The sample is
              (1) the exits of the edit: tensors produced by an added / changed op that the original also has,
              (2) tensors the rewrite declares recomputable (``tensor_map``: old tensor = slice of a new one),
              (3) an even stratified sample of the other common tensors in topological order, (4) the outputs.
              A model whose output barely depends on its input (SqueezeNet's, < 1e-3 relative across inputs) passes any
              output-only check, so a mismatch in the sample always rejects, and with an input-insensitive output at least
              one exit / declared tensor must be comparable or the rewrite is "unverifiable"
  differential the response of the output to the input (output(x1) - output(x0)) must match, against a noise floor:
              catches a rewrite whose output is dominated by an input-independent term
  exact64     when the weights derived by the rewrite are exact (structural derivations only), edited and original must
              also agree in float64 without store rounding to 1e-9 relative (a rewire is exact math)
  sol         an optional SOL hook: the edited walk on the SOL inputs must pass SOL's own tolerance against
              ``sol_reference_outputs()`` wherever the original does (docs/sol_execbench.md)
  shape variants  ``shape_variants=[(orig_dir, edited_dir), ...]``: the same numeric checks on other shapes of the problem

The shared logic of the pilot's ``gt_edit.equivalence`` (walk_free, sampled intermediates, input sensitivity) is ported
here; the pilot's tree is not imported.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import cuda_dtypes
from . import graph_edit as ge
from . import weight_derivations as wd
from .kernel_abi import WEIGHT_ROLES, BindError, bind, parse_signature

ALIAS_OPS = ("view", "chunk2_c1")
_TENSOR_DTYPES = tuple(cuda_dtypes.TENSOR_DTYPE.values()) + cuda_dtypes.INT_TENSOR
_WEIGHT_FIELDS = tuple(dict.fromkeys(WEIGHT_ROLES.values()))
# constants of the numeric checks
EXACT_REL, EXACT_ABS = 1e-9, 1e-12            # float64 agreement of an exact rewrite
SENSITIVITY_MIN = 1e-3                          # below this relative change across inputs the output is "input-insensitive"
MAX_TRACK_ELEMS = 50_000_000
EXACT_AUTO_MAX_ELEMS = 20_000_000               # exact="auto": float64 walks only when the activations are this small
_EPS = {"fp32": 2.0 ** -23, "fp16": 2.0 ** -10, "bf16": 2.0 ** -7}


# ------------------------------------------------------------------------------ the verdict
@dataclass
class Verdict:
    ok: bool = True
    reasons: list = field(default_factory=list)       # [{"code", "msg", "where"}]: why it was rejected
    warnings: list = field(default_factory=list)
    checks: dict = field(default_factory=dict)        # name -> {"ok", ...numbers}
    max_errors: dict = field(default_factory=dict)    # name -> {"abs", "rel_to_ref", "bound", "tensor"}
    info: dict = field(default_factory=dict)

    def fail(self, code: str, msg: str, where=None) -> None:
        self.ok = False
        self.reasons.append({"code": code, "msg": msg, **({"where": where} if where else {})})

    def warn(self, code: str, msg: str, where=None) -> None:
        self.warnings.append({"code": code, "msg": msg, **({"where": where} if where else {})})

    def check(self, name: str, ok: bool, **kw) -> dict:
        self.checks[name] = {"ok": bool(ok), **kw}
        return self.checks[name]

    @property
    def codes(self) -> list[str]:
        return [r["code"] for r in self.reasons]

    def summary(self) -> str:
        if self.ok:
            return "ok" + (f" ({len(self.warnings)} warning(s))" if self.warnings else "")
        return "rejected: " + "; ".join(f"[{r['code']}] {r['msg']}" for r in self.reasons[:4]) + \
               (f" (+{len(self.reasons) - 4} more)" if len(self.reasons) > 4 else "")

    def to_dict(self) -> dict:
        return {"format": "modelblaster.graph_verify/v1", "ok": self.ok, "reasons": self.reasons, "warnings": self.warnings,
                "checks": self.checks, "max_errors": self.max_errors, "info": self.info}


# ------------------------------------------------------------------------------ loading
def _load_graph(d: Path) -> dict:
    return json.loads((Path(d) / "graph.json").read_text())


def _npz(p) -> dict:
    z = np.load(p)
    return {k: z[k] for k in z.files}


def _graph_weight_keys(g: dict) -> dict[str, list]:
    """weight key -> ops using it (op names), for every weight field of every op / sub-op and every constant tensor."""
    use: dict[str, list] = {}
    for o in g["ops"]:
        for s in [o] + list(o.get("sub_ops", [])):
            for f in _WEIGHT_FIELDS:
                k = s.get(f)
                if isinstance(k, str) and k:
                    use.setdefault(k, []).append(o["name"])
    for t, m in g["tensors"].items():
        if m.get("const"):
            use.setdefault(m["const"], []).append(f"tensor:{t}")
    return use


def _specs_for(g: dict, specs: dict | None) -> dict:
    from .kb_proposals import _all_specs, with_fused_specs
    return with_fused_specs(g, dict(specs or _all_specs()))


# ------------------------------------------------------------------------------ structural checks
def _bind_op(o: dict, specs: dict) -> None:
    """What kb_export does to bind an op to its kernel signature (raises BindError / KeyError)."""
    kind = o["op"]
    if kind.startswith("region__"):
        from .region_slots import bind_region
        bind_region(o, specs)
    elif "sub_ops" in o:
        from .kb_proposals import bind_fused
        bind_fused(o, specs)
    else:
        bind(o, parse_signature(specs[kind].signature)[1])


def _float_dtype_of(g: dict, t: str) -> str | None:
    d = (g["tensors"].get(t) or {}).get("dtype")
    return d if d in cuda_dtypes.FLOAT_TENSOR else None


def _dtype_contract(v: Verdict, g: dict) -> None:
    bad = []
    q = g.get("quant")
    mixed = cuda_dtypes.is_mixed(g)
    if not mixed and q in ("fp16", "bf16", "fp32", None, ""):
        want = cuda_dtypes.TENSOR_DTYPE.get(q, "f32")
        # a precision-policy IR keeps fp32 islands (op kinds without the suffix): per-op rule below covers them
        for t, m in g["tensors"].items():
            if m.get("dtype") in cuda_dtypes.FLOAT_TENSOR and m["dtype"] != want and not _policy_island(g, t):
                bad.append(f"tensor {t!r} is {m['dtype']} in a {q or 'fp32'} IR (every float tensor is {want})")
    for o in g["ops"]:
        for s in (o.get("sub_ops") or [o]):
            kind = s["op"]
            if kind in ALIAS_OPS:
                continue
            c = cuda_dtypes.parse_cast(kind)
            ins = [t for t in s.get("inputs", []) if _float_dtype_of(g, t)]
            outs = [t for t in s.get("outputs", []) if _float_dtype_of(g, t)]
            if c:
                for t in s.get("inputs", []):
                    if g["tensors"].get(t, {}).get("dtype") not in (c[0], None) and not _internal(o, t):
                        bad.append(f"op {s['name']} ({kind}) reads {t!r} of dtype {g['tensors'][t]['dtype']}, expected {c[0]}")
                for t in s.get("outputs", []):
                    if g["tensors"].get(t, {}).get("dtype") not in (c[1], None):
                        bad.append(f"op {s['name']} ({kind}) writes {t!r} of dtype {g['tensors'][t]['dtype']}, expected {c[1]}")
                continue
            if kind.startswith("region__"):
                continue
            sdt = cuda_dtypes.op_dtype(kind)
            wantd = cuda_dtypes.TENSOR_DTYPE[sdt]
            for t in ins + outs:
                if g["tensors"][t]["dtype"] != wantd:
                    bad.append(f"op {s['name']} ({kind}) is {sdt}: tensor {t!r} is {g['tensors'][t]['dtype']}, expected {wantd} "
                               f"(insert an explicit cast_<src>_to_<dst>)")
    if bad:
        v.fail("dtype_contract", "; ".join(bad[:4]) + (f" (+{len(bad) - 4} more)" if len(bad) > 4 else ""))
    v.check("dtype", not bad, violations=len(bad))


def _internal(o: dict, t: str) -> bool:
    return t in (o.get("internal") or [])


def _policy_island(g: dict, t: str) -> bool:
    """fp32 tensors in an fp16 / bf16 IR are legal when every op touching them is an fp32 kind or a precision cast."""
    for o in g["ops"]:
        for s in (o.get("sub_ops") or [o]):
            if t in s.get("inputs", []) or t in s.get("outputs", []):
                k = s["op"]
                if k in ALIAS_OPS or cuda_dtypes.parse_cast(k) or k in ("cast_f16_to_f32", "cast_f32_to_f16") \
                        or cuda_dtypes.op_dtype(k) == cuda_dtypes.FP32:
                    return True
    return False


def _shape_guards(v: Verdict, g: dict, w1: dict) -> None:
    bad = []
    numel = lambda t: int(np.prod(g["tensors"][t]["shape"])) if t in g["tensors"] else None  # noqa: E731
    for o in g["ops"]:
        for s in (o.get("sub_ops") or [o]):
            k = s["op"].removesuffix("_f16").removesuffix("_bf16")
            sh = s.get("shape") or {}
            try:
                if k == "conv2d" and {"IC", "OC", "KH", "KW"} <= set(sh):
                    wk = s.get("weight")
                    gr = sh.get("G", 1)
                    if wk in w1 and list(w1[wk].shape) != [sh["OC"], sh["IC"] // gr, sh["KH"], sh["KW"]]:
                        bad.append(f"op {s['name']}: conv2d weight {wk!r} has shape {list(w1[wk].shape)}, the op's shape says "
                                   f"{[sh['OC'], sh['IC'] // gr, sh['KH'], sh['KW']]}")
                    if s.get("inputs") and numel(s["inputs"][0]) is not None and "N" in sh and "IH" in sh \
                            and numel(s["inputs"][0]) != sh["N"] * sh["IC"] * sh["IH"] * sh["IW"]:
                        bad.append(f"op {s['name']}: input {s['inputs'][0]!r} has {numel(s['inputs'][0])} elements, the op reads "
                                   f"{sh['N'] * sh['IC'] * sh['IH'] * sh['IW']}")
                    if s.get("outputs") and numel(s["outputs"][0]) is not None and "OH" in sh \
                            and numel(s["outputs"][0]) != sh["N"] * sh["OC"] * sh["OH"] * sh["OW"]:
                        bad.append(f"op {s['name']}: output {s['outputs'][0]!r} has {numel(s['outputs'][0])} elements, the op writes "
                                   f"{sh['N'] * sh['OC'] * sh['OH'] * sh['OW']}")
                elif k == "linear" and {"M", "K", "N"} <= set(sh):
                    wk = s.get("weight")
                    if wk in w1 and list(w1[wk].shape) != [sh["N"], sh["K"]]:
                        bad.append(f"op {s['name']}: linear weight {wk!r} has shape {list(w1[wk].shape)}, the op's shape says "
                                   f"{[sh['N'], sh['K']]}")
                    if s.get("outputs") and numel(s["outputs"][0]) is not None and numel(s["outputs"][0]) != sh["M"] * sh["N"]:
                        bad.append(f"op {s['name']}: output {s['outputs'][0]!r} has {numel(s['outputs'][0])} elements, the op writes "
                                   f"{sh['M'] * sh['N']}")
            except (KeyError, TypeError, IndexError):
                continue
    if bad:
        v.fail("shape_guard", "; ".join(bad[:4]) + (f" (+{len(bad) - 4} more)" if len(bad) > 4 else ""))
    v.check("shape", not bad, violations=len(bad))


def structural(v: Verdict, g0: dict, g1: dict, w0: dict, w1: dict, specs: dict | None, *, check_base_values: bool = True) -> dict:
    """All structural checks; fills ``v``; -> {"patch", "walkable"}."""
    import copy as _copy
    from .graph_hints import unique_names
    from .graph_prune import dead_ops

    # ---- io
    in0, out0 = ge.io_names(g0)
    try:
        in1, out1 = ge.io_names(g1)
    except (KeyError, TypeError) as e:
        v.fail("io", f"edited graph has no valid input / output declaration: {e}")
        return {"walkable": False}
    io_ok = g1["input"] == g0["input"] and g1["output"] == g0["output"]
    if not io_ok:
        v.fail("io", "graph input / output declarations changed (they must be identical to the original's)")
    for t in in0 + out0:
        if t not in g1["tensors"]:
            v.fail("io", f"graph io tensor {t!r} is missing from the edited graph's tensors")
            io_ok = False
        elif g1["tensors"][t] != g0["tensors"].get(t):
            v.fail("io", f"graph io tensor {t!r} changed shape/dtype")
            io_ok = False
    v.check("io", io_ok)
    # ---- names
    names1 = [o["name"] for o in g1["ops"]]
    dup1 = {n for n in names1 if names1.count(n) > 1}
    names0 = [o["name"] for o in g0["ops"]]
    dup0 = {n for n in names0 if names0.count(n) > 1}
    new_dup = sorted(dup1 - dup0)
    prod_count: dict[str, int] = {}
    for o in g1["ops"]:
        for t in list(o.get("outputs", [])) + list(o.get("internal", [])):
            prod_count[t] = prod_count.get(t, 0) + 1
    multi = sorted(t for t, c in prod_count.items() if c > 1)
    if new_dup:
        v.fail("names", f"duplicate op names introduced by the rewrite: {new_dup[:6]}")
    if multi:
        v.fail("names", f"tensors with more than one producer: {multi[:6]}")
    v.check("names", not new_dup and not multi)
    # ---- tensors
    bad_t = []
    for t, m in g1["tensors"].items():
        sh = m.get("shape")
        if not isinstance(sh, list) or not all(isinstance(x, int) and x > 0 for x in sh):
            bad_t.append(f"tensor {t!r}: bad shape {sh}")
        if m.get("dtype") not in _TENSOR_DTYPES:
            bad_t.append(f"tensor {t!r}: unknown dtype {m.get('dtype')!r}")
        if t in g0["tensors"] and (g0["tensors"][t].get("shape") != sh or g0["tensors"][t].get("dtype") != m.get("dtype")):
            bad_t.append(f"tensor {t!r} keeps its name but changes shape/dtype {g0['tensors'][t].get('shape')}/"
                         f"{g0['tensors'][t].get('dtype')} -> {sh}/{m.get('dtype')} (a name is a value: give the new value a new name)")
    unreg = []
    for o in g1["ops"]:
        for t in list(o.get("inputs", [])) + list(o.get("outputs", [])) + list(o.get("internal", [])):
            if t not in g1["tensors"]:
                unreg.append(f"op {o['name']}: tensor {t!r} is not registered in tensors")
        for s in o.get("sub_ops", []):
            for t in list(s.get("inputs", [])) + list(s.get("outputs", [])):
                if t not in g1["tensors"]:
                    unreg.append(f"op {s['name']}: tensor {t!r} is not registered in tensors")
    for t in out1:
        if t not in prod_count and t not in in1:
            bad_t.append(f"graph output {t!r} is not produced by any op")
    bad_t += unreg
    if bad_t:
        v.fail("tensors", "; ".join(bad_t[:4]) + (f" (+{len(bad_t) - 4} more)" if len(bad_t) > 4 else ""))
    v.check("tensors", not bad_t, violations=len(bad_t))
    # ---- dag (list order)
    dag_bad = None
    avail = set(in1) | {t for t, m in g1["tensors"].items() if m.get("const")}
    prod_idx = {}
    for i, o in enumerate(g1["ops"]):
        for t in list(o.get("outputs", [])) + list(o.get("internal", [])):
            prod_idx.setdefault(t, i)
    for i, o in enumerate(g1["ops"]):
        for t in o.get("inputs", []):
            if t in avail:
                continue
            if t in prod_idx and prod_idx[t] == i:
                continue
            if t in prod_idx:
                dag_bad = f"op {o['name']!r} (position {i}) reads {t!r}, produced later by {g1['ops'][prod_idx[t]]['name']!r}: ops must be in dependency order"
            else:
                dag_bad = f"op {o['name']!r} reads tensor {t!r} that nothing produces (not a graph input or constant)"
            break
        if dag_bad:
            break
        avail.update(list(o.get("outputs", [])) + list(o.get("internal", [])))
    if dag_bad is None:
        try:
            ge.toposort(_copy.deepcopy(g1))
        except ge.PatchError as e:
            dag_bad = str(e)
    if dag_bad:
        v.fail("dag", dag_bad)
    v.check("dag", dag_bad is None)
    # ---- weights
    _weights(v, g0, g1, w0, w1, check_base_values)
    # ---- kinds + bind
    try:
        sp = _specs_for(g1, specs)
        spec_err = None
    except Exception as e:  # noqa: BLE001 -- a fused / region kind that cannot be built
        sp, spec_err = dict(specs or {}), f"{type(e).__name__}: {e}"
        if not sp:
            from .kb_proposals import _all_specs
            sp = dict(_all_specs())
    kinds_bad = []
    if spec_err:
        kinds_bad.append(f"a fused / region kind cannot be built: {spec_err[:200]}")
    k0 = {o["op"] for o in g0["ops"]}
    new_kinds = sorted({o["op"] for o in g1["ops"]} - k0)
    for o in g1["ops"]:
        if o["op"] in ALIAS_OPS:
            continue
        if o["op"] not in sp:
            kinds_bad.append(f"op {o['name']}: kind {o['op']!r} has no kernel spec (a NEW op kind needs a kernel spec: "
                             f"reference semantics, tolerance, CUDA seed, op class; docs/graph_rewrite.md, edit type 3)")
            continue
        try:
            _bind_op(o, sp)
        except (BindError, KeyError, StopIteration, IndexError, TypeError) as e:
            kinds_bad.append(f"op {o['name']} ({o['op']}): cannot be bound to its kernel signature: {str(e)[:200]}")
    if kinds_bad:
        v.fail("op_kind", "; ".join(kinds_bad[:4]) + (f" (+{len(kinds_bad) - 4} more)" if len(kinds_bad) > 4 else ""))
    v.check("kinds", not kinds_bad, new_kinds=new_kinds)
    # ---- dtype contract, shape guards
    _dtype_contract(v, g1)
    _shape_guards(v, g1, w1)
    # ---- slots / cache keys (informational + collisions)
    _slots(v, g0, g1, sp)
    # ---- dead ops
    try:
        d1 = {g1["ops"][i]["name"] for i in dead_ops(g1)}
        d0 = {g0["ops"][i]["name"] for i in dead_ops(g0)}
        newly = sorted(d1 - d0)
        v.info["dead_ops_edited"] = sorted(d1)[:20]
        if newly:
            v.warn("dead_ops", f"{len(newly)} op(s) of the edited graph reach no output and are pruned at export: {newly[:6]}")
        v.check("dead", True, n_dead=len(d1), newly_dead=len(newly))
    except Exception as e:  # noqa: BLE001
        v.check("dead", True, error=str(e)[:200])
    # ---- the patch (what a rewrite hint stores)
    patch = None
    try:
        patch = ge.diff(g0, g1)
        v.info["patch"] = {"remove_ops": len(patch["remove_ops"]), "add_ops": len(patch["add_ops"]),
                           "add_tensors": len(patch["add_tensors"]), "derived_weights": sorted(patch["weight_derivations"])}
        if patch["changed_tensors"]:
            v.fail("tensors", f"tensors changed shape/dtype in place: {patch['changed_tensors'][:6]}")
        # replayable: the patch applied to the original graph reproduces the edited graph (what a rewrite hint stores)
        try:
            g_re, _ = ge.apply_patch(g0, patch)
            same = ge.graph_digest(g_re) == ge.graph_digest(unique_names(_copy.deepcopy(g1)))
            if not same:
                v.fail("patch_replay", "the edit is not reproducible as a patch: applying diff(original, edited) to the original does "
                                       "not give the edited graph")
            v.check("patch_replay", same)
        except ge.PatchError as e:
            v.fail("patch_replay", f"the edit's patch does not apply to the original graph: {e}")
            v.check("patch_replay", False)
        v.info["ops_before"], v.info["ops_after"] = len(g0["ops"]), len(g1["ops"])
        v.info["new_op_kinds"] = new_kinds
    except Exception as e:  # noqa: BLE001
        v.warn("patch", f"could not diff the graphs: {type(e).__name__}: {e}")
    walkable = all(v.checks.get(c, {}).get("ok", False) for c in ("io", "names", "tensors", "dag", "weights", "kinds"))
    return {"patch": patch, "walkable": walkable, "specs": sp}


def _slots(v: Verdict, g0: dict, g1: dict, sp: dict) -> None:
    from .kb_export import slot_id_of
    bad, slots1, slots0 = [], {}, {}
    sig_of = {}
    for g, out in ((g0, slots0), (g1, slots1)):
        for o in g["ops"]:
            if o["op"] in ALIAS_OPS or o["op"] not in sp:
                continue
            try:
                sid = slot_id_of(o, sp)
            except Exception as e:  # noqa: BLE001
                bad.append(f"op {o['name']}: no slot id ({type(e).__name__}: {e})")
                continue
            if not re.fullmatch(r"[A-Za-z0-9_]+", sid):
                bad.append(f"op {o['name']}: slot id {sid!r} is not a valid identifier")
            out.setdefault(sid, []).append(o["name"])
            if g is g1:
                sig = " ".join(sp[o["op"]].signature.split())
                if sig_of.setdefault(sid.split("__")[0], (sig, o["op"]))[0] != sig and not o.get("slot_variant"):
                    bad.append(f"slot {sid!r} would hold two kernel signatures ({o['op']} vs {sig_of[sid.split('__')[0]][1]})")
    if bad:
        v.fail("slots", "; ".join(bad[:4]))
    new = sorted(set(slots1) - set(slots0))
    shapes0 = {(o["op"], json.dumps(o.get("shape"), sort_keys=True)) for o in g0["ops"]}
    new_keys = sorted({(o["op"], json.dumps(o.get("shape"), sort_keys=True)) for o in g1["ops"]
                       if (o["op"], json.dumps(o.get("shape"), sort_keys=True)) not in shapes0 and o["op"] not in ALIAS_OPS})
    v.info["new_slots"] = new
    v.info["new_cache_keys"] = [{"op": k, "shape": json.loads(s)} for k, s in new_keys][:20]   # kernel-cache misses: seeded by the hand kernel
    v.check("slots", not bad, new_slots=len(new))


def _weights(v: Verdict, g0: dict, g1: dict, w0: dict, w1: dict, check_base_values: bool) -> None:
    use = _graph_weight_keys(g1)
    der = wd.derivations(g1)
    # the original's own derived weights (an earlier accepted rewrite) are not base weights: they replay from the base
    w0 = {k: a for k, a in w0.items() if k not in wd.derivations(g0)}
    problems = []
    unmapped = []
    # derivation table
    errs = wd.validate(g1, set(w0), g0)
    for e in errs:
        v.fail("derivation_" + e["code"], e["msg"], e.get("key"))
        if e["code"] in ("unmapped", "shadows_base"):
            unmapped.append(e["key"])
    replayed = {}
    if der and not errs:
        try:
            replayed = wd.replay(g1, w0, g0)
        except wd.DerivationError as e:
            v.fail("derivation_" + e.code, str(e))
            errs = [{"code": e.code}]
    # original keys of the edited npz must be the originals
    changed, missing = [], []
    for k in sorted(w0):
        if k in der:
            continue
        if k in w1:
            if check_base_values and (w1[k].dtype != w0[k].dtype or w1[k].shape != w0[k].shape or not np.array_equal(w1[k], w0[k])):
                changed.append(k)
        elif k in use:
            missing.append(k)
    if changed:
        v.fail("base_weight_modified", f"original weights changed in the edited weights.npz (a derived weight must have its own d.* key): "
                                       f"{changed[:6]} (IR weights not mapped to the Model)")
    if missing:
        v.fail("weight_missing", f"weights used by the edited graph are missing from its weights.npz: {missing[:6]}")
    for k in sorted(use):
        if k in w0 or k in der:
            continue
        unmapped.append(k)
    unmapped = [k for k in dict.fromkeys(unmapped) if k]
    stale = [k for k in w1 if k not in w0 and k not in der and k not in wd.derivations(g0)]
    if unmapped:
        v.fail("weight_not_mapped", f"IR weights not mapped to the Model (weight provenance lost): {unmapped[:8]}: every weight must be an "
                                    f"original weight or a recorded derivation (weight_derivations) of original weights")
    elif stale:
        v.fail("weight_not_mapped", f"edited weights.npz holds arrays that are neither original weights nor recorded derivations: "
                                    f"{stale[:8]} (IR weights not mapped to the Model: weight provenance lost)")
    # stored derived arrays must be the replay, bit for bit
    mism = []
    for k, arr in replayed.items():
        if k in w1:
            a = w1[k]
            if a.dtype != arr.dtype or a.shape != arr.shape or not np.array_equal(a, arr, equal_nan=True):
                mism.append(k)
    if mism:
        v.fail("derivation_mismatch", f"stored derived weights differ from their derivation replayed on the original weights: {mism[:6]}")
    read_by_derivations = set().union(*[wd.expr_leaves(r['expr']) for r in der.values()]) if der else set()
    unused = sorted(k for k in der if k not in use and k not in read_by_derivations)
    if unused:
        v.warn("unused_derived_weight", f"derived weights no op reads: {unused[:6]}")
    ok = not (errs or changed or missing or unmapped or stale or mism)
    v.check("weights", ok, derived=sorted(der), derived_exact=wd.is_exact(g1) if der else None, rebuilt_by_loader=sorted(k for k in der if k not in w1))
    if der:
        v.info["derived_weights"] = {k: {"dtype": der[k].get("dtype"), "shape": der[k].get("shape")} for k in sorted(der)}


# ------------------------------------------------------------------------------ numerical checks
class _Inputs:
    """``ir`` with its graph inputs replaced (everything else delegated)."""

    def __init__(self, ir, inputs: dict):
        self._ir, self._in = ir, inputs

    def __getattr__(self, k):
        return getattr(self._ir, k)

    def input_tensors(self, device):
        return {k: v.to(device).clone() for k, v in self._in.items()}


def _round_np(a: np.ndarray, dt: str) -> np.ndarray:
    if dt == "f16":
        return a.astype(np.float16).astype(np.float32)
    if dt == "bf16":
        return cuda_dtypes.round_bf16(a.astype(np.float32))
    return a.astype(np.float32)


def make_inputs(ir, variant: str, seed: int):
    """Random graph inputs shaped like ``ir``'s golden ones: floats per ``variant`` (uniform | normal | wide | small | spiky),
    rounded to the tensor's storage dtype; integer-valued (index) inputs drawn uniformly inside the golden range."""
    import torch
    gold = ir.input_tensors("cpu")
    rng = np.random.default_rng([seed, sum(map(ord, variant))])
    out = {}
    for name, t in gold.items():
        dt = ir.tdtype(name)
        a = t.numpy()
        if t.dtype in (torch.int64, torch.int32, torch.uint8, torch.bool) or dt in cuda_dtypes.INT_TENSOR or \
                (a.size and np.all(a == np.round(a)) and np.ptp(a) > 0):
            lo, hi = int(a.min()), int(a.max())
            x = rng.integers(lo, hi + 1, size=a.shape)
            out[name] = torch.from_numpy(x.astype(a.dtype if t.dtype != torch.bool else np.float32))
            continue
        n = a.size
        if variant == "uniform":
            x = rng.random(n, dtype=np.float32)
        elif variant == "normal":
            x = rng.standard_normal(n).astype(np.float32)
        elif variant == "wide":
            x = rng.random(n, dtype=np.float32) * 6.0 - 1.0
        elif variant == "small":
            x = rng.random(n, dtype=np.float32) * 0.05
        elif variant == "spiky":
            x = rng.standard_normal(n).astype(np.float32) * (rng.random(n) < 0.05) * 8.0
        else:
            raise ValueError(f"unknown input variant {variant!r}")
        out[name] = torch.from_numpy(_round_np(x, dt).reshape(a.shape))
    return out


def select_tensors(g0: dict, g1: dict, patch: dict | None, tensor_map: dict | None, n_track: int = 12,
                   max_elems: int = MAX_TRACK_ELEMS) -> dict:
    """The sampled intermediates: {"exits": [...], "mapped": {old: spec}, "strata": [...], "outputs": [...], "dropped": [...]}
    (names; see the module docstring)."""
    _, outs = ge.io_names(g1)
    p0, p1 = ge.produced(g0), ge.produced(g1)
    consts = {t for t, m in g1["tensors"].items() if m.get("const")}
    ok_size = lambda t: 0 < int(np.prod(g1["tensors"][t]["shape"])) <= max_elems  # noqa: E731
    common = [t for t in p1 if t in p0 and t in g0["tensors"] and t not in consts
              and g0["tensors"][t].get("shape") == g1["tensors"][t].get("shape") and ok_size(t)]
    changed = set()
    if patch:
        changed = set(patch["remove_ops"]) | {o["name"] for o in patch["add_ops"]}
    exits = [t for t in common if p1[t] in changed and t not in outs]
    order = {t: i for i, o in enumerate(g1["ops"]) for t in list(o.get("outputs", [])) + list(o.get("internal", []))}
    exits.sort(key=lambda t: order.get(t, 1 << 30))
    rest = sorted((t for t in common if t not in exits and t not in outs), key=lambda t: order.get(t, 1 << 30))
    # downstream of the edit first (where an error would be amplified), spread evenly
    first_changed = min((order.get(t, 1 << 30) for t in exits), default=-1)
    down = [t for t in rest if order.get(t, 1 << 30) > first_changed] if exits else rest
    k = max(0, n_track - len(exits[:n_track]))
    strata = down[:: max(1, len(down) // k)][:k] if k and down else []
    mapped = {}
    for old, spec in (tensor_map or {}).items():
        if old in g0["tensors"] and spec.get("from") in g1["tensors"] and old not in p1:
            mapped[old] = spec
    dropped = sorted(t for t in p0 if t not in p1 and t not in mapped and t not in g0["output"].get("tensors", []))
    return {"exits": exits[:max(n_track, 16)], "mapped": mapped, "strata": strata, "outputs": list(outs), "dropped": dropped}


def _flat(t) -> np.ndarray:
    return t.detach().double().cpu().numpy().reshape(-1)


def _bound(ref: np.ndarray, rel: float, abs_: float, cond: float = 0.0, cond_mult: float = 2.0) -> tuple[float, float]:
    fin = np.isfinite(ref)
    scale = float(np.abs(ref[fin]).max()) if fin.any() else 0.0
    return scale, rel * scale + abs_ + cond_mult * cond


def _cmp(a: np.ndarray, ref: np.ndarray, rel: float, abs_: float, cond: float = 0.0, cond_mult: float = 2.0, exact_int: bool = False) -> dict:
    """a (edited) against ref (original): max error, bound, ok. Matching infinities / NaNs are exact."""
    if a.shape != ref.shape:
        return {"ok": False, "abs": None, "bound": None, "why": f"shape {a.shape} != {ref.shape}"}
    same = (a == ref) | (np.isnan(a) & np.isnan(ref))
    fin = np.isfinite(ref)
    with np.errstate(invalid="ignore"):
        err = np.where(same, 0.0, np.abs(a - ref))
    nonfinite = bool((~np.isfinite(a) & ~same).any())
    scale, bound = _bound(ref, rel, abs_, cond, cond_mult)
    e = float(np.nanmax(err)) if err.size else 0.0
    if exact_int:
        bound = 0.0
    return {"ok": bool(e <= bound and not nonfinite), "abs": e, "bound": bound, "scale": scale, "nonfinite": nonfinite,
            "rel_to_ref": e / scale if scale else (0.0 if e == 0 else float("inf"))}


def _sol_check(out, ref, tol: dict) -> dict:
    """SOL's rule (docs/sol_execbench.md): |out - ref| <= max_atol + max_rtol * |ref| on >= required_matched_ratio of elements,
    no inf / NaN (unless both are -inf and allowed), an all-zero output against a non-zero reference fails, max_error_cap."""
    o, r = np.asarray(out, np.float64).reshape(-1), np.asarray(ref, np.float64).reshape(-1)
    if o.shape != r.shape:
        return {"ok": False, "why": f"shape {o.shape} != {r.shape}"}
    neg_inf = np.isneginf(o) & np.isneginf(r) if tol.get("allow_negative_inf") else np.zeros(o.shape, bool)
    bad_fin = (~np.isfinite(o) | ~np.isfinite(r)) & ~neg_inf
    with np.errstate(invalid="ignore"):
        err = np.abs(o - r)
        match = (err <= tol.get("max_atol", 1e-2) + tol.get("max_rtol", 1e-2) * np.abs(r)) | neg_inf
    ratio = float(match.mean()) if match.size else 1.0
    ok = ratio >= tol.get("required_matched_ratio", 0.99) and not bad_fin.any()
    if np.all(o == 0) and np.any(r != 0):
        ok = False
    cap = tol.get("max_error_cap")
    maxe = float(np.nanmax(np.where(bad_fin, 0, err))) if err.size else 0.0
    if cap is not None and maxe > cap:
        ok = False
    return {"ok": bool(ok), "matched_ratio": ratio, "max_abs_err": maxe}


def _device(device: str) -> str:
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _activation_elems(g: dict) -> int:
    return int(sum(int(np.prod(m["shape"])) for t, m in g["tensors"].items() if not m.get("const")))


def numeric(v: Verdict, orig_dir: Path, edited_dir: Path, g0: dict, g1: dict, patch: dict | None, *, w0: dict | None = None,
            tag: str = "", seeds=(0, 1, 2), variants=("uniform", "normal", "wide", "small"), tolerance: dict | None = None,
            tensor_map: dict | None = None, n_track: int = 12, exact="auto", conditioning="auto", device="cpu",
            sol: dict | None = None, golden: bool = True) -> None:
    """The numerical checks of one (original, edited) pair; failures go to ``v`` with ``tag`` in their ``where``."""
    import torch

    from .kb_export import OUTPUT_COND, _harness
    mbir, walk_torch = _harness()
    device = _device(device)
    ir0, ir1 = mbir.load_ir(orig_dir), mbir.load_ir(edited_dir)
    if w0 is not None:
        base0 = {k: a for k, a in w0.items() if k not in wd.derivations(g0)}
        ir0 = mbir.IR(ir0.name, ir0.graph, wd.fill_missing(g0, base0), ir0.golden_in, ir0.golden_out, ir0.io_tensors)
    sdt = cuda_dtypes.ir_dtype(g1)
    tol = {**cuda_dtypes.TOLERANCE[sdt], **(tolerance or {})}
    rel, abs_ = tol["output_rel"], tol["output_abs"]
    from .graph_prune import dead_ops
    skip0, skip1 = set(dead_ops(g0)), set(dead_ops(g1))
    sel = select_tensors(g0, g1, patch, tensor_map, n_track)
    outs = sel["outputs"]
    common_track = list(dict.fromkeys(sel["exits"] + sel["strata"]))
    mapped = sel["mapped"]
    grab0 = set(common_track) | set(mapped)
    grab1 = set(common_track) | {s["from"] for s in mapped.values()}
    v.info.setdefault("sampled" + tag, {"exits": len(sel["exits"]), "strata": len(sel["strata"]), "mapped": len(mapped),
                                        "dropped": len(sel["dropped"]), "tensors": (common_track + sorted(mapped))[:24]})
    big = _activation_elems(g0) > EXACT_AUTO_MAX_ELEMS
    do_exact = (exact == "on") or (exact == "auto" and not big)
    do_cond = (conditioning == "on") or (conditioning == "auto" and not big)
    exact_class = wd.is_exact(g1)

    def walk(ir, inputs, grab, skip, **kw):
        wi = _Inputs(ir, inputs) if inputs is not None else ir
        with torch.no_grad():
            return walk_torch.walk_free(wi, device, grab=grab, skip=skip, **kw)

    def where(*a):
        return ":".join(str(x) for x in (tag, *a) if x != "")

    runs, worst_out, worst_mid = [], {"abs": 0.0, "bound": 0.0, "rel_to_ref": 0.0}, {"abs": 0.0, "rel_to_ref": 0.0}
    bad_out, bad_mid, mid_compared, exit_compared = [], [], 0, 0
    refs_out0, edit_out0 = None, None
    sens, n_in = 0.0, 0
    diff_bad, diff_worst = [], {"abs": 0.0, "bound": 0.0}
    exact_bad, exact_worst, exact_n = [], 0.0, 0
    t0 = time.time()
    for si, seed in enumerate(seeds):
        ref_first = edit_first = None
        for vi, var in enumerate(variants):
            inp = make_inputs(ir0, var, seed)
            a, ga = walk(ir0, inp, grab0, skip0)
            b, gb = walk(ir1, inp, grab1, skip1)
            a_np = {t: _flat(x) for t, x in ga.items()}
            b_np = {t: _flat(x) for t, x in gb.items()}
            # the conditioning term: the original's own error against the exact walk, per compared tensor
            cond, exact_g0 = {}, {}
            if (do_cond or do_exact) and vi < 2:
                try:
                    _, eg0 = walk(ir0, inp, grab0 | set(outs), skip0, dtype=torch.float64, store_rounding=False)
                    exact_g0 = {t: _flat(x) for t, x in eg0.items()}
                    for t in a_np:
                        if t in exact_g0 and exact_g0[t].shape == a_np[t].shape and do_cond:
                            cond[t] = float(np.abs(a_np[t] - exact_g0[t]).max()) if a_np[t].size else 0.0
                except Exception as e:  # noqa: BLE001 -- best effort, as kb_export.exact_values
                    v.warn("conditioning_unavailable", f"float64 reference walk failed ({type(e).__name__}: {str(e)[:120]})", where())
            out_cond = 0.0
            if do_cond and vi < 2:
                try:
                    full0, _ = walk(ir0, inp, set(), skip0, dtype=torch.float64, store_rounding=False)
                    out_cond = float(np.abs(_flat(a) - _flat(full0)).max())
                except Exception:  # noqa: BLE001
                    out_cond = 0.0
            # ---- outputs
            av, bv = _flat(a), _flat(b)
            r = _cmp(bv, av, rel, abs_, out_cond, OUTPUT_COND)
            runs.append({"seed": seed, "variant": var, "ok": r["ok"], "abs": r["abs"], "bound": r["bound"]})
            if not r["ok"]:
                bad_out.append((seed, var, r))
            if (r["abs"] or 0) / max(r["bound"] or 1e-30, 1e-30) >= worst_out["abs"] / max(worst_out["bound"], 1e-30):
                worst_out = {"abs": r["abs"] or 0.0, "bound": r["bound"] or 0.0, "rel_to_ref": r.get("rel_to_ref") or 0.0, "seed": seed, "variant": var}
            # ---- sampled intermediates (always a hard check)
            for t in common_track:
                if t in a_np and t in b_np:
                    isint = g0["tensors"][t]["dtype"] in cuda_dtypes.INT_TENSOR
                    c = _cmp(b_np[t], a_np[t], rel, abs_, cond.get(t, 0.0), OUTPUT_COND, exact_int=isint)
                    mid_compared += 1
                    exit_compared += t in sel["exits"]
                    if (c["abs"] or 0) > worst_mid["abs"]:
                        worst_mid = {"abs": c["abs"], "bound": c["bound"], "rel_to_ref": c.get("rel_to_ref"), "tensor": t}
                    if not c["ok"]:
                        bad_mid.append({"tensor": t, "seed": seed, "variant": var, "abs": c["abs"], "bound": c["bound"]})
            for old, spec in mapped.items():
                if old in a_np and spec["from"] in b_np:
                    try:
                        full = gb[spec["from"]].reshape(g1["tensors"][spec["from"]]["shape"])
                        ax = spec.get("axis", 1)
                        sl = [slice(None)] * full.ndim
                        sl[ax] = slice(spec["start"], spec["stop"])
                        piece = _flat(full[tuple(sl)].reshape(g0["tensors"][old]["shape"]))
                    except Exception as e:  # noqa: BLE001
                        bad_mid.append({"tensor": old, "seed": seed, "variant": var, "why": f"tensor_map: {type(e).__name__}: {e}"})
                        continue
                    c = _cmp(piece, a_np[old], rel, abs_, cond.get(old, 0.0), OUTPUT_COND)
                    mid_compared += 1
                    exit_compared += 1
                    if (c["abs"] or 0) > worst_mid["abs"]:
                        worst_mid = {"abs": c["abs"], "bound": c["bound"], "rel_to_ref": c.get("rel_to_ref"), "tensor": old}
                    if not c["ok"]:
                        bad_mid.append({"tensor": old, "seed": seed, "variant": var, "abs": c["abs"], "bound": c["bound"], "via": "tensor_map"})
            # ---- sensitivity / differential (against the seed's first variant)
            if vi == 0:
                ref_first, edit_first = av, bv
            else:
                dr, de = av - ref_first, bv - edit_first
                s0 = float(np.abs(ref_first).max()) or 1.0
                sens = max(sens, float(np.abs(dr).max()) / s0)
                n_in += 1
                floor = 8.0 * _EPS[sdt] * max(s0, float(np.abs(av).max())) + OUTPUT_COND * out_cond
                bnd = 2.0 * rel * float(np.abs(dr).max()) + floor + abs_
                e = float(np.abs(de - dr).max())
                if e / max(bnd, 1e-30) >= diff_worst["abs"] / max(diff_worst["bound"], 1e-30):
                    diff_worst = {"abs": e, "bound": bnd, "seed": seed, "variant": var}
                if e > bnd:
                    diff_bad.append({"seed": seed, "variant": var, "abs": e, "bound": bnd})
            # ---- exact float64 agreement of an exact rewrite
            if do_exact and exact_class and vi < 2:
                try:
                    ea, _ = walk(ir0, inp, set(), skip0, dtype=torch.float64, store_rounding=False)
                    eb, _ = walk(ir1, inp, set(), skip1, dtype=torch.float64, store_rounding=False)
                    c = _cmp(_flat(eb), _flat(ea), EXACT_REL, EXACT_ABS)
                    exact_n += 1
                    exact_worst = max(exact_worst, c["abs"] or 0.0)
                    if not c["ok"]:
                        exact_bad.append({"seed": seed, "variant": var, "abs": c["abs"], "bound": c["bound"]})
                except Exception as e:  # noqa: BLE001
                    v.warn("exact_unavailable", f"float64 walk failed ({type(e).__name__}: {str(e)[:120]})", where())
                    do_exact = False
    # ---- verdicts
    if bad_out:
        s, var, r = bad_out[0]
        v.fail("output_mismatch", f"outputs differ from the original: max |edited - original| = {r['abs']:.3g} > bound {r['bound']:.3g} "
                                  f"(seed {s}, input '{var}'; {len(bad_out)}/{len(runs)} runs)", where())
    v.check("outputs" + tag, not bad_out, runs=len(runs), conditioning=do_cond)
    v.max_errors["outputs" + tag] = worst_out
    insensitive = n_in > 0 and sens < SENSITIVITY_MIN
    v.info["output_input_sensitivity" + tag] = round(sens, 8)
    if bad_mid:
        b = bad_mid[0]
        v.fail("intermediate_mismatch", f"sampled intermediate {b['tensor']!r} differs from the original"
                                        + (f" (max error {b['abs']:.3g} > bound {b['bound']:.3g})" if b.get("abs") is not None else f" ({b.get('why')})")
                                        + f" [{len(bad_mid)} mismatches over {mid_compared} comparisons; the output is "
                                          f"{'input-insensitive, so an output-only check would not see this' if insensitive else 'checked too'}]", where())
    v.check("sampled" + tag, not bad_mid, compared=mid_compared, exits_compared=exit_compared, insensitive=insensitive)
    v.max_errors["sampled" + tag] = worst_mid
    if insensitive and exit_compared == 0:
        v.fail("unverifiable", "the output hardly depends on the input (relative change across inputs "
                               f"{sens:.2g} < {SENSITIVITY_MIN:g}) and the rewrite leaves no tensor that both graphs define (and no tensor_map) "
                               "to compare: an output-only check cannot tell a wrong rewrite from a right one", where())
    elif insensitive:
        v.warn("input_insensitive_output", f"output changes < {SENSITIVITY_MIN:g} relative across inputs: verified through {exit_compared} "
                                           f"exit / mapped comparisons and {mid_compared - exit_compared} sampled ones", where())
    if diff_bad:
        d = diff_bad[0]
        v.fail("differential_mismatch", f"the output's response to the input differs from the original's by {d['abs']:.3g} > {d['bound']:.3g} "
                                        f"(seed {d['seed']}, input '{d['variant']}')", where())
    v.check("differential" + tag, not diff_bad, pairs=n_in)
    v.max_errors["differential" + tag] = diff_worst
    if exact_class and do_exact:
        if exact_bad:
            e = exact_bad[0]
            v.fail("exact_mismatch", f"the weights are exact derivations, so the float64 walks must agree to {EXACT_REL:g}: max error {e['abs']:.3g}", where())
        v.check("exact64" + tag, not exact_bad, runs=exact_n, max_abs=exact_worst)
    # ---- golden
    if golden:
        with torch.no_grad():
            b, _ = walk_torch.walk_free(ir1, device, skip=skip1)
            a, _ = walk_torch.walk_free(ir0, device, skip=skip0)
        gb_, ga_ = mbir.compare(b, ir0.golden_out, sdt), mbir.compare(a, ir0.golden_out, sdt)
        vs = _cmp(_flat(b), _flat(a), rel, abs_, 0.0)
        ok = (gb_["ok"] or not ga_["ok"]) and vs["ok"]
        if not ok:
            v.fail("golden_mismatch", f"on the stored golden input the edited walk is {'off the golden output' if not gb_['ok'] and ga_['ok'] else 'off the original walk'} "
                                      f"(golden: {gb_.get('max_abs_err')}, vs original: {vs['abs']})", where())
        v.check("golden" + tag, ok, edited_vs_golden=gb_.get("max_abs_err"), original_vs_golden=ga_.get("max_abs_err"), vs_original=vs["abs"])
    # ---- SOL hook
    if sol is not None:
        _sol(v, ir0, ir1, skip0, skip1, sol, device, tag)
    v.info["numeric_seconds" + tag] = round(time.time() - t0, 2)


def _sol(v: Verdict, ir0, ir1, skip0, skip1, sol: dict, device: str, tag: str) -> None:
    """``sol`` = {"refs": [tensors] | callable -> [tensors], "tolerance": {max_atol, max_rtol, required_matched_ratio, ...}}."""
    import torch

    from .kb_export import _harness
    _, walk_torch = _harness()
    refs = sol["refs"]() if callable(sol["refs"]) else sol["refs"]
    ref = np.concatenate([np.asarray(r.detach().float().cpu() if hasattr(r, "detach") else r, np.float64).reshape(-1) for r in refs])
    tol = sol.get("tolerance") or {}
    with torch.no_grad():
        b, _ = walk_torch.walk_free(ir1, device, skip=skip1)
        a, _ = walk_torch.walk_free(ir0, device, skip=skip0)
    cb, ca = _sol_check(_flat(b), ref, tol), _sol_check(_flat(a), ref, tol)
    ok = cb["ok"] or (not ca["ok"] and cb.get("max_abs_err", 1e30) <= 1.1 * ca.get("max_abs_err", 0.0) + 1e-12)
    if not ok:
        v.fail("sol_mismatch", f"the edited graph fails SOL's tolerance ({cb}) where the original {'passes' if ca['ok'] else 'also fails but by less'} ({ca})", tag or None)
    v.check("sol" + tag, ok, edited=cb, original=ca)
    v.max_errors["sol" + tag] = {"abs": cb.get("max_abs_err"), "matched_ratio": cb.get("matched_ratio")}


# ------------------------------------------------------------------------------ entry points
def verify(orig_dir, edited_dir, *, orig_weights=None, specs: dict | None = None, seeds=(0, 1, 2),
           variants=("uniform", "normal", "wide", "small"), shape_variants=(), tolerance: dict | None = None,
           tensor_map: dict | None = None, n_track: int = 12, exact="auto", conditioning="auto", device="cpu",
           sol: dict | None = None, structural_only: bool = False, check_base_values: bool = True) -> Verdict:
    """Verify ``edited_dir`` (an IR dir: graph.json, weights.npz, io.npz) against ``orig_dir``. ``orig_weights``: the original
    weights (a path to a weights.npz or a mapping) when they are not ``orig_dir``'s. Never raises: an unexpected failure
    is a ``gate_error`` reason (a candidate that cannot be verified is not accepted). -> Verdict."""
    v = Verdict()
    t0 = time.time()
    try:
        orig_dir, edited_dir = Path(orig_dir), Path(edited_dir)
        g0, g1 = _load_graph(orig_dir), _load_graph(edited_dir)
        w0 = dict(orig_weights) if isinstance(orig_weights, dict) else _npz(orig_weights or orig_dir / "weights.npz")
        w1 = _npz(edited_dir / "weights.npz")
        r = structural(v, g0, g1, w0, w1, specs, check_base_values=check_base_values)
        v.info["structural_ok"] = v.ok
        if not structural_only and r.get("walkable"):
            # the walkers run the original with the supplied original weights, the edited with what its dir holds
            # (+ derived weights the loader rebuilds)
            numeric(v, orig_dir, edited_dir, g0, g1, r.get("patch"), w0=w0, seeds=tuple(seeds), variants=tuple(variants),
                    tolerance=tolerance, tensor_map=tensor_map, n_track=n_track, exact=exact, conditioning=conditioning,
                    device=device, sol=sol)
            for i, (o2, e2) in enumerate(shape_variants):
                o2, e2 = Path(o2), Path(e2)
                try:
                    h0, h1 = _load_graph(o2), _load_graph(e2)
                    sv = Verdict()
                    pw = ge.diff(h0, h1)
                    structural(sv, h0, h1, _npz(o2 / "weights.npz"), _npz(e2 / "weights.npz"), specs, check_base_values=False)
                    for rs in sv.reasons:
                        v.fail(rs["code"], rs["msg"], f"shape_variant[{i}]")
                    if sv.ok:
                        numeric(v, o2, e2, h0, h1, pw, tag=f"[shape_variant {i}]", seeds=tuple(seeds[:1]), variants=tuple(variants[:2]),
                                tolerance=tolerance, tensor_map=tensor_map, n_track=n_track, exact=exact, conditioning=conditioning,
                                device=device, golden=True)
                except Exception as e:  # noqa: BLE001
                    v.fail("gate_error", f"shape variant {i}: {type(e).__name__}: {e}", f"shape_variant[{i}]")
        elif not structural_only and not r.get("walkable"):
            v.warn("numeric_skipped", "numerical checks skipped: the edited graph failed a structural check that makes it unwalkable")
    except Exception as e:  # noqa: BLE001 -- the gate itself must never raise
        v.fail("gate_error", f"{type(e).__name__}: {e}", "; ".join(traceback.format_exc().splitlines()[-3:])[:300])
    v.info["seconds"] = round(time.time() - t0, 2)
    v.info.setdefault("seeds", list(seeds))
    v.info["variants"] = list(variants)
    return v


def verify_or_reject(orig_dir, edited_dir, *, on_reject=None, **kw) -> Verdict:
    """``verify``, and when the verdict is a rejection call ``on_reject(verdict)`` (e.g. remove the staged IR). The caller
    decides what rollback means; nothing here touches the original."""
    v = verify(orig_dir, edited_dir, **kw)
    if not v.ok and on_reject is not None:
        try:
            on_reject(v)
        except Exception as e:  # noqa: BLE001
            v.warn("rollback_failed", f"{type(e).__name__}: {e}")
    return v


def commit_if_ok(v: Verdict, edited_dir, dst) -> bool:
    """Copy a verified IR dir to ``dst`` (symlinked npz are copied as files); False (and nothing written) when rejected."""
    if not v.ok:
        return False
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("graph.json", "weights.npz", "io.npz"):
        shutil.copyfile(Path(edited_dir) / f, dst / f)
    return True


def sol_from_bench(path) -> dict:
    """A ``sol`` hook from a SOL bench file made by gpu_eval/tools/sol_upstream.py (its ``sol_reference_outputs()`` and
    ``SOL_TOLERANCE``)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("sol_bench", str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return {"refs": m.sol_reference_outputs, "tolerance": getattr(m, "SOL_TOLERANCE", {})}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("orig_ir")
    ap.add_argument("edited_ir")
    ap.add_argument("--weights", help="the original weights.npz when it is not ORIG_IR/weights.npz")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--variants", default="uniform,normal,wide,small")
    ap.add_argument("--shape-variant", action="append", default=[], metavar="ORIG:EDITED")
    ap.add_argument("--tensor-map", help="JSON file {old tensor: {from, axis, start, stop}}")
    ap.add_argument("--sol-bench")
    ap.add_argument("--exact", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--conditioning", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-track", type=int, default=12)
    ap.add_argument("--structural-only", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    v = verify(a.orig_ir, a.edited_ir, orig_weights=a.weights, seeds=tuple(range(a.seeds)), variants=tuple(a.variants.split(",")),
               shape_variants=[tuple(s.split(":", 1)) for s in a.shape_variant],
               tensor_map=json.loads(Path(a.tensor_map).read_text()) if a.tensor_map else None, n_track=a.n_track,
               exact=a.exact, conditioning=a.conditioning, device=a.device,
               sol=sol_from_bench(a.sol_bench) if a.sol_bench else None, structural_only=a.structural_only)
    print(json.dumps(v.to_dict(), indent=1, default=str) if a.json is None else v.summary())
    if a.json:
        Path(a.json).write_text(json.dumps(v.to_dict(), indent=1, default=str))
    return 0 if v.ok else 1


if __name__ == "__main__":
    sys.exit(main())
