"""Hint kind ``merge_sibling_convs``: sibling convolutions joined by a channel concat become ONE convolution
(``--merge-sibling-convs off|auto``, env ``MB_MERGE_SIBLING_CONVS``; default off; docs/merge_sibling_convs.md).

The pattern (SqueezeNet's Fire module, the graph-agent pilot's ``merge_expand_fused``, +34% on its own)::

    s -> conv1x1 -> [act] -+
    s -> conv3x3 -> [act] -+-> cat(dim=1) -> t          =>        s -> conv3x3(W, b) -> [act] -> t
                                                                  W = concat_OC(zero_pad_centre(W1x1), W3x3)   b = concat(b1, b3)

Merge conditions (every one is re-checked by the applier on the graph it is applied to, so a stale hint is skipped
with a reason, never applied wrongly):

* the branches are 2-D convolutions (``conv2d`` / ``conv2d_f16`` / ``conv2d_bf16``) reading the SAME tensor, whose
  outputs (and optional activation outputs) are read only by the concat and are no graph outputs; the branches are
  CONSECUTIVE inputs of the concat (channels stay contiguous), a concat with other inputs keeps them;
* odd kernels, equal stride and dilation, equal output size, and CENTRED padding: ``P - D*(K-1)/2`` is the same for
  every branch (0 = the usual "same" padding; any common value works, the merged padding is ``D*(K_max-1)/2`` plus it);
* ``groups == 1`` (``conv2d_grouped`` / ``conv2d_dw`` / ``convnd`` / ``conv_transpose*`` are skipped, with the reason in
  the proposal report: a grouped merge needs per-group interleaving that does not preserve the channel layout);
* one op kind (one dtype suffix) and one weight / bias dtype per branch (``--quant native``: ``weight_dtypes``);
* an activation, when present, is the SAME op with the same parameters on every branch (relu / relu6 / silu / sigmoid
  / tanh / elu / leaky_relu / hardtanh / ewise1 / gelu...): it is applied once to the merged output. Branches with
  different activations are different keys and are not merged with each other.
* biases: all branches with a bias (concatenated), none (none), or a mix (zeros stand in: ``const`` derivation).

Weights are DERIVED WEIGHTS (weight_derivations.py): ``d.merge.<tag>.w`` = concat(axis 0) of ``pad`` nodes over the
original filters, ``d.merge.<tag>.b`` = concat of the biases (``const`` zeros for a missing one). Pure structural
derivations, so the verifier / the loader / ``write_variant_ir`` replay them from the original weights.npz bit-exactly
and per-tensor dtypes are kept (a derivation never changes dtype). The hint stores only the member names; the derivation
records are rebuilt from the graph's shapes by the applier (no weight values needed to apply it).

NUMERICS. In real arithmetic the merged convolution equals the originals (the padded taps are exact zeros). In floating
point it is NOT bit-identical: the reduction order of a different kernel, a longer reduction (extra ``0*x`` terms; with
an infinite or NaN input ``0*inf`` would turn a finite output into NaN) and, in fp16 / bf16, a different accumulation
split. The acceptance policy is therefore the gate's, not equality: ``graph_verify`` compares the model output and sampled
intermediates against the dtype's output tolerance plus the conditioning term and, with ``exact`` on, a float64 walk of
both graphs that must agree to 1e-9 (the derivations are exact, so only reordering error remains there).

COST MODEL (proposal only; the decision is the paired variant-vs-original timing like any hint). Padding multiplies the
FLOPs of the small branches: ``waste = (merged FLOPs - original FLOPs) / original FLOPs`` (1x1(64)+3x3(64): 0.80;
3x3+5x5 equal widths: 0.47; 1x1(128)+3x3(16): 3.8). A candidate is proposed only when
``waste <= max_waste`` (default 1.0; ``--merge-sibling-max-waste`` / env MB_MERGE_SIBLING_MAX_WASTE) and the estimated net
gain is positive: ``saved = concat time + launch_us * (ops removed - ops added)`` (profile times when present, else bytes
at ``bw_gbps``) minus ``extra = compute_share * sum(conv time) * waste`` (default compute_share 0.25: small-IC convs are
mostly memory / launch bound; configurable). The estimate ranks hints and gates the proposal; it never accepts one.
"""
from __future__ import annotations

import copy
import hashlib
import os
from dataclasses import asdict, dataclass, field

KIND = "merge_sibling_convs"
ENV = "MB_MERGE_SIBLING_CONVS"
ENV_WASTE = "MB_MERGE_SIBLING_MAX_WASTE"
MODES = ("off", "auto")

CONV_KINDS = {"conv2d": "f32", "conv2d_f16": "f16", "conv2d_bf16": "bf16"}
SKIP_CONV = {"conv2d_grouped": "grouped convolution", "conv2d_dw": "depthwise convolution", "convnd": "N-d convolution",
             "conv3d": "3-D convolution", "conv_transpose2d": "transposed convolution", "conv_transpose3d": "transposed convolution"}
ACT_KINDS = ("relu", "relu6", "silu", "sigmoid", "tanh", "elu", "leaky_relu", "hardtanh", "gelu", "gelu_exact", "ewise1")
SUFFIXES = ("_f16", "_bf16")
_VOLATILE = ("name", "inputs", "outputs", "dispatch_id", "depends_on", "hint", "slot_variant", "internal", "shape")


def resolve_mode(cfg=None) -> str:
    """``--merge-sibling-convs`` wins, else ``$MB_MERGE_SIBLING_CONVS``, else off. Validated."""
    v = getattr(cfg, "merge_sibling_convs", None)
    if v is None:
        v = os.environ.get(ENV, "off")
    v = str(v).strip().lower() or "off"
    if v in ("0", "false", "no"):
        v = "off"
    if v in ("1", "true", "yes", "on"):
        v = "auto"
    if v not in MODES:
        raise ValueError(f"--merge-sibling-convs must be one of {MODES}, not {v!r}")
    return v


@dataclass
class Policy:
    """Cost-model knobs of the proposal (see the module docstring)."""
    max_waste: float = 1.0           # reject when merged FLOPs exceed the originals' by more than this ratio
    compute_share: float = 0.25      # share of a conv's time that scales with its FLOPs (the rest: reads / launch)
    bw_gbps: float = 2000.0          # bytes model when the profile has no time for the concat
    launch_us: float = 0.0           # per removed kernel (the profile's launch_us overrides)
    tflops: dict = field(default_factory=lambda: {"f32": 20.0, "f16": 100.0, "bf16": 100.0})

    @staticmethod
    def from_cfg(cfg=None) -> "Policy":
        p = Policy()
        v = getattr(cfg, "merge_sibling_max_waste", None)
        if v is None:
            v = os.environ.get(ENV_WASTE)
        if v not in (None, ""):
            p.max_waste = float(v)
        if p.max_waste < 0:
            raise ValueError("merge sibling max waste must be >= 0")
        return p


class Skip(Exception):
    """A branch / instance that cannot be merged, with the reason."""


# ------------------------------------------------------------------------------ graph context
class Ctx:
    """Lookup tables over a graph (names made unique the way the flow does)."""

    def __init__(self, graph: dict):
        self.g = graph
        self.ops = {o["name"]: o for o in graph["ops"]}
        self.prod = {}
        self.readers: dict[str, list] = {}
        for o in graph["ops"]:
            for t in o.get("outputs", []):
                self.prod[t] = o
            for t in o.get("inputs", []):
                self.readers.setdefault(t, []).append(o)
        outs = graph["output"].get("tensors") or [graph["output"]["tensor"]]
        self.outs = set(outs)

    def exclusive(self, t: str, reader: dict) -> bool:
        r = self.readers.get(t, [])
        return len(r) == 1 and r[0] is reader and reader["inputs"].count(t) == 1 and t not in self.outs


def _base_kind(kind: str) -> str:
    for s in SUFFIXES:
        if kind.endswith(s):
            return kind[: -len(s)]
    return kind


def _logical(kind: str) -> str:
    return "bf16" if kind.endswith("_bf16") else "f16" if kind.endswith("_f16") else "f32"


def _act_sig(o: dict | None):
    if o is None:
        return None
    return (o["op"], tuple(sorted((k, repr(v)) for k, v in o.items() if k not in _VOLATILE)),
            tuple(sorted((k, repr(v)) for k, v in (o.get("shape") or {}).items() if k != "n")))


def _wdtype(graph: dict, key: str | None, kind: str) -> str | None:
    if key is None:
        return None
    return (graph.get("weight_dtypes") or {}).get(key) or _logical(kind)


@dataclass
class Branch:
    conv: dict
    act: dict | None
    key: tuple
    tensor: str                      # the concat input this branch produces

    @property
    def sh(self):
        return self.conv["shape"]


def _branch(ctx: Ctx, cat: dict, t: str) -> Branch:
    """The branch of ``cat`` producing input tensor ``t`` or Skip(reason)."""
    g = ctx.g
    p = ctx.prod.get(t)
    if p is None:
        raise Skip(f"input {t!r} is not produced by an op")
    if not ctx.exclusive(t, cat):
        raise Skip(f"{t!r} has other readers (or is a graph output)")
    act = None
    if _base_kind(p["op"]) in ACT_KINDS and len(p["inputs"]) == 1 and "sub_ops" not in p:
        act = p
        t2 = p["inputs"][0]
        c = ctx.prod.get(t2)
        if c is None or not ctx.exclusive(t2, act):
            raise Skip(f"the input of {act['name']} has other readers")
        p = c
    if "sub_ops" in p:
        raise Skip(f"{p['name']} is a fused op (merge applies before fusion)")
    if p["op"] in SKIP_CONV:
        sh = p.get("shape") or {}
        raise Skip(f"{p['name']}: {SKIP_CONV[p['op']]} ({p['op']}" + (f", G={sh['G']}" if "G" in sh else "") + ") is not merged")
    if p["op"] not in CONV_KINDS:
        raise Skip(f"{p['name']} ({p['op']}) is not a 2-D convolution")
    if act is not None and not ctx.exclusive(p["outputs"][0], act):
        raise Skip(f"the output of {p['name']} has other readers")
    if act is None and p["outputs"][0] != t:
        raise Skip(f"{p['name']} has several outputs")
    if len(p["inputs"]) != 1 or not p.get("weight"):
        raise Skip(f"{p['name']}: not a plain weight convolution")
    sh = p["shape"]
    if sh.get("G", 1) != 1:
        raise Skip(f"{p['name']}: groups != 1")
    if sh["KH"] % 2 == 0 or sh["KW"] % 2 == 0:
        raise Skip(f"{p['name']}: even kernel {sh['KH']}x{sh['KW']}")
    ch = sh["PH"] - sh.get("DH", 1) * (sh["KH"] - 1) // 2
    cw = sh["PW"] - sh.get("DW", 1) * (sh["KW"] - 1) // 2
    kind = p["op"]
    if act is not None and _logical(act["op"]) != _logical(kind):
        raise Skip(f"{act['name']}: dtype differs from {p['name']}")
    wdt = _wdtype(g, p["weight"], kind)
    bdt = _wdtype(g, p.get("bias"), kind)
    ot = g["tensors"].get(p["outputs"][0], {}).get("dtype")
    key = (p["inputs"][0], kind, sh["N"], sh["IC"], sh["IH"], sh["IW"], sh["OH"], sh["OW"], sh["SH"], sh["SW"],
           sh.get("DH", 1), sh.get("DW", 1), ch, cw, wdt, ot, _act_sig(act))
    return Branch(p, act, key, t)


_KEYF = ("input", "kind", "N", "IC", "IH", "IW", "OH", "OW", "SH", "SW", "DH", "DW", "padding offset H", "padding offset W",
         "weight dtype", "output dtype", "activation")


def _why_differ(a: Branch, b: Branch) -> str:
    for n, x, y in zip(_KEYF, a.key, b.key):
        if x != y:
            nm = {"SH": "stride", "SW": "stride", "DH": "dilation", "DW": "dilation", "padding offset H": "padding (not centred alike)",
                  "padding offset W": "padding (not centred alike)", "OH": "output size", "OW": "output size"}.get(n, n)
            return f"{a.conv['name']} / {b.conv['name']}: different {nm}" + ("" if n == "input" else f" ({x} vs {y})" if n != "activation" else "")
    return ""


# ------------------------------------------------------------------------------ a merge plan
def _cat_channels(cat: dict, ctx: Ctx) -> list[int] | None:
    """Channel counts of the concat's inputs when it is a plain channel concat, else None."""
    sh = cat.get("shape") or {}
    base = _base_kind(cat["op"])
    ins = cat["inputs"]
    if base in ("cat2_c1", "cat3_c1", "cat4_c1"):
        ci = sh.get("C_inputs")
        return [int(c) for c in ci] if ci and len(ci) == len(ins) else None
    if base == "catn":
        ks = [f"c{k}" for k in range(len(ins))]
        if any(k not in sh for k in ks):
            return None
        t0 = ctx.g["tensors"].get(ins[0])
        if t0 and len(t0["shape"]) == 4 and (sh.get("outer") != t0["shape"][0] or sh.get("inner") != t0["shape"][2] * t0["shape"][3]):
            return None
        return [int(sh[k]) for k in ks]
    return None


def plan(ctx: Ctx, cat: dict, run: list[Branch], policy: Policy | None = None) -> dict:
    """Merge plan of the consecutive branches ``run`` of ``cat`` (>= 2, one key), or Skip."""
    chans = _cat_channels(cat, ctx)
    if chans is None:
        raise Skip(f"{cat['name']} ({cat['op']}) is not a plain channel concat")
    pos = cat["inputs"].index(run[0].tensor)
    if [b.tensor for b in run] != cat["inputs"][pos:pos + len(run)]:
        raise Skip("the branches are not consecutive inputs of the concat")
    ocs = [b.sh["OC"] for b in run]
    if ocs != chans[pos:pos + len(run)]:
        raise Skip("the conv output channels do not match the concat's channel counts")
    s0 = run[0].sh
    kh, kw = max(b.sh["KH"] for b in run), max(b.sh["KW"] for b in run)
    dh, dw = s0.get("DH", 1), s0.get("DW", 1)
    ch, cw = run[0].key[12], run[0].key[13]
    ph, pw = dh * (kh - 1) // 2 + ch, dw * (kw - 1) // 2 + cw
    if ph < 0 or pw < 0:
        raise Skip("the merged padding would be negative")
    bdts = {_wdtype(ctx.g, b.conv.get("bias"), b.conv["op"]) for b in run if b.conv.get("bias")}
    if len(bdts) > 1:
        raise Skip("bias dtype differs between the branches")
    orig = sum(b.sh["OC"] * b.sh["IC"] * b.sh["KH"] * b.sh["KW"] for b in run)
    merged = sum(ocs) * s0["IC"] * kh * kw
    return {"cat": cat["name"], "pos": pos, "convs": [b.conv["name"] for b in run],
            "acts": [b.act["name"] for b in run] if run[0].act is not None else [],
            "x": run[0].key[0], "kh": kh, "kw": kw, "ph": ph, "pw": pw, "oc": ocs, "ic": s0["IC"],
            "waste": round((merged - orig) / orig, 4), "orig_flops_per_px": orig, "merged_flops_per_px": merged,
            "pattern": pattern_name(run), "dtype": _logical(run[0].conv["op"])}


def pattern_name(run: list[Branch]) -> str:
    a = run[0].act
    return "+".join(f"conv{b.sh['KH']}x{b.sh['KW']}" for b in run) + "+" + (_base_kind(a["op"]) if a is not None else "noact")


def find(graph: dict, policy: Policy | None = None, *, op_us: dict | None = None, total_us: float | None = None,
         launch_us: float = 0.0) -> tuple[list[dict], list[dict]]:
    """Every merge candidate in ``graph`` -> (plans, report). ``plans`` carry the cost estimate (``est``); ``report`` lists
    what was skipped and why ({"cat" | "op", "why"}), including candidates dropped by the cost gate."""
    from .graph_hints import unique_names
    policy = policy or Policy()
    g = unique_names(copy.deepcopy(graph))
    ctx = Ctx(g)
    plans, report = [], []
    for cat in g["ops"]:
        if _base_kind(cat["op"]) not in ("cat2_c1", "cat3_c1", "cat4_c1", "catn"):
            continue
        bs: list[Branch | None] = []
        reasons = []
        for t in cat["inputs"]:
            try:
                bs.append(_branch(ctx, cat, t))
            except Skip as e:
                bs.append(None)
                reasons.append(str(e))
        if sum(b is not None for b in bs) < 2:
            if any("convolution (" in r or "groups" in r or "grouped" in r for r in reasons):
                report.append({"cat": cat["name"], "why": "; ".join(r for r in reasons if "onvolution" in r or "groups" in r)[:300]})
            continue
        i = 0
        while i < len(bs):
            if bs[i] is None:
                i += 1
                continue
            j = i + 1
            while j < len(bs) and bs[j] is not None and bs[j].key == bs[i].key:
                j += 1
            if j - i >= 2:
                run = bs[i:j]
                try:
                    p = plan(ctx, cat, run, policy)
                    p["est"] = estimate(p, ctx, run, policy, op_us or {}, total_us, launch_us)
                    if p["waste"] > policy.max_waste:
                        report.append({"cat": cat["name"], "convs": p["convs"], "why": f"padding waste {p['waste']:.2f} > max {policy.max_waste:g}"})
                    elif p["est"]["net_us"] <= 0:
                        report.append({"cat": cat["name"], "convs": p["convs"], "why": f"estimated net gain {p['est']['net_us']:.1f} us <= 0"})
                    else:
                        plans.append(p)
                except Skip as e:
                    report.append({"cat": cat["name"], "why": str(e)})
            elif j < len(bs) and bs[j] is not None:
                why = _why_differ(bs[i], bs[j])
                if why:
                    report.append({"cat": cat["name"], "why": why})
            i = j
    return plans, report


def estimate(p: dict, ctx: Ctx, run: list[Branch], policy: Policy, op_us: dict, total_us: float | None, launch_us: float) -> dict:
    """Heuristic gain of one merge (module docstring, COST MODEL)."""
    cat = ctx.ops[p["cat"]]
    launch = launch_us or policy.launch_us
    removed = len(p["convs"]) + len(p["acts"]) + (1 if len(p["convs"]) == len(cat["inputs"]) else 0)
    added = 1 + (1 if p["acts"] else 0) + (0 if len(p["convs"]) == len(cat["inputs"]) else 1)
    esz = 2 if p["dtype"] in ("f16", "bf16") else 4
    sh = run[0].sh
    out_bytes = sum(p["oc"]) * sh["N"] * sh["OH"] * sh["OW"] * esz
    t_cat = op_us.get(cat["name"])
    if t_cat is None or t_cat <= 0:
        t_cat = 2.0 * out_bytes / (policy.bw_gbps * 1e3)           # read + write the concatenated tensor, in us
    saved = t_cat + launch * (removed - added)
    t_conv = [op_us.get(b.conv["name"]) for b in run]
    if all(t for t in t_conv):
        conv_us = float(sum(t_conv))
    else:
        flops = 2.0 * sh["N"] * sh["OH"] * sh["OW"] * p["orig_flops_per_px"]
        conv_us = flops / (policy.tflops.get(p["dtype"], 20.0) * 1e6)
    extra = policy.compute_share * conv_us * p["waste"]
    net = saved - extra
    tot = total_us or sum(op_us.values()) or None
    return {"saved_us": round(saved, 6), "extra_us": round(extra, 6), "net_us": round(net, 6), "concat_us": round(t_cat, 6),
            "conv_us": round(conv_us, 6), "removed_ops": removed, "added_ops": added,
            "pct": round(100.0 * max(net, 0.0) / tot, 3) if tot else None}


# ------------------------------------------------------------------------------ hints
def hint_id(pattern: str) -> str:
    return f"merge:{pattern}"


def propose(graph: dict, profile: dict | None, specs: dict, doc: dict | None, *, policy: Policy | None = None, min_gain_pct: float = 0.0,
            ir: str | None = None, ts: str | None = None) -> tuple[list[dict], list[dict]]:
    """Merge hints for one IR (one hint per pattern, the instances in ``instances`` / ``chains``) and the skip report.
    Patterns already decided in ``doc`` are not proposed again. The caller merges the hints into the hints doc."""
    from . import graph_hints as gh
    policy = policy or Policy()
    profile = profile or {}
    op_us = {o["name"]: o.get("us") or 0.0 for o in profile.get("ops", [])}
    total = profile.get("e2e_us") or sum(op_us.values()) or None
    plans, report = find(graph, policy, op_us=op_us, total_us=total, launch_us=profile.get("launch_us") or 0.0)
    done = gh.decided(doc) if doc else set()
    by: dict[str, list[dict]] = {}
    for p in plans:
        by.setdefault(p["pattern"], []).append(p)
    out = []
    for pat, ps in sorted(by.items(), key=lambda kv: -sum(p["est"]["net_us"] for p in kv[1])):
        hid = hint_id(pat)
        net = sum(p["est"]["net_us"] for p in ps)
        pct = round(100.0 * net / total, 3) if total else None
        if hid in done or (pct is not None and pct < min_gain_pct):
            if pct is not None and pct < min_gain_pct:
                report.append({"pattern": pat, "why": f"estimated gain {pct}% < min_gain_pct {min_gain_pct}"})
            continue
        out.append({"id": hid, "kind": KIND, "target": "cuda", "pattern": pat, "policy": asdict(policy),
                    "instances": [{"ir": ir, "cat": p["cat"], "convs": p["convs"], "acts": p["acts"]} for p in ps],
                    "chains": [p["convs"] + p["acts"] + [p["cat"]] for p in ps],
                    "status": "proposed",
                    "source": [{"by": "mb-cost", "ir": ir, "est_gain_us": round(net, 2), "est_gain_pct": pct,
                                "max_waste": round(max(p["waste"] for p in ps), 4), "instances": len(ps),
                                "cost": [{"cat": p["cat"], "waste": p["waste"], **p["est"]} for p in ps][:16],
                                "evidence": f"{len(ps)} x {pat}: padding waste <= {max(p['waste'] for p in ps):.2f} FLOP ratio, est. net {net:.1f} us",
                                "ts": ts or gh._now()}],
                    "evidence": []})
    return out, report


# ------------------------------------------------------------------------------ the applier
def _resolve(ctx: Ctx, inst: dict, policy: Policy | None = None) -> dict:
    """Re-derive the plan of one recorded instance on ``ctx`` (Skip with the reason when it no longer matches)."""
    cat = ctx.ops.get(inst["cat"])
    if cat is None:
        raise Skip(f"op {inst['cat']!r} is not in the graph")
    missing = [n for n in inst["convs"] + inst.get("acts", []) if n not in ctx.ops]
    if missing:
        raise Skip(f"op(s) {missing} are not in the graph")
    by_conv = {}
    for t in cat["inputs"]:
        try:
            b = _branch(ctx, cat, t)
        except Skip:
            continue
        by_conv[b.conv["name"]] = b
    run = []
    for cn in inst["convs"]:
        if cn not in by_conv:
            raise Skip(f"{cn} no longer feeds {inst['cat']} through a mergeable branch")
        run.append(by_conv[cn])
    if [b.conv["name"] for b in run] != inst["convs"]:
        raise Skip("the branches changed since the hint was made")
    if any(b.key != run[0].key for b in run):
        raise Skip(_why_differ(run[0], next(b for b in run if b.key != run[0].key)) or "the branches differ")
    if (inst.get("acts") or []) != ([b.act["name"] for b in run] if run[0].act else []):
        raise Skip("the activations changed since the hint was made")
    p = plan(ctx, cat, run, policy)
    p["_run"] = run
    return p


def _derive(g: dict, tag: str, p: dict) -> tuple[str, str | None]:
    """Record the derived weight / bias of one merge into ``g['weight_derivations']`` (declarative, no values)."""
    from . import cuda_dtypes as cd
    from . import weight_derivations as wd
    run = p["_run"]
    wdt = run[0].key[14]
    args = []
    for b in run:
        k = {"w": b.conv["weight"]}
        dh, dw = (p["kh"] - b.sh["KH"]) // 2, (p["kw"] - b.sh["KW"]) // 2
        if dh or dw:
            k = {"op": "pad", "pads": [[0, 0], [0, 0], [dh, dh], [dw, dw]], "value": 0.0, "args": [k]}
        args.append(k)
    D = g.setdefault(wd.KEY, {})
    wk = f"d.merge.{tag}.w"
    if wk in D:
        raise Skip(f"derived weight {wk!r} already exists")
    D[wk] = {"dtype": wdt, "shape": [sum(p["oc"]), p["ic"], p["kh"], p["kw"]], "expr": {"op": "concat", "axis": 0, "args": args}}
    bk = None
    if any(b.conv.get("bias") for b in run):
        bk = f"d.merge.{tag}.b"
        bdt = next(_wdtype(g, b.conv["bias"], b.conv["op"]) for b in run if b.conv.get("bias"))
        bargs = [{"w": b.conv["bias"]} if b.conv.get("bias") else
                 {"op": "const", "shape": [b.sh["OC"]], "value": 0.0, "dtype": bdt} for b in run]
        D[bk] = {"dtype": bdt, "shape": [sum(p["oc"])], "expr": {"op": "concat", "axis": 0, "args": bargs}}
    if cd.is_mixed(g):
        for k in (wk, bk):
            if k and D[k]["dtype"] in ("f32", "f16", "bf16"):
                g.setdefault("weight_dtypes", {})[k] = D[k]["dtype"]
    return wk, bk


def _apply_one(g: dict, ctx: Ctx, p: dict, hid: str) -> dict:
    run = p["_run"]
    cat = ctx.ops[p["cat"]]
    tag = hashlib.sha1(("+".join(p["convs"]) + ">" + p["cat"]).encode()).hexdigest()[:8]
    names = {o["name"] for o in g["ops"]} | set(g["tensors"])

    def uniq(b):
        n, k = b, 1
        while n in names:
            n, k = f"{b}_{k}", k + 1
        names.add(n)
        return n
    wk, bk = _derive(g, tag, p)
    s0 = run[0].sh
    full = len(p["convs"]) == len(cat["inputs"])
    oc_t = sum(p["oc"])
    out_t = cat["outputs"][0]
    old_tensors = [b.conv["outputs"][0] for b in run] + [b.act["outputs"][0] for b in run if b.act]
    if full:
        cat_meta = g["tensors"][out_t]
    mdt = g["tensors"][run[-1].conv["outputs"][0]]["dtype"]
    conv_out = uniq(f"merge_{tag}_o")
    g["tensors"][conv_out] = {"shape": [s0["N"], oc_t, s0["OH"], s0["OW"]], "dtype": mdt, "quant": None}
    cop = copy.deepcopy(run[0].conv)
    for k in ("dispatch_id", "depends_on", "hint", "slot_variant"):
        cop.pop(k, None)
    cop.update(name=uniq(f"merge_{tag}"), inputs=[p["x"]], outputs=[conv_out], weight=wk, bias=bk,
               shape={**s0, "OC": oc_t, "KH": p["kh"], "KW": p["kw"], "PH": p["ph"], "PW": p["pw"]})
    if bk is None:
        cop.pop("bias", None)
    new_ops = [cop]
    cur = conv_out
    if run[0].act is not None:
        aop = copy.deepcopy(run[0].act)
        for k in ("dispatch_id", "depends_on", "hint", "slot_variant"):
            aop.pop(k, None)
        act_out = out_t if full else uniq(f"merge_{tag}_a")
        if not full:
            g["tensors"][act_out] = {"shape": [s0["N"], oc_t, s0["OH"], s0["OW"]], "dtype": g["tensors"][run[0].act["outputs"][0]]["dtype"], "quant": None}
        aop.update(name=uniq(f"merge_{tag}_act"), inputs=[conv_out], outputs=[act_out])
        if "n" in (aop.get("shape") or {}):
            aop["shape"] = {**aop["shape"], "n": s0["N"] * oc_t * s0["OH"] * s0["OW"]}
        new_ops.append(aop)
        cur = act_out
    elif full:                                           # the conv writes the concat's tensor directly
        del g["tensors"][conv_out]
        cop["outputs"] = [out_t]
        g["tensors"][out_t] = cat_meta
        cur = out_t
    removed = set(p["convs"] + p["acts"])
    ncat = None
    if full:
        removed.add(cat["name"])
    else:
        pos = p["pos"]
        ins = cat["inputs"][:pos] + [cur] + cat["inputs"][pos + len(run):]
        chans = [int(c) for c in _cat_channels(cat, ctx)]
        chans = chans[:pos] + [oc_t] + chans[pos + len(run):]
        ncat = copy.deepcopy(cat)
        ncat["inputs"] = ins
        sh = dict(ncat["shape"])
        base = _base_kind(cat["op"])
        if base == "catn":
            for k in [k for k in sh if len(k) > 1 and k[0] == "c" and k[1:].isdigit()]:
                del sh[k]
            sh.update({f"c{k}": c for k, c in enumerate(chans)})
        else:
            sh["C_inputs"] = chans
            ncat["op"] = f"cat{len(ins)}_c1" + next((s for s in SUFFIXES if cat["op"].endswith(s)), "")
        ncat["shape"] = sh
        removed.add(cat["name"])
        new_ops.append(ncat)
    idx = 0
    for o in g["ops"]:
        if o["name"] == cat["name"]:
            break
        idx += o["name"] not in removed
    g["ops"] = [o for o in g["ops"] if o["name"] not in removed]
    # the new ops go where the concat was (everything the removed ops read is produced before it)
    g["ops"][idx:idx] = new_ops
    for t in old_tensors:
        if t != out_t:
            g["tensors"].pop(t, None)
    new_ops[0]["hint"] = hid
    start, tmap = 0, {}
    for b, oc in zip(run, p["oc"]):                      # branch outputs that vanished = slices of the merged tensor (compared by the gate)
        tmap[(b.act or b.conv)["outputs"][0]] = {"from": cur, "axis": 1, "start": start, "stop": start + oc}
        start += oc
    return {"removed": sorted(removed), "added": [o["name"] for o in new_ops], "weights": [k for k in (wk, bk) if k],
            "tensor_map": {t: m for t, m in tmap.items() if t != cur}}


def apply_hint(graph: dict, h: dict, specs: dict, prov: dict):
    """``merge_sibling_convs`` hint -> (graph, {}). Applies every recorded instance that still matches the graph and skips
    the rest with the reason (``prov['skipped']``); no instance applying with an explicit scope is a skip, not an error."""
    from . import graph_edit as ge
    g = graph
    n_ok = 0
    instances = list(h.get("instances") or [])
    if h.get("status") == "accepted":
        # an accepted hint applies wherever its pattern matches (the rule of ``fuse``): the other instances of the same pattern in this graph,
        # found by the detector under the policy the hint was proposed with (waste cap and cost gate), join the measured ones
        pol = Policy(**{k: v for k, v in (h.get("policy") or {}).items() if k in Policy.__dataclass_fields__})
        plans, _ = find(g, pol)
        known = {(i["cat"], tuple(i["convs"])) for i in instances}
        instances += [{"cat": p["cat"], "convs": p["convs"], "acts": p["acts"]} for p in plans
                      if p["pattern"] == h.get("pattern") and (p["cat"], tuple(p["convs"])) not in known]
    for inst in instances:
        ctx = Ctx(g)
        try:
            p = _resolve(ctx, inst)
        except Skip as e:
            prov["skipped"].append({"id": h["id"], "group": inst.get("convs"), "why": str(e)})
            continue
        try:
            info = _apply_one(g, ctx, p, h["id"])
        except Skip as e:
            prov["skipped"].append({"id": h["id"], "group": inst.get("convs"), "why": str(e)})
            continue
        n_ok += 1
        for n in info["removed"]:
            prov["renamed"][n] = info["added"][0]
        prov.setdefault("added", []).extend(info["added"])
        prov["applied"].append({"id": h["id"], "kind": KIND, "op": info["added"][0], "from": info["removed"], "added": info["added"],
                                "derived_weights": info["weights"], "waste": p["waste"], "tensor_map": info["tensor_map"]})
    if n_ok:
        try:
            ge.toposort(g)
        except ge.PatchError as e:                      # cannot happen for a valid plan; never leave a broken graph
            prov["skipped"].append({"id": h["id"], "why": f"merge produced an invalid graph: {e}"})
            raise
    elif not h.get("instances"):
        prov["skipped"].append({"id": h["id"], "why": "no instances recorded"})
    return g, {}


def scope_ops(h: dict, graph: dict, specs: dict) -> list[str]:
    """Current op names the hint replaces in ``graph`` (instances that still match)."""
    from .graph_hints import unique_names
    g = unique_names(copy.deepcopy(graph))
    ctx = Ctx(g)
    names: list[str] = []
    for inst in h.get("instances") or []:
        try:
            p = _resolve(ctx, inst)
        except Skip:
            continue
        names += p["convs"] + p["acts"] + [p["cat"]]
    return list(dict.fromkeys(names))


# ------------------------------------------------------------------------------ the hook the round loop calls
class MergeHook:
    """``kb_graph_opt``'s handle (same shape as ``capabilities.Caps`` / ``graph_rewrite.RewriteHook``). Inert when off."""

    def __init__(self, cfg, work=None):
        self.cfg, self.work = cfg, work
        self.mode = resolve_mode(cfg)
        self.policy = Policy.from_cfg(cfg) if self.mode != "off" else None
        self.log: list[dict] = []

    def active(self) -> bool:
        return self.mode != "off"

    def round_key(self) -> dict:
        return {"merge_sibling_convs": self.mode, **({"merge_max_waste": self.policy.max_waste} if self.policy.max_waste != 1.0 else {})} \
            if self.active() else {}

    def decide_kinds(self) -> tuple:
        return (KIND,) if self.active() else ()

    def summary(self) -> dict:
        return {"mode": self.mode, "max_waste": self.policy.max_waste if self.policy else None, "log": self.log}

    def propose(self, graphs: dict, profiles: list, doc: dict, specs: dict) -> list:
        if not self.active():
            return []
        from . import graph_hints as gh
        new = []
        for p in profiles:
            g = graphs.get(p["ir"])
            if g is None:
                continue
            hints, report = propose(g, p, specs, doc, policy=self.policy, min_gain_pct=getattr(self.cfg, "min_gain_pct", 2.0), ir=p["ir"])
            self.log.append({"ir": p["ir"], "proposed": [h["id"] for h in hints], "skipped": report[:20]})
            new += gh.merge(doc, hints)
            for h in hints:                                  # gh.merge extends ``chains`` of a known id; the instances (the applier's scope) too
                cur = gh.get(doc, h["id"])
                if cur is not None and cur is not h:
                    have = {(i["cat"], tuple(i["convs"])) for i in cur.get("instances", [])}
                    cur.setdefault("instances", []).extend(i for i in h["instances"] if (i["cat"], tuple(i["convs"])) not in have)
        return new


# ------------------------------------------------------------------------------ CLI (candidates, skip report, a hints file for graph_hints apply)
def main(argv=None) -> int:
    """``python -m modelblaster.pipeline.merge_convs IR_DIR [--profile P.json] [--max-waste X] [--out HINTS.json [--accept]]``: print the
    merge candidates of an IR (cost per instance) and the skipped list with reasons; ``--out`` writes a graph_hints document with the hint
    (``--accept``: status accepted, for ``graph_hints apply IR HINTS OUT --verify``). CPU only."""
    import argparse
    import json
    from pathlib import Path

    from . import graph_hints as gh
    from .kb_proposals import _all_specs
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("ir")
    ap.add_argument("--profile", help="a kb_plan.profile_ir style JSON ({'ops': [{'name', 'us'}], 'e2e_us', 'launch_us'}) for the cost model")
    ap.add_argument("--max-waste", type=float, default=None)
    ap.add_argument("--min-gain-pct", type=float, default=0.0)
    ap.add_argument("--out")
    ap.add_argument("--accept", action="store_true")
    a = ap.parse_args(argv)
    g = json.loads((Path(a.ir) / "graph.json").read_text())
    pol = Policy.from_cfg(None)
    if a.max_waste is not None:
        pol.max_waste = a.max_waste
    prof = json.loads(Path(a.profile).read_text()) if a.profile else None
    hints, report = propose(g, prof, dict(_all_specs()), None, policy=pol, min_gain_pct=a.min_gain_pct, ir=g.get("name"))
    print(json.dumps({"hints": [{k: h[k] for k in ("id", "pattern", "instances")} | {"source": h["source"][0]} for h in hints], "skipped": report}, indent=1))
    if a.out:
        doc = gh.new_doc()
        for h in hints:
            h["status"] = "accepted" if a.accept else "proposed"
        gh.merge(doc, hints)
        gh.save(doc, Path(a.out))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
