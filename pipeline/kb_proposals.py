"""KernelSpec proposals from KernelBlaster: build, verify and decide graph changes.

KernelBlaster proposes (format ``kb-proposal/1``) graph-level changes its
profile says would pay: fusing a producer with the elementwise op that is
the only reader of its output, or splitting a slot whose instances have very
different shapes. ModelBlaster owns the graph, so it builds the variant
(the rewrite itself is the unified hint layer, ``pipeline/graph_hints.py``),
exports it, has KernelBlaster optimise it and decides.

Fusion. A fused op is an IR op ``fused__<k0>__<k1>[__<p1>]__<k2>[__<p2>]...``
with ``sub_ops`` (the stages, in order) and ``internal`` (stage i's output,
read by stage i+1 as its parameter ``p<i+1>``; the param is spelled out only
when that stage has more than one activation input, e.g. ``add``). Chains
grow by fusing a fused op with its consumer (or a producer with a fused op):
the stages are flattened, so conv -> bn -> relu is one three-stage op
whichever order it was built in.

* Its KernelSpec signature is stage 0's parameters (``a_`` prefixed), then
  stage 1's minus the one that read the intermediate (``b_``), and so on.
  Every intermediate output pointer stays in the signature as scratch, so
  the seed -- the stages' kernels called back to back -- is correct by
  construction, and a truly fused kernel may simply not write them;
* the seed calls the hand kernels, or (``sub_sources``) the current best
  kernel of each stage, each in its own namespace, so a fusion variant starts
  from the unfused optimum rather than from the hand kernels;
* the torch IR walker runs ``sub_ops`` in order, so references (and the
  partition golden) come from the same oracle as every other op.

:func:`decide` compares the variant's best partition time against the
original's -- re-measured back to back (``remeasure``) because the two runs'
own numbers come from different moments on a shared GPU -- and records the
outcome on the hint in the hints file (``graph_hints.json``,
``modelblaster.graph_hints/v1``; a Phase 1 ``mb-fusion-registry/1`` file is
read as one). Accepted hints are what later exports apply; promoting a fused
op to ``KERNEL_SPECS`` stays a reviewed change.

    python -m modelblaster.pipeline.kb_proposals variant PROPOSALS.json ORIG_PROBLEM_DIR OUT_DIR \\
        [--only HINT_ID] [--seed-from KB_RESULT_DIR]
    python -m modelblaster.pipeline.kb_proposals decide ORIG_RESULT.json VARIANT_RESULT.json \\
        [--hints graph_hints.json] [--remeasure 5 --arch 75] \\
        [--evaluator local|ssh --ssh-host H --remote-dir D ...]   # re-time through KernelBlaster's
                                                                 # evaluator, on the runs' GPU model
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .kernel_abi import WEIGHT_ROLES, Param, bind, parse_signature
from .reference_kernels import KernelSpec

MB_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = MB_ROOT / "kernels" / "cuda" / "graph_hints.json"
ACCEPT_GAIN = 0.03               # variant must beat the original by >= 3%
STAGE_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def stage_prefix(i: int) -> str:
    return STAGE_LETTERS[i] + "_"


def _act_inputs(kind: str, specs: dict) -> list[str]:
    """Activation (non-weight) pointer inputs of an op kind's signature."""
    _, ps = parse_signature(specs[kind].signature)
    return [p.name for p in ps if p.is_const_ptr and p.name not in WEIGHT_ROLES]


@dataclass
class FusedKernelSpec(KernelSpec):
    sub_ops: list = field(default_factory=list)          # stage kinds, in order
    t_params: list = field(default_factory=list)         # stage i+1's param that reads stage i's output

    @property
    def t_param(self) -> str:                            # two-stage name (Phase 1)
        return self.t_params[0] if self.t_params else ""

    def cuda_seed(self, lib, sub_sources: dict | None = None) -> str:
        """Seed slot source: the stages' kernels called back to back.

        A stage whose kind is in ``sub_sources`` calls that source (a complete
        slot source defining the stage's entry, e.g. the current best kernel),
        wrapped in its own namespace; the others call the hand kernels from
        ``lib``."""
        from .kb_export import namespaced

        specs = _all_specs()
        sub_sources = sub_sources or {}
        stages = [(k, *parse_signature(specs[k].signature)) for k in self.sub_ops]
        hand = [e for k, e, _ in stages if k not in sub_sources]
        names = {n for e in hand for n in lib.closure(e)}
        includes, blocks = [], []
        if names:
            blocks.append("\n\n".join(lib.items[k].text for k in lib.items if k in names))
        for k in dict.fromkeys(k for k, _, _ in stages if k in sub_sources):
            inc, body = namespaced(f"stage_{k}", sub_sources[k])
            includes += [i for i in inc if i not in includes]
            blocks.append(f"// current best {k} kernel\n{body}")
        outs = [next(p.name for p in ps if p.is_ptr and not p.is_const_ptr) for _, _, ps in stages]
        calls = []
        for i, (k, e, ps) in enumerate(stages):
            args = [stage_prefix(i - 1) + outs[i - 1] if i and p.name == self.t_params[i - 1]
                    else stage_prefix(i) + p.name for p in ps]
            callee = f"stage_{k}::{e}" if k in sub_sources else e
            calls.append(f"    {callee}({', '.join(args)});")
        # the entry is CUDA: `_Float16` / `__bf16` pointers are spelled `__half` / `__nv_bfloat16`,
        # as the stage kernels it calls (the manifest's slot signature is the same spelling)
        from .cuda_dtypes import cuda_signature
        entry, params = parse_signature(cuda_signature(self.signature))
        scratch = ", ".join(f"`{stage_prefix(i)}{o}`" for i, o in enumerate(outs[:-1]))
        plist = ", ".join(f"{p.ctype} {p.name}" for p in params)
        return ("\n".join(includes) + ("\n\n" if includes else "") + "\n\n".join(blocks) +
                f"\n\n// Seed: {' then '.join(self.sub_ops)}; {scratch} hold the intermediates\n"
                f"// (scratch). A fused kernel need not write them.\n"
                f"void {entry}({plist}) {{\n" + "\n".join(calls) + "\n}\n")


_SPECS_CACHE: dict = {}


def _all_specs() -> dict:
    if not _SPECS_CACHE:
        from .reference_kernels import KERNEL_SPECS
        _SPECS_CACHE.update(KERNEL_SPECS)
    return _SPECS_CACHE


def _t_param(b_op: dict, t: str, specs: dict) -> str:
    _, pb = parse_signature(specs[b_op["op"]].signature)
    return next(a.param.name for a in bind(b_op, pb) if a.kind == "in" and a.value == t)


def chain_kind(kinds: list[str], t_params: list[str], specs: dict) -> str:
    parts = ["fused", kinds[0]]
    for k, tp in zip(kinds[1:], t_params):
        if "__" in k:
            raise ValueError(f"{k}: op kinds with '__' can't be fused")
        parts.append(k)
        if len(_act_inputs(k, specs)) > 1:
            parts.append(tp)
    return "__".join(parts)


def fused_kind(a_kind: str, b_kind: str, t_param: str, specs: dict) -> str:
    return chain_kind([a_kind, b_kind], [t_param], specs)


def parse_kind(kind: str, specs: dict) -> tuple[list[str], list[str]]:
    """``fused__...`` -> (stage kinds, t_params). Positional: a param token
    follows a stage kind exactly when that kind has > 1 activation input."""
    parts = kind.split("__")
    if parts[0] != "fused" or len(parts) < 3:
        raise ValueError(f"not a fused kind: {kind}")
    kinds, tps, i = [parts[1]], [], 2
    while i < len(parts):
        k = parts[i]
        if k not in specs:
            raise ValueError(f"{kind}: unknown stage kind {k!r}")
        acts = _act_inputs(k, specs)
        kinds.append(k)
        i += 1
        if len(acts) > 1:
            if i >= len(parts) or parts[i] not in acts:
                raise ValueError(f"{kind}: stage {k} needs the param that reads the intermediate")
            tps.append(parts[i])
            i += 1
        else:
            tps.append(acts[0])
    return kinds, tps


def make_chain_spec(kinds: list[str], t_params: list[str], specs: dict) -> FusedKernelSpec:
    kind = chain_kind(kinds, t_params, specs)
    sigs = [parse_signature(specs[k].signature)[1] for k in kinds]
    params = [Param(p.ctype, stage_prefix(i) + p.name) for i, ps in enumerate(sigs)
              for p in ps if not (i and p.name == t_params[i - 1])]
    sig = f"void kernel_{kind}(" + ", ".join(f"{p.ctype} {p.name}" for p in params) + ");"
    outs = [next(p.name for p in ps if p.is_ptr and not p.is_const_ptr) for ps in sigs]
    steps = [f"{kinds[0]} (parameters a_*) into a_{outs[0]}"]
    for i in range(1, len(kinds)):
        steps.append(f"{kinds[i]} (parameters {stage_prefix(i)}*) reading that result as its "
                     f"`{t_params[i - 1]}`" + (f", into {stage_prefix(i)}{outs[i]}" if i < len(kinds) - 1 else ""))
    scratch = ", ".join(f"{stage_prefix(i)}{o}" for i, o in enumerate(outs[:-1]))
    sem = (f"{' followed by '.join(kinds)}: computes " + "; then ".join(steps) +
           f". Only the {kinds[-1]} output is checked; {scratch} {'is' if len(kinds) == 2 else 'are'} scratch.\n" +
           "\n".join(f"[{k}] {specs[k].semantics}" for k in dict.fromkeys(kinds)))
    refs = "\n\n".join(specs[k].reference_impl for k in dict.fromkeys(kinds))
    ref = f"/* reference = the {len(kinds)} reference kernels in sequence */\n{refs}\n"
    return FusedKernelSpec(op=kind, signature=sig, semantics=sem, reference_impl=ref,
                           sub_ops=list(kinds), t_params=list(t_params))


def make_fused_spec(a_kind: str, b_kind: str, t_param: str, specs: dict) -> FusedKernelSpec:
    return make_chain_spec([a_kind, b_kind], [t_param], specs)


def spec_from_kind(kind: str, specs: dict) -> FusedKernelSpec:
    """Rebuild a fused spec from its kind name."""
    spec = make_chain_spec(*parse_kind(kind, specs), specs)
    assert spec.op == kind, (spec.op, kind)
    return spec


def with_fused_specs(graph: dict, specs: dict) -> dict:
    """``specs`` plus a FusedKernelSpec for every fused kind ``graph`` uses."""
    missing = {o["op"] for o in graph["ops"] if o["op"].startswith("fused__") and o["op"] not in specs}
    return {**specs, **{k: spec_from_kind(k, specs) for k in missing}} if missing else specs


def stages_of(op: dict) -> list[dict]:
    return op["sub_ops"] if "sub_ops" in op else [op]


def _graph_outputs(graph: dict) -> list[str]:
    return graph["output"].get("tensors") or [graph["output"]["tensor"]]


def fuse_pair(graph: dict, a_name: str, b_name: str, specs: dict) -> tuple[dict, FusedKernelSpec]:
    """Return (new graph, fused spec) with ops a_name -> b_name fused.

    Either side may already be fused: the stages are flattened into one chain.
    The producer's single output must be read by the consumer only (once, by
    its first stage) and must not be a graph output."""
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
    if readers != [b_name] or t in _graph_outputs(g) or B["inputs"].count(t) != 1:
        raise ValueError(f"{t}: not exclusively read by {b_name}")
    sa, sb = stages_of(A), stages_of(B)
    if t not in sb[0]["inputs"]:
        raise ValueError(f"{t}: read by a later stage of {b_name}, not a chain")
    stages = [copy.deepcopy(s) for s in sa + sb]
    internal = A.get("internal", []) + [t] + B.get("internal", [])
    tps = [_t_param(stages[i + 1], internal[i], specs) for i in range(len(stages) - 1)]
    spec = make_chain_spec([s["op"] for s in stages], tps, specs)
    ins = list(dict.fromkeys(A["inputs"] + [x for x in B["inputs"] if x != t]))
    fused = {"name": f"{a_name}+{b_name}", "op": spec.op, "inputs": ins, "outputs": list(B["outputs"]),
             "internal": internal, "sub_ops": stages,
             "shape": {STAGE_LETTERS[i]: s.get("shape", {}) for i, s in enumerate(stages)}}
    ops[ib] = fused
    del ops[ia]
    return g, spec


def fuse_chain(graph: dict, names: list[str], specs: dict) -> tuple[dict, FusedKernelSpec]:
    """Fuse ops ``names`` (a producer -> consumer chain) into one op."""
    g, cur, spec = graph, names[0], None
    for nxt in names[1:]:
        g, spec = fuse_pair(g, cur, nxt, specs)
        cur = f"{cur}+{nxt}"
    if spec is None:
        raise ValueError("a fusion chain needs at least two ops")
    return g, spec


def find_chains(graph: dict, kind: str, specs: dict) -> list[list[str]]:
    """Every unfused op chain in ``graph`` that ``kind`` fuses: stage kinds
    match, each intermediate is read only by the next stage (through the
    kind's param) and is not a graph output. Greedy in graph order, disjoint."""
    kinds, tps = parse_kind(kind, specs)
    ops = graph["ops"]
    outs = set(_graph_outputs(graph))
    readers: dict[str, list[dict]] = {}
    for o in ops:
        for t in o["inputs"]:
            readers.setdefault(t, []).append(o)
    chains, used = [], set()
    for o in ops:
        if o["op"] != kinds[0] or "sub_ops" in o or o["name"] in used:
            continue
        chain = [o]
        for k, tp in zip(kinds[1:], tps):
            cur = chain[-1]
            t = cur["outputs"][0] if len(cur["outputs"]) == 1 else None
            r = readers.get(t, [])
            if t is None or t in outs or len(r) != 1:
                break
            nxt = r[0]
            if nxt["op"] != k or "sub_ops" in nxt or nxt["name"] in used or nxt["inputs"].count(t) != 1:
                break
            try:
                if _t_param(nxt, t, specs) != tp:
                    break
            except (KeyError, StopIteration):
                break
            chain.append(nxt)
        if len(chain) == len(kinds):
            chains.append([c["name"] for c in chain])
            used.update(chains[-1])
    return chains


def bind_fused(op: dict, specs: dict) -> list:
    """kernel_abi.bind for a fused op, in the fused signature's order.

    Every intermediate stage output becomes kind ``scratch`` (its internal
    tensor); the next stage's input that reads it is dropped."""
    stages, internal = op["sub_ops"], op["internal"]
    args = []
    for i, s in enumerate(stages):
        _, ps = parse_signature(specs[s["op"]].signature)
        aa = bind(s, ps)
        for x in aa:
            if i < len(stages) - 1 and x.kind == "out" and x.value == internal[i]:
                x.kind = "scratch"
        if i:
            aa = [x for x in aa if not (x.kind == "in" and x.value == internal[i - 1])]
        args += aa
    return args


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


def best_sources(kb_result_dir: Path) -> dict[str, str]:
    """{slot id: source} of a KernelBlaster run's composite best."""
    d = Path(kb_result_dir) / "best" / "slots"
    return {p.stem: p.read_text() for p in sorted(d.glob("*.cu"))} if d.is_dir() else {}


def build_variant(hints, orig_problem: Path, out: Path, *, seed_sources: dict | None = None,
                  specs: dict | None = None) -> dict:
    """Apply ``hints`` (graph-hint dicts, or a kb-proposal/1 document: all its
    proposals) to the original partition's IR and export the variant
    partition under ``out/problem``.

    ``seed_sources`` ({slot id or op kind: source}, e.g. the original run's
    best) seed the variant's slots, and the fused slots' stages, so the
    variant starts where the original ended."""
    from . import graph_hints as gh
    from .kb_export import export_problem

    man = json.loads((Path(orig_problem) / "manifest.json").read_text())
    ir_dir = Path(man["source_ir"])
    if isinstance(hints, dict) and hints.get("format") == "kb-proposal/1":
        hints = gh.from_kb_proposals(hints)
    specs = dict(specs or _all_specs())
    graph = json.loads((ir_dir / "graph.json").read_text())
    g2, added, prov = gh.apply_hints(graph, hints, specs, statuses=None)
    specs.update(added)
    vir = write_variant_ir(ir_dir, g2, out / "ir")
    keep = {prov["renamed"].get(n, n) for n in man["partition"]["op_names"]}
    idx = [i for i, o in enumerate(g2["ops"]) if o["name"] in keep]
    prob = export_problem(vir, out / "problem", op_indices=idx, specs=specs, strategy="variant",
                          name=man["name"] + "__variant", slot_sources=seed_sources,
                          slot_source_tags={k: "original-best" for k in (seed_sources or {})})
    rec = {"original_problem": str(orig_problem), "variant_problem": str(prob),
           "hints": [h["id"] for h in hints], "hint_docs": hints, "applied": prov["applied"],
           "skipped": prov["skipped"]}
    (out / "variant.json").write_text(json.dumps(rec, indent=1))
    return rec


# ---------------------------------------------------------------- registry + decision

def accepted_fused_specs(registry: Path = DEFAULT_REGISTRY, specs: dict | None = None) -> dict:
    """{kind: FusedKernelSpec} for every accepted fusion in the hints file."""
    from . import graph_hints as gh

    specs = specs or _all_specs()
    hints = gh.load(Path(registry))
    return {h["fused_kind"]: spec_from_kind(h["fused_kind"], specs) for h in hints["hints"]
            if h["kind"] == "fuse" and h.get("status") == "accepted"}


def apply_accepted(graph: dict, registry: Path = DEFAULT_REGISTRY, specs: dict | None = None):
    """Apply every accepted fusion of the hints file wherever it matches.
    -> (graph, {kind: spec}, [fused op names])."""
    from . import graph_hints as gh

    hints = [h for h in gh.load(Path(registry))["hints"] if h["kind"] == "fuse"]
    g, used, prov = gh.apply_hints(graph, hints, dict(specs or _all_specs()))
    return g, used, [a["op"] for a in prov["applied"]]


def remeasure(problems: list[tuple[Path, dict]], *, arch: str, reps: int = 5, lease=None,
              timeout: int = 1800, kbeval=None, gpu_model: str | None = None, info: dict | None = None,
              work: Path | None = None) -> list[list[float]]:
    """Build each (problem dir, sources) once, then run them round-robin
    ``reps`` times -> partition times per problem. Interleaving keeps a
    shared GPU's load changes from favouring one side.

    ``lease`` (optional) is a context manager yielding the environment for
    the runs (e.g. ``CUDA_VISIBLE_DEVICES`` of one GPU, held exclusively):
    it is entered after the builds, so every run of every problem goes to
    the same GPU while nothing else times on it.

    ``kbeval`` (a :class:`kb_eval.KBEval`, e.g. pinned to one remote GPU with
    ``for_device``): the runs go through KernelBlaster's evaluator instead --
    the same rounds, on one GPU of ``gpu_model`` (the model the two
    KernelBlaster runs were timed on; none there is an error), local or over
    ssh. ``info`` (a dict) receives the device and evaluator used, and
    ``failures``: for each problem without a passing run, why (``run_failure``:
    the stage, the error text, whether it is an infrastructure failure)."""
    import contextlib
    import os
    import subprocess

    from .kb_export import build_and_run, parse_driver_output

    if kbeval is not None:
        jobs = [{"id": f"p{k}", "problem": str(d), "sources": dict(src), "reps": reps} for k, (d, src) in enumerate(problems)]
        wd = Path(work) if work is not None else Path(problems[0][0]).parent / "remeasure"
        res = kbeval.run(jobs, arch=arch, work=wd, gpu_model=gpu_model, name="remeasure")
        times, failures = [], []
        for j, (d, _) in zip(jobs, problems):
            runs = res["jobs"][j["id"]]["runs"]
            times.append([r.get("e2e_us") or r.get("partition_graph_us") or r["partition_us"] for r in runs
                          if r.get("ok") and (r.get("e2e_us") or r.get("partition_us"))])
            if not times[-1]:
                failures.append(run_failure(d, runs, log=wd / "remeasure.result.json"))
        if failures:                     # why, next to the runs (each run's own errors are in its row)
            of = wd / "remeasure.result.json"
            try:
                doc = json.loads(of.read_text())
                doc["failures"] = failures
                of.write_text(json.dumps(doc, indent=1))
            except (OSError, ValueError):
                pass
        if info is not None:
            info.update(device=res["jobs"][jobs[0]["id"]].get("gpu"), evaluator=kbeval.describe(),
                        devices=sorted({lab for j in jobs for lab in res["jobs"][j["id"]].get("gpus") or []}),
                        failures=failures)
        return times
    exes = []
    for k, (d, src) in enumerate(problems):
        # as deployed: built for and timed as a CUDA graph (eager where not capturable)
        r = build_and_run(d, arch=arch, sources=src, work=Path(d) / f"build_remeasure{k}", runs=0, graph=True)
        if r.get("stage") == "compile":
            raise RuntimeError(f"{d}: {r.get('log', '')[-800:]}")
        exes.append(r["exe"])
    times: list[list[float]] = [[] for _ in problems]
    last: dict[int, str] = {}
    with (lease if lease is not None else contextlib.nullcontext({})) as env:
        run_env = {**os.environ, **(env or {}), "MB_GRAPH": "1", "MB_ATTR": "0", "MB_OP_US_PASS": "0"}
        for _ in range(reps):
            for k, exe in enumerate(exes):
                rp = subprocess.run([str(exe)], capture_output=True, text=True, timeout=timeout, env=run_env)
                r = parse_driver_output(rp.stdout)
                if r.get("passed") and (r.get("e2e_us") or r.get("partition_us")):
                    times[k].append(r.get("e2e_us") or r["partition_us"])
                else:
                    last[k] = f"rc {rp.returncode}: " + (rp.stdout + rp.stderr)[-1500:]
    if info is not None:
        info["failures"] = [{"problem": str(problems[k][0]), "stage": "run", "error": last.get(k, "no timed run"),
                             "infra": is_infra_text(last.get(k))} for k in range(len(problems)) if not times[k]]
    return times


# KernelBlaster's infrastructure-failure patterns (resources._INFRA_PATTERNS) plus its "[infra]" tag:
# the GPU, the driver or the link failed, not the candidate
_INFRA = re.compile(
    r"\[infra\]|no CUDA-capable device is detected|all CUDA-capable devices are busy or unavailable|"
    r"CUDA driver version is insufficient|cudaErrorNoDevice|cudaErrorDevicesUnavailable|cudaErrorInsufficientDriver|"
    r"cudaErrorSystemNotReady|CUDA_ERROR_NO_DEVICE|Failed to initialize NVML|ssh: connect to host|"
    r"Connection timed out|Connection refused|Connection reset by peer|Could not resolve hostname|"
    r"Host key verification failed|Permission denied \(publickey", re.I)


def is_infra_text(text) -> bool:
    return bool(text) and bool(_INFRA.search(str(text)))


def run_failure(problem, runs: list[dict], log=None) -> dict:
    """Why a problem has no passing timed run (``time_problems`` rows) -> {"problem", "stage",
    "error" (the evaluator's error text, else its log's tail), "infra", "runs", "log"}."""
    bad = [r for r in runs if not r.get("ok")]
    r = bad[0] if bad else (runs[0] if runs else {})
    err = "; ".join(str(e) for e in (r.get("errors") or []) if e) or (r.get("log") or "")[-1500:]
    if not runs:
        err = "no run"
    elif not bad:
        err = err or "passed without a partition time"
    out = {"problem": str(problem), "stage": r.get("stage"), "error": " ".join(err.split())[:1500],
           "infra": is_infra_text(err) or is_infra_text(r.get("log")), "runs": len(runs)}
    if log is not None:
        out["log"] = str(log)
    return out


def _median(xs: list[float]) -> float | None:
    xs = sorted(xs)
    return (xs[len(xs) // 2] if len(xs) % 2 else 0.5 * (xs[len(xs) // 2 - 1] + xs[len(xs) // 2])) if xs else None


def _per_instance_us(result: dict, manifest: dict) -> dict:
    """op name -> its final time: its kernels' GPU time (the driver's attribution,
    ``attr_op_us``) or else its event-timed time (MB_OP_US) when the result has
    them, else its slot's time / number of instances."""
    if result.get("attr_op_us"):
        return dict(result["attr_op_us"])
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
           registry: Path = DEFAULT_REGISTRY, *, remeasure_reps: int = 0, arch: str = "75",
           accept_gain: float = ACCEPT_GAIN, ir: str | None = None, kbeval=None,
           allow_gpu_mismatch: bool = False) -> dict:
    """Accept the hints of a variant if its best beats the original's best by
    ``accept_gain``; record the evidence either way.

    With ``remeasure_reps`` both composite bests are rebuilt and re-timed
    back to back (median). Otherwise, when both runs report per-op times and
    the variant records fused pairs, the gain is taken on the touched ops
    only (per fused kind, as a share of the original partition): whole-
    partition bests also move with unrelated slots (a variant run finding a
    better GEMM made a 2.3% epilogue fusion look like +29%)."""
    from . import graph_hints as gh

    o = json.loads(Path(orig_result).read_text())
    v = json.loads(Path(variant_result).read_text())
    vj = Path(variant_json) if variant_json else Path(v["problem_dir"]).parent / "variant.json"
    var = json.loads(vj.read_text())
    # end to end as deployed (CUDA graph replay) where both runs report it
    basis = "e2e" if o.get("final_e2e_us") and v.get("final_e2e_us") else "partition"
    ou, vu = (o["final_e2e_us"], v["final_e2e_us"]) if basis == "e2e" else (o["final_partition_us"], v["final_partition_us"])
    ok = bool(o.get("final_ok") and v.get("final_ok") and ou and vu)
    measured = None
    if ok and remeasure_reps:
        info = {}
        gpu_model = None
        if kbeval is not None:                       # on the GPU model the two runs were timed on
            from .kb_eval import recorded_gpus, required_hardware
            hw = required_hardware(recorded_gpus([Path(orig_result).parent, Path(variant_result).parent]),
                                   arch=arch, allow_mismatch=allow_gpu_mismatch)
            gpu_model = hw["gpu_model"]
        t = remeasure([(Path(o["problem_dir"]), best_sources(Path(orig_result).parent)),
                       (Path(v["problem_dir"]), best_sources(Path(variant_result).parent))],
                      arch=arch, reps=remeasure_reps, kbeval=kbeval, gpu_model=gpu_model, info=info,
                      work=Path(variant_result).parent / "remeasure" if kbeval is not None else None)
        if t[0] and t[1]:
            ou, vu = _median(t[0]), _median(t[1])
            measured = {"original_us": t[0], "variant_us": t[1], **({"device": info["device"]} if info else {})}
        else:
            ok = False
    gain = (ou - vu) / ou if ok else None
    kinds = _touched_gains(o, v, var, ou, accept_gain) if ok and not measured else {}
    if kinds and all(k["gain"] is not None for k in kinds.values()):
        gain = sum(k["gain"] for k in kinds.values())
    status = "accepted" if ok and gain is not None and gain >= accept_gain else "rejected"
    ev = {"problem": o["problem"], "ir": ir, "original_us": ou, "variant_us": vu, "gain": gain, "basis": basis,
          "decision": status, "remeasured": measured, "touched": kinds or None, "variant": str(vj),
          "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    doc = gh.load(Path(registry))
    for h in var.get("hint_docs") or var.get("hints", []):
        gh.record_decision(doc, h, ev)
    gh.save(doc, Path(registry))
    return {"status": status, "gain": gain, "original_us": ou, "variant_us": vu,
            "partition_gain": (o["final_partition_us"] - v["final_partition_us"]) / o["final_partition_us"]
            if ok else None, "kinds": kinds, "hints": var.get("hints", []), "remeasured": measured}


def _touched_gains(o: dict, v: dict, var: dict, orig_us: float, accept_gain: float) -> dict:
    """Per fused kind: (original time of the ops it replaced - fused time) /
    original partition, from per-op times; {} when they're unavailable."""
    if not (o.get("op_us") or o.get("slots")) or not var.get("applied"):
        return {}
    o_inst = _per_instance_us(o, json.loads((Path(o["problem_dir"]) / "manifest.json").read_text()))
    v_inst = _per_instance_us(v, json.loads((Path(v["problem_dir"]) / "manifest.json").read_text()))
    per = {}
    for a in var["applied"]:
        if "pair" not in a:
            continue
        k = per.setdefault(a["kind"], {"orig_us": 0.0, "fused_us": 0.0, "pairs": 0, "missing": False})
        names = list(a["pair"])
        fused = "+".join(names)
        if all(n in o_inst for n in names) and fused in v_inst:
            k["orig_us"] += sum(o_inst[n] for n in names)
            k["fused_us"] += v_inst[fused]
            k["pairs"] += 1
        else:
            k["missing"] = True
    out = {}
    for kind, k in per.items():
        g = (k["orig_us"] - k["fused_us"]) / orig_us if k["pairs"] and not k["missing"] else None
        out[kind] = {"gain": g, "status": "accepted" if g is not None and g >= accept_gain else "rejected",
                     "touched_original_us": round(k["orig_us"], 3), "fused_us": round(k["fused_us"], 3),
                     "pairs": k["pairs"]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("variant")
    v.add_argument("proposals")
    v.add_argument("orig_problem")
    v.add_argument("out")
    v.add_argument("--only", action="append", help="build only these hint ids (repeatable)")
    v.add_argument("--seed-from", help="KernelBlaster result dir of the original (best/slots seeds the variant)")
    d = sub.add_parser("decide")
    d.add_argument("orig_result")
    d.add_argument("variant_result")
    d.add_argument("--variant-json")
    d.add_argument("--hints", "--registry", dest="hints", default=str(DEFAULT_REGISTRY))
    d.add_argument("--remeasure", type=int, default=0, help="re-time both bests N times, interleaved")
    d.add_argument("--arch", default="75")
    d.add_argument("--allow-gpu-mismatch", action="store_true",
                   help="re-time even where no GPU of the runs' model is available (with warnings)")
    from . import kb_eval as ke
    ke.add_args(d, prefix_help="--remeasure runs; default: ModelBlaster's own runs on the default GPU")
    d.set_defaults(evaluator="builtin")
    a = ap.parse_args()
    if a.cmd == "variant":
        from . import graph_hints as gh

        hints = gh.from_kb_proposals(json.loads(Path(a.proposals).read_text()))
        if a.only:
            hints = [h for h in hints if h["id"] in a.only]
        seeds = best_sources(Path(a.seed_from)) if a.seed_from else None
        print(json.dumps(build_variant(hints, Path(a.orig_problem), Path(a.out), seed_sources=seeds), indent=1))
    else:
        print(json.dumps(decide(Path(a.orig_result), Path(a.variant_result),
                                Path(a.variant_json) if a.variant_json else None, Path(a.hints),
                                remeasure_reps=a.remeasure, arch=a.arch, kbeval=ke.from_args(a),
                                allow_gpu_mismatch=a.allow_gpu_mismatch), indent=1))


if __name__ == "__main__":
    main()
