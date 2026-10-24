"""Export an IR partition as a multi-slot KernelBlaster CUDA problem.

ModelBlaster decides *what* a problem is (a partition of the graph);
KernelBlaster decides *how* each kernel in it is written. A problem is a
directory:

  manifest.json      format "mb-kb-problem/1": slots, ops, buffers, tolerances
  slots/<id>.cu      one editable source per slot (seeded from the hand
                     kernels or a cached winner); defines the slot's entry
                     with exactly its KernelSpec signature
  slots/<id>.ref.c   the op's C reference implementation (prompt context)
  walker_head.cu     shared preamble (includes, MB_* launch macros)
  walker_tail.cu     mb_run_op(i) dispatch + launch_gpu_implementation()
  driver.cpp         self-contained checker/timer (data.bin is .incbin'd)
  data.bin           reference values of every partition tensor + weights

A *slot* is one op kind in the partition: every instance of that op calls
the same slot entry (shapes are runtime arguments, as in MB's C ABI).
Assembly (both repos implement it from the manifest): head, then each slot
wrapped in ``namespace mbslot_<id> { ... }`` with its #includes hoisted out
and its #defines #undef'd after, then tail. The namespace keeps slot-local
helpers from colliding and makes every kernel name carry its slot.

The driver, run with no arguments:
  1. runs the whole partition and checks every output against the reference
     (err <= rel * max|finite ref| + abs);
  2. runs every op alone on the reference's inputs and checks its outputs
     (attributes a failure to a slot);
  3. times the partition and every op with CUDA events;
and prints MB_CHECK / MB_SLOT_CHECK / MB_PARTITION_US / MB_SLOT_US / MB_OP_US
lines and the wall time of each phase (MB_PHASE_MS), followed by "passed" or
"failed". With ``MB_MODE=profile`` it instead runs every op once (for ncu) and
prints MB_PROFILE_DONE. ``MB_ONLY_SLOTS``, ``MB_SCREEN`` and ``MB_GRAPH`` (see
_DRIVER_TEMPLATE, docs/gpu_harness_perf.md) limit the isolated checks to the
slots a candidate changed, shorten the timing, and add a CUDA-graph time.

    python -m modelblaster.pipeline.kb_export IR_DIR OUT_DIR [--ops a,b|--range i:j|--strategy S]
        [--hints graph_hints.json] [--build-check --arch 75]
    python -m modelblaster.pipeline.kb_export --plan PLAN.json OUT_DIR [--cache DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import cuda_dtypes
from .kernel_abi import bind, parse_signature
from .op_classes import op_class

MB_ROOT = Path(__file__).resolve().parents[1]
HARNESS = MB_ROOT / "gpu_eval" / "harness"
DEFAULT_HAND_CU = HARNESS / "cuda" / "mb_cuda_kernels.cu"
DEFAULT_HAND_CU_F16 = HARNESS / "cuda" / "mb_cuda_kernels_f16.cu"    # generated (f16_kernels.py)
FORMAT = "mb-kb-problem/1"
NS_PREFIX = "mbslot_"
ALIAS_OPS = ("view", "chunk2_c1")
DEFAULT_TOL = {"output_rel": 1e-3, "output_abs": 1e-5, "slot_rel": 1e-3}
DTYPE_TOL = {d: {k: t[k] for k in (*DEFAULT_TOL, "slot_abs")} for d, t in cuda_dtypes.TOLERANCE.items()}


def _harness():
    """The torch IR walker (reference values) lives with the GPU harness."""
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    import mbir
    import walk_torch
    return mbir, walk_torch


# ---------------------------------------------------------------- storage dtypes (fp32 / fp16)

class _CudaSpelled:
    """A KernelSpec seen with the CUDA spelling of its signature (`_Float16`
    -> `__half`, cuda_dtypes.cuda_signature); everything else passes through."""

    def __init__(self, spec):
        self._spec = spec

    def __getattr__(self, k):
        return getattr(self._spec, k)

    @property
    def signature(self) -> str:
        return cuda_dtypes.cuda_signature(self._spec.signature)


def _problem_dtype(kinds: list[str]) -> str:
    """fp16 when every op of the partition is an `_f16` op, else fp32."""
    dts = {cuda_dtypes.op_dtype(k) for k in kinds}
    return cuda_dtypes.FP16 if dts == {cuda_dtypes.FP16} else cuda_dtypes.FP32


def _hand_source(hand_cu: Path, dtype: str) -> str:
    """The hand kernels for a dtype: fp16 uses the generated `<stem>_f16.cu`
    beside ``hand_cu`` (cuda/mb_cuda_kernels_f16.cu for the default)."""
    hand_cu = Path(hand_cu)
    if dtype == cuda_dtypes.FP16:
        hand_cu = hand_cu.with_name(hand_cu.stem + "_f16" + hand_cu.suffix)
    return hand_cu.read_text()


def _tensor_dtype(ir, t: str) -> str:
    """Buffer dtype of IR tensor t: a constant's is its weights.npz array's,
    else graph.json's (f16 for an fp16 extract)."""
    key = ir.graph["tensors"].get(t, {}).get("const")
    if key:
        return cuda_dtypes.buffer_dtype(ir.weights[key].dtype)
    dt = ir.graph["tensors"].get(t, {}).get("dtype")
    # i64: the index outputs of argmax/argmin (kernel_argmax_dim writes int64_t)
    return {"f16": "f16", "i64": "i64"}.get(dt, "f32")


_NP_OF_BUF = {"f32": np.float32, "f16": np.float16, "i32": np.int32, "i64": np.int64}


# ---------------------------------------------------------------- CUDA source library

@dataclass
class _Item:
    name: str
    text: str                      # with its leading comment
    code: str                      # comments/strings blanked, for dependency scan
    deps: set = field(default_factory=set)


def _blank_comments(src: str) -> str:
    """Same length as src, with comments and string/char literals blanked."""
    out, i, n = list(src), 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            out[i:j] = " " * (j - i)
            i = j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out[i:j] = [ch if ch == "\n" else " " for ch in src[i:j]]
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            out[i + 1:j] = " " * (j - i - 1)
            i = j + 1
        else:
            i += 1
    return "".join(out)


class CudaLibrary:
    """Top-level definitions of a CUDA source and their dependency closure.

    Splits the file into preprocessor lines (the shared head) and top-level
    items (functions, kernels, globals), each with the comment above it, so a
    slot can be seeded with exactly the definitions its entry point needs.
    """

    def __init__(self, src: str):
        self.head: list[str] = []
        self.items: dict[str, _Item] = {}
        blank = _blank_comments(src)
        i, n, depth, start, pending = 0, len(src), 0, 0, 0
        while i < n:
            c = blank[i]
            if depth == 0 and c == "#" and blank[start:i].strip() == "":
                j = i
                while True:
                    k = src.find("\n", j)
                    k = n if k < 0 else k
                    if src[k - 1:k] == "\\":
                        j = k + 1
                        continue
                    break
                self.head.append(src[pending:k].strip("\n"))
                i = start = pending = k + 1
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    self._add(src, blank, pending, i + 1)
                    start = pending = i + 1
            elif c == ";" and depth == 0:
                self._add(src, blank, pending, i + 1)
                start = pending = i + 1
            elif c == "\n" and depth == 0 and blank[start:i].strip() == "":
                start = i + 1          # comment-only lines stay pending
            i += 1
        names = set(self.items)
        for it in self.items.values():
            it.deps = {w for w in re.findall(r"\b[A-Za-z_]\w*\b", it.code) if w in names and w != it.name}

    def _add(self, src, blank, a, b):
        # keep only the comment block directly above the definition (no blank
        # line in between); section banners and file notes are dropped
        c0 = a + len(blank[a:b]) - len(blank[a:b].lstrip())
        lead = src[a:c0].rstrip(" \t").split("\n\n")[-1].strip("\n") if c0 > a else ""
        text = ((lead + "\n") if lead.strip() else "") + src[c0:b].strip("\n")
        code = blank[a:b]
        m = re.search(r"([A-Za-z_]\w*)\s*(?:\(|\[|=|\{)", code.split("{")[0] + "{")
        if not m:
            return
        name = m.group(1)
        if name in ("struct", "class", "union", "enum"):
            m = re.search(r"\b(?:struct|class|union|enum)\s+([A-Za-z_]\w*)", code)
            name = m.group(1)
        self.items[name] = _Item(name, text, code)

    def closure(self, entry: str) -> list[str]:
        if entry not in self.items:
            raise KeyError(f"no definition of {entry}")
        seen, stack = set(), [entry]
        while stack:
            x = stack.pop()
            if x not in seen:
                seen.add(x)
                stack.extend(self.items[x].deps)
        return [k for k in self.items if k in seen]          # source order

    def slot_source(self, entry: str) -> str:
        return "\n\n".join(self.items[k].text for k in self.closure(entry)) + "\n"

    def head_source(self) -> str:
        return "\n".join(self.head) + "\n"


# ---------------------------------------------------------------- assembly

def namespaced(ns: str, code: str) -> tuple[list[str], str]:
    """(hoisted #include lines, ``namespace ns { body }`` + #undef of its macros)."""
    # C linkage would make two slots' entries the same symbol across namespaces
    code = code.replace('extern "C"', "")
    includes = [l for l in code.splitlines() if l.strip().startswith("#include")]
    body = "\n".join(l for l in code.splitlines() if not l.strip().startswith("#include"))
    defines = sorted(set(re.findall(r"^\s*#\s*define\s+([A-Za-z_]\w*)", body, re.M)))
    undefs = "".join(f"#undef {d}\n" for d in defines)
    return includes, f"namespace {ns} {{\n{body}\n}}  // namespace {ns}\n{undefs}"


def wrap_slot(slot_id: str, code: str) -> str:
    includes, body = namespaced(f"{NS_PREFIX}{slot_id}", code)
    return ("\n".join(includes) + "\n" if includes else "") + f"// ==== slot {slot_id} ====\n{body}"


def assemble(problem_dir: Path, sources: dict[str, str] | None = None) -> str:
    """The kernel TU: head + wrapped slots (``sources`` overrides slot files) + tail."""
    problem_dir = Path(problem_dir)
    man = json.loads((problem_dir / "manifest.json").read_text())
    asm = man["assembly"]
    parts = [(problem_dir / asm["head"]).read_text()]
    for s in man["slots"]:
        code = (sources or {}).get(s["id"]) or (problem_dir / s["file"]).read_text()
        parts.append(wrap_slot(s["id"], code))
    parts.append((problem_dir / asm["tail"]).read_text())
    return "\n".join(parts)


# ---------------------------------------------------------------- partitioning

def partition_ir(graph: dict, strategy: str = "whole", max_ops: int = 12) -> list[list[int]]:
    """Op-index ranges (topological order) to export as problems.

    whole    one problem with every op
    per_op   one problem per kernel op (views ride with their consumer)
    window   consecutive windows of at most ``max_ops`` kernel ops
    """
    ops = graph["ops"]
    kern = [i for i, o in enumerate(ops) if o["op"] not in ALIAS_OPS]
    if strategy == "whole":
        return [list(range(len(ops)))]
    size = 1 if strategy == "per_op" else max_ops
    if strategy not in ("per_op", "window"):
        raise ValueError(f"unknown partition strategy {strategy}")
    parts, lo = [], 0
    for k in range(0, len(kern), size):
        hi = kern[min(k + size, len(kern)) - 1] + 1
        parts.append(list(range(lo, hi)))
        lo = hi
    if lo < len(ops):
        parts[-1].extend(range(lo, len(ops)))
    return parts


# ---------------------------------------------------------------- export

def _c_scalar(base: str, v) -> str:
    if base in ("float", "double"):
        v = float(v)
        if math.isinf(v):
            return "INFINITY" if v > 0 else "-INFINITY"
        if math.isnan(v):
            return "NAN"
        s = repr(v)
        return s + ("f" if base == "float" else "")
    return str(int(v))


def _slot_id(op: str) -> str:
    return re.sub(r"\W", "_", op)


def base_slot_id(kind: str, specs: dict) -> str:
    """One slot per C entry point (kernel_<x> -> x)."""
    entry, _ = parse_signature(specs[kind].signature)
    return _slot_id(entry[len("kernel_"):] if entry.startswith("kernel_") else entry)


def slot_id_of(op: dict, specs: dict) -> str:
    """The slot an op instance calls: its entry's slot, or (entry,
    ``slot_variant``) when a split hint gave it a shape-specialised variant."""
    sid = base_slot_id(op["op"], specs)
    return f"{sid}__{op['slot_variant']}" if op.get("slot_variant") else sid


def export_problem(ir_dir, out_dir, *, op_indices: list[int] | None = None, name: str | None = None,
                   hand_cu: Path = DEFAULT_HAND_CU, slot_sources: dict[str, str] | None = None,
                   slot_source_tags: dict[str, str] | None = None, specs: dict | None = None,
                   tolerance: dict | None = None, strategy: str = "custom",
                   sub_sources: dict[str, dict[str, str]] | None = None, target: str = "cuda") -> Path:
    """Write a problem directory for the ops ``op_indices`` of the IR at ``ir_dir``.

    ``slot_sources`` maps a slot id (or an op kind) to a seed source
    replacing the hand kernel (e.g. a cached KernelBlaster winner);
    ``slot_source_tags`` records where each came from in the manifest. A
    fused slot without a source of its own is seeded with its stages called
    back to back: each stage from ``sub_sources[fused slot id][stage kind]``
    when given (e.g. cache entries guarded on the stage's shapes), else -- if
    ``sub_sources`` is None -- from ``slot_sources`` by the stage's slot id or
    kind, else the hand kernel. ``target="opencl"`` writes OpenCL slots,
    runtime and driver (``kb_export_opencl``); the partition, buffers, ops and
    data are the same.
    """
    import torch

    from .graph_hints import unique_names

    if specs is None:
        from .reference_kernels import KERNEL_SPECS
        specs = KERNEL_SPECS
    specs = dict(specs)
    tgt = _target(target)                        # None: CUDA (this module)
    mbir, walk_torch = _harness()
    ir_dir, out = Path(ir_dir).resolve(), Path(out_dir).resolve()
    ir = mbir.load_ir(ir_dir)
    g = unique_names(ir.graph)
    ops = g["ops"]
    sel = sorted(op_indices) if op_indices is not None else list(range(len(ops)))
    sel_set = set(sel)
    tensors = g["tensors"]
    specs = {k: _CudaSpelled(v) for k, v in specs.items()}     # slots are CUDA: __half

    with torch.no_grad():
        _, env = walk_torch.walk(ir, dev="cpu")

    # fused ops keep their internal tensor (A's output) as a scratch buffer
    produced = [t for i in sel for t in ops[i]["outputs"] + ops[i].get("internal", [])]
    produced_set = set(produced)
    consumed_outside = {t for i, o in enumerate(ops) if i not in sel_set for t in o["inputs"]}
    graph_outs = set(ir.output_names())
    outputs = [t for t in produced if t in consumed_outside or t in graph_outs]
    boundary = []
    for i in sel:
        for t in ops[i]["inputs"]:
            if t not in produced_set and t not in boundary:
                boundary.append(t)

    # ---- buffers: every tensor the partition touches, then the weights
    bufs, index = [], {}

    def add_buf(key, **kw):
        index[key] = len(bufs)
        bufs.append({"name": key, **kw})
        return index[key]

    alias_of = {}                               # tensor -> (base tensor, elem offset)
    for i in sel:
        o = ops[i]
        if o["op"] == "view":
            src = o["inputs"][0]
            base, off = alias_of.get(src, (src, 0))
            alias_of[o["outputs"][0]] = (base, off + int(o.get("offset", 0)))
        elif o["op"] == "chunk2_c1":
            src = o["inputs"][0]
            base, off = alias_of.get(src, (src, 0))
            hw = int(np.prod(tensors[src]["shape"][2:]))
            for k, t in enumerate(o["outputs"]):
                alias_of[t] = (base, off + k * int(o["shape"]["c_each"]) * hw)

    def kind_of(t):
        if tensors.get(t, {}).get("const"):
            return "const"
        if t in outputs:
            return "output"
        return "input" if t in boundary else "intermediate"

    order = boundary + [t for t in produced if t not in boundary]
    for t in order:                              # bases before aliases
        if t not in alias_of:
            add_buf(t, kind=kind_of(t), dtype=_tensor_dtype(ir, t),
                    numel=int(np.prod(ir.tshape(t))), alias=None)
    for t in order:
        if t in alias_of:
            base, off = alias_of[t]
            add_buf(t, kind=kind_of(t), dtype=_tensor_dtype(ir, t), numel=int(np.prod(ir.tshape(t))),
                    alias={"base": index[base], "offset": off})

    # ---- slots + per-op calls
    slots, slot_of, calls, op_recs = {}, {}, [], []
    dtype = _problem_dtype([ops[i]["op"] for i in sel if ops[i]["op"] not in ALIAS_OPS])
    lib = CudaLibrary(_hand_source(hand_cu, dtype)) if tgt is None else \
        tgt.library(None if Path(hand_cu) == DEFAULT_HAND_CU else hand_cu, dtype)
    for i in sel:
        o = ops[i]
        if o["op"] in ALIAS_OPS:
            continue
        kind = o["op"]
        if kind not in specs and kind.startswith("fused__"):      # rebuilt from its name
            from .kb_proposals import spec_from_kind
            specs[kind] = spec_from_kind(kind, specs)
        if kind not in specs:
            raise KeyError(f"{kind}: no KernelSpec")
        spec = specs[kind]
        entry, params = parse_signature(spec.signature)
        # one slot per C entry point: op kinds that share an entry (sum_dim /
        # max_dim -> kernel_reduce with a different opc) share its code
        sid, base_sid = slot_id_of(o, specs), base_slot_id(kind, specs)
        if sid not in slots:
            key = next((x for x in (sid, base_sid, kind) if slot_sources and x in slot_sources), None)
            if key is not None:
                src, tag = slot_sources[key], (slot_source_tags or {}).get(key, "override")
            elif tgt is not None:
                src, tag = tgt.seed(lib, entry, spec)
            elif hasattr(spec, "cuda_seed"):           # fused ops: stage kernels back to back
                if sub_sources is not None:
                    subs = sub_sources.get(sid, {})
                else:
                    subs = {}
                    for k in spec.sub_ops:
                        ssid = base_slot_id(k, specs)
                        hit = next((x for x in (ssid, k) if slot_sources and x in slot_sources), None)
                        if hit is not None:
                            subs[k] = slot_sources[hit]
                src = spec.cuda_seed(lib, sub_sources=subs)
                tag = "fused-seed" + (":" + ",".join(sorted(subs)) if subs else "")
            else:
                src, tag = lib.slot_source(entry), "hand"
            slots[sid] = {"id": sid, "op": kind, "ops": [], "op_class": op_class(kind), "entry": entry,
                          "signature": " ".join(spec.signature.split()),
                          "file": f"slots/{sid}{'.cu' if tgt is None else tgt.slot_ext}",
                          "reference_file": f"slots/{sid}.ref.c", "semantics": spec.semantics,
                          **({"variant_of": base_sid} if o.get("slot_variant") else {}),
                          "seed_source": tag, "instances": [], "_src": src,
                          "_ref": spec.reference_impl}
        if kind not in slots[sid]["ops"]:
            slots[sid]["ops"].append(kind)
            if kind != slots[sid]["op"] and spec.semantics:
                slots[sid]["semantics"] += f"\n[{kind}] {spec.semantics}"
        k = len(op_recs)
        args, ins, outs = [], [], []
        if "sub_ops" in o:
            from .kb_proposals import bind_fused
            arglist = bind_fused(o, specs)
        else:
            arglist = bind(o, params)
        for a in arglist:
            p = a.param
            if a.kind in ("in", "out", "scratch"):
                b = index[a.value]
                if a.kind != "scratch":
                    (ins if a.kind == "in" else outs).append(b)
                args.append(f"({p.ctype})b[{b}]")
            elif a.kind == "weight":
                key = f"w:{a.value}"
                if key not in index:
                    arr = ir.weights[a.value]
                    add_buf(key, kind="weight", dtype=cuda_dtypes.buffer_dtype(arr.dtype),
                            numel=int(arr.size), alias=None)
                args.append(f"({p.ctype})b[{index[key]}]")
            elif a.kind == "null":
                args.append("nullptr")
            else:
                args.append(_c_scalar(p.base_type, a.value))
        calls.append(f"    case {k}: {NS_PREFIX}{sid}::{entry}({', '.join(args)}); break;  // {o['name']}")
        slots[sid]["instances"].append({"op_name": o["name"], "op": kind, "index": k, "shape": o.get("shape", {})})
        op_recs.append({"index": k, "name": o["name"], "op": kind, "slot": sid,
                        "inputs": ins, "outputs": outs})

    # ---- data.bin: reference values of every non-alias tensor, then weights
    blob, off = bytearray(), 0

    def put(arr: np.ndarray) -> list[int]:
        nonlocal off
        pad = (-len(blob)) % 256
        blob.extend(b"\0" * pad)
        start = len(blob)
        blob.extend(np.ascontiguousarray(arr).tobytes())
        return [start, len(blob) - start]

    for b in bufs:
        if b["alias"] is not None:
            b["ref"] = None
            continue
        if b["kind"] == "weight":
            arr = ir.weights[b["name"][2:]]
        else:
            v = env[b["name"]]
            arr = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        arr = arr.astype(_NP_OF_BUF[b["dtype"]]).reshape(-1)       # fp16 values are exact
        assert arr.size == b["numel"], (b["name"], arr.size, b["numel"])
        b["ref"] = put(arr)

    # ---- write
    out.mkdir(parents=True, exist_ok=True)
    (out / "slots").mkdir(exist_ok=True)
    for s in slots.values():
        (out / s["file"]).write_text(s.pop("_src"))
        (out / s["reference_file"]).write_text(s.pop("_ref"))
    (out / "data.bin").write_bytes(bytes(blob))
    head, tail_name, driver_name = ("walker_head.cu", "walker_tail.cu", "driver.cpp") if tgt is None else \
        (tgt.head_name, tgt.tail_name, tgt.driver_name)
    (out / head).write_text(
        "// Shared preamble for every slot (from the ModelBlaster hand kernels).\n" + lib.head_source()
        if tgt is None else tgt.head_source(lib))
    tail = [
        "// Generated by modelblaster.pipeline.kb_export: one call per op, in IR order.",
        "void mb_run_op(int i, void **b) {", "    switch (i) {", *calls, "    default: break;", "    }", "}",
        f"int mb_num_ops() {{ return {len(op_recs)}; }}",
        "void launch_gpu_implementation(void **b) {",
        f"    for (int i = 0; i < {len(op_recs)}; ++i) mb_run_op(i, b);", "}", ""]
    (out / tail_name).write_text("\n".join(tail))
    tol = {**DTYPE_TOL[dtype], **(tolerance or {})}
    slot_list = list(slots.values())
    (out / driver_name).write_text((_driver_source if tgt is None else tgt.driver_source)(
        out / "data.bin", bufs, op_recs, slot_list, tol))
    man = {
        "format": FORMAT,
        "name": name or f"{ir.name}__{ops[sel[0]]['name']}__{ops[sel[-1]]['name']}",
        "source_ir": str(ir_dir), "ir_name": ir.name,
        "partition": {"strategy": strategy, "op_names": [ops[i]["name"] for i in sel],
                      "op_indices": sel, "is_whole_graph": len(sel) == len(ops)},
        "target": target, "dtype": dtype, "tolerance": tol,
        "assembly": {"head": head, "tail": tail_name,
                     "namespace_prefix": NS_PREFIX, "order": [s["id"] for s in slot_list]},
        "driver": driver_name, "data": "data.bin",
        "data_sha1": hashlib.sha1(bytes(blob)).hexdigest(),
        "slots": slot_list, "ops": op_recs, "buffers": bufs,
        "inputs": [index[t] for t in boundary], "outputs": [index[t] for t in outputs],
    }
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    return out


def _target(name: str):
    if name == "cuda":
        return None
    if name == "opencl":
        from .kb_export_opencl import OpenCLTarget
        return OpenCLTarget()
    raise ValueError(f"unknown target {name!r} (cuda|opencl)")


# ---------------------------------------------------------------- driver

def _driver_source(data_path: Path, bufs, op_recs, slots, tol) -> str:
    slot_ix = {s["id"]: k for k, s in enumerate(slots)}

    def arr(name, ctype, vals):
        vals = list(vals) or [0]
        return f"static const {ctype} {name}[] = {{{', '.join(str(v) for v in vals)}}};"

    nb = len(bufs)
    op_in = [i for r in op_recs for i in r["inputs"]]
    op_out = [i for r in op_recs for i in r["outputs"]]
    in_ofs, out_ofs, a, c = [], [], 0, 0
    for r in op_recs:
        in_ofs.append(a); a += len(r["inputs"])
        out_ofs.append(c); c += len(r["outputs"])
    in_ofs.append(a); out_ofs.append(c)
    kinds = {"input": 0, "const": 1, "weight": 2, "intermediate": 3, "output": 4}
    tables = "\n".join([
        f"static const int NB = {nb}, NOPS = {len(op_recs)}, NSLOTS = {len(slots)};",
        arr("B_KIND", "int", (kinds[b["kind"]] for b in bufs)),
        arr("B_INT", "int", (int(b["dtype"] in ("i32", "i64")) for b in bufs)),
        arr("B_I64", "int", (int(b["dtype"] == "i64") for b in bufs)),
        arr("B_HALF", "int", (int(b["dtype"] == "f16") for b in bufs)),
        arr("B_NUMEL", "long", (b["numel"] for b in bufs)),
        arr("B_BASE", "int", (b["alias"]["base"] if b["alias"] else -1 for b in bufs)),
        arr("B_AOFF", "long", (b["alias"]["offset"] if b["alias"] else 0 for b in bufs)),
        arr("B_REF", "long", (b["ref"][0] if b["ref"] else -1 for b in bufs)),
        arr("OP_IN_OFS", "int", in_ofs), arr("OP_IN", "int", op_in),
        arr("OP_OUT_OFS", "int", out_ofs), arr("OP_OUT", "int", op_out),
        arr("OP_SLOT", "int", (slot_ix[r["slot"]] for r in op_recs)),
        "static const char *B_NAME[] = {" + ", ".join(json.dumps(b["name"]) for b in bufs) + "};",
        "static const char *SLOT_NAME[] = {" + ", ".join(json.dumps(s["id"]) for s in slots) + "};",
        "static const char *OP_NAME[] = {" + ", ".join(json.dumps(r["name"]) for r in op_recs) + "};",
    ])
    return _DRIVER_TEMPLATE.replace("@BLOB@", _blob_source(data_path)).replace("@TABLES@", tables) \
        .replace("@OUT_REL@", repr(tol["output_rel"])).replace("@OUT_ABS@", repr(tol["output_abs"])) \
        .replace("@SLOT_REL@", repr(tol["slot_rel"])).replace("@SLOT_ABS@", repr(tol.get("slot_abs", 0.0)))


# data.bin is linked into the driver (.incbin) so the binary is self-contained
# (KernelBlaster's exec server only receives the executable); past ~1 GiB the
# relocations overflow, so big partitions read it at start-up instead
INCBIN_LIMIT = 1 << 30


def _blob_source(data_path: Path) -> str:
    if Path(data_path).stat().st_size < INCBIN_LIMIT:
        return ('__asm__(".section .rodata\\n.balign 256\\n.global mb_blob\\nmb_blob:\\n'
                f'.incbin \\"{data_path}\\"\\n.previous\\n");\n'
                'extern "C" const unsigned char mb_blob[];')
    # mmap, not fread: pages are read on first touch (the checks touch every
    # reference once; a 2.3 GB SwinV2 blob took 1.6 s to fread per run)
    return ("#include <fcntl.h>\n#include <sys/mman.h>\n#include <sys/stat.h>\n#include <unistd.h>\n"
            "static const unsigned char *mb_blob = [] {\n"
            f'    const char *path = getenv("MB_DATA_PATH") ? getenv("MB_DATA_PATH") : "{data_path}";\n'
            "    int fd = open(path, O_RDONLY); struct stat st;\n"
            "    if (fd < 0 || fstat(fd, &st) != 0) { printf(\"MB_ERROR cannot open %s\\n\", path); printf(\"failed\\n\"); exit(1); }\n"
            "    void *p = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);\n"
            "    close(fd);\n"
            "    if (p == MAP_FAILED) { printf(\"MB_ERROR cannot map %s\\n\", path); printf(\"failed\\n\"); exit(1); }\n"
            "    return (const unsigned char *)p;\n}();")


# The driver (see the module docstring for the default flow). Environment:
#   MB_MODE=profile     every op once on the reference inputs (for ncu), nothing else
#   MB_ONLY_SLOTS=a,b   isolated-op checks only for these slots' ops (the candidate
#                       changed only them); the partition outputs are still checked,
#                       and the partition and every op still timed
#   MB_SCREEN=1         screening budgets: 20 ms partition timing (1 warm-up, at
#                       least 1 rep), per-op passes until they reach the budget (1-5)
#   MB_TIME_MS / MB_OP_REPS   override the partition timing budget / per-op rep cap
#   MB_GRAPH=1          also capture the partition into a CUDA graph and time its
#                       replays (MB_PARTITION_GRAPH_US); needs the kernels built with
#                       --default-stream per-thread, else MB_GRAPH_SKIP <why>
# Every run ends with MB_PHASE_MS <phase> <ms> lines (wall time per phase).
_DRIVER_TEMPLATE = r'''// Generated by modelblaster.pipeline.kb_export -- checks and times one partition.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#ifdef MB_NVTX
#include <nvtx3/nvToolsExt.h>
#define MB_RANGE(n) nvtxRangePushA(n)
#define MB_RANGE_END() nvtxRangePop()
#else
#define MB_RANGE(n) ((void)0)
#define MB_RANGE_END() ((void)0)
#endif

void launch_gpu_implementation(void **bufs);
void mb_run_op(int i, void **bufs);
int mb_num_ops();

static double mb_now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static const double MB_T_START = mb_now_ms();      // before the blob below is mapped

@BLOB@

@TABLES@
static const double OUT_REL = @OUT_REL@, OUT_ABS = @OUT_ABS@, SLOT_REL = @SLOT_REL@, SLOT_ABS = @SLOT_ABS@;
enum { K_INPUT, K_CONST, K_WEIGHT, K_INTER, K_OUTPUT };

static void *dev[NB];
static long esz(int b) { return B_HALF[b] ? 2 : B_I64[b] ? 8 : 4; }   // f16: IEEE binary16; i64: argmax indices

// wall time per phase, printed as MB_PHASE_MS at the end
static std::vector<std::pair<const char *, double>> phases;
static double phase_t0 = MB_T_START;
static void phase(const char *name) { double t = mb_now_ms(); phases.push_back({name, t - phase_t0}); phase_t0 = t; }
static double env_num(const char *k, double dflt) { const char *v = getenv(k); return v && *v ? atof(v) : dflt; }

static bool ok_cuda(const char *what) {
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("MB_ERROR %s: %s\n", what, cudaGetErrorString(e)); return false; }
    return true;
}
static void upload(int b) {   // reference values into a non-alias buffer
    if (B_REF[b] >= 0) cudaMemcpy(dev[b], mb_blob + B_REF[b], B_NUMEL[b] * esz(b), cudaMemcpyHostToDevice);
}
// raw reference bytes of a buffer; an alias's are a slice of its base's
static const unsigned char *ref_raw(int b) {
    return B_BASE[b] >= 0 ? mb_blob + B_REF[B_BASE[b]] + B_AOFF[b] * esz(b) : mb_blob + B_REF[b];
}
static float half_to_float(unsigned short h) {
    unsigned s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ffu, f;
    if (e == 0x1f) f = s | 0x7f800000u | (m << 13);                  // inf / nan
    else if (e) f = s | ((e + 112) << 23) | (m << 13);               // normal
    else if (!m) f = s;                                                // +-0
    else { e = 113; while (!(m & 0x400u)) { m <<= 1; --e; } f = s | (e << 23) | ((m & 0x3ffu) << 13); }
    float x; memcpy(&x, &f, 4); return x;
}
// values as float (f16 widened; i32 buffers keep their bits, as before). An
// f32 buffer is used in place (no copy); `tmp` holds a widened copy otherwise.
static const float *as_float(const unsigned char *p, int b, std::vector<float> &tmp) {
    if (!B_HALF[b] && !B_I64[b]) return (const float *)p;
    tmp.resize(B_NUMEL[b]);
    if (B_HALF[b]) for (long i = 0; i < B_NUMEL[b]; ++i) { unsigned short h; memcpy(&h, p + 2 * i, 2); tmp[i] = half_to_float(h); }
    else for (long i = 0; i < B_NUMEL[b]; ++i) { int64_t q; memcpy(&q, p + 8 * i, 8); tmp[i] = (float)q; }
    return tmp.data();
}
static const float *ref_f(int b, std::vector<float> &tmp) { return as_float(ref_raw(b), b, tmp); }
// device values via one pinned staging buffer (grown on demand; pageable if
// pinned memory is not available)
static unsigned char *stage = nullptr;
static size_t stage_cap = 0;
static std::vector<unsigned char> stage_pageable;
static const float *dev_f(int b, std::vector<float> &tmp) {
    size_t n = B_NUMEL[b] * esz(b);
    unsigned char *dst;
    if (n <= stage_cap) dst = stage;
    else {
        if (stage) cudaFreeHost(stage);
        stage = nullptr; stage_cap = 0;
        if (cudaMallocHost((void **)&stage, n) == cudaSuccess) { stage_cap = n; dst = stage; }
        else { cudaGetLastError(); stage_pageable.resize(n); dst = stage_pageable.data(); }
    }
    cudaMemcpy(dst, dev[b], n, cudaMemcpyDeviceToHost);
    return as_float(dst, b, tmp);
}
// the checks scan every reference once (GBs for a whole model): big buffers
// are split across threads (max is order-independent: same result)
template <class F> static double par_max(long n, F f) {
    unsigned nt = std::min(8u, std::max(1u, std::thread::hardware_concurrency()));
    if (n < (1L << 20) || nt == 1) return f(0L, n);
    std::vector<double> part(nt, 0.0);
    std::vector<std::thread> th;
    long step = (n + nt - 1) / nt;
    for (unsigned t = 0; t < nt; ++t)
        th.emplace_back([&, t] { long a = t * step, b = std::min(n, a + step); part[t] = a < b ? f(a, b) : 0.0; });
    for (auto &x : th) x.join();
    return *std::max_element(part.begin(), part.end());
}
static double max_finite(const float *v, long n) {
    return par_max(n, [v](long a, long b) {
        double m = 0; for (long i = a; i < b; ++i) if (std::isfinite(v[i])) m = std::max(m, (double)std::fabs(v[i]));
        return m; });
}
// max |finite ref| per buffer, computed once
static std::vector<double> ref_scale(NB, -1.0);
static double ref_max_finite(int b) {
    if (ref_scale[b] < 0) { std::vector<float> t; ref_scale[b] = max_finite(ref_f(b, t), B_NUMEL[b]); }
    return ref_scale[b];
}
// max |a - r| (matching infinities/NaNs are exact; any other non-finite: inf)
static double max_err(const float *a, const float *r, long n) {
    return par_max(n, [a, r](long lo, long hi) {
        double e = 0;
        for (long i = lo; i < hi; ++i) {
            float x = a[i], y = r[i];
            if (x == y || (std::isnan(x) && std::isnan(y))) continue;
            if (!std::isfinite(x) || !std::isfinite(y)) return (double)INFINITY;
            e = std::max(e, (double)std::fabs(x - y));
        }
        return e; });
}
static void upload_inputs() {
    for (int b = 0; b < NB; ++b)
        if (B_BASE[b] < 0 && (B_KIND[b] == K_INPUT || B_KIND[b] == K_CONST || B_KIND[b] == K_WEIGHT)) upload(b);
}
static void zero_produced() {   // intermediates and outputs back to their start-up state
    for (int b = 0; b < NB; ++b)
        if (B_BASE[b] < 0 && (B_KIND[b] == K_INTER || B_KIND[b] == K_OUTPUT))
            cudaMemset(dev[b], 0, std::max(B_NUMEL[b], 1L) * esz(b));
}
// every partition output vs the reference (prints MB_CHECK lines with `tag`)
static bool check_outputs(const char *tag) {
    bool all = true;
    for (int b = 0; b < NB; ++b) {
        if (B_KIND[b] != K_OUTPUT) continue;
        std::vector<float> tr, ta;
        const float *r = ref_f(b, tr), *a = dev_f(b, ta);
        double e = max_err(a, r, B_NUMEL[b]), bound = OUT_REL * ref_max_finite(b) + OUT_ABS;
        bool ok = e <= bound;
        all &= ok;
        printf("%s %s max_abs_err=%.6g bound=%.6g ok=%d\n", tag, B_NAME[b], e, bound, (int)ok);
    }
    return all;
}
struct Dist { double mean, min, med, max; };
static Dist dist(std::vector<float> v) {
    Dist d{0, 0, 0, 0};
    if (v.empty()) return d;
    for (float x : v) d.mean += x;
    d.mean /= v.size();
    std::sort(v.begin(), v.end());
    d.min = v.front(); d.max = v.back(); d.med = v[v.size() / 2];
    return d;
}
// CUDA events on the legacy stream: an event there waits for every blocking
// stream, so work a slot puts on its own streams is inside the measurement
static cudaStream_t const TS = cudaStreamLegacy;
// time `body` (ms per call): warm-up, then at least min_reps and until `budget` ms or max_reps
template <class F> static std::vector<float> time_reps(F body, cudaStream_t s, int warm, int min_reps,
                                                       int max_reps, double budget) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int w = 0; w < warm; ++w) body();
    cudaDeviceSynchronize();
    std::vector<float> ms;
    double total = 0;
    while ((int)ms.size() < min_reps || (total < budget && (int)ms.size() < max_reps)) {
        float t; cudaEventRecord(e0, s); body(); cudaEventRecord(e1, s);
        cudaEventSynchronize(e1); cudaEventElapsedTime(&t, e0, e1); ms.push_back(t); total += t;
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms;
}
// time a and b alternately, rep by rep (both see the same clocks and caches)
template <class A, class B> static void time_pair(A a, cudaStream_t sa, B b, cudaStream_t sb, int warm, int min_reps,
                                                  int max_reps, double budget, std::vector<float> &ma,
                                                  std::vector<float> &mb) {
    cudaEvent_t e[4];
    for (auto &x : e) cudaEventCreate(&x);
    for (int w = 0; w < warm; ++w) { a(); b(); }
    cudaDeviceSynchronize();
    double total = 0;
    while ((int)ma.size() < min_reps || (total < 2 * budget && (int)ma.size() < max_reps)) {
        float t0, t1;
        cudaEventRecord(e[0], sa); a(); cudaEventRecord(e[1], sa); cudaEventSynchronize(e[1]);
        cudaEventRecord(e[2], sb); b(); cudaEventRecord(e[3], sb); cudaEventSynchronize(e[3]);
        cudaEventElapsedTime(&t0, e[0], e[1]); cudaEventElapsedTime(&t1, e[2], e[3]);
        ma.push_back(t0); mb.push_back(t1); total += t0 + t1;
    }
    for (auto &x : e) cudaEventDestroy(x);
}
// MB_GRAPH=1: the partition captured once on the per-thread default stream and
// checked (outputs of a replay on zeroed buffers). Not capturable (a slot
// syncs, allocates, copies to the host, launches on the legacy stream, ...) or
// not reproducing the outputs: MB_GRAPH_SKIP with the reason, no graph time.
static cudaGraphExec_t graph_build(size_t *nodes) {
    cudaStream_t s = cudaStreamPerThread;
    cudaGraph_t g = nullptr;
    cudaGraphExec_t ge = nullptr;
    cudaDeviceSynchronize();
    cudaGetLastError();
    MB_RANGE("graph_capture");
    cudaError_t e = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    if (e == cudaSuccess) {
        launch_gpu_implementation(dev);
        cudaError_t e2 = cudaStreamEndCapture(s, &g), e3 = cudaGetLastError();
        e = e2 != cudaSuccess ? e2 : e3;
    }
    *nodes = 0;
    if (e == cudaSuccess && g) cudaGraphGetNodes(g, nullptr, nodes);
    if (e == cudaSuccess && *nodes > 0) e = cudaGraphInstantiate(&ge, g, 0);
    if (g) cudaGraphDestroy(g);
    MB_RANGE_END();
    cudaDeviceSynchronize();
    cudaGetLastError();
    if (e != cudaSuccess || *nodes == 0 || !ge) {
        printf("MB_GRAPH_SKIP %s\n", e != cudaSuccess ? cudaGetErrorString(e)
               : "empty capture (kernels not on the per-thread default stream: build with --default-stream per-thread)");
        return nullptr;
    }
    zero_produced(); upload_inputs();
    cudaGraphLaunch(ge, s);
    if (!(ok_cuda("graph replay") && check_outputs("MB_GRAPH_CHECK output"))) {
        printf("MB_GRAPH_SKIP replay does not reproduce the eager outputs\n");
        cudaGraphExecDestroy(ge);
        ok_cuda("graph");
        return nullptr;
    }
    return ge;
}

int main() {
    bool all_ok = true;
    phase("start");                                         // process start + blob map, until main
    cudaFree(0);
    phase("cuda_init");
    for (int b = 0; b < NB; ++b)
        if (B_BASE[b] < 0) {
            if (cudaMalloc(&dev[b], std::max(B_NUMEL[b], 1L) * esz(b)) != cudaSuccess) { printf("MB_ERROR alloc %s\n", B_NAME[b]); printf("failed\n"); return 1; }
            cudaMemset(dev[b], 0, std::max(B_NUMEL[b], 1L) * esz(b));
        }
    for (int b = 0; b < NB; ++b)
        if (B_BASE[b] >= 0) dev[b] = (char *)dev[B_BASE[b]] + B_AOFF[b] * esz(b);
    if (mb_num_ops() != NOPS) { printf("MB_ERROR walker has %d ops, driver %d\n", mb_num_ops(), NOPS); printf("failed\n"); return 1; }

    // MB_MODE=profile: every op exactly once on the reference inputs, no checks
    // or timing reps -- what a profiler (ncu -k regex:mbslot_<slot>::) should see
    const char *mode = getenv("MB_MODE");
    if (mode && !strcmp(mode, "profile")) {
        upload_inputs();
        for (int i = 0; i < NOPS; ++i) mb_run_op(i, dev);
        bool ok = ok_cuda("profile");
        printf(ok ? "MB_PROFILE_DONE\n" : "failed\n");
        return ok ? 0 : 1;
    }
    // MB_ONLY_SLOTS: which slots' ops get the isolated check
    std::vector<int> sel_slot(NSLOTS, 1);
    const char *only = getenv("MB_ONLY_SLOTS");
    bool partial = only && *only;
    if (partial) {
        std::fill(sel_slot.begin(), sel_slot.end(), 0);
        std::string list(only);
        for (size_t p = 0; p <= list.size();) {
            size_t q = list.find(',', p);
            if (q == std::string::npos) q = list.size();
            std::string id = list.substr(p, q - p);
            p = q + 1;
            if (id.empty()) continue;
            int k = -1;
            for (int s = 0; s < NSLOTS; ++s) if (id == SLOT_NAME[s]) k = s;
            if (k < 0) { printf("MB_ERROR MB_ONLY_SLOTS: no slot %s\n", id.c_str()); printf("failed\n"); return 1; }
            sel_slot[k] = 1;
        }
        printf("MB_ONLY_SLOTS %s\n", only);
    }
    std::vector<int> sel_op(NOPS);
    for (int i = 0; i < NOPS; ++i) sel_op[i] = sel_slot[OP_SLOT[i]];
    bool screen = env_num("MB_SCREEN", 0) != 0;
    double budget = env_num("MB_TIME_MS", screen ? 20.0 : 200.0);
    int warm = screen ? 1 : 2, min_reps = screen ? 1 : 3, max_reps = 200;
    int op_cap = (int)env_num("MB_OP_REPS", screen ? 5 : 20);
    phase("alloc");

    // 1. whole partition vs reference
    MB_RANGE("check_partition");
    upload_inputs();
    launch_gpu_implementation(dev);
    if (!ok_cuda("partition")) { printf("failed\n"); return 1; }
    all_ok &= check_outputs("MB_CHECK output");
    MB_RANGE_END();
    phase("check_partition");

    // 2. every (selected) op alone on the reference's inputs: each of its
    // inputs, weights included, is reset to the reference first
    MB_RANGE("check_ops");
    std::vector<double> slot_err(NSLOTS, 0.0);
    std::vector<int> slot_bad(NSLOTS, 0);
    for (int i = 0; i < NOPS; ++i) {
        if (!sel_op[i]) continue;
        for (int k = OP_IN_OFS[i]; k < OP_IN_OFS[i + 1]; ++k) {
            int b = OP_IN[k];
            cudaMemcpy(dev[b], ref_raw(b), B_NUMEL[b] * esz(b), cudaMemcpyHostToDevice);
        }
        mb_run_op(i, dev);
        if (!ok_cuda(OP_NAME[i])) { slot_bad[OP_SLOT[i]] = 1; slot_err[OP_SLOT[i]] = INFINITY; continue; }
        double scale = 0;
        for (int k = OP_IN_OFS[i]; k < OP_IN_OFS[i + 1]; ++k)
            if (!B_INT[OP_IN[k]]) scale = std::max(scale, ref_max_finite(OP_IN[k]));
        for (int k = OP_OUT_OFS[i]; k < OP_OUT_OFS[i + 1]; ++k) {
            int b = OP_OUT[k];
            std::vector<float> tr, ta;
            const float *r = ref_f(b, tr), *a = dev_f(b, ta);
            double s = std::max(scale, ref_max_finite(b)), ae = max_err(a, r, B_NUMEL[b]), e = ae / std::max(s, 1e-12);
            slot_err[OP_SLOT[i]] = std::max(slot_err[OP_SLOT[i]], e);
            if (!(ae <= SLOT_REL * s + SLOT_ABS)) slot_bad[OP_SLOT[i]] = 1;   // abs: fp16 subnormals
        }
    }
    for (int s = 0; s < NSLOTS; ++s) {
        if (!sel_slot[s]) continue;
        all_ok &= !slot_bad[s];
        printf("MB_SLOT_CHECK %s rel_err=%.6g ok=%d\n", SLOT_NAME[s], slot_err[s], (int)!slot_bad[s]);
    }
    MB_RANGE_END();
    phase("check_ops");

    // 3. timing (CUDA events): whole partition (and its CUDA graph), then every op
    if (all_ok) {
        cudaGraphExec_t ge = nullptr;
        size_t nodes = 0;
        if (env_num("MB_GRAPH", 0) != 0) { ge = graph_build(&nodes); phase("graph_build"); }
        upload_inputs();
        MB_RANGE("time_partition");
        std::vector<float> pms, gms;
        if (ge) time_pair([] { launch_gpu_implementation(dev); }, TS,
                          [&] { cudaGraphLaunch(ge, cudaStreamPerThread); }, cudaStreamPerThread,
                          warm, min_reps, max_reps, budget, pms, gms);
        else pms = time_reps([] { launch_gpu_implementation(dev); }, TS, warm, min_reps, max_reps, budget);
        MB_RANGE_END();
        Dist d = dist(pms);
        int reps = (int)pms.size();
        printf("MB_PARTITION_US %.3f reps=%d\n", 1000.0 * d.mean, reps);
        printf("MB_PARTITION_DIST min=%.3f med=%.3f max=%.3f\n", 1000.0 * d.min, 1000.0 * d.med, 1000.0 * d.max);
        if (ge) {
            Dist g = dist(gms);
            printf("MB_PARTITION_GRAPH_US %.3f reps=%d nodes=%zu\n", 1000.0 * g.mean, (int)gms.size(), nodes);
            printf("MB_PARTITION_GRAPH_DIST min=%.3f med=%.3f max=%.3f\n", 1000.0 * g.min, 1000.0 * g.med, 1000.0 * g.max);
            cudaGraphExecDestroy(ge);
        }
        phase("time_partition");
        // per op: events around each op, every op in call order, also with
        // MB_ONLY_SLOTS (an op's time depends on what ran before it -- caches,
        // clocks -- so timing a subset would measure it differently)
        MB_RANGE("time_ops");
        std::vector<double> slot_us(NSLOTS, 0.0), op_us(NOPS, 0.0);
        // reps: as many as the partition got (at most MB_OP_REPS); screening:
        // until the timed passes add up to the budget (at least 1, at most MB_OP_REPS)
        int oreps = screen ? op_cap : std::max(1, std::min(reps, op_cap)), done = 0;
        std::vector<cudaEvent_t> ev(2 * NOPS);
        for (auto &e : ev) cudaEventCreate(&e);
        double timed_ms = 0;
        for (int r = 0; r < oreps && NOPS > 0; ++r) {
            for (int i = 0; i < NOPS; ++i) { cudaEventRecord(ev[2 * i], TS); mb_run_op(i, dev); cudaEventRecord(ev[2 * i + 1], TS); }
            cudaEventSynchronize(ev[2 * NOPS - 1]);
            for (int i = 0; i < NOPS; ++i) {
                float ms; cudaEventElapsedTime(&ms, ev[2 * i], ev[2 * i + 1]);
                slot_us[OP_SLOT[i]] += 1000.0 * ms; op_us[i] += 1000.0 * ms; timed_ms += ms;
            }
            ++done;
            if (screen && timed_ms >= budget) break;
        }
        for (auto &v : slot_us) v /= std::max(done, 1);
        for (auto &v : op_us) v /= std::max(done, 1);
        for (auto &e : ev) cudaEventDestroy(e);
        MB_RANGE_END();
        for (int s = 0; s < NSLOTS; ++s) printf("MB_SLOT_US %s %.3f\n", SLOT_NAME[s], slot_us[s]);
        for (int i = 0; i < NOPS; ++i) printf("MB_OP_US %s %.3f\n", OP_NAME[i], op_us[i]);
        all_ok &= ok_cuda("timing");
        phase("time_ops");
    }
    double total = 0;
    for (auto &p : phases) { printf("MB_PHASE_MS %s %.1f\n", p.first, p.second); total += p.second; }
    printf("MB_PHASE_MS total %.1f\n", total);
    printf(all_ok ? "passed\n" : "failed\n");
    return all_ok ? 0 : 1;
}
'''


# ---------------------------------------------------------------- local build/run (MB-side checks)

def parse_driver_output(text: str) -> dict:
    res = {"passed": bool(re.search(r"^passed$", text, re.M)), "outputs": {}, "slots": {},
           "slot_us": {}, "op_us": {}, "partition_us": None, "partition_graph_us": None,
           "graph_skip": None, "phase_ms": {},
           "errors": re.findall(r"^MB_ERROR (.*)$", text, re.M)}
    for m in re.finditer(r"^MB_CHECK output (\S+) max_abs_err=(\S+) bound=(\S+) ok=(\d)", text, re.M):
        res["outputs"][m.group(1)] = {"max_abs_err": float(m.group(2)), "bound": float(m.group(3)),
                                      "ok": m.group(4) == "1"}
    for m in re.finditer(r"^MB_SLOT_CHECK (\S+) rel_err=(\S+) ok=(\d)", text, re.M):
        res["slots"][m.group(1)] = {"rel_err": float(m.group(2)), "ok": m.group(3) == "1"}
    for m in re.finditer(r"^MB_SLOT_US (\S+) (\S+)", text, re.M):
        res["slot_us"][m.group(1)] = float(m.group(2))
    res["op_us"] = {m.group(1): float(m.group(2)) for m in re.finditer(r"^MB_OP_US (\S+) (\S+)", text, re.M)}
    m = re.search(r"^MB_PARTITION_US (\S+)", text, re.M)
    if m:
        res["partition_us"] = float(m.group(1))
    m = re.search(r"^MB_PARTITION_GRAPH_US (\S+)", text, re.M)
    if m:
        res["partition_graph_us"] = float(m.group(1))
    m = re.search(r"^MB_GRAPH_SKIP (.*)$", text, re.M)
    if m:
        res["graph_skip"] = m.group(1).strip()
    for k, pat in (("partition_dist", "MB_PARTITION_DIST"), ("partition_graph_dist", "MB_PARTITION_GRAPH_DIST")):
        m = re.search(rf"^{pat} min=(\S+) med=(\S+) max=(\S+)", text, re.M)
        if m:
            res[k] = {"min": float(m.group(1)), "med": float(m.group(2)), "max": float(m.group(3))}
    res["phase_ms"] = {m.group(1): float(m.group(2)) for m in re.finditer(r"^MB_PHASE_MS (\S+) (\S+)", text, re.M)}
    return res


# nvcc flags of a local build; PTDS_FLAGS puts every <<<>>> launch on the
# per-thread default stream, which MB_GRAPH=1 needs to capture the partition
NVCC_FLAGS = ["-O3", "-std=c++17"]
PTDS_FLAGS = ["--default-stream", "per-thread"]


def run_built(exe, timeout: int = 1800, env: dict | None = None) -> dict:
    """Run a built driver; ``env`` adds driver variables (MB_ONLY_SLOTS, MB_SCREEN, MB_GRAPH, ...)."""
    import os

    rp = subprocess.run([str(exe)], capture_output=True, text=True, timeout=timeout,
                        env={**os.environ, **{k: str(v) for k, v in (env or {}).items()}})
    res = parse_driver_output(rp.stdout)
    res.update(stage="run", log=(rp.stdout + rp.stderr)[-4000:])
    return res


def build_and_run(problem_dir, *, arch: str = "75", sources: dict[str, str] | None = None,
                  work: Path | None = None, nvcc: str = "nvcc", timeout: int = 1800, runs: int = 1,
                  env: dict | None = None, graph: bool = False) -> dict:
    """Assemble + nvcc-build + run the problem outside KernelBlaster.

    ``runs`` > 1 runs the binary that many times and keeps the run with the
    median partition time (``partition_us_runs`` has them all); ``runs=0``
    only builds (``exe`` is the binary). ``graph`` builds with the per-thread
    default stream and runs with ``MB_GRAPH=1``; ``env`` is passed to the driver."""
    problem_dir = Path(problem_dir)
    work = Path(work or problem_dir / "build")
    work.mkdir(parents=True, exist_ok=True)
    (work / "kernels.cu").write_text(assemble(problem_dir, sources))
    exe = work / "problem"
    cp = subprocess.run([nvcc, *NVCC_FLAGS, f"-arch=sm_{arch}", *(PTDS_FLAGS if graph else []), "-o", str(exe),
                         str(work / "kernels.cu"), str(problem_dir / "driver.cpp")],
                        capture_output=True, text=True, timeout=timeout)
    if cp.returncode != 0:
        return {"passed": False, "stage": "compile", "log": (cp.stdout + cp.stderr)[-4000:]}
    if runs <= 0:
        return {"passed": None, "stage": "built", "exe": str(exe)}
    env = {**({"MB_GRAPH": "1"} if graph else {}), **(env or {})}
    rs = [run_built(exe, timeout, env) for _ in range(runs)]
    ok = [r for r in rs if r["passed"] and r["partition_us"]]
    if not ok:
        return {**rs[0], "exe": str(exe)}
    ok.sort(key=lambda r: r["partition_us"])
    return {**ok[len(ok) // 2], "exe": str(exe), "partition_us_runs": [r["partition_us"] for r in rs if r["partition_us"]]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ir_dir", nargs="?")
    ap.add_argument("out_dir")
    ap.add_argument("--ops", help="comma-separated op names")
    ap.add_argument("--range", help="op index range i:j")
    ap.add_argument("--strategy", default=None, choices=["whole", "per_op", "window"],
                    help="export every partition of the strategy under OUT_DIR/pNN")
    ap.add_argument("--max-ops", type=int, default=12)
    ap.add_argument("--plan", help="export the problems of a kb_plan file (modelblaster.kb_plan/v1); "
                                   "IR_DIR is not used")
    ap.add_argument("--cache", help="with --plan: seed the problems from this kernel cache")
    ap.add_argument("--hand-cu", default=str(DEFAULT_HAND_CU))
    ap.add_argument("--hints", "--fusion-registry", dest="hints",
                    help="graph hints file: apply its accepted hints first")
    ap.add_argument("--build-check", action="store_true", help="build and run each problem (nvcc / g++ + OpenCL)")
    ap.add_argument("--arch", default="75")
    ap.add_argument("--target", default="cuda", choices=["cuda", "opencl"])
    a = ap.parse_args()
    if a.plan:
        from .kb_ingest import CudaCache
        from .kb_plan import export_plan
        cache = CudaCache(Path(a.cache)) if a.cache else None
        for d in export_plan(json.loads(Path(a.plan).read_text()), Path(a.out_dir), cache=cache, arch=a.arch):
            print(json.dumps({"problem": str(d)}), flush=True)
        return
    ir_dir = Path(a.ir_dir)
    g = json.loads((ir_dir / "graph.json").read_text())
    if a.hints:
        from . import graph_hints as gh
        ir_dir, _, prov = gh.apply_to_ir(ir_dir, gh.load(Path(a.hints)), Path(a.out_dir) / "hinted_ir")
        if prov["applied"]:
            g = json.loads((ir_dir / "graph.json").read_text())
            print(json.dumps({"applied": prov["applied"]}), flush=True)
    if a.strategy:
        parts = [(f"p{k:02d}", p) for k, p in enumerate(partition_ir(g, a.strategy, a.max_ops))]
    elif a.ops:
        from .graph_hints import unique_names
        names = a.ops.split(",")
        parts = [("", [i for i, o in enumerate(unique_names(g)["ops"]) if o["name"] in names])]
    elif a.range:
        i, j = (int(x) for x in a.range.split(":"))
        parts = [("", list(range(i, j)))]
    else:
        parts = [("", None)]
    for sub, idx in parts:
        d = export_problem(ir_dir, Path(a.out_dir) / sub if sub else a.out_dir, op_indices=idx,
                           hand_cu=Path(a.hand_cu), strategy=a.strategy or "custom", target=a.target)
        line = {"problem": str(d)}
        if a.build_check:
            if a.target == "opencl":
                from .kb_export_opencl import build_and_run as build_and_run_cl
                r = build_and_run_cl(d)
            else:
                r = build_and_run(d, arch=a.arch)
            line.update({k: r.get(k) for k in ("passed", "stage", "partition_us", "slot_us", "slots", "errors")})
            if not r.get("passed"):
                line["log"] = r.get("log", "")[-1500:]
        print(json.dumps(line), flush=True)


if __name__ == "__main__":
    main()
