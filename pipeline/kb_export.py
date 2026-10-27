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
DEFAULT_HAND_CU_BF16 = HARNESS / "cuda" / "mb_cuda_kernels_bf16.cu"  # generated (f16_kernels.py)
# the casts of a mixed IR (cast_<src>_to_<dst>), beside the hand kernels
CAST_CU = "mb_cuda_kernels_cast.cu"
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


# ---------------------------------------------------------------- storage dtypes (fp32 / fp16 / bf16)

class _CudaSpelled:
    """A KernelSpec seen with the CUDA spelling of its signature (`_Float16`
    -> `__half`, `__bf16` -> `__nv_bfloat16`, cuda_dtypes.cuda_signature);
    everything else passes through."""

    def __init__(self, spec):
        self._spec = spec

    def __getattr__(self, k):
        return getattr(self._spec, k)

    @property
    def signature(self) -> str:
        return cuda_dtypes.cuda_signature(self._spec.signature)


def _problem_dtype(kinds: list[str], mixed: bool = False) -> str:
    """fp16 / bf16 when every op of the partition is an `_f16` / `_bf16` op, else fp32.
    A mixed IR's partition (``mixed``; --quant native) whose ops have more than one
    storage dtype is ``mixed``: each slot then carries its own (manifest slot ``dtype``)."""
    dts = {cuda_dtypes.op_dtype(k) for k in kinds}
    if mixed and len(dts) > 1:
        return MIXED
    return dts.pop() if len(dts) == 1 else cuda_dtypes.FP32


MIXED = "mixed"


def _hand_source(hand_cu: Path, dtype: str) -> str:
    """The hand kernels for a dtype: fp16 / bf16 use the generated `<stem>_f16.cu`
    / `<stem>_bf16.cu` beside ``hand_cu`` (cuda/mb_cuda_kernels_f16.cu /
    _bf16.cu for the default)."""
    hand_cu = Path(hand_cu)
    if dtype in cuda_dtypes.SUFFIX:
        hand_cu = hand_cu.with_name(hand_cu.stem + cuda_dtypes.SUFFIX[dtype] + hand_cu.suffix)
    return hand_cu.read_text()


def _tensor_dtype(ir, t: str) -> str:
    """Buffer dtype of IR tensor t: a constant's is its weights.npz array's
    (a mixed IR names it in ``weight_dtypes``), else graph.json's (f16 / bf16 for
    an fp16 / bf16 extract; every tensor's own in a mixed IR)."""
    key = ir.graph["tensors"].get(t, {}).get("const")
    if key:
        return cuda_dtypes.weight_dtype(ir.graph, key, ir.weights[key].dtype)
    dt = ir.graph["tensors"].get(t, {}).get("dtype")
    # i64: the index outputs of argmax/argmin (kernel_argmax_dim writes int64_t);
    # i32 / b8 / u8: a mixed IR's integer / bool graph inputs and outputs
    return {"f16": "f16", "bf16": "bf16", "i64": "i64", "i32": "i32", "b8": "b8", "u8": "u8"}.get(dt, "f32")


_NP_OF_BUF = {"f32": np.float32, "f16": np.float16, "i32": np.int32, "i64": np.int64, "b8": np.uint8,
              "u8": np.uint8}


def _buffer_values(arr: np.ndarray, dtype: str) -> np.ndarray:
    """A buffer's reference values as stored on the device (bf16: the 2-byte
    patterns; the walk's values are bf16-exact, so this is exact; b8: bool as
    one byte 0 / 1)."""
    if dtype == "bf16":
        return cuda_dtypes.bf16_bits(arr)
    if dtype == "b8":
        arr = np.asarray(arr) != 0
    return arr.astype(_NP_OF_BUF[dtype]).reshape(-1)               # fp16 values are exact


class _Libraries:
    """The hand-kernel libraries a partition draws its slot seeds from: one per
    storage dtype (fp32 / `_f16` / `_bf16` hand files) plus the casts file for a
    mixed IR. Every entry name is unique across them (``kernel_relu`` /
    ``kernel_relu_bf16`` / ``kernel_cast_bf16_to_f32``), so a slot's seed is the
    closure of its entry in the library that defines it, and the shared head is
    the union of the used libraries' preprocessor lines. A single-dtype partition
    uses exactly one library, as before."""

    def __init__(self, hand_cu: Path, dtypes, casts: bool):
        self.libs = {d: CudaLibrary(_hand_source(hand_cu, d)) for d in dtypes}
        if casts:
            self.libs["cast"] = CudaLibrary((Path(hand_cu).with_name(CAST_CU)).read_text())

    def of(self, kind: str) -> CudaLibrary:
        return self.libs["cast" if cuda_dtypes.parse_cast(kind) else cuda_dtypes.op_dtype(kind)]

    def head_source(self) -> str:
        if len(self.libs) == 1:
            return next(iter(self.libs.values())).head_source()
        lines = []
        for lib in self.libs.values():
            lines += [ln for ln in lib.head if ln not in lines]
        return "\n".join(lines) + "\n"


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
    kinds = [ops[i]["op"] for i in sel if ops[i]["op"] not in ALIAS_OPS]
    dtype = _problem_dtype(kinds, cuda_dtypes.is_mixed(g))
    casts = any(cuda_dtypes.parse_cast(k) for k in kinds)
    if tgt is not None:
        if casts:
            raise NotImplementedError("per-tensor casts (--quant native, docs/tensor_dtypes.md) are "
                                      f"CUDA-only; no {target} seeds")
        lib = libs = tgt.library(None if Path(hand_cu) == DEFAULT_HAND_CU else hand_cu, dtype)
    else:
        libs = _Libraries(hand_cu, sorted({cuda_dtypes.op_dtype(k) for k in kinds if not cuda_dtypes.parse_cast(k)})
                          or [cuda_dtypes.FP32 if dtype == MIXED else dtype], casts)
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
                lib = libs.of(kind)
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
                src, tag = libs.of(kind).slot_source(entry), "hand"
            sdt = cuda_dtypes.op_dtype(kind)
            slots[sid] = {"id": sid, "op": kind, "ops": [], "op_class": op_class(kind), "entry": entry,
                          "signature": " ".join(spec.signature.split()),
                          "file": f"slots/{sid}{'.cu' if tgt is None else tgt.slot_ext}",
                          "reference_file": f"slots/{sid}.ref.c", "semantics": spec.semantics,
                          **({"variant_of": base_sid} if o.get("slot_variant") else {}),
                          "seed_source": tag, "instances": [], "_src": src,
                          "_ref": spec.reference_impl,
                          # the slot's storage dtype and its isolated-op bound (a mixed problem's
                          # slots differ; a cast slot's is its destination's)
                          "dtype": sdt, "tolerance": {k: DTYPE_TOL[sdt][k] for k in ("slot_rel", "slot_abs")}}
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
                    add_buf(key, kind="weight", dtype=cuda_dtypes.weight_dtype(ir.graph, a.value, arr.dtype),
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
        if dtype == MIXED:                       # the driver's isolated bound of this op (its dtype's)
            op_recs[-1]["tolerance"] = {**slots[sid]["tolerance"], **{
                k: v for k, v in (tolerance or {}).items() if k in ("slot_rel", "slot_abs")}}

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
        arr = _buffer_values(arr, b["dtype"])
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
        "// Shared preamble for every slot (from the ModelBlaster hand kernels).\n" + libs.head_source()
        if tgt is None else tgt.head_source(lib))
    tail = [
        "// Generated by modelblaster.pipeline.kb_export: one call per op, in IR order.",
        "void mb_run_op(int i, void **b) {", "    switch (i) {", *calls, "    default: break;", "    }", "}",
        f"int mb_num_ops() {{ return {len(op_recs)}; }}",
        "void launch_gpu_implementation(void **b) {",
        f"    for (int i = 0; i < {len(op_recs)}; ++i) mb_run_op(i, b);", "}",
        *([] if tgt is not None else SO_ABI), ""]
    (out / tail_name).write_text("\n".join(tail))
    tol = {**DTYPE_TOL[dtype if dtype != MIXED else _mixed_bound(slots)], **(tolerance or {})}
    if dtype == MIXED:
        # per output: its dtype's bound, or a narrower one it was computed through
        # (an fp32 output of bf16 intermediates carries their rounding)
        cone = cuda_dtypes.cone_dtypes(g, [b["name"] for b in bufs if b["kind"] == "output"])
        for b in bufs:
            if b["kind"] == "output":
                b["tolerance"] = {k: v for k, v in {**DTYPE_TOL[cone[b["name"]]], **(tolerance or {})}.items()
                                  if k in ("output_rel", "output_abs")}
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


# C entry points of a candidate's shared object (the slot units + this tail,
# built -shared): what the persistent host (driver.cpp built -DMB_HOST) dlopens
SO_ABI = [
    "// persistent-host ABI (driver.cpp -DMB_HOST dlopens these from the candidate's .so)",
    'extern "C" int mb_so_abi() { return 1; }',
    'extern "C" void mb_so_run_op(int i, void **b) { mb_run_op(i, b); }',
    'extern "C" void mb_so_launch(void **b) { launch_gpu_implementation(b); }',
    'extern "C" int mb_so_num_ops() { return mb_num_ops(); }',
]


def _mixed_bound(slots: dict) -> str:
    """A mixed problem's manifest-level dtype for its bounds: its narrowest slot dtype."""
    return cuda_dtypes.narrowest(s["dtype"] for s in slots.values())


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
        arr("B_BF16", "int", (int(b["dtype"] == "bf16") for b in bufs)),
        # mixed IRs: i32 / bool / u8 graph inputs and outputs, read as numbers
        arr("B_I32", "int", (int(b["dtype"] == "i32") for b in bufs)),
        arr("B_BYTE", "int", (int(b["dtype"] in ("b8", "u8")) for b in bufs)),
        # bounds per output buffer / per op (a mixed problem's differ by dtype)
        arr("B_OUT_REL", "double", (repr(float(b.get("tolerance", tol)["output_rel"])) for b in bufs)),
        arr("B_OUT_ABS", "double", (repr(float(b.get("tolerance", tol)["output_abs"])) for b in bufs)),
        arr("OP_SLOT_REL", "double", (repr(float(r.get("tolerance", tol)["slot_rel"])) for r in op_recs)),
        arr("OP_SLOT_ABS", "double", (repr(float(r.get("tolerance", tol).get("slot_abs", 0.0))) for r in op_recs)),
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
#   MB_WALL=1           also the host wall clock per forward pass and the steady-state
#                       throughput of every timed body (MB_TIMING lines; MB_THRU_N passes)
#   MB_WARM=N           warm-up passes before the partition timing (default 2, screening 1)
# Every run ends with MB_PHASE_MS <phase> <ms> lines (wall time per phase).
_DRIVER_TEMPLATE = r'''// Generated by modelblaster.pipeline.kb_export -- checks and times one partition.
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
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

// the kernels: linked in (this driver run as one process), or loaded from a
// candidate's shared object by the persistent host (-DMB_HOST, below)
struct MbKernels {
    void (*run_op)(int, void **) = nullptr;
    void (*launch)(void **) = nullptr;
    int (*num_ops)() = nullptr;
    void *handle = nullptr;
};
#ifndef MB_HOST
void launch_gpu_implementation(void **bufs);
void mb_run_op(int i, void **bufs);
int mb_num_ops();
static MbKernels K{mb_run_op, launch_gpu_implementation, mb_num_ops, nullptr};
#else
#include <dlfcn.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/file.h>
#include <unistd.h>
static MbKernels K;
#endif
static MbKernels K_REF;                 // the incumbent (host only): paired end-to-end time

static double mb_now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static const double MB_T_START = mb_now_ms();      // before the blob below is mapped

@BLOB@

@TABLES@
static const double OUT_REL = @OUT_REL@, OUT_ABS = @OUT_ABS@, SLOT_REL = @SLOT_REL@, SLOT_ABS = @SLOT_ABS@;
enum { K_INPUT, K_CONST, K_WEIGHT, K_INTER, K_OUTPUT };

static void *dev[NB];
static long esz(int b) { return B_HALF[b] || B_BF16[b] ? 2 : B_I64[b] ? 8 : B_BYTE[b] ? 1 : 4; }   // f16: IEEE binary16; bf16: bfloat16; i64: argmax indices; bool / u8: one byte

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
// values as float (f16 / bf16 widened; i64, and a mixed IR's i32 / bool / u8 io,
// converted; other i32 buffers -- gather indices, never checked -- keep their bits,
// as before). An f32 buffer is used in place (no copy); `tmp` holds a widened copy
// otherwise.
static const float *as_float(const unsigned char *p, int b, std::vector<float> &tmp) {
    if (!B_HALF[b] && !B_BF16[b] && !B_I64[b] && !B_I32[b] && !B_BYTE[b]) return (const float *)p;
    tmp.resize(B_NUMEL[b]);
    if (B_I32[b]) { for (long i = 0; i < B_NUMEL[b]; ++i) { int32_t q; memcpy(&q, p + 4 * i, 4); tmp[i] = (float)q; } return tmp.data(); }
    if (B_BYTE[b]) { for (long i = 0; i < B_NUMEL[b]; ++i) tmp[i] = (float)p[i]; return tmp.data(); }
    if (B_HALF[b]) for (long i = 0; i < B_NUMEL[b]; ++i) { unsigned short h; memcpy(&h, p + 2 * i, 2); tmp[i] = half_to_float(h); }
    else if (B_BF16[b]) for (long i = 0; i < B_NUMEL[b]; ++i) {   // bf16 = the top half of an fp32
        unsigned short h; memcpy(&h, p + 2 * i, 2); unsigned f = (unsigned)h << 16; memcpy(&tmp[i], &f, 4); }
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
        double e = max_err(a, r, B_NUMEL[b]), bound = B_OUT_REL[b] * ref_max_finite(b) + B_OUT_ABS[b];
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
// time `bodies` round-robin, rep by rep (ms per call each; all see the same
// clocks and caches): warm-up, then at least min_reps and until every body has
// had `budget` ms or max_reps. One body is a plain timing loop.
static std::vector<std::vector<float>> time_multi(const std::vector<std::function<void()>> &bodies,
                                                  const std::vector<cudaStream_t> &st, int warm, int min_reps,
                                                  int max_reps, double budget) {
    size_t n = bodies.size();
    std::vector<cudaEvent_t> e(2 * n);
    for (auto &x : e) cudaEventCreate(&x);
    for (int w = 0; w < warm; ++w) for (auto &b : bodies) b();
    cudaDeviceSynchronize();
    std::vector<std::vector<float>> ms(n);
    double total = 0;
    while ((int)ms[0].size() < min_reps || (total < n * budget && (int)ms[0].size() < max_reps)) {
        for (size_t k = 0; k < n; ++k) {
            cudaEventRecord(e[2 * k], st[k]); bodies[k](); cudaEventRecord(e[2 * k + 1], st[k]);
            cudaEventSynchronize(e[2 * k + 1]);
        }
        for (size_t k = 0; k < n; ++k) {
            float t; cudaEventElapsedTime(&t, e[2 * k], e[2 * k + 1]); ms[k].push_back(t); total += t;
        }
    }
    for (auto &x : e) cudaEventDestroy(x);
    return ms;
}
// MB_WALL=1: next to the event times of time_multi, for every body (k: eager, graph,
// ref), the host wall clock per forward pass -- from issuing it to the return of
// cudaDeviceSynchronize, the GPU idle before each, as many reps as the events got, round
// robin -- and the steady-state throughput: MB_THRU_N (default: the reps, at least 10)
// passes back to back after one warm pass, one sync at the end, per pass (wall and
// events). One MB_TIMING line per body (us; med is the driver's upper median).
static double stdev(const std::vector<float> &v) {
    if (v.size() < 2) return 0;
    double m = 0, q = 0;
    for (float x : v) m += x;
    m /= v.size();
    for (float x : v) q += (x - m) * (x - m);
    return std::sqrt(q / (v.size() - 1));
}
static void time_wall(const std::vector<std::function<void()>> &bodies, const std::vector<cudaStream_t> &st,
                      const std::vector<const char *> &names, const std::vector<std::vector<float>> &gpu) {
    size_t n = bodies.size();
    int reps = (int)gpu[0].size(), thru = (int)env_num("MB_THRU_N", std::max(10, reps));
    std::vector<std::vector<float>> wall(n);
    cudaDeviceSynchronize();
    for (int r = 0; r < reps; ++r)
        for (size_t k = 0; k < n; ++k) {
            double t0 = mb_now_ms();
            bodies[k]();
            cudaDeviceSynchronize();
            wall[k].push_back((float)(mb_now_ms() - t0));
        }
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (size_t k = 0; k < n; ++k) {
        bodies[k](); cudaDeviceSynchronize();
        double t0 = mb_now_ms();
        cudaEventRecord(e0, st[k]);
        for (int i = 0; i < thru; ++i) bodies[k]();
        cudaEventRecord(e1, st[k]);
        cudaEventSynchronize(e1);
        cudaDeviceSynchronize();
        double tw = mb_now_ms() - t0;
        float tg = 0; cudaEventElapsedTime(&tg, e0, e1);
        Dist g = dist(gpu[k]), w = dist(wall[k]);
        printf("MB_TIMING %s gpu_med=%.3f gpu_mean=%.3f gpu_min=%.3f gpu_max=%.3f gpu_std=%.3f gpu_n=%d "
               "wall_med=%.3f wall_mean=%.3f wall_min=%.3f wall_max=%.3f wall_std=%.3f wall_n=%d "
               "thru_wall=%.3f thru_gpu=%.3f thru_n=%d\n", names[k],
               1000.0 * g.med, 1000.0 * g.mean, 1000.0 * g.min, 1000.0 * g.max, 1000.0 * stdev(gpu[k]), (int)gpu[k].size(),
               1000.0 * w.med, 1000.0 * w.mean, 1000.0 * w.min, 1000.0 * w.max, 1000.0 * stdev(wall[k]), (int)wall[k].size(),
               1000.0 * tw / thru, 1000.0 * tg / thru, thru);
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
}
// MB_GRAPH=1: the partition captured once on the per-thread default stream and
// checked (outputs of a replay on zeroed buffers). Not capturable (a slot
// syncs, allocates, copies to the host, launches on the legacy stream, ...) or
// not reproducing the outputs: <tag>_SKIP with the reason, no graph time.
static cudaGraphExec_t graph_build(const MbKernels &k, size_t *nodes, const char *tag) {
    cudaStream_t s = cudaStreamPerThread;
    cudaGraph_t g = nullptr;
    cudaGraphExec_t ge = nullptr;
    cudaDeviceSynchronize();
    cudaGetLastError();
    MB_RANGE("graph_capture");
    cudaError_t e = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    if (e == cudaSuccess) {
        k.launch(dev);
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
        printf("%s_SKIP %s\n", tag, e != cudaSuccess ? cudaGetErrorString(e)
               : "empty capture (kernels not on the per-thread default stream: build with --default-stream per-thread)");
        return nullptr;
    }
    zero_produced(); upload_inputs();
    cudaGraphLaunch(ge, s);
    std::string ctag = std::string(tag) + "_CHECK output";
    if (!(ok_cuda("graph replay") && check_outputs(ctag.c_str()))) {
        printf("%s_SKIP replay does not reproduce the eager outputs\n", tag);
        cudaGraphExecDestroy(ge);
        ok_cuda("graph");
        return nullptr;
    }
    return ge;
}
// Per-op attribution: what each op's kernels (and memsets/copies) take on the
// GPU, from CUPTI's activity records -- the durations nsys reports -- for ops
// run back to back as in the partition, without events between them (an
// event pair around an op also holds a few us of launch and event gaps, which
// made tiny ops look 2-3x bigger). libcupti is dlopen'ed at run time (MB_CUPTI_LIB,
// else the usual names); without it, or built without cupti.h, MB_ATTR_SKIP.
#if !defined(MB_NO_CUPTI) && __has_include(<cupti.h>)
#include <cupti.h>
#include <dlfcn.h>
#include <mutex>
#define MB_HAVE_CUPTI 1
namespace mbcupti {
static CUptiResult (*enable_)(CUpti_ActivityKind), (*disable_)(CUpti_ActivityKind);
static CUptiResult (*register_)(CUpti_BuffersCallbackRequestFunc, CUpti_BuffersCallbackCompleteFunc);
static CUptiResult (*flush_)(uint32_t);
static CUptiResult (*next_)(uint8_t *, size_t, CUpti_Activity **);
static CUptiResult (*push_)(CUpti_ExternalCorrelationKind, uint64_t);
static CUptiResult (*pop_)(CUpti_ExternalCorrelationKind, uint64_t *);
static CUptiResult (*finalize_)();
static void *lib = nullptr;
static int state = 0;                                   // 0 not tried, 1 ready, -1 unavailable
static std::string why;
struct Rec { uint32_t corr; uint64_t ns; };
static std::vector<Rec> recs;
static std::vector<std::pair<uint32_t, uint64_t>> ext;  // correlation id -> op index + 1
static std::mutex mu;
static const CUpti_ActivityKind KINDS[] = {CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, CUPTI_ACTIVITY_KIND_MEMCPY, CUPTI_ACTIVITY_KIND_RUNTIME,
                                           CUPTI_ACTIVITY_KIND_MEMSET, CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION};
static void CUPTIAPI buf_req(uint8_t **b, size_t *sz, size_t *maxrec) {
    *sz = 8 << 20; *b = (uint8_t *)aligned_alloc(8, *sz); *maxrec = 0;
}
static void CUPTIAPI buf_done(CUcontext, uint32_t, uint8_t *b, size_t, size_t valid) {
    CUpti_Activity *r = nullptr;
    std::lock_guard<std::mutex> g(mu);
    while (b && next_(b, valid, &r) == CUPTI_SUCCESS) {
        if (r->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || r->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
            auto *k = (CUpti_ActivityKernel9 *)r; recs.push_back({(uint32_t)k->correlationId, (uint64_t)(k->end - k->start)});
        } else if (r->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
            auto *k = (CUpti_ActivityMemcpy5 *)r; recs.push_back({(uint32_t)k->correlationId, (uint64_t)(k->end - k->start)});
        } else if (r->kind == CUPTI_ACTIVITY_KIND_MEMSET) {
            auto *k = (CUpti_ActivityMemset4 *)r; recs.push_back({(uint32_t)k->correlationId, (uint64_t)(k->end - k->start)});
        } else if (r->kind == CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) {
            auto *e = (CUpti_ActivityExternalCorrelation *)r; ext.push_back({(uint32_t)e->correlationId, (uint64_t)e->externalId});
        }
    }
    free(b);
}
template <class F> static bool sym(void *h, F &f, const char *n) { f = (F)dlsym(h, n); return f != nullptr; }
static bool init() {
    if (state) return state > 0;
    state = -1;
    const char *names[] = {getenv("MB_CUPTI_LIB"), "libcupti.so.12", "libcupti.so",
                           "/usr/local/cuda/extras/CUPTI/lib64/libcupti.so"};
    void *h = lib;
    for (const char *n : names) if (!h && n && *n) h = dlopen(n, RTLD_NOW | RTLD_GLOBAL);
    if (!h) { why = "libcupti not found (set MB_CUPTI_LIB)"; return false; }
    lib = h;
    finalize_ = (CUptiResult (*)())dlsym(h, "cuptiFinalize");
    if (!(sym(h, enable_, "cuptiActivityEnable") && sym(h, disable_, "cuptiActivityDisable") &&
          sym(h, register_, "cuptiActivityRegisterCallbacks") && sym(h, flush_, "cuptiActivityFlushAll") &&
          sym(h, next_, "cuptiActivityGetNextRecord") && sym(h, push_, "cuptiActivityPushExternalCorrelationId") &&
          sym(h, pop_, "cuptiActivityPopExternalCorrelationId"))) { why = "libcupti lacks the activity API"; return false; }
    if (register_(buf_req, buf_done) != CUPTI_SUCCESS) { why = "cuptiActivityRegisterCallbacks failed"; return false; }
    state = 1;
    return true;
}
static void end_kinds() { for (auto k : KINDS) disable_(k); }
// detach CUPTI after a pass: attached, it keeps adding to every launch (+1 % on a
// graph replay), which the persistent host's next request would time
static void detach() {
    if (finalize_ && finalize_() == CUPTI_SUCCESS) state = 0;
}
static bool begin() {
    recs.clear(); ext.clear();
    for (auto k : KINDS) if (enable_(k) != CUPTI_SUCCESS) { why = "cuptiActivityEnable failed"; end_kinds(); return false; }
    return true;
}
}  // namespace mbcupti
#endif

static int mb_setup() {
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
    return 0;
}

// One evaluation of the kernels in K (and, in the host, the incumbent in K_REF
// for a paired end-to-end comparison): checks, then timing. Prints the MB_*
// lines and "passed"/"failed"; 0 if passed.
static int mb_eval() {
    bool all_ok = true;
    if (K.num_ops() != NOPS) { printf("MB_ERROR walker has %d ops, driver %d\n", K.num_ops(), NOPS); printf("failed\n"); return 1; }
    zero_produced();                     // the state a fresh process starts from

    // MB_MODE=profile: every op exactly once on the reference inputs, no checks
    // or timing reps -- what a profiler (ncu -k regex:mbslot_<slot>::) should see
    const char *mode = getenv("MB_MODE");
    if (mode && !strcmp(mode, "profile")) {
        upload_inputs();
        for (int i = 0; i < NOPS; ++i) K.run_op(i, dev);
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
    int warm = (int)env_num("MB_WARM", screen ? 1 : 2), min_reps = screen ? 1 : 3, max_reps = 200;
    int op_cap = (int)env_num("MB_OP_REPS", screen ? 5 : 20);
    phase("alloc");

    // 1. whole partition vs reference
    MB_RANGE("check_partition");
    upload_inputs();
    K.launch(dev);
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
        K.run_op(i, dev);
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
            if (!(ae <= OP_SLOT_REL[i] * s + OP_SLOT_ABS[i])) slot_bad[OP_SLOT[i]] = 1;   // abs: fp16 subnormals (0 for fp32 / bf16)
        }
    }
    for (int s = 0; s < NSLOTS; ++s) {
        if (!sel_slot[s]) continue;
        all_ok &= !slot_bad[s];
        printf("MB_SLOT_CHECK %s rel_err=%.6g ok=%d\n", SLOT_NAME[s], slot_err[s], (int)!slot_bad[s]);
    }
    MB_RANGE_END();
    phase("check_ops");

    // 3. timing (CUDA events): the whole partition as deployed (eager, its CUDA
    // graph, and the incumbent's in the host), the per-op attribution, every op
    if (all_ok) {
        cudaGraphExec_t ge = nullptr, gr = nullptr;
        size_t nodes = 0, rnodes = 0;
        bool want_graph = env_num("MB_GRAPH", 0) != 0, have_ref = K_REF.launch != nullptr;
        if (want_graph) { ge = graph_build(K, &nodes, "MB_GRAPH"); phase("graph_build"); }
        // paired with the incumbent on the same basis: both graph replays when both
        // capture, else both eager (a candidate that breaks capture is compared eager to
        // eager, interleaved; its own deployed time is still its eager time)
        if (have_ref && ge) gr = graph_build(K_REF, &rnodes, "MB_REF_GRAPH");
        upload_inputs();
        MB_RANGE("time_partition");
        // bodies: candidate eager [, candidate graph] [, incumbent as deployed]
        std::vector<std::function<void()>> bodies{[] { K.launch(dev); }};
        std::vector<cudaStream_t> st{TS};
        if (ge) { bodies.push_back([&] { cudaGraphLaunch(ge, cudaStreamPerThread); }); st.push_back(cudaStreamPerThread); }
        if (have_ref && gr) { bodies.push_back([&] { cudaGraphLaunch(gr, cudaStreamPerThread); }); st.push_back(cudaStreamPerThread); }
        else if (have_ref) { bodies.push_back([] { K_REF.launch(dev); }); st.push_back(TS); }
        std::vector<std::vector<float>> tm = time_multi(bodies, st, warm, min_reps, max_reps, budget);
        if (env_num("MB_WALL", 0) != 0) {
            std::vector<const char *> names{"eager"};
            if (ge) names.push_back("graph");
            if (have_ref) names.push_back("ref");
            time_wall(bodies, st, names, tm);
        }
        MB_RANGE_END();
        const std::vector<float> &pms = tm[0];
        Dist d = dist(pms);
        int reps = (int)pms.size();
        printf("MB_PARTITION_US %.3f reps=%d\n", 1000.0 * d.mean, reps);
        printf("MB_PARTITION_DIST min=%.3f med=%.3f max=%.3f\n", 1000.0 * d.min, 1000.0 * d.med, 1000.0 * d.max);
        Dist g{0, 0, 0, 0};
        double ge_us = 0;
        if (ge) {
            g = dist(tm[1]);
            ge_us = 1000.0 * g.mean;
            printf("MB_PARTITION_GRAPH_US %.3f reps=%d nodes=%zu\n", 1000.0 * g.mean, (int)tm[1].size(), nodes);
            printf("MB_PARTITION_GRAPH_DIST min=%.3f med=%.3f max=%.3f\n", 1000.0 * g.min, 1000.0 * g.med, 1000.0 * g.max);
        }
        // end to end as deployed: the graph replay where the partition is capturable, else eager
        const std::vector<float> &dep = ge ? tm[1] : tm[0];
        Dist e2e = ge ? g : d;
        printf("MB_E2E_US %.3f basis=%s mean=%.3f reps=%d\n", 1000.0 * e2e.med, ge ? "graph" : "eager",
               1000.0 * e2e.mean, (int)dep.size());
        if (have_ref) {
            const std::vector<float> &rm = tm.back();
            Dist rd = dist(rm);
            std::vector<float> ratio;
            const std::vector<float> &cm = gr ? tm[1] : tm[0];
            for (size_t r = 0; r < rm.size() && r < cm.size(); ++r) if (rm[r] > 0) ratio.push_back(cm[r] / rm[r]);
            Dist q = dist(ratio);
            printf("MB_REF_E2E_US %.3f basis=%s mean=%.3f reps=%d\n", 1000.0 * rd.med, gr ? "graph" : "eager",
                   1000.0 * rd.mean, (int)rm.size());
            printf("MB_E2E_RATIO med=%.5f mean=%.5f min=%.5f max=%.5f n=%d basis=%s\n", q.med, q.mean, q.min, q.max,
                   (int)ratio.size(), gr ? "graph" : "eager");
        }
        if (ge) cudaGraphExecDestroy(ge);
        if (gr) cudaGraphExecDestroy(gr);
        phase("time_partition");

        // per op: every op in call order, also with MB_ONLY_SLOTS (an op's time
        // depends on what ran before it -- caches, clocks -- so timing a subset
        // would measure it differently). reps: as many as the partition got (at
        // most MB_OP_REPS); screening: until the timed passes add up to the
        // budget (at least 1, at most MB_OP_REPS)
        int oreps = screen ? op_cap : std::max(1, std::min(reps, op_cap));
        std::vector<cudaEvent_t> ev(2 * NOPS);
        for (auto &e : ev) cudaEventCreate(&e);
        // 3a. attribution (MB_ATTR=0: off): GPU time of every op's kernels, run
        // back to back as in the partition (CUPTI activity records)
        if (env_num("MB_ATTR", 1) != 0 && NOPS > 0) {
            MB_RANGE("time_attr");
#ifdef MB_HAVE_CUPTI
            if (mbcupti::init() && mbcupti::begin()) {
                std::vector<double> op_a(NOPS, 0.0);
                std::vector<long> op_n(NOPS, 0);
                int done = 0;
                double t_wall = mb_now_ms();
                cudaDeviceSynchronize();
                for (int r = 0; r < oreps; ++r) {
                    for (int i = 0; i < NOPS; ++i) {
                        mbcupti::push_(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, (uint64_t)i + 1);
                        K.run_op(i, dev);
                        uint64_t id; mbcupti::pop_(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &id);
                    }
                    cudaDeviceSynchronize();
                    ++done;
                    if (screen && mb_now_ms() - t_wall >= budget) break;
                }
                mbcupti::flush_(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
                mbcupti::end_kinds();
                mbcupti::flush_(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
                std::vector<std::pair<uint32_t, uint64_t>> ext;
                std::vector<mbcupti::Rec> recs;
                { std::lock_guard<std::mutex> g(mbcupti::mu); ext.swap(mbcupti::ext); recs.swap(mbcupti::recs); }
                mbcupti::detach();
                std::sort(ext.begin(), ext.end());
                long unmatched = 0;
                for (auto &rc : recs) {
                    auto it = std::lower_bound(ext.begin(), ext.end(), std::make_pair(rc.corr, (uint64_t)0));
                    if (it == ext.end() || it->first != rc.corr || it->second < 1 || it->second > (uint64_t)NOPS) { ++unmatched; continue; }
                    op_a[it->second - 1] += rc.ns / 1000.0; op_n[it->second - 1]++;
                }
                std::vector<double> slot_a(NSLOTS, 0.0);
                std::vector<long> slot_n(NSLOTS, 0);
                double sum = 0;
                long launches = 0;
                for (int i = 0; i < NOPS; ++i) {
                    op_a[i] /= std::max(done, 1);
                    slot_a[OP_SLOT[i]] += op_a[i]; slot_n[OP_SLOT[i]] += op_n[i]; sum += op_a[i]; launches += op_n[i];
                }
                for (int s = 0; s < NSLOTS; ++s)
                    printf("MB_ATTR_SLOT_US %s %.3f launches=%.1f\n", SLOT_NAME[s], slot_a[s], (double)slot_n[s] / std::max(done, 1));
                for (int i = 0; i < NOPS; ++i) printf("MB_ATTR_OP_US %s %.3f\n", OP_NAME[i], op_a[i]);
                printf("MB_ATTR_SUM_US %.3f basis=cupti reps=%d launches=%.1f unmatched=%ld\n", sum, done,
                       (double)launches / std::max(done, 1), unmatched);
                // what the partition costs beyond its kernels: launch latency and gaps
                printf("MB_LAUNCH_OVERHEAD_US eager=%.3f", 1000.0 * d.mean - sum);
                if (ge_us > 0) printf(" graph=%.3f", ge_us - sum);
                printf(" frac_eager=%.4f\n", d.mean > 0 ? std::max(0.0, 1.0 - sum / (1000.0 * d.mean)) : 0.0);
            } else printf("MB_ATTR_SKIP %s\n", mbcupti::why.c_str());
#else
            printf("MB_ATTR_SKIP built without cupti.h (add -I<cuda>/extras/CUPTI/include)\n");
#endif
            MB_RANGE_END();
            all_ok &= ok_cuda("attribution");
            phase("time_attr");
        }
        // 3b. events around each op as launched (MB_SLOT_US / MB_OP_US; MB_OP_US_PASS=0: off)
        if (env_num("MB_OP_US_PASS", 1) != 0) {
            MB_RANGE("time_ops");
            std::vector<double> slot_us(NSLOTS, 0.0), op_us(NOPS, 0.0);
            int done = 0;
            double timed_ms = 0;
            for (int r = 0; r < oreps && NOPS > 0; ++r) {
                for (int i = 0; i < NOPS; ++i) { cudaEventRecord(ev[2 * i], TS); K.run_op(i, dev); cudaEventRecord(ev[2 * i + 1], TS); }
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
            MB_RANGE_END();
            for (int s = 0; s < NSLOTS; ++s) printf("MB_SLOT_US %s %.3f\n", SLOT_NAME[s], slot_us[s]);
            for (int i = 0; i < NOPS; ++i) printf("MB_OP_US %s %.3f\n", OP_NAME[i], op_us[i]);
            phase("time_ops");
        }
        for (auto &e : ev) cudaEventDestroy(e);
        all_ok &= ok_cuda("timing");
    }
    double total = 0;
    for (auto &p : phases) { printf("MB_PHASE_MS %s %.1f\n", p.first, p.second); total += p.second; }
    printf("MB_PHASE_MS total %.1f\n", total);
    printf(all_ok ? "passed\n" : "failed\n");
    return all_ok ? 0 : 1;
}

#ifndef MB_HOST
int main() {
    if (mb_setup()) return 1;
    return mb_eval();
}
#else
// ---- the persistent host (built with -DMB_HOST, kernels not linked): keeps
// the CUDA context, the buffers and the reference data, and evaluates one
// candidate per request, its kernels dlopen'ed from a shared object built from
// the same units (the MB_SO_* entry points of walker_tail.cu). Protocol, one
// request per stdin line, fields separated by tabs:
//   EVAL <nonce> so=<path> [ref=<path>] [lock=<path>] [lock_wait=<s>] [MB_*=<value> ...]
//   QUIT
// The reply is what the one-process driver prints for the same MB_* variables
// (ref= adds the incumbent's paired end-to-end time), then
//   MB_HOST_DONE <nonce> rc=<0|1> poisoned=<0|1> recycle=<0|1>
// After a request whose kernels left the context unusable (poisoned=1), or
// once the process has leaked MB_HOST_LEAK_MB of device memory or served
// MB_HOST_MAX_EVALS requests (recycle=1), the host exits; the client starts a
// new one. A crash ends the stream without MB_HOST_DONE.
static const char *HOST_VARS[] = {"MB_MODE", "MB_ONLY_SLOTS", "MB_SCREEN", "MB_TIME_MS", "MB_OP_REPS", "MB_GRAPH",
                                  "MB_ATTR", "MB_OP_US_PASS", "MB_WALL", "MB_WARM", "MB_THRU_N"};
static bool load_so(const char *path, MbKernels &k, std::string &err) {
    void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) { err = dlerror(); return false; }
    auto abi = (int (*)())dlsym(h, "mb_so_abi");
    k.run_op = (void (*)(int, void **))dlsym(h, "mb_so_run_op");
    k.launch = (void (*)(void **))dlsym(h, "mb_so_launch");
    k.num_ops = (int (*)())dlsym(h, "mb_so_num_ops");
    if (!abi || abi() != 1 || !k.run_op || !k.launch || !k.num_ops) {
        err = "not an MB slot library (mb_so_abi 1)"; dlclose(h); k = MbKernels{}; return false;
    }
    k.handle = h;
    return true;
}
static void unload_so(MbKernels &k) {
    if (k.handle) dlclose(k.handle);
    k = MbKernels{};
}
// the GPU's cross-process lock (flock, as KernelBlaster's gpu_lock_wrap): -1
// unlocked (no file, or not ours to open), -2 timed out
static int gpu_lock(const std::string &path, double wait_s) {
    if (path.empty()) return -1;
    int fd = open(path.c_str(), O_RDONLY | O_CREAT, 0666);
    if (fd < 0) return -1;
    double t0 = mb_now_ms();
    while (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        if (mb_now_ms() - t0 > 1000.0 * wait_s) { close(fd); return -2; }
        usleep(2000);
    }
    return fd;
}
static void gpu_unlock(int fd) { if (fd >= 0) { flock(fd, LOCK_UN); close(fd); } }
static bool poisoned() {   // a sticky error leaves every later call failing
    static void *probe = nullptr;
    cudaGetLastError();
    cudaError_t e = cudaDeviceSynchronize();
    if (e == cudaSuccess && !probe) e = cudaMalloc(&probe, 256);
    if (e == cudaSuccess) e = cudaMemset(probe, 0, 256);
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) printf("MB_HOST_POISONED %s\n", cudaGetErrorString(e));
    return e != cudaSuccess;
}
int main() {
    setvbuf(stdout, nullptr, _IOLBF, 1 << 16);
    // the client is gone (its end of stdin closed: killed, or the ssh channel
    // dropped) -> leave, even while a candidate's kernel hangs
    std::thread([] {
        for (;;) {
            struct pollfd p{0, 0, 0};
            if (poll(&p, 1, 200) > 0 && (p.revents & (POLLHUP | POLLERR | POLLNVAL))) _exit(4);
        }
    }).detach();
    if (mb_setup()) { fflush(stdout); return 1; }
    size_t free0 = 0, total = 0, freen = 0;
    cudaMemGetInfo(&free0, &total);
    double leak_mb = env_num("MB_HOST_LEAK_MB", 1024), max_evals = env_num("MB_HOST_MAX_EVALS", 500);
    printf("MB_HOST_READY abi=1 nb=%d nops=%d nslots=%d init_ms=%.1f\n", NB, NOPS, NSLOTS, mb_now_ms() - MB_T_START);
    fflush(stdout);
    char *line = nullptr;
    size_t cap = 0;
    int evals = 0;
    while (getline(&line, &cap, stdin) > 0) {
        std::vector<std::string> f;
        for (char *p = strtok(line, "\t\r\n"); p; p = strtok(nullptr, "\t\r\n")) f.push_back(p);
        if (f.empty()) continue;
        if (f[0] == "QUIT") break;
        if (f[0] != "EVAL" || f.size() < 2) { printf("MB_HOST_BAD_REQUEST %s\n", f[0].c_str()); fflush(stdout); continue; }
        std::string nonce = f[1], so, ref, lock;
        double lock_wait = 3600;
        for (const char *v : HOST_VARS) unsetenv(v);
        for (size_t i = 2; i < f.size(); ++i) {
            size_t eq = f[i].find('=');
            if (eq == std::string::npos) continue;
            std::string k = f[i].substr(0, eq), v = f[i].substr(eq + 1);
            if (k == "so") so = v;
            else if (k == "ref") ref = v;
            else if (k == "lock") lock = v;
            else if (k == "lock_wait") lock_wait = atof(v.c_str());
            else if (k.rfind("MB_", 0) == 0) setenv(k.c_str(), v.c_str(), 1);
        }
        phases.clear();
        phase_t0 = mb_now_ms();
        int rc = 1, fd = gpu_lock(lock, lock_wait);
        std::string err;
        if (fd == -2) printf("MB_ERROR [infra] GPU lock %s not acquired in %.0f s\nfailed\n", lock.c_str(), lock_wait);
        else if (!load_so(so.c_str(), K, err)) printf("MB_ERROR cannot load %s: %s\nfailed\n", so.c_str(), err.c_str());
        else if (!ref.empty() && !load_so(ref.c_str(), K_REF, err))
            printf("MB_ERROR cannot load incumbent %s: %s\nfailed\n", ref.c_str(), err.c_str());
        else {
            phase("load");
            rc = mb_eval();
        }
        bool bad = poisoned();
        if (bad) {                 // no dlclose on a broken context: report and leave
            gpu_unlock(fd);
            printf("MB_HOST_DONE %s rc=1 poisoned=1 recycle=1\n", nonce.c_str());
            fflush(stdout);
            _exit(3);
        }
        unload_so(K_REF);
        unload_so(K);
        gpu_unlock(fd);
        ++evals;
        cudaMemGetInfo(&freen, &total);
        bool recycle = evals >= max_evals || (double)free0 - (double)freen > leak_mb * 1e6;
        printf("MB_HOST_DONE %s rc=%d poisoned=0 recycle=%d\n", nonce.c_str(), rc, (int)recycle);
        fflush(stdout);
        if (recycle) break;
    }
    free(line);
    return 0;
}
#endif
'''


# ---------------------------------------------------------------- local build/run (MB-side checks)

def parse_driver_output(text: str) -> dict:
    from . import kb_trace
    kb_trace.get().output("driver", text)      # raw run output (graph optimiser trace; no-op otherwise)
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
    res.update(parse_e2e(text))
    return res


def parse_e2e(text: str) -> dict:
    """The end-to-end and attribution lines (MB_E2E_US, MB_REF_E2E_US,
    MB_E2E_RATIO, MB_ATTR_*, MB_LAUNCH_OVERHEAD_US); absent lines -> None / {}."""
    res = {"e2e_us": None, "e2e_basis": None, "e2e_mean_us": None, "ref_e2e_us": None, "ref_e2e_basis": None,
           "e2e_ratio": None, "attr_slot_us": {}, "attr_op_us": {}, "attr_sum_us": None, "attr_basis": None,
           "attr_launches": None, "attr_slot_launches": {}, "attr_skip": None,
           "launch_overhead_us": None, "launch_overhead_graph_us": None, "launch_overhead_frac": None,
           "graph_nodes": None, "ref_graph_skip": None}
    m = re.search(r"^MB_E2E_US (\S+) basis=(\w+) mean=(\S+)", text, re.M)
    if m:
        res.update(e2e_us=float(m.group(1)), e2e_basis=m.group(2), e2e_mean_us=float(m.group(3)))
    m = re.search(r"^MB_REF_E2E_US (\S+) basis=(\w+)", text, re.M)
    if m:
        res.update(ref_e2e_us=float(m.group(1)), ref_e2e_basis=m.group(2))
    m = re.search(r"^MB_E2E_RATIO med=(\S+) mean=(\S+) min=(\S+) max=(\S+) n=(\d+)(?: basis=(\w+))?", text, re.M)
    if m:
        res["e2e_ratio"] = {"med": float(m.group(1)), "mean": float(m.group(2)), "min": float(m.group(3)),
                            "max": float(m.group(4)), "n": int(m.group(5)), "basis": m.group(6) or "eager"}
    for m in re.finditer(r"^MB_ATTR_SLOT_US (\S+) (\S+)(?: launches=(\S+))?", text, re.M):
        res["attr_slot_us"][m.group(1)] = float(m.group(2))
        if m.group(3):
            res["attr_slot_launches"][m.group(1)] = float(m.group(3))
    res["attr_op_us"] = {m.group(1): float(m.group(2)) for m in re.finditer(r"^MB_ATTR_OP_US (\S+) (\S+)", text, re.M)}
    m = re.search(r"^MB_ATTR_SUM_US (\S+) basis=(\w+) reps=\d+ launches=(\S+)", text, re.M)
    if m:
        res.update(attr_sum_us=float(m.group(1)), attr_basis=m.group(2), attr_launches=float(m.group(3)))
    m = re.search(r"^MB_ATTR_SKIP (.*)$", text, re.M)
    if m:
        res["attr_skip"] = m.group(1).strip()
    m = re.search(r"^MB_LAUNCH_OVERHEAD_US eager=(\S+)(?: graph=(\S+))? frac_eager=(\S+)", text, re.M)
    if m:
        res.update(launch_overhead_us=float(m.group(1)), launch_overhead_frac=float(m.group(3)),
                   launch_overhead_graph_us=float(m.group(2)) if m.group(2) else None)
    m = re.search(r"^MB_PARTITION_GRAPH_US \S+ reps=\d+ nodes=(\d+)", text, re.M)
    if m:
        res["graph_nodes"] = int(m.group(1))
    m = re.search(r"^MB_REF_GRAPH_SKIP (.*)$", text, re.M)
    if m:
        res["ref_graph_skip"] = m.group(1).strip()
    res["timing"] = parse_timing(text)
    return res


def parse_timing(text: str) -> dict:
    """``MB_TIMING`` lines (``MB_WALL=1``) -> {body: {"gpu": {med, mean, min, max, std, n},
    "wall": {...}, "thru_wall_us", "thru_gpu_us", "thru_n"}} (us); none -> {}."""
    out = {}
    for m in re.finditer(r"^MB_TIMING (\S+) (.*)$", text, re.M):
        kv = dict(x.split("=", 1) for x in m.group(2).split() if "=" in x)
        d = {}
        for side in ("gpu", "wall"):
            d[side] = {k: (int(kv[f"{side}_{k}"]) if k == "n" else float(kv[f"{side}_{k}"]))
                       for k in ("med", "mean", "min", "max", "std", "n") if f"{side}_{k}" in kv}
        d.update(thru_wall_us=float(kv["thru_wall"]) if "thru_wall" in kv else None,
                 thru_gpu_us=float(kv["thru_gpu"]) if "thru_gpu" in kv else None,
                 thru_n=int(kv["thru_n"]) if "thru_n" in kv else None)
        out[m.group(1)] = d
    return out


# nvcc flags of a local build; PTDS_FLAGS puts every <<<>>> launch on the
# per-thread default stream, which MB_GRAPH=1 needs to capture the partition
NVCC_FLAGS = ["-O3", "-std=c++17"]
PTDS_FLAGS = ["--default-stream", "per-thread"]


def cupti_flags(nvcc: str = "nvcc") -> list[str]:
    """-I for CUPTI's headers beside nvcc (the driver's per-op attribution);
    [] without them (the driver then reports MB_ATTR_SKIP)."""
    import shutil

    exe = shutil.which(nvcc) or nvcc
    inc = Path(exe).resolve().parents[1] / "extras" / "CUPTI" / "include"
    return [f"-I{inc}"] if (inc / "cupti.h").exists() else []


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
                  env: dict | None = None, graph: bool = False, runtime: str = "process") -> dict:
    """Assemble + nvcc-build + run the problem outside KernelBlaster.

    ``runtime="host"`` builds the persistent host and the kernels' shared
    object and evaluates them in one host request (CUDA graph on, as the host
    always builds with the per-thread default stream); ``runs`` requests."""
    if runtime == "host":
        return _build_and_run_host(problem_dir, arch=arch, sources=sources, work=work, nvcc=nvcc,
                                   timeout=timeout, runs=runs, env=env)
    return _build_and_run_process(problem_dir, arch=arch, sources=sources, work=work, nvcc=nvcc,
                                  timeout=timeout, runs=runs, env=env, graph=graph)


def _build_and_run_host(problem_dir, *, arch, sources, work, nvcc, timeout, runs, env) -> dict:
    problem_dir = Path(problem_dir)
    work = Path(work or problem_dir / "build")
    host, log = build_host(problem_dir, arch=arch, work=work, nvcc=nvcc, timeout=timeout)
    so, log2 = build_so(problem_dir, arch=arch, sources=sources, work=work, nvcc=nvcc, timeout=timeout) \
        if host else (None, "")
    if host is None or so is None:
        return {"passed": False, "stage": "compile", "log": log if host is None else log2}
    if runs <= 0:
        return {"passed": None, "stage": "built", "exe": str(host), "so": str(so)}
    h = DriverHost(host, timeout=timeout)
    try:
        rs = [h.evaluate(so, env={"MB_GRAPH": "1", **(env or {})}) for _ in range(runs)]
    finally:
        h.close()
    ok = sorted((r for r in rs if r["passed"] and r["partition_us"]), key=lambda r: r["partition_us"])
    best = ok[len(ok) // 2] if ok else rs[0]
    return {**best, "exe": str(host), "so": str(so), "partition_us_runs": [r["partition_us"] for r in rs if r["partition_us"]]}


def _build_and_run_process(problem_dir, *, arch, sources, work, nvcc, timeout, runs, env, graph) -> dict:
    """The one-process driver: kernels and driver linked into one binary.

    ``runs`` > 1 runs the binary that many times and keeps the run with the
    median partition time (``partition_us_runs`` has them all); ``runs=0``
    only builds (``exe`` is the binary). ``graph`` builds with the per-thread
    default stream and runs with ``MB_GRAPH=1``; ``env`` is passed to the driver."""
    problem_dir = Path(problem_dir)
    work = Path(work or problem_dir / "build")
    work.mkdir(parents=True, exist_ok=True)
    (work / "kernels.cu").write_text(assemble(problem_dir, sources))
    exe = work / "problem"
    cmd = [nvcc, *NVCC_FLAGS, f"-arch=sm_{arch}", *(PTDS_FLAGS if graph else []), *cupti_flags(nvcc), "-o", str(exe),
           str(work / "kernels.cu"), str(problem_dir / "driver.cpp")]
    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    from . import kb_trace
    kb_trace.get().tool("nvcc", cmd, cp.returncode, cp.stdout + cp.stderr)
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


# ---------------------------------------------------------------- the persistent host (driver.cpp -DMB_HOST)
#
# One long-lived process per (problem, GPU) keeps the CUDA context, the
# buffers and the reference data; each candidate's kernels are a shared object
# (the slot units + walker_tail.cu, built -shared with the per-thread default
# stream) that the host dlopens for one request and dlcloses after. The checks
# and timing are the one-process driver's code (mb_eval), so a request gives
# the same lines as running the driver with the same MB_* variables.

HOST_FLAGS = ["-DMB_HOST"]
SO_FLAGS = ["-shared", "-Xcompiler", "-fPIC", *PTDS_FLAGS]
HOST_MARKER = "MB_HOST_READY"          # in a driver.cpp that has the host


def supports_host(problem_dir) -> bool:
    """Exported with the persistent host (older drivers are one-process only)."""
    return HOST_MARKER in (Path(problem_dir) / "driver.cpp").read_text()


def _nvcc(args, timeout):
    from . import kb_trace
    cp = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    kb_trace.get().tool("nvcc", list(args), cp.returncode, cp.stdout + cp.stderr)
    return cp.returncode, (cp.stdout + cp.stderr)[-4000:]


def build_host(problem_dir, *, arch: str = "75", work: Path | None = None, nvcc: str = "nvcc",
               timeout: int = 1800) -> tuple[Path | None, str]:
    """driver.cpp as the persistent host -> (binary or None, log)."""
    problem_dir = Path(problem_dir)
    work = Path(work or problem_dir / "build")
    work.mkdir(parents=True, exist_ok=True)
    exe = work / "host"
    rc, log = _nvcc([nvcc, *NVCC_FLAGS, f"-arch=sm_{arch}", *HOST_FLAGS, *cupti_flags(nvcc), "-o", str(exe),
                     str(problem_dir / "driver.cpp"), "-ldl"], timeout)
    return (exe if rc == 0 else None), log


def build_so(problem_dir, *, arch: str = "75", sources: dict[str, str] | None = None, work: Path | None = None,
             name: str = "kernels.so", nvcc: str = "nvcc", timeout: int = 1800) -> tuple[Path | None, str]:
    """The assembled kernels as the host's shared object -> (path or None, log)."""
    problem_dir = Path(problem_dir)
    work = Path(work or problem_dir / "build")
    work.mkdir(parents=True, exist_ok=True)
    cu = work / (Path(name).stem + ".cu")
    cu.write_text(assemble(problem_dir, sources))
    so = work / name
    rc, log = _nvcc([nvcc, *NVCC_FLAGS, f"-arch=sm_{arch}", *SO_FLAGS, "-o", str(so), str(cu)], timeout)
    return (so if rc == 0 else None), log


class DriverHost:
    """A running persistent host: ``evaluate(so)`` sends one request and parses
    the reply. A host that crashed, hung past ``timeout`` or reported a
    poisoned context is gone afterwards (``alive`` False); the result says so
    (``stage="run"``, ``host_crash``) and the next ``evaluate`` starts a new one."""

    def __init__(self, exe, env: dict | None = None, timeout: int = 1800):
        import os

        self.exe, self.timeout = Path(exe), timeout
        self.env = {**os.environ, **{k: str(v) for k, v in (env or {}).items()}}
        self.proc = None
        self._buf = b""
        self.starts = 0
        self.ready_line = ""

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _readline(self, deadline: float) -> str | None:
        """The next output line; "" at EOF, None at the deadline (raw fd
        reads: a buffered reader would hide lines from select)."""
        import os
        import select
        import time

        while b"\n" not in self._buf:
            left = deadline - time.time()
            if left <= 0:
                return None
            r, _, _ = select.select([self.proc.stdout], [], [], left)
            if r:
                chunk = os.read(self.proc.stdout.fileno(), 1 << 16)
                if not chunk:
                    line, self._buf = self._buf, b""
                    return line.decode(errors="replace")   # "" when nothing was pending
                self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        return line.decode(errors="replace") + "\n"

    def start(self):
        import time

        self.close()
        self.proc = subprocess.Popen([str(self.exe)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, env=self.env)
        self._buf = b""
        self.starts += 1
        out, deadline = [], time.time() + self.timeout
        while True:
            line = self._readline(deadline)
            if not line:
                self.kill()
                raise RuntimeError("host did not start: " + "".join(out)[-2000:])
            out.append(line)
            if line.startswith(HOST_MARKER):
                self.ready_line = line.strip()
                return

    def evaluate(self, so, ref=None, env: dict | None = None, timeout: int | None = None) -> dict:
        import time
        import uuid

        if not self.alive:
            self.start()
        nonce = uuid.uuid4().hex[:12]
        fields = ["EVAL", nonce, f"so={so}"] + ([f"ref={ref}"] if ref else []) + \
                 [f"{k}={v}" for k, v in (env or {}).items()]
        t0 = time.time()
        try:
            self.proc.stdin.write(("\t".join(fields) + "\n").encode())
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass                                           # the read below sees the EOF
        out, deadline, done = [], t0 + (timeout or self.timeout), None
        while True:
            line = self._readline(deadline)
            if not line:                                   # EOF (crash) or timeout
                if line is None:
                    why = "timed out"
                else:
                    try:
                        why = f"exited (rc {self.proc.wait(timeout=10)})"
                    except subprocess.TimeoutExpired:
                        why = "closed its output"
                self.kill()
                text = "".join(out)
                res = parse_driver_output(text)
                res.update(passed=False, stage="run", host_crash=why, log=text[-4000:],
                           errors=res["errors"] + [f"candidate run {why} in the persistent host"])
                return res
            if line.startswith(f"MB_HOST_DONE {nonce} "):
                done = line
                break
            out.append(line)
        text = "".join(out)
        res = parse_driver_output(text)
        res.update(stage="run", log=text[-4000:], host_crash=None, wall_s=time.time() - t0,
                   poisoned="poisoned=1" in done, recycle="recycle=1" in done)
        if res["poisoned"] or res["recycle"]:
            self.proc.wait(timeout=60)
            self.proc = None
        if res["poisoned"]:
            res["passed"] = False
            res["errors"] = res["errors"] + ["the candidate left the CUDA context unusable"]
        return res

    def kill(self):
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait()
            self.proc = None

    def close(self):
        if self.alive:
            try:
                self.proc.stdin.write(b"QUIT\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=30)
            except Exception:
                pass
        self.kill()


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
    ap.add_argument("--runtime", default="process", choices=["process", "host"],
                    help="with --build-check: one-process driver, or the persistent host + kernels .so")
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
                r = build_and_run(d, arch=a.arch, runtime=a.runtime)
            line.update({k: r.get(k) for k in ("passed", "stage", "partition_us", "slot_us", "slots", "errors",
                                               *(("e2e_us", "e2e_basis", "graph_skip") if a.runtime == "host" else ()))})
            if not r.get("passed"):
                line["log"] = r.get("log", "")[-1500:]
        print(json.dumps(line), flush=True)


if __name__ == "__main__":
    main()
