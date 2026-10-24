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
and prints MB_CHECK / MB_SLOT_CHECK / MB_SLOT_US / MB_PARTITION_US lines
followed by "passed" or "failed". With ``MB_MODE=profile`` it instead runs
every op once (for ncu) and prints MB_PROFILE_DONE.

    python -m modelblaster.pipeline.kb_export IR_DIR OUT_DIR [--ops a,b|--range i:j]
        [--build-check --arch 75]
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

from .kernel_abi import bind, parse_signature
from .op_classes import op_class

MB_ROOT = Path(__file__).resolve().parents[1]
HARNESS = MB_ROOT / "gpu_eval" / "harness"
DEFAULT_HAND_CU = HARNESS / "cuda" / "mb_cuda_kernels.cu"
FORMAT = "mb-kb-problem/1"
NS_PREFIX = "mbslot_"
ALIAS_OPS = ("view", "chunk2_c1")
DEFAULT_TOL = {"output_rel": 1e-3, "output_abs": 1e-5, "slot_rel": 1e-3}


def _harness():
    """The torch IR walker (reference values) lives with the GPU harness."""
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    import mbir
    import walk_torch
    return mbir, walk_torch


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

def wrap_slot(slot_id: str, code: str) -> str:
    # C linkage would make two slots' entries the same symbol across namespaces
    code = code.replace('extern "C"', "")
    includes = [l for l in code.splitlines() if l.strip().startswith("#include")]
    body = "\n".join(l for l in code.splitlines() if not l.strip().startswith("#include"))
    defines = sorted(set(re.findall(r"^\s*#\s*define\s+([A-Za-z_]\w*)", body, re.M)))
    undefs = "".join(f"#undef {d}\n" for d in defines)
    return ("\n".join(includes) + "\n" if includes else "") + (
        f"// ==== slot {slot_id} ====\nnamespace {NS_PREFIX}{slot_id} {{\n{body}\n}}  // namespace {NS_PREFIX}{slot_id}\n{undefs}")


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


def export_problem(ir_dir, out_dir, *, op_indices: list[int] | None = None, name: str | None = None,
                   hand_cu: Path = DEFAULT_HAND_CU, slot_sources: dict[str, str] | None = None,
                   slot_source_tags: dict[str, str] | None = None, specs: dict | None = None,
                   tolerance: dict | None = None, strategy: str = "custom", target: str = "cuda") -> Path:
    """Write a problem directory for the ops ``op_indices`` of the IR at ``ir_dir``.

    ``slot_sources`` maps an op kind to a seed source replacing the hand
    kernel (e.g. a cached KernelBlaster winner); ``slot_source_tags`` records
    where each came from in the manifest. ``target="opencl"`` writes OpenCL
    slots, runtime and driver (``kb_export_opencl``); the partition, buffers,
    ops and data are the same.
    """
    import torch

    if specs is None:
        from .reference_kernels import KERNEL_SPECS
        specs = KERNEL_SPECS
    tgt = _target(target)                        # None: CUDA (this module)
    mbir, walk_torch = _harness()
    ir_dir, out = Path(ir_dir).resolve(), Path(out_dir).resolve()
    ir = mbir.load_ir(ir_dir)
    g = ir.graph
    ops = g["ops"]
    sel = sorted(op_indices) if op_indices is not None else list(range(len(ops)))
    sel_set = set(sel)
    tensors = g["tensors"]

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
            const_key = tensors.get(t, {}).get("const")
            is_int = bool(const_key) and ir.weights[const_key].dtype.kind in "iub"
            dtype = "f32" if tgt is None else tgt.buffer_dtype(tensors.get(t, {}))
            add_buf(t, kind=kind_of(t), dtype="i32" if is_int else dtype,
                    numel=int(np.prod(ir.tshape(t))), alias=None)
    for t in order:
        if t in alias_of:
            base, off = alias_of[t]
            add_buf(t, kind=kind_of(t), dtype="f32", numel=int(np.prod(ir.tshape(t))),
                    alias={"base": index[base], "offset": off})

    # ---- slots + per-op calls
    slots, slot_of, calls, op_recs = {}, {}, [], []
    lib = CudaLibrary(Path(hand_cu).read_text()) if tgt is None else \
        tgt.library(None if Path(hand_cu) == DEFAULT_HAND_CU else hand_cu)
    for i in sel:
        o = ops[i]
        if o["op"] in ALIAS_OPS:
            continue
        kind = o["op"]
        if kind not in specs:
            raise KeyError(f"{kind}: no KernelSpec")
        spec = specs[kind]
        entry, params = parse_signature(spec.signature)
        # one slot per C entry point: op kinds that share an entry (sum_dim /
        # max_dim -> kernel_reduce with a different opc) share its code
        sid = _slot_id(entry[len("kernel_"):] if entry.startswith("kernel_") else entry)
        if sid not in slots:
            key = next((x for x in (sid, kind) if slot_sources and x in slot_sources), None)
            if key is not None:
                src, tag = slot_sources[key], (slot_source_tags or {}).get(key, "override")
            elif tgt is not None:
                src, tag = tgt.seed(lib, entry, spec)
            elif hasattr(spec, "cuda_seed"):           # fused ops: sub-op kernels back to back
                src, tag = spec.cuda_seed(lib), "fused-seed"
            else:
                src, tag = lib.slot_source(entry), "hand"
            slots[sid] = {"id": sid, "op": kind, "ops": [], "op_class": op_class(kind), "entry": entry,
                          "signature": " ".join(spec.signature.split()),
                          "file": f"slots/{sid}{'.cu' if tgt is None else tgt.slot_ext}",
                          "reference_file": f"slots/{sid}.ref.c", "semantics": spec.semantics,
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
                    add_buf(key, kind="weight", dtype="i32" if arr.dtype.kind in "iub" else "f32",
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
        arr = arr.astype({"i32": np.int32, "i64": np.int64}.get(b["dtype"], np.float32)).reshape(-1)
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
    tol = {**DEFAULT_TOL, **(tolerance or {})}
    slot_list = list(slots.values())
    (out / driver_name).write_text((_driver_source if tgt is None else tgt.driver_source)(
        out / "data.bin", bufs, op_recs, slot_list, tol))
    man = {
        "format": FORMAT,
        "name": name or f"{ir.name}__{ops[sel[0]]['name']}__{ops[sel[-1]]['name']}",
        "source_ir": str(ir_dir), "ir_name": ir.name,
        "partition": {"strategy": strategy, "op_names": [ops[i]["name"] for i in sel],
                      "op_indices": sel, "is_whole_graph": len(sel) == len(ops)},
        "target": target, "dtype": "fp32", "tolerance": tol,
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
        arr("B_INT", "int", (int(b["dtype"] == "i32") for b in bufs)),
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
        .replace("@SLOT_REL@", repr(tol["slot_rel"]))


# data.bin is linked into the driver (.incbin) so the binary is self-contained
# (KernelBlaster's exec server only receives the executable); past ~1 GiB the
# relocations overflow, so big partitions read it at start-up instead
INCBIN_LIMIT = 1 << 30


def _blob_source(data_path: Path) -> str:
    if Path(data_path).stat().st_size < INCBIN_LIMIT:
        return ('__asm__(".section .rodata\\n.balign 256\\n.global mb_blob\\nmb_blob:\\n'
                f'.incbin \\"{data_path}\\"\\n.previous\\n");\n'
                'extern "C" const unsigned char mb_blob[];')
    return ("static const unsigned char *mb_blob = [] {\n"
            f'    const char *path = getenv("MB_DATA_PATH") ? getenv("MB_DATA_PATH") : "{data_path}";\n'
            "    FILE *f = fopen(path, \"rb\");\n"
            "    if (!f) { printf(\"MB_ERROR cannot open %s\\n\", path); printf(\"failed\\n\"); exit(1); }\n"
            "    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);\n"
            "    unsigned char *p = (unsigned char *)malloc(n);\n"
            "    if (fread(p, 1, n, f) != (size_t)n) { printf(\"MB_ERROR short read\\n\"); exit(1); }\n"
            "    fclose(f); return (const unsigned char *)p;\n}();")


_DRIVER_TEMPLATE = r'''// Generated by modelblaster.pipeline.kb_export -- checks and times one partition.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

void launch_gpu_implementation(void **bufs);
void mb_run_op(int i, void **bufs);
int mb_num_ops();

@BLOB@

@TABLES@
static const double OUT_REL = @OUT_REL@, OUT_ABS = @OUT_ABS@, SLOT_REL = @SLOT_REL@;
enum { K_INPUT, K_CONST, K_WEIGHT, K_INTER, K_OUTPUT };

static void *dev[NB];

static bool ok_cuda(const char *what) {
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("MB_ERROR %s: %s\n", what, cudaGetErrorString(e)); return false; }
    return true;
}
static void upload(int b) {   // reference values into a non-alias buffer
    if (B_REF[b] >= 0) cudaMemcpy(dev[b], mb_blob + B_REF[b], B_NUMEL[b] * 4, cudaMemcpyHostToDevice);
}
static const float *ref_of(int b) { return B_REF[b] >= 0 ? (const float *)(mb_blob + B_REF[b]) : nullptr; }
// reference for an alias buffer = slice of its base's reference
static std::vector<float> ref_vec(int b) {
    std::vector<float> v(B_NUMEL[b]);
    if (B_BASE[b] >= 0) memcpy(v.data(), ref_of(B_BASE[b]) + B_AOFF[b], B_NUMEL[b] * 4);
    else memcpy(v.data(), ref_of(b), B_NUMEL[b] * 4);
    return v;
}
static std::vector<float> dev_vec(int b) {
    std::vector<float> v(B_NUMEL[b]);
    cudaMemcpy(v.data(), dev[b], B_NUMEL[b] * 4, cudaMemcpyDeviceToHost);
    return v;
}
static double max_finite(const std::vector<float> &v) {
    double m = 0; for (float x : v) if (std::isfinite(x)) m = std::max(m, (double)std::fabs(x)); return m;
}
static double max_err(const std::vector<float> &a, const std::vector<float> &r) {
    double e = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        float x = a[i], y = r[i];
        if (x == y || (std::isnan(x) && std::isnan(y))) continue;
        if (!std::isfinite(x) || !std::isfinite(y)) return INFINITY;
        e = std::max(e, (double)std::fabs(x - y));
    }
    return e;
}
static void upload_inputs() {
    for (int b = 0; b < NB; ++b)
        if (B_BASE[b] < 0 && (B_KIND[b] == K_INPUT || B_KIND[b] == K_CONST || B_KIND[b] == K_WEIGHT)) upload(b);
}

int main() {
    bool all_ok = true;
    for (int b = 0; b < NB; ++b)
        if (B_BASE[b] < 0) {
            if (cudaMalloc(&dev[b], std::max(B_NUMEL[b], 1L) * 4) != cudaSuccess) { printf("MB_ERROR alloc %s\n", B_NAME[b]); printf("failed\n"); return 1; }
            cudaMemset(dev[b], 0, std::max(B_NUMEL[b], 1L) * 4);
        }
    for (int b = 0; b < NB; ++b)
        if (B_BASE[b] >= 0) dev[b] = (float *)dev[B_BASE[b]] + B_AOFF[b];
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

    // 1. whole partition vs reference
    upload_inputs();
    launch_gpu_implementation(dev);
    if (!ok_cuda("partition")) { printf("failed\n"); return 1; }
    for (int b = 0; b < NB; ++b) {
        if (B_KIND[b] != K_OUTPUT) continue;
        std::vector<float> r = ref_vec(b), a = dev_vec(b);
        double e = max_err(a, r), bound = OUT_REL * max_finite(r) + OUT_ABS;
        bool ok = e <= bound;
        all_ok &= ok;
        printf("MB_CHECK output %s max_abs_err=%.6g bound=%.6g ok=%d\n", B_NAME[b], e, bound, (int)ok);
    }

    // 2. every op alone on the reference's inputs
    std::vector<double> slot_err(NSLOTS, 0.0);
    std::vector<int> slot_bad(NSLOTS, 0);
    for (int i = 0; i < NOPS; ++i) {
        upload_inputs();
        for (int k = OP_IN_OFS[i]; k < OP_IN_OFS[i + 1]; ++k) {
            int b = OP_IN[k];
            if (B_KIND[b] == K_CONST || B_KIND[b] == K_WEIGHT) continue;
            std::vector<float> r = ref_vec(b);
            cudaMemcpy(dev[b], r.data(), B_NUMEL[b] * 4, cudaMemcpyHostToDevice);
        }
        mb_run_op(i, dev);
        if (!ok_cuda(OP_NAME[i])) { slot_bad[OP_SLOT[i]] = 1; slot_err[OP_SLOT[i]] = INFINITY; continue; }
        double scale = 0;
        for (int k = OP_IN_OFS[i]; k < OP_IN_OFS[i + 1]; ++k)
            if (!B_INT[OP_IN[k]]) scale = std::max(scale, max_finite(ref_vec(OP_IN[k])));
        for (int k = OP_OUT_OFS[i]; k < OP_OUT_OFS[i + 1]; ++k) {
            int b = OP_OUT[k];
            std::vector<float> r = ref_vec(b), a = dev_vec(b);
            double s = std::max(scale, max_finite(r)), e = max_err(a, r) / std::max(s, 1e-12);
            slot_err[OP_SLOT[i]] = std::max(slot_err[OP_SLOT[i]], e);
            if (!(e <= SLOT_REL)) slot_bad[OP_SLOT[i]] = 1;
        }
    }
    for (int s = 0; s < NSLOTS; ++s) {
        all_ok &= !slot_bad[s];
        printf("MB_SLOT_CHECK %s rel_err=%.6g ok=%d\n", SLOT_NAME[s], slot_err[s], (int)!slot_bad[s]);
    }

    // 3. timing (CUDA events): whole partition, then every op
    if (all_ok) {
        upload_inputs();
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        for (int w = 0; w < 2; ++w) launch_gpu_implementation(dev);
        cudaDeviceSynchronize();
        int reps = 0; float total = 0;
        const char *env = getenv("MB_TIME_MS");
        float budget = env ? atof(env) : 200.0f;
        while (reps < 3 || (total < budget && reps < 200)) {
            float ms; cudaEventRecord(e0); launch_gpu_implementation(dev); cudaEventRecord(e1);
            cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); total += ms; ++reps;
        }
        printf("MB_PARTITION_US %.3f reps=%d\n", 1000.0 * total / reps, reps);
        std::vector<double> slot_us(NSLOTS, 0.0), op_us(NOPS, 0.0);
        int oreps = std::max(1, std::min(reps, 20));
        std::vector<cudaEvent_t> ev(2 * NOPS);
        for (auto &e : ev) cudaEventCreate(&e);
        for (int r = 0; r < oreps; ++r) {
            for (int i = 0; i < NOPS; ++i) { cudaEventRecord(ev[2 * i]); mb_run_op(i, dev); cudaEventRecord(ev[2 * i + 1]); }
            cudaEventSynchronize(ev[2 * NOPS - 1]);
            for (int i = 0; i < NOPS; ++i) {
                float ms; cudaEventElapsedTime(&ms, ev[2 * i], ev[2 * i + 1]);
                slot_us[OP_SLOT[i]] += 1000.0 * ms / oreps; op_us[i] += 1000.0 * ms / oreps;
            }
        }
        for (int s = 0; s < NSLOTS; ++s) printf("MB_SLOT_US %s %.3f\n", SLOT_NAME[s], slot_us[s]);
        for (int i = 0; i < NOPS; ++i) printf("MB_OP_US %s %.3f\n", OP_NAME[i], op_us[i]);
        all_ok &= ok_cuda("timing");
    }
    printf(all_ok ? "passed\n" : "failed\n");
    return all_ok ? 0 : 1;
}
'''


# ---------------------------------------------------------------- local build/run (MB-side checks)

def parse_driver_output(text: str) -> dict:
    res = {"passed": bool(re.search(r"^passed$", text, re.M)), "outputs": {}, "slots": {},
           "slot_us": {}, "partition_us": None, "errors": re.findall(r"^MB_ERROR (.*)$", text, re.M)}
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
    return res


def build_and_run(problem_dir, *, arch: str = "75", sources: dict[str, str] | None = None,
                  work: Path | None = None, nvcc: str = "nvcc", timeout: int = 1800) -> dict:
    """Assemble + nvcc-build + run the problem outside KernelBlaster."""
    problem_dir = Path(problem_dir)
    work = Path(work or problem_dir / "build")
    work.mkdir(parents=True, exist_ok=True)
    (work / "kernels.cu").write_text(assemble(problem_dir, sources))
    exe = work / "problem"
    cp = subprocess.run([nvcc, "-O3", f"-arch=sm_{arch}", "-std=c++17", "-o", str(exe),
                         str(work / "kernels.cu"), str(problem_dir / "driver.cpp")],
                        capture_output=True, text=True, timeout=timeout)
    if cp.returncode != 0:
        return {"passed": False, "stage": "compile", "log": (cp.stdout + cp.stderr)[-4000:]}
    rp = subprocess.run([str(exe)], capture_output=True, text=True, timeout=timeout)
    res = parse_driver_output(rp.stdout)
    res.update(stage="run", log=(rp.stdout + rp.stderr)[-4000:])
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ir_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--ops", help="comma-separated op names")
    ap.add_argument("--range", help="op index range i:j")
    ap.add_argument("--strategy", default=None, choices=["whole", "per_op", "window"],
                    help="export every partition of the strategy under OUT_DIR/pNN")
    ap.add_argument("--max-ops", type=int, default=12)
    ap.add_argument("--hand-cu", default=str(DEFAULT_HAND_CU))
    ap.add_argument("--fusion-registry", help="apply the fusions accepted in this registry first")
    ap.add_argument("--build-check", action="store_true", help="build and run each problem (nvcc / g++ + OpenCL)")
    ap.add_argument("--arch", default="75")
    ap.add_argument("--target", default="cuda", choices=["cuda", "opencl"])
    a = ap.parse_args()
    ir_dir, specs = Path(a.ir_dir), None
    g = json.loads((ir_dir / "graph.json").read_text())
    if a.fusion_registry:
        from .kb_proposals import apply_accepted, write_variant_ir
        from .reference_kernels import KERNEL_SPECS
        g, fused, done = apply_accepted(g, Path(a.fusion_registry), KERNEL_SPECS)
        if done:
            specs = {**KERNEL_SPECS, **fused}
            ir_dir = write_variant_ir(Path(a.ir_dir), g, Path(a.out_dir) / "fused_ir")
            print(json.dumps({"fused": done}), flush=True)
    if a.strategy:
        parts = [(f"p{k:02d}", p) for k, p in enumerate(partition_ir(g, a.strategy, a.max_ops))]
    elif a.ops:
        names = a.ops.split(",")
        parts = [("", [i for i, o in enumerate(g["ops"]) if o["name"] in names])]
    elif a.range:
        i, j = (int(x) for x in a.range.split(":"))
        parts = [("", list(range(i, j)))]
    else:
        parts = [("", None)]
    for sub, idx in parts:
        d = export_problem(ir_dir, Path(a.out_dir) / sub if sub else a.out_dir, op_indices=idx,
                           hand_cu=Path(a.hand_cu), strategy=a.strategy or "custom", specs=specs, target=a.target)
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
