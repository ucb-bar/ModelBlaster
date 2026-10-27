#!/usr/bin/env python3
"""Convert SOL-ExecBench problems (the original repo layout) to KernelBench-style
bench files for ModelBlaster's ``extract_graph --bench-file``.

Input: ``<bench>/<set>/<problem>/{definition.json, workload.jsonl}`` from
NVIDIA SOL-ExecBench (sets L1, L2, Quant, FlashInfer-Bench). ``definition.json``
names the axes (const / var / expr), the inputs and outputs (shape as axis
names, dtype) and the reference (``run(...)``, plus an optional custom input
generator). ``workload.jsonl`` binds the var axes and says how every input is
made (random / scalar / custom / safetensors), with a per-workload tolerance.

Output, per selected workload: ``<out>/src/<dtype>/<stem>.py``, a self-contained
bench file (torch only), plus ``<out>/manifest.<dtype>.json``.

Rules (docs/sol_execbench.md):

* **Native shapes.** The workload's axes are used as given. They live in the
  ``SOL_AXES`` dict, not as module-level ints, so ``extract_graph``'s and
  ``torch_baseline``'s shrinkers find nothing to resize. A workload whose io
  (inputs + outputs at the variant dtype) is over ``--cap-mb`` is skipped with
  that reason; it is never shrunk.
* **Native inputs.** ``get_inputs()`` reproduces SOL's ``gen_inputs``: seed
  (``SOL_SEED``, SOL's default 200), the custom entrypoint first, then every
  random input in definition order with SOL's name heuristics (norm weights 1,
  causal masks, softmaxed weights, rope cos/sin, ...). It runs the *original*
  reference source on the CPU, so the values are the same for every dtype
  variant. Floating tensors are then cast to the variant dtype.
* **Model.** ``forward`` is the reference ``run`` (renamed ``_sol_run``). For a
  non-native variant the reference's low-precision dtypes are rewritten to the
  variant's (``torch.bfloat16`` -> ``torch.float32``; ``.bfloat16()`` ->
  ``.float()``); ``bf16`` keeps the reference byte-for-byte. A dict return is
  packed in definition output order.
* **Refusals** are named: unsupported dtypes (fp8 / fp4 / int8 storage),
  safetensors inputs without a blob root, no tensor inputs.

Usage:
    sol_upstream.py convert BENCH OUT [--sets L1,L2] [--dtype fp32|bf16|fp16]
        [--cap-mb 256] [--pick smallest,median,largest] [--problems REGEX]
        [--blob-root DIR]
    sol_upstream.py sizes BENCH [--sets ...]        # workload io sizes (MiB)
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from pathlib import Path

SETS = ("L1", "L2", "Quant", "FlashInfer-Bench")
SET_TAG = {"L1": "L1", "L2": "L2", "Quant": "Q", "FlashInfer-Bench": "FIB"}
SOL_SEED = 200                       # sol_execbench BenchmarkConfig.seed
DEFAULT_TOL = {"max_atol": 1e-2, "max_rtol": 1e-2, "required_matched_ratio": 0.99,
               "max_error_cap": None, "allow_negative_inf": False}

# variant -> the torch dtype the definition's float dtypes become
VARIANTS = {
    "fp32": {"bfloat16": "float32", "float16": "float32", "float32": "float32", "float64": "float64"},
    "bf16": {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32", "float64": "float64"},
    "fp16": {"bfloat16": "float16", "float16": "float16", "float32": "float32", "float64": "float64"},
}
# AST rewrite of the reference's dtype tokens per variant (bf16 = native: none)
REWRITE = {
    "fp32": {"bfloat16": "float32", "float16": "float32", "half": "float32"},
    "bf16": {},
    "fp16": {"bfloat16": "float16"},
}
METHOD_REWRITE = {"fp32": {"bfloat16": "float", "half": "float"}, "bf16": {}, "fp16": {"bfloat16": "half"}}
ITEMSIZE = {"float64": 8, "float32": 4, "bfloat16": 2, "float16": 2, "int64": 8, "int32": 4,
            "int16": 2, "int8": 1, "uint8": 1, "bool": 1, "float8_e4m3fn": 1, "float8_e5m2": 1,
            "float4_e2m1": 0.5, "float4_e2m1fn_x2": 1}
# storage MB/KB cannot carry: refused with this reason
UNSUPPORTED_DTYPES = {
    "float8_e4m3fn": "fp8 (e4m3) storage", "float8_e5m2": "fp8 (e5m2) storage",
    "float4_e2m1": "fp4 (e2m1) storage", "float4_e2m1fn_x2": "packed fp4 (e2m1 x2) storage",
    "int8": "int8 tensor storage", "int16": "int16 tensor storage",
}


# ── axes ──────────────────────────────────────────────────────────────────────

_BIN = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b, ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a ** b}


def _eval_expr(expr: str, env: dict):
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.Name):
            return env[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            return _BIN[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        raise ValueError(f"unsupported axis expression {expr!r}")
    return ev(ast.parse(expr, mode="eval"))


def resolve_axes(defn: dict, wl_axes: dict) -> dict:
    """const + var (from the workload) + expr axes -> {name: int} (sol Definition semantics)."""
    out = {}
    for k, a in defn["axes"].items():
        if a["type"] == "const":
            out[k] = int(a["value"])
        elif a["type"] == "var":
            if k not in wl_axes:
                raise KeyError(f"workload has no value for var axis {k}")
            out[k] = int(wl_axes[k])
    for _ in range(len(defn["axes"])):
        for k, a in defn["axes"].items():
            if a["type"] == "expr" and k not in out:
                try:
                    out[k] = int(_eval_expr(a["expression"], out))
                except KeyError:
                    pass
    missing = [k for k in defn["axes"] if k not in out]
    if missing:
        raise ValueError(f"unresolved axes {missing}")
    return out


def shape_of(spec: dict, axes: dict):
    if spec["shape"] is None:
        return None
    # an entry is an axis name, or a literal size ("1" / 1)
    return [axes[a] if isinstance(a, str) and a in axes else int(a) for a in spec["shape"]]


def io_bytes(defn: dict, axes: dict, variant: str) -> int:
    """Bytes of every tensor input + output at the variant's dtypes."""
    tot = 0
    for sec in ("inputs", "outputs"):
        for spec in defn[sec].values():
            sh = shape_of(spec, axes)
            if sh is None:
                continue
            dt = VARIANTS[variant].get(spec["dtype"], spec["dtype"])
            tot += math.prod(sh) * ITEMSIZE.get(dt, 4)
    return int(tot)


def tolerance(wl: dict) -> dict:
    """The workload's ToleranceSpec (pydantic defaults; unknown keys such as L2's
    ``required_match_ratio`` are ignored by SOL and recorded here as such)."""
    raw = dict(wl.get("tolerance") or {})
    tol = dict(DEFAULT_TOL)
    ignored = {}
    for k, v in raw.items():
        if k in tol:
            tol[k] = v
        else:
            ignored[k] = v
    if ignored:
        tol["ignored_by_sol"] = ignored
    return tol


# ── problem loading / refusal ────────────────────────────────────────────────

def load_problem(pdir: Path) -> tuple[dict, list]:
    defn = json.loads((pdir / "definition.json").read_text())
    wls = [json.loads(line) for line in (pdir / "workload.jsonl").read_text().splitlines() if line.strip()]
    return defn, wls


def refusal(defn: dict, wl: dict, blob_root: Path | None) -> str | None:
    """Why this workload cannot be converted (None = convertible)."""
    bad = sorted({UNSUPPORTED_DTYPES[s["dtype"]] for sec in ("inputs", "outputs")
                  for s in defn[sec].values() if s["dtype"] in UNSUPPORTED_DTYPES})
    if bad:
        return "unsupported dtype: " + ", ".join(bad)
    if not any(s["shape"] is not None for s in defn["inputs"].values()):
        return "no tensor inputs (forward takes scalars only)"
    for name, spec in wl["inputs"].items():
        if spec.get("type") == "safetensors":
            if blob_root is None:
                return "safetensors inputs (no --blob-root)"
            if _blob_path(blob_root, spec["path"]) is None:
                return f"safetensors blob missing: {spec['path']}"
    return None


def _blob_path(root: Path, rel: str) -> Path | None:
    parts = Path(rel).parts
    for start in range(len(parts)):
        c = root / Path(*parts[start:])
        if c.exists():
            return c
    return None


def pick(sizes: list, picks: list[str]) -> dict:
    """sizes: [(bytes, index)] of the fitting workloads -> {pick_name: index}."""
    order = sorted(sizes)
    if not order:
        return {}
    named = {"smallest": order[0][1], "median": order[(len(order) - 1) // 2][1], "largest": order[-1][1]}
    out, seen = {}, set()
    for p in picks:
        if p.startswith("index"):
            i = int(p[5:])
            if any(ix == i for _, ix in order):
                named[p] = i
        if p in named and named[p] not in seen:
            out[p] = named[p]
            seen.add(named[p])
    return out


# ── reference rewriting ──────────────────────────────────────────────────────

class _Rename(ast.NodeTransformer):
    def __init__(self, names: dict):
        self.names = names

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.name in self.names:
            node.name = self.names[node.name]
        return node

    def visit_Name(self, node):
        if node.id in self.names:
            node.id = self.names[node.id]
        return node


class _Dtype(ast.NodeTransformer):
    def __init__(self, variant: str):
        self.attr = REWRITE[variant]
        self.meth = METHOD_REWRITE[variant]
        self.count = 0

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "torch" and node.attr in self.attr:
            self.count += 1
            return ast.copy_location(ast.Attribute(node.value, self.attr[node.attr], node.ctx), node)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in self.meth and not node.args:
            self.count += 1
            f.attr = self.meth[f.attr]
        return node

    def visit_Constant(self, node):
        # a hard-coded CUDA device in run(): the bench is traced on the CPU
        if isinstance(node.value, str) and re.fullmatch(r"cuda(:\d+)?", node.value):
            self.count += 1
            return ast.copy_location(ast.Constant("cpu"), node)
        return node


def model_source(reference: str, variant: str, entry: str | None) -> tuple[str, int]:
    """The reference as module code: run -> _sol_run, the custom entrypoint -> _sol_custom_inputs,
    dtype tokens rewritten for the variant -> (source, rewrites)."""
    tree = ast.parse(reference)
    names = {"run": "_sol_run"}
    if entry:
        names[entry] = "_sol_custom_inputs"
    for n in ("get_inputs", "get_init_inputs", "Model"):
        names.setdefault(n, f"_sol_ref_{n}")
    tree = _Rename(names).visit(tree)
    dt = _Dtype(variant)
    if REWRITE[variant] or METHOD_REWRITE[variant]:
        tree = dt.visit(tree)
    else:
        tree = _DeviceOnly().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), dt.count


class _DeviceOnly(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str) and re.fullmatch(r"cuda(:\d+)?", node.value):
            return ast.copy_location(ast.Constant("cpu"), node)
        return node


# ── bench file ───────────────────────────────────────────────────────────────

# SOL-ExecBench's input generation (sol_execbench/core/bench/io.py @834c7c1,
# Apache-2.0, NVIDIA), reduced to the CPU and the dtypes this port accepts.
_GEN = r'''
_ORIG = {n: getattr(torch, n) for n in ("rand", "randn", "zeros", "ones", "empty", "full", "randint",
                                        "arange", "tensor", "eye", "linspace", "randperm", "normal",
                                        "rand_like", "randn_like", "zeros_like", "ones_like",
                                        "empty_like", "full_like")}


class _cpu_creation:
    """Every creation op on the CPU, with its own dtype (undoes extract_graph's fp16->fp32 patch, so
    the values do not depend on who calls get_inputs)."""

    def __enter__(self):
        self.saved = {n: getattr(torch, n) for n in _ORIG}
        for n, fn in _ORIG.items():
            def wrap(fn=fn):
                def inner(*a, **k):
                    if k.get("device") is not None:
                        k["device"] = "cpu"
                    return fn(*a, **k)
                return inner
            setattr(torch, n, wrap())

    def __exit__(self, *exc):
        for n, fn in self.saved.items():
            setattr(torch, n, fn)


def _rand_tensor(shape, dtype):
    if dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
        return torch.randn(shape, dtype=dtype)
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool)
    if dtype in (torch.int32, torch.int64):
        return torch.randint(-1024, 1024, shape, dtype=dtype)
    raise ValueError(f"unsupported random dtype {dtype}")


def _heuristic(name, shape, dtype, desc):
    if not dtype.is_floating_point:
        return None
    d = (desc or "").lower()
    def norm_like(suffix):
        if name == "norm" + suffix:
            return True
        if name.endswith(suffix):
            p = name[: -len(suffix)]
            if p.endswith(("_norm", "_layernorm", "layernorm")):
                return True
            s = p.rstrip("0123456789")
            return bool(s) and s.endswith(("norm", "layernorm"))
        return False
    if norm_like("_weight"):
        return torch.ones(shape, dtype=dtype)
    if norm_like("_bias"):
        return torch.zeros(shape, dtype=dtype)
    if (len(shape) >= 2 and shape[-1] == shape[-2]
            and (name in ("attention_mask", "causal_mask") or ("attention mask" in d and "causal" in d))):
        n = shape[-1]
        m = torch.zeros(shape, dtype=dtype)
        m[..., torch.triu(torch.ones(n, n), diagonal=1).bool()] = torch.finfo(dtype).min
        return m
    if name in ("x_mask", "text_mask", "aspect_ratio_mask", "drop_mask", "attention_mask") or (
            name.endswith("_mask") and desc and any(
                k in d for k in ("binary", "{0, 1}", "0 or 1", "1.0 for valid", "0.0 for masked"))):
        return torch.randint(0, 2, shape).to(dtype)
    if name in ("cos", "sin", "cos_cached", "sin_cached", "rope_cos", "rope_sin"):
        t = torch.randn(shape, dtype=torch.float32).clamp_(-math.pi, math.pi)
        return (torch.cos(t) if name in ("cos", "cos_cached", "rope_cos") else torch.sin(t)).to(dtype)
    base = name.rstrip("0123456789")
    if (name in ("rstd", "std", "variance", "var") or name.endswith(("_rstd", "_std", "_variance", "_var"))
            or base in ("rstd", "std", "variance", "var") or base.endswith(("_rstd", "_std", "_var"))):
        return torch.randn(shape, dtype=dtype).abs() + 0.1
    if name in ("A", "A_log", "A_cumsum", "g"):
        if name == "A_cumsum":
            return torch.empty(shape, dtype=dtype).uniform_(-0.1, 0.0).cumsum(dim=-1)
        lo, hi = {"A_log": (-5.0, -1.0), "A": (-1.0, -0.001), "g": (-5.0, 0.0)}[name]
        return torch.empty(shape, dtype=dtype).uniform_(lo, hi)
    if name in ("attn_weights", "attention_weights", "routing_weights") or ("softmax" in d and "output" in d):
        return torch.softmax(torch.randn(shape, dtype=torch.float32), dim=-1).to(dtype)
    sfx = ("_weight", "_weights", "_proj", "_projs", "_proj_weight", "_proj_weights", "_weight_matrix")
    if len(shape) >= 2 and (name.endswith(sfx) or name == "weight" or base == "weight" or base.endswith(sfx)):
        return torch.randn(shape, dtype=dtype) / math.sqrt(shape[-1])
    return None


def _native_inputs():
    """SOL's gen_inputs for this workload on the CPU, at the definition's own dtypes."""
    torch.manual_seed(SOL_SEED)
    out = []
    with _cpu_creation(), torch.no_grad():
        ns = {}
        exec(compile(SOL_REFERENCE, "sol_reference", "exec"), ns)
        custom = None
        if any(s["type"] == "custom" for s in SOL_WORKLOAD_INPUTS.values()):
            custom = ns[SOL_CUSTOM_ENTRY]({**SOL_AXES, **SOL_SCALARS}, torch.device("cpu"))
        for name, spec in SOL_INPUTS.items():
            kind = SOL_WORKLOAD_INPUTS.get(name, {"type": "random"})
            dtype = getattr(torch, spec["dtype"])
            shape = spec["shape_v"]
            if kind["type"] == "scalar":
                v = kind["value"]
            elif kind["type"] == "custom":
                v = custom[name]
            elif kind["type"] == "safetensors":
                v = _load_blob(kind)
            elif shape is None:
                v = _rand_tensor((), dtype).item()
            else:
                v = _heuristic(name, tuple(shape), dtype, spec.get("description"))
                if v is None:
                    v = _rand_tensor(shape, dtype)
                if SOL_OP_TYPE == "sampling" and name == "probs":
                    v = torch.softmax(v, dim=-1)
            if torch.is_tensor(v):
                v = v.to("cpu")
            out.append(v)
    return out


_ST_DTYPES = {"F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16", "I64": "int64",
              "I32": "int32", "I16": "int16", "I8": "int8", "U8": "uint8", "BOOL": "bool"}


def _load_blob(kind):
    """One tensor of a .safetensors file (8-byte header length, JSON header, raw little-endian data)."""
    import json
    import struct
    with open(SOL_BLOBS[kind["path"]], "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))[kind["tensor_key"]]
        lo, hi = meta["data_offsets"]
        f.seek(8 + n + lo)
        raw = bytearray(f.read(hi - lo))
    dt = getattr(torch, _ST_DTYPES[meta["dtype"]])
    return torch.frombuffer(raw, dtype=dt).reshape(meta["shape"]).clone() if raw else \
        torch.empty(meta["shape"], dtype=dt)


def get_inputs():
    """The workload's inputs in definition order; floating tensors in the variant's dtypes."""
    out = []
    for (name, spec), v in zip(SOL_INPUTS.items(), _native_inputs()):
        if torch.is_tensor(v) and v.is_floating_point():
            v = v.to(getattr(torch, SOL_DTYPE_MAP.get(spec["dtype"], spec["dtype"])))
        out.append(v)
    return out


def get_init_inputs():
    return []


def sol_reference_outputs(inputs=None, device="cpu"):
    """The SOL ground truth: the original reference on the native inputs -> list of tensors in
    definition output order (``device``: where run() executes, SOL uses the GPU)."""
    ns = {}
    exec(compile(SOL_REFERENCE, "sol_reference", "exec"), ns)
    ins = _native_inputs() if inputs is None else inputs
    ins = [v.to(device) if torch.is_tensor(v) else v for v in ins]
    with torch.no_grad():
        return [t.cpu() if torch.is_tensor(t) else torch.tensor(t) for t in _pack(ns["run"](*ins))]
'''

_MODEL = r'''

def _pack(out):
    if isinstance(out, dict):
        return tuple(out[k] for k in SOL_OUTPUTS)
    if isinstance(out, (tuple, list)):
        return tuple(out)
    return (out,)


class Model(nn.Module):
    """SOL reference ``run`` as a module (no parameters: every SOL input is a forward input)."""

    def forward(self, __ARGS__):
        out = _sol_run(__ARGS__)
        return out if torch.is_tensor(out) else _pack(out)
'''


def bench_source(set_name: str, pname: str, defn: dict, wl: dict, idx: int, pick_name: str,
                 variant: str, blobs: dict) -> tuple[str, dict]:
    axes = resolve_axes(defn, wl["axes"])
    scalars = {k: v["value"] for k, v in wl["inputs"].items() if v.get("type") == "scalar"}
    inputs = {}
    for k, s in defn["inputs"].items():
        inputs[k] = {"dtype": s["dtype"], "shape_v": shape_of(s, axes), "shape": s["shape"]}
        if s.get("description"):
            inputs[k]["description"] = s["description"]
    entry = defn.get("custom_inputs_entrypoint")
    msrc, nrew = model_source(defn["reference"], variant, entry)
    head = (f'"""SOL-ExecBench {set_name}/{pname}, workload {wl["uuid"]} (#{idx}, {pick_name}), '
            f'variant {variant}.\n\nGenerated by modelblaster/gpu_eval/tools/sol_upstream.py; native axes, '
            f'never resized.\n"""\n'
            "import math\nimport torch\nimport torch.nn as nn\n\n")
    consts = (f"SOL_SET = {set_name!r}\nSOL_PROBLEM = {pname!r}\nSOL_WORKLOAD = {wl['uuid']!r}\n"
              f"SOL_WORKLOAD_INDEX = {idx}\nSOL_PICK = {pick_name!r}\nSOL_VARIANT = {variant!r}\n"
              f"SOL_SEED = {SOL_SEED}\nSOL_OP_TYPE = {defn.get('op_type')!r}\n"
              f"SOL_CUSTOM_ENTRY = {entry!r}\n"
              f"SOL_AXES = {json.dumps(axes)}\n"
              f"SOL_SCALARS = {json.dumps(scalars)}\n"
              f"SOL_INPUTS = {json.dumps(inputs)}\n"
              f"SOL_WORKLOAD_INPUTS = {json.dumps(wl['inputs'])}\n"
              f"SOL_OUTPUTS = {json.dumps({k: {'dtype': s['dtype'], 'shape_v': shape_of(s, axes)} for k, s in defn['outputs'].items()})}\n"
              f"SOL_TOLERANCE = {json.dumps(tolerance(wl))}\n"
              f"SOL_DTYPE_MAP = {json.dumps(VARIANTS[variant])}\n"
              f"SOL_BLOBS = {json.dumps(blobs)}\n"
              f"SOL_REFERENCE = {defn['reference']!r}\n")
    consts = _pyify(consts)
    # fx traces named parameters only (not *args): one per definition input, in order
    args = ", ".join(f"in{i}_" + re.sub(r"\W", "_", k) for i, k in enumerate(defn["inputs"]))
    src = head + consts + "\n# ── reference (run -> _sol_run" + (", dtypes rewritten" if nrew else "") + \
        ") ──\n" + msrc + "\n" + _MODEL.replace("__ARGS__", args) + _GEN
    meta = {"axes": axes, "scalars": scalars, "rewrites": nrew}
    return src, meta


def _pyify(text: str) -> str:
    """json.dumps literals -> Python (null/true/false) outside strings, line by line."""
    out = []
    for line in text.splitlines(keepends=True):
        if line.startswith("SOL_REFERENCE"):
            out.append(line)
            continue
        out.append(re.sub(r'("(?:[^"\\]|\\.)*")|\b(null|true|false)\b',
                          lambda m: m.group(1) or {"null": "None", "true": "True", "false": "False"}[m.group(2)],
                          line))
    return "".join(out)


def stem_for(set_name: str, pname: str, pick_name: str) -> str:
    short = {"smallest": "min", "median": "med", "largest": "max"}.get(pick_name, pick_name)
    return f"{SET_TAG[set_name]}_{pname}__{short}"


def convert(bench: Path, out: Path, sets, variant: str, cap_mb: float, picks, problems=None,
            blob_root: Path | None = None) -> dict:
    rows, problems_out = [], []
    rx = re.compile(problems) if problems else None
    for s in sets:
        for pdir in sorted((bench / s).iterdir()):
            if not (pdir / "definition.json").exists():
                continue
            if rx and not rx.search(f"{s}/{pdir.name}"):
                continue
            defn, wls = load_problem(pdir)
            prob = {"set": s, "problem": pdir.name, "n_workloads": len(wls), "op_type": defn.get("op_type"),
                    "dtypes": sorted({v["dtype"] for sec in ("inputs", "outputs") for v in defn[sec].values()}),
                    "selected": [], "skipped": []}
            fits = []
            for i, wl in enumerate(wls):
                why = refusal(defn, wl, blob_root)
                try:
                    axes = resolve_axes(defn, wl["axes"])
                    nb = io_bytes(defn, axes, variant)
                except Exception as e:  # noqa: BLE001
                    why, nb = why or f"axes: {e}", 0
                if why is None and nb > cap_mb * 2 ** 20:
                    why = f"io {nb / 2 ** 20:.0f} MiB > cap {cap_mb:g} MiB"
                if why:
                    prob["skipped"].append({"index": i, "uuid": wl["uuid"], "io_mib": round(nb / 2 ** 20, 2),
                                            "reason": why})
                else:
                    fits.append((nb, i))
            chosen = pick(fits, picks)
            if not chosen:
                reasons = sorted({k["reason"].split(":")[0] if not k["reason"].startswith("io ") else "over cap"
                                  for k in prob["skipped"]})
                prob["refusal"] = "; ".join(reasons)
            for pname_, i in chosen.items():
                wl = wls[i]
                blobs = {}
                for v in wl["inputs"].values():
                    if v.get("type") == "safetensors":
                        blobs[v["path"]] = str(_blob_path(blob_root, v["path"]))
                src, meta = bench_source(s, pdir.name, defn, wl, i, pname_, variant, blobs)
                stem = stem_for(s, pdir.name, pname_)
                f = out / "src" / variant / f"{stem}.py"
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(src)
                row = {"stem": stem, "file": str(f), "set": s, "problem": pdir.name, "pick": pname_,
                       "workload_index": i, "uuid": wl["uuid"], "axes": meta["axes"], "scalars": meta["scalars"],
                       "io_mib": round(dict((ix, b) for b, ix in fits)[i] / 2 ** 20, 3),
                       "variant": variant, "dtype_map": {k: VARIANTS[variant].get(k, k) for k in prob["dtypes"]},
                       "native_quant": ("bf16" if "bfloat16" in prob["dtypes"] else
                                        "fp16" if "float16" in prob["dtypes"] else "fp32"),
                       "dtype_rewrites": meta["rewrites"], "tolerance": tolerance(wl),
                       "input_types": sorted({v["type"] for v in wl["inputs"].values()})}
                rows.append(row)
                prob["selected"].append({"pick": pname_, "index": i, "uuid": wl["uuid"], "stem": stem})
            problems_out.append(prob)
    man = {"format": "modelblaster.sol_upstream/v1", "source": str(bench), "variant": variant,
           "cap_mb": cap_mb, "picks": list(picks), "seed": SOL_SEED, "workloads": rows,
           "problems": problems_out}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"manifest.{variant}.json").write_text(json.dumps(man, indent=1))
    return man


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("bench")
    c.add_argument("out")
    c.add_argument("--sets", default="L1,L2")
    c.add_argument("--dtype", default="fp32", choices=sorted(VARIANTS))
    c.add_argument("--cap-mb", type=float, default=256)
    c.add_argument("--pick", default="smallest,median,largest")
    c.add_argument("--problems", default=None, help="regex over <set>/<problem>")
    c.add_argument("--blob-root", default=None, help="SOL flashinfer-trace root (safetensors blobs)")
    z = sub.add_parser("sizes")
    z.add_argument("bench")
    z.add_argument("--sets", default="L1,L2")
    z.add_argument("--dtype", default="fp32", choices=sorted(VARIANTS))
    a = ap.parse_args(argv)
    sets = [s for s in a.sets.split(",") if s]
    if a.cmd == "convert":
        man = convert(Path(a.bench), Path(a.out), sets, a.dtype, a.cap_mb, a.pick.split(","), a.problems,
                      Path(a.blob_root) if a.blob_root else None)
        n = len(man["workloads"])
        np_ = sum(1 for p in man["problems"] if p["selected"])
        print(f"{n} workloads from {np_}/{len(man['problems'])} problems -> {a.out}/src/{a.dtype}")
        for p in man["problems"]:
            if not p["selected"]:
                print(f"  refused {p['set']}/{p['problem']}: {p.get('refusal')}")
    else:
        for s in sets:
            for pdir in sorted((Path(a.bench) / s).iterdir()):
                defn, wls = load_problem(pdir)
                mib = sorted(io_bytes(defn, resolve_axes(defn, w["axes"]), a.dtype) / 2 ** 20 for w in wls)
                print(f"{s}/{pdir.name}\t{len(wls)}\t{mib[0]:.1f}\t{mib[len(mib) // 2]:.1f}\t{mib[-1]:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
