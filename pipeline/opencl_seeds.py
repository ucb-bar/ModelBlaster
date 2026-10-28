"""OpenCL seed sources for multi-slot slots (target ``opencl``).

A seed is a complete slot source: one or more ``static mb_program cl_* =
MB_CL(R"CL(...)CL");`` OpenCL C programs and the host launcher with exactly the
op's KernelSpec signature (see ``gpu_eval/harness/opencl/mb_opencl_runtime.h``).

Three families are generated from compact templates, because their kernels
differ only in one expression -- the kernel takes the launcher's parameters
one-to-one, so a template is (range, body) written in the signature's names:

  MAP     elementwise ops (relu, gelu, hardtanh(min_val, max_val), add, ...)
  AXIS    [outer, reduce, inner] reductions and normalisations (sum_dim,
          argmax_dim, rms_norm, ...), one work-item per (outer, inner)
  LOSS    scalar losses (mse, huber, kldiv, cross-entropy, ...): per-group
          compensated partial sums, then one group adds them up

Everything else (conv, pooling, GEMM family, row ops, layout, indexing,
scans, recurrent ops) is hand-written in ``mb_opencl_kernels.cpp`` and a
slot gets its launcher's dependency closure (the CUDA library mechanism).

    python -m modelblaster.pipeline.opencl_seeds OUT_DIR   # every seed + one TU that builds them all
"""
from __future__ import annotations

import argparse
import re as _re
from pathlib import Path

from .kb_export import CudaLibrary
from .kernel_abi import parse_signature


def _cl_param(ctype: str, name: str) -> str:
    if "*" in ctype:
        base = ctype.replace("const", "").replace("*", "").strip()
        base = {"int64_t": "long", "int32_t": "int"}.get(base, base)
        return f"__global {'const ' if 'const' in ctype else ''}{base} *{name}"
    return f"{ctype} {name}"


# ---------------------------------------------------------------- MAP: one work-item per element
# op -> expression of `v` (the input element) and the signature's scalars
MAP1 = {
    "relu": "fmax(v, 0.0f)",
    "relu6": "fmin(fmax(v, 0.0f), 6.0f)",
    "sigmoid": "1.0f / (1.0f + exp(-v))",
    "silu": "v / (1.0f + exp(-v))",
    "swish": "v * (1.0f / (1.0f + exp(-v)))",
    "elu": "v > 0.0f ? v : alpha * (exp(v) - 1.0f)",
    "selu": "1.0507009873554805f * (v > 0.0f ? v : 1.6732632423543772f * (exp(v) - 1.0f))",
    "gelu": "0.5f * v * (1.0f + erf(v * 0.70710678118654752f))",
    # MB's gelu_exact is the tanh (MinGPT) form
    "gelu_exact": "0.5f * v * (1.0f + tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)))",
    "tanh": "tanh(v)",
    "log": "log(v)",
    "hardsigmoid": "v <= -3.0f ? 0.0f : (v >= 3.0f ? 1.0f : v * (1.0f / 6.0f) + 0.5f)",
    "hardtanh": "fmin(fmax(v, min_val), max_val)",
    "leaky_relu": "v > 0.0f ? v : negative_slope * v",
    "softplus": "v > 0.0f ? v + log1p(exp(-v)) : log1p(exp(v))",
    "softsign": "v / (1.0f + fabs(v))",
    "mul_scalar": "v * s",
}
# op -> expression of `va`, `vb` (same-shape operands)
MAP2 = {"add": "va + vb", "mul": "va * vb"}

# ---------------------------------------------------------------- AXIS: [outer, reduce, inner]
_AXIS_HEAD = """    long t = MB_TID; if (t >= (long)outer * inner) return;
    long o = t / inner, i = t % inner;
    __global const float *x = input + o * (long)reduce * inner + i;
"""
_NORM_TAIL = """    __global float *y = output + o * (long)reduce * inner + i;
    for (long r = 0; r < reduce; r++) y[r * inner] = x[r * inner] * inv;
"""
AXIS = {
    "sum_dim": "    mb_ksum s = mb_ksum0();\n    for (long r = 0; r < reduce; r++) mb_kadd(&s, x[r * inner]);\n"
               "    output[t] = s.s;\n",
    "mean_dim": "    mb_ksum s = mb_ksum0();\n    for (long r = 0; r < reduce; r++) mb_kadd(&s, x[r * inner]);\n"
                "    output[t] = s.s / (float)reduce;\n",
    "max_dim": "    float m = x[0];\n    for (long r = 1; r < reduce; r++) { float v = x[r * inner]; if (v > m || isnan(v)) m = v; }\n"
               "    output[t] = m;\n",
    "min_dim": "    float m = x[0];\n    for (long r = 1; r < reduce; r++) { float v = x[r * inner]; if (v < m || isnan(v)) m = v; }\n"
               "    output[t] = m;\n",
    "prod_dim": "    float p = 1.0f;\n    for (long r = 0; r < reduce; r++) p *= x[r * inner];\n    output[t] = p;\n",
    # first index on ties, as torch
    "argmax_dim": "    float m = -FLT_MAX; long idx = 0;\n"
                  "    for (long r = 0; r < reduce; r++) { float v = x[r * inner]; if (v > m) { m = v; idx = r; } }\n"
                  "    output[t] = idx;\n",
    "argmin_dim": "    float m = FLT_MAX; long idx = 0;\n"
                  "    for (long r = 0; r < reduce; r++) { float v = x[r * inner]; if (v < m) { m = v; idx = r; } }\n"
                  "    output[t] = idx;\n",
    "l1_norm": "    mb_ksum s = mb_ksum0();\n    for (long r = 0; r < reduce; r++) mb_kadd(&s, fabs(x[r * inner]));\n"
               "    float inv = 1.0f / s.s;\n" + _NORM_TAIL,
    "l2_norm": "    mb_ksum s = mb_ksum0();\n    for (long r = 0; r < reduce; r++) { float v = x[r * inner]; mb_kadd(&s, v * v); }\n"
               "    float inv = 1.0f / sqrt(s.s);\n" + _NORM_TAIL,
    "rms_norm": "    mb_ksum s = mb_ksum0();\n    for (long r = 0; r < reduce; r++) { float v = x[r * inner]; mb_kadd(&s, v * v); }\n"
                "    float inv = 1.0f / sqrt(s.s / (float)reduce + eps);\n" + _NORM_TAIL,
    "mean_abs_norm": "    mb_ksum s = mb_ksum0();\n    for (long r = 0; r < reduce; r++) mb_kadd(&s, fabs(x[r * inner]));\n"
                     "    float inv = 1.0f / (s.s / (float)reduce);\n" + _NORM_TAIL,
}

# ---------------------------------------------------------------- LOSS: scalar output[0]
# op -> (items to reduce over, code setting `float term` for item k, host scale)
LOSS = {
    "mse_loss": ("(long)n", "float d = a[k] - b[k]; float term = d * d;", "1.0f / (float)n"),
    "huber_loss": ("(long)n", "float d = fabs(a[k] - b[k]);\n        "
                   "float term = d < beta ? 0.5f * d * d / beta : d - 0.5f * beta;", "1.0f / (float)n"),
    "hinge_loss": ("(long)n", "float h = 1.0f - pred[k] * targ[k % targ_len]; float term = h > 0.0f ? h : 0.0f;",
                   "1.0f / (float)n"),
    "kldiv_loss": ("(long)N * C", "float tk = target[k];\n        "
                   "float term = tk > 0.0f ? tk * (log(tk) - log_input[k]) : 0.0f;", "1.0f / (float)N"),
    "cross_entropy_loss": ("(long)N", """__global const float *row = logits + k * C;
        float mx = row[0];
        for (int c = 1; c < C; c++) mx = fmax(mx, row[c]);
        mb_ksum se = mb_ksum0();
        for (int c = 0; c < C; c++) mb_kadd(&se, exp(row[c] - mx));
        int tc = (int)(targets[k] + 0.5f);
        float term = mx + log(se.s) - row[tc];""", "1.0f / (float)N"),
    "triplet_loss": ("(long)B", """__global const float *pa = anchor + k * F, *pp = pos + k * F, *pn = neg + k * F;
        mb_ksum sp = mb_ksum0(), sn = mb_ksum0();
        for (int f = 0; f < F; f++) { float dp = pa[f] - pp[f], dn = pa[f] - pn[f]; mb_kadd(&sp, dp * dp); mb_kadd(&sn, dn * dn); }
        float l = sqrt(sp.s) - sqrt(sn.s) + margin;
        float term = l > 0.0f ? l : 0.0f;""", "1.0f / (float)B"),
}


def _sig(spec_signature: str):
    entry, params = parse_signature(spec_signature)
    host = f"void {entry}({', '.join(f'{p.ctype} {p.name}' for p in params)})"
    dev = ", ".join(_cl_param(p.ctype, p.name) for p in params)
    names = [p.name for p in params]
    return entry, params, host, dev, names


def template_source(op: str, signature: str) -> str | None:
    """Generated slot source for ``op``, or None if no template covers it."""
    entry, params, host, dev, names = _sig(signature)
    ptrs = [p for p in params if p.is_ptr]
    args = ", ".join(names)
    if op in MAP1:
        x, y = ptrs[0].name, ptrs[1].name
        body = f"    long i = MB_TID;\n    if (i < n) {{ float v = {x}[i]; {y}[i] = {MAP1[op]}; }}\n"
        rng = "mb_grid(n)"
    elif op in MAP2:
        a, b, y = (p.name for p in ptrs[:3])
        body = f"    long i = MB_TID;\n    if (i < n) {{ float va = {a}[i], vb = {b}[i]; {y}[i] = {MAP2[op]}; }}\n"
        rng = "mb_grid(n)"
    elif op in AXIS:
        body = _AXIS_HEAD + AXIS[op]
        rng = "mb_grid((long)outer * inner)"
    elif op in LOSS:
        return _loss_source(op, entry, host, dev, names)
    else:
        return None
    return (f"static mb_program cl_{op} = MB_CL(R\"CL(\n__kernel void k_{op}({dev}) {{\n{body}}}\n)CL\");\n"
            f"{host} {{\n    mb_launch(MB_KERNEL(cl_{op}, k_{op}), {rng}, {args});\n}}\n")


def _loss_source(op, entry, host, dev, names) -> str:
    count, term, scale = LOSS[op]
    args = ", ".join(names)
    return f"""// per-group compensated partial sums of the loss terms, then one group adds them up
static mb_program cl_{op} = MB_CL(R"CL(
__kernel void k_{op}_partial({dev}, __global float *part) {{
    __local float sh[MB_THREADS];
    int t = get_local_id(0);
    mb_ksum acc = mb_ksum0();
    for (long k = get_global_id(0); k < {count}; k += get_global_size(0)) {{
        {term}
        mb_kadd(&acc, term);
    }}
    sh[t] = acc.s; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {{ if (t < s) sh[t] += sh[t + s]; barrier(CLK_LOCAL_MEM_FENCE); }}
    if (t == 0) part[get_group_id(0)] = sh[0];
}}
__kernel void k_{op}_final(__global const float *part, int np, __global float *out, float scale) {{
    __local float sh[MB_THREADS];
    int t = get_local_id(0);
    mb_ksum acc = mb_ksum0();
    for (int k = t; k < np; k += get_local_size(0)) mb_kadd(&acc, part[k]);
    sh[t] = acc.s; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {{ if (t < s) sh[t] += sh[t + s]; barrier(CLK_LOCAL_MEM_FENCE); }}
    if (t == 0) out[0] = sh[0] * scale;
}}
)CL");
{host} {{
    long cnt = {count};
    int np = (int)std::min(1024L, std::max(1L, (cnt + MB_THREADS - 1) / MB_THREADS));
    float *part = (float *)MB_SCRATCH(cl_{op}, np * sizeof(float));
    mb_launch(MB_KERNEL(cl_{op}, k_{op}_partial), mb_groups(np), {args}, part);
    mb_launch(MB_KERNEL(cl_{op}, k_{op}_final), mb_groups(1), (const float *)part, np, output, {scale});
}}
"""


# ---------------------------------------------------------------- fp16 (binary16 storage, fp32 math)
#
# The `_f16` seeds follow the CUDA fp16 contract (docs/fp16_cuda.md): data is
# stored as IEEE binary16 and every element is widened to fp32 on load; each
# stored value is rounded once (round-to-nearest-even). In OpenCL that is
# `__global half *` with vload_half / vstore_half -- core OpenCL 1.2, no
# cl_khr_fp16 needed. They are generated from the fp32 seeds: data pointers
# become half (scratch partials stay float), element reads p[e] become
# vload_half(e, p), stores p[e] = v become vstore_half(v, e, p), compound
# stores p[e] op= v become vstore_half(vload_half(e, p) op (v), e, p) (and
# `(long)` of a loaded half goes through int, see f16_device); on the
# host `float *` becomes `__half *` (cl_half, the CUDA spelling of _Float16)
# and entries get their _f16 names. Ops whose _f16 spec has a different
# signature are hand-written in mb_opencl_kernels_f16_hand.cpp.


F16_KEEP = {"part"}                 # float scratch (loss / norm partial sums)
F16_UNSUPPORTED = {"kernel_scatter_dim"}   # scatter-add needs half atomics
_CL_CHUNK = _re.compile(r'(R"CL\()(.*?)(\)CL")', _re.S)
_PTR_DECL = _re.compile(r"__global\s+(const\s+)?float\s*\*\s*(\w+)")


def _match(text: str, i: int, open_c: str = "[", close_c: str = "]") -> int:
    """Index just past the bracket that closes text[i] (an opening bracket)."""
    depth = 0
    for j in range(i, len(text)):
        if text[j] in "([{":
            depth += 1
        elif text[j] in ")]}":
            depth -= 1
            if depth == 0:
                return j + 1
    raise ValueError("unbalanced brackets")


def _stmt_end(text: str, i: int) -> int:
    """Index of the `;` (or unmatched `)`/`,` at depth 0) ending the expression starting at i."""
    depth = 0
    for j in range(i, len(text)):
        c = text[j]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0:
                return j
            depth -= 1
        elif c in ";," and depth == 0:
            return j
    return len(text)


def _half_access(code: str, names: set) -> str:
    if not names:
        return code
    pat = _re.compile(r"\b(" + "|".join(sorted(map(_re.escape, names))) + r")\s*\[")
    out, i = [], 0
    while True:
        m = pat.search(code, i)
        if not m:
            out.append(code[i:])
            return "".join(out)
        name, lb = m.group(1), m.end() - 1
        rb = _match(code, lb)
        idx = _half_access(code[lb + 1:rb - 1], names)
        out.append(code[i:m.start()])
        rest = code[rb:]
        am = _re.match(r"\s*([-+*/]?)=(?!=)", rest)
        if am:
            rs = rb + am.end()
            re_ = _stmt_end(code, rs)
            rhs = _half_access(code[rs:re_], names).strip()
            if am.group(1):
                rhs = f"vload_half({idx}, {name}) {am.group(1)} ({rhs})"
            out.append(f"vstore_half({rhs}, {idx}, {name})")
            i = re_
        else:
            out.append(f"vload_half({idx}, {name})")
            i = rb


def f16_device(cl: str) -> str:
    names = {m.group(2) for m in _PTR_DECL.finditer(cl)} - F16_KEEP
    cl = _PTR_DECL.sub(lambda m: m.group(0) if m.group(2) in F16_KEEP else
                       f"__global {m.group(1) or ''}half *{m.group(2)}", cl)
    # `__global const float *a = ..., *b = ...`: the later declarators too
    for m in _re.finditer(r"__global\s+(?:const\s+)?half\s*\*[^;]*;", cl):
        names |= set(_re.findall(r",\s*\*\s*(\w+)", m.group(0)))
    # an index stored as half: convert through int. Adreno's compiler aborts
    # (QGPUI64Emulate: "Unsupported FtoI/ItoF") on a half -> 64-bit int conversion
    return _re.sub(r"\(long\)\s*vload_half\(", "(long)(int)vload_half(", _half_access(cl, names))


def f16_host(code: str, renames: dict) -> str:
    lines = []
    for line in code.split("\n"):
        if "MB_SCRATCH" not in line and not _re.search(r"\bpart\b", line):
            line = _re.sub(r"\bfloat\s*\*", "__half *", line).replace("sizeof(float)", "sizeof(__half)")
        lines.append(line)
    code = "\n".join(lines)
    for a, b in renames.items():
        code = _re.sub(rf"\b{a}\s*\(", f"{b}(", code)
    return code


def to_f16(src: str, renames: dict) -> str:
    """fp32 slot source -> its fp16 variant (see above)."""
    parts, last = [], 0
    for m in _CL_CHUNK.finditer(src):
        parts.append(f16_host(src[last:m.start()], renames))
        parts.append(m.group(1) + f16_device(m.group(2)) + m.group(3))
        last = m.end()
    parts.append(f16_host(src[last:], renames))
    return "".join(parts)


TEMPLATED = set(MAP1) | set(MAP2) | set(AXIS) | set(LOSS)


class OpenCLSeeds:
    """Seed sources by C entry point: templates first, then the hand library."""

    def __init__(self, hand_path: Path, f16_hand_path: Path | None = None):
        from .reference_kernels import KERNEL_SPECS
        self.lib = CudaLibrary(Path(hand_path).read_text())
        f16 = Path(f16_hand_path) if f16_hand_path else Path(hand_path).with_name("mb_opencl_kernels_f16_hand.cpp")
        self.lib16 = CudaLibrary(f16.read_text()) if f16.exists() else None
        self._op_of = {}
        for op, spec in KERNEL_SPECS.items():
            self._op_of.setdefault(parse_signature(spec.signature)[0], (op, spec.signature))

    def _f16_base(self, entry: str):
        if entry.endswith("_f16") and not (self.lib16 and entry in self.lib16.items):
            base = entry[:-len("_f16")]
            return None if base in F16_UNSUPPORTED else base
        return None

    def has(self, entry: str) -> bool:
        if self.lib16 is not None and entry in self.lib16.items:
            return True
        base = self._f16_base(entry)
        if base is not None:
            return self.has(base)
        if entry.endswith("_f16"):
            return False
        op = self._op_of.get(entry, (None,))[0]
        return op in TEMPLATED or entry in self.lib.items

    def origin(self, entry: str) -> str:
        if self.lib16 is not None and entry in self.lib16.items:
            return "hand-f16"
        base = self._f16_base(entry)
        if base is not None:
            return self.origin(base) + "-f16"
        return "template" if self._op_of.get(entry, (None,))[0] in TEMPLATED else "hand"

    def slot_source(self, entry: str) -> str:
        if self.lib16 is not None and entry in self.lib16.items:
            return self.lib16.slot_source(entry)
        base = self._f16_base(entry)
        if base is not None:
            src = self.slot_source(base)
            renames = {e: e + "_f16" for e in _re.findall(r"\bvoid\s+(kernel_\w+)\s*\(", src)
                       if e + "_f16" in self._op_of}
            return to_f16(src, renames)
        op, sig = self._op_of.get(entry, (None, None))
        src = template_source(op, sig) if op in TEMPLATED else None
        return src if src is not None else self.lib.slot_source(entry)

    def head_source(self) -> str:
        return ""

    def fused_seed(self, spec, sub_sources: dict | None = None) -> str:
        """A fused op's seed: its stages' seeds called back to back (the CUDA ``cuda_seed``).

        Each stage is a complete slot source -- ``sub_sources[stage kind]`` when given (e.g.
        the current best OpenCL kernel of that stage), else its seed -- wrapped in its own
        namespace (``stage_<kind>``), so stage programs and helpers never collide; chains of
        any length (``make_chain_spec``)."""
        from .cuda_dtypes import cuda_signature
        from .kb_export import namespaced
        from .kb_proposals import _all_specs, stage_prefix

        specs = _all_specs()
        sub_sources = sub_sources or {}
        stages = [(k, *parse_signature(specs[k].signature)) for k in spec.sub_ops]
        includes, blocks = [], []
        for k, e, _ in dict.fromkeys((k, e, None) for k, e, _ in stages):
            src = sub_sources.get(k) or self.slot_source(e)
            inc, body = namespaced(f"stage_{k}", src)
            includes += [i for i in inc if i not in includes]
            blocks.append(f"// {'current best' if k in sub_sources else 'seed'} {k} kernel\n{body}")
        outs = [next(p.name for p in ps if p.is_ptr and not p.is_const_ptr) for _, _, ps in stages]
        calls = []
        for i, (k, e, ps) in enumerate(stages):
            args = [stage_prefix(i - 1) + outs[i - 1] if i and p.name == spec.t_params[i - 1]
                    else stage_prefix(i) + p.name for p in ps]
            calls.append(f"    stage_{k}::{e}({', '.join(args)});")
        entry, params = parse_signature(cuda_signature(spec.signature))
        scratch = ", ".join(f"`{stage_prefix(i)}{o}`" for i, o in enumerate(outs[:-1]))
        plist = ", ".join(f"{p.ctype} {p.name}" for p in params)
        return ("\n".join(includes) + ("\n\n" if includes else "") + "\n\n".join(blocks) +
                f"\n\n// Seed: {' then '.join(spec.sub_ops)}; {scratch} hold the intermediates\n"
                f"// (scratch). A fused kernel need not write them.\n"
                f"void {entry}({plist}) {{\n" + "\n".join(calls) + "\n}\n")


def all_entries(dtype: str = "fp32") -> dict[str, str]:
    """entry -> op for every fp32 (or `_f16`) KernelSpec: the target set of the seeds. The
    per-tensor casts of a mixed IR (`cast_<src>_to_<dst>`) are CUDA-only and not in it."""
    from .cuda_dtypes import op_dtype, parse_cast
    from .reference_kernels import KERNEL_SPECS
    out = {}
    for op, spec in KERNEL_SPECS.items():
        if any(s in op for s in ("_s8", "_i8")) or "int8_t" in spec.signature or op_dtype(op) != dtype \
                or parse_cast(op):
            continue
        out.setdefault(parse_signature(spec.signature)[0], op)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    a = ap.parse_args()
    from .kb_export import wrap_slot
    from .kb_export_opencl import HAND_LIB, RUNTIME_H
    seeds, out = OpenCLSeeds(HAND_LIB), Path(a.out_dir)
    (out / "slots").mkdir(parents=True, exist_ok=True)
    parts, missing = [RUNTIME_H.read_text()], []
    for entry, op in sorted(all_entries(a.dtype).items()):
        if not seeds.has(entry):
            missing.append(op)
            continue
        src = seeds.slot_source(entry)
        (out / "slots" / f"{op}.cpp").write_text(src)
        parts.append(wrap_slot(op, src))
    (out / "all_slots.cpp").write_text("\n".join(parts))
    print(f"{len(parts) - 1} seeds -> {out}; no seed for: {missing}")


if __name__ == "__main__":
    main()

