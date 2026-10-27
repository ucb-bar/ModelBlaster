"""Generate the fp16 (`_f16`) and bf16 (`_bf16`) hand CUDA kernels from the fp32 ones.

ModelBlaster derives every `_f16` KernelSpec from its fp32 spec mechanically
(pipeline/reference_kernels.py `_make_f16_variant`): float POINTERS become
`_Float16 *`, scalars and locals stay `float`, so storage is fp16 and the math
is fp32. This applies the same rule to cuda/mb_cuda_kernels.cu, with the
CUDA spelling of the type (`__half`; nvcc cannot generate code for
`_Float16`, and the two are the same IEEE binary16 in memory):

  * `const float *` / `float *` -> `const __half *` / `__half *`, except the
    names in KEEP_FLOAT (shared-memory reduction scratch);
  * every element read `p[i]` of a half pointer -> `mb_ld(p[i])` (a float), so
    no expression is evaluated in fp16; stores `p[i] = <float expr>` round once
    (round-to-nearest-even); `p[i] op= v` -> `p[i] = mb_ld(p[i]) op (v)`;
  * extern "C" entries `kernel_<op>` -> `kernel_<op>_f16` (calls included).

Entries whose `_f16` KernelSpec is hand-written with a different signature
(softmax_f16's input_scale, adaptive_avg_pool2d_f16's OH/OW, cat2_c1_f16's
argument order) are replaced by cuda/mb_cuda_kernels_f16_hand.cu.

The bf16 file (cuda/mb_cuda_kernels_bf16.cu, docs/bf16_cuda.md) is the same
transform with `__nv_bfloat16` (<cuda_bf16.h>; `__bfloat162float` loads,
round-to-nearest-even stores, which compile for sm_75 too). Every `_bf16`
KernelSpec is derived from its fp32 spec, so only kernels whose fp32 code
would round an intermediate in the storage type are replaced, by
cuda/mb_cuda_kernels_bf16_hand.cu: softmax (stores the exponentials, then
rescales them) and sdpa (accumulates P.V in the output row).

    python f16_kernels.py [--check]     # (re)write both generated files
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FP32_CU = HERE / "cuda" / "mb_cuda_kernels.cu"
HAND_CU = HERE / "cuda" / "mb_cuda_kernels_f16_hand.cu"
OUT_CU = HERE / "cuda" / "mb_cuda_kernels_f16.cu"
KEEP_FLOAT = {"sh"}          # mb_block_sum's __shared__ float scratch
HALF = "__half"


def _prelude(t: str, header: str, label: str, widen: str) -> str:
    return f"""#include <{header}>

// {label} storage, fp32 math: every element read of a {t} buffer goes through
// mb_ld; a store of a float/double rounds once (round-to-nearest-even).
__device__ __forceinline__ float mb_ld({t} v) {{ return {widen}(v); }}
"""


# storage type -> (CUDA type, suffix, prelude, hand file, generated file)
DTYPES = {
    "f16": (HALF, "_f16", _prelude(HALF, "cuda_fp16.h", "fp16", "__half2float"), HAND_CU, OUT_CU),
    "bf16": ("__nv_bfloat16", "_bf16", _prelude("__nv_bfloat16", "cuda_bf16.h", "bf16", "__bfloat162float"),
             HERE / "cuda" / "mb_cuda_kernels_bf16_hand.cu", HERE / "cuda" / "mb_cuda_kernels_bf16.cu"),
}
PRELUDE = DTYPES["f16"][2]


def _top_level_chunks(src: str) -> list[str]:
    """Split at top-level `}` / `;` so each function is processed alone
    (comments and strings in this file contain no braces)."""
    out, depth, start = [], 0, 0
    for i, c in enumerate(src):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                out.append(src[start:i + 1])
                start = i + 1
        elif c == ";" and depth == 0:
            out.append(src[start:i + 1])
            start = i + 1
    out.append(src[start:])
    return out


def _match_bracket(s: str, i: int) -> int:
    """s[i] == '['; index just past its matching ']'."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "[":
            depth += 1
        elif s[j] == "]":
            depth -= 1
            if depth == 0:
                return j + 1
    raise ValueError("unbalanced [")


def _stmt_end(s: str, i: int) -> int:
    """Index of the `;` ending the expression that starts at i (same nesting)."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] in "([{":
            depth += 1
        elif s[j] in ")]}":
            depth -= 1
        elif s[j] == ";" and depth == 0:
            return j
    raise ValueError("no ;")


def _wrap_loads(chunk: str, names: set[str]) -> str:
    if not names:
        return chunk
    pat = re.compile(r"\b(" + "|".join(sorted(names, key=len, reverse=True)) + r")\[")
    out, i = [], 0
    while True:
        m = pat.search(chunk, i)
        if m is None:
            out.append(chunk[i:])
            return "".join(out)
        a, b = m.start(), _match_bracket(chunk, m.end() - 1)
        inner = _wrap_loads(chunk[m.end():b - 1], names)     # nested index reads
        ref = f"{m.group(1)}[{inner}]"
        rest = chunk[b:]
        out.append(chunk[i:a])
        cm = re.match(r"\s*([+\-*/])=(?!=)", rest)
        if cm:                                                # p[i] op= v
            e = _stmt_end(chunk, b + cm.end())
            rhs = _wrap_loads(chunk[b + cm.end():e], names).strip()
            out.append(f"{ref} = mb_ld({ref}) {cm.group(1)} ({rhs})")
            i = e
        elif re.match(r"\s*=(?!=)", rest):                    # store
            out.append(ref)
            i = b
        else:                                                 # read
            out.append(f"mb_ld({ref})")
            i = b


def _half_names(chunk: str, t: str = HALF) -> set[str]:
    names = set()
    for m in re.finditer(rf"\b{t} \*(\w+)(\[)?", chunk):
        if not m.group(2):                       # `__half *ins[8]` is an array of pointers
            names.add(m.group(1))
        # further declarators of the same declaration: `__half *v = ..., *ix = ...`
        depth, j = 0, m.end()
        while j < len(chunk) and not (depth == 0 and chunk[j] in ";)"):
            depth += chunk[j] in "([{"
            depth -= chunk[j] in ")]}"
            if depth == 0 and chunk[j] == ",":
                d = re.match(r"\s*\*(\w+)", chunk[j + 1:])
                if d:
                    names.add(d.group(1))
            j += 1
    return names


def _defined_entries(src: str) -> set[str]:
    return set(re.findall(r'extern\s+"C"\s+void\s+(kernel_\w+)\s*\(', src))


def _drop_entries(src: str, names: set[str]) -> str:
    """Remove the extern "C" definitions of `names` (with the comment above)."""
    out = []
    for ch in _top_level_chunks(src):
        m = re.search(r'extern\s+"C"\s+void\s+(kernel_\w+)\s*\(', ch)
        if m and m.group(1) in names:
            ch = ch[:ch.find('extern "C"')]
            ch = "\n".join(ch.split("\n")[:-1]) if not ch.strip() else ch.rstrip(" \t")
            ch = re.sub(r"(\n\s*//[^\n]*)+\s*$", "\n", ch)
        out.append(ch)
    return "".join(out)


def generate(fp32_src: str, hand_src: str, dtype: str = "f16") -> str:
    t, sfx, prelude, hand_cu, _ = DTYPES[dtype]
    src = fp32_src
    keep = "|".join(sorted(KEEP_FLOAT))
    src = re.sub(rf"\bconst float \*(?!(?:{keep})\b)", f"const {t} *", src)
    src = re.sub(rf"(?<!const )\bfloat \*(?!(?:{keep})\b)", f"{t} *", src)
    src = re.sub(rf"\bkernel_(\w+?)({sfx})?\s*\(", lambda m: f"kernel_{m.group(1)}{sfx}(", src)
    src = "".join(_wrap_loads(ch, _half_names(ch, t)) for ch in _top_level_chunks(src))
    src = _drop_entries(src, _defined_entries(hand_src))
    head = ("// GENERATED by gpu_eval/harness/f16_kernels.py from mb_cuda_kernels.cu and\n"
            f"// {hand_cu.name} -- do not edit; regenerate.\n//\n")
    src = src.replace("#include <math.h>\n", "#include <math.h>\n" + prelude, 1)
    hand_body = "\n".join(l for l in hand_src.splitlines() if not l.startswith("#include"))
    return head + src.rstrip("\n") + "\n\n" + hand_body.strip("\n") + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 if a committed file is stale")
    ap.add_argument("--dtype", choices=sorted(DTYPES), action="append",
                    help="only this storage type (repeatable; default: all)")
    ap.add_argument("--out", help="output path (with a single --dtype)")
    a = ap.parse_args()
    dts = a.dtype or sorted(DTYPES, reverse=True)          # f16, bf16
    if a.out and len(dts) != 1:
        ap.error("--out needs exactly one --dtype")
    stale = False
    for dt in dts:
        _, _, _, hand_cu, out_cu = DTYPES[dt]
        text = generate(FP32_CU.read_text(), hand_cu.read_text(), dt)
        out = Path(a.out) if a.out else out_cu
        if a.check:
            ok = out.exists() and out.read_text() == text
            print(f"{out}: {'up to date' if ok else 'STALE'}")
            stale |= not ok
            continue
        out.write_text(text)
        print(f"wrote {out}")
    sys.exit(1 if stale else 0)


if __name__ == "__main__":
    main()
