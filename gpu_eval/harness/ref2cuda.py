"""Harness 2: auto-port every ModelBlaster reference C kernel to CUDA.

For each KernelSpec, rewrite `reference_impl` as __device__ code and wrap it
in a single-thread __global__ plus an extern "C" launcher with the ORIGINAL
signature. The result is a slow but trivially-correct CUDA seed for every
op ModelBlaster supports; this script measures how many survive nvcc.

    python ref2cuda.py specs.json OUT_DIR [--arch sm_89] [--jobs 16]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mbabi import parse_signature

_CTYPES = r"(?:void|float|double|int|long|short|char|unsigned|signed|size_t|_Float16|__fp16|u?int(?:8|16|32|64)_t)"
# A top-level function header (definition or prototype) at column 0.
_FN_HDR = re.compile(rf"^((?:static\s+)?(?:inline\s+)?(?:const\s+)?{_CTYPES}(?:\s+{_CTYPES})*\s*\**\s*\w+\s*\()", re.M)
# File-scope data that device code reads must live in device memory.
_FILE_DATA = re.compile(rf"^((?:static\s+)?(?:const\s+)?{_CTYPES}\s+\w+\s*(?:\[[^\]]*\])+\s*=)", re.M)
_SCALAR_DECL = re.compile(r"\b(float|int|double|int8_t|int16_t|int32_t|int64_t|uint8_t|_Float16)\s+(\w+)\s*\[([^\]]+)\]\s*;")


def _is_vla(size_expr: str) -> bool:
    # Constant sizes are digits / ALL_CAPS macros / arithmetic on those.
    return re.search(r"\b[a-z_]\w*\b", size_expr) is not None or re.search(r"\b[A-Z]{1,2}\b", size_expr) is not None


def port(op: str, spec: dict, fix_vla: bool) -> tuple[str, dict]:
    name, params = parse_signature(spec["signature"])
    src = spec["reference_impl"]
    notes = collections.Counter()
    src = re.sub(rf"\b{name}\b", f"mbdev_{name}", src)
    src, n = _FN_HDR.subn(r"__device__ \1", src)
    notes["device_fns"] = n
    src, n = _FILE_DATA.subn(r"__device__ \1", src)
    notes["device_data"] = n
    vlas = [m for m in _SCALAR_DECL.finditer(src) if _is_vla(m.group(3))]
    notes["vla"] = len(vlas)
    if fix_vla and vlas:
        # Device malloc stands in for a stack VLA; the seed leaks it, which is
        # fine for one verify/profile run (heap limit raised in the launcher).
        src = _SCALAR_DECL.sub(
            lambda m: (f"{m.group(1)} *{m.group(2)} = ({m.group(1)} *)malloc(sizeof({m.group(1)}) * ({m.group(3)}));"
                       if _is_vla(m.group(3)) else m.group(0)), src)
    plist = ", ".join(f"{p.ctype} {p.name}" for p in params)
    alist = ", ".join(p.name for p in params)
    # Raising the device heap is only legal before the context first uses
    # device malloc, and each seed .so tries independently -- so only VLA
    # seeds ask, and a refusal is cleared rather than left sticky.
    heap = ("    static bool heap_set = false;\n"
            "    if (!heap_set) { cudaDeviceSetLimit(cudaLimitMallocHeapSize, (size_t)512 << 20);"
            " cudaGetLastError(); heap_set = true; }\n") if (fix_vla and vlas) else ""
    wrapper = f"""
__global__ void mbg_{name}({plist}) {{ mbdev_{name}({alist}); }}

extern "C" void {name}({plist}) {{
{heap}    mbg_{name}<<<1, 1>>>({alist});
}}

extern "C" int mb_last_error(void) {{ return (int)cudaGetLastError(); }}
"""
    # nvcc cannot code-generate C23 _Float16 in device code ("unsupported
    # float variant" ICE), and __half's operator overloads make the
    # reference's mixed half/float expressions ambiguous. mb_half is 2-byte
    # storage that computes in float -- what GCC does for _Float16 on x86.
    # Its float ctor is explicit (else `c ? h : 0.0f` is ambiguous), so
    # copy-initialisation from an expression is rewritten to direct-init.
    src, n = re.subn(r"^(\s*)((?:const\s+)?_Float16\s+\w+)\s*=\s*([^;]+);",
                     r"\1\2((float)(\3));", src, flags=re.M)
    notes["f16_inits"] = n
    prelude = ("#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n#include <stdint.h>\n#include <math.h>\n"
               "struct mb_half {\n"
               "    __half h;\n"
               "    mb_half() = default;\n"
               "    __host__ __device__ explicit mb_half(float f) : h(__float2half_rn(f)) {}\n"
               "    __host__ __device__ operator float() const { return __half2float(h); }\n"
               "    __host__ __device__ mb_half &operator=(float f) { h = __float2half_rn(f); return *this; }\n"
               "    __host__ __device__ mb_half &operator+=(float f) { return *this = float(*this) + f; }\n"
               "    __host__ __device__ mb_half &operator-=(float f) { return *this = float(*this) - f; }\n"
               "    __host__ __device__ mb_half &operator*=(float f) { return *this = float(*this) * f; }\n"
               "    __host__ __device__ mb_half &operator/=(float f) { return *this = float(*this) / f; }\n"
               "};\n"
               "static_assert(sizeof(mb_half) == 2, \"_Float16 ABI\");\n"
               "#define _Float16 mb_half\n")
    return prelude + src + wrapper, dict(notes)


def first_error(log: str) -> str:
    lines = log.splitlines()
    return next((l for l in lines if re.search(r"\berror\b|Error:", l)), lines[-1] if lines else "")


def classify(log: str) -> str:
    first = first_error(log)
    rules = [("Internal Compiler Error", "nvcc_ice"),("variable length array|expression must have a constant value", "vla"),
             ("calling a __host__ function", "host_fn_call"),
             ("_Float16|__fp16|half", "fp16_type"),
             ("identifier .* is undefined", "undefined_identifier"),
             ("cannot be directly read in a device function|__device__ variable", "file_scope_data")]
    for pat, cls in rules:
        if re.search(pat, first):
            return cls
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs")
    ap.add_argument("out")
    ap.add_argument("--arch", default="sm_89")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--no-fix-vla", action="store_true")
    a = ap.parse_args()
    specs = json.load(open(a.specs))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def build(op):
        rec = {"op": op}
        try:
            text, notes = port(op, specs[op], fix_vla=not a.no_fix_vla)
        except Exception as e:
            return {**rec, "ok": False, "stage": "port", "error": str(e)}
        cu = out / f"{op}.cu"
        cu.write_text(text)
        so = out / f"lib_{op}.so"
        p = subprocess.run(["nvcc", "-O2", f"-arch={a.arch}", "-Xcompiler", "-fPIC", "-shared",
                            "-o", str(so), str(cu)], capture_output=True, text=True)
        rec.update(notes)
        if p.returncode != 0:
            (out / f"{op}.err").write_text(p.stderr)
            return {**rec, "ok": False, "stage": "nvcc", "class": classify(p.stderr),
                    "error": first_error(p.stderr)}
        return {**rec, "ok": True}

    with ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(build, sorted(specs)))
    json.dump(results, open(out / "ref2cuda_results.json", "w"), indent=1)
    ok = [r for r in results if r["ok"]]
    fails = collections.Counter(r.get("class", r.get("stage")) for r in results if not r["ok"])
    by_dtype = collections.defaultdict(lambda: [0, 0])
    for r in results:
        dt = "s8" if "_s8" in r["op"] or r["op"].endswith("_i8") else ("f16" if "f16" in r["op"] else "fp32")
        by_dtype[dt][0] += r["ok"]
        by_dtype[dt][1] += 1
    print(f"compiled {len(ok)}/{len(results)} reference kernels to CUDA seeds")
    print("by dtype:", {k: f"{v[0]}/{v[1]}" for k, v in by_dtype.items()})
    print("failure classes:", dict(fails))
    print("vla rewrites applied:", sum(1 for r in results if r.get("vla")))


if __name__ == "__main__":
    main()
