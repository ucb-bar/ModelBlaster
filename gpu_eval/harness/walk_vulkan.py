"""Walk a ModelBlaster IR on a Vulkan GPU (AMD RDNA via RADV, ...), one kernel per op.

The Vulkan counterpart of walk_cuda.py: every non-alias op is dispatched
through ModelBlaster's own C ABI (KernelSpec.signature) to a launcher in
vulkan/libmb_vk.so (mbvk.py), on host-mapped device buffers. The output is
checked against the io.npz golden, every kernel is checked in isolation on the
PyTorch walk's own inputs (walk_torch on the CPU: no GPU torch is needed, which
is the point on a ROCm-less APU), and the walk is timed with Vulkan timestamps.

    python walk_vulkan.py IR_DIR... --specs specs.json [--lib libmb_vk.so] \
        [--reps 10] [--no-isolation] [--json out.json]

fp32 IRs (`extract_graph --quant fp32`) are checked as walk_cuda checks them.
int8 IRs (`--quant int8`: the `_s8` ops, e.g. SmolVLA's layernorm / rmsnorm /
gelu / linear / matmul / softmax / mul / sin / cos) are held to bit-exactness:
the output must equal the io.npz golden, and every kernel's output must equal
ModelBlaster's reference C implementation (KernelSpec.reference_impl, built
with the host C compiler, or given prebuilt with --ref-lib) on the same inputs.
int8_mixed IRs (`--quant int8 --int8-mixed`, pipeline/int8_mixed.py: int8 linears
in an fp32 graph) are held to both: every int8 kernel equal to the reference C
and every fp32 kernel within the fp32 bound, in isolation on the CPU reference
walk's inputs; the output within MIXED_GOLDEN_REL of the golden.
An op without a Vulkan launcher is reported in `missing`.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

import walk_torch
from mbabi import bind, parse_signature
from mbir import IR, compare, cuda_dtypes, load_ir, sol_for
from mbvk import VkBackend

_NP = {"f32": np.float32, "i64": np.int64, "i32": np.int32, "i8": np.int8, "b8": np.bool_, "u8": np.uint8}


def bindable(op: dict) -> dict:
    """The op as kernel_abi.bind reads it. int8 records carry their scalars in
    op["quant"] and name a norm's gamma / beta `weight` / `bias` (generate_skeleton
    spells those calls out by hand); fp32 records pass through unchanged."""
    if "quant" not in op or not isinstance(op["quant"], dict):
        return op
    o = dict(op)
    for k, v in op["quant"].items():
        o.setdefault(k, v)
    if op["op"] in ("layernorm_s8", "rmsnorm_s8"):
        o.setdefault("gamma_key", op.get("weight"))
        o.setdefault("beta_key", op.get("bias"))
    return o


def is_int8(ir: IR) -> bool:
    return ir.graph.get("quant") == "int8"


def is_int8_mixed(ir: IR) -> bool:
    """An int8 IR with fp32 islands (pipeline/int8_mixed.py): f32 and i8 tensors side by side."""
    return ir.graph.get("quant") == "int8_mixed"


def _int8_mixed():
    """pipeline/int8_mixed.py, loaded by path (numpy + torch only at import), like mbir's cuda_dtypes."""
    import importlib.util
    from pathlib import Path
    if "mb_int8_mixed" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "mb_int8_mixed", Path(__file__).resolve().parents[2] / "pipeline" / "int8_mixed.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["mb_int8_mixed"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["mb_int8_mixed"]


def _weight_array(a: np.ndarray) -> np.ndarray:
    """weights.npz entry as the kernels read it: float32, int8 (quantized weights),
    int32 (biases, indices; int64 narrowed)."""
    a = np.ascontiguousarray(a)
    if a.dtype.kind == "f":
        return a.astype(np.float32)
    if a.dtype == np.int8:
        return a
    if a.dtype.kind in "iu":
        return a.astype(np.int32)
    return a


class VkKernelSet:
    def __init__(self, specs: dict, vk: VkBackend):
        self.specs, self.vk = specs, vk
        self._cache: dict[str, tuple] = {}

    def resolve(self, op: str):
        """-> (callable, params) or raises KeyError."""
        if op not in self._cache:
            if op not in self.specs:
                raise KeyError(f"no KernelSpec for {op}")
            name, params = parse_signature(self.specs[op]["signature"])
            if not hasattr(self.vk.lib, name):
                raise KeyError(f"no Vulkan kernel for {op}")
            fn = getattr(self.vk.lib, name)
            fn.argtypes = [p.ctypes_type() for p in params]
            fn.restype = None
            self._cache[op] = (fn, params)
        return self._cache[op]


def build_plan(ir: IR, ks: VkKernelSet):
    plan, missing = [], []
    for op in ir.ops:
        if op["op"] in ("view", "chunk2_c1"):
            plan.append((op, None))
            continue
        try:
            plan.append((op, ks.resolve(op["op"])))
        except KeyError:
            missing.append(op["op"])
    return plan, sorted(set(missing))


SKINNY_MAX_M = 64        # vulkan/mb_vk_kernels.cpp: linears of <= 64 rows run the skinny GEMM on a packed weight


def pack_weight(w: np.ndarray, dtype=np.float32) -> np.ndarray:
    """An [N, K] linear weight in the skinny GEMM's layout (shaders/skinny_body.glsl):
    [ceil(N/64), K/4, 64, 4] -- per 64-column block and k-quad, the 64 columns'
    4 values contiguous; columns zero-padded to a multiple of 64."""
    n, k = w.shape
    npad = -(-n // 64) * 64
    p = np.zeros((npad, k), dtype)
    p[:n] = w
    return np.ascontiguousarray(p.reshape(npad // 64, 64, k // 4, 4).transpose(0, 2, 1, 3))


class Weights:
    """weights.npz entries uploaded once (they outlive every run's arena mark).

    pack (default: on unless $MB_VK_PACK=0): the weight of every fp32 linear
    with <= 64 rows also gets a packed copy registered with the library, which
    then runs those linears on the skinny GEMM (same fp32 arithmetic). f16:
    that packed copy is fp16 (half the weight bytes; activations and
    accumulation stay fp32) -- an approximation, off by default."""

    def __init__(self, ir: IR, vk: VkBackend, pack: bool | None = None, f16: bool = False):
        self.ir, self.vk, self.buf, self.packed = ir, vk, {}, {}
        self.pack = (os.environ.get("MB_VK_PACK", "1") != "0") if pack is None else pack
        self.f16 = f16
        if f16 and not self.pack:
            raise ValueError("fp16 weights need the packed (skinny GEMM) path")

    def ptr(self, key):
        if key not in self.buf:
            self.buf[key] = self.vk.upload(_weight_array(self.ir.weights[key]))
        return self.vk.ptr(self.buf[key])

    def _pack(self, op):
        key, sh = op.get("weight"), op.get("shape") or {}
        if (op["op"] != "linear" or not key or key in self.packed or is_int8(self.ir)
                or not hasattr(self.vk.lib, "mb_vk_set_packed")):
            return
        w = self.ir.weights[key]
        if w.ndim != 2 or sh.get("M", SKINNY_MAX_M + 1) > SKINNY_MAX_M or w.shape[1] % 16:
            return
        p = self.vk.upload(pack_weight(np.asarray(w, np.float32), np.float16 if self.f16 else np.float32))
        self.packed[key] = p
        self.vk.lib.mb_vk_set_packed(ctypes.c_void_p(self.ptr(key)), ctypes.c_void_p(self.vk.ptr(p)), int(self.f16))

    def preload(self, plan):
        for op, k in plan:
            if k is None:
                continue
            for role in ("weight", "bias", "gamma_key", "beta_key", "w_scale"):
                if op.get(role):
                    self.ptr(op[role])
            if self.pack:
                self._pack(op)


def _tdtype(ir: IR, t: str):
    return _NP.get(ir.graph["tensors"].get(t, {}).get("dtype", "f32"), np.float32)


def run(ir: IR, plan, vk: VkBackend, weights: Weights, feed=None, time_ops=False, consts=None, bound=None):
    """One forward pass. Returns (flat fp32 output, env, per-op timestamp pairs).

    feed: an env of reference tensors (torch) whose values replace each kernel's
    inputs, so a kernel is checked in isolation from upstream rounding. The
    caller owns the arena: wrap in vk.mark() / vk.release(). bound: graph inputs
    that read an existing device array in place (another graph's output)."""
    env: dict[str, np.ndarray] = {}
    for t, v in ir.input_tensors("cpu").items():
        if bound and t in bound:
            env[t] = bound[t]
        else:
            env[t] = vk.upload(v.numpy().astype(_tdtype(ir, t)))
    for t, v in (consts if consts is not None else ir.const_tensors("cpu", int_dtype=torch.int32)).items():
        env[t] = vk.upload(v.numpy())
    per_op = []
    for op, k in plan:
        if op["op"] == "view":
            off, n = op.get("offset", 0), op["shape"]["n"]
            env[op["outputs"][0]] = env[op["inputs"][0]].reshape(-1)[off:off + n].reshape(ir.tshape(op["outputs"][0]))
            continue
        if op["op"] == "chunk2_c1":
            x = env[op["inputs"][0]]
            assert x.shape[0] == 1, "chunk2_c1 alias needs N==1"
            c = op["shape"]["c_each"]
            for i, t in enumerate(op["outputs"]):
                env[t] = x[:, i * c:(i + 1) * c]
            continue
        fn, params = k
        fed = {}
        if feed is not None:
            for t in op["inputs"]:
                if t in feed and not ir.graph["tensors"].get(t, {}).get("const"):
                    v = feed[t].detach().cpu().numpy() if torch.is_tensor(feed[t]) else np.asarray(feed[t])
                    fed[t] = vk.upload(v.astype(env[t].dtype).reshape(env[t].shape))
        for t in op["outputs"]:
            env[t] = vk.array(ir.tshape(t), _tdtype(ir, t))
        args = bind(bindable(op), params, lambda t: vk.ptr(fed.get(t, env[t])), weights.ptr)
        t0 = vk.lib.mb_vk_ts() if time_ops else -1
        fn(*args)
        if vk.lib.mb_last_error():
            vk.check(f"{op['name']} ({op['op']})")
        if time_ops:
            per_op.append((op, t0))
    t_end = vk.lib.mb_vk_ts() if time_ops else -1
    vk.sync()
    out = np.concatenate([env[t].reshape(-1).astype(np.float32) for t in ir.output_names()])
    if time_ops:
        per_op = [(op, t0, (per_op[i + 1][1] if i + 1 < len(per_op) else t_end)) for i, (op, t0) in enumerate(per_op)]
    return out, env, per_op


class RefC:
    """ModelBlaster's reference C kernels (KernelSpec.reference_impl) as a host
    library, the bit-exact oracle of the int8 ops."""

    def __init__(self, specs: dict, ops, lib: str | None = None):
        if lib is None:
            cc = os.environ.get("CC", "cc")
            if not shutil.which(cc):
                raise FileNotFoundError(f"no C compiler ({cc}) for the reference kernels; pass --ref-lib")
            d = tempfile.mkdtemp(prefix="mb_refc_")
            src = ["#include <stdint.h>", "#include <stddef.h>", "#include <math.h>"]
            src += [specs[o]["reference_impl"] for o in sorted(set(ops))]
            open(f"{d}/ref.c", "w").write("\n".join(src))
            lib = f"{d}/libmb_ref.so"
            subprocess.run([cc, "-O2", "-shared", "-fPIC", f"{d}/ref.c", "-o", lib, "-lm"], check=True)
        self.lib, self.specs = ctypes.CDLL(lib), specs

    def fn(self, op: str):
        name, params = parse_signature(self.specs[op]["signature"])
        f = getattr(self.lib, name)
        f.argtypes, f.restype = [p.ctypes_type() for p in params], None
        return f, params

    def walk(self, ir: IR) -> dict:
        """The IR on the CPU through the reference kernels -> env of numpy arrays."""
        env = {t: v.numpy().astype(_tdtype(ir, t)) for t, v in ir.input_tensors("cpu").items()}
        w = {k: _weight_array(v) for k, v in ir.weights.items()}
        for op in ir.ops:
            if op["op"] == "view":
                off, n = op.get("offset", 0), op["shape"]["n"]
                env[op["outputs"][0]] = env[op["inputs"][0]].reshape(-1)[off:off + n].reshape(ir.tshape(op["outputs"][0]))
                continue
            f, params = self.fn(op["op"])
            for t in op["outputs"]:
                env[t] = np.zeros(ir.tshape(t), _tdtype(ir, t))
            f(*bind(bindable(op), params, lambda t: env[t].ctypes.data, lambda k: w[k].ctypes.data))
        return env


def evaluate_int8(ir: IR, plan, ks: "VkKernelSet", reps: int, ref_lib: str | None, rec: dict) -> dict:
    """Bit-exactness gates of an int8 IR (see the module doc) + timing."""
    vk = ks.vk
    base = vk.mark()
    weights = Weights(ir, vk)
    weights.preload(plan)
    m = vk.mark()
    out, env, _ = run(ir, plan, vk, weights)
    gold = ir.golden_out.reshape(-1).astype(np.int32)
    got = np.concatenate([env[t].reshape(-1).astype(np.int32) for t in ir.output_names()])
    rec["golden_mismatches"] = int((got != gold).sum())
    rec["golden_max_lsb"] = int(np.abs(got - gold).max()) if got.size else 0
    vk.release(m)
    ok = rec["golden_mismatches"] == 0
    try:
        ref = RefC(ks.specs, [op["op"] for op, k in plan if k], ref_lib).walk(ir)
    except (FileNotFoundError, OSError, subprocess.CalledProcessError) as e:
        rec["isolation"] = f"skipped: {e}"
    else:
        m = vk.mark()
        _, iso, _ = run(ir, plan, vk, weights, feed=ref)
        per, n_bad = {}, 0
        for op, k in plan:
            if k is None:
                continue
            for t in op["outputs"]:
                d = int((np.asarray(iso[t]) != ref[t]).sum())
                per[op["op"]] = per.get(op["op"], 0) + d
                n_bad += d > 0
        vk.release(m)
        rec["ref_c_mismatches_by_op"] = per
        rec["n_ops_not_bit_exact"] = n_bad
        ok = ok and n_bad == 0
    walls = []
    for _ in range(reps):
        m = vk.mark()
        t0 = time.perf_counter()
        run(ir, plan, vk, weights)
        walls.append(time.perf_counter() - t0)
        vk.release(m)
    rec["model_ms"] = 1e3 * float(np.median(walls)) if walls else None
    vk.release(base)
    rec["ok"] = bool(ok)
    return rec


# end-to-end bound of an int8_mixed IR against its golden (relative to the golden's range): the fp32
# ops agree with the CPU only to fp32 rounding, and a row whose value lands that close to a rounding
# boundary of quantize_rows_s8 moves one int8 step
MIXED_GOLDEN_REL = 2e-2


def evaluate_int8_mixed(ir: IR, plan, ks: "VkKernelSet", reps: int, ref_lib: str | None, rec: dict,
                        isolation: bool = True) -> dict:
    """Gates of an int8_mixed IR: the output against the golden (MIXED_GOLDEN_REL of its range); in
    isolation on the CPU reference walk's inputs (pipeline/int8_mixed.walk), every int8 kernel equal
    to the reference C and every fp32 kernel within the fp32 per-slot bound; replay; timing."""
    vk = ks.vk
    base = vk.mark()
    weights = Weights(ir, vk)
    weights.preload(plan)
    consts = ir.const_tensors("cpu", int_dtype=torch.int32)
    m = vk.mark()
    out, env, _ = run(ir, plan, vk, weights, consts=consts)
    out = out.copy()
    g = ir.golden_out.reshape(-1).astype(np.float64)
    o = out.astype(np.float64)
    d = np.abs(o - g)
    rec.update(golden_max_abs_err=float(d.max()), golden_rel_to_range=float(d.max()) / max(float(np.abs(g).max()), 1e-12),
               golden_cosine=float(o @ g / max(np.linalg.norm(o) * np.linalg.norm(g), 1e-300)))
    ok = bool(np.isfinite(out).all() and rec["golden_rel_to_range"] <= MIXED_GOLDEN_REL)
    vk.release(m)
    if isolation:
        names = {t for op in ir.ops for t in op["inputs"] + op["outputs"]}
        ref = _int8_mixed().walk(ir, ir.input_tensors("cpu"), keep=names)
        try:
            refc = RefC(ks.specs, [op["op"] for op, k in plan if k and op["op"].endswith("_s8")], ref_lib)
        except (FileNotFoundError, OSError, subprocess.CalledProcessError) as e:
            refc, rec["isolation_s8"] = None, f"skipped: {e}"
        m = vk.mark()
        _, iso, _ = run(ir, plan, vk, weights, feed=ref, consts=consts)
        w = {k: _weight_array(v) for k, v in ir.weights.items()}
        s8_bad, f32_bad, worst, first_bad = {}, 0, 0.0, None
        for op, k in plan:
            if k is None:
                continue
            if op["op"].endswith("_s8"):
                if refc is None:
                    continue
                f, params = refc.fn(op["op"])
                ins = {t: np.ascontiguousarray(ref[t].numpy().astype(_tdtype(ir, t))) for t in op["inputs"]}
                outs = {t: np.zeros(ir.tshape(t), _tdtype(ir, t)) for t in op["outputs"]}
                f(*bind(bindable(op), params, lambda t: (outs[t] if t in outs else ins[t]).ctypes.data,
                        lambda kk: w[kk].ctypes.data))
                n = sum(int((np.asarray(iso[t]) != outs[t]).sum()) for t in op["outputs"])
                s8_bad[op["op"]] = s8_bad.get(op["op"], 0) + n
                if n and first_bad is None:
                    first_bad = f"{op['name']} ({op['op']}): {n} differ from the reference C"
                continue
            for t in op["outputs"]:
                a = torch.from_numpy(np.array(iso[t])).float()
                b = ref[t].float()
                dd = torch.where(a == b, torch.zeros_like(a), (a - b).abs())
                scale = 0.0
                for v in [b] + [ref[i] for i in op["inputs"] if i in ref and ref[i].is_floating_point()]:
                    fin = v.float()[torch.isfinite(v.float())]
                    scale = max(scale, fin.abs().max().item() if fin.numel() else 0.0)
                e = dd.max().item() / max(scale, 1e-12) if dd.numel() else 0.0
                worst = max(worst, e)
                tol = cuda_dtypes.TOLERANCE[cuda_dtypes.op_dtype(op["op"])]
                if (dd.numel() and dd.max().item() > tol["slot_rel"] * scale + tol["slot_abs"]) or \
                        not (torch.isfinite(a) == torch.isfinite(b)).all():
                    f32_bad += 1
                    if first_bad is None:
                        first_bad = f"{op['name']} ({op['op']}) rel_err={e:.2e}"
        vk.release(m)
        rec.update(ref_c_mismatches_by_op=s8_bad, n_s8_ops_not_bit_exact=sum(1 for v in s8_bad.values() if v),
                   worst_fp32_intermediate_rel_err=worst, n_fp32_ops_over_slot_bound=f32_bad,
                   first_divergent_op=first_bad)
        ok = ok and f32_bad == 0 and not any(s8_bad.values())
    walls = []
    for _ in range(reps):
        m = vk.mark()
        t0 = time.perf_counter()
        run(ir, plan, vk, weights, consts=consts)
        walls.append(time.perf_counter() - t0)
        vk.release(m)
    rec["model_ms"] = 1e3 * float(np.median(walls)) if walls else None
    m = vk.mark()
    gr = VkGraph(ir, plan, vk, weights, consts=consts)
    ins = [t.numpy() for t in ir.input_tensors("cpu").values()]
    o2 = np.concatenate([x.reshape(-1).astype(np.float32) for x in gr(*ins)])
    rec["replay_equal"] = bool(np.array_equal(o2, out))
    walls = []
    for _ in range(reps):
        t0 = time.perf_counter()
        gr(*ins)
        walls.append(time.perf_counter() - t0)
    rec["model_ms_replay"] = 1e3 * float(np.median(walls)) if walls else None
    vk.release(m)
    vk.release(base)
    rec["ok"] = bool(ok and rec["replay_equal"])
    return rec


class VkGraph:
    """An IR recorded once as a replayable Vulkan graph (mb_vk_capture_*), the
    deployment path: every tensor has a fixed buffer, ``__call__`` writes the
    inputs into theirs, resubmits the recorded command buffers and returns the
    outputs -- no per-op Python or launch work. Lives in the caller's arena."""

    def __init__(self, ir: IR, plan, vk: VkBackend, weights: Weights, consts=None, bound=None):
        """bound: {input name: device array} -- inputs read in place from another
        graph's output buffer (chaining graphs without a host copy); __call__
        then takes only the remaining inputs, in order."""
        self.ir, self.vk = ir, vk
        consts = consts if consts is not None else ir.const_tensors("cpu", int_dtype=torch.int32)
        run(ir, plan, vk, weights, consts=consts, bound=bound)   # warm-up: pipelines built, scratch at full size
        self.graph = vk.lib.mb_vk_capture_begin()
        _, self.env, _ = run(ir, plan, vk, weights, consts=consts, bound=bound)
        g = vk.lib.mb_vk_capture_end()
        vk.check("capture")
        assert g == self.graph
        self.inputs = [t for t in ir.input_tensors("cpu") if not (bound and t in bound)]
        self.outputs = ir.output_names()

    def __call__(self, *args, **kw) -> list[np.ndarray]:
        for name, v in list(zip(self.inputs, args)) + list(kw.items()):
            dst = self.env[name]
            dst[...] = np.asarray(v, dtype=dst.dtype).reshape(dst.shape)
        self.vk.replay(self.graph)
        return [self.env[t] for t in self.outputs]


def evaluate(d: str, ks: VkKernelSet, reps: int, isolation: bool = True, ref_lib: str | None = None) -> dict:
    vk = ks.vk
    ir = load_ir(d)
    plan, missing = build_plan(ir, ks)
    rec = {"ir": ir.name, "backend": "vulkan", "device": vk.device, "dtype": "int8" if is_int8(ir) else ir.dtype,
           "n_kernel_ops": sum(1 for _, k in plan if k), "missing": missing}
    if missing:
        rec["ok"] = False
        return rec
    if is_int8(ir):
        return evaluate_int8(ir, plan, ks, reps, ref_lib, rec)
    if is_int8_mixed(ir):
        rec["dtype"] = "int8_mixed"
        return evaluate_int8_mixed(ir, plan, ks, reps, ref_lib, rec, isolation=isolation)
    if ir.dtype != "fp32":
        rec.update(ok=False, why=f"{ir.dtype} IR: the Vulkan backend runs fp32 IRs")
        return rec
    base = vk.mark()
    weights = Weights(ir, vk)
    weights.preload(plan)
    consts = ir.const_tensors("cpu", int_dtype=torch.int32)
    m = vk.mark()
    t0 = time.perf_counter()
    out, env, _ = run(ir, plan, vk, weights, consts=consts)
    rec["first_run_s"] = round(time.perf_counter() - t0, 3)
    rec.update(compare(torch.from_numpy(out), ir.golden_out, ir.dtype, sol=sol_for(ir)))
    out_copy = out.copy()
    vk.release(m)
    if isolation:
        with torch.no_grad():
            _, ref_env = walk_torch.walk(ir, dev="cpu")
        m = vk.mark()
        _, iso_env, _ = run(ir, plan, vk, weights, feed=ref_env, consts=consts)
        ref_out = torch.cat([ref_env[t].reshape(-1).float() for t in ir.output_names()]).numpy()
        rec["output_vs_ref_walk"] = compare(torch.from_numpy(out_copy), ref_out, ir.dtype)["rel_to_range"]
        worst, first_bad, by_kind, n_bad = 0.0, None, {}, 0
        for op, k in plan:
            if k is None:
                continue
            for t in op["outputs"]:
                if t not in ref_env:
                    continue
                a = torch.from_numpy(np.array(iso_env[t])).float()
                b = ref_env[t].float()
                dd = torch.where(a == b, torch.zeros_like(a), (a - b).abs())
                scale = 0.0
                for v in [b] + [ref_env[i] for i in op["inputs"] if i in ref_env]:
                    fin = v.float()[torch.isfinite(v.float())]
                    scale = max(scale, fin.abs().max().item() if fin.numel() else 0.0)
                e = dd.max().item() / max(scale, 1e-12) if dd.numel() else 0.0
                worst = max(worst, e)
                by_kind[op["op"]] = max(by_kind.get(op["op"], 0.0), e)
                tol = cuda_dtypes.TOLERANCE[cuda_dtypes.op_dtype(op["op"])]
                bad = dd.numel() and dd.max().item() > tol["slot_rel"] * scale + tol["slot_abs"]
                if bad or not (torch.isfinite(a) == torch.isfinite(b)).all():
                    n_bad += 1
                    if first_bad is None:
                        first_bad = f"{op['name']} ({op['op']}) rel_err={e:.2e}"
        vk.release(m)
        rec["worst_intermediate_rel_err"] = worst
        rec["n_ops_over_slot_bound"] = n_bad
        rec["first_divergent_op"] = first_bad
        rec["intermediate_rel_err_by_op"] = {k: float(f"{v:.3g}") for k, v in sorted(by_kind.items())}
        rec["ok"] = bool(rec["ok"] and n_bad == 0)
    # timing: host wall clock of whole passes (launch + GPU), then per-op GPU time
    for _ in range(2):
        m = vk.mark()
        run(ir, plan, vk, weights, consts=consts)
        vk.release(m)
    walls = []
    for _ in range(reps):
        m = vk.mark()
        t0 = time.perf_counter()
        run(ir, plan, vk, weights, consts=consts)
        walls.append(time.perf_counter() - t0)
        vk.release(m)
    rec["model_ms"] = 1e3 * float(np.median(walls))
    # as deployed: the walk captured once and replayed (inputs rewritten in place)
    m = vk.mark()
    g = VkGraph(ir, plan, vk, weights, consts=consts)
    ins = [t.numpy() for t in ir.input_tensors("cpu").values()]
    outs = g(*ins)
    rep = compare(torch.from_numpy(np.concatenate([o.reshape(-1).astype(np.float32) for o in outs])),
                  ir.golden_out, ir.dtype)
    walls = []
    for _ in range(reps):
        t0 = time.perf_counter()
        g(*ins)
        walls.append(time.perf_counter() - t0)
    vk.release(m)
    rec["replay_ok"] = rep["ok"]
    rec["model_ms_replay"] = 1e3 * float(np.median(walls))
    rec["ok"] = bool(rec["ok"] and rep["ok"])
    vk.lib.mb_vk_timestamps(1)
    vk.lib.mb_vk_ts_reset()
    m = vk.mark()
    _, _, per_op = run(ir, plan, vk, weights, consts=consts, time_ops=True)
    vk.release(m)
    vk.lib.mb_vk_timestamps(0)
    by_type, gpu = {}, 0.0
    for op, a, b in per_op:
        dt = (vk.lib.mb_vk_ts_ns(b) - vk.lib.mb_vk_ts_ns(a)) / 1e6
        by_type[op["op"]] = by_type.get(op["op"], 0.0) + dt
        gpu += dt
    rec["gpu_ms"] = gpu
    rec["ms_by_op_type"] = {k: round(v, 4) for k, v in sorted(by_type.items(), key=lambda kv: -kv[1])}
    vk.release(base)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("irs", nargs="+")
    ap.add_argument("--specs", required=True)
    ap.add_argument("--lib", default=None, help="libmb_vk.so (default $MB_VK_LIB or vulkan/build/)")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--no-isolation", action="store_true",
                    help="skip the per-kernel check against walk_torch (saves the CPU reference walk)")
    ap.add_argument("--ref-lib", default=None, help="prebuilt reference-C library for the int8 checks")
    ap.add_argument("--json")
    a = ap.parse_args()
    vk = VkBackend(a.lib)
    print(f"# {vk.info}", flush=True)
    ks = VkKernelSet(json.load(open(a.specs)), vk)
    results = []
    for d in a.irs:
        try:
            r = evaluate(d, ks, a.reps, isolation=not a.no_isolation, ref_lib=a.ref_lib)
        except Exception as e:
            r = {"ir": d, "backend": "vulkan", "ok": False, "error": f"{type(e).__name__}: {e}"}
        results.append(r)
        print(json.dumps(r), flush=True)
    if a.json:
        json.dump(results, open(a.json, "w"), indent=1)
    sys.exit(0 if all(r.get("ok") for r in results) else 1)


if __name__ == "__main__":
    main()
