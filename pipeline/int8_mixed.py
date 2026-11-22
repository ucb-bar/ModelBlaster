"""int8 extraction with fp32 islands (``extract_graph --quant int8 --int8-mixed POLICY``).

The FX int8 extractor (``extract_int8``) lowers a model only when EVERY node has
an int8 lowering, and it runs the whole graph on per-tensor int8 grids. A
transformer like SmolVLA has neither property: its graphs are full of
constants (additive -inf masks, RoPE tables, position rows), broadcasts,
slices and residual streams whose outliers no per-tensor int8 scale holds,
and the numerically fragile parts (softmax inputs, norms, the flow-matching
time embedding, the action head) lose accuracy in int8 for no speed: the
time is in the linears (~75% of a SmolVLA chunk on the Deck).

This pass therefore starts from the fp32 FX lowering (every op the generic
fp32 kernels run, ``get_attr`` constants materialised as constant tensors,
``expand`` a zero-stride ``strided_copy``) and moves the policy's ops -- the
linears -- to int8:

* ``linear`` -> ``quantize_rows_s8`` + ``linear_dq_s8``. The activation is
  quantized per ROW (per token) at run time: ``quantize_rows_s8`` writes the
  int8 rows and each row's scale (max |x'| / 127 of that row). Before that,
  each input column is multiplied by a calibrated SmoothQuant factor
  (x' = x * col_inv[k], col_inv = 1 / sm) which the weight absorbs
  (W' = W * sm, so x' W'^T = x W^T), so an outlier channel no longer sets
  every row's grid: sm_k = max|x_k|^a / max|W_:,k|^(1-a) over the
  calibration set and over every int8 reader of x (q / k / v share one
  quantized tensor and one sm), a = ``alpha`` (default 0.5). The weight is
  int8 per output channel. ``linear_dq_s8`` accumulates in int32 and writes
  fp32: ``acc * row_scale[m] * w_scale[n]`` rounded once, + fp32 bias. So
  the residual stream, RoPE, attention, softmax and the norms stay fp32 and
  never see an int8 grid.
* ops named in ``keep_fp32`` (fnmatch patterns over IR op names) stay fp32.
* constants stay the fp32 constant tensors of the fp32 lowering (no linear
  reads one in these graphs; one that did would be quantized like any
  activation, its row scales computed by quantize_rows_s8 on the device).
  Additive masks stay in the add-before-softmax fp32 domain: softmax is fp32,
  so exp(-inf) = 0 exactly as in the model (softmax_s8 would have to clip
  -inf onto an int8 grid, which is not softmax).

Calibration (``extract_int8``'s rule: representative inputs, max over the
set, ``MB_INT8_CALIB_POLICY=pNN`` for a pooled percentile) sets the
per-column maxima the smoothing factors come from; the row scales are
measured at run time, so an input unlike the calibration set cannot saturate
a row.

The resulting IR has ``quant: int8_mixed``: f32 and i8 tensors side by side.
Its golden output is the IR walked on the CPU: fp32 ops as ``walk_torch``
computes them, the int8 kinds bit-exactly as their reference C
(``TORCH_RULES`` reproduce it with exact float64 / IEEE fp32 steps; the tests
compare them with the compiled reference_impl). Walkers hold the int8
kernels to bit-exactness against the reference C and the fp32 ops to the
fp32 per-slot bound (walk_vulkan.evaluate_int8_mixed).
"""
from __future__ import annotations

import ctypes
import fnmatch
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

MIXED = "int8_mixed"
QMAX = 127.0
POLICIES = {
    # every linear -> quantize_rows_s8 + linear_dq_s8; everything else fp32
    "linear": {"int8_kinds": ["linear"], "alpha": 0.5},
}
KINDS = ("quantize_rows_s8", "linear_dq_s8")

# ------------------------------------------------------------------ kernel specs
_SPECS = [
    ("quantize_rows_s8",
     "void kernel_quantize_rows_s8(const float *input, const float *col_inv, int8_t *output, "
     "float *row_scale, int M, int K)",
     "Dynamic per-row int8 quantization of an fp32 [M, K] tensor, after a per-column factor\n"
     "(SmoothQuant smoothing; all ones for none):\n"
     "  x'[m, k]     = input[m, k] * col_inv[k]                    (fp32)\n"
     "  amax         = max_k |x'[m, k]|                            (NaN ignored)\n"
     "  row_scale[m] = amax > 0 ? amax / 127.0f : 1.0f             (fp32 division)\n"
     "  inv          = 1.0f / row_scale[m]\n"
     "  output[m, k] = clamp(roundf(x'[m, k] * inv), -127, 127)    (roundf: half away from zero;\n"
     "                                                             NaN -> -127)\n"
     "Inserted by the int8_mixed pass (pipeline/int8_mixed.py) where a linear_dq_s8 reads an\n"
     "fp32 tensor; the row scales are linear_dq_s8's.",
     "void kernel_quantize_rows_s8(const float *input, const float *col_inv, int8_t *output,\n"
     "                             float *row_scale, int M, int K) {\n"
     "    for (int m = 0; m < M; m++) {\n"
     "        const float *x = input + (size_t)m * K;\n"
     "        float amax = 0.0f;\n"
     "        for (int k = 0; k < K; k++) {\n"
     "            float v = fabsf(x[k] * col_inv[k]);\n"
     "            if (v > amax) amax = v;\n"
     "        }\n"
     "        const float s = amax > 0.0f ? amax / 127.0f : 1.0f;\n"
     "        const float inv = 1.0f / s;\n"
     "        row_scale[m] = s;\n"
     "        for (int k = 0; k < K; k++) {\n"
     "            const float v = x[k] * col_inv[k];\n"
     "            float q = roundf(v * inv);\n"
     "            if (!(q >= -127.0f)) q = -127.0f;\n"
     "            if (q > 127.0f) q = 127.0f;\n"
     "            output[(size_t)m * K + k] = (int8_t)q;\n"
     "        }\n"
     "    }\n}\n",
     [{"M": 1, "K": 16}, {"M": 5, "K": 257}]),
    ("linear_dq_s8",
     "void kernel_linear_dq_s8(const int8_t *input, const float *row_scale, const int8_t *weight, "
     "const float *bias, const float *w_scale, float *output, int M, int K, int N)",
     "int8 x int8 fully-connected layer with an fp32 (dequantized) output:\n"
     "  input int8 [M, K] with per-row scales row_scale[M] (quantize_rows_s8), weight int8 [N, K]\n"
     "  with per-output-channel scales w_scale[N], bias fp32 [N] or NULL, output fp32 [M, N]\n"
     "  acc = sum_k input[m, k] * weight[n, k]                       (int32, exact)\n"
     "  y   = (float)((double)acc * ((double)row_scale[m] * (double)w_scale[n]))\n"
     "  output[m, n] = bias ? y + bias[n] : y                         (fp32 add)\n"
     "The scale product is exact in double; the dequantized value is rounded once to fp32.",
     "void kernel_linear_dq_s8(const int8_t *input, const float *row_scale, const int8_t *weight,\n"
     "                         const float *bias, const float *w_scale, float *output, int M, int K, int N) {\n"
     "    for (int m = 0; m < M; m++) {\n"
     "        for (int n = 0; n < N; n++) {\n"
     "            int32_t acc = 0;\n"
     "            for (int k = 0; k < K; k++)\n"
     "                acc += (int32_t)input[(size_t)m * K + k] * (int32_t)weight[(size_t)n * K + k];\n"
     "            const double s = (double)row_scale[m] * (double)w_scale[n];\n"
     "            float y = (float)((double)acc * s);\n"
     "            if (bias) y = y + bias[n];\n"
     "            output[(size_t)m * N + n] = y;\n"
     "        }\n"
     "    }\n}\n",
     [{"M": 1, "K": 16, "N": 8}, {"M": 5, "K": 36, "N": 9}]),
]


def kernel_specs(kernel_spec_cls) -> dict:
    """KernelSpecs of the int8_mixed kinds (reference_kernels registers them)."""
    i8p, fp = ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_float)
    argt = {"quantize_rows_s8": [fp, fp, i8p, fp, ctypes.c_int, ctypes.c_int],
            "linear_dq_s8": [i8p, fp, i8p, fp, fp, fp, ctypes.c_int, ctypes.c_int, ctypes.c_int]}
    return {op: kernel_spec_cls(op=op, signature=sig, semantics=sem, reference_impl=impl, extra_shapes=shapes,
                                argtypes_factory=(lambda a=argt[op]: list(a)))
            for op, sig, sem, impl, shapes in _SPECS}


# ------------------------------------------------------------------ the int8 kinds on the CPU (bit-exact)
def _round_half_away(v: torch.Tensor) -> torch.Tensor:
    """C roundf / round: half away from zero (v - trunc(v) is exact)."""
    t = torch.trunc(v)
    return torch.where((v - t).abs() >= 0.5, t + torch.sign(v), t)


def quantize_rows(x: torch.Tensor, col_inv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """kernel_quantize_rows_s8 on [M, K] (IEEE fp32 steps in the reference's order)."""
    v = x.float() * col_inv.float()
    a = v.abs()
    a = torch.where(torch.isnan(a), torch.zeros_like(a), a).amax(1)      # (NaN never wins `v > amax`)
    s = torch.where(a > 0, a / torch.tensor(127.0), torch.ones_like(a))
    inv = torch.tensor(1.0) / s
    q = _round_half_away(v * inv[:, None])
    q = torch.where(q >= -QMAX, q, torch.full_like(q, -QMAX)).clamp(max=QMAX)
    return q.to(torch.int8), s


def _r_quantize_rows(ir, op, ins):
    sh = op["shape"]
    q, s = quantize_rows(ins[0].reshape(sh["M"], sh["K"]), ins[1].reshape(-1))
    return [q, s]


def _r_linear_dq(ir, op, ins):
    sh = op["shape"]
    x = ins[0].reshape(sh["M"], sh["K"]).double()
    w = torch.from_numpy(np.asarray(ir.weights[op["weight"]])).double()
    acc = x @ w.T                                   # integer-valued, |partial sums| < 2^53: exact
    ws = torch.from_numpy(np.asarray(ir.weights[op["w_scale"]], np.float32)).double()
    s = ins[1].reshape(-1, 1).double() * ws[None, :]                 # exact in double (24 x 24 bits)
    y = (acc * s).float()
    if op.get("bias"):
        y = y + torch.from_numpy(np.asarray(ir.weights[op["bias"]], np.float32))
    return [y]


TORCH_RULES = {"quantize_rows_s8": _r_quantize_rows, "linear_dq_s8": _r_linear_dq}


# ------------------------------------------------------------------ IR walk on the CPU
def _walk_torch():
    h = str(Path(__file__).resolve().parents[1] / "gpu_eval" / "harness")
    if h not in sys.path:
        sys.path.insert(0, h)
    import walk_torch  # noqa: PLC0415
    return walk_torch


def as_ir(graph: dict, weights: dict, golden_in=None, golden_out=None):
    _walk_torch()
    from mbir import IR  # noqa: PLC0415
    e = np.zeros(0, np.float32)
    return IR(graph["name"], graph, weights, e if golden_in is None else golden_in,
              e if golden_out is None else golden_out)


def walk(ir, inputs: dict, record=None, keep=None) -> dict:
    """The IR on the CPU: fp32 ops by walk_torch's rules, the int8 kinds bit-exactly
    (TORCH_RULES; other `_s8` kinds are not supported here). inputs: {name: tensor}.
    record(name, value) sees every produced tensor; keep: names kept in the returned
    env (default: the graph outputs), everything else leaves at its last use."""
    wt = _walk_torch()
    env = {k: (v if torch.is_tensor(v) else torch.from_numpy(np.asarray(v))).reshape(ir.tshape(k))
           for k, v in inputs.items()}
    env.update(ir.const_tensors("cpu"))
    keep = set(keep or ()) | set(ir.output_names())
    last = {}
    for k, op in enumerate(ir.ops):
        for t in op["inputs"]:
            last[t] = k
    with torch.no_grad():
        for k, op in enumerate(ir.ops):
            ins = [env[t] for t in op["inputs"]]
            rule = TORCH_RULES.get(op["op"])
            if rule is not None:
                outs = rule(ir, op, ins)
            else:
                fn = wt._rule(op["op"])
                if fn is None or op["op"].endswith("_s8"):
                    raise NotImplementedError(f"int8_mixed walk: no CPU rule for {op['op']}")
                outs = fn(ir, op, ins, "cpu")
            for name, val in zip(op["outputs"], outs):
                env[name] = val.reshape(ir.tshape(name))
                if record is not None:
                    record(name, env[name])
            for t in op["inputs"]:
                if t not in keep and last.get(t, -1) <= k and not ir.graph["tensors"].get(t, {}).get("const"):
                    env.pop(t, None)
    return {k: v for k, v in env.items() if k in keep}


# ------------------------------------------------------------------ the pass
def _calib_col_ranges(ir, samples: list[dict], names: dict) -> dict:
    """Per-column max |x| (finite values) of each tensor in `names` ({tensor: K}) over all the
    calibration samples, or a pooled per-column percentile under MB_INT8_CALIB_POLICY=pNN."""
    pol = os.environ.get("MB_INT8_CALIB_POLICY", "max")
    if pol != "max" and not (pol.startswith("p") and pol[1:].replace(".", "", 1).isdigit()):
        raise SystemExit(f"MB_INT8_CALIB_POLICY={pol!r}: expected 'max' or 'pNN[.N]'")
    q = None if pol == "max" else float(pol[1:]) / 100.0
    rng, pool = {}, {}

    def rec(name, v):
        if name not in names or not v.is_floating_point():
            return
        a = v.detach().abs().reshape(-1, names[name]).double()
        a = torch.where(torch.isfinite(a), a, torch.zeros_like(a))
        m = a.amax(0)
        rng[name] = m if name not in rng else torch.maximum(rng[name], m)
        if q is not None:
            pool.setdefault(name, []).append(a)

    for s in samples:
        for k in set(names) & set(s):              # graph inputs
            rec(k, torch.as_tensor(np.asarray(s[k])))
        walk(ir, s, record=rec)
    if q is not None:
        for name, chunks in pool.items():
            rng[name] = torch.minimum(rng[name], torch.quantile(torch.cat(chunks), q, dim=0))
    return rng


def resolve_policy(policy) -> dict:
    """A policy name (POLICIES), a JSON string or a dict ({"base": name, "alpha": a,
    "keep_fp32": [patterns]}) -> the full policy dict."""
    if isinstance(policy, str):
        policy = policy.strip()
        if policy.startswith("{"):
            policy = json.loads(policy)
        elif policy in POLICIES:
            policy = {"base": policy}
        else:
            raise ValueError(f"int8_mixed policy {policy!r}: expected one of {sorted(POLICIES)} or a JSON dict")
    base = policy.get("base", "linear")
    if base not in POLICIES:
        raise ValueError(f"int8_mixed policy base {base!r}: expected one of {sorted(POLICIES)}")
    p = dict(POLICIES[base])
    p.update(policy)
    p["base"] = base
    p.setdefault("keep_fp32", [])
    return p


def quantize_graph(ir, samples: list[dict], policy="linear") -> tuple[dict, dict, dict]:
    """fp32 IR (mbir.IR) + calibration samples ({input name: tensor}) -> (graph, weights, report)
    of the int8_mixed IR. The fp32 IR is not modified."""
    pol = resolve_policy(policy)
    alpha = float(pol["alpha"])
    keep = [str(p) for p in pol["keep_fp32"]]
    sel = [op for op in ir.ops if op["op"] in pol["int8_kinds"]
           and not any(fnmatch.fnmatchcase(op["name"], p) for p in keep)]
    for op in sel:
        if op["op"] != "linear":
            raise NotImplementedError(f"int8_mixed: no int8 lowering for {op['op']}")
    tensors = {k: dict(v) for k, v in ir.graph["tensors"].items()}
    weights = dict(ir.weights)
    readers: dict[str, list] = {}
    for op in sel:
        readers.setdefault(op["inputs"][0], []).append(op)
    cols = _calib_col_ranges(ir, samples, {x: rd[0]["shape"]["K"] for x, rd in readers.items()})

    # SmoothQuant factors, one per quantized tensor (shared by all of its int8 readers)
    col_inv = {}
    for x, rd in readers.items():
        K = rd[0]["shape"]["K"]
        wmax = np.zeros(K)
        for op in rd:
            wmax = np.maximum(wmax, np.abs(np.asarray(ir.weights[op["weight"]], np.float64)).max(axis=0))
        xmax = cols[x].numpy() if x in cols else np.ones(K)
        sm = np.ones(K)
        if alpha > 0:
            sm = np.maximum(xmax, 1e-5) ** alpha / np.maximum(wmax, 1e-5) ** (1.0 - alpha)
            sm = np.where(np.isfinite(sm) & (sm > 0), sm, 1.0)
        col_inv[x] = (1.0 / sm).astype(np.float32)

    sel_ids = {id(op) for op in sel}
    qname: dict[str, tuple[str, str]] = {}
    ops = []
    for op in ir.ops:
        if id(op) not in sel_ids:
            ops.append(op)
            continue
        x = op["inputs"][0]
        sh = op["shape"]
        M, K = int(sh["M"]), int(sh["K"])
        if x not in qname:
            q, s, ci = f"{x}__q8", f"{x}__q8_scale", f"{x}__col_inv"
            qname[x] = (q, s)
            weights[f"const.{ci}"] = col_inv[x]
            tensors[ci] = {"shape": [K], "dtype": "f32", "quant": None, "const": f"const.{ci}"}
            tensors[q] = {"shape": [M, K], "dtype": "i8", "quant": {"scale": "per_row", "zero_point": 0}}
            tensors[s] = {"shape": [M], "dtype": "f32", "quant": None}
            ops.append({"name": q, "op": "quantize_rows_s8", "inputs": [x, ci], "outputs": [q, s],
                        "shape": {"M": M, "K": K}})
        q, s = qname[x]
        # W' = W * sm, with sm = 1 / col_inv as stored (so x' W'^T = x W^T up to the fp32 x')
        w = np.asarray(ir.weights[op["weight"]], np.float64) / col_inv[x].astype(np.float64)[None, :]
        ws = (np.maximum(np.abs(w).max(axis=1), 1e-30) / QMAX).astype(np.float32)
        wq = _round_half_away(torch.from_numpy(w / ws.astype(np.float64)[:, None]))
        wkey, skey = f"{op['weight']}.q8", f"{op['weight']}.wscale"
        weights[wkey] = wq.clamp(-QMAX, QMAX).to(torch.int8).numpy()
        weights[skey] = ws
        new = {k: v for k, v in op.items() if k not in ("dispatch_id", "depends_on", "hardware_target")}
        new.update(op="linear_dq_s8", inputs=[q, s], weight=wkey, w_scale=skey)
        ops.append(new)
    # the fp32 weights of the linears that went to int8 leave the IR (unless something else reads them)
    used = {op.get(r) for op in ops for r in ("weight", "bias", "w_scale", "gamma_key", "beta_key")}
    gone = {op["weight"] for op in sel} - used
    weights = {k: v for k, v in weights.items() if k not in gone}
    from modelblaster.pipeline.extract_graph import _annotate_dispatches  # noqa: PLC0415
    g = dict(ir.graph)
    g.update(quant=MIXED, ops=ops, tensors=tensors)
    g["dispatches"] = _annotate_dispatches(ops)
    kinds = {}
    for op in ops:
        if op["op"] != "view":
            kinds[op["op"]] = kinds.get(op["op"], 0) + 1
    report = {"policy": pol, "n_calibration": len(samples),
              "op_kinds": dict(sorted(kinds.items(), key=lambda kv: -kv[1])),
              "n_int8_linears": len(sel),
              "n_fp32_linears": sum(1 for op in ops if op["op"] == "linear"),
              "fp32_linears": [op["name"] for op in ops if op["op"] == "linear"]}
    g["int8_mixed"] = dict(report)
    return g, weights, report


def inputs_of(ir, sample) -> dict:
    """A model sample (tensor / list, placeholder order) -> {graph input name: float32 tensor}."""
    names = [p["name"] for p in ir.graph["input"].get("packed_inputs") or [{"name": ir.graph["input"]["tensor"]}]]
    xs = list(sample) if isinstance(sample, (list, tuple)) else [sample]
    if len(xs) != len(names):
        raise ValueError(f"sample has {len(xs)} tensors, the IR {len(names)} inputs")
    return {n: torch.as_tensor(x).detach().float().reshape(ir.tshape(n)) for n, x in zip(names, xs)}


def extract_mixed(fp32_dir: str, out_dir: str, calibration_samples: list | None, policy="linear") -> dict:
    """fp32 IR dir (extract_graph --quant fp32) -> int8_mixed IR in out_dir. The golden input
    is the fp32 IR's; calibration_samples (model samples) default to it."""
    _walk_torch()
    from mbir import load_ir  # noqa: PLC0415
    ir = load_ir(fp32_dir)
    golden = dict(ir.input_tensors("cpu"))
    samples = [inputs_of(ir, s) for s in calibration_samples] if calibration_samples else [golden]
    g, w, report = quantize_graph(ir, samples, policy)
    mixed = as_ir(g, w)
    out = walk(mixed, golden)
    o = torch.cat([out[t].reshape(-1).float() for t in mixed.output_names()])
    r = torch.from_numpy(ir.golden_out.reshape(-1).astype(np.float32))
    d = o.double() - r.double()
    report["vs_fp32_golden"] = {"cosine": float(torch.nn.functional.cosine_similarity(o.double(), r.double(), dim=0)),
                                "max_abs_err": float(d.abs().max()), "ref_abs_max": float(r.abs().max())}
    g["int8_mixed"]["vs_fp32_golden"] = report["vs_fp32_golden"]
    os.makedirs(out_dir, exist_ok=True)
    Path(out_dir, "graph.json").write_text(json.dumps(g, indent=1))
    np.savez(Path(out_dir, "weights.npz"), **w)
    io = np.load(Path(fp32_dir, "io.npz"))
    np.savez(Path(out_dir, "io.npz"), input=io["input"], output=o.numpy().astype(np.float32))
    v = report["vs_fp32_golden"]
    print(f"[int8_mixed] {g['name']}: {report['n_int8_linears']} int8 linears ({report['n_fp32_linears']} "
          f"kept fp32), calibrated on {len(samples)} samples; vs the fp32 golden: cosine {v['cosine']:.6f}, "
          f"max |err| {v['max_abs_err']:.3g} (|ref| <= {v['ref_abs_max']:.3g})", flush=True)
    return report
