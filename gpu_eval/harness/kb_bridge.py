"""Harness 4: turn ModelBlaster ops into KernelBlaster CUDA problems.

For each selected IR op this emits a KB-format problem directory
(init.cu + driver.cpp) whose `launch_gpu_implementation` has EXACTLY the
op's ModelBlaster KernelSpec signature, so a kernel KB optimizes drops
back into ModelBlaster's kernels.h by a rename. It then builds it the way
KB's compile server does (KB's split_files_for_compilation + KB's
cuda_env/CMakeLists.txt), runs it, and profiles it with ncu through KB's
NCU log parser.

    python kb_bridge.py --kb KB_ROOT --specs specs.json --hand-cu mb_cuda_kernels.cu \
        --out OUT IR_DIR:OP_NAME [IR_DIR:OP_NAME ...] [--seed hand|ref] [--ncu]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from mbabi import parse_signature
from mbir import load_ir

# libtorch reference per op type; `{x}`/`{w}`/... are tensor variables the
# driver declares, scalar shape params are substituted by name.
TORCH_REF = {
    "relu": "torch::relu({in0})",
    "relu6": "torch::clamp({in0}, 0.0, 6.0)",
    "silu": "torch::silu({in0})",
    "sigmoid": "torch::sigmoid({in0})",
    "elu": "torch::elu({in0}, {alpha})",
    "add": "{in0} + {in1}",
    "batchnorm2d": "{in0} * {weight}.view({{1, -1, 1, 1}}) + {bias}.view({{1, -1, 1, 1}})",
    # explicit vectors: a braced {0, 0} padding also matches the string_view overload
    "conv2d": "torch::conv2d({in0}, {weight}, {bias}, std::vector<int64_t>{{{SH}, {SW}}}, std::vector<int64_t>{{{PH}, {PW}}})",
    "conv2d_dw": "torch::conv2d({in0}, {weight}, {bias}, std::vector<int64_t>{{{SH}, {SW}}}, std::vector<int64_t>{{{PH}, {PW}}}, std::vector<int64_t>{{1, 1}}, {C})",
    "maxpool2d": "torch::max_pool2d({in0}, {{{KH}, {KW}}}, {{{SH}, {SW}}}, {{{PH}, {PW}}}, {{{DH}, {DW}}})",
    "linear": "torch::linear({in0}.reshape({{{M}, {K}}}), {weight}, {bias})",
    "matmul": "torch::matmul({in0}, {in1})",
    "cat2_c1": "torch::cat({{{in0}, {in1}}}, 1)",
    "cat3_c1": "torch::cat({{{in0}, {in1}, {in2}}}, 1)",
    "cat4_c1": "torch::cat({{{in0}, {in1}, {in2}, {in3}}}, 1)",
    "upsample_nearest": "torch::upsample_nearest2d({in0}, std::vector<int64_t>{{{IH} * {scale}, {IW} * {scale}}})",
    "avgpool2d": "torch::avg_pool2d({in0}, {{{KH}, {KW}}}, {{{SH}, {SW}}}, {{{PH}, {PW}}}, false, true)",
    "adaptive_avg_pool2d": "{in0}.mean(std::vector<int64_t>{{2, 3}}, true)",
    "softmax": "torch::softmax({in0}.reshape({{{M}, {K}}}), 1)",
    "layer_norm": "torch::layer_norm({in0}.reshape({{{M}, {K}}}), std::vector<int64_t>{{{K}}}, {gamma_key}, {beta_key}, {eps})",
    "leaky_relu": "torch::leaky_relu({in0}, {negative_slope})",
    "hardtanh": "torch::clamp({in0}, {min_val}, {max_val})",
    "softplus": "torch::softplus({in0})",
    "conv3d": "torch::conv3d({in0}, {weight}, {bias}, std::vector<int64_t>{{{SD}, {SH}, {SW}}}, std::vector<int64_t>{{{PD}, {PH}, {PW}}}, std::vector<int64_t>{{{DD}, {DH}, {DW}}}, {G})",
    "conv_transpose3d": "torch::conv_transpose3d({in0}, {weight}, {bias}, std::vector<int64_t>{{{SD}, {SH}, {SW}}}, std::vector<int64_t>{{{PD}, {PH}, {PW}}}, std::vector<int64_t>{{{OPD}, {OPH}, {OPW}}}, {G}, std::vector<int64_t>{{{DD}, {DH}, {DW}}})",
    "maxpool3d": "torch::max_pool3d({in0}, std::vector<int64_t>{{{KD}, {KH}, {KW}}}, std::vector<int64_t>{{{SD}, {SH}, {SW}}}, std::vector<int64_t>{{{PD}, {PH}, {PW}}}, std::vector<int64_t>{{{DD}, {DH}, {DW}}})",
    "avgpool3d": "torch::avg_pool3d({in0}, std::vector<int64_t>{{{KD}, {KH}, {KW}}}, std::vector<int64_t>{{{SD}, {SH}, {SW}}}, std::vector<int64_t>{{{PD}, {PH}, {PW}}}, false, true, {KD} * {KH} * {KW})",
    "group_norm": "torch::group_norm({in0}.reshape({{{N}, {C}, {HW}}}), {G}, {gamma_key}, {beta_key}, {eps})",
}


def _kb_split():
    """KB's own file-split, or a faithful copy if its deps are missing here."""
    try:
        from kernelblaster.servers.compile import write_compilation_files  # noqa
        return write_compilation_files, "kernelblaster.servers.compile"
    except Exception as e:  # pydantic/fastapi/config not installed on this box
        reason = f"{type(e).__name__}: {e}"

    def write_compilation_files(work_dir: Path, main_file: str, cuda_file: str):
        main = Path(main_file).read_text()
        cuda = Path(cuda_file).read_text()
        hdrs = re.findall(r"(void launch_gpu_implementation\(.*?\);)", main, flags=re.DOTALL)
        assert len(hdrs) == 1, hdrs
        main = main.replace(hdrs[0], "")
        needs_torch = bool(re.search(r"\b(torch|c10|at)::", cuda) or re.search(r"\b(torch|c10|at)::", hdrs[0]))
        inc = "#include <torch/torch.h>\n" if needs_torch else "#ifndef __CUDACC__\n#include <torch/torch.h>\n#endif\n"
        (work_dir / "cuda_model.cuh").write_text("#include <cstdint>\n" + inc + hdrs[0] + "\n")
        (work_dir / "main.cpp").write_text('#include "cuda_model.cuh"\n' + main)
        cuda = cuda.replace("inline void launch_gpu_implementation", "void launch_gpu_implementation").replace('extern "C"', "")
        (work_dir / "cuda_model.cu").write_text('#include "cuda_model.cuh"\n' + cuda)
    return write_compilation_files, f"vendored copy ({reason})"


def _kb_ncu_parser():
    try:
        from kernelblaster.agents.utils.parsing import get_elapsed_cycles_ncu_log  # noqa
        return get_elapsed_cycles_ncu_log, "kernelblaster.agents.utils.parsing"
    except Exception as e:
        reason = f"{type(e).__name__}: {e}"

    def get_elapsed_cycles_ncu_log(log: str) -> int:
        for pat in (r"Elapsed Cycles\s+\w+\s+(\d[\d,]*)", r"Elapsed Cycles[,\s:]+(?:cycle[,\s]+)?(\d[\d,]*)"):
            m = re.search(pat, log, re.I | re.M)
            if m:
                return int(m.group(1).replace(",", ""))
        raise RuntimeError("no Elapsed Cycles in ncu log")
    return get_elapsed_cycles_ncu_log, f"vendored copy ({reason})"


def find_names(cu_path: Path) -> list[str]:
    """__global__ kernel names in a candidate, via KB's own scanner if present."""
    try:
        from kernelblaster.agents.utils.parsing import find_kernel_names  # noqa
        return list(find_kernel_names(cu_path))
    except ImportError:
        return re.findall(r"__global__\s+void\s+(\w+)", Path(cu_path).read_text())


# ---- generic / level3 ops: rule(op, sh) -> (setup lines, [ref expr per output])
_E1_REF = {"neg": "-{x}", "exp": "torch::exp({x})", "log": "torch::log({x})",
           "sqrt": "torch::sqrt({x})", "rsqrt": "torch::rsqrt({x})", "abs": "torch::abs({x})",
           "tanh": "torch::tanh({x})", "sigmoid": "torch::sigmoid({x})", "relu": "torch::relu({x})",
           "sin": "torch::sin({x})", "cos": "torch::cos({x})", "erf": "torch::erf({x})",
           "gelu": "torch::gelu({x})", "gelu_tanh": "torch::gelu({x}, \"tanh\")",
           "silu": "torch::silu({x})", "clamp": "torch::clamp({x}, {p0}, {p1})",
           "reciprocal": "torch::reciprocal({x})", "square": "torch::square({x})",
           "softplus": "torch::softplus({x})", "leaky_relu": "torch::leaky_relu({x}, {p0})",
           "elu": "torch::elu({x}, {p0})", "hardswish": "torch::hardswish({x})",
           "hardsigmoid": "torch::hardsigmoid({x})", "log1p": "torch::log1p({x})",
           "expm1": "torch::expm1({x})", "sign": "torch::sign({x})", "floor": "torch::floor({x})",
           "ceil": "torch::ceil({x})", "round": "torch::round({x})",
           "logical_not": "({x} == 0).to(torch::kFloat32)", "copy": "{x}.clone()",
           "mish": "torch::mish({x})"}
_E2_REF = {"add": "({a} + {alpha} * {b})", "sub": "({a} - {alpha} * {b})", "mul": "({a} * {b})",
           "div": "({a} / {b})", "max": "torch::maximum({a}, {b})", "min": "torch::minimum({a}, {b})",
           "pow": "torch::pow({a}, {b})",
           "masked_fill": "torch::where({b} != 0, torch::full_like({a}, {alpha}), {a})",
           "eq": "({a} == {b})", "ne": "({a} != {b})", "lt": "({a} < {b})", "le": "({a} <= {b})",
           "gt": "({a} > {b})", "ge": "({a} >= {b})"}
_ES_REF = {"add": "({x} + {c})", "sub": "({x} - {c})", "rsub": "({c} - {x})", "mul": "({x} * {c})",
           "div": "({x} / {c})", "rdiv": "({c} / {x})", "pow": "torch::pow({x}, {c})",
           "rpow": "torch::pow(torch::full_like({x}, {c}), {x})",
           "max": "torch::clamp_min({x}, {c})", "min": "torch::clamp_max({x}, {c})",
           "eq": "({x} == {c})", "ne": "({x} != {c})", "lt": "({x} < {c})", "le": "({x} <= {c})",
           "gt": "({x} > {c})", "ge": "({x} >= {c})"}
_RED_REF = {"sum": "{x}.sum(1)", "mean": "{x}.mean(1)", "max": "std::get<0>({x}.max(1))",
            "min": "std::get<0>({x}.min(1))", "prod": "{x}.prod(1)", "sumsq": "({x} * {x}).sum(1)",
            "l2": "{x}.norm(2, 1)", "argmax": "{x}.argmax(1)", "argmin": "{x}.argmin(1)",
            "var": "{x}.var(1, false)", "var_unbiased": "{x}.var(1, true)"}


def _f(v) -> str:
    v = float(v)
    if v == float("inf"):
        return "INFINITY"
    if v == float("-inf"):
        return "-INFINITY"
    return repr(v) + "f"


def _vec(xs) -> str:
    return "std::vector<int64_t>{" + ", ".join(str(int(x)) for x in xs) + "}"


def _r6(sh, p):
    return [sh[f"{p}{i}"] for i in range(6)]


def _strided_index(sh, p="s") -> str:
    """libtorch expression for off + sum_k i_k * s_k over [d0..d5] (negative
    strides allowed, unlike as_strided)."""
    d = _r6(sh, "d")
    terms = [f"(torch::arange({n}, opt.dtype(torch::kLong)) * {st}).reshape("
             f"{_vec([n if j == k else 1 for j in range(6)])})"
             for k, (n, st) in enumerate(zip(d, _r6(sh, p)))]
    return f"({sh.get('off', 0)} + " + " + ".join(terms) + ")"


def _ref_generic(op, sh):
    k = op["op"]
    if k == "strided_copy":
        return [], [f"in0.reshape(-1).index({{{_strided_index(sh)}}})"]
    if k == "strided_scatter":
        pre = ["    auto y = in0.reshape(-1).clone();",
               f"    y.index_put_({{{_strided_index(sh)}.reshape(-1)}}, in1.reshape(-1));"]
        return pre, ["y"]
    if k == "gather":
        return [], ["torch::where(in1 >= 0, in0.reshape(-1).index({in1.clamp_min(0).to(torch::kLong)}), "
                    "torch::zeros({1}, opt))"]
    if k == "catn":
        parts = ", ".join(f"in{i}.reshape({_vec([sh['outer'], sh[f'c{i}'], sh['inner']])})"
                          for i in range(sh["n_in"]))
        return [], [f"torch::cat({{{parts}}}, 1)"]
    if k == "ewise1":
        return [], [_E1_REF[op["fn"]].format(x="in0", p0=_f(op["p0"]), p1=_f(op["p1"]))]
    if k == "ewise2":
        d = _vec(_r6(sh, "d"))
        a = f"in0.reshape(-1).as_strided({d}, {_vec(_r6(sh, 'sa'))})"
        b = f"in1.reshape(-1).as_strided({d}, {_vec(_r6(sh, 'sb'))})"
        return [], [_E2_REF[op["fn"]].format(a=a, b=b, alpha=_f(op.get("alpha", 1.0))) + ".to(torch::kFloat32)"]
    if k == "ewise_scalar":
        return [], [_ES_REF[op["fn"]].format(x="in0", c=_f(op["c"])) + ".to(torch::kFloat32)"]
    if k == "reduce":
        x = f"in0.reshape({_vec([sh['outer'], sh['R'], sh['inner']])})"
        return [], [_RED_REF[op["fn"]].format(x=x) + ".to(torch::kFloat32)"]
    if k in ("sum_dim", "mean_dim", "max_dim", "min_dim"):
        x = f"in0.reshape({_vec([sh['outer'], sh['reduce'], sh['inner']])})"
        return [], [_RED_REF[k[:-4]].format(x=x)]
    if k == "softmax_dim":
        fn = "torch::log_softmax" if op.get("log_out") else "torch::softmax"
        return [], [f"{fn}(in0.reshape({_vec([sh['outer'], sh['R'], sh['inner']])}), 1)"]
    # no bias input -> no `bias` variable in the driver
    bias = "bias" if op.get("bias") else "torch::Tensor()"
    if k == "conv2d_grouped":
        return [], [f"torch::conv2d(in0.reshape({_vec([sh['N'], sh['IC'], sh['IH'], sh['IW']])}), weight, {bias}, "
                    f"{_vec([sh['SH'], sh['SW']])}, {_vec([sh['PH'], sh['PW']])}, "
                    f"{_vec([sh['DH'], sh['DW']])}, {sh['G']})"]
    if k == "maxpool2d_ex":
        return [], [f"torch::max_pool2d(in0, {_vec([sh['KH'], sh['KW']])}, {_vec([sh['SH'], sh['SW']])}, "
                    f"{_vec([sh['PH'], sh['PW']])}, {_vec([sh['DH'], sh['DW']])}, true)"
                    f".narrow(2, 0, {sh['OH']}).narrow(3, 0, {sh['OW']})"]
    if k == "conv_transpose2d":
        opad = [max(0, sh[f"O{a}"] - ((sh[f"I{a}"] - 1) * sh[f"S{a}"] - 2 * sh[f"P{a}"]
                                      + sh[f"D{a}"] * (sh[f"K{a}"] - 1) + 1)) for a in "HW"]
        return [], [f"torch::conv_transpose2d(in0, weight, {bias}, {_vec([sh['SH'], sh['SW']])}, "
                    f"{_vec([sh['PH'], sh['PW']])}, {_vec(opad)}, {sh['G']}, {_vec([sh['DH'], sh['DW']])})"
                    f".narrow(2, 0, {sh['OH']}).narrow(3, 0, {sh['OW']})"]
    if k == "matmul_ta":   # A is [K, M]
        return [], [f"torch::matmul(in0.reshape({_vec([sh['K'], sh['M']])}).t(), in1.reshape({_vec([sh['K'], sh['N']])}))"]
    if k == "matmul_tb":   # B is [N, K]
        return [], [f"torch::matmul(in0.reshape({_vec([sh['M'], sh['K']])}), in1.reshape({_vec([sh['N'], sh['K']])}).t())"]
    if k == "bmm":
        return [], [f"torch::bmm(in0.reshape({_vec([sh['batch'], sh['M'], sh['K']])}), "
                    f"in1.reshape({_vec([sh['batch'], sh['K'], sh['N']])}))"]
    if k == "mul_scalar":
        return [], [f"in0 * {_f(op['scalar'])}"]
    if k == "mul":
        return [], ["in0 * in1"]
    if k == "gelu_exact":   # MB's gelu_exact is the tanh form
        return [], ['torch::gelu(in0, "tanh")']
    if k in ("tanh", "gelu"):
        return [], [f"torch::{k}(in0)"]
    if k in ("lstm_seq", "gru_seq"):
        B, T, I, H = sh["B"], sh["T"], sh["I"], sh["H"]
        x = f"in0.reshape({_vec([B, T, I])})"
        if sh["reverse"]:
            x += ".flip(1)"
        flip = ".flip(1)" if sh["reverse"] else ""
        if k == "lstm_seq":
            pre = [f"    auto r = at::lstm({x}, {{in1.reshape({_vec([1, B, H])}), in2.reshape({_vec([1, B, H])})}}, "
                   f"{{in3, in4, in5, torch::zeros_like(in5)}}, true, 1, 0.0, false, false, true);"]
            return pre, [f"std::get<0>(r){flip}", "std::get<1>(r)", "std::get<2>(r)"]
        pre = [f"    auto r = at::gru({x}, in1.reshape({_vec([1, B, H])}), {{in2, in3, in4, in5}}, "
               f"true, 1, 0.0, false, false, true);"]
        return pre, [f"std::get<0>(r){flip}", "std::get<1>(r)"]
    if k in ("cumsum", "cumprod"):
        return [], [f"torch::{k}(in0.reshape({_vec([sh['outer'], sh['axis'], sh['inner']])}), 1)"]
    if k in ("index_select", "gather_dim", "scatter_dim", "topk"):
        o, A, i = sh["outer"], sh["A"], sh["inner"]
        x = f"in0.reshape({_vec([o, A, i])})"
        if k == "topk":
            pre = [f"    auto r = torch::sort({x}, /*stable=*/true, 1, {'true' if sh['largest'] else 'false'});"]
            return pre, [f"std::get<0>(r).narrow(1, 0, {sh['K']})",
                         f"std::get<1>(r).narrow(1, 0, {sh['K']})"]
        J = sh["J"]

        def wrap(t):
            return f"torch::where({t} < 0, {t} + {A}, {t}).to(torch::kLong)"
        if k == "index_select":
            return [], [f"torch::index_select({x}, 1, {wrap('in1.reshape(-1)')})"]
        idx = wrap(f"in1.reshape({_vec([o, J, i])})")
        if k == "gather_dim":
            return [], [f"torch::gather({x}, 1, {idx})"]
        src = f"in2.reshape({_vec([o, J, i])})"
        return [], [f"{x}.{'scatter_add' if sh['mode'] else 'scatter'}(1, {idx}, {src})"]
    if k == "convnd":
        x = f"in0.reshape({_vec([sh['N'], sh['IC'], sh['ID'], sh['IH'], sh['IW']])})"
        w = f"in1.reshape({_vec([sh['OC'], sh['IC'] // sh['G'], sh['KD'], sh['KH'], sh['KW']])})"
        b = "in2" if len(op["inputs"]) > 2 else "torch::Tensor()"
        return [], [f"torch::conv3d({x}, {w}, {b}, {_vec([sh['SD'], sh['SH'], sh['SW']])}, "
                    f"{_vec([sh['PD'], sh['PH'], sh['PW']])}, {_vec([sh['DD'], sh['DH'], sh['DW']])}, "
                    f"{sh['G']})"]
    raise KeyError(k)


def _index_gen(op, shape, dims) -> str | None:
    """Valid runtime indices for the index ops' `index` input (float-valued,
    negatives included to test the wrap); unique per column for scatter-set,
    whose result is unspecified on duplicates."""
    if op["op"] not in ("index_select", "gather_dim", "scatter_dim"):
        return None
    o, A, J, i = shape["outer"], shape["A"], shape["J"], shape["inner"]
    if op["op"] == "scatter_dim" and not shape["mode"] and J <= A:
        return (f"(torch::argsort(torch::rand({{{o}, {A}, {i}}}, opt), /*dim=*/(int64_t)1).narrow(1, 0, {J})"
                f".to(torch::kFloat32) - {A}).reshape({{{dims}}})")
    return f"torch::randint(-{A}, {A}, {{{dims}}}, opt)"


# index of the first weight input of the recurrent ops (x, h0[, c0], w_ih, ...)
_RNN_FIRST_WEIGHT = {"lstm_seq": 3, "gru_seq": 2}
# ewise1 ops whose reference is only defined on positive inputs
_POSITIVE_IN = {"log", "sqrt", "rsqrt", "reciprocal"}


def emit_driver(op: dict, ir, sig_params) -> str:
    """libtorch driver: random inputs/weights at the IR op's shapes, torch
    reference, call launch_gpu_implementation with MB's ABI, allclose on every
    output."""
    shape = dict(op["shape"])
    decls, call, subs = [], [], {}
    ins = list(op["inputs"])
    outs = list(op["outputs"])
    W = {"weight": "weight", "scale": "weight", "bias": "bias", "gamma": "gamma_key", "beta": "beta_key"}
    n_out = 0
    for p in sig_params:
        if p.is_ptr and not p.is_const_ptr:
            outs.pop(0)
            call.append(f"out{n_out}.data_ptr<float>()")
            n_out += 1
        elif p.is_ptr:
            role = W.get(p.name)
            if role and role in op:
                key = op.get(role)
                if key is None:
                    subs[role] = "torch::Tensor()"
                    call.append("nullptr")
                    continue
                dims = ", ".join(map(str, ir.weights[key].shape))
                decls.append(f"    auto {role} = torch::randn({{{dims}}}, opt);")
                subs[role] = role
                call.append(f"{role}.data_ptr<float>()")
            elif not ins:  # variadic tail (catn in4..in7)
                call.append("nullptr")
            else:
                t = ins.pop(0)
                idx = len(op["inputs"]) - len(ins) - 1
                dims = ", ".join(map(str, ir.tshape(t)))
                if p.base_type == "int":
                    n_src = 1
                    for d in ir.tshape(op["inputs"][0]):
                        n_src *= d
                    decls.append(f"    auto in{idx} = torch::randint(-1, {n_src}, {{{dims}}}, "
                                 f"opt.dtype(torch::kInt32));")
                    call.append(f"in{idx}.data_ptr<int>()")
                elif p.name == "index" and _index_gen(op, shape, dims):
                    decls.append(f"    auto in{idx} = {_index_gen(op, shape, dims)};")
                    call.append(f"in{idx}.data_ptr<float>()")
                else:
                    pos = op["op"] == "ewise1" and op.get("fn") in _POSITIVE_IN
                    gen = f"torch::rand({{{dims}}}, opt) + 0.1" if pos else f"torch::randn({{{dims}}}, opt)"
                    if idx >= _RNN_FIRST_WEIGHT.get(op["op"], 99):
                        # PyTorch-init scale: N(0,1) weights make a 512-step
                        # recurrence chaotic, so fp32 rounding alone diverges
                        gen += f" * {1.0 / shape['H'] ** 0.5!r}"
                    decls.append(f"    auto in{idx} = {gen};")
                    call.append(f"in{idx}.data_ptr<float>()")
                subs[f"in{idx}"] = f"in{idx}"
        else:
            if re.fullmatch(r"c\d", p.name) and "C_inputs" in shape:
                v = shape["C_inputs"][int(p.name[1])]
            else:
                v = shape.get(p.name, op.get(p.name, op.get("scalar") if p.name == "s" else 1))
            call.append(_f(v) if p.ctype == "float" else str(v))
    subs.update({k: v for k, v in shape.items() if not isinstance(v, list)})
    subs.setdefault("DH", 1)
    subs.setdefault("DW", 1)
    subs["alpha"] = op.get("alpha", 1.0)
    subs["eps"] = op.get("eps", 1e-5)
    for k in ("negative_slope", "min_val", "max_val"):
        if k in op:
            subs[k] = _f(op[k])
    if op["op"] == "conv_transpose3d":  # output_padding = O - default transposed size
        for a in "DHW":
            subs[f"OP{a}"] = shape[f"O{a}"] - ((shape[f"I{a}"] - 1) * shape[f"S{a}"] - 2 * shape[f"P{a}"]
                                                + shape[f"D{a}"] * (shape[f"K{a}"] - 1) + 1)
    if op["op"] in TORCH_REF:
        pre, refs = [], [TORCH_REF[op["op"]].format(**subs)]
    else:
        pre, refs = _ref_generic(op, shape)
    checks, outdecl = [], []
    for i, (t, ref) in enumerate(zip(op["outputs"], refs)):
        dims = ", ".join(map(str, ir.tshape(t)))
        outdecl.append(f"    torch::Tensor ref{i} = ({ref}).to(torch::kFloat32).reshape({{{dims}}}).contiguous();")
        outdecl.append(f"    torch::Tensor out{i} = torch::empty({{{dims}}}, opt);")
        checks.append(f"    ok = ok && torch::allclose(out{i}, ref{i}, 1e-3, 1e-3, /*equal_nan=*/true);\n"
                      f"    err = std::max(err, torch::nan_to_num((out{i} - ref{i}).abs(), 0.0, 0.0, 0.0)"
                      f".max().item<float>());")
    proto = ", ".join(f"{p.ctype} {p.name}" for p in sig_params)
    return f"""#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include <iostream>

// Declaration for GPU implementation (ModelBlaster ABI for `{op["op"]}`)
void launch_gpu_implementation({proto});

int main() {{
    // ModelBlaster IR op: {op["name"]} ({op["op"]}) shape={json.dumps(op["shape"])}
    torch::manual_seed(0);
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
{chr(10).join(decls)}
{chr(10).join(pre)}
{chr(10).join(outdecl)}
    launch_gpu_implementation({", ".join(call)});
    torch::cuda::synchronize();
    bool ok = true;
    float err = 0.0f;
{chr(10).join(checks)}
    std::cout << (ok ? "passed" : "failed") << " max_abs_err=" << err << std::endl;
    return 0;
}}
"""


def emit_init_cu(op_type: str, specs: dict, hand_cu: str, seed_kind: str) -> str:
    name, params = parse_signature(specs[op_type]["signature"])
    plist = ", ".join(f"{p.ctype} {p.name}" for p in params)
    alist = ", ".join(p.name for p in params)
    if seed_kind == "ref":
        from ref2cuda import port
        body, _ = port(op_type, specs[op_type], fix_vla=True)
        body = body.replace(f'extern "C" void {name}(', f"static void {name}_seed(")
        fwd = f"{name}_seed({alist})"
    else:
        body = Path(hand_cu).read_text()
        fwd = f"{name}({alist})"
    return body + f"""
// KernelBlaster entry point == ModelBlaster `{name}` ABI.
void launch_gpu_implementation({plist}) {{
    {fwd};
    cudaDeviceSynchronize();
}}
"""


def build_and_run(prob: Path, kb_root: Path, arch: str, ncu: bool, split, parse_ncu) -> dict:
    work = prob / "kb_build"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(kb_root / "src/kernelblaster/servers/cuda_env", work)
    split(work, str(prob / "driver.cpp"), str(prob / "init.cu"))
    import torch
    import torch.utils
    prefix = f"{torch.utils.cmake_prefix_path}"
    # Outside KB's Docker image, libtorch_cuda's NEEDED libs (NCCL etc.) come
    # from the pip `nvidia-*` wheels; without rpath-link the linker picks up
    # an older system NCCL and fails on missing symbols.
    nv = sorted(str(p) for p in (Path(torch.__file__).parent.parent / "nvidia").glob("*/lib"))
    ldflags = " ".join(f"-Wl,-rpath-link,{p}" for p in nv)
    build = work / f"build_sm_{arch}"
    rec = {}
    for stage, cmd, cwd in [
        ("cmake", f"mkdir -p {build} && cd {build} && cmake -DCMAKE_PREFIX_PATH='{prefix}' "
                  f"-DCMAKE_EXE_LINKER_FLAGS='{ldflags}' "
                  f"-DCMAKE_BUILD_TYPE=Release -DGPU_ARCH_VERSION={arch} ..", work),
        ("make", "make -j8", build),
    ]:
        p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        if p.returncode:
            return {"ok": False, "stage": stage, "error": (p.stderr or p.stdout)[-1500:]}
    p = subprocess.run([str(build / "main")], capture_output=True, text=True, timeout=600,
                       env={**os.environ, "NVIDIA_TF32_OVERRIDE": "0"})
    rec["run_stdout"] = p.stdout.strip()[-300:]
    rec["ok"] = p.returncode == 0 and "passed" in p.stdout and "failed" not in p.stdout
    if ncu and rec["ok"]:
        # Same targeting as KB's find_kernel_names_ncu: kernels ncu sees in
        # the run, intersected with __global__ names in the candidate source
        # (the driver's own libtorch kernels run first and must be skipped).
        env = "NVIDIA_TF32_OVERRIDE=0"
        q = subprocess.run(f"{env} ncu -k invalid_magic_kernel_name_here {build / 'main'}",
                           shell=True, capture_output=True, text=True, timeout=900)
        seen = re.findall(r"\s*\d+\.\s*(\w+)", q.stdout.split("Available Kernels:")[-1])
        names = sorted(set(seen) & set(find_names(prob / "init.cu")))
        rec["ncu_kernels"] = names
        kb_metric, total, launches, logs = 0, 0, 0, []
        for k in names:
            q = subprocess.run(f"{env} ncu -k {k} --section SpeedOfLight {build / 'main'}",
                               shell=True, capture_output=True, text=True, timeout=1800)
            logs.append(q.stdout + q.stderr)
            try:
                # KB's parser returns the FIRST launch only ...
                kb_metric += parse_ncu(q.stdout)
            except Exception as e:
                rec["ncu_error"] = f"{e}: {(q.stdout + q.stderr)[-600:]}"
            # ... but an op may launch a kernel several times (cat_* = 1/input).
            every = [int(c.replace(",", "")) for c in re.findall(r"Elapsed Cycles\s+\w+\s+(\d[\d,]*)", q.stdout)]
            total += sum(every)
            launches += len(every)
        (prob / "ncu.log").write_text("\n".join(logs))
        if names and "ncu_error" not in rec:
            rec["ncu_elapsed_cycles_kb_parser"] = kb_metric
            rec["ncu_elapsed_cycles_all_launches"] = total
            rec["ncu_launches"] = launches
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+", help="IR_DIR:OP_NAME")
    ap.add_argument("--kb", required=True, help="KernelBlaster checkout (for cuda_env + split code)")
    ap.add_argument("--specs", required=True)
    ap.add_argument("--hand-cu", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", default="hand", choices=["hand", "ref"])
    ap.add_argument("--arch", default="89")
    ap.add_argument("--ncu", action="store_true")
    a = ap.parse_args()
    kb = Path(a.kb).resolve()
    sys.path.insert(0, str(kb / "src"))
    split, split_src = _kb_split()
    parse_ncu, ncu_src = _kb_ncu_parser()
    print(json.dumps({"kb_split": split_src, "kb_ncu_parser": ncu_src}), flush=True)
    specs = json.load(open(a.specs))
    results = []
    for tgt in a.targets:
        d, opname = tgt.rsplit(":", 1)
        ir = load_ir(d)
        if "@" in opname:   # NAME@INDEX: generic lowerings emit several ops per name
            opname, idx = opname.rsplit("@", 1)
            op = ir.ops[int(idx)]
            assert op["name"] == opname, (op["name"], opname)
            opname = f"{opname}_{idx}"
        else:
            op = next(o for o in ir.ops if o["name"] == opname)
        _, params = parse_signature(specs[op["op"]]["signature"])
        prob = Path(a.out) / f"{ir.name}__{re.sub(r'[^A-Za-z0-9_]', '_', opname)}__{a.seed}"
        prob.mkdir(parents=True, exist_ok=True)
        try:
            (prob / "driver.cpp").write_text(emit_driver(op, ir, params))
            (prob / "init.cu").write_text(emit_init_cu(op["op"], specs, a.hand_cu, a.seed))
            r = build_and_run(prob, kb, a.arch, a.ncu, split, parse_ncu)
        except Exception as e:
            r = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        r = {"problem": prob.name, "op": op["op"], "seed": a.seed, **r}
        results.append(r)
        print(json.dumps(r), flush=True)
    json.dump(results, open(Path(a.out) / f"kb_bridge_{a.seed}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
