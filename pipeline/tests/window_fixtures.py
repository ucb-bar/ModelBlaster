"""Fixtures for the region-window tests: synthetic chain / recurrence / dense-block IRs (pure graph dicts, no weights needed for
the structural tests) and ``ir_dir_from_graph`` (a real IR dir with synthetic weights for a final graph.json of the flow, so
``export_problem`` can be run on 34 / 15 without their trained weights: the walk only needs the right shapes).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

H = 16


def _lin(name, inp, out, m=2, k=H, n=H):
    return {"name": name, "op": "linear_f16", "inputs": [inp], "outputs": [out], "weight": f"{name}.weight", "bias": f"{name}.bias",
            "shape": {"M": m, "K": k, "N": n}}


def _act(name, inp, out, op="tanh_f16", n=2 * H):
    return {"name": name, "op": op, "inputs": [inp], "outputs": [out], "shape": {"n": n}}


def _finish(ops, tensors, inp, out):
    for i, o in enumerate(ops):
        o["dispatch_id"], o["depends_on"] = i, [i - 1] if i else []
    return {"name": "synthetic", "version": 1, "quant": "fp16", "input": {"tensor": inp}, "output": {"tensors": [out], "tensor": out},
            "tensors": {t: {"shape": s, "dtype": "f16", "quant": None} for t, s in tensors.items()}, "ops": ops,
            "dispatches": list(range(len(ops)))}


def mlp_chain(n_layers: int = 12) -> dict:
    """x -> (linear -> relu) x n_layers -> out: every cut has exactly one tensor crossing (a pure chain)."""
    ops, ts, prev = [], {"x": [2, H]}, "x"
    for i in range(n_layers):
        ops.append(_lin(f"fc{i}", prev, f"fc{i}_o"))
        ops.append(_act(f"relu{i}", f"fc{i}_o", f"relu{i}_o", "relu_f16"))
        ts[f"fc{i}_o"] = ts[f"relu{i}_o"] = [2, H]
        prev = f"relu{i}_o"
    return _finish(ops, ts, "x", prev)


def rnn_unrolled(steps: int = 24, fused: bool = True) -> dict:
    """The 34_VanillaRNNHidden shape: per step view(x slice) -> cat2(x_t, h) -> linear+tanh (fused) -> linear (output head); every
    head output is kept (stacked at the end by a two-level tree of catn ops of at most 8 inputs, as 34's 37 catn are), so the
    head outputs stay live across any cut. ``steps`` a multiple of 8, <= 64. ``fused=False``: linear and tanh are separate ops (no fused kinds,
    so no cached stage sources are needed by an exported region)."""
    assert steps % 8 == 0 and 8 <= steps <= 64, "steps: a multiple of 8 up to 64"
    ops, ts = [], {"x": [steps, 2, H], "h0": [2, H // 2]}
    ops.append({"name": "to", "op": "view", "inputs": ["h0"], "outputs": ["to"], "shape": {"n": 2 * (H // 2)}})
    ts["to"] = [2, H // 2]
    h = "to"
    outs = []
    for t in range(steps):
        ops.append({"name": f"getitem_{t}", "op": "view", "inputs": ["x"], "outputs": [f"xt{t}"], "shape": {"n": 2 * H}, "offset": t * 2 * H})
        ops.append({"name": f"cat_{t}", "op": "cat2_c1_f16", "inputs": [f"xt{t}", h], "outputs": [f"cat{t}"],
                    "shape": {"N": 2, "H": 1, "W": 1, "C_inputs": [H, H // 2], "C_total": H + H // 2}})
        if fused:
            ops.append({"name": f"i2h{t}+tanh{t}", "op": "fused__linear_f16__tanh_f16", "inputs": [f"cat{t}"], "outputs": [f"tanh{t}"],
                        "internal": [f"i2h{t}_o"], "shape": {"a": {"M": 2, "K": H + H // 2, "N": H // 2}, "b": {"n": H}},
                        "sub_ops": [dict(_lin(f"i2h{t}", f"cat{t}", f"i2h{t}_o", 2, H + H // 2, H // 2)),
                                    dict(_act(f"tanh{t}", f"i2h{t}_o", f"tanh{t}", n=H))]})
        else:
            ops.append(_lin(f"i2h{t}", f"cat{t}", f"i2h{t}_o", 2, H + H // 2, H // 2))
            ops.append(_act(f"tanh{t}", f"i2h{t}_o", f"tanh{t}", n=H))
        ops.append(_lin(f"h2o{t}", f"tanh{t}", f"h2o{t}_o", 2, H // 2, 4))
        for n_, s_ in ((f"xt{t}", [2, H]), (f"cat{t}", [2, H + H // 2]), (f"i2h{t}_o", [2, H // 2]), (f"tanh{t}", [2, H // 2]), (f"h2o{t}_o", [2, 4])):
            ts[n_] = s_
        h = f"tanh{t}"
        outs.append(f"h2o{t}_o")

    def stack(name, ins, c):
        ops.append({"name": name, "op": "catn_f16", "inputs": list(ins), "outputs": [name + "_o"],
                    "shape": {"n_in": len(ins), "outer": 1, "inner": 8, **{f"c{i}": (c if i < len(ins) else 0) for i in range(8)}}})
        ts[name + "_o"] = [len(ins) * c * 8]
        return name + "_o"

    groups = [stack(f"stack_g{i}", outs[i:i + 8], 1) for i in range(0, len(outs), 8)]
    last = stack("stack", groups, 8)
    g = _finish(ops, ts, "x", last)
    nx = steps * 2 * H
    g["input"] = {"tensor": "x", "packed_inputs": [{"name": "x", "offset": 0, "size": nx}, {"name": "h0", "offset": nx, "size": 2 * (H // 2)}]}
    return g


def dense_blocks(blocks=(4, 5), growth: int = 4) -> dict:
    """DenseNet-like: per layer conv -> view (alias read by EVERY later concat of the block) and a concat of all earlier layers;
    blocks are joined by a transition (1x1 conv): the cut between blocks has ONE tensor crossing, the cuts inside a block have
    the block's views crossing (alias outputs: illegal window boundaries)."""
    ops, ts = [], {"x": [1, growth, 4, 4]}
    prev, k = "x", 0
    for b, nl in enumerate(blocks):
        feats = [prev]
        for l in range(nl):
            inp = feats[0] if len(feats) == 1 else f"cat{b}_{l}"
            if len(feats) > 1:
                ops.append({"name": f"cat{b}_{l}", "op": "catn_f16", "inputs": list(feats), "outputs": [f"cat{b}_{l}"],
                            "shape": {"n_in": len(feats), "outer": 1, "inner": 16, **{f"c{i}": (growth if i < len(feats) else 0) for i in range(8)}}})
                ts[f"cat{b}_{l}"] = [1, growth * len(feats), 4, 4]
            ops.append({"name": f"conv{b}_{l}", "op": "conv2d_f16", "inputs": [inp], "outputs": [f"conv{b}_{l}"], "weight": f"conv{b}_{l}.weight",
                        "bias": None, "shape": {"N": 1, "IC": growth * len(feats), "IH": 4, "IW": 4, "OC": growth, "OH": 4, "OW": 4, "KH": 3,
                                                "KW": 3, "SH": 1, "SW": 1, "PH": 1, "PW": 1, "DH": 1, "DW": 1}})
            ops.append({"name": f"view{b}_{l}", "op": "view", "inputs": [f"conv{b}_{l}"], "outputs": [f"v{b}_{l}"], "shape": {"n": growth * 16}})
            ts[f"conv{b}_{l}"] = ts[f"v{b}_{l}"] = [1, growth, 4, 4]
            feats.append(f"v{b}_{l}")
        ops.append({"name": f"tcat{b}", "op": "catn_f16", "inputs": list(feats), "outputs": [f"tcat{b}"],
                    "shape": {"n_in": len(feats), "outer": 1, "inner": 16, **{f"c{i}": (growth if i < len(feats) else 0) for i in range(8)}}})
        ts[f"tcat{b}"] = [1, growth * len(feats), 4, 4]
        ops.append({"name": f"trans{b}", "op": "conv2d_f16", "inputs": [f"tcat{b}"], "outputs": [f"trans{b}"], "weight": f"trans{b}.weight",
                    "bias": None, "shape": {"N": 1, "IC": growth * len(feats), "IH": 4, "IW": 4, "OC": growth, "OH": 4, "OW": 4, "KH": 1, "KW": 1,
                                            "SH": 1, "SW": 1, "PH": 0, "PW": 0, "DH": 1, "DW": 1}})
        ts[f"trans{b}"] = [1, growth, 4, 4]
        prev = f"trans{b}"
    return _finish(ops, ts, "x", prev)


# ---------------------------------------------------------------------------------------------- real final graphs
def synthetic_weights(g: dict, seed: int = 0) -> dict:
    """{key: array} with the right SHAPE for every weight key of a final graph.json (linear / conv2d / batchnorm2d /
    layernorm-like vectors); values are small random: the exporter's walk needs shapes, not trained values."""
    r = np.random.default_rng(seed)
    shapes: dict = {}

    def put(key, shape):
        if isinstance(key, str) and key:
            shapes.setdefault(key, tuple(int(x) for x in shape))

    def leaves(o):
        yield o
        for s in o.get("sub_ops", []):
            yield from leaves(s)

    for o in g["ops"]:
        for s in leaves(o):
            k = s.get("op", "")
            sh = s.get("shape", {})
            if k.startswith("linear"):
                put(s.get("weight"), (sh["N"], sh["K"]))
                put(s.get("bias"), (sh["N"],))
            elif k.startswith("conv2d"):
                put(s.get("weight"), (sh["OC"], sh["IC"], sh["KH"], sh["KW"]))
                put(s.get("bias"), (sh["OC"],))
            elif k.startswith("batchnorm2d"):
                put(s.get("weight"), (sh["C"],))
                put(s.get("bias"), (sh["C"],))
            else:
                for f in ("weight", "bias"):
                    if isinstance(s.get(f), str):
                        raise KeyError(f"no shape rule for {f} of {k}")
    return {k: (r.standard_normal(v) * 0.05).astype(np.float16) for k, v in shapes.items()}


def ir_dir_from_graph(d: Path, graph: dict, seed: int = 0) -> Path:
    """An IR dir (graph.json + synthetic weights.npz + io.npz with the torch walk's golden output) for ``graph``."""
    import torch
    from modelblaster.pipeline.kb_export import _harness
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    (d / "graph.json").write_text(json.dumps(graph))
    np.savez(d / "weights.npz", **synthetic_weights(graph, seed))
    inp = graph["input"]
    nin = int(sum(p["size"] for p in inp["packed_inputs"])) if inp.get("packed_inputs") else int(np.prod(graph["tensors"][inp["tensor"]]["shape"]))
    x = (np.random.default_rng(1234).random(nin).astype(np.float16)).astype(np.float32)
    np.savez(d / "io.npz", input=x, output=np.zeros(1, np.float32))
    mbir, walk_torch = _harness()
    ir = mbir.load_ir(d)
    with torch.no_grad():
        out, _ = walk_torch.walk(ir, dev="cpu")
    np.savez(d / "io.npz", input=x, output=out.numpy().astype(np.float32).reshape(-1))
    return d
