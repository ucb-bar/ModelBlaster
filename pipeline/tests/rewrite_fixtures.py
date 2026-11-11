"""Fixtures for the graph-rewrite tests: a tiny SqueezeNet-style fire module IR, written as a real IR dir
(graph.json / weights.npz / io.npz, golden made by the torch walker), and the known rewrites of it.

    x[1,4,6,6] -> sq 1x1 conv(4->3) -> relu -> s
    s -> e1 1x1 conv(3->2) -> relu -> e1r        s -> e3 3x3 conv(3->3, pad 1) -> relu -> e3r
    cat2_c1(e1r, e3r) -> cat_out[1,5,6,6] -> adaptive_avg_pool2d -> pooled[1,5,1,1] -> linear(5->4) -> out[1,4]

``fire_ir(..., tiny_head=True)``: the head's weights are ~1e-7 and its bias ~100, so the model output barely depends on
its input (the SqueezeNet situation: an output-only comparison passes any rewrite).

Known rewrites (``merge_expand``): the two expand convs become ONE 3x3 conv with the 1x1 filters zero-padded to 3x3 and
stacked (37 -> 21 ops in SqueezeNet, 7 -> 5 here); ``wrong`` selects a mistake: "order" (filters stacked in the wrong
order), "pad" (1x1 filter padded off-centre).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline import graph_edit as ge  # noqa: E402
from modelblaster.pipeline.kb_export import _harness  # noqa: E402

C, S, E1, E3, H, W = 4, 3, 2, 3, 6, 6


def _suffix(quant: str) -> str:
    return {"fp16": "_f16", "native": "_f16", "bf16": "_bf16"}.get(quant, "")      # "native": per-tensor dtypes, all f16 here


def _tdt(quant: str) -> str:
    return {"fp16": "f16", "native": "f16", "bf16": "bf16"}.get(quant, "f32")


def _wdt(quant: str):
    return np.float16 if quant in ("fp16", "native") else np.float32


def conv_op(name, inp, out, w, b, ic, oc, k, quant, hw=(H, W)):
    return {"name": name, "op": "conv2d" + _suffix(quant), "inputs": [inp], "outputs": [out], "weight": w, "bias": b,
            "shape": {"N": 1, "IC": ic, "IH": hw[0], "IW": hw[1], "OC": oc, "OH": hw[0], "OW": hw[1], "KH": k, "KW": k,
                      "SH": 1, "SW": 1, "PH": k // 2, "PW": k // 2, "DH": 1, "DW": 1}}


def relu_op(name, inp, out, n, quant):
    return {"name": name, "op": "relu" + _suffix(quant), "inputs": [inp], "outputs": [out], "shape": {"n": n}}


def fire_graph(quant: str = "fp32", hw=(H, W)) -> dict:
    H, W = hw
    t = _tdt(quant)
    sh = {"x": [1, C, H, W], "sq_o": [1, S, H, W], "s": [1, S, H, W], "e1_o": [1, E1, H, W], "e1r": [1, E1, H, W],
          "e3_o": [1, E3, H, W], "e3r": [1, E3, H, W], "cat_out": [1, E1 + E3, H, W], "pooled": [1, E1 + E3, 1, 1], "out": [1, 4]}
    ops = [conv_op("sq", "x", "sq_o", "sq.weight", "sq.bias", C, S, 1, quant, hw),
           relu_op("sq_relu", "sq_o", "s", S * H * W, quant),
           conv_op("e1", "s", "e1_o", "e1.weight", "e1.bias", S, E1, 1, quant, hw),
           relu_op("e1_relu", "e1_o", "e1r", E1 * H * W, quant),
           conv_op("e3", "s", "e3_o", "e3.weight", "e3.bias", S, E3, 3, quant, hw),
           relu_op("e3_relu", "e3_o", "e3r", E3 * H * W, quant),
           {"name": "cat", "op": "cat2_c1" + _suffix(quant), "inputs": ["e1r", "e3r"], "outputs": ["cat_out"],
            "shape": {"N": 1, "H": H, "W": W, "C_inputs": [E1, E3]}},
           {"name": "pool", "op": "adaptive_avg_pool2d" + _suffix(quant), "inputs": ["cat_out"], "outputs": ["pooled"],
            "shape": {"N": 1, "C": E1 + E3, "IH": H, "IW": W}},
           {"name": "fc", "op": "linear" + _suffix(quant), "inputs": ["pooled"], "outputs": ["out"], "weight": "fc.weight",
            "bias": "fc.bias", "shape": {"M": 1, "K": E1 + E3, "N": 4}}]
    for i, o in enumerate(ops):
        o["dispatch_id"], o["depends_on"] = i, [i - 1] if i else []
    g = {"name": "fire", "version": 1, "quant": quant, "input": {"tensor": "x"},
         "output": {"tensors": ["out"], "tensor": "out"},
         "tensors": {k: {"shape": v, "dtype": t, "quant": None} for k, v in sh.items()}, "ops": ops,
         "dispatches": list(range(len(ops)))}
    if quant == "native":                       # --quant native: every float weight names its dtype (docs/tensor_dtypes.md)
        g["weight_dtypes"] = {k: "f16" for k in ("sq.weight", "sq.bias", "e1.weight", "e1.bias", "e3.weight", "e3.bias", "fc.weight", "fc.bias")}
    return g


def fire_weights(quant: str = "fp32", seed: int = 0, tiny_head: bool = False) -> dict:
    r = np.random.default_rng(seed)
    dt = _wdt(quant)
    g = lambda *s: (r.standard_normal(s) * 0.4).astype(np.float32)  # noqa: E731
    w = {"sq.weight": g(S, C, 1, 1), "sq.bias": g(S), "e1.weight": g(E1, S, 1, 1), "e1.bias": g(E1),
         "e3.weight": g(E3, S, 3, 3), "e3.bias": g(E3), "fc.weight": g(4, E1 + E3), "fc.bias": g(4)}
    if tiny_head:
        w["fc.weight"] = (w["fc.weight"] * 1e-7).astype(np.float32)
        w["fc.bias"] = (100.0 + r.random(4) * 10).astype(np.float32)
    return {k: v.astype(dt) for k, v in w.items()}


def write_ir(d: Path, graph: dict, weights: dict, golden_in: np.ndarray | None = None) -> Path:
    """An IR dir for ``graph`` / ``weights``; the golden output is the torch walk's."""
    import torch
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    nin = int(np.prod(graph["tensors"][graph["input"]["tensor"]]["shape"]))
    q = graph.get("quant")
    if golden_in is None:
        golden_in = np.random.default_rng(1234).random(nin).astype(np.float32)
        if q in ("fp16", "native"):
            golden_in = golden_in.astype(np.float16).astype(np.float32)
    (d / "graph.json").write_text(json.dumps(graph, indent=1))
    np.savez(d / "weights.npz", **weights)
    np.savez(d / "io.npz", input=golden_in, output=np.zeros(1, np.float32))
    mbir, walk_torch = _harness()
    ir = mbir.load_ir(d)
    with torch.no_grad():
        out, _ = walk_torch.walk(ir, dev="cpu")
    np.savez(d / "io.npz", input=golden_in, output=out.numpy().astype(np.float32).reshape(-1))
    return d


def fire_ir(d: Path, quant: str = "fp32", seed: int = 0, tiny_head: bool = False, hw=(H, W)) -> Path:
    return write_ir(d, fire_graph(quant, hw), fire_weights(quant, seed, tiny_head))


def merge_expand(orig_dir: Path, out_dir: Path, wrong: str | None = None, *, with_map: bool = False):
    """Apply the merge/expand rewrite to the fire IR at ``orig_dir`` -> (edited IR dir, tensor_map).

    One 3x3 conv (OC = E1 + E3) over ``s`` with weights  concat([pad(e1.weight 1x1 -> 3x3), e3.weight], axis 0)  and
    bias concat([e1.bias, e3.bias]), then one relu producing ``cat_out`` (the cat and the four ops it replaces vanish)."""
    g0 = json.loads((Path(orig_dir) / "graph.json").read_text())
    z = np.load(Path(orig_dir) / "weights.npz")
    base = {k: z[k] for k in z.files}
    q = g0["quant"]
    H, W = g0["tensors"]["x"]["shape"][2:]
    ed = ge.Editor(g0, base)
    D = ed.derive
    pads = [[0, 0], [0, 0], [0, 2], [0, 2]] if wrong == "pad" else [[0, 0], [0, 0], [1, 1], [1, 1]]
    D.pad("d.fire.e1_3x3", "e1.weight", pads)
    D.concat("d.fire.w", ["e3.weight", "d.fire.e1_3x3"] if wrong == "order" else ["d.fire.e1_3x3", "e3.weight"], 0)
    D.concat("d.fire.b", ["e3.bias", "e1.bias"] if wrong == "order" else ["e1.bias", "e3.bias"], 0)
    ed.remove_ops(["e1", "e1_relu", "e3", "e3_relu", "cat"])
    t = _tdt(q)
    ed.add_tensor("mc_o", [1, E1 + E3, H, W], t)
    ed.add_op(conv_op("merged", "s", "mc_o", "d.fire.w", "d.fire.b", S, E1 + E3, 3, q, (H, W)))
    ed.g["tensors"].pop("cat_out")                     # re-registered below as the relu's output (same name, shape, dtype)
    ed.add_tensor("cat_out", [1, E1 + E3, H, W], t)
    ed.add_op(relu_op("merged_relu", "mc_o", "cat_out", (E1 + E3) * H * W, q))
    for k in ("e1_o", "e1r", "e3_o", "e3r"):
        ed.g["tensors"].pop(k)
    g1 = ed.result()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "graph.json").write_text(json.dumps(g1, indent=1))
    from modelblaster.pipeline import weight_derivations as wd
    wd.write_weights(Path(orig_dir) / "weights.npz", g1, out / "weights.npz")
    import shutil
    shutil.copyfile(Path(orig_dir) / "io.npz", out / "io.npz")
    tmap = {"e1r": {"from": "cat_out", "axis": 1, "start": 0, "stop": E1},
            "e3r": {"from": "cat_out", "axis": 1, "start": E1, "stop": E1 + E3}}
    return out, tmap
