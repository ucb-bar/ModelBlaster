"""For IRs that miss the fp32 golden bound: is the IR wrong, or the problem
ill-conditioned? Walks the IR in float64 (truth, checked against the model in
fp64 when ADAPTED_PY is given) and reports how far the fp32 golden and the
hand-CUDA walk each are from it.  argv: SPECS LIB_HAND IR_DIR [ADAPTED_PY]"""
import os, sys, numpy as np, torch
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness")
sys.path.insert(0, H)
import mbir, walk_torch, walk_cuda

specs, lib, d = sys.argv[1:4]
ir = mbir.load_ir(d)
import json
plan, missing = walk_cuda.build_plan(ir, walk_cuda.KernelSet(json.load(open(specs)), hand=lib, seeds=None, impl="hand"))
assert not missing, missing
with torch.no_grad():
    gpu, _, _ = walk_cuda.run(ir, plan)
gpu = gpu.double().cpu()

# float64 walk: inputs/weights in double, and the harness ops' .float() casts kept double
packed = ir.graph["input"].get("packed_inputs") or [
    {"name": ir.graph["input"]["tensor"], "offset": 0, "size": ir.golden_in.size}]
flat = torch.from_numpy(ir.golden_in.astype(np.float64))
xs = {p["name"]: flat[p["offset"]:p["offset"] + p["size"]].reshape(ir.tshape(p["name"])).clone() for p in packed}
f32 = torch.Tensor.float
torch.Tensor.float = lambda self: self.double()
mbir.IR.input_tensors = lambda self, device: {k: v.clone() for k, v in xs.items()}
mbir.IR.weight = lambda self, key, device: None if key is None else \
    torch.from_numpy(np.ascontiguousarray(self.weights[key], dtype=np.float64))
with torch.no_grad():
    truth, _ = walk_torch.walk(ir, "cpu")
truth = truth.double()
r = truth.abs().max().item()
gold = torch.from_numpy(ir.golden_out.astype(np.float64)).reshape(-1)
rel = lambda a: (a - truth).abs().max().item() / r
line = f"{ir.name}\tgolden32_vs_fp64 {rel(gold):.3g}\tcuda_vs_fp64 {rel(gpu):.3g}"
if len(sys.argv) > 4:
    import importlib.util
    torch.Tensor.float = f32
    spec = importlib.util.spec_from_file_location("m", sys.argv[4])
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    torch.manual_seed(0); ins = m.get_inputs()
    tens = iter(xs[p["name"]] for p in packed)
    args = [next(tens).to(torch.float64 if t.is_floating_point() else t.dtype) if torch.is_tensor(t) else t for t in ins]
    mod = m.Model(*(m.get_init_inputs() if hasattr(m, "get_init_inputs") else [])).double()
    with torch.no_grad():
        o = mod(*args)
    ref = torch.cat([t.reshape(-1).double() for t in (o if isinstance(o, (tuple, list)) else [o])])
    line += f"\tir64_vs_model64 {(ref - truth).abs().max().item() / r:.3g}"
print(line)
