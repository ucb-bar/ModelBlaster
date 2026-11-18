. /tmp/krisd/venv/bin/activate
mkdir -p /tmp/krisd/mamba_chk && cd /tmp/krisd/mamba_chk
python - <<'PY'
import torch, importlib.util
spec=importlib.util.spec_from_file_location('m','/tmp/krisd/mamba_chk/p48.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(0)
init=m.get_init_inputs(); xs=m.get_inputs()
def run(dt):
    torch.manual_seed(0)
    model=m.Model(*init).cuda().to(dt).eval()
    ins=[x.cuda().to(dt) if torch.is_tensor(x) and x.is_floating_point() else (x.cuda() if torch.is_tensor(x) else x) for x in xs]
    with torch.no_grad(): out=model(*ins)
    return out if torch.is_tensor(out) else out[0]
o32=run(torch.float32); o16=run(torch.float16)
print('shape',tuple(o32.shape))
print('fp32 finite',bool(torch.isfinite(o32).all()),'absmax',float(o32.abs().max()))
f=torch.isfinite(o16)
print('fp16 nonfinite count',int((~f).sum()),'of',o16.numel(),'; finite absmax',float(o16[f].abs().max()) if f.any() else None)
if f.all(): print('fp16 vs fp32 max abs err',float((o16.float()-o32).abs().max()))
PY
