#!/usr/bin/env python3
"""Progressive per-op fp16 characterization for the N-channel stacked DroNet
(adapts fp16_char.py). Locates where the int8 loss lives and whether a fuller
fp16 promotion (deployable: convs int8, rest fp16) recovers float accuracy.
Faithful: non-promoted ops = int8 golden; hybrid(empty) verified bit-exact.
Runs on the FULL leakage-free valid set (natural distribution)."""
import os, sys, json, csv, glob, time
N=int(os.environ["N"]); NORM=90.0
N_CALIB=int(os.environ.get("N_CALIB","64")); N_SUB=int(os.environ.get("N_SUB","6000"))
OUT=os.environ.get("OUT_DIR",f"/tmp/stack{N}_sweep")
X="/scratch2/dima/misc_sw/XPU-RT"; M="/scratch2/dima/misc_sw/xpurt_repro_wt"; SP="/scratch2/dima/misc_sw/dronet_int8_eval"
for p in (X,M,SP): sys.path.insert(0,p)
import numpy as np, torch
from PIL import Image, ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES=True
import torchvision.transforms.functional as TF
from modelblaster.models import dronet_arch
from modelblaster.pipeline import extract_graph as EG
from int8_sim import simulate_int8
from hybrid_sim import simulate_hybrid
ROOT=f"{X}/datasets/pulp_dronet_himax/raw/Dataset_PULP_Dronet_v3"; CKPT=f"{X}/logs/dronet_himax_stack{N}/best.pt"; HUGE=1e12
def load_stack(paths,size=112):
    ch=[]
    for p in paths:
        with Image.open(p) as im: im=im.convert("L").resize((size,size),Image.BILINEAR)
        ch.append(TF.to_tensor(im))
    return torch.cat(ch,dim=0)
def build(part):
    st=[]
    for c in sorted(glob.glob(os.path.join(ROOT,"**","labels_partitioned.csv"),recursive=True)):
        d=os.path.dirname(c); imgd=os.path.join(d,"images"); keyed=[]
        for r in csv.DictReader(open(c)):
            try: k=int(r["filename"].rsplit(".",1)[0])
            except: continue
            if k>HUGE: continue
            try: y=max(-1.0,min(1.0,float(r["label_yaw_rate"])/NORM))
            except: continue
            p=os.path.join(imgd,r["filename"]);
            if not os.path.exists(p): continue
            keyed.append((k,r["partition"],p,y))
        keyed.sort(key=lambda t:t[0]); parts=[t[1] for t in keyed]
        for i in range(len(keyed)):
            if parts[i]!=part or i-(N-1)<0: continue
            win=keyed[i-(N-1):i+1]
            if any(w[1]!=part for w in win): continue
            st.append((tuple(w[2] for w in win), keyed[i][3]))
    return st
tr=build("train"); va=build("valid")
sub=va[:N_SUB] if N_SUB<len(va) else va
gt=np.array([y for _,y in sub]); nz=np.abs(gt)>0.02
print(f"[data] N={N} valid={len(va)} eval_on={len(sub)} turn_frac={nz.mean():.3f}",flush=True)
model=dronet_arch.DronetTorch(img_dims=(112,112),img_channels=N,output_dim=1,small=True).eval()
model.load_state_dict(torch.load(CKPT,map_location="cpu",weights_only=False),strict=False)
fparams={}
for i,c in enumerate(model.conv_modules): fparams[f"conv_modules.{i}"]={"W":c.weight.detach().numpy(),"b":(c.bias.detach().numpy() if c.bias is not None else None)}
for i,bn in enumerate(model.bn_modules): fparams[f"bn_modules.{i}"]={"bn":(bn.weight.detach().numpy(),bn.bias.detach().numpy(),bn.running_mean.detach().numpy(),bn.running_var.detach().numpy(),float(bn.eps))}
fparams["linear1"]={"W":model.linear1.weight.detach().numpy(),"b":model.linear1.bias.detach().numpy()}
dev="cuda" if torch.cuda.is_available() else "cpu"; model.to(dev)
def E(pr,g): v=g.var(); return float(1-((pr-g).var()/v)) if v>0 else float('nan')
with torch.no_grad():
    P=[]
    for j in range(0,len(sub),256):
        b=torch.stack([load_stack(p) for p,_ in sub[j:j+256]]).to(dev); P.append(model(b)[0].squeeze(1).cpu().numpy())
    sf=np.concatenate(P).astype(np.float64)
model.cpu()
ci=np.linspace(0,len(tr)-1,N_CALIB).astype(int); calib=[load_stack(tr[i][0]).unsqueeze(0) for i in ci]
os.makedirs(OUT,exist_ok=True)
EG.extract_int8(model,calib[0],f"s{N}",OUT,calibration_samples=calib,fold_conv_bn=False)
ir=json.load(open(f"{OUT}/graph.json")); wn=np.load(f"{OUT}/weights.npz"); wb={k:wn[k] for k in wn.files}
ops=ir["ops"]; inn=ir["input"]["tensors"]; st=ir["output"]["tensors"][0]
scales={n:ir["tensors"][n]["quant"]["scale"] for n in ir["tensors"] if "quant" in ir["tensors"][n]}
insc=scales[inn[0]]; ssc=scales[st]; insh=[1]+list(ir["tensors"][inn[0]]["shape"])[1:]
conv=set(); bn=set(); add=set(); relu=set(); fc=set()
for op in ops:
    k=op["op"]; nm=op["name"]
    if k=="conv2d_s8": conv.add(nm)
    elif k=="batchnorm2d_s8": bn.add(nm)
    elif k=="add_s8": add.add(nm)
    elif k=="relu_s8": relu.add(nm)
    elif k=="linear_s8" and nm=="linear1": fc.add(nm)
    elif k=="conv2d_batchnorm2d_s8": conv.add(op["sub_ops"][0]["name"]); bn.add(op["sub_ops"][1]["name"])
def q_in(paths): return np.clip(np.round(load_stack(paths).numpy().reshape(insh)/insc),-128,127).astype(np.int8)
x0=q_in(sub[0][0])
print(f"[verify] bit_exact={int(simulate_int8(ops,wb,[x0],inn)[st].reshape(-1)[0])==int(simulate_hybrid(ops,wb,[x0],inn,scales,set(),fparams)[st].reshape(-1)[0])}",flush=True)
CFG={"int8":set(),"tail(relu6+fc)":relu|fc,"only_bn":bn,"only_convs":conv,"bn+add":bn|add,
     "bn+add+relu":bn|add|relu,"DEPLOY(bn+add+relu+fc, convs int8)":bn|add|relu|fc,"all_fp16":conv|bn|add|relu|fc}
def steer(cfg):
    out=np.zeros(len(sub))
    for j,(p,_) in enumerate(sub):
        a=simulate_hybrid(ops,wb,[q_in(p)],inn,scales,cfg,fparams)[st].reshape(-1)[0]
        out[j]=(float(a)*ssc) if (hasattr(a,'dtype') and a.dtype==np.int8) else float(a)
    return out
efloat=E(sf,gt); res={}; t0=time.time()
for name,cs in CFG.items():
    pr=steer(cs); res[name]={"EVA":E(pr,gt),"EVA_nz":E(pr[nz],gt[nz]),"MAE_vs_float":float(np.abs(pr-sf).mean())}
    print(f"[{name}] EVA={res[name]['EVA']:.3f} nz={res[name]['EVA_nz']:.3f} MAEvf={res[name]['MAE_vs_float']:.4f} ({time.time()-t0:.0f}s)",flush=True)
gap=efloat-res["int8"]["EVA"]
summary={"N":N,"n":len(sub),"EVA_float":efloat,"EVA_float_nz":E(sf[nz],gt[nz]),"configs":res,
  "recovery_pct_overall":{k:(round(100*(v["EVA"]-res["int8"]["EVA"])/gap,1) if abs(gap)>1e-9 else None) for k,v in res.items()}}
json.dump(summary,open(f"{OUT}/sweep.json","w"),indent=1); print("=== SUMMARY ==="); print(json.dumps(summary,indent=1)); print(f"STACK_SWEEP_DONE N={N}",flush=True)
