#!/usr/bin/env python3
"""Per-op fp16 characterization for DroNet int8->hybrid.
Which ops, run in fp16 (compute+accumulate, full-precision weights), recover the
most of the float steering EVA? Measures a progressive promotion sweep + isolates.
DATASET=simforest|himax. Faithful: non-promoted ops use the int8 golden; hybrid
with no promotions is verified bit-exact vs int8_sim."""
import os, sys, json, csv, glob, time
os.environ["MODELBLASTER_DRONET_CHANNELS"]="1"
DATASET=os.environ.get("DATASET","simforest")
N_SUB=int(os.environ.get("N_SUB","40")); N_CALIB=int(os.environ.get("N_CALIB","64"))
OUT=os.environ.get("OUT_DIR",f"/tmp/fp16char_{DATASET}")
X="/scratch2/dima/misc_sw/XPU-RT"; M="/scratch2/dima/misc_sw/xpurt_repro_wt"; SP="/scratch2/dima/misc_sw/dronet_int8_eval"
for p in (X,M,SP,os.path.dirname(os.path.abspath(__file__))): sys.path.insert(0,p)
import numpy as np, torch
from PIL import Image, ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES=True
from torchvision import transforms
from modelblaster.models import dronet as dmod
from modelblaster.pipeline import extract_graph as EG
from int8_sim import simulate_int8
from hybrid_sim import simulate_hybrid

# ---- data ----
_tx=transforms.Compose([transforms.Resize((112,112)),transforms.ToTensor()])
def load(p):
    with Image.open(p) as im: return _tx(im.convert("L"))
if DATASET=="himax":
    os.environ["MODELBLASTER_DRONET_CKPT"]=f"{X}/logs/dronet_himax/best.pt"
    ROOT=f"{X}/datasets/pulp_dronet_himax/raw/Dataset_PULP_Dronet_v3"; NORM=90.0
    tr=[];va=[]
    for c in sorted(glob.glob(os.path.join(ROOT,"**","labels_partitioned.csv"),recursive=True)):
        d=os.path.dirname(c); imgd=os.path.join(d,"images")
        for r in csv.DictReader(open(c)):
            try:y=max(-1.,min(1.,float(r["label_yaw_rate"])/NORM))
            except:continue
            p=os.path.join(imgd,r["filename"])
            if not os.path.exists(p):continue
            (tr if r["partition"]=="train" else (va if r["partition"]=="valid" else [])).append((p,y))
else:  # simforest
    os.environ["MODELBLASTER_DRONET_CKPT"]=f"{X}/logs/dronet_simforest/best.pt"
    DATA=f"{X}/datasets/sim_forest/extracted"
    rows=[]
    for seg in sorted(os.listdir(DATA)):
        lf=f"{DATA}/{seg}/labels.csv"
        if os.path.exists(lf):
            for r in csv.DictReader(open(lf)): rows.append((f"{DATA}/{seg}/{r['filename']}",float(r["steering_label"])))
    rng=np.random.RandomState(0); idx=rng.permutation(len(rows)); nval=len(rows)//6
    va=[rows[i] for i in idx[:nval]]; tr=[rows[i] for i in idx[nval:]]
print(f"[data] {DATASET} train={len(tr)} valid={len(va)} ckpt={os.environ['MODELBLASTER_DRONET_CKPT']}",flush=True)
vy=np.array([y for _,y in va]); order=np.argsort(vy)
si=np.linspace(0,len(va)-1,min(N_SUB,len(va))).astype(int); sub=[va[order[i]] for i in si]
gt=np.array([y for _,y in sub])

# ---- model + float params ----
model=dmod.get_model().eval()
def find_dronet(m):
    if hasattr(m,"conv_modules"): return m
    for a in ("model","net","module","dronet"):
        if hasattr(m,a): return find_dronet(getattr(m,a))
    return m
dn=find_dronet(model)
fparams={}
for i,c in enumerate(dn.conv_modules):
    fparams[f"conv_modules.{i}"]={"W":c.weight.detach().cpu().numpy(),
        "b":(c.bias.detach().cpu().numpy() if c.bias is not None else None)}
for i,bn in enumerate(dn.bn_modules):
    fparams[f"bn_modules.{i}"]={"bn":(bn.weight.detach().cpu().numpy(),bn.bias.detach().cpu().numpy(),
        bn.running_mean.detach().cpu().numpy(),bn.running_var.detach().cpu().numpy(),float(bn.eps))}
fparams["linear1"]={"W":dn.linear1.weight.detach().cpu().numpy(),"b":dn.linear1.bias.detach().cpu().numpy()}

# ---- float reference ----
dev="cuda" if torch.cuda.is_available() else "cpu"; model.to(dev)
with torch.no_grad():
    b=torch.stack([load(p) for p,_ in sub]).to(dev); sf=model(b)[0].squeeze(1).cpu().numpy().astype(np.float64)

# ---- extract int8 (fold=False = BN separate, deploy) ----
ci=np.linspace(0,len(tr)-1,N_CALIB).astype(int); calib=[load(tr[i][0]).unsqueeze(0) for i in ci]
os.makedirs(OUT,exist_ok=True)
EG.extract_int8(model.cpu(),calib[0],"c",OUT,calibration_samples=calib,fold_conv_bn=False)
ir=json.load(open(f"{OUT}/graph.json")); wn=np.load(f"{OUT}/weights.npz"); wb={k:wn[k] for k in wn.files}
ops=ir["ops"]; inn=ir["input"]["tensors"]; st=ir["output"]["tensors"][0]
scales={n:ir["tensors"][n]["quant"]["scale"] for n in ir["tensors"] if "quant" in ir["tensors"][n]}
insc=scales[inn[0]]; ssc=scales[st]
insh=[1]+list(ir["tensors"][inn[0]]["shape"])[1:]

# ---- op groups ----
conv_names=set(); bn_names=set(); add_names=set(); relu_names=set(); fc_names=set()
for op in ops:
    k=op["op"]; nm=op["name"]
    if k=="conv2d_s8": conv_names.add(nm)
    elif k=="batchnorm2d_s8": bn_names.add(nm)
    elif k=="add_s8": add_names.add(nm)
    elif k=="relu_s8": relu_names.add(nm)
    elif k=="linear_s8" and nm=="linear1": fc_names.add(nm)
    elif k=="conv2d_batchnorm2d_s8":
        conv_names.add(op["sub_ops"][0]["name"]); bn_names.add(op["sub_ops"][1]["name"])
print(f"[groups] conv={len(conv_names)} bn={len(bn_names)} add={len(add_names)} relu={len(relu_names)} fc={len(fc_names)}",flush=True)

# ---- verify hybrid(empty)==int8 golden ----
def q_in(p): return np.clip(np.round(load(p).numpy().reshape(insh)/insc),-128,127).astype(np.int8)
x0=q_in(sub[0][0])
g_i8=simulate_int8(ops,wb,[x0],inn)[st].reshape(-1)[0]
g_hy=simulate_hybrid(ops,wb,[x0],inn,scales,set(),fparams)[st].reshape(-1)[0]
print(f"[verify] int8={g_i8} hybrid_empty={g_hy} match={int(g_i8)==int(g_hy)}",flush=True)

# ---- configs ----
CFG={
 "int8": set(),
 "bn": bn_names,
 "bn+add": bn_names|add_names,
 "bn+add+relu": bn_names|add_names|relu_names,
 "bn+add+relu+fc (DEPLOY: convs int8)": bn_names|add_names|relu_names|fc_names,
 "all_fp16": conv_names|bn_names|add_names|relu_names|fc_names,
 "only_bn": bn_names,
 "only_add": add_names,
 "only_relu": relu_names,
 "only_fc": fc_names,
 "only_convs": conv_names,
 "only_conv0": {"conv_modules.0"},
 "add+relu": add_names|relu_names,
 "relu+fc": relu_names|fc_names,
 "add2+relu": {"add_2"}|relu_names,
}
def steer(cfg_set,p):
    a=simulate_hybrid(ops,wb,[q_in(p)],inn,scales,cfg_set,fparams)[st].reshape(-1)[0]
    return float(a)*ssc if isinstance(a,(np.int8,)) or (hasattr(a,'dtype') and a.dtype==np.int8) else float(a)
def E(pr,g): v=g.var(); return float(1-((pr-g).var()/v)) if v>0 else float('nan')

res={}; t0=time.time()
for name,cs in CFG.items():
    pr=np.array([steer(cs,p) for p,_ in sub])
    res[name]={"EVA_vs_GT":E(pr,gt),"MAE_vs_float":float(np.abs(pr-sf).mean()),
               "MAE_vs_GT":float(np.abs(pr-gt).mean())}
    print(f"[cfg {name}] EVA={res[name]['EVA_vs_GT']:.3f} MAE_vs_float={res[name]['MAE_vs_float']:.4f} ({time.time()-t0:.0f}s)",flush=True)

eva_float=E(sf,gt); eva_i8=res["int8"]["EVA_vs_GT"]; gap=eva_float-eva_i8
summary={"dataset":DATASET,"n":len(sub),"EVA_float":eva_float,"EVA_int8":eva_i8,"steer_scale":ssc,
  "configs":res,"recovery_pct":{k:(round(100*(v["EVA_vs_GT"]-eva_i8)/gap,1) if gap>1e-9 else None) for k,v in res.items()}}
json.dump(summary,open(f"{OUT}/fp16_char.json","w"),indent=1)
print("=== SUMMARY ==="); print(json.dumps(summary,indent=1)); print("FP16_CHAR_DONE",flush=True)
