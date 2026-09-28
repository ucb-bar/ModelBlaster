#!/usr/bin/env python3
"""Train DroNet steering head on the REAL HM01B0 PULP-DroNet v3 dataset.
Domain-matched: same Himax HM01B0 monochrome sensor as riskybird. Continuous yaw-rate.
Grayscale 112x112 (deploy input). Official partition split. Steering head only.
label_yaw_rate is raw deg/s (+-90) -> normalize /90 to [-1,+1] (+CCW=turn left, our convention).
BALANCE=1 -> WeightedRandomSampler by |yaw| bins to counter the ~83% straight-flight zeros."""
import os, sys, time, csv, glob, argparse
X="/scratch2/dima/misc_sw/XPU-RT"; sys.path.insert(0, X)
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image, ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES=True
from torchvision import transforms
from qnn_models.dronet import DronetTorch

ap=argparse.ArgumentParser()
ap.add_argument("--epochs",type=int,default=int(os.environ.get("EPOCHS","45")))
ap.add_argument("--bs",type=int,default=64); ap.add_argument("--lr",type=float,default=1e-3)
ap.add_argument("--root",default=f"{X}/datasets/pulp_dronet_himax/raw/Dataset_PULP_Dronet_v3")
ap.add_argument("--out",default=f"{X}/logs/dronet_himax")
ap.add_argument("--norm",type=float,default=90.0)  # deg/s -> [-1,1]
ap.add_argument("--res",type=int,default=int(os.environ.get("RES","112")))  # 112=small(deploy) 200=original DroNet
a=ap.parse_args()
BALANCE=os.environ.get("BALANCE","0")=="1"
BAL_POW=float(os.environ.get("BAL_POW","1.0"))  # 1=inverse-freq(aggressive) 0.5=sqrt(mild) 0=natural
RES=a.res; SMALL=(RES<200)  # linear_in 2048(4x4@112) vs 6272(7x7@200)
dev="cuda" if torch.cuda.is_available() else "cpu"

# gather (imgpath, yaw_norm, partition) from every labels_partitioned.csv
rows_tr=[]; rows_va=[]
for c in sorted(glob.glob(os.path.join(a.root,"**","labels_partitioned.csv"),recursive=True)):
    d=os.path.dirname(c); imgd=os.path.join(d,"images")
    for r in csv.DictReader(open(c)):
        try: y=float(r["label_yaw_rate"])/a.norm
        except: continue
        y=max(-1.0,min(1.0,y))
        p=os.path.join(imgd,r["filename"])
        if not os.path.exists(p): continue
        part=r["partition"]
        if part=="train": rows_tr.append((p,y))
        elif part=="valid": rows_va.append((p,y))
print(f"[data] train={len(rows_tr)} valid={len(rows_va)}", flush=True)
ytr=np.array([y for _,y in rows_tr]); yva=np.array([y for _,y in rows_va])
print(f"[yaw norm] train std={ytr.std():.3f} frac|y|>0.02={np.mean(np.abs(ytr)>0.02):.3f}  valid std={yva.std():.3f} frac_nz={np.mean(np.abs(yva)>0.02):.3f}", flush=True)

_rz=int(round(RES*1.14))  # resize larger then random-crop for aug
_txt=transforms.Compose([transforms.Resize((_rz,_rz)),transforms.RandomCrop(RES),transforms.ToTensor()])
_txv=transforms.Compose([transforms.Resize((RES,RES)),transforms.ToTensor()])
class HX(Dataset):
    def __init__(self,items,train): self.items=items; self.train=train
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        p,y=self.items[i]
        with Image.open(p) as im: im=im.convert("L")
        img=(_txt if self.train else _txv)(im)
        if self.train and np.random.rand()<0.5: img=torch.flip(img,dims=[2]); y=-y
        return img, torch.tensor([y],dtype=torch.float32)

if BALANCE:
    # weight by |yaw| bins: give turning frames more sampling weight
    absb=np.abs(ytr); bins=np.array([0.0,0.02,0.1,0.3,2.0]); bi=np.digitize(absb,bins)-1
    cnt=np.bincount(bi,minlength=len(bins)); w=np.power(1.0/np.maximum(cnt[bi],1),BAL_POW)
    sampler=WeightedRandomSampler(torch.as_tensor(w,dtype=torch.double),num_samples=len(rows_tr),replacement=True)
    tl=DataLoader(HX(rows_tr,True),batch_size=a.bs,sampler=sampler,num_workers=8,pin_memory=True,drop_last=True)
    print(f"[balance] bin counts={cnt.tolist()} pow={BAL_POW}", flush=True)
else:
    tl=DataLoader(HX(rows_tr,True),batch_size=a.bs,shuffle=True,num_workers=8,pin_memory=True,drop_last=True)
vl=DataLoader(HX(rows_va,False),batch_size=256,shuffle=False,num_workers=8,pin_memory=True)

def eva(pred,gt):
    v=gt.var(); return float(1-((pred-gt).var()/v)) if v>0 else float('nan')
def evaluate(m):
    m.eval(); P=[];G=[]
    with torch.no_grad():
        for x,y in vl:
            s,_=m(x.to(dev)); P.append(s.squeeze(1).cpu().numpy()); G.append(y.squeeze(1).numpy())
    P=np.concatenate(P);G=np.concatenate(G)
    nz=np.abs(G)>0.02
    return float(((P-G)**2).mean()), eva(P,G), (eva(P[nz],G[nz]) if nz.sum()>2 else float('nan')), float(np.abs(P-G).mean())

model=DronetTorch(img_dims=(RES,RES),img_channels=1,output_dim=1,small=SMALL).to(dev)
opt=optim.Adam(model.parameters(),lr=a.lr,weight_decay=1e-4)
sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
outd=__import__("pathlib").Path(a.out); outd.mkdir(parents=True,exist_ok=True); best=-1e9
print(f"[model] res={RES} params={sum(p.numel() for p in model.parameters())} small={SMALL} balance={BALANCE}", flush=True)
for ep in range(1,a.epochs+1):
    model.train(); t0=time.time(); ls=0;nc=0
    for x,y in tl:
        x=x.to(dev);y=y.to(dev); opt.zero_grad()
        s,_=model(x); loss=nn.functional.mse_loss(s,y); loss.backward(); opt.step()
        ls+=loss.item()*len(x); nc+=len(x)
    sch.step(); vmse,veva,veva_nz,vmae=evaluate(model)
    print(f"[ep{ep:02d}] train_mse={ls/nc:.4f} val_mse={vmse:.4f} val_EVA={veva:.3f} val_EVA_nz={veva_nz:.3f} val_MAE={vmae:.4f} ({time.time()-t0:.1f}s)",flush=True)
    torch.save(model.state_dict(),outd/"last.pt")
    if veva>best: best=veva; torch.save(model.state_dict(),outd/"best.pt"); print(f"  ^best EVA={best:.3f}",flush=True)
print(f"[done] best_val_EVA={best:.3f} ckpt={outd/'best.pt'}",flush=True); print("HIMAX_TRAIN_DONE",flush=True)
