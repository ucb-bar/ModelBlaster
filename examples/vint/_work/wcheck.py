import torch, numpy as np, sys
sys.path.insert(0, "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw")
from modelblaster.models import vint as V
m = V.get_model(); sd = m.state_dict()
k = "obs_encoder._conv_stem.weight"
w = sd[k].detach().numpy()
ck = torch.load("/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/sims/external/visualnav-transformer/deployment/model_weights/vint.pth", map_location="cpu", weights_only=False)
loaded = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
csd = loaded.module.state_dict() if hasattr(loaded, "module") else loaded.state_dict()
cw = csd[k].detach().numpy()
m2 = V._build_module(V._load_config()).eval(); rw = m2.state_dict()[k].detach().numpy()
print("model==checkpoint :", bool(np.array_equal(w, cw)), " sum=%.6f" % w.sum())
print("model==randominit :", bool(np.array_equal(w, rw)), " randinit sum=%.6f" % rw.sum())
print("n_params:", sum(p.numel() for p in m.parameters()))
