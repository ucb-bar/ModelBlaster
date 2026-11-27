"""obs_L<L>.bin (count x [pixels, lang, state, noise] f32) + names, and temb.bin (steps x 720) for k3ir chunk."""
import glob, json, math, sys
import numpy as np, torch
src, out, steps = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 5
def temb(t, dim=720, lo=4e-3, hi=4.0):
    f = np.linspace(0.0, 1.0, dim // 2, dtype=np.float64)
    x = (1.0 / (lo * (hi / lo) ** f) * 2 * math.pi) * t
    return np.concatenate([np.sin(x), np.cos(x)]).astype(np.float32)
dt = -1.0 / steps
np.stack([temb(1.0 + s * dt) for s in range(steps)]).tofile(f"{out}/temb.bin")
for d in sorted(glob.glob(f"{src}/L*")):
    L = d.rsplit("L", 1)[1]
    files = sorted(glob.glob(f"{d}/*.pt"))
    recs = []
    for f in files:
        o = torch.load(f, weights_only=False)
        recs.append(np.concatenate([o[k].float().numpy().reshape(-1) for k in ("pixels", "lang", "state", "noise")]))
    np.stack(recs).astype(np.float32).tofile(f"{out}/obs_L{L}.bin")
    json.dump([f.rsplit("/", 1)[1] for f in files], open(f"{out}/obs_L{L}.json", "w"))
    print(L, len(files))
