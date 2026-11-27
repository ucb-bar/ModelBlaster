"""Open-loop chunk accuracy of K3 (or any) SmolVLA actions vs the lerobot fp32 PyTorch chunk on the same
observation + noise, in PILOT units, as smolvla_gate/code/i8_summary.py reports it.

  python chunk_summary.py OBS_DIR STATS_JSON TAG=ACTS_DIR [TAG=ACTS_DIR ...] [--json OUT]
ACTS_DIR holds acts_L<L>.bin (count x 50 x 32 f32, k3ir chunk --out) and obs_L<L>.json (obs names), or
<q>_L<L>__<obs>.npy files (validate.py --save) when TAG starts with npy:<q>."""
import glob, json, sys
import numpy as np, torch
obs_dir, st = sys.argv[1], json.load(open(sys.argv[2]))
asd = np.array(st["action_std"])
args = [a for a in sys.argv[3:] if "=" in a]
out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
res = {}
for a in args:
    tag, d = a.split("=", 1)
    e10, eall, cos = [], [], []
    for Ld in sorted(glob.glob(f"{obs_dir}/L*")):
        L = Ld.rsplit("L", 1)[1]
        if tag.startswith("npy:"):
            q = tag[4:]
            names = sorted(f.rsplit("/", 1)[1] for f in glob.glob(f"{Ld}/*.pt"))
            acts = [np.load(f"{d}/{q}_L{L}__{n[:-3]}.npy").reshape(50, 32) for n in names]
        else:
            names = json.load(open(f"{d}/obs_L{L}.json"))
            raw = np.fromfile(f"{d}/acts_L{L}.bin", np.float32)
            acts = list(raw.reshape(-1, 50, 32))[:len(names)]
        for n, act in zip(names, acts):
            o = torch.load(f"{Ld}/{n}", weights_only=False)
            a4, r = act[:, :4], o["ref"][0, :, :4].numpy()
            dd = np.abs(a4 - r) * asd
            e10.append(dd[:10].max(0)); eall.append(dd.max(0))
            cos.append(float((a4 * r).sum() / np.linalg.norm(a4) / np.linalg.norm(r)))
    e10, eall = np.array(e10), np.array(eall)
    res[tag] = dict(n=len(cos), cos_min=min(cos), cos_median=float(np.median(cos)),
                    maxerr_first10_median=np.median(e10, 0).round(4).tolist(),
                    maxerr_first10_p95=np.percentile(e10, 95, 0).round(4).tolist(),
                    maxerr_first10_max=e10.max(0).round(4).tolist(), maxerr_chunk_max=eall.max(0).round(4).tolist())
res["units"] = "PILOT [vx m/s, vy m/s, yawrate rad/s, alt m]; first10 = the executed actions at n_exec 10"
print(json.dumps(res, indent=1))
if out:
    json.dump(res, open(out, "w"), indent=1)
