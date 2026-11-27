"""int8 SmolVLA numerics validation on the CPU reference walk (int8 kinds bit-exact with the reference C,
fp32 ops as walk_torch): action-space metrics over a held-out set, per-layer error attribution,
accumulator / saturation headroom. Run where the fp32 IRs and calibration samples live.

  python validate.py metrics  OUT.jsonl COMBOS_JSON [--save DIR]      per-observation action metrics
  python validate.py local    OUT.json  POLICY_JSON  [--n 16]          per-linear local error + headroom
  python validate.py groups   OUT.jsonl POLICY_JSON  GROUPS_JSON [--obs-from metrics.jsonl --worst 12]
  python validate.py irs      OUT.jsonl IR_ROOT TAG [--save DIR]      extracted IRs (ROOT/{vision,prefix,step}) as they are
COMBOS: [[tag, {part: policy | null (fp32)}], ...]   (a part missing from a combo: its policy is "linear")
GROUPS: [[tag, {part: [fnmatch patterns kept fp32 on top of the policy]}], ...]
Common: --root (fp32 IRs: ROOT/{vision,prefix,step}; or MB_SMOLVLA_IR_ROOT), --cal (dir with cal_{part}.pt and the
default observation sets obs/, obs14/, heldout/; or MB_SMOLVLA_CAL), --obs (glob, repeatable)."""
import argparse, copy, fnmatch, glob, json, math, os, sys, time
import numpy as np, torch
from modelblaster.pipeline import int8_mixed as M
M._walk_torch()
from mbir import load_ir

PARTS = ("vision", "prefix", "step")
ap = argparse.ArgumentParser()
ap.add_argument("mode"); ap.add_argument("out"); ap.add_argument("arg1"); ap.add_argument("arg2", nargs="?")
ap.add_argument("--root", default=os.environ.get("MB_SMOLVLA_IR_ROOT"), required="MB_SMOLVLA_IR_ROOT" not in os.environ)
ap.add_argument("--cal", default=os.environ.get("MB_SMOLVLA_CAL"), required="MB_SMOLVLA_CAL" not in os.environ)
ap.add_argument("--obs", action="append")
ap.add_argument("--n", type=int, default=0)
ap.add_argument("--save")
ap.add_argument("--obs-from"); ap.add_argument("--worst", type=int, default=12)
ap.add_argument("--threads", type=int, default=12)
a = ap.parse_args()
torch.set_num_threads(a.threads)
f32 = {p: load_ir(f"{a.root}/{p}") for p in PARTS}
cal = {} if a.mode == "irs" else {p: [M.inputs_of(f32[p], s) for s in torch.load(f"{a.cal}/cal_{p}.pt")] for p in PARTS}  # irs mode needs no calibration dump
obs_files = []
for g in (a.obs or [f"{a.cal}/obs/eval0.pt", f"{a.cal}/obs14/*.pt", f"{a.cal}/heldout/*.pt"]):
    obs_files += sorted(glob.glob(g))
if a.n:
    obs_files = obs_files[:a.n]
_cache = {}
def ir_for(p, pol):
    if pol is None:
        return f32[p]
    k = (p, json.dumps(pol, sort_keys=True))
    if k not in _cache:
        g, w, rep = M.quantize_graph(f32[p], cal[p], pol)
        _cache[k] = M.as_ir(g, w)
    return _cache[k]
def temb(t, dim=720, lo=4e-3, hi=4.0):
    f = np.linspace(0.0, 1.0, dim // 2, dtype=np.float64)
    x = (1.0 / (lo * (hi / lo) ** f) * 2 * math.pi) * t
    return torch.from_numpy(np.concatenate([np.sin(x), np.cos(x)])[None, :].astype(np.float32))
def run(ir, *xs):
    return M.walk(ir, M.inputs_of(ir, list(xs)))[ir.output_names()[0]]
def e2e(v, obs, steps=5):
    kv = run(v["prefix"], run(v["vision"], obs["pixels"].float()), obs["lang"].float(), obs["state"].float())
    dt, x = -1.0 / steps, obs["noise"].float().clone()
    for s in range(steps):
        x = x + dt * run(v["step"], kv, x, temb(1.0 + s * dt)).reshape(x.shape)
    return x
def metrics(o, r, ad=6):
    o, r = o.double().reshape(-1, 32), r.double().reshape(-1, 32)
    d = (o - r).abs()
    of, rf = o.reshape(-1), r.reshape(-1)
    return {"cos": float(of @ rf / (of.norm() * rf.norm())), "maxerr": float(d.max()), "rmse": float(d.pow(2).mean().sqrt()),
            "ref_absmax": float(r.abs().max()),
            "cos6": float((o[:, :ad].reshape(-1) @ r[:, :ad].reshape(-1)) / (o[:, :ad].norm() * r[:, :ad].norm())),
            "joint_maxerr": [round(float(x), 5) for x in d[:, :ad].amax(0)],
            "joint_meanerr": [round(float(x), 5) for x in d[:, :ad].mean(0)],
            "joint_ref_std": [round(float(x), 4) for x in r[:, :ad].std(0)],
            "t_maxerr": [round(float(x), 4) for x in d[:, :ad].amax(1)[::7]]}
def policies(combo):
    return {p: combo.get(p, "linear") for p in PARTS}

if a.mode == "irs":
    v = {p: load_ir(f"{a.arg1}/{p}") for p in PARTS}
    tag = a.arg2 or a.arg1
    with open(a.out, "a") as fo:
        for f in obs_files:
            obs = torch.load(f)
            x = e2e(v, obs)
            fo.write(json.dumps({"tag": tag, "obs": f.split("/")[-1], "image": obs.get("image"), **metrics(x, obs["ref"])}) + "\n")
            fo.flush()
            if a.save:
                np.save(f"{a.save}/{tag.replace(' ', '_')}__{f.split('/')[-1][:-3]}.npy", x.numpy())
    print(tag, "done", len(obs_files), flush=True)

elif a.mode == "metrics":
    combos = json.loads(a.arg1 if not a.arg1.endswith(".json") else open(a.arg1).read())
    with open(a.out, "a") as fo:
        for tag, combo in combos:
            v = {p: ir_for(p, pol) for p, pol in policies(combo).items()}
            t0 = time.time()
            for f in obs_files:
                obs = torch.load(f)
                x = e2e(v, obs)
                rec = {"tag": tag, "obs": f.split("/")[-1], "image": obs.get("image"), **metrics(x, obs["ref"])}
                fo.write(json.dumps(rec) + "\n"); fo.flush()
                if a.save:
                    np.save(f"{a.save}/{tag.replace(' ', '_')}__{f.split('/')[-1][:-3]}.npy", x.numpy())
            print(tag, "done", len(obs_files), "obs", round(time.time() - t0), "s", flush=True)

elif a.mode == "local":
    pol = json.loads(a.arg1)
    v = {p: ir_for(p, pol.get(p, "linear")) for p in PARTS}
    stats = {}
    stash = {}
    orig_q, orig_l = M.TORCH_RULES["quantize_rows_s8"], M.TORCH_RULES["linear_dq_s8"]
    cur = {"part": None}
    def q_rule(ir, op, ins):
        out = orig_q(ir, op, ins)
        sh = op["shape"]
        x = ins[0].reshape(sh["M"], sh["K"]).float()
        ci = ins[1].reshape(-1).float()
        vv = x * ci
        amax = vv.abs().amax(1, keepdim=True)
        s = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
        pre = vv / s
        st = stats.setdefault((cur["part"], op["name"]), {"kind": "quantize"})
        st["n"] = st.get("n", 0) + pre.numel()
        st["clamped"] = st.get("clamped", 0) + int((pre.abs() > 127.5).sum())
        st["at_127"] = st.get("at_127", 0) + int((out[0].abs() == 127).sum())
        st["zero_frac_sum"] = st.get("zero_frac_sum", 0.0) + float((out[0] == 0).float().mean())
        st["calls"] = st.get("calls", 0) + 1
        # per-row: how much of the row's grid is used by the median element (resolution indicator)
        stash[op["outputs"][0]] = x
        return out
    def l_rule(ir, op, ins):
        out = orig_l(ir, op, ins)
        sh = op["shape"]
        xq = ins[0].reshape(sh["M"], sh["K"]).double()
        w = torch.from_numpy(np.asarray(ir.weights[op["weight"]])).double()
        acc = xq @ w.T
        st = stats.setdefault((cur["part"], op["name"]), {"kind": "linear", "K": sh["K"], "M": sh["M"], "N": sh["N"]})
        st["acc_absmax"] = max(st.get("acc_absmax", 0), float(acc.abs().max()))
        st["acc_bound"] = sh["K"] * 127 * 127
        # local error: the same fp32 input through the original fp32 weights
        x = stash.get(op["inputs"][0])
        fop = f32ops[cur["part"]].get(op["name"].replace("__i8part", ""))
        if x is not None and fop is not None:
            W = torch.from_numpy(np.asarray(f32[cur["part"]].weights[fop["weight"]], np.float32)).double()
            ref = x.double() @ W.T
            if fop.get("bias"):
                ref = ref + torch.from_numpy(np.asarray(f32[cur["part"]].weights[fop["bias"]], np.float32)).double()
            y = out[0].double()
            if op["name"].endswith("__i8part") or "outl" in op["outputs"][0]:
                pass
            e = (y - ref)
            st["err2"] = st.get("err2", 0.0) + float(e.pow(2).sum()); st["ref2"] = st.get("ref2", 0.0) + float(ref.pow(2).sum())
            st["err_absmax"] = max(st.get("err_absmax", 0.0), float(e.abs().max()))
        st["calls"] = st.get("calls", 0) + 1
        return out
    f32ops = {p: {op["name"]: op for op in f32[p].ops if op["op"] == "linear"} for p in PARTS}
    M.TORCH_RULES["quantize_rows_s8"], M.TORCH_RULES["linear_dq_s8"] = q_rule, l_rule
    def run_p(p, *xs):
        cur["part"] = p
        return run(v[p], *xs)
    for f in obs_files:
        obs = torch.load(f)
        kv = run_p("prefix", run_p("vision", obs["pixels"].float()), obs["lang"].float(), obs["state"].float())
        dt, x = -0.2, obs["noise"].float().clone()
        for s in range(5):
            x = x + dt * run_p("step", kv, x, temb(1.0 + s * dt)).reshape(x.shape)
        stash.clear()
    rows = []
    for (p, name), st in stats.items():
        r = {"part": p, "name": name, **st}
        if "err2" in st:
            r["local_rel_err"] = math.sqrt(st["err2"] / max(st["ref2"], 1e-30))
        if "acc_absmax" in st:
            r["acc_headroom_bits"] = math.log2(2**31 / max(st["acc_absmax"], 1))
        rows.append(r)
    json.dump(rows, open(a.out, "w"), indent=0)
    print("local done", len(rows), "ops", len(obs_files), "obs", flush=True)

elif a.mode == "groups":
    base = json.loads(a.arg1)
    groups = json.loads(a.arg2)
    files = obs_files
    if a.obs_from:                        # the worst observations of the baseline
        recs = [json.loads(l) for l in open(a.obs_from)]
        recs = [r for r in recs if r["tag"] == recs[0]["tag"]]
        recs.sort(key=lambda r: r["cos"])
        want = {r["obs"] for r in recs[:a.worst]}
        files = [f for f in obs_files if f.split("/")[-1] in want]
    with open(a.out, "a") as fo:
        for tag, keep in groups:
            combo = {}
            for p in PARTS:
                bp = base.get(p, "linear")
                if bp is None:
                    combo[p] = None
                    continue
                bp = copy.deepcopy(M.resolve_policy(bp))
                bp["keep_fp32"] = list(bp.get("keep_fp32", [])) + list(keep.get(p, []))
                combo[p] = bp
            v = {p: ir_for(p, pol) for p, pol in combo.items()}
            cs = []
            for f in files:
                obs = torch.load(f)
                m = metrics(e2e(v, obs), obs["ref"])
                cs.append((m["cos"], m["maxerr"]))
            cs = np.array(cs)
            rec = {"tag": tag, "n": len(files), "cos_mean": float(cs[:, 0].mean()), "cos_min": float(cs[:, 0].min()),
                   "maxerr_mean": float(cs[:, 1].mean()), "maxerr_max": float(cs[:, 1].max()),
                   "per_obs_cos": [round(float(c), 6) for c in cs[:, 0]]}
            fo.write(json.dumps(rec) + "\n"); fo.flush()
            print(json.dumps({k: v for k, v in rec.items() if k != "per_obs_cos"}), flush=True)
