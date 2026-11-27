"""The diffusion locomotion student (DiffuseLoco method, the `dlm_fused` checkpoint) as one
FX-traceable denoiser pass, for extract_graph --quant fp32 | int8 (int8_mixed).

The release's TransformerForDiffusion (6 x nn.TransformerDecoderLayer, norm_first, GELU, width 256, 8 heads,
causal self-attention over the 16 action tokens, masked cross-attention to 17 condition tokens produced by an
MLP "encoder") is re-stated with plain ops (linear / layer_norm / bmm / softmax with additive -inf masks /
gelu / mish), so it lowers to the generic fp32 ops; `check()` measures it against the release module.

  denoiser  x [1, 16, 12] (noisy action chunk), temb [1, 256] (sinusoid of the integer timestep, host),
            cond [1, 8, 45] (normalised observation history, host) -> eps [1, 16, 12]

The DDPM update (10 steps), the normalisers and the timestep sinusoid stay on the host: they are ~1k flops.
Environment (all required): MB_DLM_CKPT (the student checkpoint), MB_DLM_REPO (the DiffuseLoco release),
MB_DLM_POLICIES (directory with the `policies` module that defines Student), MB_DLM_DATA (an .npz with
`state` [N, 45] and `episode_ends`, for calibration / sample inputs).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT = os.environ.get("MB_DLM_CKPT", "")
REPO = os.environ.get("MB_DLM_REPO", "")
DATA = os.environ.get("MB_DLM_DATA", "")
POLICIES_DIR = os.environ.get("MB_DLM_POLICIES", "")


def _need(name: str, value: str) -> str:
    if not value:
        raise RuntimeError(f"dlm_student: set {name} (see the module docstring)")
    return value


def _student():
    pdir = _need("MB_DLM_POLICIES", POLICIES_DIR)
    if pdir not in sys.path:
        sys.path.insert(0, pdir)
    import policies as P  # noqa: PLC0415
    return P.Student(_need("MB_DLM_CKPT", CKPT), _need("MB_DLM_REPO", REPO))


def temb(t: int, dim: int = 256) -> torch.Tensor:
    """SinusoidalPosEmb(dim)(tensor([t])) of the release, in fp32."""
    half = dim // 2
    e = math.log(10000) / (half - 1)
    e = torch.exp(torch.arange(half) * -e)
    e = torch.tensor([float(t)])[:, None] * e[None, :]
    return torch.cat((e.sin(), e.cos()), dim=-1)


def _mha(q, k, v, heads, mask):
    """q [S, E], k / v [T, E] -> [S, E]; softmax(q k^T / sqrt(d) + mask) v per head."""
    S, E = q.shape
    T = k.shape[0]
    d = E // heads
    qh = q.reshape(S, heads, d).permute(1, 0, 2)
    kh = k.reshape(T, heads, d).permute(1, 2, 0)
    vh = v.reshape(T, heads, d).permute(1, 0, 2)
    s = torch.bmm(qh, kh) * (d ** -0.5) + mask
    p = F.softmax(s, dim=-1)
    return torch.bmm(p, vh).permute(1, 0, 2).reshape(S, E)


class Layer(nn.Module):
    def __init__(self, E=256, heads=8):
        super().__init__()
        self.heads = heads
        self.sa_q, self.sa_k, self.sa_v, self.sa_o = (nn.Linear(E, E) for _ in range(4))
        self.ca_q, self.ca_k, self.ca_v, self.ca_o = (nn.Linear(E, E) for _ in range(4))
        self.linear1, self.linear2 = nn.Linear(E, 4 * E), nn.Linear(4 * E, E)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(E), nn.LayerNorm(E), nn.LayerNorm(E)

    def forward(self, x, mem, tmask, mmask):
        h = self.norm1(x)
        x = x + self.sa_o(_mha(self.sa_q(h), self.sa_k(h), self.sa_v(h), self.heads, tmask))
        h = self.norm2(x)
        x = x + self.ca_o(_mha(self.ca_q(h), self.ca_k(mem), self.ca_v(mem), self.heads, mmask))
        h = self.norm3(x)
        return x + self.linear2(F.gelu(self.linear1(h)))


class Denoiser(nn.Module):
    def __init__(self, E=256, layers=6, T=16, Tc=17, Da=12, Do=45):
        super().__init__()
        self.input_emb = nn.Linear(Da, E)
        self.cond_obs_emb = nn.Linear(Do - 3, E)
        self.cond_obs_emb_2 = nn.Linear(3, E)
        self.enc1, self.enc2 = nn.Linear(E, 4 * E), nn.Linear(4 * E, E)
        self.layers = nn.ModuleList(Layer(E) for _ in range(layers))
        self.ln_f, self.head = nn.LayerNorm(E), nn.Linear(E, Da)
        self.register_buffer("pos_emb", torch.zeros(T, E))
        self.register_buffer("cond_pos_emb", torch.zeros(Tc, E))
        self.register_buffer("tmask", torch.zeros(T, T))
        self.register_buffer("mmask", torch.zeros(T, Tc))

    def forward(self, x, temb, cond):
        c = cond[0]
        obs = torch.cat([c[:, :6], c[:, 9:]], dim=-1)
        ce = torch.cat([temb, self.cond_obs_emb(obs), self.cond_obs_emb_2(c[:, 6:9])], dim=0) + self.cond_pos_emb
        e = self.enc1(ce)
        mem = self.enc2(e * torch.tanh(F.softplus(e)))                    # Mish
        h = self.input_emb(x[0]) + self.pos_emb
        for layer in self.layers:
            h = layer(h, mem, self.tmask, self.mmask)
        return self.head(self.ln_f(h))[None]


def build_from(student) -> Denoiser:
    m = student.model
    d = Denoiser()
    sd = {"input_emb": m.input_emb, "cond_obs_emb": m.cond_obs_emb, "cond_obs_emb_2": m.cond_obs_emb_2,
          "enc1": m.encoder[0], "enc2": m.encoder[2], "ln_f": m.ln_f, "head": m.head}
    for k, src in sd.items():
        getattr(d, k).load_state_dict(src.state_dict())
    for i, L in enumerate(m.decoder.layers):
        D = d.layers[i]
        for a, b in (("sa", L.self_attn), ("ca", L.multihead_attn)):
            W, B = b.in_proj_weight.detach(), b.in_proj_bias.detach()
            E = W.shape[1]
            for j, n in enumerate("qkv"):
                lin = getattr(D, f"{a}_{n}")
                lin.weight.data.copy_(W[j * E:(j + 1) * E])
                lin.bias.data.copy_(B[j * E:(j + 1) * E])
            getattr(D, f"{a}_o").load_state_dict(b.out_proj.state_dict())
        for n in ("linear1", "linear2", "norm1", "norm2", "norm3"):
            getattr(D, n).load_state_dict(getattr(L, n).state_dict())
    d.pos_emb.copy_(m.pos_emb[0, :16])
    d.cond_pos_emb.copy_(m.cond_pos_emb[0, :17])
    d.tmask.copy_(m.mask)
    d.mmask.copy_(m.memory_mask)
    return d.eval()


_CACHE = {}


def _parts():
    if "d" not in _CACHE:
        s = _student()
        _CACHE["s"], _CACHE["d"] = s, build_from(s)
    return _CACHE["s"], _CACHE["d"]


def windows(n: int, offset: int = 0, stride: int = 37) -> list[torch.Tensor]:
    """n observation histories [8, 45] (raw), consecutive states inside one episode, spread over the data."""
    z = np.load(_need("MB_DLM_DATA", DATA))
    st, ends = z["state"], z["episode_ends"]
    starts = np.concatenate([[0], ends[:-1]])
    ok = [i for a, b in zip(starts, ends) for i in range(a, b - 8)]
    idx = [ok[(offset + k * stride * 101) % len(ok)] for k in range(n)]
    return [torch.from_numpy(st[i:i + 8].copy()) for i in idx]


def trajectory_inputs(hist_raw: torch.Tensor, seed: int, steps: int = 10):
    """[x_t, temb(t), cond] for every denoising step of the fp32 student on one history (cond normalised)."""
    s, d = _parts()
    cond = (hist_raw * s.o_scale + s.o_off)[None]
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((1, 16, 12), generator=g)
    out = []
    with torch.no_grad():
        for t in s.sched.timesteps(steps):
            te = temb(t)
            out.append([x.clone(), te, cond])
            x = s.sched.step(d(x, te, cond), t, x, steps, g)
    return out, x


def get_model():
    return _parts()[1]


def get_sample_input():
    return trajectory_inputs(windows(1, offset=5)[0], seed=1)[0][0]


def get_calibration_samples(n: int) -> list:
    out = []
    for i, h in enumerate(windows(n, offset=11)):
        out += trajectory_inputs(h, seed=100 + i)[0]
    return out


def get_int8_mixed_policy():
    env = os.environ.get("MB_DLM_INT8")
    if env == "keep_attn_out":
        return {"base": "linear", "alpha": 0.5, "keep_fp32": ["*_o", "head"]}
    return {"base": "linear", "alpha": float(os.environ.get("MB_DLM_ALPHA", 0.5)), "keep_fp32": ["head"]}


def check() -> dict:
    """re-stated denoiser vs the release module on held-out windows: max |eps difference|."""
    s, d = _parts()
    worst = 0.0
    with torch.no_grad():
        for i, h in enumerate(windows(4, offset=3)):
            for x, te, cond in trajectory_inputs(h, seed=i)[0]:
                t = int(round(float(torch.atan2(te[0, 0], te[0, 128]))))  # unused; recompute below
                ref = None
                for tt in range(10):
                    if torch.allclose(temb(tt), te):
                        ref = s.model(x, torch.tensor(tt), cond)
                worst = max(worst, float((d(x, te, cond) - ref).abs().max()))
    return {"max_abs_eps_diff": worst}


if __name__ == "__main__":
    print(check())
