"""SmolVLA (lerobot/smolvla_base) as three FX-traceable graphs for the fp32 flow.

``models/smolvla.py`` wraps the policy for torch.export (the int8 embedded
path). This module instead re-states SmolVLA's KV-cached inference
(``VLAFlowMatching.sample_actions``) as three plain ``nn.Module`` graphs, so
the FX extractor (``extract_graph --quant fp32``) lowers each to the generic
fp32 ops every GPU backend implements (linear / bmm / softmax / layer_norm /
ewise / strided_copy / catn / conv2d ...):

  vision  pixels [1,3,H,W] in [-1,1]   -> image tokens [1, Ni, 960]
          (SigLIP patch conv + 12 encoder layers + post-LN, pixel-shuffle
          connector, * sqrt(960))
  prefix  image tokens, language embeddings [1, L, 960] (embedding lookup
          * sqrt(960), done on the host), state [1, 32]
          -> KV cache [2 * n_layers, P, 320] (layer l: rows 2l = K after
          RoPE, 2l+1 = V), P = Ni + L + 1
  step    KV cache, x_t [1, 50, 32], time embedding [1, 720] (sinusoid of the
          flow time, host) -> v_t [1, 50, 32]

and the host runs the flow loop exactly as lerobot does:
``x_{t+dt} = x_t + dt * step(kv, x_t, emb(t))`` with dt = -1/num_steps from
t = 1. What stays on the host is what has no tensor compute: the token
embedding gather, the 720-wide time sinusoid, and the 5-step Euler update.

Shapes (image size, prompt length) are fixed at build time; the attention
masks, RoPE tables and SigLIP position embeddings for them become constant
buffers. Masks are additive -inf (exp(-inf) = 0: the masked probabilities
are exactly lerobot's zeros), and grouped-query attention groups the 3 query
heads of each KV head into the GEMM rows instead of expanding K/V.

Numerics follow lerobot 0.4.4 + transformers 4.57 in fp32 (``check()`` below
measures it against ``policy.model.sample_actions``).

``build(policy=None, ...)`` returns the three modules; with no policy they
get seeded random weights of the smolvla_base shapes (tests, op inventory).
As a ``--model`` for ``extract_graph``: set ``MB_SMOLVLA_PART`` to
vision | prefix | step (default vision); ``MB_SMOLVLA_REAL=1`` loads the
checkpoint (lerobot + HF cache), else random weights;
``MB_SMOLVLA_IMG`` (256), ``MB_SMOLVLA_LANG`` (prompt tokens, 16),
``MB_SMOLVLA_LAYERS`` (16) shrink it; ``MB_SMOLVLA_INPUTS`` (a torch.save'd
observation, see get_sample_input) makes the goldens a real observation's.
``--quant int8`` is an int8 IR with fp32 islands (get_int8_mixed_policy), its
smoothing calibrated on ``--num-calibration N`` observations from
``MB_SMOLVLA_CALIB`` (get_calibration_samples).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Cfg:
    img: int = 256
    lang: int = 16                    # prompt tokens (lerobot pads to the longest = no padding at batch 1)
    patch: int = 16
    pos_side: int = 32                # SigLIP position grid (image_size 512 / patch 16)
    v_hidden: int = 768
    v_layers: int = 12
    v_heads: int = 12
    v_inter: int = 3072
    v_eps: float = 1e-6
    scale_factor: int = 4
    hidden: int = 960
    inter: int = 2560
    heads: int = 15
    kv_heads: int = 5
    head_dim: int = 64
    rms_eps: float = 1e-5
    layers: int = 16
    e_hidden: int = 720
    e_inter: int = 2048
    state_dim: int = 32
    action_dim: int = 32
    chunk: int = 50
    self_attn_every: int = 2
    rope_base: float = 10000.0
    min_period: float = 4e-3
    max_period: float = 4.0

    @property
    def n_img(self) -> int:
        return (self.img // self.patch) ** 2 // self.scale_factor ** 2

    @property
    def prefix_len(self) -> int:
        return self.n_img + self.lang + 1

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim


class RMSNorm(nn.Module):
    """LlamaRMSNorm in fp32: w * x * rsqrt(mean(x^2) + eps)."""

    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        v = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(v + self.eps))


def rope_tables(positions: torch.Tensor, head_dim: int, base: float):
    """lerobot apply_rope's cos / sin for [L] positions -> [L, 1, head_dim/2] each."""
    d_half = head_dim // 2
    freq_exponents = (2.0 / head_dim) * torch.arange(d_half, dtype=torch.float32)
    timescale = base ** freq_exponents
    radians = positions[:, None].to(torch.float32) / timescale[None, :].to(torch.float32)
    return torch.cos(radians)[:, None, :], torch.sin(radians)[:, None, :]


def apply_rope(x, cos, sin, d_half: int):
    """x [1, L, H, D] (not interleaved: halves), as lerobot's apply_rope."""
    x1 = x[..., :d_half]
    x2 = x[..., d_half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def gqa_attention(q, k, v, heads: int, kv_heads: int, head_dim: int, bias=None):
    """softmax(q k^T / sqrt(d) + bias) v with grouped KV heads.

    q [1, S, heads, D]; k, v [1, T, kv_heads, D]. The g = heads / kv_heads query
    heads that share KV head h become the rows of one [g*S, D] x [D, T] product,
    exactly the eager attention's expand-then-matmul, without materialising the
    expanded K / V. bias: additive [S, T] (0 / -inf) or None. -> [1, S, heads*D]."""
    S, T, g = q.shape[1], k.shape[1], heads // kv_heads
    qh = q.reshape(S, kv_heads, g, head_dim).permute(1, 2, 0, 3).reshape(kv_heads, g * S, head_dim)
    kh = k.reshape(T, kv_heads, head_dim).permute(1, 2, 0)                  # [kv, D, T]
    vh = v.reshape(T, kv_heads, head_dim).permute(1, 0, 2)                  # [kv, T, D]
    s = torch.bmm(qh, kh) * (head_dim ** -0.5)                               # [kv, g*S, T]
    if bias is not None:
        s = (s.reshape(kv_heads, g, S, T) + bias).reshape(kv_heads, g * S, T)
    p = F.softmax(s, dim=-1)
    o = torch.bmm(p, vh)                                                     # [kv, g*S, D]
    return o.reshape(kv_heads, g, S, head_dim).permute(2, 0, 1, 3).reshape(1, S, heads * head_dim)


class Mlp(nn.Module):
    def __init__(self, d, inter):
        super().__init__()
        self.gate_proj = nn.Linear(d, inter, bias=False)
        self.up_proj = nn.Linear(d, inter, bias=False)
        self.down_proj = nn.Linear(inter, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------- vision
class SiglipLayer(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        d = c.v_hidden
        self.c = c
        self.layer_norm1 = nn.LayerNorm(d, eps=c.v_eps)
        self.layer_norm2 = nn.LayerNorm(d, eps=c.v_eps)
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(d, d) for _ in range(4))
        self.fc1 = nn.Linear(d, c.v_inter)
        self.fc2 = nn.Linear(c.v_inter, d)

    def forward(self, x):
        c, S = self.c, x.shape[1]
        hd = c.v_hidden // c.v_heads
        h = self.layer_norm1(x)
        q = self.q_proj(h).reshape(S, c.v_heads, hd).permute(1, 0, 2)       # [H, S, hd]
        k = self.k_proj(h).reshape(S, c.v_heads, hd).permute(1, 2, 0)       # [H, hd, S]
        v = self.v_proj(h).reshape(S, c.v_heads, hd).permute(1, 0, 2)
        p = F.softmax(torch.bmm(q, k) * (hd ** -0.5), dim=-1)
        a = torch.bmm(p, v).permute(1, 0, 2).reshape(1, S, c.v_hidden)
        x = x + self.out_proj(a)
        h = self.layer_norm2(x)
        return x + self.fc2(F.gelu(self.fc1(h), approximate="tanh"))


def siglip_position_ids(c: Cfg) -> torch.Tensor:
    """SmolVLMVisionEmbeddings' bucketized position ids for a full (unmasked) image."""
    nb = c.img // c.patch
    boundaries = torch.arange(1 / c.pos_side, 1.0, 1 / c.pos_side)
    idx = torch.arange(nb, dtype=torch.float32)
    frac = idx / nb * (1 - 1e-6)
    b = torch.bucketize(frac, boundaries, right=True)
    return (b[:, None] * c.pos_side + b).flatten()


class Vision(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.patch_embedding = nn.Conv2d(3, c.v_hidden, c.patch, c.patch)
        self.position_embedding = nn.Embedding(c.pos_side ** 2, c.v_hidden)
        self.layers = nn.ModuleList(SiglipLayer(c) for _ in range(c.v_layers))
        self.post_layernorm = nn.LayerNorm(c.v_hidden, eps=c.v_eps)
        self.proj = nn.Linear(c.v_hidden * c.scale_factor ** 2, c.hidden, bias=False)
        self.register_buffer("pos", torch.zeros((c.img // c.patch) ** 2, c.v_hidden), persistent=False)

    def finalize(self):
        """Gather the position embedding rows for this image size into the constant buffer."""
        with torch.no_grad():
            self.pos.copy_(self.position_embedding.weight[siglip_position_ids(self.c)])

    def forward(self, pixels):
        c = self.c
        x = self.patch_embedding(pixels).flatten(2).transpose(1, 2) + self.pos
        for layer in self.layers:
            x = layer(x)
        x = self.post_layernorm(x)
        # SmolVLMConnector.pixel_shuffle (scale_factor s)
        s, side = c.scale_factor, c.img // c.patch
        x = x.reshape(1, side, side // s, c.v_hidden * s).permute(0, 2, 1, 3)
        x = x.reshape(1, side // s, side // s, c.v_hidden * s * s).permute(0, 2, 1, 3)
        x = x.reshape(1, (side * side) // (s * s), c.v_hidden * s * s)
        return self.proj(x) * (c.hidden ** 0.5)


# ---------------------------------------------------------------- VLM prefix (KV-cache fill)
class Attn(nn.Module):
    def __init__(self, d_in, heads, kv_heads, head_dim, d_out, kv_in=None):
        super().__init__()
        self.q_proj = nn.Linear(d_in, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(kv_in or d_in, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(kv_in or d_in, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, d_out, bias=False)


class TextLayer(nn.Module):
    def __init__(self, d, inter, c: Cfg, kv_in=None):
        super().__init__()
        self.input_layernorm = RMSNorm(d, c.rms_eps)
        self.post_attention_layernorm = RMSNorm(d, c.rms_eps)
        self.self_attn = Attn(d, c.heads, c.kv_heads, c.head_dim, d, kv_in)
        self.mlp = Mlp(d, inter)


class Prefix(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.layers = nn.ModuleList(TextLayer(c.hidden, c.inter, c) for _ in range(c.layers))
        self.state_proj = nn.Linear(c.state_dim, c.hidden)
        P = c.prefix_len
        cos, sin = rope_tables(torch.arange(P), c.head_dim, c.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        # prefix-LM mask: image + language tokens attend each other, not the
        # state token (att mask 1); the state token attends everything
        bias = torch.zeros(P, P)
        bias[:P - 1, P - 1] = float("-inf")
        self.register_buffer("bias", bias, persistent=False)

    def forward(self, img_emb, lang_emb, state):
        c = self.c
        P, dh = c.prefix_len, c.head_dim // 2
        x = torch.cat([img_emb, lang_emb, self.state_proj(state).reshape(1, 1, c.hidden)], dim=1)
        kv = []
        for i, layer in enumerate(self.layers):
            h = layer.input_layernorm(x)
            a = layer.self_attn
            q = apply_rope(a.q_proj(h).reshape(1, P, c.heads, c.head_dim), self.cos, self.sin, dh)
            k = apply_rope(a.k_proj(h).reshape(1, P, c.kv_heads, c.head_dim), self.cos, self.sin, dh)
            v = a.v_proj(h)
            kv += [k.reshape(1, P, c.kv_dim), v.reshape(1, P, c.kv_dim)]
            if i == len(self.layers) - 1:
                break               # the last layer's output feeds nothing the expert reads
            o = gqa_attention(q, k, v.reshape(1, P, c.kv_heads, c.head_dim), c.heads, c.kv_heads, c.head_dim, self.bias)
            x = x + a.o_proj(o)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return torch.cat(kv, dim=0)                                          # [2*layers, P, kv_dim]


# ---------------------------------------------------------------- action expert (one flow step)
class Step(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.layers = nn.ModuleList(
            TextLayer(c.e_hidden, c.e_inter, c, kv_in=None if self._is_sa(i) else c.kv_dim)
            for i in range(c.layers))
        self.norm = RMSNorm(c.e_hidden, c.rms_eps)
        self.action_in_proj = nn.Linear(c.action_dim, c.e_hidden)
        self.action_out_proj = nn.Linear(c.e_hidden, c.action_dim)
        self.action_time_mlp_in = nn.Linear(2 * c.e_hidden, c.e_hidden)
        self.action_time_mlp_out = nn.Linear(c.e_hidden, c.e_hidden)
        P, A = c.prefix_len, c.chunk
        cos, sin = rope_tables(torch.arange(P, P + A), c.head_dim, c.rope_base)   # self-attn layers
        self.register_buffer("cos_sa", cos, persistent=False)
        self.register_buffer("sin_sa", sin, persistent=False)
        cos, sin = rope_tables(torch.arange(A), c.head_dim, c.rope_base)          # cross-attn layers
        self.register_buffer("cos_ca", cos, persistent=False)
        self.register_buffer("sin_ca", sin, persistent=False)
        # self-attn rows: every prefix key, then causal over the action tokens
        bias = torch.zeros(A, P + A)
        bias[:, P:] = torch.triu(torch.full((A, A), float("-inf")), diagonal=1)
        self.register_buffer("bias", bias, persistent=False)

    def _is_sa(self, i):
        return self.c.self_attn_every > 0 and i % self.c.self_attn_every == 0

    def forward(self, kv, x_t, time_emb):
        c = self.c
        P, A, dh = c.prefix_len, c.chunk, c.head_dim // 2
        a_emb = self.action_in_proj(x_t)                                     # [1, A, e]
        t_emb = time_emb.reshape(1, 1, c.e_hidden).expand(1, A, c.e_hidden)
        x = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(torch.cat([a_emb, t_emb], dim=2))))
        for i, layer in enumerate(self.layers):
            h = layer.input_layernorm(x)
            a = layer.self_attn
            k_cache, v_cache = kv[2 * i].reshape(1, P, c.kv_dim), kv[2 * i + 1].reshape(1, P, c.kv_dim)
            if self._is_sa(i):
                q = apply_rope(a.q_proj(h).reshape(1, A, c.heads, c.head_dim), self.cos_sa, self.sin_sa, dh)
                k = apply_rope(a.k_proj(h).reshape(1, A, c.kv_heads, c.head_dim), self.cos_sa, self.sin_sa, dh)
                k = torch.cat([k_cache, k.reshape(1, A, c.kv_dim)], dim=1)
                v = torch.cat([v_cache, a.v_proj(h)], dim=1)
                o = gqa_attention(q, k.reshape(1, P + A, c.kv_heads, c.head_dim),
                                  v.reshape(1, P + A, c.kv_heads, c.head_dim), c.heads, c.kv_heads, c.head_dim, self.bias)
            else:
                q = apply_rope(a.q_proj(h).reshape(1, A, c.heads, c.head_dim), self.cos_ca, self.sin_ca, dh)
                k = a.k_proj(k_cache).reshape(1, P, c.kv_heads, c.head_dim)
                v = a.v_proj(v_cache).reshape(1, P, c.kv_heads, c.head_dim)
                o = gqa_attention(q, k, v, c.heads, c.kv_heads, c.head_dim)
            x = x + a.o_proj(o)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.action_out_proj(self.norm(x))


# ---------------------------------------------------------------- host side
def time_embedding(t: float, c: Cfg) -> torch.Tensor:
    """lerobot create_sinusoidal_pos_embedding(t, e_hidden, min, max) (float64 -> fp32) -> [1, e_hidden]."""
    fraction = torch.linspace(0.0, 1.0, c.e_hidden // 2, dtype=torch.float64)
    period = c.min_period * (c.max_period / c.min_period) ** fraction
    x = (1.0 / period * 2 * math.pi)[None, :] * torch.tensor([t], dtype=torch.float64)[:, None]
    return torch.cat([torch.sin(x), torch.cos(x)], dim=1).to(torch.float32)


def flow_loop(step_fn, kv, noise, num_steps: int, c: Cfg):
    """lerobot's Euler integration from t = 1 to 0; step_fn(kv, x_t, time_emb) -> v_t."""
    dt = -1.0 / num_steps
    x = noise
    for s in range(num_steps):
        x = x + dt * step_fn(kv, x, time_embedding(1.0 + s * dt, c))
    return x


@dataclass
class Parts:
    cfg: Cfg
    vision: Vision
    prefix: Prefix
    step: Step
    embed: torch.Tensor | None = None    # token embedding table [vocab, hidden] (real weights only)

    def lang_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embed[tokens.reshape(-1)].reshape(1, -1, self.cfg.hidden) * math.sqrt(self.cfg.hidden)

    @torch.no_grad()
    def sample_actions(self, pixels, lang_emb, state, noise, num_steps):
        kv = self.prefix(self.vision(pixels), lang_emb, state)
        return flow_loop(self.step, kv, noise, num_steps, self.cfg)


def _copy(dst: nn.Module, src: nn.Module):
    missing = dst.load_state_dict({k: v.float() for k, v in src.state_dict().items()}, strict=False)
    bad = [k for k in missing.missing_keys if not k.startswith(("pos", "cos", "sin", "bias"))]
    if bad or missing.unexpected_keys:
        raise RuntimeError(f"weight copy {type(src).__name__} -> {type(dst).__name__}: "
                           f"missing {bad[:5]} unexpected {missing.unexpected_keys[:5]}")


def cfg_from_policy(policy, img: int, lang: int) -> Cfg:
    vwe = policy.model.vlm_with_expert
    tc, vc = vwe.config.text_config, vwe.config.vision_config
    return Cfg(img=img, lang=lang, patch=vc.patch_size, pos_side=vc.image_size // vc.patch_size,
               v_hidden=vc.hidden_size, v_layers=vc.num_hidden_layers, v_heads=vc.num_attention_heads,
               v_inter=vc.intermediate_size, v_eps=vc.layer_norm_eps, scale_factor=vwe.config.scale_factor,
               hidden=tc.hidden_size, inter=tc.intermediate_size, heads=tc.num_attention_heads,
               kv_heads=tc.num_key_value_heads, head_dim=tc.head_dim, rms_eps=tc.rms_norm_eps,
               layers=vwe.num_vlm_layers, e_hidden=vwe.expert_hidden_size,
               e_inter=vwe.lm_expert.config.intermediate_size,
               state_dim=policy.config.max_state_dim, action_dim=policy.config.max_action_dim,
               chunk=policy.config.chunk_size, self_attn_every=policy.config.self_attn_every_n_layers,
               min_period=policy.config.min_period, max_period=policy.config.max_period)


def build(policy=None, cfg: Cfg | None = None, seed: int = 0) -> Parts:
    """The three graphs, with the checkpoint's weights (policy: a lerobot SmolVLAPolicy) or seeded random ones."""
    c = cfg or Cfg()
    torch.manual_seed(seed)
    vision, prefix, step = Vision(c), Prefix(c), Step(c)
    embed = None
    if policy is not None:
        vwe = policy.model.vlm_with_expert
        vm = vwe.get_vlm_model()
        ve = vm.vision_model
        vision.patch_embedding.load_state_dict({k: v.float() for k, v in ve.embeddings.patch_embedding.state_dict().items()})
        vision.position_embedding.load_state_dict({k: v.float() for k, v in ve.embeddings.position_embedding.state_dict().items()})
        for dst, src in zip(vision.layers, ve.encoder.layers):
            sd = {k.replace("self_attn.", "").replace("mlp.", ""): v.float() for k, v in src.state_dict().items()}
            dst.load_state_dict(sd)
        vision.post_layernorm.load_state_dict({k: v.float() for k, v in ve.post_layernorm.state_dict().items()})
        vision.proj.load_state_dict({k: v.float() for k, v in vm.connector.modality_projection.proj.state_dict().items()})
        for dst, src in zip(prefix.layers, vm.text_model.layers):
            _copy(dst, src)
        prefix.state_proj.load_state_dict({k: v.float() for k, v in policy.model.state_proj.state_dict().items()})
        for dst, src in zip(step.layers, vwe.lm_expert.layers):
            _copy(dst, src)
        step.norm.load_state_dict({k: v.float() for k, v in vwe.lm_expert.norm.state_dict().items()})
        for name in ("action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"):
            getattr(step, name).load_state_dict({k: v.float() for k, v in getattr(policy.model, name).state_dict().items()})
        embed = vm.text_model.get_input_embeddings().weight.detach().float()
    else:
        with torch.no_grad():   # keep random activations O(1) through 16 layers
            for m in list(vision.modules()) + list(prefix.modules()) + list(step.modules()):
                if isinstance(m, nn.Linear):
                    m.weight.mul_(0.5)
    vision.finalize()
    for m in (vision, prefix, step):
        m.eval()
    return Parts(c, vision, prefix, step, embed)


# ---------------------------------------------------------------- extract_graph --model smolvla_split
def _env_cfg() -> Cfg:
    c = Cfg(img=int(os.environ.get("MB_SMOLVLA_IMG", 256)), lang=int(os.environ.get("MB_SMOLVLA_LANG", 16)))
    if os.environ.get("MB_SMOLVLA_LAYERS"):
        c = replace(c, layers=int(os.environ["MB_SMOLVLA_LAYERS"]))
    if os.environ.get("MB_SMOLVLA_VLAYERS"):
        c = replace(c, v_layers=int(os.environ["MB_SMOLVLA_VLAYERS"]))
    return c


_CACHE: dict = {}


def _parts() -> Parts:
    if "p" not in _CACHE:
        c = _env_cfg()
        if os.environ.get("MB_SMOLVLA_REAL"):
            policy = load_policy(c.img)
            _CACHE["p"] = build(policy, cfg_from_policy(policy, c.img, c.lang))
        else:
            _CACHE["p"] = build(None, c)
    return _CACHE["p"]


def _part() -> str:
    return os.environ.get("MB_SMOLVLA_PART", "vision")


def get_model():
    return getattr(_parts(), _part())


def get_sample_input():
    p = _parts()
    c = p.cfg
    g = torch.Generator().manual_seed(1)
    part = _part()
    if os.environ.get("MB_SMOLVLA_INPUTS"):
        # a real observation (torch.save'd dict: pixels [1,3,H,W] in [-1,1], lang
        # [1,L,hidden] = embedded prompt, state [1,32], noise [1,50,32]); each part's
        # golden is then that observation's actual intermediate
        r = torch.load(os.environ["MB_SMOLVLA_INPUTS"])
        if part == "vision":
            return r["pixels"].float()
        img = p.vision(r["pixels"].float()).detach()
        if part == "prefix":
            return [img, r["lang"].float(), r["state"].float()]
        kv = p.prefix(img, r["lang"].float(), r["state"].float()).detach()
        return [kv, r["noise"].float(), time_embedding(1.0, c)]
    if part == "vision":
        return torch.rand(1, 3, c.img, c.img, generator=g) * 2 - 1
    if part == "prefix":
        img = p.vision(torch.rand(1, 3, c.img, c.img, generator=g) * 2 - 1).detach()
        lang = torch.randn(1, c.lang, c.hidden, generator=g) * 0.05 * math.sqrt(c.hidden)
        return [img, lang, torch.randn(1, c.state_dim, generator=g)]
    kv = p.prefix(p.vision(torch.rand(1, 3, c.img, c.img, generator=g) * 2 - 1),
                  torch.randn(1, c.lang, c.hidden, generator=g) * 0.05 * math.sqrt(c.hidden),
                  torch.randn(1, c.state_dim, generator=g)).detach()
    return [kv, torch.randn(1, c.chunk, c.action_dim, generator=g), time_embedding(1.0, c)]


def _fit_lang(lang: torch.Tensor, L: int) -> torch.Tensor:
    """A prompt embedding [1, L', hidden] cycled / cut to the graphs' L tokens (calibration only:
    the shapes are fixed at build time, and the activation ranges need representative tokens,
    not this exact prompt)."""
    return lang[:, torch.arange(L) % lang.shape[1]]


def _calib_obs(n: int) -> list[dict]:
    """MB_SMOLVLA_CALIB: a directory of torch.save'd observations (as MB_SMOLVLA_INPUTS; *.pt,
    sorted), or none: n seeded random ones."""
    c = _parts().cfg
    src = os.environ.get("MB_SMOLVLA_CALIB")
    if src:
        from pathlib import Path
        files = sorted(Path(src).glob("*.pt"))[:n]
        if not files:
            raise SystemExit(f"MB_SMOLVLA_CALIB={src}: no *.pt observations")
        return [torch.load(f) for f in files]
    out = []
    for i in range(n):
        g = torch.Generator().manual_seed(1000 + i)
        out.append({"pixels": torch.rand(1, 3, c.img, c.img, generator=g) * 2 - 1,
                    "lang": torch.randn(1, c.lang, c.hidden, generator=g) * 0.05 * math.sqrt(c.hidden),
                    "state": torch.randn(1, c.state_dim, generator=g),
                    "noise": torch.randn(1, c.chunk, c.action_dim, generator=g)})
    return out


def get_calibration_samples(n: int) -> list:
    """int8 calibration inputs of the current part (MB_SMOLVLA_PART) from n observations
    (_calib_obs): vision -> pixels; prefix -> [image tokens, prompt, state]; step -> one sample
    per flow step of each observation's own fp32 trajectory ([kv, x_t, time_emb(t)]), so every
    time t the deployed loop visits is calibrated (MB_SMOLVLA_STEPS, default 5)."""
    p, part = _parts(), _part()
    c = p.cfg
    steps = int(os.environ.get("MB_SMOLVLA_STEPS", 5))
    out = []
    with torch.no_grad():
        for r in _calib_obs(n):
            pix = r["pixels"].float().reshape(1, 3, c.img, c.img)
            if part == "vision":
                out.append(pix)
                continue
            lang, state = _fit_lang(r["lang"].float(), c.lang), r["state"].float()
            img = p.vision(pix)
            if part == "prefix":
                out.append([img, lang, state])
                continue
            kv = p.prefix(img, lang, state)
            dt, x = -1.0 / steps, r["noise"].float()
            for s_ in range(steps):
                te = time_embedding(1.0 + s_ * dt, c)
                out.append([kv, x.clone(), te])
                x = x + dt * p.step(kv, x, te)
    return out


# extract_graph --quant int8 (pipeline/int8_mixed.py: linears -> dynamic per-token int8 activations,
# SmoothQuant factors, per-channel int8 weights). Chosen on 13 held-out observations against
# lerobot's sample_actions (docs/vulkan_amd.md, "int8"): the prefix's KV cache feeds every layer of
# every flow step, and int8 there costs the most accuracy for the least time (its linears are ~7%
# of the chunk), so it stays fp32; SigLIP's fc2 (input: the GELU output, outlier channels) too.
INT8_POLICY = {
    "vision": {"base": "linear", "alpha": 0.8, "keep_fp32": ["*.fc2"]},
    "prefix": {"base": "linear", "alpha": 0.8, "keep_fp32": ["*"]},
    "step": {"base": "linear", "alpha": 0.65, "keep_fp32": []},
}
# every linear int8 (MB_SMOLVLA_INT8=all): the fastest mix, measured too
INT8_ALL = {"vision": {"base": "linear", "alpha": 0.8}, "prefix": {"base": "linear", "alpha": 0.65},
            "step": {"base": "linear", "alpha": 0.65}}


def get_int8_mixed_policy():
    """The current part's int8 policy (pipeline/int8_mixed.py). MB_SMOLVLA_INT8: 'all' (INT8_ALL),
    or a policy (name / JSON dict) for this part."""
    env = os.environ.get("MB_SMOLVLA_INT8")
    if env == "all":
        return dict(INT8_ALL[_part()])
    return env or dict(INT8_POLICY[_part()])


def load_policy(img: int = 256, model_id: str = "lerobot/smolvla_base"):
    """lerobot SmolVLAPolicy in fp32 on the CPU, all weights from the policy checkpoint
    (load_vlm_weights=False: the SmolVLM2 backbone is built from its config only)."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    cfg = PreTrainedConfig.from_pretrained(model_id)
    cfg.device = "cpu"
    cfg.load_vlm_weights = False
    cfg.resize_imgs_with_padding = (img, img)
    policy = SmolVLAPolicy.from_pretrained(model_id, config=cfg).to(torch.float32).eval()
    return policy
