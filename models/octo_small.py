"""Octo — generalist robot policy (`octo-small-1.5`), PyTorch reimplementation.

Upstream Octo is JAX/Flax (`octo/octo/model/`). ModelBlaster traces PyTorch, so
this is a from-scratch PyTorch reimplementation of the *policy* whose weights
are converted out of the published Flax checkpoint by
``experiments/octo_port/convert_weights.py``. There is no shortcut through the
upstream code: of the upstream tree's files, 36 import jax and exactly 1
imports torch.

Full scoping — variant, measured param counts, token budget, the numbers below
and what was deliberately left out — lives in
``experiments/octo_port/NOTES.md``. Read that first.

Architecture, as read out of the checkpoint's own ``config.json`` (NOT from the
paper):

    token_embedding_size = 384      # ViT-S
    transformer           = 12 layers, 6 heads, mlp_dim 1536, pre-LN
    window_size (horizon) = 2       (max_horizon 10 -> pos-emb tables are [1,10,...])
    obs tokenizers        = primary (256x256) + wrist (128x128), both
                            ImageTokenizer + SmallStem16
    task tokenizer        = language, T5-base embeddings  (16, 768)
    repeat_task_tokens    = True    # language tokens ALSO tiled per timestep
    readouts              = {"action": 1}
    head                  = DiffusionActionHead(action_dim=7, action_horizon=4,
                                                diffusion_steps=20)
    use_correct_attention = True
    dropout               = 0.0 everywhere -> eval graph == train graph

Token budget per timestep: 256 (primary) + 64 (wrist) + 16 (language, tiled)
+ 1 (readout) = 337. Sequence = 16 prefix + 2*337 = **690 tokens**.

THE T5 TEXT ENCODER IS NOT PART OF THIS MODEL, BY DESIGN.
It is 109,628,544 of the checkpoint's 136,670,604 parameters and it is frozen
(`finetune_encoder=False`). A robot running a fixed task has a fixed
instruction, so its embedding is a constant: computed once offline and handed
to the policy as a `(B, 16, 768)` input tensor. Upstream's `LanguageTokenizer`
already accepts a raw array on exactly this path, so this is the deployment
contract and not an approximation. The port is COMPLETE without it. See
NOTES.md §7.

Three things here are easy to get quietly wrong; all three are handled and
each has a numeric check in ``experiments/octo_port/validate.py``:

  1. `StdConv` — the stems' convs are weight-standardised: upstream overrides
     parameter *read* to return `(w - mean(w)) / (std(w) + 1e-5)` reduced over
     the kernel and input-channel axes. For a frozen checkpoint that is a pure
     function of the weights, so the CONVERTER folds it in and these are plain
     `nn.Conv2d` here. A naive "transpose HWIO->OIHW" conversion without the
     fold is wrong by a per-output-channel scale and shift, and still *looks*
     plausible.
  2. epsilons — flax `LayerNorm`/`GroupNorm` default to **1e-6**, torch to
     1e-5. Both are pinned to 1e-6 below.
  3. gelu — flax's `nn.gelu` is `jax.nn.gelu` whose default is
     `approximate=True`, i.e. the **tanh** approximation. torch's `F.gelu`
     defaults to exact. `approximate="tanh"` is pinned below.

Also: the stems take **6** input channels, not 3. `ImageTokenizer` concatenates
the goal image onto the observation along channels, substituting zeros when
there is no goal image; `normalize_images` (`x/127.5 - 1`) then maps that zero
half to **-1.0**. Confirmed by `StdConv_0/kernel` being `[3,3,6,32]`.

Shape / lowering knobs, same convention as the other models here:

  MODELBLASTER_OCTO_WINDOW     default 2. Observation history length. <= 10.
                               window=1 halves the timestep tokens and is the
                               single biggest latency knob (see NOTES.md §6).
  MODELBLASTER_OCTO_PRIMARY    default 256. Primary camera resolution. Must be
                               a multiple of 16 (four stride-2 convs then a
                               1x1). Tokens per timestep = (N/16)^2, so this
                               dominates the sequence length.
  MODELBLASTER_OCTO_WRIST      default 128. Wrist camera resolution, same
                               constraint. 0 disables the wrist tokenizer
                               entirely (-64 tokens/timestep).
  MODELBLASTER_OCTO_LAYERS     default 12. Truncates the transformer stack.
                               For latency scaling studies only -- any value
                               other than 12 discards trained weights.
  MODELBLASTER_OCTO_PART       default "full". Which graph get_model() returns:
                                 "full"     backbone + one score-net step
                                            (one denoising evaluation)
                                 "backbone" observations -> readout embedding
                                 "score"    (embedding, noisy_action, t) -> eps
                               The runtime wants backbone x1 + score x20, so
                               "backbone" and "score" are the two graphs you
                               actually schedule; "full" is the single
                               self-contained graph for extraction + drift.
  MODELBLASTER_OCTO_GN         default "native" (nn.GroupNorm). "layernorm"
                               lowers each GroupNorm to reshape + layer_norm +
                               affine mul/add, which is mathematically
                               identical and lands on ops ModelBlaster already
                               has -- `group_norm` is NOT in its vocabulary.
  MODELBLASTER_OCTO_ATTN       default "sdpa" (F.scaled_dot_product_attention,
                               which extract_graph_export decomposes into
                               matmul/softmax/matmul). "matmul" writes the
                               attention out explicitly instead.
  MODELBLASTER_OCTO_GOAL       default 0 (language-conditioned: the goal-image
                               half of the stem input is the constant -1.0
                               that upstream's zero-fill normalises to). 1 adds
                               goal_primary/goal_wrist forward inputs.
  MODELBLASTER_OCTO_NORM       default 1: the /127.5 - 1 image normalisation
                               is in the graph, as it is upstream. 0 expects
                               input already in [-1, 1] (what a camera driver
                               or the harness would hand you) and removes the
                               `div` + `sub` nodes -- `sub` being one of only
                               two op kinds in this port that the export
                               extractor does not know.
                               NOTE the normalisation canNOT be folded into
                               conv0's weights: the scale folds cleanly but the
                               -1 shift does not, for the same zero-padding
                               reason the goal-channel fold fails (see
                               ImageTokenizer).
  MODELBLASTER_OCTO_TIME       default "fourier" (faithful: the score net takes
                               the raw timestep and runs FourierFeatures + the
                               cond MLP). "lut" makes it take the precomputed
                               32-d conditioning vector instead, which the host
                               looks up from a 20x32 table -- exact, and it
                               removes the `cos`/`sin` ops, the only other pair
                               neither extractor covers. See
                               ScoreNet.cond_table.
  MODELBLASTER_OCTO_MASK_NEG   default 10.0. Magnitude of the additive
                               attention-mask bias in disallowed slots.
                               Upstream flax uses finfo(float32).min; that
                               value cannot be quantized alongside the logits
                               (see the comment where mask_bias is built), so
                               a finite magnitude is used instead. Trade-off:
                               the value bounds both the residual weight on
                               masked slots (exp(-mask_neg)) and the int8
                               resolution of the real logits (mask_neg/127).

  MODELBLASTER_OCTO_CKPT       path to the converted PyTorch state_dict
                               (`experiments/octo_port/octo_small_torch.pt`).
                               WITHOUT IT THE WEIGHTS ARE RANDOM and every
                               number the model produces is meaningless -- it
                               is then a shape/throughput benchmark only.

No `.shape` is read inside any `forward` -- every shape is a compile-time
constant off the config -- so the module traces cleanly under BOTH
`torch.fx.symbolic_trace` and `torch.export`.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

MAX_HORIZON = 10          # size of the checkpoint's positional-embedding tables
T5_HIDDEN = 768           # t5-base last_hidden_state width
LANG_TOKENS = 16          # HFTokenizer(max_length=16, padding="max_length")
EMBED_DIM = 384           # token_embedding_size (ViT-S)
NUM_HEADS = 6
MLP_DIM = 1536
STEM_FEATURES = (32, 96, 192, 384)
STEM_NUM_FEATURES = 512   # SmallStem "embedding" 1x1 conv output width
GROUP_NORM_GROUPS = 32    # flax nn.GroupNorm default
NORM_EPS = 1.0e-6         # flax LayerNorm/GroupNorm default (torch's is 1e-5)
STD_CONV_EPS = 1.0e-5     # StdConv weight_standardize eps

ACTION_DIM = 7
ACTION_HORIZON = 4
DIFFUSION_STEPS = 20
TIME_DIM = 32
SCORE_HIDDEN = 256
SCORE_BLOCKS = 3
MAX_ACTION = 5.0


def _cfg() -> dict:
    window = int(os.environ.get("MODELBLASTER_OCTO_WINDOW", 2))
    primary = int(os.environ.get("MODELBLASTER_OCTO_PRIMARY", 256))
    wrist = int(os.environ.get("MODELBLASTER_OCTO_WRIST", 128))
    layers = int(os.environ.get("MODELBLASTER_OCTO_LAYERS", 12))
    part = os.environ.get("MODELBLASTER_OCTO_PART", "full")
    gn = os.environ.get("MODELBLASTER_OCTO_GN", "native")
    attn = os.environ.get("MODELBLASTER_OCTO_ATTN", "sdpa")
    goal = os.environ.get("MODELBLASTER_OCTO_GOAL", "0") == "1"
    norm = os.environ.get("MODELBLASTER_OCTO_NORM", "1") == "1"
    time_mode = os.environ.get("MODELBLASTER_OCTO_TIME", "fourier")
    mask_neg = float(os.environ.get("MODELBLASTER_OCTO_MASK_NEG", 10.0))

    if not 1 <= window <= MAX_HORIZON:
        raise ValueError(
            f"MODELBLASTER_OCTO_WINDOW={window} must be in 1..{MAX_HORIZON}: the "
            f"checkpoint's positional-embedding tables are [1,{MAX_HORIZON},n,384] "
            f"and are sliced to the window, so a longer window has no weights.")
    if primary % 16 or primary <= 0:
        raise ValueError(
            f"MODELBLASTER_OCTO_PRIMARY={primary} must be a positive multiple of 16: "
            f"SmallStem16 applies four stride-2 convs and then a 1x1, so any other "
            f"size does not land on an integer token grid.")
    if wrist and (wrist % 16):
        raise ValueError(
            f"MODELBLASTER_OCTO_WRIST={wrist} must be 0 (disabled) or a multiple of 16.")
    if not 1 <= layers <= 12:
        raise ValueError(f"MODELBLASTER_OCTO_LAYERS={layers} must be in 1..12.")
    if part not in ("full", "backbone", "score"):
        raise ValueError(
            f"MODELBLASTER_OCTO_PART={part!r} must be one of full/backbone/score.")
    if gn not in ("native", "layernorm"):
        raise ValueError(f"MODELBLASTER_OCTO_GN={gn!r} must be native or layernorm.")
    if attn not in ("sdpa", "matmul"):
        raise ValueError(f"MODELBLASTER_OCTO_ATTN={attn!r} must be sdpa or matmul.")
    if time_mode not in ("fourier", "lut"):
        raise ValueError(
            f"MODELBLASTER_OCTO_TIME={time_mode!r} must be fourier or lut.")
    if not mask_neg > 0.0:
        raise ValueError(
            f"MODELBLASTER_OCTO_MASK_NEG={mask_neg} must be positive; it is a "
            f"magnitude and is negated to build the additive bias.")

    n_primary = (primary // 16) ** 2
    n_wrist = (wrist // 16) ** 2 if wrist else 0
    return dict(window=window, primary=primary, wrist=wrist, layers=layers,
                part=part, gn=gn, attn=attn, goal=goal, norm=norm,
                time_mode=time_mode, mask_neg=mask_neg,
                n_primary=n_primary, n_wrist=n_wrist)


# --------------------------------------------------------------------------
# Norms
# --------------------------------------------------------------------------

class GroupNormLN(nn.Module):
    """GroupNorm expressed as reshape + layer_norm + per-channel affine.

    `group_norm` is not in ModelBlaster's op vocabulary (`extract_graph_export`
    has `layer_norm.default`, not `group_norm`). It does not need to be: flax's
    GroupNorm normalises over the channels *within* a group and the spatial
    axes together, which for `(N, C, H, W)` with `G` groups is exactly a
    LayerNorm over the trailing `(C/G)*H*W` of a `(N, G, (C/G)*H*W)` view.
    Affine is applied afterwards per channel, so it stays a `mul` + `add`.

    Identical arithmetic to `nn.GroupNorm`, on ops that already have kernels.
    """

    def __init__(self, num_groups: int, num_channels: int, hw: int,
                 eps: float = NORM_EPS):
        super().__init__()
        if num_channels % num_groups:
            raise ValueError(f"{num_channels} channels not divisible into "
                             f"{num_groups} groups")
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.hw = hw                      # H (== W); kept static, never read off the tensor
        self.group_numel = (num_channels // num_groups) * hw * hw
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.reshape(-1, self.num_groups, self.group_numel)
        y = F.layer_norm(y, (self.group_numel,), None, None, self.eps)
        y = y.reshape(-1, self.num_channels, self.hw, self.hw)
        return y * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


def _make_group_norm(channels: int, hw: int, mode: str) -> nn.Module:
    if mode == "layernorm":
        return GroupNormLN(GROUP_NORM_GROUPS, channels, hw, NORM_EPS)
    return nn.GroupNorm(GROUP_NORM_GROUPS, channels, eps=NORM_EPS)


# --------------------------------------------------------------------------
# SmallStem16  (octo/model/components/vit_encoders.py::SmallStem)
# --------------------------------------------------------------------------

class SmallStem16(nn.Module):
    """4 x [StdConv 3x3/s2/p1 -> GroupNorm -> relu] then a 1x1 "embedding" conv.

    `patch_size=16` and the "patchify" conv is `kernel_size = patch_size//16 = 1`,
    so the final stage is pointwise -- the four stride-2 convs have already done
    the 16x downsample. Input resolution N gives an (N/16, N/16) token grid.

    The convs are plain `nn.Conv2d`: StdConv's weight standardisation is folded
    into the weights by the converter (see module docstring, item 1).
    """

    def __init__(self, in_channels: int, resolution: int, gn_mode: str):
        super().__init__()
        convs, norms = [], []
        cin, hw = in_channels, resolution
        for feat in STEM_FEATURES:
            convs.append(nn.Conv2d(cin, feat, kernel_size=3, stride=2,
                                   padding=1, bias=True))
            hw = hw // 2
            norms.append(_make_group_norm(feat, hw, gn_mode))
            cin = feat
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.embedding = nn.Conv2d(cin, STEM_NUM_FEATURES, kernel_size=1,
                                   stride=1, padding=0, bias=True)
        self.out_hw = hw
        self.num_tokens = hw * hw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, in_channels, R, R) -> (N, num_tokens, 512)."""
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x)))
        x = self.embedding(x)
        # NCHW -> token-major with W fastest, matching upstream's NHWC
        # reshape(b, t, -1, C).
        x = x.permute(0, 2, 3, 1).reshape(-1, self.num_tokens, STEM_NUM_FEATURES)
        return x


class ImageTokenizer(nn.Module):
    """Stem + the observation/goal channel stack, per upstream ImageTokenizer.

    Upstream concatenates the goal image onto the observation along channels and
    zero-fills it when there is no goal image; `normalize_images` then maps that
    zero half to the constant **-1.0**. So the stem always sees **6** input
    channels -- confirmed by `StdConv_0/kernel` being `[3,3,6,32]`.

    The constant half is supplied by a `(1, 3, 1, 1)` buffer broadcast with
    `expand_as`, which costs 3 floats of weights and lands on ops that already
    exist (`expand_as` is a free alias, `cat` is supported).

    NOTE, because it is a trap worth recording: folding that constant into the
    first conv's bias -- `conv([obs, -1]) + b == conv_3ch(obs) + b - sum(W[:,3:])`
    -- is WRONG for these convs. They use `padding=1`, so in the pad ring the
    "constant -1" input is actually **0**, and the constant's contribution is
    therefore spatially varying rather than a per-channel bias. It is exact only
    on the interior. The first attempt at this port did fold it, and the
    signature was instructive: cosine similarity stayed at 0.998 while
    `max_abs` hit 1.0e+01 on a tensor whose own max was 3.6e+01, with the SAME
    max_abs for two completely different inputs -- the tell-tale of a constant
    offset on a border. `validate.py` check [2] pins this down.
    """

    def __init__(self, resolution: int, window: int, gn_mode: str, goal: bool,
                 normalize: bool = True):
        super().__init__()
        self.resolution = resolution
        self.window = window
        self.goal = goal
        self.normalize = normalize
        self.stem = SmallStem16(6, resolution, gn_mode)
        self.num_tokens = self.stem.num_tokens
        if not goal:
            # normalize_images(0) == 0/127.5 - 1 == -1.0
            self.register_buffer("goal_const",
                                 torch.full((1, 3, 1, 1), -1.0), persistent=False)

    def forward(self, images: torch.Tensor,
                goal: Optional[torch.Tensor] = None) -> torch.Tensor:
        """images (B, W, 3, R, R) -> (B, W, num_tokens, 512).

        In [0, 255] when `normalize` (the default); already in [-1, 1] when
        `MODELBLASTER_OCTO_NORM=0`, which is the deployment-realistic form --
        a camera driver or the harness does the scaling -- and which removes
        the `div`/`sub` nodes from the graph.
        """
        x = images.reshape(-1, 3, self.resolution, self.resolution)
        if self.normalize:
            x = x / 127.5 - 1.0
        if self.goal:
            if goal is None:
                raise ValueError("MODELBLASTER_OCTO_GOAL=1 but no goal image passed")
            g = goal / 127.5 - 1.0 if self.normalize else goal
            # upstream: task_inputs[:, None].repeat(horizon, axis=1). Written as
            # a cat of a static number of copies rather than `expand`/`repeat`,
            # neither of which is in the export extractor's alias set.
            g = torch.cat([g.unsqueeze(1)] * self.window, dim=1)
            g = g.reshape(-1, 3, self.resolution, self.resolution)
        else:
            g = self.goal_const.expand_as(x)
        x = torch.cat([x, g], dim=1)
        tokens = self.stem(x)
        return tokens.reshape(-1, self.window, self.num_tokens, STEM_NUM_FEATURES)


# --------------------------------------------------------------------------
# Transformer  (octo/model/components/transformer.py)
# --------------------------------------------------------------------------

class MultiHeadDotProductAttention(nn.Module):
    """flax `nn.MultiHeadDotProductAttention`, self-attention only.

    flax stores q/k/v kernels as `(features, num_heads, head_dim)` and the out
    kernel as `(num_heads, head_dim, features)`; the converter flattens the head
    axes and transposes into torch `Linear` layout. Queries are scaled by
    1/sqrt(head_dim), which is also what `scaled_dot_product_attention` does.

    Masking uses an ADDITIVE bias of `finfo(float32).min` in disallowed slots,
    matching flax's `jnp.where(mask, logits, big_neg)` -- both leave exactly
    zero weight after the softmax. A bool mask would work too; the float form
    keeps the graph to `add` + `softmax`, both already supported ops.
    """

    def __init__(self, dim: int = EMBED_DIM, num_heads: int = NUM_HEADS,
                 attn_mode: str = "sdpa"):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} not divisible by {num_heads} heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.attn_mode = attn_mode
        self.query = nn.Linear(dim, dim, bias=True)
        self.key = nn.Linear(dim, dim, bias=True)
        self.value = nn.Linear(dim, dim, bias=True)
        self.out = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor, mask_bias: torch.Tensor) -> torch.Tensor:
        h, d = self.num_heads, self.head_dim
        q = self.query(x).unflatten(-1, (h, d)).transpose(1, 2)
        k = self.key(x).unflatten(-1, (h, d)).transpose(1, 2)
        v = self.value(x).unflatten(-1, (h, d)).transpose(1, 2)
        if self.attn_mode == "sdpa":
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_bias)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            scores = scores + mask_bias
            o = torch.matmul(F.softmax(scores, dim=-1), v)
        o = o.transpose(1, 2).flatten(-2)
        return self.out(o)


class MlpBlock(nn.Module):
    """flax MlpBlock: Dense -> gelu(tanh) -> Dense. Dropout is 0.0, so omitted."""

    def __init__(self, dim: int = EMBED_DIM, mlp_dim: int = MLP_DIM):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # approximate="tanh": flax's nn.gelu is jax.nn.gelu, default
        # approximate=True. torch's default is the exact erf form.
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class Encoder1DBlock(nn.Module):
    """Pre-LN transformer encoder block."""

    def __init__(self, dim: int = EMBED_DIM, num_heads: int = NUM_HEADS,
                 mlp_dim: int = MLP_DIM, attn_mode: str = "sdpa"):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=NORM_EPS)
        self.attn = MultiHeadDotProductAttention(dim, num_heads, attn_mode)
        self.norm2 = nn.LayerNorm(dim, eps=NORM_EPS)
        self.mlp = MlpBlock(dim, mlp_dim)

    def forward(self, x: torch.Tensor, mask_bias: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask_bias)
        return x + self.mlp(self.norm2(x))


class Transformer(nn.Module):
    def __init__(self, num_layers: int = 12, dim: int = EMBED_DIM,
                 num_heads: int = NUM_HEADS, mlp_dim: int = MLP_DIM,
                 attn_mode: str = "sdpa"):
        super().__init__()
        self.blocks = nn.ModuleList([
            Encoder1DBlock(dim, num_heads, mlp_dim, attn_mode)
            for _ in range(num_layers)])
        self.encoder_norm = nn.LayerNorm(dim, eps=NORM_EPS)

    def forward(self, x: torch.Tensor, mask_bias: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, mask_bias)
        return self.encoder_norm(x)


# --------------------------------------------------------------------------
# Attention mask — a compile-time constant for the deployment case
# --------------------------------------------------------------------------

def build_attention_mask(n_prefix_per_group: list[int],
                         n_timestep_per_group: list[int],
                         horizon: int,
                         readout_group_index: int) -> np.ndarray:
    """Reproduce `BlockTransformer.generate_attention_mask` in numpy.

    Upstream builds this from group sizes and `AttentionRule`s, then ANDs it
    with the padding mask. With both cameras present, language present and no
    padded timesteps -- the deployment case -- the pad mask is all-ones, so the
    whole thing is a **constant**. It is baked as a buffer here and asserted
    equal to the mask JAX produces (`validate.py`).

    Group order matters and is fixed by the config's dict insertion order:
      prefix   : [task_language]
      timestep : [obs_primary, obs_wrist, obs_task_language, readout_action]

    Rules (from `OctoTransformer.__call__`):
      task_*    attends to task_*                          (CAUSAL)
      obs_*     attends to task_* and obs_*                (CAUSAL)
      readout_* attends to task_*, obs_*, and ITS OWN group (CAUSAL)
    A readout is never attended to by anything else, which is the whole point
    of a readout: it reads without writing.

    `use_correct_attention=True` selects `side="right"` in upstream's
    `np.searchsorted`, which is what decides group membership at a boundary
    index. Getting that wrong shifts every group by one token.
    """
    n_prefix = sum(n_prefix_per_group)
    n_per_step = sum(n_timestep_per_group)
    total = n_prefix + n_per_step * horizon

    prefix_cum = np.cumsum(n_prefix_per_group)
    step_cum = np.cumsum(n_timestep_per_group)

    # group id, and timestep (-1 for prefix), for every token index
    gid = np.empty(total, dtype=np.int64)
    tstep = np.empty(total, dtype=np.int64)
    for i in range(total):
        if i < n_prefix:
            gid[i] = int(np.searchsorted(prefix_cum, i, side="right"))
            tstep[i] = -1
        else:
            j = i - n_prefix
            t, r = divmod(j, n_per_step)
            gid[i] = len(n_prefix_per_group) + int(
                np.searchsorted(step_cum, r, side="right"))
            tstep[i] = t

    n_pg = len(n_prefix_per_group)
    is_prefix = gid < n_pg                       # task_language
    is_readout = gid == (n_pg + readout_group_index)

    # rows: what each token is allowed to look at
    #   prefix rows -> only prefix (task_*) columns
    #   obs rows    -> prefix + non-readout timestep columns
    #   readout row -> prefix + non-readout timestep columns + own readout group
    causal = tstep[None, :] <= tstep[:, None]

    allow = np.zeros((total, total), dtype=bool)
    # every row may attend to the task prefix
    allow |= is_prefix[None, :]
    # non-prefix rows may additionally attend to non-readout timestep tokens
    allow |= (~is_prefix[:, None]) & (~is_prefix[None, :]) & (~is_readout[None, :])
    # readout rows may attend to their own readout group
    allow |= is_readout[:, None] & is_readout[None, :] & (gid[:, None] == gid[None, :])
    # prefix rows must NOT see timestep tokens
    allow &= ~(is_prefix[:, None] & (~is_prefix[None, :]))

    return allow & causal


# --------------------------------------------------------------------------
# Backbone
# --------------------------------------------------------------------------

class OctoBackbone(nn.Module):
    """Observations + language embedding -> the action readout embedding.

    Runs ONCE per control step. This is ~99.8% of the policy's FLOPs
    (~17.5 GMAC at window=2 with both cameras, vs ~34 MMAC for the entire
    20-step denoising loop) -- see NOTES.md §6.

    Output is `(B, window, 384)`: upstream mean-pools the readout tokens, and
    since `readouts={"action": 1}` there is exactly one readout token per
    timestep, so that "mean" is a squeeze over a length-1 axis rather than a
    reduction. No `mean` op is emitted.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        w = cfg["window"]
        self.window = w
        self.use_wrist = cfg["wrist"] > 0

        self.tok_primary = ImageTokenizer(cfg["primary"], w, cfg["gn"],
                                          cfg["goal"], cfg["norm"])
        self.proj_primary = nn.Linear(STEM_NUM_FEATURES, EMBED_DIM)
        self.pos_primary = nn.Parameter(
            torch.zeros(1, MAX_HORIZON, self.tok_primary.num_tokens, EMBED_DIM))

        if self.use_wrist:
            self.tok_wrist = ImageTokenizer(cfg["wrist"], w, cfg["gn"],
                                            cfg["goal"], cfg["norm"])
            self.proj_wrist = nn.Linear(STEM_NUM_FEATURES, EMBED_DIM)
            self.pos_wrist = nn.Parameter(
                torch.zeros(1, MAX_HORIZON, self.tok_wrist.num_tokens, EMBED_DIM))

        self.proj_language = nn.Linear(T5_HIDDEN, EMBED_DIM)
        self.pos_language = nn.Parameter(torch.zeros(1, LANG_TOKENS, EMBED_DIM))
        self.pos_readout = nn.Parameter(torch.zeros(1, MAX_HORIZON, 1, EMBED_DIM))

        self.transformer = Transformer(cfg["layers"], EMBED_DIM, NUM_HEADS,
                                       MLP_DIM, cfg["attn"])

        n_ts = [self.tok_primary.num_tokens]
        if self.use_wrist:
            n_ts.append(self.tok_wrist.num_tokens)
        n_ts += [LANG_TOKENS, 1]                       # tiled language, readout
        readout_idx = len(n_ts) - 1
        mask = build_attention_mask([LANG_TOKENS], n_ts, w, readout_idx)
        # Disallowed slots get a FINITE negative bias, not finfo.min.
        # finfo.min is what flax uses and is exact in float, but it is
        # unrepresentable in int8: per-tensor calibration of this buffer
        # picks scale = 3.4e38/127, the add's output scale follows it, and
        # every real logit then quantizes to 0 -- softmax degenerates to
        # uniform-over-unmasked. -mask_neg keeps the same "zero weight
        # after softmax" semantics (exp(-32) = 1.3e-14 relative) while
        # leaving the add's output scale at mask_neg/127, which is what
        # bounds the surviving logits' resolution.
        bias = np.where(mask, 0.0, -cfg["mask_neg"]).astype(np.float32)
        self.register_buffer("mask_bias",
                             torch.from_numpy(bias)[None, None], persistent=False)
        self.n_prefix = LANG_TOKENS
        self.n_per_step = sum(n_ts)
        self.total_tokens = self.n_prefix + self.n_per_step * w

    def forward(self, *args, **kwargs):
        """Eager-mode entry point.

        Delegates to `_run`. NOTE: do not trace THIS -- `*args` collapses to a
        single fx placeholder holding an immutable_list. `get_model()` replaces
        `__class__` with a `_traceable(...)` subclass whose `forward` has one
        explicit parameter per real input; that is what the extractors see.
        """
        return self._run(*args, **kwargs)

    def _run(self, img_primary: torch.Tensor,
             img_wrist: Optional[torch.Tensor] = None,
             lang: Optional[torch.Tensor] = None,
             goal_primary: Optional[torch.Tensor] = None,
             goal_wrist: Optional[torch.Tensor] = None) -> torch.Tensor:
        w = self.window

        # --- task prefix: language ------------------------------------------
        task = self.proj_language(lang) + self.pos_language        # (B, 16, 384)

        # --- observation groups ---------------------------------------------
        groups = []
        p = self.tok_primary(img_primary, goal_primary)
        groups.append(self.proj_primary(p) + self.pos_primary[:, :w])
        if self.use_wrist:
            q = self.tok_wrist(img_wrist, goal_wrist)
            groups.append(self.proj_wrist(q) + self.pos_wrist[:, :w])

        # repeat_task_tokens=True: the ALREADY projected + position-embedded
        # prefix tokens are tiled across timesteps as an extra group. No second
        # projection and no extra positional embedding -- there is no
        # obs_task_language_pos_embedding in the checkpoint.
        # `cat` of a static number of copies, not `expand`/`repeat`: only
        # `expand_as` is in the export extractor's alias set, and neither
        # `expand` nor `repeat` is.
        groups.append(torch.cat([task.unsqueeze(1)] * w, dim=1))

        # readout tokens are content-free: zeros + their positional embedding.
        # expand_as off an existing (B, w, n, 384) group gets the batch size
        # without reading .shape (which would defeat fx tracing).
        groups.append(self.pos_readout[:, :w].expand_as(groups[0][:, :, :1, :]))

        # --- assemble: prefix, then (horizon x tokens) flattened ------------
        ts = torch.cat(groups, dim=2).reshape(-1, w * self.n_per_step, EMBED_DIM)
        x = torch.cat([task, ts], dim=1)

        x = self.transformer(x, self.mask_bias)

        # --- split out the readout group ------------------------------------
        x = x[:, self.n_prefix:].reshape(-1, w, self.n_per_step, EMBED_DIM)
        # readout is the LAST timestep group, 1 token wide -> index -1, then
        # the length-1 token axis is squeezed (== upstream's mean over it).
        return x[:, :, -1, :]


# --------------------------------------------------------------------------
# Diffusion score network  (octo/model/components/diffusion.py)
# --------------------------------------------------------------------------

class FourierFeatures(nn.Module):
    """Learnable Fourier time embedding: `concat(cos(2*pi*t@w.T), sin(...))`.

    One parameter, `kernel [TIME_DIM//2, 1]`. Runs on a `(B, W, 1)` scalar
    time, so it is ~600 flops -- irrelevant to cost, but `cos`/`sin` are not in
    ModelBlaster's op vocabulary, so it is called out in the coverage report.
    """

    def __init__(self, output_size: int = TIME_DIM):
        super().__init__()
        self.linear = nn.Linear(1, output_size // 2, bias=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        f = 2.0 * math.pi * self.linear(t)
        return torch.cat([torch.cos(f), torch.sin(f)], dim=-1)


def _swish(x: torch.Tensor) -> torch.Tensor:
    """flax `nn.swish` == `x * sigmoid(x)`.

    Written out rather than `F.silu` so the graph lands on `sigmoid` + `mul`,
    both of which ModelBlaster already has (`sigmoid_s8`, `mul_s8`); a bare
    `aten.silu` is not in the export extractor's vocabulary.
    """
    return x * torch.sigmoid(x)


class MLPResNetBlock(nn.Module):
    """LayerNorm -> Dense(4f) -> swish -> Dense(f) -> + residual.

    `dropout_rate=0.0` upstream, so the dropout branch never fires and is
    omitted. `use_layer_norm=True` for this checkpoint.
    """

    def __init__(self, features: int = SCORE_HIDDEN):
        super().__init__()
        self.norm = nn.LayerNorm(features, eps=NORM_EPS)
        self.fc1 = nn.Linear(features, features * 4)
        self.fc2 = nn.Linear(features * 4, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.fc2(_swish(self.fc1(y)))
        return x + y


class ScoreNet(nn.Module):
    """The DDPM score network: (embedding, noisy_action, t) -> predicted noise.

    Runs 20 times per control step (`diffusion_steps=20`) but is only
    1,703,564 params / ~1.7 MMAC per call, so the whole loop is ~0.2% of the
    policy's arithmetic. Input to `reverse_network` is
    `concat([cond(32), obs(384), action(28)]) = 444`, which is exactly the
    checkpoint's `reverse_network/Dense_0/kernel [444, 256]`.
    """

    def __init__(self, obs_dim: int = EMBED_DIM, time_mode: str = "fourier"):
        super().__init__()
        self.time_mode = time_mode
        self.time_preprocess = FourierFeatures(TIME_DIM)
        # cond_encoder = MLP((2*time_dim, time_dim)); activate_final=False and
        # use_layer_norm defaults False, so swish goes between the two Denses
        # only.
        self.cond1 = nn.Linear(TIME_DIM, 2 * TIME_DIM)
        self.cond2 = nn.Linear(2 * TIME_DIM, TIME_DIM)

        self.action_flat = ACTION_DIM * ACTION_HORIZON     # 28
        in_dim = TIME_DIM + obs_dim + self.action_flat     # 444
        self.fc_in = nn.Linear(in_dim, SCORE_HIDDEN)
        self.blocks = nn.ModuleList(
            [MLPResNetBlock(SCORE_HIDDEN) for _ in range(SCORE_BLOCKS)])
        self.fc_out = nn.Linear(SCORE_HIDDEN, self.action_flat)

    def forward(self, *args, **kwargs):
        """Eager-mode entry point; see OctoBackbone.forward."""
        return self._run(*args, **kwargs)

    def _run(self, obs_enc: torch.Tensor, noisy_actions: torch.Tensor,
             t: torch.Tensor) -> torch.Tensor:
        """`t` is the raw timestep (B, W, 1) in "fourier" mode, or the
        precomputed 32-d conditioning vector (B, W, 32) in "lut" mode."""
        if self.time_mode == "lut":
            c = t
        else:
            c = self.cond_embed(t)
        x = torch.cat([c, obs_enc, noisy_actions], dim=-1)
        x = self.fc_in(x)
        for blk in self.blocks:
            x = blk(x)
        return self.fc_out(_swish(x))

    def cond_embed(self, t: torch.Tensor) -> torch.Tensor:
        """FourierFeatures + cond_encoder: raw timestep -> 32-d conditioning."""
        return self.cond2(_swish(self.cond1(self.time_preprocess(t))))

    @torch.no_grad()
    def cond_table(self, steps: int = DIFFUSION_STEPS) -> torch.Tensor:
        """The whole time-conditioning branch, precomputed: (steps, 32).

        `cond_embed` is a frozen function of the timestep alone, and DDPM
        sampling visits exactly `diffusion_steps` = 20 integer timesteps. So
        the entire branch -- FourierFeatures (cos/sin) plus a two-layer MLP
        with a swish -- collapses to a 20x32 lookup the host computes once at
        init. Exact for integer t in [0, steps).

        Same argument as not porting T5 (NOTES.md 7): a frozen function of a
        fixed, small input set is a constant, not compute. It also removes the
        only two ops in the whole port that neither extractor knows (`cos`,
        `sin`), which is why `MODELBLASTER_OCTO_TIME=lut` exists.
        """
        t = torch.arange(steps, dtype=torch.float32).reshape(steps, 1, 1)
        return self.cond_embed(t).reshape(steps, TIME_DIM)


def cosine_beta_schedule(timesteps: int = DIFFUSION_STEPS,
                         s: float = 0.008) -> np.ndarray:
    """Upstream's `cosine_beta_schedule`. A constant -- folded in as a buffer."""
    t = np.linspace(0, timesteps, timesteps + 1) / timesteps
    ac = np.cos((t + s) / (1 + s) * np.pi * 0.5) ** 2
    ac = ac / ac[0]
    return np.clip(1 - (ac[1:] / ac[:-1]), 0.0, 0.999)


# --------------------------------------------------------------------------
# Whole policy
# --------------------------------------------------------------------------

class OctoSmall(nn.Module):
    """backbone + one score-net evaluation == one denoising step.

    `forward` is a single self-contained graph over every op in the port, which
    is what gets extracted and drift-checked. The *deployment* decomposition is
    different and better: run `backbone` once, cache the embedding, then run
    `score` 20 times (`MODELBLASTER_OCTO_PART`). `sample_actions` does that and
    is deliberately NOT part of the traced graph.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.backbone = OctoBackbone(cfg)
        self.score = ScoreNet(EMBED_DIM, cfg["time_mode"])
        betas = cosine_beta_schedule(DIFFUSION_STEPS)
        alphas = 1.0 - betas
        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("alphas", torch.tensor(alphas, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("alpha_hats",
                             torch.tensor(np.cumprod(alphas), dtype=torch.float32),
                             persistent=False)

    def forward(self, *args, **kwargs):
        """Eager-mode entry point.

        Delegates to `_run`. NOTE: do not trace THIS -- `*args` collapses to a
        single fx placeholder holding an immutable_list. `get_model()` replaces
        `__class__` with a `_traceable(...)` subclass whose `forward` has one
        explicit parameter per real input; that is what the extractors see.
        """
        return self._run(*args, **kwargs)

    def _run(self, img_primary: torch.Tensor,
             img_wrist: Optional[torch.Tensor] = None,
             lang: Optional[torch.Tensor] = None,
             noisy_actions: Optional[torch.Tensor] = None,
             t: Optional[torch.Tensor] = None,
             goal_primary: Optional[torch.Tensor] = None,
             goal_wrist: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.backbone._run(img_primary, img_wrist, lang,
                                 goal_primary, goal_wrist)
        return self.score(emb, noisy_actions, t)

    # -- not traced -------------------------------------------------------
    @torch.no_grad()
    def sample_actions(self, embeddings: torch.Tensor,
                       noise: torch.Tensor,
                       step_noise: torch.Tensor) -> torch.Tensor:
        """DDPM ancestral sampling, upstream `DiffusionActionHead.predict_action`.

        Noise is passed in rather than drawn so this is bit-comparable against
        the JAX reference: JAX's PRNG and torch's are different generators, so
        the only honest way to compare the loop is to drive both from the same
        arrays.

        `noise` is `(B, W, 28)` (x_T); `step_noise` is `(20, B, W, 28)`, indexed
        by loop iteration (iteration i handles time = 19 - i).

        The `where(action_mask, ...)` upstream applies is omitted: it is
        identity whenever `embodiment_action_dim == action_dim == 7`, which is
        this checkpoint's case. It matters only for embodiments with fewer than
        7 action dims.
        """
        x = noise
        lut = (self.score.cond_table() if self.score.time_mode == "lut"
               else None)
        for i, time in enumerate(range(DIFFUSION_STEPS - 1, -1, -1)):
            if lut is not None:
                t = lut[time].expand(x.shape[:-1] + (TIME_DIM,))
            else:
                t = torch.full(x.shape[:-1] + (1,), float(time), dtype=x.dtype)
            eps = self.score(embeddings, x, t)
            a1 = 1.0 / torch.sqrt(self.alphas[time])
            a2 = (1.0 - self.alphas[time]) / torch.sqrt(1.0 - self.alpha_hats[time])
            x = a1 * (x - a2 * eps)
            if time > 0:
                x = x + torch.sqrt(self.betas[time]) * step_noise[i]
            x = torch.clamp(x, -MAX_ACTION, MAX_ACTION)
        x = x.reshape(x.shape[0], x.shape[1], ACTION_HORIZON, ACTION_DIM)
        return x[:, -1]          # only the last timestep in the window


# --------------------------------------------------------------------------
# Traceable signatures
# --------------------------------------------------------------------------

def forward_arg_names(cfg: dict) -> list[str]:
    """Positional forward() parameter names for the active knob combination.

    `get_sample_input` returns exactly these, in this order.
    """
    if cfg["part"] == "score":
        return ["obs_enc", "noisy_actions", "t"]
    names = ["img_primary"]
    if cfg["wrist"]:
        names.append("img_wrist")
    names.append("lang")
    if cfg["goal"]:
        names.append("goal_primary")
        if cfg["wrist"]:
            names.append("goal_wrist")
    if cfg["part"] == "full":
        names += ["noisy_actions", "t"]
    return names


def _traceable(base: type, arg_names: list[str]) -> type:
    """Subclass `base` with a `forward` whose signature is EXACTLY arg_names.

    Why this exists: `torch.fx.symbolic_trace` turns a `forward(self, *args)`
    into a SINGLE placeholder holding an `immutable_list` of proxies, and the
    extractor's `_tensor_meta` then fails with
    `'immutable_list' object has no attribute 'shape'`. Parameters that merely
    have `None` defaults are no better -- they become dangling placeholders
    with no tensor meta.

    The knobs change the arity (wrist on/off x goal on/off x part), so the
    signature cannot be written out statically without four near-duplicate
    classes per part. Generating it once at build time keeps a single
    implementation (`base._run`, keyword-dispatched) and gives fx and
    torch.export a clean, explicit placeholder per real input.

    Subclassing rather than wrapping keeps the state_dict keys unchanged, so
    the converter's `backbone.*` / `score.*` names still apply.
    """
    src = (f"def forward(self, {', '.join(arg_names)}):\n"
           f"    return self._run({', '.join(f'{n}={n}' for n in arg_names)})\n")
    ns: dict = {}
    exec(src, ns)                       # noqa: S102 - generated from a fixed list
    fwd = ns["forward"]
    fwd.__doc__ = f"Generated signature: forward({', '.join(arg_names)})"
    return type(f"{base.__name__}Traceable", (base,), {"forward": fwd})


# --------------------------------------------------------------------------
# modelblaster entry points
# --------------------------------------------------------------------------

def get_model(seed: int = 0) -> nn.Module:
    """Build the policy and load converted weights if available.

    WITHOUT `MODELBLASTER_OCTO_CKPT` the weights are random and every output is
    meaningless. That is fine for op-histogram / latency work and useless for
    anything else, so it warns loudly rather than failing quietly -- the same
    contract models/vint.py uses.
    """
    torch.manual_seed(seed)
    cfg = _cfg()
    names = forward_arg_names(cfg)
    if cfg["part"] == "backbone":
        cls = _traceable(OctoBackbone, names)
    elif cfg["part"] == "score":
        cls = _traceable(ScoreNet, names)
    else:
        cls = _traceable(OctoSmall, names)

    # Always build the FULL policy so the converted state_dict loads under its
    # own key names, then hand back the requested part.
    full = OctoSmall(cfg).eval()

    ckpt = os.environ.get("MODELBLASTER_OCTO_CKPT", "")
    if not ckpt:
        default = (os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                   + "/experiments/octo_port/octo_small_torch.pt")
        ckpt = default if os.path.isfile(default) else ""
    if not ckpt:
        print("[octo_small.get_model] WARN: no converted checkpoint found; using "
              "RANDOM-INIT weights (outputs are meaningless). Build one with "
              "experiments/octo_port/convert_weights.py, or set "
              "MODELBLASTER_OCTO_CKPT.", flush=True)
    else:
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        missing, unexpected = full.load_state_dict(sd, strict=False)
        real_missing = [k for k in missing if not k.endswith("mask_bias")]
        if real_missing:
            print(f"[octo_small.get_model] {len(real_missing)} missing keys "
                  f"(e.g. {real_missing[:3]})", flush=True)
        if unexpected:
            print(f"[octo_small.get_model] {len(unexpected)} unexpected keys "
                  f"(e.g. {unexpected[:3]})", flush=True)

    part = {"backbone": full.backbone, "score": full.score}.get(cfg["part"], full)
    part.__class__ = cls              # swap in the generated explicit signature
    return part.eval()


def get_sample_input(seed: int = 1):
    """One positional tuple matching `OctoSmall.forward` for the active PART.

    Images are float tensors valued in [0, 255] -- the /127.5 - 1 normalisation
    lives inside the model, as it does upstream, so the harness feeds raw
    camera values.

    Random gaussian language embeddings and random images are fine for tracing
    and for shape/latency work. They are NOT a calibration set: see
    get_calibration_spec.
    """
    cfg = _cfg()
    g = torch.Generator().manual_seed(seed)
    w = cfg["window"]

    def _img(n: int, batch_dims: tuple) -> torch.Tensor:
        return torch.randint(0, 256, batch_dims + (3, n, n), generator=g).float()

    # Built by name so this cannot drift out of step with forward_arg_names().
    made = {
        "img_primary": lambda: _img(cfg["primary"], (1, w)),
        "img_wrist": lambda: _img(cfg["wrist"], (1, w)),
        "lang": lambda: torch.randn(1, LANG_TOKENS, T5_HIDDEN, generator=g) * 0.5,
        "goal_primary": lambda: _img(cfg["primary"], (1,)),
        "goal_wrist": lambda: _img(cfg["wrist"], (1,)),
        "noisy_actions": lambda: torch.randn(
            1, w, ACTION_DIM * ACTION_HORIZON, generator=g) * 0.3,
        # NB the parens: `lambda: A if c else lambda: B` parses as
        # `lambda: (A if c else <lambda>)` and hands torch.export a function.
        "t": ((lambda: torch.randn(1, w, TIME_DIM, generator=g) * 0.5)
              if cfg["time_mode"] == "lut"
              else (lambda: torch.full((1, w, 1), 7.0))),
        "obs_enc": lambda: torch.randn(1, w, EMBED_DIM, generator=g),
    }
    return tuple(made[n]() for n in forward_arg_names(cfg))


def get_calibration_spec(num_samples: int = 8) -> "dict | None":
    """Declarative calibration spec, or None if no real data source is set.

    int8 PTQ on this model needs real robot frames: the stems' activation
    ranges after weight standardisation + GroupNorm have nothing to do with
    what `torch.randint(0,256)` produces, and per-tensor scales fitted to noise
    would still verify bit-exact against their own golden while being the wrong
    scales. `MODELBLASTER_OCTO_BRIDGE` should point at the BridgeData episodes
    (`/scratch2/dima/misc_sw/octo_work/bridge_episodes.pkl` is the pickle the
    upstream benchmarks used); returning None is the honest answer when it is
    unset, rather than silently calibrating on noise.
    """
    src = os.environ.get("MODELBLASTER_OCTO_BRIDGE", "")
    if not src:
        return None
    cfg = _cfg()
    return {
        "num_samples": num_samples,
        "inputs": {
            "img_primary": {"loader": "bridge_episodes", "path": src,
                            "key": "image_primary",
                            "image_size": [cfg["primary"], cfg["primary"]],
                            "compose": {"kind": "rolling_window",
                                        "frames_per_sample": cfg["window"]}},
            "img_wrist": {"loader": "bridge_episodes", "path": src,
                          "key": "image_wrist",
                          "image_size": [cfg["wrist"], cfg["wrist"]],
                          "compose": {"kind": "rolling_window",
                                      "frames_per_sample": cfg["window"]}},
        },
    }
