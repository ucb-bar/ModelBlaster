"""A large held-out SmolVLA observation set (pixels / lang / state / noise + lerobot reference actions)
for the int8 numerics validation. Prompts are kept only if they tokenize to exactly L tokens (the IRs
are extracted for one prompt length). Images: the riskybird frames NOT used for calibration, light
augmentations of them, real HM01B0 frames from the PULP-DroNet v3 set (grayscale, other drones),
and synthetic scenes.
   python make_heldout.py OUT_DIR [--frames DIR] [--himax DIR] [--n-himax 120] [--L 14]"""
import argparse, glob, os, random, sys, torch, numpy as np
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance
torch.set_num_threads(8)
from modelblaster.models import smolvla_split as S
from lerobot.policies.factory import make_pre_post_processors

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--frames", default=os.environ.get("MB_SMOLVLA_FRAMES"), required="MB_SMOLVLA_FRAMES" not in os.environ,
                help="calibration frames directory (or MB_SMOLVLA_FRAMES)")
ap.add_argument("--himax", default=os.environ.get("MB_HIMAX_DATASET"), required="MB_HIMAX_DATASET" not in os.environ,
                help="PULP-DroNet v3 HM01B0 dataset root (or MB_HIMAX_DATASET)")
ap.add_argument("--n-himax", type=int, default=120)
ap.add_argument("--L", type=int, default=14)
ap.add_argument("--seed0", type=int, default=1000)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

policy = S.load_policy(256)
pre, post = make_pre_post_processors(policy.config, pretrained_path="lerobot/smolvla_base",
                                     preprocessor_overrides={"device_processor": {"device": "cpu"}})
policy.config.num_steps = 5
key = list(policy.config.image_features)[0]
CAL_FRAMES = {"clipf0-colour.png", "clipf3-colour.png", "wf1-colour.png", "wf5-colour.png", "wf3-colour.png",
              "clipf5-colour.png"}

def ntok(t):
    return int(pre({"observation.state": torch.zeros(policy.config.robot_state_feature.shape[0]), "task": t,
                    key: torch.zeros(3, 64, 64)})["observation.language.attention_mask"].sum())

verbs = ["fly forward", "fly backward", "turn left", "turn right", "go up", "go down", "move left", "move right",
         "climb slowly", "descend slowly", "rotate in place", "slide sideways"]
objs = ["the open door", "the red box", "the white cable", "the ceiling light", "the far wall", "the green chair",
        "the small table", "the bright window", "the black backpack", "the tall plant", "the person waving",
        "the landing pad"]
ends = ["and hover there", "and land gently", "and stop right there", "then hold position", "and wait a moment",
        "then come back home"]
fill = ["", " please", " slowly", " now", " carefully", " right away", " very slowly", " for me please"]
rng = random.Random(7)
good, seen, tries = [], set(), 0
while len(good) < 120 and tries < 3000:
    tries += 1
    t = rng.choice([f"{rng.choice(verbs)} toward {rng.choice(objs)} {rng.choice(ends)}",
                    f"please {rng.choice(verbs)} until you reach {rng.choice(objs)} {rng.choice(ends)}",
                    f"{rng.choice(verbs)} past {rng.choice(objs)} {rng.choice(ends)}"]) + rng.choice(fill)
    if t in seen:
        continue
    seen.add(t)
    n = ntok(t)
    if n == a.L:
        good.append(t)
    elif n < a.L - 4 or n > a.L + 4:
        continue
print("prompt candidates tried", tries, flush=True)
rng.shuffle(good)
print("prompts with L", a.L, ":", len(good), flush=True)

def synth(seed):
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 256), torch.linspace(0, 1, 256), indexing="ij")
    im = torch.stack([xx, yy, 0.5 * (xx + yy)]).roll(seed, 0) * 0.8 + 0.1 * torch.rand(3, 256, 256, generator=g)
    p, q = (torch.randint(20, 180, (2,), generator=g)).tolist()
    im[:, p:p + 60, q:q + 50] = torch.rand(3, generator=g)[:, None, None]
    return im.clamp(0, 1)

def t_of(pil):
    return torch.from_numpy(np.asarray(pil.convert("RGB"), np.float32) / 255.0).permute(2, 0, 1)

items = []                                          # (tag, image tensor)
real = [f for f in sorted(Path(a.frames).glob("*.png")) if f.name not in CAL_FRAMES]
for f in real:
    im = Image.open(f)
    items.append((f"real:{f.name}", t_of(im)))
    items.append((f"real:{f.name}#flip", t_of(ImageOps.mirror(im))))
    items.append((f"real:{f.name}#dark", t_of(ImageEnhance.Brightness(im.convert('RGB')).enhance(0.55))))
    items.append((f"real:{f.name}#bright", t_of(ImageEnhance.Brightness(im.convert('RGB')).enhance(1.4))))
hm = sorted(glob.glob(f"{a.himax}/*/*/*/images/*.jpeg"))
hrng = random.Random(11)
for f in hrng.sample(hm, min(a.n_himax, len(hm))):
    items.append((f"himax:{Path(f).relative_to(a.himax)}", t_of(Image.open(f))))
for s in range(30, 46):
    items.append((f"synth:{s}", synth(s)))
print("observations:", len(items), flush=True)

for i, (tag, im) in enumerate(items):
    task = good[i % len(good)]
    g = torch.Generator().manual_seed(a.seed0 + i)
    st = torch.randn(policy.config.robot_state_feature.shape[0], generator=g) * 0.3
    batch = policy._prepare_batch(pre({"observation.state": st, "task": task, key: im}))
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    tokens, lmask = batch["observation.language.tokens"], batch["observation.language.attention_mask"]
    noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, generator=g)
    valid = tokens[lmask.bool()].reshape(1, -1)
    assert valid.shape[1] == a.L, (task, valid.shape)
    parts = S.build(policy, S.cfg_from_policy(policy, 256, valid.shape[1]))
    with torch.no_grad():
        ref = policy.model.sample_actions(images, img_masks, tokens, lmask, state, noise=noise.clone())
    d = {"pixels": images[0], "lang": parts.lang_embedding(valid), "state": state, "noise": noise,
         "tokens": valid, "task": task, "image": tag, "ref": ref}
    torch.save(d, out / f"h{i:03d}.pt")
    print(i, tag, "|", task, flush=True)
print("DONE", flush=True)
