"""A broader int8 calibration set (observations without reference actions) for SmolVLA: the 8 original
calibration observations' frames at other brightnesses, plus HM01B0 frames from the PULP-DroNet v3 set
that are NOT in the held-out set (make_heldout.py samples with seed 11; this samples others).
   python make_calib.py OUT_DIR HELDOUT_DIR [--n-himax 16]"""
import argparse, glob, os, random, torch, numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance
torch.set_num_threads(8)
from modelblaster.models import smolvla_split as S
from lerobot.policies.factory import make_pre_post_processors
ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("heldout")
ap.add_argument("--frames", default=os.environ.get("MB_SMOLVLA_FRAMES"), required="MB_SMOLVLA_FRAMES" not in os.environ,
                help="calibration frames directory (or MB_SMOLVLA_FRAMES)")
ap.add_argument("--himax", default=os.environ.get("MB_HIMAX_DATASET"), required="MB_HIMAX_DATASET" not in os.environ,
                help="PULP-DroNet v3 HM01B0 dataset root (or MB_HIMAX_DATASET)")
ap.add_argument("--n-himax", type=int, default=16)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
policy = S.load_policy(256)
pre, post = make_pre_post_processors(policy.config, pretrained_path="lerobot/smolvla_base",
                                     preprocessor_overrides={"device_processor": {"device": "cpu"}})
key = list(policy.config.image_features)[0]
t_of = lambda pil: torch.from_numpy(np.asarray(pil.convert("RGB"), np.float32) / 255.0).permute(2, 0, 1)
CAL = [("clipf0-colour.png", "fly forward and land on the table"), ("clipf3-colour.png", "turn left and hover"),
       ("wf1-colour.png", "go up slowly and stop"), ("wf5-colour.png", "pick up the cup"),
       ("wf3-colour.png", "fly through the doorway"), ("clipf5-colour.png", "follow the cable to the wall")]
items = []
for f, task in CAL:
    im = Image.open(Path(a.frames) / f).convert("RGB")
    for b in (0.5, 1.5):
        items.append((t_of(ImageEnhance.Brightness(im).enhance(b)), task))
used = {str(torch.load(f)["image"]).split(":", 1)[1] for f in glob.glob(f"{a.heldout}/*.pt")
        if str(torch.load(f)["image"]).startswith("himax:")}
hm = sorted(glob.glob(f"{a.himax}/*/*/*/images/*.jpeg"))
rng = random.Random(99)
tasks = [t for _, t in CAL] + ["move to the red box", "fly backward and land"]
picked = 0
for f in rng.sample(hm, 400):
    rel = str(Path(f).relative_to(a.himax))
    if rel in used:
        continue
    items.append((t_of(Image.open(f)), tasks[picked % len(tasks)]))
    picked += 1
    if picked >= a.n_himax:
        break
for i, (im, task) in enumerate(items):
    g = torch.Generator().manual_seed(500 + i)
    st = torch.randn(policy.config.robot_state_feature.shape[0], generator=g) * 0.3
    batch = policy._prepare_batch(pre({"observation.state": st, "task": task, key: im}))
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    tokens, lmask = batch["observation.language.tokens"], batch["observation.language.attention_mask"]
    noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, generator=g)
    valid = tokens[lmask.bool()].reshape(1, -1)
    parts = S.build(policy, S.cfg_from_policy(policy, 256, valid.shape[1]))
    torch.save({"pixels": images[0], "lang": parts.lang_embedding(valid), "state": state, "noise": noise,
                "tokens": valid, "task": task}, out / f"cal{100 + i}.pt")
    print(i, task, flush=True)
print("DONE")
