"""DroNet checkpoint selection for scaled / grayscale geometries.

A non-default geometry (MODELBLASTER_DRONET_CHANNELS != 3 or
MODELBLASTER_DRONET_INPUT != 112) cannot use the default 3x112x112
checkpoint. It must:

- stay at its seeded random init when MODELBLASTER_DRONET_CKPT is unset, so
  images built without a checkpoint keep their exact weights;
- load a checkpoint named by MODELBLASTER_DRONET_CKPT strictly when every key
  and shape matches;
- refuse a checkpoint for another geometry with an error that names the
  mismatch, never fall back to random weights.

models/dronet.py reads the environment at import, so each case runs in a
fresh interpreter.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    import torch  # noqa: F401
    _HAVE_TORCH = True
except Exception:                            # torch absent in some envs
    _HAVE_TORCH = False

_PRELUDE = (
    "import sys; sys.path[:0] = [{src!r}];"
    "import torch;"
    "from modelblaster.models import dronet, dronet_arch;"
).format(src=str(ROOT / "src"))


def _run(code: str, **env) -> subprocess.CompletedProcess:
    e = {k: v for k, v in os.environ.items() if not k.startswith("MODELBLASTER_DRONET_")}
    e.update(env)
    return subprocess.run([sys.executable, "-c", _PRELUDE + code], env=e,
                          capture_output=True, text=True, timeout=300)


@unittest.skipUnless(_HAVE_TORCH, "torch not installed")
class DronetCheckpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        cls.gray = d / "gray.pt"
        cls.rgb = d / "rgb.pt"
        r = _run(
            "torch.manual_seed(123);"
            "m = dronet_arch.DronetTorch(img_dims=(112, 112), img_channels=1, output_dim=1, small=True);"
            f"torch.save({{'model_state_dict': m.state_dict()}}, {str(cls.gray)!r});"
            "m = dronet_arch.DronetTorch(img_dims=(112, 112), img_channels=3, output_dim=1, small=True);"
            f"torch.save({{'model_state_dict': m.state_dict()}}, {str(cls.rgb)!r})")
        assert r.returncode == 0, r.stderr

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_no_checkpoint_keeps_seeded_init(self):
        r = _run(
            "m = dronet.get_model();"
            "torch.manual_seed(0);"
            "ref = dronet_arch.DronetTorch(img_dims=(112, 112), img_channels=1, output_dim=1, small=True);"
            "sd, rd = m.state_dict(), ref.state_dict();"
            "assert all(torch.equal(sd[k], rd[k]) for k in rd), 'weights differ from seed 0';"
            "print('OK')",
            MODELBLASTER_DRONET_CHANNELS="1")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("OK", r.stdout)

    def test_matched_checkpoint_loads(self):
        r = _run(
            "m = dronet.get_model();"
            f"ck = torch.load({str(self.gray)!r}, weights_only=False)['model_state_dict'];"
            "assert all(torch.equal(m.state_dict()[k], ck[k]) for k in ck);"
            "print('OK')",
            MODELBLASTER_DRONET_CHANNELS="1", MODELBLASTER_DRONET_CKPT=str(self.gray))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("loaded 1x112x112 checkpoint", r.stdout)

    def test_mismatched_checkpoint_is_an_error(self):
        r = _run("dronet.get_model()",
                 MODELBLASTER_DRONET_CHANNELS="1", MODELBLASTER_DRONET_CKPT=str(self.rgb))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("does not match the 1x112x112 geometry", r.stderr)
        self.assertIn("conv_modules.0.weight", r.stderr)

    def test_missing_checkpoint_is_an_error(self):
        r = _run("dronet.get_model()",
                 MODELBLASTER_DRONET_CHANNELS="1",
                 MODELBLASTER_DRONET_CKPT=str(Path(self.tmp.name) / "absent.pt"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("does not exist", r.stderr)


if __name__ == "__main__":
    unittest.main()
