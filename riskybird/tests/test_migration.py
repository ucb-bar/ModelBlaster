"""Host checks for recovered model sources, explicit inputs and failure propagation.

Run with a Python providing torch/numpy/Pillow/torchvision:
  python -m unittest discover -s riskybird/tests -v
"""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "riskybird/dronet_temporal"))
import _paths


class MigrationTests(unittest.TestCase):
    def test_artifact_assembly_uses_movable_basenames(self):
        for p in (ROOT / "riskybird/artifacts").rglob("test_io.S"):
            for line in p.read_text().splitlines():
                if '.incbin "' in line:
                    self.assertNotIn('/', line.split('"')[1], str(p))

    def test_dataset_required_and_stack_order(self):
        import numpy as np
        from PIL import Image
        from modelblaster.models import dronet_stack3 as model
        old_root = model._DATA_ROOT
        try:
            model._STACKS_CACHE.clear()
            model._DATA_ROOT = ""
            with unittest.mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, 'MODELBLASTER_DRONET_STACK3_DATA'):
                    model.get_sample_input()
            with tempfile.TemporaryDirectory() as folder:
                d = Path(folder)
                (d / "images").mkdir()
                (d / "labels_partitioned.csv").write_text(
                    'filename,partition,label_yaw_rate\n10.jpeg,train,0\n20.jpeg,train,0\n30.jpeg,train,45\n40.jpeg,valid,0\n')
                for n, val in [(10, 20), (20, 60), (30, 100), (40, 140)]:
                    Image.fromarray(np.full((112, 112), val, dtype=np.uint8)).save(d / "images" / f"{n}.jpeg")
                model._DATA_ROOT = folder
                model._STACKS_CACHE.clear()
                stacks = model._build("train")
                self.assertEqual(len(stacks), 1)
                self.assertEqual(stacks[0][1], 0.5)
                self.assertEqual(model._build("valid"), [])
                x = model.get_sample_input().numpy()
                self.assertEqual(x.shape, (1, 3, 112, 112))
                self.assertTrue(np.all(np.diff(x.mean(axis=(0, 2, 3))) > 0))
        finally:
            model._DATA_ROOT = old_root
            model._STACKS_CACHE.clear()

    def test_recovered_simulators_agree_without_fp16_promotions(self):
        import numpy as np
        from int8_sim import simulate_int8
        from hybrid_sim import simulate_hybrid
        x = np.array([-10, 0, 37], dtype=np.int8)
        ops = [{"name": "relu", "op": "relu_s8", "inputs": ["x"], "outputs": ["y"]}]
        a = simulate_int8(ops, {}, [x], ["x"])["y"]
        b = simulate_hybrid(ops, {}, [x], ["x"], {"x": 1., "y": 1.}, set(), {})["y"]
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, [0, 0, 37])

    def test_shell_stops_on_failed_generation(self):
        scripts = [ROOT / "riskybird/dronet_temporal/stack3_build.sh",
                   ROOT / "examples/dronet/int8/build_ku040_dual.sh",
                   ROOT / "examples/dronet/int8/build_ku040_dual_nhwc.sh"]
        with tempfile.TemporaryDirectory() as folder:
            d = Path(folder)
            for name in ['python', 'west']:
                p = d / name
                p.write_text('#!/bin/sh\nexit 37\n')
                p.chmod(0o755)
            fragment = d / 'console.conf'
            fragment.touch()
            env = dict(os.environ, PATH=f'{d}:/usr/bin:/bin', MB=str(ROOT),
                       ZCS=folder, ZEPHYR_BASE=folder, ZEPHYR_SDK_INSTALL_DIR=folder,
                       CONSOLE_CONF=str(fragment), CONSOLE_OVERLAY=str(fragment))
            for script in scripts:
                result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 37, (str(script), result.stdout, result.stderr))
                self.assertNotIn('STACK3_BUILD_DONE', result.stdout)


# unittest.mock is not loaded by importing unittest alone on every interpreter.
import unittest.mock
