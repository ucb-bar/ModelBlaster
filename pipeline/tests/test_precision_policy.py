"""Precision policy (A2 item 3): fp32 islands for overflow-prone ops in fp16 IRs.

Pure python: the detector, the island rewrite on a hand-made graph (kinds, casts, dtypes, constants, golden), the
mixed-partition helpers of kb_export. With torch: a tiny model whose folded exp() overflows fp16 is extracted with
the policy (island applied, walk finite and within 1e-3 of the fp32 model) and exported (mixed partition: fp16
problem rules, fp32 island slots, casts); a model without overflow is left alone; with GPU + nvcc the mixed problem
builds and passes the driver.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline import precision_policy as pp  # noqa: E402


def _have_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _have_gpu() -> bool:
    try:
        import shutil
        return _have_torch() and shutil.which("nvcc") is not None and subprocess.run(
            ["nvidia-smi"], capture_output=True).returncode == 0
    except Exception:
        return False


class DetectorTest(unittest.TestCase):
    def test_detect(self):
        g16 = np.array([1.0, np.inf, 2.0], dtype=np.float16)
        self.assertTrue(pp.detect(g16, np.array([1.0, 1e9, 2.0]))["overflow"])
        self.assertFalse(pp.detect(np.array([1.0, 2.0], np.float16), np.array([1.0, 2.0]))["overflow"])
        # inf in the fp32 model too (e.g. a masked -inf output): not an fp16 problem
        self.assertFalse(pp.detect(g16, np.array([1.0, np.inf, 2.0]))["overflow"])
        self.assertFalse(pp.detect(g16, None)["overflow"])

    def test_switch(self):
        self.assertEqual(pp.enabled("auto"), "auto")
        self.assertEqual(pp.enabled(None), "off")
        with self.assertRaises(ValueError):
            pp.enabled("on")


class MixedExportHelpersTest(unittest.TestCase):
    def test_is_mixed(self):
        from modelblaster.pipeline.kb_export import _is_mixed
        self.assertFalse(_is_mixed(["bmm_f16", "add_f16"]))
        self.assertFalse(_is_mixed(["bmm", "add"]))
        self.assertTrue(_is_mixed(["bmm_f16", "bmm"]))
        self.assertTrue(_is_mixed(["strided_copy_f16", "cast_f16_to_f32"]))
        self.assertTrue(_is_mixed(["cast_f16_to_f32", "bmm"]))       # casts: always the mixed library
        self.assertTrue(_is_mixed(["cast_f32_to_f16"]))               # a plan item holding one cast
        self.assertFalse(_is_mixed(["bmm", "strided_copy"]))          # all-fp32 island partition

    def test_mixed_library(self):
        from modelblaster.pipeline.kb_export import DEFAULT_HAND_CU, _MixedLibrary
        lib = _MixedLibrary(DEFAULT_HAND_CU)
        self.assertIn("__half", lib.slot_source("kernel_bmm_f16"))
        self.assertNotIn("__half", lib.slot_source("kernel_bmm"))
        self.assertIn("__float2half_rn", lib.slot_source("kernel_cast_f32_to_f16"))
        self.assertIn("cuda_fp16.h", lib.head_source())

    def test_cast_specs_registered(self):
        from modelblaster.pipeline.reference_kernels import KERNEL_SPECS
        from modelblaster.pipeline.cuda_dtypes import op_dtype
        self.assertIn("cast_f16_to_f32", KERNEL_SPECS)
        self.assertIn("cast_f32_to_f16", KERNEL_SPECS)
        self.assertNotIn("cast_f16_to_f32_f16", KERNEL_SPECS)
        self.assertEqual(op_dtype("cast_f32_to_f16"), "fp16")
        self.assertEqual(op_dtype("cast_f16_to_f32"), "fp32")


OVF = '''import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = nn.Parameter(torch.linspace(-1.0, 1.0, 64).reshape(8, 8))
        self.lin = nn.Linear(8, 8)
    def forward(self, x):
        w = torch.exp(self.A * 14.0)            # folded at extraction: up to e^14 = 1.2e6 > fp16 max
        h = torch.relu(self.lin(x))
        return torch.matmul(h, w) + 1.0
def get_inputs():
    return [torch.rand(4, 8)]
def get_init_inputs():
    return []
'''

PLAIN = '''import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)
    def forward(self, x):
        return torch.relu(self.lin(x))
def get_inputs():
    return [torch.rand(4, 8)]
def get_init_inputs():
    return []
'''


@unittest.skipUnless(_have_torch(), "needs torch")
class PolicyTest(unittest.TestCase):
    def _run(self, src, d: Path, policy="auto"):
        (d / "m.py").write_text(src)
        return pp.extract(d / "m.py", d / "ir", extra=["--bench-target-mb", "0"], policy=policy, log=lambda *a: None)

    def test_overflow_model_gets_islands(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rep = self._run(OVF, d)
            self.assertTrue(rep["detector"]["overflow"], rep)
            self.assertTrue(rep["applied"], rep)
            g = json.loads((d / "ir" / "graph.json").read_text())
            kinds = [o["op"] for o in g["ops"]]
            self.assertTrue(any(k.endswith("_f16") for k in kinds))           # the linear/relu part stays fp16
            self.assertTrue(any(not k.endswith("_f16") and k not in ("view",) for k in kinds))
            self.assertIn("cast_f16_to_f32", kinds)
            self.assertIn("cast_f32_to_f16", kinds)                          # back to fp16 for the epilogue
            # the output (max ~1.1e4) is inside the fp16 range: it stays fp16, only the folded exp() is an island
            self.assertEqual(g["precision_policy"]["fp32_outputs"], [])
            self.assertIn("_mb_folded_0", g["precision_policy"]["fp32_consts"])
            self.assertTrue(rep["detector"]["nonfinite_consts"])
            io = np.load(d / "ir" / "io.npz")
            self.assertTrue(np.isfinite(io["output"].astype(np.float32)).all())
            chk = rep["check"]
            self.assertEqual(chk["nonfinite_walk"], 0)
            self.assertLess(chk["max_rel_err"], 1e-3)
            # export: one mixed partition
            from modelblaster.pipeline.kb_export import export_problem
            man = json.loads((export_problem(d / "ir", d / "prob") / "manifest.json").read_text())
            self.assertEqual(man["dtype"], "fp16")
            self.assertEqual(man["tolerance"]["output_rel"], 1e-2)
            byid = {s["id"]: s for s in man["slots"]}
            self.assertIn("cast_f16_to_f32", byid)
            f32_slots = [s for s in man["slots"] if not s["op"].endswith("_f16") and not s["op"].startswith("cast")]
            self.assertTrue(f32_slots)
            for s in f32_slots:
                self.assertIn("fp32 island", s["semantics"])
                self.assertNotIn("__half", (d / "prob" / s["file"]).read_text())
            self.assertEqual(man["buffers"][man["outputs"][0]]["dtype"], "f16")
            if _have_gpu():
                from modelblaster.pipeline.kb_export import build_and_run
                arch = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                                      capture_output=True, text=True).stdout.split()[0].replace(".", "")
                res = build_and_run(d / "prob", arch=arch)
                self.assertTrue(res["passed"], res.get("log"))

    def test_policy_off_and_no_overflow(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rep = self._run(PLAIN, d)
            self.assertFalse(rep["applied"])
            self.assertFalse(rep["detector"]["overflow"])
            g = json.loads((d / "ir" / "graph.json").read_text())
            self.assertNotIn("precision_policy", g)
            self.assertTrue(all(o["op"].endswith("_f16") or o["op"] == "view" for o in g["ops"]))
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rep = self._run(OVF, d, policy="off")
            self.assertFalse(rep["applied"])
            self.assertTrue(rep["detector"]["overflow"])                     # reported, not acted on
            self.assertNotIn("precision_policy", json.loads((d / "ir" / "graph.json").read_text()))


if __name__ == "__main__":
    unittest.main()
