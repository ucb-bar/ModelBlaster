"""The PyTorch baseline of the model run (pipeline/torch_baseline.py, torch_bench.py;
docs/gpu_harness_perf.md "Against PyTorch"): the method (ABBA blocks, upper medians,
same-device enforcement), the tolerance check, TF32 detection, the bundle (source at
the extractor's dims, IR weights in the state dict), the result fields, the driver's
MB_WALL lines; and, with a CUDA GPU, the whole comparison on a tiny model.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline import torch_baseline as tb  # noqa: E402
from modelblaster.pipeline.kb_export import parse_driver_output  # noqa: E402

UUID = "GPU-11111111-2222-3333-4444-555555555555"
TIMING = ("MB_TIMING {b} gpu_med={g} gpu_mean={g} gpu_min=1.0 gpu_max=99.0 gpu_std=0.5 gpu_n=50 "
          "wall_med={w} wall_mean={w} wall_min=1.0 wall_max=99.0 wall_std=1.0 wall_n=50 "
          "thru_wall={t} thru_gpu={t} thru_n=100")

MODEL = '''import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(3, c, 3)
        self.bn = nn.BatchNorm2d(c)
        self.scale = nn.Parameter(torch.randn(c, 1, 1))
    def forward(self, x):
        return torch.relu(self.bn(self.conv(x))) * self.scale
batch_size = 2
channels = 8
def get_inputs():
    return [torch.randn(batch_size, 3, 12, 12)]
def get_init_inputs():
    return [channels]
'''


def _blk(g, w, t=None):
    s = {"med": g, "mean": g, "min": g, "max": g, "std": 0.01 * g, "n": 10}
    return {"gpu": s, "wall": {**s, "med": w, "mean": w}, "thru_wall_us": t or w, "thru_gpu_us": t or g, "thru_n": 100}


class FakeServer:
    def __init__(self, mode, variants, uuid=UUID, log=None):
        self.mode, self.variants, self.uuid, self.log = mode, variants, uuid, log

    def request(self, obj, timeout):
        if self.log is not None:
            self.log.append(f"torch{self.mode}")
        return {"event": "timing", "device": {"uuid": self.uuid},
                "variants": {v: _blk(g * (0.8 if self.mode else 1.0), g + 20) for v, g in self.variants.items()}}

    def close(self):
        pass


def _ready(mode, ok=("eager", "graph"), bad=()):
    v = {k: {"ok": True, "check": {"ok": True}, "setup_s": 0.1} for k in ok}
    v.update({k: {"ok": False, "error": "TypeError: boom", "setup_s": 0.1} for k in bad})
    return {"event": "ready", "device": {"uuid": UUID, "name": "NVIDIA L40S"}, "variants": v,
            "versions": {"torch": "2.x", "cuda": "12.8", "cudnn": 91900, "driver": "580"},
            "flags": {"allow_tf32_cudnn": bool(mode)}}


class MethodTest(unittest.TestCase):
    def test_parse_timing_lines(self):
        r = parse_driver_output("\n".join([TIMING.format(b="eager", g=12.5, w=20.0, t=11.0),
                                           TIMING.format(b="graph", g=10.0, w=15.0, t=9.0), "passed"]))
        self.assertEqual(set(r["timing"]), {"eager", "graph"})
        self.assertEqual(r["timing"]["graph"]["gpu"]["med"], 10.0)
        self.assertEqual(r["timing"]["graph"]["wall"]["n"], 50)
        self.assertEqual((r["timing"]["eager"]["thru_wall_us"], r["timing"]["eager"]["thru_n"]), (11.0, 100))
        self.assertEqual(parse_driver_output("passed\n")["timing"], {})          # MB_WALL off: no lines

    def test_aggregate_upper_median_cv_spread(self):
        a = tb.aggregate([_blk(10, 20), _blk(12, 22), _blk(11, 21), _blk(30, 40)])
        self.assertEqual(a["gpu_us"], 12)                          # upper median of the block medians
        self.assertEqual(a["wall_us"], 22)
        self.assertAlmostEqual(a["gpu_spread"], (30 - 10) / 12)
        self.assertAlmostEqual(a["gpu_cv"], 0.01)
        self.assertEqual((a["n_blocks"], a["reps"]), (4, 40))
        self.assertEqual((a["gpu_mean_us"], a["wall_mean_us"]), (12, 22))  # upper median of the block means

    def test_rounds_are_abba_on_one_device(self):
        log = []

        def ours(k):
            log.append("ours")
            return {"ok": True, "device": {"uuid": UUID.lower(), "label": "h:2"},
                    "timing": {"graph": _blk(50, 60), "eager": _blk(55, 70)}}
        srv = {0: FakeServer(0, {"eager": 100}, log=log), 1: FakeServer(1, {"eager": 100}, log=log)}
        raw = tb.run_rounds(ours, srv, rounds=3, uuid=UUID, params={})
        self.assertEqual(log, ["ours", "torch0", "torch1", "torch1", "torch0", "ours", "ours", "torch0", "torch1"])
        self.assertEqual(len(raw["ours"]), 3)
        self.assertEqual(len(raw["torch"][1]), 3)
        self.assertTrue(all(d == UUID.lower() for d in raw["devices"]))

    def test_same_device_enforced(self):
        good = lambda k: {"ok": True, "device": {"uuid": UUID}, "timing": {"graph": _blk(1, 2)}}  # noqa: E731
        other = lambda k: {"ok": True, "device": {"uuid": "GPU-other", "label": "h:3"},  # noqa: E731
                           "timing": {"graph": _blk(1, 2)}}
        with self.assertRaises(tb.DeviceMismatch):
            tb.run_rounds(other, {0: FakeServer(0, {"eager": 1})}, rounds=1, uuid=UUID, params={})
        with self.assertRaises(tb.DeviceMismatch):
            tb.run_rounds(good, {0: FakeServer(0, {"eager": 1}, uuid="GPU-other")}, rounds=1, uuid=UUID, params={})
        failing = lambda k: {"ok": False, "device": {"uuid": UUID}, "log": "MB_CHECK bad", "timing": {}}  # noqa: E731
        with self.assertRaises(RuntimeError):
            tb.run_rounds(failing, {0: FakeServer(0, {"eager": 1})}, rounds=1, uuid=UUID, params={})
        no_lines = lambda k: {"ok": True, "device": {"uuid": UUID}, "timing": {}}  # noqa: E731
        with self.assertRaises(RuntimeError):                     # a driver from before MB_WALL
            tb.run_rounds(no_lines, {0: FakeServer(0, {"eager": 1})}, rounds=1, uuid=UUID, params={})
        self.assertEqual(tb.norm_uuid("22350958-ee15"), "gpu-22350958-ee15")    # torch's uuid has no prefix

    def test_check_tolerance(self):
        import numpy as np

        spec = [{"name": "y", "numel": 4}, {"name": "z", "numel": 2}]
        tol = {"out_rel": 1e-3, "out_abs": 1e-5}
        ref = np.array([1, -2, 4, 0, np.nan, np.inf], np.float32)
        ok = tb.check_flat([ref[:4] + 1e-3, ref[4:]], ref, spec, tol)
        self.assertTrue(ok["ok"])                                   # 1e-3 <= 1e-3 * 4 + 1e-5; nan / inf match
        bad = tb.check_flat([ref[:4] + 1e-2, ref[4:]], ref, spec, tol)
        self.assertFalse(bad["outputs"]["y"]["ok"])
        self.assertFalse(tb.check_flat([ref[:4], np.array([0, np.inf], np.float32)], ref, spec, tol)["ok"])
        self.assertFalse(tb.check_flat([ref[:4]], ref, spec, tol)["ok"])           # an output missing

    def test_detect_math(self):
        fp32 = "__global__ void k(const float4 *x, float *y) { y[0] = x[0].x * 2.0f; }  // no tf32 here"
        tf32 = ("#include <mma.h>\nusing namespace nvcuda;\n"
                "wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a;")
        half = "__global__ void k(const __half *x) {}"
        self.assertEqual(tb.detect_math({"conv2d": fp32})["mode"], "fp32")
        m = tb.detect_math({"conv2d": fp32, "linear": tf32})
        self.assertEqual(m["mode"], "tf32")
        self.assertIn("linear", m["evidence"]["tf32"])
        self.assertEqual(tb.detect_math({"a": "wmma::mma_sync(c, a, b, c);"})["mode"], "tensor_cores")
        self.assertEqual(tb.detect_math({"a": half})["mode"], "reduced")
        self.assertEqual(tb.detect_math({"a": "/* tf32 later */ float x;"})["mode"], "fp32")
        self.assertIn("libraries", tb.detect_math({"a": "cublasSgemm(h, ...);"})["evidence"])
        self.assertEqual(tb.headline_mode({"mode": "fp32"}), "tf32_off")
        self.assertEqual(tb.headline_mode({"mode": "tf32"}), "best")

    def _raw(self):
        return {"ours": [{"graph": _blk(50, 60), "eager": _blk(55, 70)}] * 2,
                "torch": {0: [{"eager": _blk(200, 230), "graph": _blk(150, 170)}] * 2,
                          1: [{"eager": _blk(100, 130), "graph": _blk(90, 110)}] * 2},
                "devices": [UUID.lower()] * 6, "ours_runs": [{"ok": True}] * 2}

    def test_rows_and_headline(self):
        ready = {0: (None, _ready(0, bad=("compile_mna",))), 1: (None, _ready(1))}
        rows, head = tb.build_rows(self._raw(), ready, {"mode": "fp32"})
        labels = [(r["side"], r["variant"], r["tf32"]) for r in rows]
        self.assertEqual(labels[:2], [("ours", "graph", None), ("ours", "eager", None)])
        self.assertIn(("torch", "compile_mna", False), labels)
        failed = next(r for r in rows if r["variant"] == "compile_mna")
        self.assertFalse(failed["check"])
        self.assertIsNone(failed.get("gpu_us"))                     # reported, never timed
        g0 = next(r for r in rows if r["variant"] == "graph" and r["tf32"] is False)
        self.assertAlmostEqual(g0["speedup_gpu"], 150 / 50)
        self.assertAlmostEqual(g0["speedup_wall"], 170 / 60)
        # ours fp32: against TF32 off only, though TF32 on is faster
        self.assertEqual((head["mode"], head["torch_best_variant"], head["torch_best_us"]), ("tf32_off", "graph", 150))
        self.assertAlmostEqual(head["speedup_vs_torch"], 3.0)
        self.assertAlmostEqual(head["speedup_vs_torch_eager"], 4.0)
        self.assertAlmostEqual(head["speedup_vs_torch_wall"], 170 / 60)
        # ours TF32: the faster torch mode
        _, head = tb.build_rows(self._raw(), ready, {"mode": "tf32"})
        self.assertEqual((head["mode"], head["torch_best_variant"], head["torch_best_us"]), ("best", "graph+tf32", 90))
        f = tb.summary_fields({"status": "ok", "headline": head})
        self.assertEqual((f["torch_us"], f["torch_graph_us"], f["torch_eager_us"]), (90, 90, 100))
        self.assertAlmostEqual(f["speedup_vs_torch"], 90 / 50)
        self.assertEqual((f["torch_status"], f["torch_reason"]), ("ok", None))
        f = tb.summary_fields({"status": "skipped", "reason": "no torch python"})
        self.assertEqual((f["torch_status"], f["torch_reason"], f["torch_us"]), ("skipped", "no torch python", None))

    def test_fp16_headline(self):
        """An fp16 IR: torch in half, one math mode (TF32 does not apply), headline 'fp16'."""
        raw = self._raw()
        raw["torch"] = {0: raw["torch"][0]}
        rows, head = tb.build_rows(raw, {0: (None, _ready(0))}, {"mode": "tensor_cores"}, "float16")
        t = [r for r in rows if r["side"] == "torch"]
        self.assertTrue(t and all(r["tf32"] is None and r["label"].endswith(" fp16") for r in t))
        self.assertEqual((head["mode"], head["torch_best_variant"], head["torch_best_us"]), ("fp16", "graph", 150))
        self.assertEqual(tb.headline_mode({"mode": "fp32"}, "float16"), "fp16")

    def test_compare_fp16_uses_one_mode(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            ir, prob = self._tiny_ir(tmp)
            g = json.loads((ir / "graph.json").read_text())
            (ir / "graph.json").write_text(json.dumps({**g, "quant": "fp16"}))
            (prob / "slots").mkdir(parents=True)
            (prob / "slots" / "conv2d_f16.cu").write_text("__global__ void k(__half *x) { asm(\"mma.sync.aligned\"); }")
            (prob / "manifest.json").write_text(json.dumps({"slots": [{"id": "conv2d_f16", "file": "slots/conv2d_f16.cu"}],
                                                            "tolerance": {"output_rel": 1e-2, "output_abs": 1e-5}}))
            log = []
            ours = lambda k: {"ok": True, "device": {"uuid": UUID}, "timing": self._raw()["ours"][0]}  # noqa: E731
            srv = {0: (FakeServer(0, {"eager": 200, "graph": 150}, log=log), _ready(0))}
            res = tb.compare(ir_dir=ir, problem=prob, device={"uuid": UUID, "host": "h"}, work=tmp / "w", arch="89",
                             rounds=2, ours=ours, servers=srv)
            self.assertEqual(res["method"]["dtype"], "float16")
            self.assertEqual(res["math"]["mode"], "tensor_cores")
            self.assertEqual(res["math"]["headline_mode"], "fp16")
            self.assertEqual(log, ["torch0", "torch0"])
            self.assertIn("| torch CUDA graph fp16 | - | pass |", tb.table(res))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_ablation_uses_the_paired_ratio(self):
        from modelblaster.pipeline import kb_ablate

        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "work").mkdir()
            (tmp / "work" / "summary.json").write_text(json.dumps({"irs": {"m": {
                "model_e2e_us": 50.0, "seed_e2e_us": 1000.0, "model_passed": True, "torch_us": 200.0,
                "speedup_vs_torch": 4.2, "speedup_vs_torch_wall": 3.1}}}))
            row = kb_ablate.collect_graph_opt(tmp, None)["problems"]["m"]
            self.assertEqual((row["torch_us"], row["vs_torch_paired"], row["vs_torch_wall"]), (200.0, 4.2, 3.1))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_compare_with_injected_sides(self):
        """compare(): ours and the servers injected -> the full result and the table."""
        tmp = Path(tempfile.mkdtemp())
        try:
            ir, prob = self._tiny_ir(tmp)
            (prob / "slots").mkdir(parents=True)
            (prob / "slots" / "conv2d.cu").write_text("__global__ void k(float *x) {}")
            (prob / "manifest.json").write_text(json.dumps({"slots": [{"id": "conv2d", "file": "slots/conv2d.cu"}],
                                                            "tolerance": {"output_rel": 1e-3, "output_abs": 1e-5}}))
            ours = lambda k: {"ok": True, "device": {"uuid": UUID}, "timing": self._raw()["ours"][0]}  # noqa: E731
            srv = {m: (FakeServer(m, {"eager": 200, "graph": 150}), _ready(m)) for m in (0, 1)}
            res = tb.compare(ir_dir=ir, problem=prob, device={"uuid": UUID, "name": "NVIDIA L40S", "host": "h"},
                             work=tmp / "w", arch="89", rounds=2, ours=ours, servers=srv)
            self.assertEqual(res["status"], "ok")
            self.assertTrue(res["same_device"])
            self.assertEqual(res["math"]["mode"], "fp32")
            self.assertEqual(res["method"]["rounds"], 2)
            self.assertEqual(res["headline"]["torch_best_variant"], "graph")
            self.assertIn("| torch CUDA graph | off | pass |", tb.table(res))
            with self.assertRaises(tb.Skip):                       # nothing to pin the comparison to
                tb.compare(ir_dir=ir, problem=prob, device={}, work=tmp / "w", arch="89", ours=ours, servers=srv)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_kbeval_driver_env(self):
        from modelblaster.pipeline.kb_eval import KBEval

        cmd = KBEval(kb=Path("/kb")).command(Path("j"), Path("o"), arch="89", work=Path("w"),
                                             driver_env=tb.timing_env(10, 200, 100))
        i = cmd.index("--driver-env")
        self.assertEqual(cmd[i + 1], "MB_WALL=1")
        self.assertIn("MB_WARM=10", cmd)
        self.assertIn("MB_THRU_N=100", cmd)
        self.assertNotIn("--driver-env", KBEval(kb=Path("/kb")).command(Path("j"), Path("o"), arch="89",
                                                                        work=Path("w")))

    def test_auto_skips_with_a_reason(self):
        from modelblaster.pipeline.kb_model_run import torch_comparison

        tmp = Path(tempfile.mkdtemp())
        try:
            ir, prob = self._tiny_ir(tmp, source=False)
            ok = {"passed": True, "device": {"uuid": UUID, "host": "u@h"}}
            r = torch_comparison(ir, prob, ok, mode="auto", work=tmp / "t", kbeval=None, arch="89", gpu_model=None)
            self.assertEqual(r["status"], "skipped")
            self.assertIn("no PyTorch source", r["reason"])
            (ir.parent / "src").mkdir()
            (ir.parent / "src" / f"{ir.name}.py").write_text(MODEL)
            os.environ.pop("MB_TORCH_PYTHON", None)
            r = torch_comparison(ir, prob, ok, mode="auto", work=tmp / "t", kbeval=None, arch="89", gpu_model=None)
            self.assertIn("no torch python for u@h", r["reason"])
            r = torch_comparison(ir, prob, {**ok, "passed": False}, mode="auto", work=tmp / "t", kbeval=None,
                                 arch="89", gpu_model=None)
            self.assertIn("did not pass", r["reason"])
            with self.assertRaises(RuntimeError):
                torch_comparison(ir, prob, ok, mode="on", work=tmp / "t", kbeval=None, arch="89", gpu_model=None)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ---- the bundle: the source at the extractor's dims, the IR's weights in its state dict
    def _tiny_ir(self, tmp: Path, source: bool = True, bn_folded: bool = True, wrong: bool = False):
        """A hand-made IR of MODEL (channels shrunk to 4, as an extractor would): graph.json,
        weights.npz with the batchnorm folded, io.npz from the torch forward."""
        import numpy as np
        import torch

        sys.modules.pop("_tiny_src", None)
        set_dir = tmp / "set"
        ir = set_dir / "ir16" / "7_TinyConvBn"
        ir.mkdir(parents=True)
        if source:
            (set_dir / "src").mkdir()
            (set_dir / "src" / "7_TinyConvBn.py").write_text(MODEL)
        torch.manual_seed(1)
        conv = torch.nn.Conv2d(3, 8, 3)
        gamma, beta = torch.rand(8) + 0.5, torch.randn(8)
        mean, var, scale = torch.randn(8) * 0.1, torch.rand(8) + 0.5, torch.randn(8, 1, 1)
        x = torch.randn(2, 3, 12, 12)
        with torch.no_grad():
            a = (gamma / torch.sqrt(var + 1e-5))
            y = torch.relu(conv(x) * a[None, :, None, None] + (beta - mean * a)[None, :, None, None]) * scale
        w = {"conv.weight": conv.weight.detach().numpy(), "conv.bias": conv.bias.detach().numpy(),
             "bn.scale": a.numpy(), "bn.bias_fused": (beta - mean * a).numpy(), "const.scale": scale.numpy()}
        if not bn_folded:
            w = {k: v for k, v in w.items() if not k.startswith("bn.")}
        if wrong:
            w["conv.weight"] = w["conv.weight"] * 1.5
        np.savez(ir / "weights.npz", **w)
        np.savez(ir / "io.npz", input=x.numpy().ravel(), output=y.numpy().ravel())
        (ir / "graph.json").write_text(json.dumps({
            "name": "kb_7_TinyConvBn", "quant": "fp32", "input": {"tensor": "x"},
            "output": {"tensors": ["mul"], "tensor": "mul"},
            "tensors": {"x": {"shape": [2, 3, 12, 12]}, "mul": {"shape": [2, 8, 10, 10]}}, "ops": []}))
        return ir, tmp / "prob"

    def test_bundle_maps_weights_and_checks(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            ir, _ = self._tiny_ir(tmp)
            self.assertEqual(tb.find_source(ir), tmp / "set" / "src" / "7_TinyConvBn.py")
            tol = {"out_rel": 1e-3, "out_abs": 1e-5}
            rec = tb.build_bundle(ir, tmp / "b", tol=tol)
            self.assertTrue(rec["cpu_check"]["ok"], rec["cpu_check"])
            self.assertEqual(rec["inputs"], [{"shape": [2, 3, 12, 12], "dtype": "float32"}])
            self.assertIn("bn.running_var", rec["mapping"]["bn_folded"])
            self.assertIn("scale", rec["mapping"]["mapped"])            # const.<name>: a bare parameter
            self.assertEqual(rec["mapping"]["unused_ir_weights"], [])
            for f in ("model.py", "recipe.json", "state.npz", "io.npz"):
                self.assertTrue((tmp / "b" / f).exists(), f)
            self.assertEqual(tb.build_bundle(ir, tmp / "b", tol=tol)["key"], rec["key"])    # reused
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_bundle_fp16(self):
        """An fp16 IR (fp16 weights and io): the bundle runs torch in half, checked in fp32 on the CPU."""
        import numpy as np

        tmp = Path(tempfile.mkdtemp())
        try:
            ir, _ = self._tiny_ir(tmp)
            g = json.loads((ir / "graph.json").read_text())
            (ir / "graph.json").write_text(json.dumps({**g, "quant": "fp16"}))
            W, io = np.load(ir / "weights.npz"), np.load(ir / "io.npz")
            np.savez(ir / "weights.npz", **{k: W[k].astype(np.float16) for k in W.files})
            np.savez(ir / "io.npz", **{k: io[k].astype(np.float16) for k in io.files})
            rec = tb.build_bundle(ir, tmp / "b", tol={"out_rel": 1e-2, "out_abs": 1e-5})
            self.assertEqual(rec["dtype"], "float16")
            self.assertEqual(rec["inputs"], [{"shape": [2, 3, 12, 12], "dtype": "float16"}])
            self.assertTrue(rec["cpu_check"]["ok"], rec["cpu_check"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_bundle_refuses_what_does_not_reproduce(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            ir, _ = self._tiny_ir(tmp, wrong=True)
            with self.assertRaises(tb.Skip) as c:
                tb.build_bundle(ir, tmp / "b", tol={"out_rel": 1e-3, "out_abs": 1e-5}, sizings=[(16, 64, 0)])
            self.assertIn("CPU output check failed", str(c.exception))
            tmp2 = Path(tempfile.mkdtemp())
            ir2, _ = self._tiny_ir(tmp2, bn_folded=False)
            with self.assertRaises(tb.Skip) as c:
                tb.build_bundle(ir2, tmp2 / "b", tol={"out_rel": 1e-3, "out_abs": 1e-5}, sizings=[(16, 64, 0)])
            self.assertIn("without IR weights", str(c.exception))
            shutil.rmtree(tmp2, ignore_errors=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def _have_gpu() -> bool:
    if shutil.which("nvcc") is None or shutil.which("nvidia-smi") is None:
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@unittest.skipUnless(_have_gpu(), "needs torch, nvcc and a CUDA GPU")
class LocalGPUTest(unittest.TestCase):
    """The whole comparison on this host's GPU: a tiny model extracted, composed and timed by
    the builtin host, against its PyTorch module (eager + CUDA graph, TF32 off / on)."""

    def test_model_run_against_torch(self):
        from modelblaster.pipeline.kb_model_run import model_runtime

        root = Path(__file__).resolve().parents[3]
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "set" / "src").mkdir(parents=True)
            src = tmp / "set" / "src" / "8_TinyMLP.py"
            src.write_text(MODEL)
            cp = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file", str(src),
                                 "--bench-target-mb", "16", "--bench-target-weights-mb", "64",
                                 "--out-dir", str(tmp / "set" / "ir16" / "8_TinyMLP"), "--quant", "fp32"],
                                cwd=root, capture_output=True, text=True, timeout=600,
                                env={**os.environ, "PYTHONPATH": os.pathsep.join([str(root),
                                                                                  os.environ.get("PYTHONPATH", "")])})
            self.assertEqual(cp.returncode, 0, cp.stderr[-2000:])
            arch = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.split()[0].replace(".", "")
            res = model_runtime(tmp / "set" / "ir16" / "8_TinyMLP", work=tmp / "w", arch=arch, kbeval=None,
                                torch_baseline="on", keep=True,
                                torch_opts={"python": sys.executable, "rounds": 2, "variants": ["eager", "graph"],
                                            "budget_ms": 20.0, "warm": 3, "thru_n": 20})
            t = res["torch_baseline"]
            self.assertEqual(t["status"], "ok", t)
            self.assertTrue(t["same_device"])
            self.assertEqual(t["math"]["mode"], "fp32")
            self.assertEqual(len(t["raw"]["order"]), 2 * 3)
            rows = {(r["side"], r["variant"], r["tf32"]): r for r in t["rows"]}
            for k in (("ours", "graph", None), ("ours", "eager", None), ("torch", "eager", False),
                      ("torch", "graph", False), ("torch", "eager", True), ("torch", "graph", True)):
                self.assertIn(k, rows)
                self.assertTrue(rows[k]["check"])
                self.assertGreater(rows[k]["gpu_us"], 0)
                self.assertGreater(rows[k]["wall_us"], 0)
                self.assertGreater(rows[k]["thru_wall_us"], 0)
            self.assertEqual(t["flags"]["tf32_off"]["allow_tf32_cudnn"], False)
            self.assertEqual(t["flags"]["tf32_on"]["allow_tf32_cudnn"], True)
            self.assertIsNotNone(res["speedup_vs_torch"])
            self.assertEqual(res["torch_status"], "ok")
            self.assertTrue(t["bundle"]["cpu_check"]["ok"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
