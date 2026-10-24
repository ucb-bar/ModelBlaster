"""GPU integration: ModelBlaster as the graph optimiser, whole flow.

A tiny conv -> bn -> relu (x2) model (small enough that launches and
intermediate round trips, which fusion removes, dominate) is extracted, then
optimised over several rounds with KernelBlaster driven by a scripted
``stub:`` LLM (a ``--plugin`` file): it answers with real fused kernels for fused slots and
returns every other slot unchanged. The rounds must fuse conv->bn(->relu)
into one kernel per block -- a chain built over two rounds -- accept it on
re-measured times, reach a fixpoint and verify the final model; and a split
hint goes through variant -> optimise -> decide.

Needs torch, nvcc, a CUDA GPU and a KernelBlaster checkout
(``MB_KB_ROOT``, default ``third_party/KernelBlaster``) whose python
(``MB_KB_PYTHON``, default this one) has its dependencies; skipped otherwise.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

KB = Path(os.environ.get("MB_KB_ROOT", ROOT / "third_party" / "KernelBlaster"))
KB_PY = os.environ.get("MB_KB_PYTHON", sys.executable)

MODEL = '''import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.b1 = nn.BatchNorm2d(8)
        self.c2 = nn.Conv2d(8, 8, 3, padding=1)
        self.b2 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.b2(self.c2(self.relu(self.b1(self.c1(x))))))
def get_inputs():
    return [torch.randn(1, 3, 8, 8)]
def get_init_inputs():
    return []
'''

# One thread per output; the same arithmetic as the hand conv + bn (+ relu)
# kernels, so the fused result is bit-identical -- but one launch and no
# intermediate round trip.
FUSED_KERNEL = r'''
__global__ void k_conv_bn(const float *x, const float *w, const float *cb, const float *s, const float *sb,
                          float *y, int N, int IC, int IH, int IW, int OC, int OH, int OW, int KH, int KW,
                          int SH, int SW, int PH, int PW, int DH, int DW, int relu) {
    long i = MB_TID; long tot = (long)N * OC * OH * OW;
    if (i >= tot) return;
    int ow = i % OW; int oh = (i / OW) % OH; int oc = (i / ((long)OW * OH)) % OC; int n = i / ((long)OW * OH * OC);
    double acc = cb ? cb[oc] : 0.0f;
    for (int ic = 0; ic < IC; ic++)
        for (int kh = 0; kh < KH; kh++) {
            int ih = oh * SH - PH + kh * DH; if (ih < 0 || ih >= IH) continue;
            for (int kw = 0; kw < KW; kw++) {
                int iw = ow * SW - PW + kw * DW; if (iw < 0 || iw >= IW) continue;
                acc += x[(((long)n * IC + ic) * IH + ih) * IW + iw] * w[(((long)oc * IC + ic) * KH + kh) * KW + kw];
            }
        }
    float v = (float)acc * s[oc] + sb[oc];
    y[i] = relu ? fmaxf(v, 0.0f) : v;
}
'''
CALL = ("    long tot = (long)a_N * a_OC * b_H * b_W;\n"
        "    k_conv_bn<<<MB_GRID(tot), MB_THREADS>>>(a_input, a_weight, a_bias, b_scale, b_bias, {out}, a_N, a_IC,\n"
        "        a_IH, a_IW, a_OC, b_H, b_W, a_KH, a_KW, a_SH, a_SW, a_PH, a_PW, a_DH, a_DW, {relu});\n")
BN_RELU = r'''
__global__ void k_bn_relu(const float *x, const float *s, const float *b, float *y, long tot, int C, int HW) {
    long i = MB_TID; if (i < tot) { int c = (i / HW) % C; y[i] = fmaxf(x[i] * s[c] + b[c], 0.0f); }
}
'''

PLUGIN = '''import json
from kernelblaster.agents.opt_multislot import extract_code
from kernelblaster.agents.utils import agentic_llm

FUSED = json.loads({fused!r})


def scripted(msgs):
    user = msgs[-1]["content"]
    for sid in sorted(FUSED, key=len, reverse=True):
        if "## Slot `" + sid + "`" in user or ("failed" in user and "`" + sid + "`" in user):
            return "```cuda\\n" + FUSED[sid] + "```"
    cur = extract_code(user.split("Current source:")[-1] if "Current source:" in user
                       else user.split("Current slot source:")[-1])
    return "```cuda\\n" + (cur or "") + "```"


agentic_llm.register_stub("graphopt", scripted)
'''


def fused_sources() -> dict[str, str]:
    from modelblaster.pipeline.reference_kernels import KERNEL_SPECS
    from modelblaster.pipeline.kb_proposals import spec_from_kind

    out = {}
    for kind, body in (
            ("fused__conv2d__batchnorm2d", FUSED_KERNEL + "{sig} {{\n" + CALL.format(out="b_output", relu=0) + "}}\n"),
            ("fused__conv2d__batchnorm2d__relu", FUSED_KERNEL + "{sig} {{\n" + CALL.format(out="c_output", relu=1) + "}}\n"),
            ("fused__batchnorm2d__relu", BN_RELU + "{sig} {{\n    long tot = (long)a_N * a_C * a_H * a_W;\n"
             "    k_bn_relu<<<MB_GRID(tot), MB_THREADS>>>(a_input, a_scale, a_bias, b_output, tot, a_C, a_H * a_W);\n}}\n")):
        sig = spec_from_kind(kind, KERNEL_SPECS).signature.rstrip(";")
        out[kind] = body.replace("{sig}", sig).replace("{{", "{").replace("}}", "}")
    return out


def _have_all() -> str | None:
    if shutil.which("nvcc") is None or shutil.which("nvidia-smi") is None:
        return "needs nvcc and a CUDA GPU"
    if not (KB / "scripts" / "run_multislot.py").exists():
        return f"needs a KernelBlaster checkout (MB_KB_ROOT; tried {KB})"
    try:
        import torch  # noqa: F401
        if subprocess.run(["nvidia-smi"], capture_output=True, timeout=30).returncode != 0:
            return "nvidia-smi failed"
        if subprocess.run([KB_PY, "-c", "import loguru, pydantic"], capture_output=True).returncode != 0:
            return f"{KB_PY} lacks KernelBlaster's dependencies (MB_KB_PYTHON)"
    except Exception as e:  # noqa: BLE001
        return str(e)
    return None


SKIP = _have_all()


def _arch() -> str:
    return subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.split()[0].replace(".", "")


@unittest.skipIf(SKIP is not None, SKIP or "")
class GraphOptFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(os.environ.get("MB_TEST_TMP", tempfile.gettempdir()))
        base.mkdir(parents=True, exist_ok=True)
        cls.tmp = Path(tempfile.mkdtemp(prefix="mb_graphopt_", dir=base))
        (cls.tmp / "tiny.py").write_text(MODEL)
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT.parent), os.environ.get("PYTHONPATH", "")])}
        cp = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file",
                             str(cls.tmp / "tiny.py"), "--out-dir", str(cls.tmp / "ir"), "--quant", "fp32"],
                            cwd=ROOT, capture_output=True, text=True, timeout=600, env=env)
        assert cp.returncode == 0, cp.stderr[-2000:]
        (cls.tmp / "plugin.py").write_text(PLUGIN.format(fused=json.dumps(fused_sources())))
        cls.arch = _arch()

    @classmethod
    def tearDownClass(cls):
        if not os.environ.get("MB_TEST_KEEP"):
            shutil.rmtree(cls.tmp, ignore_errors=True)

    def _cfg(self, **kw):
        from modelblaster.pipeline.kb_graph_opt import Config
        base = dict(kb=KB, model="stub:graphopt", plugins=[str(self.tmp / "plugin.py")], python=KB_PY,
                    arch=self.arch, rounds=4, rollouts=1, steps=1, variant_steps=1, max_fix=1, top_k=2,
                    max_variants=2, remeasure=5, profile_runs=1)
        base.update(kw)
        return Config(**base)

    def test_rounds_fuse_a_chain_to_a_fixpoint(self):
        from modelblaster.pipeline import graph_hints as gh
        from modelblaster.pipeline.kb_graph_opt import run

        g = json.loads((self.tmp / "ir" / "graph.json").read_text())
        self.assertEqual([o["op"] for o in g["ops"]], ["conv2d", "batchnorm2d", "relu"] * 2)
        work = self.tmp / "work"
        # conv->bn alone gains a few % here, near the 3% bar on a shared GPU, so
        # the chain may also be built as bn->relu then conv->(bn+relu): one
        # round more before the fixpoint
        s = run([self.tmp / "ir"], work, self._cfg(rounds=6))
        doc = gh.load(work / "graph_hints.json")
        status = {h["id"]: h["status"] for h in doc["hints"]}
        chain = "fuse:fused__conv2d__batchnorm2d__relu"
        self.assertEqual(status.get(chain), "accepted", json.dumps(status))
        self.assertIsNotNone(s["fixpoint"], s)                        # stopped on its own
        ir = next(iter(s["irs"].values()))
        self.assertTrue(ir["passed"], s)
        self.assertEqual(sorted(ir["fused"]), ["c1+b1+relu", "c2+b2+relu#1"])
        self.assertGreater(ir["speedup"], 1.0, s)
        # the chain was built over two rounds: a two-stage fusion first, then its extension
        rounds = json.loads((work / "state.json").read_text())["rounds"]
        acc = [(r["round"], d["hint"]) for r in rounds for d in r["decisions"] if d["status"] == "accepted"]
        self.assertGreaterEqual(len(acc), 2, acc)
        self.assertEqual(acc[-1][1], chain)
        self.assertLess(acc[0][0], acc[-1][0])
        ev = next(h for h in doc["hints"] if h["id"] == chain)["evidence"][-1]
        self.assertEqual(len(ev["remeasured"]["variant_us"]), 5)
        self.assertGreater(ev["gain"], 0.03)
        # the fused winner was cached and the final model used it
        final = json.loads((work / "final" / "ir" / "problem" / "manifest.json").read_text()) \
            if (work / "final" / "ir" / "problem").exists() else None
        if final is not None:
            tags = {x["id"]: x["seed_source"] for x in final["slots"]}
            self.assertTrue(tags["fused__conv2d__batchnorm2d__relu"].startswith("cache:"), tags)

    def test_split_hint_variant_and_decision(self):
        """KB-style split hint on the two convs (different shape buckets):
        MB builds the variant (two conv slots, each seeded with the current
        kernel), KB optimises both problems, MB re-measures and decides."""
        from modelblaster.pipeline import graph_hints as gh
        from modelblaster.pipeline.kb_graph_opt import KBRunner, evaluate_hint
        from modelblaster.pipeline.kb_ingest import CudaCache
        from modelblaster.pipeline.kb_plan import profile_ir
        from modelblaster.pipeline.kb_proposals import _all_specs

        work = self.tmp / "split"
        cfg = self._cfg(remeasure=3)
        prof = profile_ir(self.tmp / "ir", work / "profile", arch=self.arch, runs=1)
        h = {"id": "split:conv2d", "kind": "split_slot", "target": "cuda", "slot": "conv2d", "by": "shape_bucket",
             "groups": [{"bucket": "b0", "ops": ["c1"]}, {"bucket": "b1", "ops": ["c2"]}], "status": "proposed",
             "source": [{"by": "kb", "ir": prof["ir"], "est_gain_pct": 5.0}], "evidence": []}
        doc = gh.new_doc()
        d = evaluate_hint(h, prof, work / "variant", kb=KBRunner(cfg, work / "knowledge"),
                          cache=CudaCache(work / "cache"), doc=doc, cfg=cfg, specs=dict(_all_specs()))
        self.assertIn(d["status"], ("accepted", "rejected"))
        self.assertIsNotNone(d["gain"], d)                            # both sides built, ran and were timed
        man = json.loads((work / "variant" / "variant" / "manifest.json").read_text())
        self.assertEqual(sorted(s["id"] for s in man["slots"] if s["op"] == "conv2d"), ["conv2d__v0", "conv2d__v1"])
        self.assertEqual({s.get("variant_of") for s in man["slots"] if s["op"] == "conv2d"}, {"conv2d"})
        self.assertEqual(gh.get(doc, "split:conv2d")["evidence"][-1]["decision"], d["status"])


if __name__ == "__main__":
    unittest.main()
