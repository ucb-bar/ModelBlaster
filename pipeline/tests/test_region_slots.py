"""Region slots (A2 item 4) and the capability switches (capabilities.py).

Pure python: matchers on the kept sweep IR of 41 (skipped when absent) and on 43's graph (tests/data), the rewrite's
refusals, the fixed signature (roles, dead readers ignored), the composite seed (member sources by variant, levels,
concurrent members on helper-thread streams with their own copies), the fuse_region hint applier, spec plumbing
(with_fused_specs, the in-process registry kb_ingest uses), Caps defaults / env / round key / proposals / model
context. With torch: a tiny bidirectional GRU's rnn_stack region exported (one region slot, watchdog driver); with
GPU + nvcc it builds and passes the driver's checks.
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
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline import region_slots as rs  # noqa: E402
from modelblaster.pipeline.kb_proposals import _all_specs, with_fused_specs  # noqa: E402

HERE = Path(__file__).resolve().parent
FARM = Path(__file__).resolve().parents[3].parent / "l3_farm" / "results"
if not FARM.is_dir():
    FARM = Path.home() / "Documents" / "ModelBlaster-clone" / "l3_farm" / "results"
G41 = FARM / "41_GRUBidirectional" / "work" / "final" / "41_GRUBidirectional" / "ir" / "graph.json"


def _specs(g):
    return with_fused_specs(g, dict(_all_specs()))


@unittest.skipUnless(G41.is_file(), "no local 41 IR")
class Region41Test(unittest.TestCase):
    def setUp(self):
        self.g = json.loads(G41.read_text())
        self.sp = _specs(self.g)

    def test_rnn_stack(self):
        names = rs.find_regions(self.g, "rnn_stack", self.sp)[0]
        kinds = [next(o["op"] for o in self.g["ops"] if o["name"] == n) for n in names]
        self.assertEqual(kinds.count("gru_seq_f16"), 12)
        self.assertEqual(kinds.count("catn_f16"), 6)                 # per-layer concats; the dead h_n stack is out
        self.assertEqual(kinds.count("view"), 12)                    # h0 slices
        g2, spec = rs.fuse_region(self.g, names, self.sp, "rnn_stack")
        op = next(o for o in g2["ops"] if rs.is_region(o))
        self.assertTrue(op["op"].endswith("_f16"))
        self.assertEqual(op["outputs"], ["gru__g43"])                # h_n read only by dead ops: scratch
        self.assertIn("gru__g3", op["internal"])
        roles = {r for _, r, _ in spec.params}
        self.assertEqual(roles, {"in", "out", "scratch"})
        self.assertEqual(sum(r == "out" for _, r, _ in spec.params), 1)
        self.assertIn("mb_sm_budget()", spec.semantics)
        self.assertIs(rs.known_spec(op["op"]), spec)
        # the same region from the spec plumbing every caller uses
        self.assertIn(op["op"], with_fused_specs(g2, dict(_all_specs())))
        from modelblaster.pipeline.op_classes import op_class
        self.assertEqual(op_class(op["op"], recurrent=False), "gemm")
        self.assertEqual(op_class(op["op"], recurrent=True), "recurrent")

    def test_seed(self):
        names = rs.find_regions(self.g, "rnn_stack", self.sp)[0]
        g2, spec = rs.fuse_region(self.g, names, self.sp, "rnn_stack")
        op = next(o for o in g2["ops"] if rs.is_region(o))
        srcs = {f"gru_seq_f16__v{k}": f'#include <cuda_fp16.h>\nstatic int st_v{k};\nextern "C" void '
                f'kernel_gru_seq_f16(const __half *x, const __half *h0, const __half *a, const __half *b, '
                f'const __half *c, const __half *d, __half *y, __half *hn, int B, int T, int I, int H, int r) '
                f'{{ st_v{k}++; }}\n' for k in range(4)}
        srcs["catn_f16__v0"] = 'extern "C" void kernel_catn_f16(const __half *in0, const __half *in1, const __half *in2, ' \
            'const __half *in3, const __half *in4, const __half *in5, const __half *in6, const __half *in7, ' \
            '__half *output, int n_in, int outer, int inner, int c0, int c1, int c2, int c3, int c4, int c5, int c6, ' \
            'int c7) {}\n'
        subs = rs.seed_sources(op, srcs, self.sp)
        self.assertEqual(len(subs), 18)
        from modelblaster.pipeline.kb_export import DEFAULT_HAND_CU, CudaLibrary, _hand_source
        lib = CudaLibrary(_hand_source(DEFAULT_HAND_CU, "fp16"))
        par = spec.cuda_seed(lib, subs, parallel=True)
        seq = spec.cuda_seed(lib, subs, parallel=False)
        self.assertIn("std::thread", par)
        self.assertNotIn("std::thread", seq)
        self.assertEqual(par.count("cudaEventRecord(mb_fork"), 6)     # one fork per layer (fwd || rev)
        # concurrent members never share statics: v0 (layers 1-5 fwd) and v1 (rev) each one copy, and the
        # sequential seed has one namespace per distinct source
        import re
        self.assertEqual(len(re.findall(r"namespace member\d+ \{", seq)), 5)
        # fwd / rev members already use different sources (v0 / v1, v3 / v2): no extra lane copy is needed
        self.assertEqual(len(re.findall(r"namespace member\d+ \{", par)), 5)
        self.assertIn("(in_h0 + 2560)", par)                         # h0 slice = pointer arithmetic
        # the forward winner for every GRU (A1's reuse): fwd and rev of a layer run concurrently -> two copies
        same = {**srcs, **{f"gru_seq_f16__v{k}": srcs["gru_seq_f16__v0"] for k in range(4)}}
        par1 = spec.cuda_seed(lib, rs.seed_sources(op, same, self.sp), parallel=True)
        seq1 = spec.cuda_seed(lib, rs.seed_sources(op, same, self.sp), parallel=False)
        self.assertEqual(len(re.findall(r"namespace member\d+ \{", seq1)), 2)    # gru, catn
        self.assertEqual(len(re.findall(r"namespace member\d+ \{", par1)), 3)    # gru lane 0 + lane 1, catn

    def test_all_pattern(self):
        names = rs.find_regions(self.g, "all", self.sp)[0]
        g2, _ = rs.fuse_region(self.g, names, self.sp, "all")
        op = next(o for o in g2["ops"] if rs.is_region(o))
        self.assertEqual(op["outputs"], ["getitem"])


class FusedMemberFallbackTest(unittest.TestCase):
    """A region whose member is a fused op (conv+relu, linear+tanh) with NO cached source for its kind -- a graph-first round
    has produced no kernel winner and the cache has never seen this region op -- seeds from the member's own stage-by-stage
    seed (the non-region fused slot path's rule) instead of raising 'a fused member needs its source in sub_sources'
    (abl3 17018_SqueezeNet, round 1 plan export: region:all and region:win:whole both accepted in round 0)."""

    def setUp(self):
        from modelblaster.pipeline.tests import window_fixtures as wf
        from modelblaster.pipeline.kb_export import DEFAULT_HAND_CU, CudaLibrary, _hand_source
        self.g = wf.rnn_unrolled(8, fused=True)
        self.sp = _specs(self.g)
        names = rs.find_regions(self.g, "all", self.sp)[0]
        self.g2, self.spec = rs.fuse_region(self.g, names, self.sp, "all")
        self.lib = CudaLibrary(_hand_source(DEFAULT_HAND_CU, "fp16"))

    def test_no_source_falls_back_to_the_fused_seed(self):
        src = self.spec.cuda_seed(self.lib, {}, parallel=False)
        self.assertIn("current kernel of fused__linear_f16__tanh_f16", src)

    def test_sub_ops_sources_are_used(self):
        marker = "// MARKER-LINEAR-WINNER"
        from modelblaster.pipeline.kb_export import DEFAULT_HAND_CU, _hand_source
        hand = self.lib.slot_source("kernel_linear_f16").replace("kernel_linear_f16", "kernel_linear_f16", 1)
        src = self.spec.cuda_seed(self.lib, {"linear_f16": marker + "\n" + hand}, parallel=False)
        self.assertIn(marker, src)

    def test_explicit_fused_source_still_wins(self):
        fk = "fused__linear_f16__tanh_f16"
        own = self.sp[fk].cuda_seed(self.lib, sub_sources={}) + "\n// MARKER-OWN-FUSED\n"
        src = self.spec.cuda_seed(self.lib, {fk: own}, parallel=False)
        self.assertIn("MARKER-OWN-FUSED", src)


class Region43Test(unittest.TestCase):
    def test_attention_block(self):
        g = json.loads((HERE / "data" / "mingpt43_final_graph.json").read_text())
        sp = _specs(g)
        names = rs.find_regions(g, "attention_block", sp)[0]
        self.assertEqual(names[0], "c_attn")
        self.assertEqual(names[-1], "c_proj")
        g2, spec = rs.fuse_region(g, names, sp, "attention_block")
        self.assertEqual(len(g2["ops"]), 1)
        from modelblaster.pipeline.op_classes import op_class
        self.assertEqual(op_class(g2["ops"][0]["op"]), "attention")
        self.assertEqual([r for _, r, _ in spec.params].count("out"), 1)
        self.assertEqual(rs.find_regions(g, "rnn_stack", sp), [])


class RefusalTest(unittest.TestCase):
    def g(self):
        return {"output": {"tensors": ["z"]}, "tensors": {}, "ops": [
            {"name": "a", "op": "relu_f16", "inputs": ["x"], "outputs": ["t1"], "shape": {"n": 4}},
            {"name": "b", "op": "relu_f16", "inputs": ["t1"], "outputs": ["t2"], "shape": {"n": 4}},
            {"name": "c", "op": "relu_f16", "inputs": ["t1"], "outputs": ["t3"], "shape": {"n": 4}},
            {"name": "d", "op": "add_f16", "inputs": ["t2", "t3"], "outputs": ["z"], "shape": {"n": 4}}]}

    def test_reader_between_members(self):
        with self.assertRaises(ValueError):
            rs.fuse_region(self.g(), ["a", "b", "d"], dict(_all_specs()))     # c reads t1 before the region ends

    def test_ok_and_too_small(self):
        g2, spec = rs.fuse_region(self.g(), ["a", "b", "c", "d"], dict(_all_specs()))
        self.assertEqual(len(g2["ops"]), 1)
        with self.assertRaises(ValueError):
            rs.fuse_region(self.g(), ["a"], dict(_all_specs()))

    def test_switch(self):
        self.assertEqual(rs.enabled("off"), set())
        self.assertEqual(rs.enabled("rnn_stack,all"), {"rnn_stack", "all"})
        with self.assertRaises(ValueError):
            rs.enabled("everything")

    def test_hint_applier(self):
        from modelblaster.pipeline import graph_hints as gh
        g = self.g()
        h = {"id": "region:x", "kind": "fuse_region", "members": ["a", "b", "c", "d"], "status": "accepted",
             "target": "cuda"}
        g2, added, prov = gh.apply_hints(g, [h], dict(_all_specs()))
        self.assertEqual(len(g2["ops"]), 1)
        self.assertEqual(prov["applied"][0]["kind"], "fuse_region")
        self.assertEqual(prov["renamed"]["a"], g2["ops"][0]["name"])
        self.assertIn(g2["ops"][0]["op"], added)


class CapsTest(unittest.TestCase):
    def cfg(self, **kw):
        from modelblaster.pipeline.capabilities import DEFAULTS
        return SimpleNamespace(**{**DEFAULTS, **kw})

    def test_defaults(self):
        from modelblaster.pipeline.capabilities import Caps
        c = Caps(self.cfg())
        self.assertEqual(c.round_key(), {})
        self.assertEqual(c.decide_kinds(), ())
        self.assertEqual(c.propose({}, [], {"hints": []}, {}), [])
        env = dict(os.environ)
        try:
            for k in ("MB_PRUNE_DEAD", "MB_RECURRENT_CLASS", "MB_REGION_SLOTS", "KERNELBLASTER_MULTITURN"):
                os.environ.pop(k, None)
            c.apply_env()
            self.assertNotIn("MB_PRUNE_DEAD", os.environ)
            self.assertNotIn("KERNELBLASTER_MULTITURN", os.environ)
            Caps(self.cfg(prune_dead="off", recurrent_class="on", multiturn="on")).apply_env()
            self.assertEqual(os.environ["MB_PRUNE_DEAD"], "0")
            self.assertEqual(os.environ["MB_RECURRENT_CLASS"], "1")
            self.assertEqual(os.environ["KERNELBLASTER_MULTITURN_BUDGET"], "3,4,120")
        finally:
            os.environ.clear()
            os.environ.update(env)
        self.assertEqual(Caps(self.cfg(region_slots="rnn_stack")).round_key(), {"caps_region_slots": "rnn_stack"})
        self.assertEqual(Caps(self.cfg(view_fusion="on", region_slots="all")).decide_kinds(), ("fuse_region", "fuse_view"))
        with self.assertRaises(ValueError):
            Caps(self.cfg(multiturn_budget="3,4"))

    @unittest.skipUnless(G41.is_file(), "no local 41 IR")
    def test_propose_and_context(self):
        from modelblaster.pipeline.capabilities import Caps, model_context
        g = json.loads(G41.read_text())
        from modelblaster.pipeline.graph_hints import unique_names
        ops = [{"name": o["name"], "op": o["op"], "us": 500.0 if "seq" in o["op"] else 3.0}
               for o in unique_names(g)["ops"] if o["op"] != "view"]
        prof = {"ir": "kb_problem", "ops": ops, "e2e_us": sum(o["us"] for o in ops)}
        doc = {"hints": []}
        new = Caps(self.cfg(region_slots="rnn_stack")).propose({"kb_problem": g}, [prof], doc, dict(_all_specs()))
        self.assertEqual([h["id"] for h in new], ["region:rnn_stack"])
        self.assertGreater(new[0]["source"][0]["share"], 0.9)
        ctx = model_context({"ir_name": "kb_problem", "partition": {"op_names": ["gru_l0"]}}, [prof])
        self.assertEqual(ctx["n_ops"], len(ops))
        self.assertTrue(any(o["in_this_problem"] for o in ctx["top_ops"]))


def _have_torch():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _have_gpu():
    return _have_torch() and shutil.which("nvcc") is not None and shutil.which("nvidia-smi") is not None


GRU = '''import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(8, 16, 2, True, False, dropout=0, bidirectional=True)
    def forward(self, x, h0):
        output, h_n = self.gru(x, h0)
        return output
def get_inputs():
    return [torch.rand(6, 2, 8), torch.rand(4, 2, 16)]
def get_init_inputs():
    return []
'''


@unittest.skipUnless(_have_torch(), "needs torch")
class ExportTest(unittest.TestCase):
    def test_region_export(self):
        from modelblaster.pipeline.kb_export import export_problem
        from modelblaster.pipeline.kb_proposals import write_variant_ir
        from modelblaster.pipeline.graph_hints import unique_names
        root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "m.py").write_text(GRU)
            cp = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file",
                                 str(d / "m.py"), "--bench-target-mb", "0", "--out-dir", str(d / "ir"), "--quant", "fp16"],
                                cwd=root, capture_output=True, text=True, timeout=600,
                                env={**os.environ, "PYTHONPATH": os.pathsep.join(
                                    [str(root), os.environ.get("PYTHONPATH", "")])})
            if cp.returncode:
                self.skipTest(cp.stderr[-300:])
            g = unique_names(json.loads((d / "ir" / "graph.json").read_text()))
            sp = _specs(g)
            names = rs.find_regions(g, "rnn_stack", sp)[0]
            g2, spec = rs.fuse_region(g, names, sp, "rnn_stack")
            vir = write_variant_ir(d / "ir", g2, d / "rir")
            p = export_problem(vir, d / "prob")
            man = json.loads((p / "manifest.json").read_text())
            reg = [s for s in man["slots"] if s["op"].startswith("region__")]
            self.assertEqual(len(reg), 1)
            self.assertTrue(reg[0]["seed_source"].startswith("region-seed:0/"))
            self.assertIn("concurrency", man)                         # watchdog + mb_sm_budget for region problems
            self.assertIn("MB_CONC_DEADLOCK", (p / "driver.cpp").read_text())
            off = export_problem(vir, d / "prob_off", concurrency="off")
            self.assertNotIn("concurrency", json.loads((off / "manifest.json").read_text()))
            if _have_gpu():
                from modelblaster.pipeline.kb_export import build_and_run
                arch = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                                      capture_output=True, text=True).stdout.split()[0].replace(".", "")
                res = build_and_run(p, arch=arch)
                self.assertTrue(res["passed"], (res.get("log") or "")[-2000:])


if __name__ == "__main__":
    unittest.main()


G31 = FARM / "31_VisionAttention" / "work" / "final" / "31_VisionAttention" / "ir" / "graph.json"


class ViewFusionTest(unittest.TestCase):
    """view_fusion (A2 item 6a): producer -> pure-reshape view -> flat elementwise."""

    def g(self):
        return {"output": {"tensors": ["z"]}, "tensors": {"x": {"shape": [4, 8]}, "y": {"shape": [4, 8]},
                                                         "yv": {"shape": [32]}, "z": {"shape": [32]},
                                                         "yv2": {"shape": [2, 16]}},
                "ops": [{"name": "mm", "op": "relu_f16", "inputs": ["x"], "outputs": ["y"], "shape": {"n": 32}},
                        {"name": "v", "op": "view", "inputs": ["y"], "outputs": ["yv"], "shape": {"n": 32}},
                        {"name": "s", "op": "relu_f16", "inputs": ["yv"], "outputs": ["z"], "shape": {"n": 32}}]}

    def test_transparent_and_elide(self):
        from modelblaster.pipeline import view_fusion as vf
        g = self.g()
        self.assertEqual(vf.transparent_views(g), ["v"])
        g2, done = vf.elide(g)
        self.assertEqual(done, ["v"])
        self.assertEqual([o["name"] for o in g2["ops"]], ["mm", "s"])
        self.assertEqual(g2["ops"][1]["inputs"], ["y"])
        g["ops"][1]["offset"] = 8                                      # a slice is not a reshape
        self.assertEqual(vf.transparent_views(g), [])
        g = self.g()
        g["ops"][2]["shape"] = {"M": 4, "K": 8}                      # a non-flat consumer (e.g. softmax rows)
        self.assertEqual(vf.transparent_views(g), [])
        with self.assertRaises(ValueError):
            vf.elide(g, ["v"])
        self.assertFalse(vf.enabled())
        self.assertTrue(vf.enabled("on"))

    @unittest.skipUnless(G31.is_file(), "no local 31 IR")
    def test_31_scale_behind_view(self):
        from modelblaster.pipeline import graph_hints as gh, view_fusion as vf
        g = json.loads(G31.read_text())
        sp = dict(_all_specs())
        hs = vf.propose(g, {"attn#12": 12652.4, "attn#10": 6401.4}, sp, min_gain_pct=0.5, ir="kb_problem",
                        total_us=37993.5)
        # the pair exists only through the view, but its scale stage has n = 2^31: no fused op can take that as a
        # C int (export refuses it), so it is not proposed
        self.assertEqual(hs, [])
        g = json.loads(G31.read_text())
        for o in g["ops"]:                                 # the same graph at a size the ABI takes
            if o["name"] in ("attn#10", "attn#12"):
                o["shape"] = {k: (v // 2 if k == "n" else v) for k, v in o["shape"].items()}
        hs = vf.propose(g, {"attn#12": 12652.4, "attn#10": 6401.4}, sp, min_gain_pct=0.5, ir="kb_problem",
                        total_us=37993.5)
        self.assertEqual(len(hs), 1)
        h = hs[0]
        self.assertEqual((h["kind"], h["fused_kind"], h["chains"], h["views"]),
                         ("fuse_view", "fused__bmm_f16__ewise_scalar_f16", [["attn#10", "attn#12"]], ["attn#11"]))
        self.assertGreater(h["source"][0]["est_gain_pct"], 30)
        # the plain proposer cannot see it
        plain = gh.propose_from_profile(g, {"attn#12": 12652.4}, sp, min_gain_pct=0.5)
        self.assertFalse(any(x["fused_kind"] == h["fused_kind"] for x in plain))
        g2, added, prov = gh.apply_hints(g, [{**h, "status": "accepted"}], sp)
        self.assertEqual(prov["applied"][0]["kind"], "fuse_view")  # noqa
        self.assertNotIn("attn#11", {o["name"] for o in g2["ops"]})
        fused = next(o for o in g2["ops"] if o["name"] == "attn#10+attn#12")
        self.assertEqual(fused["outputs"], ["attn__g13"])
        self.assertIn(h["fused_kind"], added)
        # applying it again on the rewritten graph is a no-op (nothing left to elide or fuse)
        g3, _, prov3 = gh.apply_hints(g2, [{**h, "status": "accepted"}], sp)
        self.assertEqual(len(g3["ops"]), len(g2["ops"]))
