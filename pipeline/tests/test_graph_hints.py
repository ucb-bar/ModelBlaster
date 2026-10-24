"""Tests for the unified graph-hint layer and fused-op chains.

Pure python (the verify test walks a tiny IR with torch on the CPU).
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline import graph_hints as gh  # noqa: E402
from modelblaster.pipeline.kb_proposals import (bind_fused, find_chains, fuse_chain, fuse_pair,  # noqa: E402
                                                make_chain_spec, parse_kind, spec_from_kind)
from modelblaster.pipeline.reference_kernels import KERNEL_SPECS as K  # noqa: E402


def conv_block(n_blocks: int = 1, reuse_relu: bool = True) -> dict:
    """conv -> bn -> relu (x n), every op with a dispatch id; a reused ReLU
    module gives repeated op names, as extract_graph does."""
    t = {"x": {"shape": [1, 3, 8, 8]}}
    ops, cur, did = [], "x", 0
    for k in range(n_blocks):
        ic = 3 if k == 0 else 4
        for kind, name in (("conv2d", f"conv{k}"), ("batchnorm2d", f"bn{k}"), ("relu", "relu" if reuse_relu else f"relu{k}")):
            out = f"{name}_{k}_out"
            t[out] = {"shape": [1, 4, 8, 8]}
            o = {"name": name, "op": kind, "inputs": [cur], "outputs": [out], "dispatch_id": did,
                 "depends_on": [did - 1] if did else []}
            if kind == "conv2d":
                o.update(weight=f"conv{k}.weight", bias=None,
                         shape={"N": 1, "IC": ic, "IH": 8, "IW": 8, "OC": 4, "OH": 8, "OW": 8, "KH": 3, "KW": 3,
                                "SH": 1, "SW": 1, "PH": 1, "PW": 1, "DH": 1, "DW": 1})
            elif kind == "batchnorm2d":
                o.update(weight=f"bn{k}.scale", bias=f"bn{k}.bias", shape={"N": 1, "C": 4, "H": 8, "W": 8})
            else:
                o["shape"] = {"n": 256}
            ops.append(o)
            cur, did = out, did + 1
    return {"name": "blk", "input": {"tensor": "x"}, "output": {"tensor": cur}, "tensors": t, "ops": ops,
            "dispatches": list(range(did))}


def ewise_graph() -> dict:
    return {"name": "t", "input": {"tensor": "x"}, "output": {"tensor": "z"},
            "tensors": {k: {"shape": [8]} for k in "xyz"},
            "ops": [{"name": "s", "op": "sigmoid", "inputs": ["x"], "outputs": ["y"], "shape": {"n": 8},
                     "dispatch_id": 0, "depends_on": []},
                    {"name": "r", "op": "relu", "inputs": ["y"], "outputs": ["z"], "shape": {"n": 8},
                     "dispatch_id": 1, "depends_on": [0]}]}


class ChainTest(unittest.TestCase):
    def test_kind_names_round_trip(self):
        # two-stage names are unchanged from Phase 1; the param appears only
        # for a stage with two activation inputs
        self.assertEqual(parse_kind("fused__sigmoid__relu", K), (["sigmoid", "relu"], ["input"]))
        self.assertEqual(parse_kind("fused__batchnorm2d__add__b", K), (["batchnorm2d", "add"], ["b"]))
        k3 = "fused__conv2d__batchnorm2d__add__a__relu"
        self.assertEqual(parse_kind(k3, K), (["conv2d", "batchnorm2d", "add", "relu"], ["input", "a", "input"]))
        self.assertEqual(spec_from_kind(k3, K).op, k3)
        with self.assertRaises(ValueError):
            parse_kind("fused__batchnorm2d__add", K)          # add needs its param

    def test_three_stage_signature_and_bind(self):
        g = conv_block()
        g2, spec = fuse_chain(g, ["conv0", "bn0", "relu"], K)
        self.assertEqual(spec.op, "fused__conv2d__batchnorm2d__relu")
        self.assertEqual([o["op"] for o in g2["ops"]], [spec.op])
        f = g2["ops"][0]
        self.assertEqual(f["name"], "conv0+bn0+relu")
        self.assertEqual(f["internal"], ["conv0_0_out", "bn0_0_out"])
        self.assertEqual(set(f["shape"]), {"a", "b", "c"})
        sig = spec.signature
        self.assertTrue(sig.startswith("void kernel_fused__conv2d__batchnorm2d__relu(const float* a_input"))
        self.assertIn("float* a_output", sig)
        self.assertIn("const float* b_scale", sig)
        self.assertNotIn("b_input", sig)
        self.assertNotIn("c_input", sig)
        self.assertTrue(sig.endswith("float* c_output, int c_n);"))
        args = bind_fused(f, K)
        kinds = [(a.kind, a.value) for a in args if a.kind in ("in", "out", "scratch")]
        self.assertEqual(kinds, [("in", "x"), ("scratch", "conv0_0_out"), ("scratch", "bn0_0_out"),
                                 ("out", "relu_0_out")])
        from modelblaster.pipeline.kernel_abi import parse_signature
        self.assertEqual(len(args), len(parse_signature(sig)[1]))

    def test_chains_flatten_either_way(self):
        g = conv_block()
        a, _ = fuse_pair(g, "conv0", "bn0", K)
        a, s1 = fuse_pair(a, "conv0+bn0", "relu", K)                # fused producer + consumer
        b, _ = fuse_pair(g, "bn0", "relu", K)
        b, s2 = fuse_pair(b, "conv0", "bn0+relu", K)                # producer + fused consumer
        self.assertEqual(s1.op, s2.op)
        self.assertEqual([s["name"] for s in a["ops"][0]["sub_ops"]], ["conv0", "bn0", "relu"])
        self.assertEqual(a["ops"][0]["internal"], b["ops"][0]["internal"])

    def test_not_a_chain(self):
        g = ewise_graph()
        g["ops"].append({"name": "r2", "op": "relu", "inputs": ["y"], "outputs": ["w"], "shape": {"n": 8}})
        with self.assertRaises(ValueError):                         # y has two readers
            fuse_pair(g, "s", "r", K)
        g = ewise_graph()
        g["ops"][1] = {"name": "m", "op": "add", "inputs": ["x", "y"], "outputs": ["z"], "shape": {"n": 8}}
        g2, spec = fuse_pair(g, "s", "m", K)
        self.assertEqual(spec.op, "fused__sigmoid__add__b")
        self.assertEqual(g2["ops"][0]["inputs"], ["x"])              # x read by both stages: one input

    def test_find_chains(self):
        u = gh.unique_names(conv_block(2))
        self.assertEqual([o["name"] for o in u["ops"]][-1], "relu#1")
        self.assertEqual(find_chains(u, "fused__conv2d__batchnorm2d__relu", K),
                         [["conv0", "bn0", "relu"], ["conv1", "bn1", "relu#1"]])
        u["output"] = {"tensors": ["bn1_1_out", "relu_1_out"]}          # bn1 output escapes
        self.assertEqual(find_chains(u, "fused__conv2d__batchnorm2d__relu", K), [["conv0", "bn0", "relu"]])

    def test_seed_composes_stage_sources(self):
        spec = make_chain_spec(["sigmoid", "relu"], ["input"], K)

        class Lib:                                                   # hand library stub
            items = {"kernel_sigmoid": type("I", (), {"text": "void kernel_sigmoid(const float*, float*, int) {}"})()}

            def closure(self, e):
                return ["kernel_sigmoid"] if e == "kernel_sigmoid" else []

        relu_best = "#include <cuda_fp16.h>\n#define TILE 4\n__global__ void k() {}\nvoid kernel_relu(const float *input, float *output, int n) {}\n"
        src = spec.cuda_seed(Lib(), sub_sources={"relu": relu_best})
        self.assertTrue(src.startswith("#include <cuda_fp16.h>"))
        self.assertIn("namespace stage_relu {", src)
        self.assertIn("#undef TILE", src)
        self.assertIn("    kernel_sigmoid(a_input, a_output, a_n);", src)
        self.assertIn("    stage_relu::kernel_relu(a_output, b_output, b_n);", src)


class ApplyTest(unittest.TestCase):
    def test_longest_first_provenance_and_ids(self):
        g = conv_block(2)
        hints = [gh.fuse_hint("fused__conv2d__batchnorm2d", None, {"by": "user"}, "accepted"),
                 gh.fuse_hint("fused__conv2d__batchnorm2d__relu", None, {"by": "user"}, "accepted"),
                 gh.fuse_hint("fused__sigmoid__relu", None, {"by": "user"}, "proposed")]
        g2, added, prov = gh.apply_hints(g, hints, K)
        self.assertEqual([o["op"] for o in g2["ops"]], ["fused__conv2d__batchnorm2d__relu"] * 2)
        self.assertEqual(set(added), {"fused__conv2d__batchnorm2d__relu"})
        self.assertEqual([o["hint"] for o in g2["ops"]], ["fuse:fused__conv2d__batchnorm2d__relu"] * 2)
        self.assertEqual([(o["dispatch_id"], o["depends_on"]) for o in g2["ops"]], [(0, []), (1, [0])])
        self.assertEqual(g2["id_remap"], {"0": [0], "1": [0], "2": [0], "3": [1], "4": [1], "5": [1]})
        self.assertEqual(g2["dispatches"], [0, 1])
        self.assertEqual(prov["renamed"]["relu#1"], "conv1+bn1+relu#1")
        self.assertEqual(g2["graph_hints"]["rounds"][-1]["applied"], prov["applied"])
        self.assertEqual(g, conv_block(2))                           # input untouched

    def test_scoped_chain_and_unfuse(self):
        g = conv_block(2)
        h = gh.fuse_hint("fused__conv2d__batchnorm2d", [["conv1", "bn1"]], {"by": "kb"})
        g2, _, prov = gh.apply_hints(g, [h], K, statuses=None)
        self.assertEqual([o["name"] for o in g2["ops"]], ["conv0", "bn0", "relu", "conv1+bn1", "relu#1"])
        # extend the fused op with its consumer by original names
        h3 = gh.fuse_hint("fused__conv2d__batchnorm2d__relu", [["conv1", "bn1", "relu#1"]], {"by": "mb-cost"})
        g3, _, _ = gh.apply_hints(g2, [h3], K, statuses=None)
        self.assertEqual(g3["ops"][-1]["name"], "conv1+bn1+relu#1")
        u = {"id": "unfuse:x", "kind": "unfuse", "target": "cuda", "fused_kind": h3["fused_kind"]}
        g4, _, p4 = gh.apply_hints(g3, [u], K, statuses=None)
        self.assertEqual([o["name"] for o in g4["ops"]], [o["name"] for o in gh.unique_names(conv_block(2))["ops"]])
        self.assertEqual([o["depends_on"] for o in g4["ops"]], [[], [0], [1], [2], [3], [4]])

    def test_split_slot_annotations(self):
        from modelblaster.pipeline.kb_export import slot_id_of
        g = conv_block(2)
        g["ops"][3]["shape"] = dict(g["ops"][3]["shape"], IC=64, OC=64)    # another shape bucket
        h = {"id": "split:conv2d", "kind": "split_slot", "target": "cuda", "slot": "conv2d", "by": "shape_bucket",
             "status": "accepted"}
        g2, _, prov = gh.apply_hints(g, [h], K)
        convs = [o for o in g2["ops"] if o["op"] == "conv2d"]
        self.assertEqual([slot_id_of(o, K) for o in convs], ["conv2d__v0", "conv2d__v1"])
        self.assertEqual(prov["applied"][0]["variants"], {"v0": ["conv0"], "v1": ["conv1"]})
        g3 = conv_block(2)
        g3["ops"][0]["shape"]["IC"] = 4
        g3, _, prov = gh.apply_hints(g3, [h], K)                         # one bucket: nothing to split
        self.assertFalse(prov["applied"])
        self.assertIn("1 shape group", prov["skipped"][0]["why"])

    def test_xpurt_hints(self):
        g = conv_block()
        fh = {"contract": "modelblaster.fusion_hints/v1", "networks": [{"network": "blk", "fuse_groups": [[0, 1, 2]]}]}
        sh = {"contract": "modelblaster.shard_hints/v1", "networks": [{"network": "blk", "shard_ops": [{"op": 0, "n_shards": 2}]}]}
        hints = gh.from_xpurt(fh) + gh.from_xpurt(sh)
        self.assertEqual([h["target"] for h in hints], ["riscv", "riscv"])
        # on CUDA the dispatch group becomes the chain fusion; the shard is skipped with the reason
        g2, added, prov = gh.apply_hints(g, hints, K)
        self.assertEqual([o["op"] for o in g2["ops"]], ["fused__conv2d__batchnorm2d__relu"])
        self.assertTrue(any("not applicable on cuda" in s["why"] for s in prov["skipped"]))
        # on RISC-V the existing appliers run unchanged
        g3, _, p3 = gh.apply_hints(g, gh.from_xpurt(fh), target="riscv")
        self.assertEqual(g3["ops"][0]["op"], "__fused__conv2d__batchnorm2d__relu")
        self.assertEqual(p3["applied"][0]["kind"], "fuse")


class DocTest(unittest.TestCase):
    def test_registry_import_decisions_and_merge(self):
        with tempfile.TemporaryDirectory() as d:
            reg = Path(d) / "r.json"
            reg.write_text(json.dumps({"format": "mb-fusion-registry/1", "fusions": {
                "fused__sigmoid__relu": {"kind": "fused__sigmoid__relu", "status": "accepted", "evidence": []}}}))
            doc = gh.load(reg)
            self.assertEqual(doc["hints"][0]["id"], "fuse:fused__sigmoid__relu")
            h = gh.fuse_hint("fused__conv2d__batchnorm2d", [["c", "b"]], {"by": "kb"})
            self.assertEqual(gh.merge(doc, [h]), [h])
            gh.merge(doc, [gh.fuse_hint("fused__conv2d__batchnorm2d", [["c2", "b2"]], {"by": "mb-cost"})])
            cur = gh.get(doc, h["id"])
            self.assertEqual(cur["chains"], [["c", "b"], ["c2", "b2"]])
            self.assertEqual(len(cur["source"]), 2)
            gh.record_decision(doc, h["id"], {"decision": "rejected"})
            self.assertEqual(cur["status"], "rejected")
            gh.record_decision(doc, h["id"], {"decision": "accepted"})      # tie rejects
            self.assertEqual(cur["status"], "rejected")
            gh.record_decision(doc, h["id"], {"decision": "accepted"})
            self.assertEqual(cur["status"], "accepted")
            self.assertNotIn("chains", cur)                                  # now applies everywhere
            self.assertEqual(cur["tried_chains"], [["c", "b"], ["c2", "b2"]])
            gh.save(doc, Path(d) / "h.json")
            self.assertEqual(gh.load(Path(d) / "h.json")["format"], gh.FORMAT)

    def test_mb_cost_proposals_grow_chains(self):
        g = conv_block(2)
        us = {"conv0": 50.0, "bn0": 10.0, "relu": 8.0, "conv1": 50.0, "bn1": 10.0, "relu#1": 8.0}
        props = gh.propose_from_profile(g, us, K)
        # greedy by consumer time, each op in one pair: conv->bn wins, bn is then taken
        self.assertEqual([p["id"] for p in props], ["fuse:fused__conv2d__batchnorm2d"])
        self.assertEqual(props[0]["chains"], [["conv0", "bn0"], ["conv1", "bn1"]])
        # after conv->bn is accepted, the fused op's relu consumer is the next proposal
        doc = gh.new_doc()
        gh.record_decision(doc, props[0], {"decision": "accepted"})
        g2, _, _ = gh.apply_hints(g, doc["hints"], K)
        us2 = {"conv0+bn0": 55.0, "relu": 8.0, "conv1+bn1": 55.0, "relu#1": 8.0}
        p2 = gh.propose_from_profile(g2, us2, K, doc=doc)
        self.assertEqual([p["id"] for p in p2], ["fuse:fused__conv2d__batchnorm2d__relu"])
        self.assertEqual(p2[0]["source"][0]["by"], "mb-cost")
        self.assertEqual(gh.propose_from_profile(g2, us2, K, doc=doc, min_gain_pct=20.0), [])

    def test_from_kb_proposals(self):
        g = conv_block(2)
        g2, _, _ = gh.apply_hints(g, [gh.fuse_hint("fused__conv2d__batchnorm2d", None, {}, "accepted")], K)
        doc = {"format": "kb-proposal/1", "problem": "p", "proposals": [
            {"kind": "fuse", "pattern": ["fused__conv2d__batchnorm2d", "relu"],
             "pairs": [["conv0+bn0", "relu", "bn0_0_out"], ["conv1+bn1", "relu#1", "bn1_1_out"], ["x", "y", "t"]],
             "evidence": {"est_gain_pct": 5.0}},
            {"kind": "split", "slot": "conv2d", "groups": [{"bucket": "a", "ops": ["conv0"]}, {"bucket": "b", "ops": ["conv1"]}],
             "evidence": {"est_gain_pct": 3.0}}]}
        hs = gh.from_kb_proposals(doc, graph=g2)
        self.assertEqual([h["id"] for h in hs], ["fuse:fused__conv2d__batchnorm2d__relu", "split:conv2d"])
        self.assertEqual(hs[0]["chains"], [["conv0", "bn0", "relu"], ["conv1", "bn1", "relu#1"]])
        self.assertEqual(hs[1]["kind"], "split_slot")

    def test_shape_bucket_matches_kb(self):
        self.assertEqual(gh.shape_bucket("conv", {"N": 1, "IC": 64, "OC": 64, "OH": 56, "OW": 56, "KH": 3, "KW": 3}),
                         "conv:N1,IC64,OC64,OH16,OW16,KH1,KW1")
        self.assertEqual(gh.shape_bucket("elementwise", {"n": 3211264}), "elementwise:n1048576")
        self.assertEqual(gh.shape_bucket("conv", {"a": {"N": 1, "IC": 3}, "b": {"n": 5}}), "conv:N1,IC1")


class VerifyTest(unittest.TestCase):
    def test_walk_vs_golden(self):
        g = ewise_graph()
        x = np.linspace(-2, 2, 8).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "graph.json").write_text(json.dumps(g))
            np.savez(d / "weights.npz", dummy=np.zeros(1, np.float32))
            np.savez(d / "io.npz", input=x, output=np.maximum(1 / (1 + np.exp(-x)), 0))
            g2, _, _ = gh.apply_hints(g, [gh.fuse_hint("fused__sigmoid__relu", None, {}, "accepted")], K)
            r = gh.verify_rewrite(d, g2)
            self.assertTrue(r["ok"], r)
            self.assertEqual(r["vs_original_walk"], 0.0)
            bad = copy.deepcopy(g2)
            bad["ops"][0]["sub_ops"].reverse()                     # relu then sigmoid: wrong
            bad["ops"][0]["sub_ops"][0]["inputs"] = ["x"]
            bad["ops"][0]["sub_ops"][1]["inputs"] = ["z"]
            bad["ops"][0]["sub_ops"][1]["outputs"] = ["y"]
            bad["output"] = {"tensor": "y"}
            self.assertFalse(gh.verify_rewrite(d, bad)["ok"])


if __name__ == "__main__":
    unittest.main()
