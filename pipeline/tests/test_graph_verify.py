"""The verification gate (pipeline/graph_verify.py) and the memory-safe walker (walk_torch.walk_free).

CPU only. The fixture is a tiny fire-module IR (rewrite_fixtures.py) in fp32 and fp16, its known correct rewrite (merge the two
expand convolutions into one with a derived zero-padded weight stack) and known wrong ones.
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
import weakref
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline import graph_verify as gv  # noqa: E402
from modelblaster.pipeline.kb_export import _harness  # noqa: E402
from modelblaster.pipeline.tests import rewrite_fixtures as F  # noqa: E402


def _edit(src: Path, dst: Path, fn, *, weights=None) -> Path:
    """A hand-edited copy of an IR dir: ``fn(graph)`` mutates the graph; ``weights(dict)`` the weights."""
    dst.mkdir(parents=True, exist_ok=True)
    g = json.loads((src / "graph.json").read_text())
    fn(g)
    (dst / "graph.json").write_text(json.dumps(g, indent=1))
    z = np.load(src / "weights.npz")
    w = {k: z[k] for k in z.files}
    if weights:
        weights(w)
    np.savez(dst / "weights.npz", **w)
    shutil.copyfile(src / "io.npz", dst / "io.npz")
    return dst


class GateAcceptRejectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="mb_gv_"))
        cls.ir = {q: F.fire_ir(cls.tmp / f"o_{q}", q) for q in ("fp32", "fp16")}
        cls.tiny = {q: F.fire_ir(cls.tmp / f"t_{q}", q, tiny_head=True) for q in ("fp32", "fp16")}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_accepts_the_correct_rewrite(self):
        for q in ("fp32", "fp16"):
            e, tmap = F.merge_expand(self.ir[q], self.tmp / f"e_{q}")
            v = gv.verify(self.ir[q], e, tensor_map=tmap)
            self.assertTrue(v.ok, (q, v.reasons))
            self.assertEqual({k: c["ok"] for k, c in v.checks.items()},
                             {k: True for k in v.checks}, q)
            for name in ("io", "names", "tensors", "dag", "weights", "kinds", "dtype", "shape", "slots", "patch_replay",
                         "outputs", "sampled", "differential", "exact64", "golden"):
                self.assertIn(name, v.checks, (q, name))
            self.assertTrue(v.checks["weights"]["derived_exact"])
            self.assertGreater(v.checks["sampled"]["exits_compared"], 0)
            self.assertEqual(v.info["ops_before"], 9)
            self.assertEqual(v.info["ops_after"], 6)                      # 9 - 5 + 2
            self.assertEqual(sorted(v.info["derived_weights"]), ["d.fire.b", "d.fire.e1_3x3", "d.fire.w"])
            self.assertEqual(v.reasons, [])
            self.assertEqual(json.loads(json.dumps(v.to_dict()))["ok"], True)      # a structured, serialisable verdict
            # also without the declared tensor_map: the exits (cat_out) are compared by name
            self.assertTrue(gv.verify(self.ir[q], e).ok)

    def test_rejects_wrong_rewrites(self):
        for q in ("fp32", "fp16"):
            for wrong in ("order", "pad"):
                e, tmap = F.merge_expand(self.ir[q], self.tmp / f"w_{q}_{wrong}", wrong=wrong)
                v = gv.verify(self.ir[q], e, tensor_map=tmap)
                self.assertFalse(v.ok, (q, wrong))
                self.assertIn("output_mismatch", v.codes)
                self.assertIn("intermediate_mismatch", v.codes)
                self.assertGreater(v.max_errors["outputs"]["abs"], v.max_errors["outputs"]["bound"])

    def test_output_only_check_passes_a_wrong_rewrite_the_sampled_check_rejects(self):
        """The model's output is ~100 whatever the input (head weights 1e-7, bias ~100): the wrong rewrites change it by ~1e-6,
        far inside the output tolerance. The sampled intermediates (cat_out, the edit's exit) catch them."""
        for q in ("fp32", "fp16"):
            o = self.tiny[q]
            good, tmap = F.merge_expand(o, self.tmp / f"tg_{q}")
            v = gv.verify(o, good, tensor_map=tmap)
            self.assertTrue(v.ok, v.reasons)
            self.assertLess(v.info["output_input_sensitivity"], 1e-3)
            self.assertTrue(v.checks["sampled"]["insensitive"])
            self.assertIn("input_insensitive_output", [w["code"] for w in v.warnings])
            for wrong in ("order", "pad"):
                bad, tmap = F.merge_expand(o, self.tmp / f"tw_{q}_{wrong}", wrong=wrong)
                v = gv.verify(o, bad, tensor_map=tmap)
                self.assertTrue(v.checks["outputs"]["ok"], "an output-only check passes it")
                self.assertTrue(v.checks["differential"]["ok"])      # (the noise floor of a ~100 output hides it too)
                self.assertFalse(v.checks["sampled"]["ok"], (q, wrong))
                self.assertEqual(v.codes, ["intermediate_mismatch"])
                self.assertFalse(v.ok)
                self.assertIn("input-insensitive", v.reasons[0]["msg"])
                self.assertEqual(gv.verify(o, bad).codes, ["intermediate_mismatch"])   # by name, without the tensor_map

    def test_input_insensitive_output_with_nothing_to_compare_is_unverifiable(self):
        o = self.tiny["fp32"]
        bad, _ = F.merge_expand(o, self.tmp / "u_bad", wrong="order")

        def rename(g):               # everything the edit touches gets new names: no tensor both graphs define depends on it
            for old in ("cat_out", "pooled"):
                for op in g["ops"]:
                    op["outputs"] = [old + "_new" if t == old else t for t in op["outputs"]]
                    op["inputs"] = [old + "_new" if t == old else t for t in op["inputs"]]
                g["tensors"][old + "_new"] = g["tensors"].pop(old)
        e = _edit(bad, self.tmp / "u_bad2", rename)
        shutil.copyfile(bad / "weights.npz", e / "weights.npz")
        v = gv.verify(o, e)
        self.assertFalse(v.ok)
        self.assertIn("unverifiable", v.codes)
        self.assertEqual(v.checks["outputs"]["ok"], True)

    def test_golden_and_shape_variants(self):
        o = self.ir["fp32"]
        good, tmap = F.merge_expand(o, self.tmp / "sv_good")
        o2 = F.fire_ir(self.tmp / "o2", "fp32", seed=3, hw=(8, 8))
        g2, _ = F.merge_expand(o2, self.tmp / "sv_g2")
        b2, _ = F.merge_expand(o2, self.tmp / "sv_b2", wrong="pad")
        v = gv.verify(o, good, tensor_map=tmap, shape_variants=[(o2, g2)])
        self.assertTrue(v.ok, v.reasons)
        self.assertIn("outputs[shape_variant 0]", v.checks)
        v = gv.verify(o, good, tensor_map=tmap, shape_variants=[(o2, b2)])
        self.assertFalse(v.ok)                                    # right at the problem's shape, wrong at another
        self.assertTrue(all(r.get("where", "").startswith("[shape_variant 0]") for r in v.reasons))
        self.assertTrue(v.checks["outputs"]["ok"])

    def test_sol_hook(self):
        import torch
        o = self.ir["fp32"]
        good, tmap = F.merge_expand(o, self.tmp / "sol_good")
        mbir, walk_torch = _harness()
        with torch.no_grad():
            ref, _ = walk_torch.walk(mbir.load_ir(o), dev="cpu")
        v = gv.verify(o, good, tensor_map=tmap, sol={"refs": lambda: [ref.double()], "tolerance": {"max_atol": 1e-2, "max_rtol": 1e-2}})
        self.assertTrue(v.ok, v.reasons)
        self.assertTrue(v.checks["sol"]["ok"])
        self.assertEqual(v.checks["sol"]["edited"]["matched_ratio"], 1.0)
        # SOL's rule, unit level
        t = {"max_atol": 1e-2, "max_rtol": 1e-2, "required_matched_ratio": 0.9}
        r = np.linspace(1, 2, 100)
        self.assertTrue(gv._sol_check(r, r, t)["ok"])
        off = r.copy()
        off[:20] += 1.0                                            # 80% match < 90%
        self.assertFalse(gv._sol_check(off, r, t)["ok"])
        off = r.copy()
        off[:5] += 1.0                                             # 95% match
        self.assertTrue(gv._sol_check(off, r, t)["ok"])
        self.assertFalse(gv._sol_check(np.where(np.arange(100) == 3, np.nan, r), r, t)["ok"])         # any NaN fails
        self.assertFalse(gv._sol_check(np.zeros(100), r, t)["ok"])                                    # all-zero vs non-zero
        ninf = np.array([-np.inf, 1.0])
        self.assertFalse(gv._sol_check(ninf, ninf, {})["ok"])
        self.assertTrue(gv._sol_check(ninf, ninf, {"allow_negative_inf": True, "required_matched_ratio": 1.0})["ok"])
        self.assertFalse(gv._sol_check(r + 0.5, r, {**t, "max_error_cap": 0.1, "required_matched_ratio": 0.0})["ok"])


class StructuralRejectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="mb_gvs_"))
        cls.o = F.fire_ir(cls.tmp / "o", "fp16")
        cls.good, _ = F.merge_expand(cls.o, cls.tmp / "good")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _v(self, name, fn, weights=None, src=None, **kw):
        e = _edit(src or self.good, self.tmp / name, fn, weights=weights)
        return gv.verify(self.o, e, structural_only=True, **kw)

    def test_weights_not_mapped(self):
        # an op reads a weight that is neither original nor derived
        v = self._v("unmapped", lambda g: g["ops"][[o["name"] for o in g["ops"]].index("merged")].update(weight="mystery.w"))
        self.assertIn("weight_not_mapped", v.codes)
        self.assertIn("IR weights not mapped to the Model", v.reasons[0]["msg"])
        self.assertFalse(v.checks["weights"]["ok"])

        # the weights were computed by arbitrary code: the array is in weights.npz but no derivation records where it came from
        def drop(g):
            g.pop("weight_derivations")
        v = self._v("nolineage", drop)
        self.assertIn("weight_not_mapped", v.codes)
        self.assertIn("provenance", v.reasons[0]["msg"])

    def test_tampered_derived_weight_and_modified_base_weight(self):
        def tamper(w):
            w["d.fire.w"] = w["d.fire.w"].copy()
            w["d.fire.w"][0, 0, 0, 0] += np.float16(0.25)
        v = self._v("tamper", lambda g: None, weights=tamper)
        self.assertEqual(v.codes, ["derivation_mismatch"])
        self.assertIn("d.fire.w", v.reasons[0]["msg"])

        def mod(w):
            w["e3.weight"] = (w["e3.weight"] * 2).astype(np.float16)
        v = self._v("modbase", lambda g: None, weights=mod)
        self.assertIn("base_weight_modified", v.codes)

    def test_derivation_errors_are_reasons(self):
        def bad_leaf(g):
            g["weight_derivations"]["d.fire.w"]["expr"]["args"][1] = {"w": "nope.weight"}
        v = self._v("badleaf", bad_leaf)
        self.assertTrue({"derivation_unmapped"} & set(v.codes), v.codes)

        def bad_op(g):
            g["weight_derivations"]["d.fire.w"]["expr"]["op"] = "exec"
        self.assertIn("derivation_unknown_op", self._v("badop", bad_op).codes)

        def bad_dtype(g):                       # an fp16 IR with an f32 derived weight
            g["weight_derivations"]["d.fire.b"]["dtype"] = "f32"
        self.assertIn("derivation_dtype_contract", self._v("baddt", bad_dtype).codes)

    def test_unknown_op_kind_needs_a_kernel_spec(self):
        v = self._v("kind", lambda g: g["ops"][0].update(op="mystery_kernel_f16"))
        self.assertIn("op_kind", v.codes)
        self.assertIn("kernel spec", v.reasons[0]["msg"])
        self.assertFalse(v.checks["kinds"]["ok"])

    def test_dtype_contract(self):
        def fp32_kind(g):                       # a fp32 kernel on f16 tensors, no cast
            for o in g["ops"]:
                if o["name"] == "merged_relu":
                    o["op"] = "relu"
        v = self._v("dt1", fp32_kind)
        self.assertIn("dtype_contract", v.codes)

        def f32_tensor(g):
            g["tensors"]["mc_o"]["dtype"] = "f32"
        v = self._v("dt2", f32_tensor)
        self.assertIn("dtype_contract", v.codes)

    def test_graph_structure(self):
        def dup(g):
            g["ops"][-1]["name"] = g["ops"][-2]["name"]
        self.assertIn("names", self._v("dup", dup).codes)

        def order(g):                           # an op before its producer
            g["ops"].reverse()
        v = self._v("order", order)
        self.assertIn("dag", v.codes)
        self.assertIn("dependency order", " ".join(r["msg"] for r in v.reasons))

        def cyc(g):
            ops = {o["name"]: o for o in g["ops"]}
            ops["merged"]["inputs"] = ["cat_out"]
        self.assertIn("dag", self._v("cyc", cyc).codes)

        def io(g):
            g["output"] = {"tensors": ["pooled"], "tensor": "pooled"}
        self.assertIn("io", self._v("io", io).codes)

        def reshape_keep_name(g):
            g["tensors"]["s"]["shape"] = [1, 3, 36]
        self.assertIn("tensors", self._v("shp", reshape_keep_name).codes)

        def unreg(g):
            g["tensors"].pop("mc_o")
        self.assertIn("tensors", self._v("unreg", unreg).codes)

    def test_shape_guard(self):
        def oc(g):
            for o in g["ops"]:
                if o["name"] == "merged":
                    o["shape"]["OC"] = 4
        v = self._v("sg", oc)
        self.assertIn("shape_guard", v.codes)

    def test_dead_ops_are_a_warning(self):
        def dead(g):
            g["tensors"]["extra"] = {"shape": [1, 5, 6, 6], "dtype": "f16", "quant": None}
            g["ops"].insert(len(g["ops"]) - 2, {"name": "extra_relu", "op": "relu_f16", "inputs": ["cat_out"], "outputs": ["extra"],
                                                "shape": {"n": 180}})
        v = self._v("dead", dead)
        self.assertTrue(v.ok, v.reasons)
        self.assertIn("dead_ops", [w["code"] for w in v.warnings])

    def test_the_gate_never_raises(self):
        e = self.tmp / "broken"
        e.mkdir()
        (e / "graph.json").write_text("{not json")
        v = gv.verify(self.o, e)
        self.assertFalse(v.ok)
        self.assertEqual(v.codes, ["gate_error"])
        v = gv.verify(self.o, self.tmp / "does_not_exist")
        self.assertEqual(v.codes, ["gate_error"])

    def test_rollback_hooks(self):
        e = _edit(self.good, self.tmp / "rb", lambda g: g["ops"][0].update(op="mystery_kernel_f16"))
        seen = []
        v = gv.verify_or_reject(self.o, e, on_reject=lambda verdict: (seen.append(verdict.summary()), shutil.rmtree(e)), structural_only=True)
        self.assertFalse(v.ok)
        self.assertTrue(seen and not e.exists())
        self.assertFalse(gv.commit_if_ok(v, e, self.tmp / "nope"))
        self.assertFalse((self.tmp / "nope").exists())
        ok = gv.verify(self.o, self.good, structural_only=True)
        self.assertTrue(ok.ok)
        self.assertTrue(gv.commit_if_ok(ok, self.good, self.tmp / "committed"))
        self.assertTrue((self.tmp / "committed" / "weights.npz").exists())

    def test_cli(self):
        rc = gv.main([str(self.o), str(self.good), "--seeds", "1", "--variants", "uniform,normal", "--json", str(self.tmp / "v.json")])
        self.assertEqual(rc, 0)
        self.assertTrue(json.loads((self.tmp / "v.json").read_text())["ok"])
        bad, _ = F.merge_expand(self.o, self.tmp / "cli_bad", wrong="order")
        self.assertEqual(gv.main([str(self.o), str(bad), "--seeds", "1", "--variants", "uniform", "--json", str(self.tmp / "v2.json")]), 1)


# ------------------------------------------------------------------------------ the memory-safe walker
class WalkFreeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="mb_wf_"))
        cls.mbir, cls.wt = _harness()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _same(self, ir, **kw):
        import torch
        with torch.no_grad():
            out, env = self.wt.walk(ir, "cpu", **kw)
            names = sorted(t for t in env if t in ir.graph["tensors"] and not ir.graph["tensors"][t].get("const"))
            out2, got = self.wt.walk_free(ir, "cpu", grab=set(names), **kw)
        self.assertTrue(torch.equal(out, out2))
        for t in names:
            if t in got:
                self.assertTrue(torch.equal(env[t], got[t]), t)
        return names, got

    def test_equals_walk_for_every_storage_dtype_and_the_exact_walk(self):
        import torch
        for quant in ("fp32", "fp16"):
            ir = self.mbir.load_ir(F.fire_ir(self.tmp / f"ir_{quant}", quant))
            names, got = self._same(ir)
            self.assertGreaterEqual(len(got), 8)
            self._same(ir, dtype=torch.float64, store_rounding=False)          # the exact walk of kb_export.exact_values
        # bf16: float32 arrays holding bf16-exact values, `_bf16` kinds
        from modelblaster.pipeline import cuda_dtypes
        g = F.fire_graph("bf16")
        w = {k: cuda_dtypes.round_bf16(v) for k, v in F.fire_weights("fp32").items()}
        ir = self.mbir.load_ir(F.write_ir(self.tmp / "ir_bf16", g, w))
        self._same(ir)

    def test_equals_walk_on_a_fused_graph(self):
        from modelblaster.pipeline import graph_hints as gh
        d = F.fire_ir(self.tmp / "irf", "fp32")
        g = json.loads((d / "graph.json").read_text())
        h = {"id": "fuse:fused__conv2d__relu", "kind": "fuse", "target": "cuda", "fused_kind": "fused__conv2d__relu", "status": "accepted"}
        g2, _, prov = gh.apply_hints(g, [h])
        self.assertTrue(prov["applied"] and any("sub_ops" in o for o in g2["ops"]))
        ir = self.mbir.IR("fire", g2, dict(np.load(d / "weights.npz")), *[np.load(d / "io.npz")[k] for k in ("input", "output")])
        names, got = self._same(ir)
        self.assertIn("e1_o", got)                                              # an internal tensor of a fused op

    def test_frees_activations_after_their_last_reader(self):
        n, big = 30, [1, 64, 16, 16]
        tens = {f"t{i}": {"shape": big, "dtype": "f32", "quant": None} for i in range(n + 1)}
        ops = [{"name": f"r{i}", "op": "relu", "inputs": [f"t{i}"], "outputs": [f"t{i + 1}"], "shape": {"n": 16384}} for i in range(n)]
        g = {"name": "chain", "version": 1, "quant": "fp32", "input": {"tensor": "t0"}, "output": {"tensors": [f"t{n}"], "tensor": f"t{n}"},
             "tensors": tens, "ops": ops}
        d = F.write_ir(self.tmp / "chain", g, {}, golden_in=np.random.default_rng(0).standard_normal(16384).astype(np.float32))
        ir = self.mbir.load_ir(d)
        orig = self.wt.OPS["relu"]
        alive = {"walk": 0, "free": 0}
        refs = []

        def count(tag):
            def wrapped(ir_, op, ins, dev):
                r = orig(ir_, op, ins, dev)
                refs.append(weakref.ref(r[0]))
                alive[tag] = max(alive[tag], sum(1 for w in refs if w() is not None))
                return r
            return wrapped
        import torch
        try:
            self.wt.OPS["relu"] = count("walk")
            with torch.no_grad():
                o1, env = self.wt.walk(ir, "cpu")
            del env
            refs.clear()
            self.wt.OPS["relu"] = count("free")
            with torch.no_grad():
                o2, _ = self.wt.walk_free(ir, "cpu")
        finally:
            self.wt.OPS["relu"] = orig
        self.assertTrue(torch.equal(o1, o2))
        self.assertGreaterEqual(alive["walk"], n)             # walk() keeps every activation (the SqueezeNet OOM)
        self.assertLessEqual(alive["free"], 3)                 # walk_free: the live set only

    def test_cpu_fallback_and_skip(self):
        import torch
        ir = self.mbir.load_ir(F.fire_ir(self.tmp / "irc", "fp32"))
        calls = []
        real = self.wt._walk_free

        def oom(ir_, dev, *a):
            calls.append(dev)
            if dev != "cpu":
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
            return real(ir_, dev, *a)
        self.wt._walk_free = oom
        try:
            with torch.no_grad():
                out, _ = self.wt.walk_free(ir, "cuda")
                ref, _ = self.wt.walk(ir, "cpu")
        finally:
            self.wt._walk_free = real
        self.assertEqual(calls, ["cuda", "cpu"])
        self.assertTrue(torch.equal(out, ref))
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            self.wt._walk_free = oom
            try:
                self.wt.walk_free(ir, "cuda", cpu_fallback=False)
            finally:
                self.wt._walk_free = real
        # skipped (dead) ops do not run
        from modelblaster.pipeline.graph_prune import dead_ops
        g = copy.deepcopy(ir.graph)
        g["tensors"]["extra"] = {"shape": [1, 5, 6, 6], "dtype": "f32", "quant": None}
        g["ops"].insert(3, {"name": "extra_relu", "op": "mystery", "inputs": ["s"], "outputs": ["extra"], "shape": {"n": 1}})
        ir2 = self.mbir.IR("fire", g, ir.weights, ir.golden_in, ir.golden_out)
        with self.assertRaises(NotImplementedError):
            self.wt.walk_free(ir2, "cpu")
        out2, _ = self.wt.walk_free(ir2, "cpu", skip=set(dead_ops(g)))
        self.assertTrue(torch.equal(out2, ref))


if __name__ == "__main__":
    unittest.main()
