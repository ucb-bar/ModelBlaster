"""The rewrite sandbox (graph_rewrite_sandbox.py): accepts the fire-module merge written as a script (and equals merge_sibling_convs'
result), rejects imports / attribute escapes / unbounded loops / untraceable weights / dangling tensors with clear messages, and a wrong
rewrite is caught by the graph_verify gate. CPU only."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline import graph_edit as ge  # noqa: E402
from modelblaster.pipeline import graph_hints as gh  # noqa: E402
from modelblaster.pipeline import graph_rewrite_sandbox as sb  # noqa: E402
from modelblaster.pipeline import graph_verify as gv  # noqa: E402
from modelblaster.pipeline import merge_convs as mc  # noqa: E402
from modelblaster.pipeline import weight_derivations as wd  # noqa: E402
from modelblaster.pipeline.kb_proposals import _all_specs, write_variant_ir  # noqa: E402
from modelblaster.pipeline.tests import merge_fixtures as MF  # noqa: E402

FIRE_SCRIPT = '''
def rewrite(g, W, P, T):
    """fire module: conv1x1+relu | conv3x3+relu -> cat   =>   one conv3x3 (zero-padded, stacked filters) + relu"""
    e1, e3, cat = g.op("c0"), g.op("c1"), g.op("cat")
    r1, r3 = g.op("a0"), g.op("a1")
    s1, s3 = e1["shape"], e3["shape"]
    d = (s3["KH"] - s1["KH"]) // 2
    pad = P.pad("d.m.e1pad", e1["weight"], [[0, 0], [0, 0], [d, d], [d, d]])
    w = P.concat("d.m.w", [pad, e3["weight"]] if not SWAP else [e3["weight"], pad], 0)
    b = P.concat("d.m.b", [e1["bias"], e3["bias"]] if not SWAP else [e3["bias"], e1["bias"]], 0)
    oc = s1["OC"] + s3["OC"]
    out = cat["outputs"][0]
    dt = T.dtype(out)
    olds = [e1["outputs"][0], r1["outputs"][0], e3["outputs"][0], r3["outputs"][0]]
    g.remove_ops(["c0", "a0", "c1", "a1", "cat"])
    for t in olds:
        g.remove_tensor(t)
    mo = g.add_tensor("mrg_o", [s3["N"], oc, s3["OH"], s3["OW"]], dt)
    g.add_op({"name": "mrg", "op": e3["op"], "inputs": e3["inputs"], "outputs": [mo], "weight": w, "bias": b,
              "shape": {**s3, "OC": oc}})
    g.add_op({"name": "mrg_act", "op": r1["op"], "inputs": [mo], "outputs": [out], "shape": {"n": s3["N"] * oc * s3["OH"] * s3["OW"]}})
    return {"description": "merge the expand convs", "tensor_map": {olds[1]: {"from": out, "axis": 1, "start": 0, "stop": s1["OC"]},
                                                                       olds[3]: {"from": out, "axis": 1, "start": s1["OC"], "stop": oc}}}
'''


def fire_script(swap=False):
    return "SWAP = %s\n" % ("True" if swap else "False") + FIRE_SCRIPT


def base_weights(d):
    z = np.load(Path(d) / "weights.npz")
    return {k: z[k] for k in z.files}


def graph_of(d):
    return json.loads((Path(d) / "graph.json").read_text())


class SandboxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="mb_sb_"))
        cls.ir = {q: MF.branchy_ir(cls.tmp / f"o_{q}", MF.FIRE, q) for q in ("fp32", "fp16", "bf16", "native")}
        cls.g = {q: graph_of(d) for q, d in cls.ir.items()}
        cls.w = {q: base_weights(d) for q, d in cls.ir.items()}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_script(self, src, q="fp32", **kw):
        return sb.run_rewrite(src, self.g[q], self.w[q], **kw)

    def rejects(self, src, code, needle="", q="fp32", **kw):
        r = self.run_script(src, q, **kw)
        self.assertFalse(r.ok, "accepted: " + src)
        self.assertEqual(r.error["code"], code, r.error)
        self.assertIn(needle, r.error["msg"])
        self.assertNotIn("Traceback", r.error["msg"])
        return r

    # -- acceptance
    def test_fire_merge_script_is_accepted_and_verifies_in_every_dtype(self):
        for q in ("fp32", "fp16", "bf16", "native"):
            with self.subTest(q=q):
                r = self.run_script(fire_script(), q)
                self.assertTrue(r.ok, r.error)
                self.assertEqual(sorted(r.patch["remove_ops"]), ["a0", "a1", "c0", "c1", "cat"])
                self.assertEqual(sorted(o["name"] for o in r.patch["add_ops"]), ["mrg", "mrg_act"])
                self.assertEqual(sorted(r.derivations), ["d.m.b", "d.m.e1pad", "d.m.w"])
                self.assertEqual(r.provenance["d.m.w"]["leaves"], ["c0.weight", "c1.weight"])            # traced back to the parameters
                self.assertTrue(r.provenance["d.m.w"]["exact"])
                self.assertEqual(wd.validate(r.graph, set(self.w[q])), [])
                e = sb.write_result_ir(r, self.ir[q], self.tmp / f"e_{q}")
                v = gv.verify(self.ir[q], e, seeds=(0, 1), tensor_map=r.tensor_map)
                self.assertTrue(v.ok, v.summary())
                self.assertEqual(r.description, "merge the expand convs")
                json.dumps(r.to_dict(), default=str)
                self.assertEqual(len(self.g[q]["ops"]), 9)                                              # the input graph is untouched
                self.assertNotIn(wd.KEY, self.g[q])

    def test_patch_replays_on_the_base_graph(self):
        r = self.run_script(fire_script())
        g2, info = ge.apply_patch(self.g["fp32"], json.loads(json.dumps(r.patch)))
        self.assertEqual(ge.graph_digest(g2), ge.graph_digest(r.graph))
        self.assertEqual(sorted(info["added"]), ["mrg", "mrg_act"])

    def test_equals_the_merge_sibling_convs_applier(self):
        """the same rewrite written as a script and made by ITEM 3's applier: same ops, shapes, edges and bit-identical weights"""
        for q in ("fp32", "fp16"):
            with self.subTest(q=q):
                g0 = self.g[q]
                hints, _ = mc.propose(g0, None, dict(_all_specs()), None, policy=mc.Policy(compute_share=0.0, max_waste=2.0))
                g_m, _, _ = gh.apply_hints(g0, [{**hints[0], "status": "accepted"}], dict(_all_specs()))
                r = self.run_script(fire_script(), q)
                self.assertTrue(r.ok, r.error)
                z_m = wd.materialize(g_m, self.w[q])
                z_s = wd.materialize(r.graph, self.w[q])

                def canon(g, z):
                    names = {}
                    ops = []
                    for o in g["ops"]:
                        for t in o["outputs"]:
                            names.setdefault(t, f"t{len(names)}")
                    for o in g["ops"]:
                        wkey = lambda k: None if k is None else (z[k].dtype.str, z[k].shape, z[k].tobytes().hex()[:64], len(z[k].tobytes()))  # noqa: E731
                        ops.append((o["op"], tuple(names.get(t, t) for t in o["inputs"]), tuple(names.get(t, t) for t in o["outputs"]),
                                    wkey(o.get("weight")), wkey(o.get("bias")), json.dumps(o.get("shape"), sort_keys=True)))
                    ts = sorted((names.get(t, t), m["dtype"], tuple(m["shape"])) for t, m in g["tensors"].items())
                    return ops, ts
                self.assertEqual(canon(g_m, z_m), canon(r.graph, z_s))

    def test_untouched_ops_pass_through_and_orphan_tensors_are_pruned(self):
        src = '''
def rewrite(g, W, P, T):
    t = g.add_tensor("unused", [1, 2], "f32")
'''
        r = self.run_script(src)
        self.assertTrue(r.ok, r.error)
        self.assertNotIn("unused", r.graph["tensors"])
        self.assertEqual(r.patch["remove_ops"], [])

    def test_reads_are_copies_and_loops_are_fine(self):
        src = '''
def rewrite(g, W, P, T):
    o = g.op("c1")
    o["shape"]["OC"] = 999            # a copy: the graph does not change
    for name in g.ops():
        if g.producer(g.op(name)["outputs"][0]) is None:
            raise ValueError("producer lookup broke")
    n = 0
    for i in range(1000):
        n += i
    xs = [k for k in W.keys() if W.shape(k)]
    print("n", n, len(xs), T.numel("cat_out"), T.shape("cat_out"), math.prod([2, 3]))
'''
        r = self.run_script(src)
        self.assertTrue(r.ok, r.error)
        self.assertEqual(next(o for o in r.graph["ops"] if o["name"] == "c1")["shape"]["OC"], 3)
        self.assertTrue(r.log[0].startswith("n 499500"))

    # -- rejections: constructs
    def test_imports_are_rejected(self):
        self.rejects("import os\ndef rewrite(g, W, P, T):\n    pass\n", "forbidden_construct", "imports")
        self.rejects("from math import sqrt\ndef rewrite(g, W, P, T):\n    pass\n", "forbidden_construct", "imports")
        self.rejects("def rewrite(g, W, P, T):\n    import os\n", "forbidden_construct", "imports")
        self.rejects("def rewrite(g, W, P, T):\n    x = __import__('os')\n", "forbidden_name", "__import__")

    def test_attribute_escapes_are_rejected(self):
        self.rejects("def rewrite(g, W, P, T):\n    x = g.__class__\n", "forbidden_attr", "underscore")
        self.rejects("def rewrite(g, W, P, T):\n    x = P._rec\n", "forbidden_attr", "underscore")
        self.rejects("def rewrite(g, W, P, T):\n    x = ().__class__.__mro__\n", "forbidden_attr", "underscore")
        self.rejects("def rewrite(g, W, P, T):\n    x = (lambda: 1).__globals__\n", "forbidden_attr", "underscore")
        self.rejects("def rewrite(g, W, P, T):\n    x = '{0.__class__}'.format(g)\n", "forbidden_attr", "format")
        self.rejects("def rewrite(g, W, P, T):\n    x = f'{g.__dict__}'\n", "forbidden_attr", "underscore")
        self.rejects("def rewrite(g, W, P, T):\n    x = g.op.__func__\n", "forbidden_attr", "underscore")
        self.rejects("def rewrite(g, W, P, T):\n    x = [].gi_frame\n", "forbidden_attr", "whitelist")
        self.rejects("def rewrite(g, W, P, T):\n    x = getattr(g, 'op')\n", "forbidden_name", "getattr")
        self.rejects("def rewrite(g, W, P, T):\n    x = type(g)\n", "forbidden_name", "type")
        self.rejects("def rewrite(g, W, P, T):\n    open('/etc/passwd')\n", "forbidden_name", "open")
        self.rejects("def rewrite(g, W, P, T):\n    eval('1')\n", "forbidden_name", "eval")
        self.rejects("def rewrite(g, W, P, T):\n    exec('x=1')\n", "forbidden_name", "exec")
        self.rejects("def rewrite(g, W, P, T):\n    import_ = np\n", "forbidden_name", "np")

    def test_other_forbidden_constructs(self):
        self.rejects("def rewrite(g, W, P, T):\n    while True:\n        pass\n", "forbidden_construct", "while")
        self.rejects("class A:\n    pass\ndef rewrite(g, W, P, T):\n    pass\n", "forbidden_construct", "ClassDef")
        self.rejects("def rewrite(g, W, P, T):\n    x = 10 ** 10\n", "forbidden_construct", "Pow")
        self.rejects("def rewrite(g, W, P, T):\n    x = 1 << 99\n", "forbidden_construct", "LShift")
        self.rejects("def rewrite(g, W, P, T):\n    x = [0] * 99999999999\n", "forbidden_construct", "larger")
        self.rejects("def rewrite(g, W, P, T):\n    with g as x:\n        pass\n", "forbidden_construct", "With")
        self.rejects("def rewrite(g, W, P, T):\n    global x\n", "forbidden_construct", "Global")
        self.rejects("def rewrite(g, W, P, T):\n    (y := 1)\n", "forbidden_construct", "NamedExpr")
        self.rejects("@print\ndef f():\n    pass\ndef rewrite(g, W, P, T):\n    pass\n", "forbidden_construct", "decorators")
        self.rejects("def rewrite(g, W, P):\n    pass\n", "no_rewrite", "def rewrite")
        self.rejects("x = (", "syntax", "syntax error")
        self.rejects("# " + "x" * 300_000, "too_large", "KB")

    # -- rejections: unbounded work
    def test_unbounded_loops_are_stopped(self):
        r = self.rejects("def rewrite(g, W, P, T):\n    xs = [1]\n    for x in xs:\n        xs.append(x)\n", "step_budget", "statements", max_steps=5000)
        self.assertEqual(r.error["line"], 4)
        self.rejects("def rewrite(g, W, P, T):\n    for i in range(10 * 1000 * 1000):\n        pass\n", "step_budget", "range")
        self.rejects("def rewrite(g, W, P, T):\n    def f():\n        return f()\n    f()\n", "step_budget", "recursion")
        self.rejects("def rewrite(g, W, P, T):\n    n = 0\n    for i in range(900000):\n        n += 1\n", "step_budget", "statements", max_steps=1000)

    def test_wall_clock_limit(self):
        import threading
        if threading.current_thread() is not threading.main_thread():
            self.skipTest("SIGALRM needs the main thread")
        src = "def rewrite(g, W, P, T):\n    n = 0\n    for i in range(1000000):\n        for j in range(1000):\n            n += 1\n"
        r = self.rejects(src, "timeout", "longer than", timeout_s=0.3, max_steps=10 ** 12)
        self.assertIsNone(r.graph)

    # -- rejections: weights
    def test_untraceable_weights_are_rejected(self):
        # a weight key the model does not have, named by an op
        src = '''
def rewrite(g, W, P, T):
    c = g.op("c1")
    g.add_tensor("o2", [1, 3, 6, 6], "f32")
    g.add_op({"name": "x", "op": "conv2d", "inputs": c["inputs"], "outputs": ["o2"], "weight": "my.fake.weight", "bias": c["bias"], "shape": c["shape"]})
'''
        r = self.rejects(src, "weight_untraceable", "my.fake.weight")
        self.assertIn("P.concat", r.error["msg"])
        # w * w computed outside P: the values are not readable at all
        self.rejects('def rewrite(g, W, P, T):\n    w = W.value("c1.weight")\n    v = w * w\n', "weight_not_readable", "provenance")
        self.rejects('def rewrite(g, W, P, T):\n    w = W.get("c1.weight")\n', "weight_not_readable", "P.")
        # the same through an existing op: updating its weight to an invented key
        self.rejects('def rewrite(g, W, P, T):\n    g.update_op("c1", weight="d.not.derived")\n', "weight_untraceable", "d.not.derived")
        # a bad P call
        self.rejects('def rewrite(g, W, P, T):\n    P.concat("w2", ["c0.weight"], 0)\n', "bad_weight", "must be strings starting with")
        self.rejects('def rewrite(g, W, P, T):\n    P.concat("d.w2", ["c0.weight", "nope.weight"], 0)\n', "weight_untraceable", "nope.weight")
        self.rejects('def rewrite(g, W, P, T):\n    P.concat("d.w2", ["c0.weight", "c1.weight"], 0)\n', "bad_weight", "P:")        # 1x1 vs 3x3 filters: shapes differ
        self.rejects('def rewrite(g, W, P, T):\n    P.pad("c0.weight", "c0.weight", [[0, 0]] * 4)\n', "bad_weight", "d.")
        self.rejects('def rewrite(g, W, P, T):\n    P.pad("d.a", "c0.weight", [[0, 0]] * 4)\n    P.pad("d.a", "c0.weight", [[0, 0]] * 4)\n', "bad_weight", "already exists")
        # derived from derived is fine and traceable
        r = self.run_script('def rewrite(g, W, P, T):\n    a = P.pad("d.a", "c0.weight", [[0, 0], [0, 0], [1, 1], [1, 1]])\n    b = P.mul("d.b", a, a)\n    P.zeros("d.z", [2], "f32")\n')
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.provenance["d.b"]["leaves"], ["c0.weight"])
        self.assertFalse(r.provenance["d.b"]["exact"])                                                  # mul is arithmetic, not structural
        self.assertTrue(r.provenance["d.a"]["exact"])

    def test_derived_weights_follow_the_dtype_contract(self):
        for q, ldt in (("fp16", "f16"), ("bf16", "bf16"), ("fp32", "f32"), ("native", "f16")):
            r = self.run_script('def rewrite(g, W, P, T):\n    P.pad("d.a", "c0.weight", [[0, 0], [0, 0], [1, 1], [1, 1]])\n', q)
            self.assertTrue(r.ok, (q, r.error))
            self.assertEqual(r.derivations["d.a"]["dtype"], ldt)
            self.assertEqual(wd.validate(r.graph, set(self.w[q])), [])
        r = self.run_script('def rewrite(g, W, P, T):\n    P.zeros("d.z", [2], "f16")\n', "fp32")             # an f16 weight in an fp32 IR: the contract is violated
        self.assertTrue(r.ok)
        self.assertTrue(any(e["code"] == "dtype_contract" for e in wd.validate(r.graph, set(self.w["fp32"]))))

    # -- rejections: graph surgery
    def test_dangling_tensors_have_a_clear_message(self):
        # removed the concat's producers and never replaced them: cat_out has no producer, pool reads it
        r = self.rejects('def rewrite(g, W, P, T):\n    g.remove_ops(["cat"])\n', "dangling_tensor", "cat_out")
        self.assertIn("nothing produces", r.error["msg"])
        r = self.rejects('def rewrite(g, W, P, T):\n    g.remove_ops(["c0"])\n', "dangling_tensor", "c0_o")
        self.assertIn("rewire", r.error["msg"])
        self.rejects('def rewrite(g, W, P, T):\n    g.remove_ops(["fc"])\n', "dangling_tensor", "graph output 'out'")
        self.rejects('def rewrite(g, W, P, T):\n    g.add_op({"name": "n", "op": "relu", "inputs": ["ghost"], "outputs": ["cat_out"], "shape": {"n": 1}})\n',
                     "dangling_tensor", "ghost")
        r = self.rejects('def rewrite(g, W, P, T):\n    g.add_op({"name": "n", "op": "relu", "inputs": ["s"], "outputs": ["unregistered"], "shape": {"n": 1}})\n',
                         "bad_edit", "g.add_tensor")
        self.rejects('def rewrite(g, W, P, T):\n    g.op("nope")\n', "bad_edit", "no op 'nope'")
        self.rejects('def rewrite(g, W, P, T):\n    g.remove_ops(["nope"])\n', "bad_edit", "no such ops")
        self.rejects('def rewrite(g, W, P, T):\n    T.shape("nope")\n', "bad_edit", "no tensor 'nope'")
        self.rejects('def rewrite(g, W, P, T):\n    g.add_tensor("s", [1], "f32")\n', "bad_edit", "already exists")
        self.rejects('def rewrite(g, W, P, T):\n    g.add_tensor("zz", [0, 2], "f32")\n', "bad_edit", "positive")
        self.rejects('def rewrite(g, W, P, T):\n    g.remove_tensor("x")\n', "bad_edit", "interface")
        self.rejects('def rewrite(g, W, P, T):\n    g.add_op({"name": "c0", "op": "relu", "inputs": [], "outputs": []})\n', "bad_edit", "exists")
        self.rejects('def rewrite(g, W, P, T):\n    g.add_op({"name": "f", "op": "fused__conv2d__relu", "inputs": [], "outputs": []})\n', "bad_op", "fusion hints")
        self.rejects('def rewrite(g, W, P, T):\n    g.add_op({"name": "f", "op": "relu", "inputs": [], "outputs": [], "k": object})\n', "forbidden_name", "object")
        self.rejects('def rewrite(g, W, P, T):\n    g.add_op({"name": "f", "op": "relu", "inputs": [], "outputs": [], "k": lambda: 1})\n', "bad_op", "JSON")

    def test_a_cycle_is_a_reason(self):
        self.rejects('def rewrite(g, W, P, T):\n    g.update_op("sq", inputs=["cat_out"])\n', "cycle", "cycle")

    def test_script_errors_carry_the_exception_and_line(self):
        r = self.rejects('def rewrite(g, W, P, T):\n    x = 1\n    y = g["ops"]\n', "script_error", "TypeError")
        self.assertEqual(r.error["line"], 3)
        r = self.rejects('def rewrite(g, W, P, T):\n    d = {}\n    return d["missing"]\n', "script_error", "KeyError")
        r = self.rejects('def rewrite(g, W, P, T):\n    undefined_name + 1\n', "script_error", "NameError")
        r = self.rejects('def rewrite(g, W, P, T):\n    raise ValueError("no merge possible here")\n', "script_error", "no merge possible here")
        r = self.rejects('def rewrite(g, W, P, T):\n    return {"tensor_map": object}\n', "forbidden_name", "object")

    # -- the gate catches what the sandbox cannot
    def test_a_wrong_rewrite_is_caught_by_the_sampled_intermediate_gate(self):
        r = self.run_script(fire_script(swap=True))                      # filters and biases stacked in the wrong order: structurally perfect
        self.assertTrue(r.ok, r.error)
        e = sb.write_result_ir(r, self.ir["fp32"], self.tmp / "e_wrong")
        v = gv.verify(self.ir["fp32"], e, seeds=(0,), tensor_map=r.tensor_map)
        self.assertFalse(v.ok)
        self.assertTrue(any(x["code"] in ("sampled_mismatch", "output_mismatch", "tensor_map_mismatch") or "mismatch" in x["code"] for x in v.reasons), v.reasons)
        good = self.run_script(fire_script())
        e = sb.write_result_ir(good, self.ir["fp32"], self.tmp / "e_good")
        self.assertTrue(gv.verify(self.ir["fp32"], e, seeds=(0,), tensor_map=good.tensor_map).ok)

    def test_cli(self):
        import subprocess
        script = self.tmp / "fire.py"
        script.write_text(fire_script())
        out = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.graph_rewrite_sandbox", str(self.ir["fp32"]), str(script),
                              "--out", str(self.tmp / "cli_out"), "--verify"], capture_output=True, text=True, timeout=300)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn('"ok": true', out.stdout)
        self.assertTrue((self.tmp / "cli_out" / "graph.json").exists())
        bad = self.tmp / "bad.py"
        bad.write_text("import os\ndef rewrite(g, W, P, T):\n    pass\n")
        out = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.graph_rewrite_sandbox", str(self.ir["fp32"]), str(bad)],
                             capture_output=True, text=True, timeout=300)
        self.assertEqual(out.returncode, 1)
        self.assertIn("forbidden_construct", out.stdout)


if __name__ == "__main__":
    unittest.main()
