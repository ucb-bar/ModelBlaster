"""kb_reuse._write_back: an incumbent outside the pool's max_cands sources is still written back (E1, X-120).

The old write-back kept an entry only for incumbents that were among the pool's ``max_cands`` candidates, but it
superseded every old entry of the kind for the IR: the shapes of the other incumbents lost their kernel and fell to the
hand seed (15_DenseNet121 2,396 us vs 1,380 us with the fix; the model guard then reverted the whole pass). Offline: a
CudaCache with one cached incumbent per shape, the count taken from the 15_DenseNet121 data.
"""
import json
import tempfile
import unittest
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
kr = CudaCache = cache_plan = SPECS = None


def setUpModule():
    global kr, CudaCache, cache_plan, SPECS
    from modelblaster.pipeline import kb_reuse as kr
    from modelblaster.pipeline.kb_ingest import CudaCache, cache_plan
    from modelblaster.pipeline.kb_proposals import _all_specs
    SPECS = _all_specs()


def shape(i):
    return {"N": 10, "IC": 16 + i, "IH": 7, "IW": 7, "OC": 32, "OH": 7, "OW": 7, "KH": 3, "KW": 3, "SH": 1, "SW": 1,
            "PH": 1, "PW": 1, "DH": 1, "DW": 1}


class WriteBackTest(unittest.TestCase):
    KIND = "conv2d_f16"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = CudaCache(Path(self.tmp.name) / "cache")

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, n_incumbents: int, max_cands: int = 6):
        from modelblaster.pipeline.kernel_abi import parse_signature
        spec = SPECS[self.KIND]
        entry = parse_signature(spec.signature)[0]
        codes, graph_ops, table, inc_sha, inc_of_op = [], [], {}, {}, {}
        for i in range(n_incumbents):
            code = f"// incumbent {i}\n"
            codes.append(code)
            self.cache.add(op=self.KIND, entry=entry, signature=spec.signature, arch="sm_89", shapes=[shape(i)],
                           code=code, slot_us=20.0, speedup=10.0, impl_id=f"w{i}", source={})
            graph_ops.append({"name": f"conv{i}", "op": self.KIND, "shape": shape(i), "inputs": ["x0", f"c{i}w", f"c{i}b"],
                              "outputs": [f"y{i}"]})
            inc_of_op[f"conv{i}"] = code
        cands = [{"code": c, "sha": kr._sha(c), "from": f"cache:w{i}", "us_per_inst": 1.0, "verified": True}
                 for i, c in enumerate(codes[:max_cands])]
        for i in range(n_incumbents):
            table[i] = {"n": 1, "shape": shape(i), "ops": [f"conv{i}"], "incumbent": {"ok": True, "us": 100.0},
                        "cands": {c["sha"][:12]: {"ok": True, "us": 90.0} for c in cands}}
            inc_sha[i] = kr._sha(codes[i])
        # one shape (the first beyond the pool) takes pool source 0; every other shape keeps its incumbent
        pick = {i: None for i in range(n_incumbents)}
        if n_incumbents > max_cands:
            pick[max_cands] = cands[0]["sha"][:12]
        t = {"name": "dn", "tensors": {"x0": {"shape": [1]}}, "ops": graph_ops, "output": {"tensor": f"y{n_incumbents - 1}"}}
        for i in range(n_incumbents):
            t["tensors"].update({f"c{i}w": {"shape": [1], "const": f"c{i}w"}, f"c{i}b": {"shape": [1], "const": f"c{i}b"},
                                 f"y{i}": {"shape": [1]}})
        krep = {}
        kr._write_back(self.cache, self.KIND, SPECS, "sm_89", "dn", table, pick, cands, inc_sha, inc_of_op, krep)
        return t, krep

    def _served(self, g):
        """{op name: impl id} of the ops the cache serves in the plan: cache_plan splits the kind's slot into one group per
        covering entry (``split_slot`` hint) plus a group with ``cache: None`` for the instances on the hand seed."""
        plan = cache_plan(g, self.cache, "sm_89", SPECS)
        return {o: grp["cache"] for h in plan["hints"] for grp in h["groups"] if grp["cache"] for o in grp["ops"]}

    def test_every_shape_keeps_a_kernel_with_more_incumbents_than_max_cands(self):
        n = 12
        g, krep = self._run(n)
        used = self._served(g)
        self.assertEqual(len(used), n, "a shape whose incumbent was beyond max_cands fell back to the hand seed")
        self.assertEqual(krep["result"], "routed")
        self.assertEqual(sum(w["shapes"] for w in krep["written"]), n)
        # the shape that took pool source 0 shares that source's entry; the rest keep their own code
        self.assertEqual(len(krep["written"]), n - 1)

    def test_densenet121_scale(self):
        d = json.loads((DATA / "seed_left_15_densenet121.json").read_text())
        n = len({o["seed_source"] for o in d["ops"] if o["op"] == self.KIND})
        self.assertGreater(n, 50)                                  # 61 distinct conv incumbents in the shipped model
        g, krep = self._run(n)
        self.assertEqual(len(self._served(g)), n)

    def test_within_max_cands_nothing_extra_is_added(self):
        g, krep = self._run(4)
        self.assertEqual(len(krep["written"]), 4)                  # all four incumbents are pool sources: as before
        self.assertEqual(len(self._served(g)), 4)

    def test_incumbent_entry_is_marked_as_such(self):
        _, krep = self._run(8)
        froms = {w["from"] for w in krep["written"]}
        self.assertIn("incumbent", froms)                          # the 2 incumbents beyond the 6-source pool


if __name__ == "__main__":
    unittest.main()
