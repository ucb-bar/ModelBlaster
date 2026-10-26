"""The CUDA-only-baseline learnings (l3_farm/analysis/cuda_learnings_impl.md), offline (no torch, no GPU):
concurrency dependency analysis + launch plan, plan-key canonicalization, winner cross-application (choice, guards,
cache routing, seeds), sibling / epilogue / attention fusion matchers and per-shape fused seeds, default-off."""
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# imported in setUpModule, not at collection: importing `modelblaster` while another test module's sys.path insert is
# active makes test_fused_op_shape_key's checkout check see the wrong tree
cl = fusion_ext = gh = kc = kb_plan = kb_reuse = CudaCache = cache_plan = None
_all_specs = bind_fused = find_siblings = fuse_siblings = make_par_spec = spec_from_kind = with_fused_specs = None


def setUpModule():
    global cl, fusion_ext, gh, kc, kb_plan, kb_reuse, CudaCache, cache_plan
    global _all_specs, bind_fused, find_siblings, fuse_siblings, make_par_spec, spec_from_kind, with_fused_specs
    from modelblaster.pipeline import cuda_learnings as cl
    from modelblaster.pipeline import fusion_ext
    from modelblaster.pipeline import graph_hints as gh
    from modelblaster.pipeline import kb_concurrency as kc
    from modelblaster.pipeline import kb_plan
    from modelblaster.pipeline import kb_reuse
    from modelblaster.pipeline.kb_ingest import CudaCache, cache_plan
    from modelblaster.pipeline.kb_proposals import (_all_specs, bind_fused, find_siblings, fuse_siblings,
                                                    make_par_spec, spec_from_kind, with_fused_specs)

DATA = Path(__file__).resolve().parent / "data"
class _LazySpecs(dict):
    """KERNEL_SPECS on first use (importing them at collection time shadows generate_skeleton for
    test_fused_op_shape_key's path check)."""

    def _load(self):
        if not dict.__len__(self):
            from modelblaster.pipeline.kb_proposals import _all_specs as _a
            self.update(_a())

    def __getitem__(self, k):
        self._load()
        return dict.__getitem__(self, k)

    def __contains__(self, k):
        self._load()
        return dict.__contains__(self, k)

    def get(self, k, d=None):
        self._load()
        return dict.get(self, k, d)

    def keys(self):
        self._load()
        return dict.keys(self)

    def items(self):
        self._load()
        return dict.items(self)

    def values(self):
        self._load()
        return dict.values(self)

    def __iter__(self):
        self._load()
        return dict.__iter__(self)

    def __len__(self):
        self._load()
        return dict.__len__(self)


SPECS = _LazySpecs()


def gru(name, x, h0, out, hn, I=512, rev=0):
    return {"name": name, "op": "gru_seq_f16",
            "inputs": [x, h0, f"{name}__w_ih", f"{name}__w_hh", f"{name}__b_ih", f"{name}__b_hh"],
            "outputs": [out, hn], "shape": {"B": 2, "T": 4, "I": I, "H": 8, "reverse": rev}}


def bidir_graph(layers=2):
    """x -> copy -> [fwd, rev] -> cat -> [fwd, rev] -> cat (41_GRUBidirectional's structure, tiny)."""
    t = {"x": {"shape": [4, 2, 16]}, "x2": {"shape": [2, 4, 16]}, "h0": {"shape": [2 * layers, 2, 8]}}
    ops = [{"name": "copy", "op": "strided_copy_f16", "inputs": ["x"], "outputs": ["x2"],
            "shape": {"off": 0, "d0": 1, "d1": 1, "d2": 1, "d3": 2, "d4": 4, "d5": 16, "s0": 0, "s1": 0, "s2": 0,
                      "s3": 16, "s4": 32, "s5": 1}}]
    cur, I = "x2", 16
    for l in range(layers):
        for d in (0, 1):
            ops.append({"name": f"h0v{l}{d}", "op": "view", "inputs": ["h0"], "outputs": [f"h0_{l}{d}"],
                        "shape": {"n": 16}, "offset": 16 * (2 * l + d)})
            t[f"h0_{l}{d}"] = {"shape": [2, 8]}
        for d, nm in ((0, f"gru_l{l}"), (1, f"gru_l{l}_reverse")):
            ops.append(gru(nm, cur, f"h0_{l}{d}", f"y{l}{d}", f"hn{l}{d}", I=I, rev=d))
            t[f"y{l}{d}"], t[f"hn{l}{d}"] = {"shape": [2, 4, 8]}, {"shape": [2, 8]}
            for w, sh in (("w_ih", [24, I]), ("w_hh", [24, 8]), ("b_ih", [24]), ("b_hh", [24])):
                t[f"{nm}__{w}"] = {"shape": sh, "const": f"{nm}__{w}"}
        ops.append({"name": f"cat{l}", "op": "catn_f16", "inputs": [f"y{l}0", f"y{l}1"], "outputs": [f"c{l}"],
                    "shape": {"n_in": 2, "outer": 8, "inner": 1, "c0": 8, "c1": 8, "c2": 0, "c3": 0, "c4": 0,
                              "c5": 0, "c6": 0, "c7": 0}})
        t[f"c{l}"] = {"shape": [2, 4, 16]}
        cur, I = f"c{l}", 16
    return {"name": "bidir", "tensors": t, "ops": ops, "output": {"tensor": cur}}


def recs_for(g, sel=None):
    """(op_recs, conc_ops, bufs) as export_problem builds them (one buffer per tensor, no aliases)."""
    ops = g["ops"]
    sel = sel if sel is not None else list(range(len(ops)))
    bufs, index = [], {}
    for o in ops:
        for t in o["inputs"] + o["outputs"]:
            if t not in index:
                index[t] = len(bufs)
                bufs.append({"name": t, "numel": 8, "alias": None})
    recs, conc = [], []
    for i in sel:
        o = ops[i]
        if o["op"] == "view":
            continue
        sid = o["op"].replace("_f16", "") + "_f16"
        recs.append({"index": len(recs), "name": o["name"], "op": o["op"], "slot": sid,
                     "inputs": [index[t] for t in o["inputs"]], "outputs": [index[t] for t in o["outputs"]]})
        conc.append((i, sid, f"mbslot_{sid}::kernel_{o['op']}(x);", []))
    return recs, conc, bufs, index


class ConcurrencyAnalysisTest(unittest.TestCase):
    def test_directions_are_concurrent_and_get_lanes(self):
        g = bidir_graph()
        recs, conc, bufs, _ = recs_for(g)
        s = kc.schedule(g, list(range(len(g["ops"]))), recs, conc, bufs)
        names = s["names"]
        k = {n: i for i, n in enumerate(names)}
        self.assertTrue(s["concurrent"])
        self.assertEqual(s["preds"][k["gru_l0"]], [k["copy"]])
        self.assertEqual(s["preds"][k["gru_l0_reverse"]], [k["copy"]])
        self.assertEqual(sorted(s["preds"][k["cat0"]]), sorted([k["gru_l0"], k["gru_l0_reverse"]]))
        self.assertEqual(s["preds"][k["gru_l1"]], [k["cat0"]])          # transitively reduced
        self.assertEqual(s["width"][k["gru_l0"]], 2)
        self.assertEqual(s["width"][k["cat0"]], 1)
        # the two directions share one slot: the reverse one runs on its own copy (lane 1)
        self.assertEqual(s["lane"][k["gru_l0"]], 0)
        self.assertEqual(s["lane"][k["gru_l0_reverse"]], 1)
        self.assertEqual(s["lanes"], {"gru_seq_f16": 2})
        tail = "\n".join(kc.tail(s, "split", len(recs), "mbslot_"))
        self.assertIn("mbslot_gru_seq_f16__lane1::kernel_gru_seq_f16", tail)
        self.assertIn("cudaStreamUpdateCaptureDependencies", tail)
        self.assertIn("MB_CONCURRENCY", tail)
        self.assertIn('extern "C" int mb_sm_budget(void)', tail)

    def test_partition_keeps_whole_graph_order(self):
        """A plan problem holding only the GRU ops: layer 1 still follows layer 0 (through the cat it lacks)."""
        g = bidir_graph()
        sel = [i for i, o in enumerate(g["ops"]) if o["op"] == "gru_seq_f16"]
        recs, conc, bufs, _ = recs_for(g, sel)
        s = kc.schedule(g, sel, recs, conc, bufs)
        k = {n: i for i, n in enumerate(s["names"])}
        self.assertEqual(sorted(s["preds"][k["gru_l1"]]), sorted([k["gru_l0"], k["gru_l0_reverse"]]))
        self.assertEqual(s["preds"][k["gru_l0_reverse"]], [])
        self.assertEqual(s["width"][k["gru_l1"]], 2)

    def test_chain_is_not_concurrent(self):
        g = {"name": "c", "tensors": {}, "output": {"tensor": "c"},
             "ops": [{"name": "a", "op": "relu_f16", "inputs": ["x"], "outputs": ["a"], "shape": {"n": 8}},
                     {"name": "b", "op": "relu_f16", "inputs": ["a"], "outputs": ["b"], "shape": {"n": 8}},
                     {"name": "c", "op": "relu_f16", "inputs": ["b"], "outputs": ["c"], "shape": {"n": 8}}]}
        recs, conc, bufs, _ = recs_for(g)
        s = kc.schedule(g, [0, 1, 2], recs, conc, bufs)
        self.assertFalse(s["concurrent"])
        tail = "\n".join(kc.tail(s, "split", 3, "mbslot_"))
        self.assertIn("for (int i = 0; i < 3; ++i) mb_run_op(i, b);", tail)
        self.assertNotIn("mb_conc_launch", tail)

    def test_aliasing_writes_are_ordered(self):
        """Two independent ops whose outputs overlap on one base buffer (an in-place / aliased write) stay ordered."""
        g = {"name": "h", "tensors": {}, "output": {"tensor": "o2"},
             "ops": [{"name": "a", "op": "relu_f16", "inputs": ["x"], "outputs": ["o1"], "shape": {"n": 8}},
                     {"name": "b", "op": "relu_f16", "inputs": ["y"], "outputs": ["o2"], "shape": {"n": 8}}]}
        recs, conc, bufs, index = recs_for(g)
        s = kc.schedule(g, [0, 1], recs, conc, bufs)
        self.assertTrue(s["concurrent"])
        bufs[index["o2"]]["alias"] = {"base": index["o1"], "offset": 4}      # o2 overlaps o1[4:]
        s = kc.schedule(g, [0, 1], recs, conc, bufs)
        self.assertFalse(s["concurrent"])
        self.assertEqual(s["preds"][1], [0])
        self.assertEqual(s["edges_hazard"], 1)
        bufs[index["o2"]]["alias"] = {"base": index["o1"], "offset": 8}      # disjoint ranges: independent again
        self.assertTrue(kc.schedule(g, [0, 1], recs, conc, bufs)["concurrent"])

    def test_policy_none_and_resolve(self):
        g = bidir_graph(1)
        recs, conc, bufs, _ = recs_for(g)
        s = kc.schedule(g, list(range(len(g["ops"]))), recs, conc, bufs)
        self.assertIn("MB_WIDTH[] = {1, 1, 1, 1}", "\n".join(kc.tail(s, "none", len(recs), "mbslot_")))
        self.assertIn("MB_STRICT = 1", "\n".join(kc.tail(s, "strict", len(recs), "mbslot_")))
        old = os.environ.pop("MB_GRAPH_CONCURRENCY", None)
        try:
            self.assertIsNone(kc.resolve(None, None))                     # default: off
            self.assertEqual(kc.resolve("on", None), "split")
            with self.assertRaises(ValueError):
                kc.resolve("maybe", None)
            with self.assertRaises(ValueError):
                kc.resolve("on", "half")
        finally:
            if old is not None:
                os.environ["MB_GRAPH_CONCURRENCY"] = old

    def test_driver_watchdog(self):
        from modelblaster.pipeline.kb_export import _DRIVER_TEMPLATE
        d = kc.driver(_DRIVER_TEMPLATE)
        self.assertNotIn("cudaGraphLaunch(ge, s)", d)
        self.assertIn("mb_graph_launch(ge, s)", d)
        self.assertIn("MB_CONC_DEADLOCK", d)
        self.assertIn('"MB_CONCURRENCY"', d)
        self.assertIn("_exit(5)", d)
        self.assertTrue(kc.deadlocked({"log": "x\nMB_CONC_DEADLOCK graph replay ...\nfailed"}))
        self.assertFalse(kc.deadlocked({"log": "passed"}))

    def test_slot_note(self):
        g = bidir_graph(1)
        recs, conc, bufs, _ = recs_for(g)
        s = kc.schedule(g, list(range(len(g["ops"]))), recs, conc, bufs)
        slots = [{"id": "gru_seq_f16", "semantics": "GRU"}, {"id": "catn_f16", "semantics": "cat"}]
        kc.annotate_slots(slots, s)
        self.assertIn("mb_sm_budget()", slots[0]["semantics"])
        self.assertEqual(slots[0]["concurrent_width"], 2)
        self.assertNotIn("concurrency", slots[1]["semantics"])


class LanesAssemblyTest(unittest.TestCase):
    def _problem(self, d: Path, lanes=None):
        (d / "slots").mkdir(parents=True)
        (d / "walker_head.cu").write_text("// head\n")
        (d / "walker_tail.cu").write_text("// tail\n")
        (d / "slots" / "a.cu").write_text("static int st = 0;\nvoid kernel_a(int x) { st += x; }\n")
        man = {"format": "mb-kb-problem/1", "name": "p", "assembly": {"head": "walker_head.cu", "tail": "walker_tail.cu",
               "namespace_prefix": "mbslot_", "order": ["a"], **({"lanes": lanes} if lanes else {})},
               "slots": [{"id": "a", "op": "a", "entry": "kernel_a", "signature": "void kernel_a(int x)",
                          "file": "slots/a.cu"}], "ops": [], "buffers": [], "driver": "driver.cpp", "data": "data.bin"}
        (d / "manifest.json").write_text(json.dumps(man))
        (d / "driver.cpp").write_text("")
        (d / "data.bin").write_bytes(b"")

    def test_units_emit_lane_copies(self):
        from modelblaster.pipeline.kb_export import assemble, units
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            self._problem(d, {"a": 2})
            names = [n for n, _ in units(d)]
            self.assertEqual(names, ["slot_a.cu", "slot_a__lane1.cu", "walker_tail.cu"])
            tail = dict(units(d))["walker_tail.cu"]
            self.assertIn("namespace mbslot_a__lane1 { void kernel_a(int x); }", tail)
            self.assertIn("namespace mbslot_a__lane1 {", assemble(d))
        with tempfile.TemporaryDirectory() as t:                    # default: unchanged
            d = Path(t)
            self._problem(d)
            self.assertEqual([n for n, _ in units(d)], ["slot_a.cu", "walker_tail.cu"])

    def test_kernelblaster_units_emit_lane_copies(self):
        kb_src = Path(__file__).resolve().parents[2] / "third_party" / "KernelBlaster" / "src"
        sys.path.insert(0, str(kb_src))
        try:
            from kernelblaster.knowledge.slots import SlotProblem
        except Exception as e:  # noqa: BLE001 -- optional deps of the package
            self.skipTest(f"KernelBlaster not importable here: {e}")
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            self._problem(d, {"a": 3})
            p = SlotProblem(d)
            names = [n for n, _ in p.units({"a": "void kernel_a(int x) {}"})]
            self.assertEqual(names, ["slot_a.cu", "slot_a__lane1.cu", "slot_a__lane2.cu", "walker_tail.cu"])
            self.assertIn("namespace mbslot_a__lane2 {", p.assemble({"a": "void kernel_a(int x) {}"}))


class CanonTest(unittest.TestCase):
    def _profile(self):
        ops = []
        for nm, rev, us in (("f", 0, 557.0), ("r", 1, 1000.0)):
            ops.append({"name": nm, "op": "gru_seq_f16", "entry": "kernel_gru_seq_f16", "signature_sha": "s",
                        "shape": {"B": 10, "T": 512, "I": 512, "H": 256, "reverse": rev}, "us": us, "seed": "hand"})
        return {"ir": "m", "ir_dir": "/x", "ops": ops, "partition_us": 1.0}

    def test_reverse_is_a_flag(self):
        self.assertEqual(kb_plan.canon_fields("gru_seq_f16", SPECS), ("reverse",))
        self.assertEqual(kb_plan.canon_fields("lstm_seq_f16", SPECS), ("reverse",))
        self.assertEqual(kb_plan.canon_fields("relu_f16", SPECS), ())
        # a listed field that is not an int parameter of the signature is kept (not verifiable)
        self.assertEqual(kb_plan.canon_fields("gru_seq_f16", SPECS, {"gru_seq": ("reverse", "nope")}), ("reverse",))

    def test_one_item_for_both_directions(self):
        off = kb_plan.rank([self._profile()])
        self.assertEqual(len(off), 2)                                   # default: exact shape keys
        on = kb_plan.rank([self._profile()], canon=kb_plan.CANON_FLAGS, specs=SPECS)
        self.assertEqual(len(on), 1)
        self.assertEqual(on[0]["occurrences"][0]["ops"], ["f", "r"])
        self.assertNotIn("reverse", on[0]["shape"])
        self.assertEqual(len(on[0]["shapes"]), 2)
        plan = kb_plan.make_plan([self._profile()], coverage=2.0, per_item=True, canon=kb_plan.CANON_FLAGS, specs=SPECS)
        self.assertEqual(plan["problems"][0]["op_names"], ["f", "r"])
        self.assertEqual(kb_plan.parse_canon_flags("gru_seq:reverse,x:y"), {"gru_seq": ("reverse",), "x": ("y",)})


def _gru_graph_for_cache():
    """4 GRU instances, 3 shapes (41-like: l0 fwd/rev with I=128, l1 fwd/rev with I=512)."""
    g = bidir_graph(2)
    for o in g["ops"]:
        if o["op"] == "gru_seq_f16" and o["name"].startswith("gru_l0"):
            o["shape"]["I"] = 128
    return g


class ReuseTest(unittest.TestCase):
    def test_choose_guards(self):
        table = {0: {"incumbent": {"ok": True, "us": 1000.0}, "cands": {"aaa": {"ok": True, "us": 557.0},
                                                                         "bbb": {"ok": False, "us": None}}},
                 1: {"incumbent": {"ok": True, "us": 557.0}, "cands": {"aaa": {"ok": True, "us": 550.0}}},   # < 3%
                 2: {"incumbent": {"ok": True, "us": 500.0}, "cands": {"ccc": {"ok": True, "us": 600.0}}},   # slower
                 3: {"incumbent": {"ok": False, "us": None}, "cands": {"ccc": {"ok": True, "us": 900.0}}}}
        pick = kb_reuse.choose(table, [], {0: "x" * 40, 1: "y" * 40, 2: "z" * 40, 3: None}, 0.03)
        self.assertEqual(pick, {0: "aaa", 1: None, 2: None, 3: "ccc"})
        # the incumbent's own code is never "picked"
        t0 = {0: {"incumbent": {"ok": True, "us": 1000.0}, "cands": {"aaa000000000": {"ok": True, "us": 557.0}}}}
        pick = kb_reuse.choose(t0, [], {0: "aaa" + "0" * 37}, 0.03)
        self.assertEqual(pick, {0: None})

    def test_write_back_routes_per_shape_and_is_ir_scoped(self):
        g = _gru_graph_for_cache()
        specs = dict(SPECS)
        sig = specs["gru_seq_f16"].signature
        with tempfile.TemporaryDirectory() as t:
            cache = CudaCache(Path(t))
            shapes = {n: o["shape"] for o in g["ops"] for n in [o["name"]] if o["op"] == "gru_seq_f16"}
            codes = {}
            for nm, ops in (("fwd", ["gru_l1"]), ("rev", ["gru_l1_reverse"]), ("l0f", ["gru_l0"]),
                            ("l0r", ["gru_l0_reverse"])):
                codes[nm] = f"// {nm}\nvoid kernel_gru_seq_f16() {{}}\n"
                cache.add(op="gru_seq_f16", entry="kernel_gru_seq_f16", signature=sig, arch="sm_89",
                          shapes=[shapes[o] for o in ops], code=codes[nm], slot_us=100.0, speedup=2.0, impl_id=nm,
                          source={})
            before = cache_plan(g, cache, "sm_89", specs)
            self.assertEqual(len([s for s in before["sources"] if s.startswith("gru_seq_f16")]), 4)   # 4 variants
            cands = kb_reuse.pool_candidates("gru_seq_f16", cache, arch="sm_89", specs=specs, ir_name="bidir")
            self.assertEqual(len(cands), 4)
            by = {c["from"]: c for c in cands}
            fwd = by["cache:fwd"]["sha"][:12]
            # forward winner fastest on every shape -> one entry covering all four instances
            table = {}
            for k, (nm, sh) in enumerate(sorted({json.dumps(s, sort_keys=True): n for n, s in shapes.items()}.items())):
                table[k] = {"n": 1, "shape": json.loads(nm), "ops": [sh], "incumbent": {"ok": True, "us": 1000.0},
                            "cands": {fwd: {"ok": True, "us": 557.0}}}
            inc_sha = {k: kb_reuse._sha(codes[{"gru_l1": "fwd", "gru_l1_reverse": "rev", "gru_l0": "l0f",
                                                "gru_l0_reverse": "l0r"}[row["ops"][0]]]) for k, row in table.items()}
            pick = kb_reuse.choose(table, cands, inc_sha, 0.03)
            krep = {}
            kb_reuse._write_back(cache, "gru_seq_f16", specs, "sm_89", "bidir", table, pick, cands, inc_sha, {}, krep)
            self.assertEqual(krep["result"], "routed")
            self.assertEqual(len(krep["written"]), 1)
            after = cache_plan(g, cache, "sm_89", specs)
            self.assertEqual(after["hints"], [])                         # no split any more: one slot
            self.assertEqual(after["sources"]["gru_seq_f16"], codes["fwd"])
            e = after["used"]["gru_seq_f16"]
            self.assertTrue(e["reuse"])
            self.assertEqual(e["source"]["from"], "cache:fwd")
            # the old entries are superseded for this IR only (the routed entry is shape-guarded like any other)
            old = [e for e in cache.entries if not e.get("reuse")]
            self.assertTrue(all(e["superseded"] == ["bidir"] for e in old))
            g2 = copy.deepcopy(g)
            g2["name"] = "other"
            from modelblaster.pipeline.kb_ingest import superseded
            self.assertFalse(any(superseded(e, "other") for e in old))

    def test_pool_global_records_and_seed_candidates(self):
        specs = dict(SPECS)
        sig = specs["gru_seq_f16"].signature
        from modelblaster.pipeline.cuda_dtypes import cuda_signature
        from modelblaster.pipeline.kb_ingest import signature_sha
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            cache = CudaCache(t / "cache")
            cache.add(op="gru_seq_f16", entry="kernel_gru_seq_f16", signature=sig, arch="sm_89",
                      shapes=[{"B": 1}], code="// a\n", slot_us=10.0, speedup=2.0, impl_id="a", source={})
            impl = t / "impl"
            (impl / "code").mkdir(parents=True)
            rows = []
            for k, (sp, op) in enumerate(((3.0, "gru_seq_f16"), (1.0, "gru_seq_f16"), (5.0, "relu_f16"))):
                code = f"// g{k}\n"
                sha = kb_reuse._sha(code)
                (impl / "code" / f"{sha}.cu").write_text(code)
                rows.append({"op": op, "signature_sha": signature_sha(cuda_signature(sig)), "arch": "sm_89",
                             "dtype": "fp16", "target": "cuda", "code_sha": sha, "shapes": [{"B": 2}], "slot_us": 5.0,
                             "speedup_vs_seed": sp, "id": f"r{k}", "problem_id": "p"})
            (impl / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            sib = kb_reuse.pool_candidates("gru_seq_f16", cache, arch="sm_89", specs=specs, ir_name="m")
            self.assertEqual([c["from"] for c in sib], ["cache:a"])
            glo = kb_reuse.pool_candidates("gru_seq_f16", cache, arch="sm_89", specs=specs, ir_name="m",
                                           mode="global", impl_dirs=[impl])
            self.assertEqual(sorted(c["from"] for c in glo), ["cache:a", "impl:r0:p"])    # below 1.02x / other op: out
            self.assertEqual(kb_reuse.pool_candidates("gru_seq_f16", cache, arch="sm_89", specs=specs, ir_name="m",
                                                      mode="off"), [])
            # seeds for a fresh problem whose slot starts from the hand kernel
            pd = t / "prob"
            (pd / "slots").mkdir(parents=True)
            (pd / "slots" / "gru_seq_f16.cu").write_text("// hand\n")
            (pd / "manifest.json").write_text(json.dumps({"ir_name": "m", "slots": [
                {"id": "gru_seq_f16", "op": "gru_seq_f16", "file": "slots/gru_seq_f16.cu", "seed_source": "hand"}]}))
            r = kb_reuse.add_seed_candidates(pd, cache, arch="89", specs=specs, mode="global", impl_dirs=[impl])
            man = json.loads((pd / "manifest.json").read_text())
            self.assertEqual(len(man["seed_candidates"]["gru_seq_f16"]), 2)
            self.assertTrue((pd / man["seed_candidates"]["gru_seq_f16"][0]["file"]).exists())
            self.assertEqual(r["gru_seq_f16"], ["impl:r0:p", "cache:a"])      # faster per instance first


class SiblingFusionTest(unittest.TestCase):
    def test_find_and_fuse_bidirectional_layers(self):
        g = gh.unique_names(json.loads(json.dumps(bidir_graph(2))))
        groups = find_siblings(g, SPECS)
        self.assertEqual(groups, [["gru_l0", "gru_l0_reverse"], ["gru_l1", "gru_l1_reverse"]])
        g2, spec = fuse_siblings(g, groups[0], SPECS)
        self.assertEqual(spec.op, "fused__par__gru_seq_f16__gru_seq_f16")
        names = [o["name"] for o in g2["ops"]]
        self.assertIn("gru_l0+gru_l0_reverse", names)
        self.assertLess(names.index("gru_l0+gru_l0_reverse"), names.index("cat0"))
        fused = next(o for o in g2["ops"] if o["name"] == "gru_l0+gru_l0_reverse")
        self.assertEqual(fused["outputs"], ["y00", "hn00", "y01", "hn01"])
        self.assertEqual(spec_from_kind(spec.op, SPECS).signature, spec.signature)
        self.assertIn("int a_reverse", spec.signature)
        self.assertIn("b_x", spec.signature)
        self.assertIn("int b_reverse", spec.signature)
        args = bind_fused(fused, with_fused_specs(g2, SPECS))
        self.assertEqual([a.value for a in args if a.kind == "out"], ["y00", "hn00", "y01", "hn01"])
        self.assertFalse(any(a.kind == "scratch" for a in args))
        from modelblaster.pipeline.op_classes import op_class
        from modelblaster.pipeline.cuda_dtypes import FP16, op_dtype
        self.assertEqual(op_class(spec.op), "gemm")
        self.assertEqual(op_dtype(spec.op), FP16)
        # a dependent pair is not a sibling group
        with self.assertRaises(ValueError):
            fuse_siblings(g, ["gru_l0", "gru_l1"], SPECS)

    def test_par_seed_calls_every_stage(self):
        spec = make_par_spec(["gru_seq_f16", "gru_seq_f16"], SPECS)

        class Lib:
            items = {}

            def closure(self, e):
                return []
        src = spec.cuda_seed(Lib(), sub_sources={"gru_seq_f16": "void kernel_gru_seq_f16(...) {}\n"})
        self.assertIn("stage0_gru_seq_f16::kernel_gru_seq_f16(a_x,", src)      # each stage its own copy (statics)
        self.assertIn("stage1_gru_seq_f16::kernel_gru_seq_f16(b_x,", src)
        self.assertIn("namespace stage1_gru_seq_f16 {", src)
        self.assertIn("independent sibling ops", src)
        # stages overlap: stage 1 on a helper thread's stream, forked / joined with events (capture-safe)
        self.assertIn("cudaStreamWaitEvent(cudaStreamPerThread, mb_fork, 0)", src)
        self.assertIn("cudaStreamWaitEvent(cudaStreamPerThread, mb_join[i], 0)", src)
        self.assertLess(src.index("std::thread([&]"), src.index("stage0_gru_seq_f16::kernel_gru_seq_f16(a_x,"))

    def test_hint_apply_and_proposal(self):
        g = bidir_graph(2)
        op_us = {"gru_l0": 557.0, "gru_l0_reverse": 1000.0, "gru_l1": 557.0, "gru_l1_reverse": 1000.0}
        hs = fusion_ext.propose_siblings(g, op_us, SPECS, total_us=4000.0, min_gain_pct=1.0, ir="bidir", done=set())
        self.assertEqual(len(hs), 1)
        h = hs[0]
        self.assertEqual(h["kind"], "fuse_siblings")
        self.assertAlmostEqual(h["source"][0]["est_gain_us"], 2 * 557.0)
        g2, added, prov = gh.apply_hints(g, [h], dict(SPECS), statuses=None)
        self.assertEqual(len([a for a in prov["applied"] if a["kind"] == "fuse_siblings"]), 2)
        self.assertIn(h["fused_kind"], added)
        self.assertEqual(sorted(fusion_ext.scope_ops(h, gh.unique_names(copy.deepcopy(g)), dict(SPECS))),
                         ["gru_l0", "gru_l0_reverse", "gru_l1", "gru_l1_reverse"])
        # accepted -> pattern level (no groups): applies wherever find_siblings matches
        acc = {**h, "status": "accepted"}
        acc.pop("groups")
        g3, _, prov3 = gh.apply_hints(g, [acc], dict(SPECS))
        self.assertEqual(len(prov3["applied"]), 2)

    def test_fused_variant_keeps_per_shape_stage_winners(self):
        """A fused slot over instances whose stage-0 winners are per shape is split by them, each part seeded with
        its own stage kernel (cache_plan split=True: what prepare_hint uses with --fusion-extended)."""
        g = _gru_graph_for_cache()
        h = fusion_ext.siblings_hint("fused__par__gru_seq_f16__gru_seq_f16",
                                     [["gru_l0", "gru_l0_reverse"], ["gru_l1", "gru_l1_reverse"]], {"by": "t"})
        specs = dict(SPECS)
        vg, added, _ = gh.apply_hints(g, [h], specs, statuses=None)
        specs.update(added)
        sig = specs["gru_seq_f16"].signature
        with tempfile.TemporaryDirectory() as t:
            cache = CudaCache(Path(t))
            for nm in ("gru_l0", "gru_l1"):
                sh = next(o["shape"] for o in g["ops"] if o["name"] == nm)
                cache.add(op="gru_seq_f16", entry="kernel_gru_seq_f16", signature=sig, arch="sm_89", shapes=[sh],
                          code=f"// {nm}\n", slot_us=1.0, speedup=2.0, impl_id=nm, source={})
            one = cache_plan(vg, cache, "sm_89", specs, split=False)
            fid = "fused__par__gru_seq_f16__gru_seq_f16"
            self.assertEqual(one["sub_sources"].get(fid), {})            # no stage kernel covers both: hand seeds
            per = cache_plan(vg, cache, "sm_89", specs, split=True)
            self.assertEqual(len(per["hints"]), 1)
            subs = {k: v for k, v in per["sub_sources"].items() if k.startswith(fid)}
            self.assertEqual(sorted(v["gru_seq_f16"] for v in subs.values() if v), ["// gru_l0\n", "// gru_l1\n"])
        self.assertTrue(fusion_ext.inherit("siblings"))
        self.assertFalse(fusion_ext.inherit("off"))


class EpilogueAttentionTest(unittest.TestCase):
    def test_conv_relu_pool_chain(self):
        conv = {"name": "conv", "op": "conv2d_f16", "inputs": ["x", "w", "b"], "outputs": ["c"],
                "shape": {"N": 1, "IC": 3, "IH": 8, "IW": 8, "OC": 4, "KH": 3, "KW": 3, "SH": 1, "SW": 1, "PH": 1,
                          "PW": 1, "DH": 1, "DW": 1, "OH": 8, "OW": 8, "G": 1}}
        relu = {"name": "relu", "op": "relu_f16", "inputs": ["c"], "outputs": ["r"], "shape": {"n": 256}}
        pool = {"name": "pool", "op": "maxpool2d_f16", "inputs": ["r"], "outputs": ["p"],
                "shape": {"N": 1, "C": 4, "IH": 8, "IW": 8, "OH": 4, "OW": 4, "KH": 2, "KW": 2, "SH": 2, "SW": 2,
                          "PH": 0, "PW": 0}}
        if any(o["op"] not in SPECS for o in (conv, relu, pool)):
            self.skipTest("conv/relu/maxpool f16 specs not present")
        g = {"name": "cnn", "tensors": {}, "ops": [conv, relu, pool], "output": {"tensor": "p"}}
        hs = fusion_ext.propose_epilogue(g, {"conv": 100.0, "relu": 20.0, "pool": 15.0}, dict(SPECS), total_us=135.0,
                                         min_gain_pct=1.0, ir="cnn", done=set())
        self.assertEqual(len(hs), 1)
        self.assertEqual(hs[0]["chains"], [["conv", "relu", "pool"]])
        self.assertTrue(hs[0]["fused_kind"].startswith("fused__conv2d_f16__relu_f16__maxpool2d_f16"))
        g2, _, prov = gh.apply_hints(g, hs, dict(SPECS), statuses=None)
        self.assertEqual([o["name"] for o in g2["ops"]], ["conv+relu+pool"])

    def test_attention_matcher_and_modes(self):
        from modelblaster.pipeline.attention_fuse import VERIFIED_TRIU, fuse
        g = json.loads((DATA / "mingpt43_final_graph.json").read_text())
        g2, rep = fuse(g, const_lookup=lambda k: VERIFIED_TRIU)
        self.assertTrue(rep["fused"])
        att = [o for o in g2["ops"] if o["op"] == "attention_f16"]
        self.assertEqual(len(att), 1)
        self.assertEqual(att[0]["shape"].get("causal"), 1)
        self.assertEqual(fusion_ext.enabled("all"), {"siblings", "epilogue", "attention"})
        self.assertEqual(fusion_ext.decide_kinds("attention"), ("fuse_attention",))
        self.assertEqual(fusion_ext.decide_kinds("off"), ())
        with self.assertRaises(ValueError):
            fusion_ext.enabled("everything")


class DefaultsTest(unittest.TestCase):
    def test_off_by_default(self):
        from modelblaster.pipeline.kb_graph_opt import Config
        cfg = Config(kb=Path("."))
        env = dict(os.environ)
        try:
            for k in ("MB_GRAPH_CONCURRENCY", "KERNELBLASTER_MULTISLOT_ROLLOUT_ACCEPT"):
                os.environ.pop(k, None)
            ext = cl.Ext(cfg, Path("/nonexistent"))
            ext.apply_env()
            self.assertEqual(ext.round_key(), {})
            self.assertNotIn("MB_GRAPH_CONCURRENCY", os.environ)
            self.assertNotIn("KERNELBLASTER_MULTISLOT_ROLLOUT_ACCEPT", os.environ)
            self.assertEqual(ext.plan_kw(SPECS), {})
            self.assertEqual(ext.decide_kinds(), ())
            self.assertEqual(ext.seed_problems([Path("x")], None, SPECS), [Path("x")])
            self.assertEqual(ext.after_round(Path("."), [], {}, None, SPECS), [])
            on = cl.Ext(Config(kb=Path("."), reuse_winners="siblings", graph_concurrency="on",
                               fusion_extended="siblings"), Path("/nonexistent"))
            self.assertEqual(set(on.round_key()), {"reuse_winners", "graph_concurrency", "fusion_extended"})
            on.apply_env()
            self.assertEqual(os.environ.get("MB_GRAPH_CONCURRENCY"), "on")
            self.assertEqual(os.environ.get("KERNELBLASTER_MULTISLOT_ROLLOUT_ACCEPT"), "local")
            with self.assertRaises(ValueError):
                cl.Ext(Config(kb=Path("."), reuse_winners="all"), Path("."))
            auto = cl.Ext(Config(kb=Path("."), graph_concurrency="auto"), Path("/nonexistent"))
            self.assertTrue(auto.wants_after_round())
            self.assertEqual(auto.round_key(), {"graph_concurrency": ["auto", "split"]})
            auto.apply_env()
            self.assertEqual(os.environ.get("MB_GRAPH_CONCURRENCY"), "on")
            self.assertFalse(cl.Ext(cfg, Path("/nonexistent")).wants_after_round())
        finally:
            os.environ.clear()
            os.environ.update(env)


if __name__ == "__main__":
    unittest.main()
